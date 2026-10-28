# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Managed installs for the video-driven lip-sync engines: LatentSync and MuseTalk.

:mod:`codai.compose.presenter` has been able to *select* these two by name for a
while, and ranks them ahead of wav2lip because wav2lip works at 96x96 and upscales
— which is where the soft, smeared mouth on a 1080p reel comes from. But selecting
them only ever worked if an operator had put the engine on PATH themselves, so in
practice both answered "neither on PATH nor installed". Coderai orchestrates the
models it serves, so it installs these too.

Three things make an engine runnable, and each is handled differently:

* **Code.** Vendored under ``third_party/`` and pinned to a commit, not cloned at
  install time — an image someone else pulls has to run without network access to
  a third party. This module seeds a writable working copy from it, the way the
  packaged wav2lip shim seeds from the baked read-only source, so a run that
  writes into its own tree cannot corrupt the vendored one.
* **Weights.** Fetched from the Hub on first use, like every other model. Never
  during a compose job: a render is the wrong place to pull several GB, which is
  the rule :func:`codai.compose.presenter.available_engines` already follows.
* **An interpreter.** Each engine pins a stack the server cannot host — LatentSync
  wants torch 2.5.1 with diffusers 0.32 and numpy 1.x, MuseTalk wants diffusers
  0.30 and mmpose — so each gets its own venv on its own python, exactly as
  :mod:`codai.api.acestep_worker` does, and is driven as a subprocess.

Neither engine is reachable from this process's interpreter, so everything here
runs under :func:`codai.api.venv_env.clean_py_env`: the engine processes export
``PYTHONHOME`` pointing at the standalone 3.13, and a foreign interpreter that
inherits it dies before running a byte ("No module named 'encodings'").
"""

import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from codai.api.venv_env import clean_py_env

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDOR_ROOT = _REPO_ROOT / "third_party"


# --------------------------------------------------------------------------- #
# What an engine is
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class WeightSet:
    """One Hub repo's worth of weights.

    ``dest`` is a directory under the install root, because both engines read
    their checkpoints from hard-coded relative paths. ``dest=None`` means "just
    make sure it is in the HF cache": LatentSync calls
    ``AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")`` itself, so the
    only way to keep the run offline is to have fetched it beforehand.
    """
    repo_id: str
    patterns: Tuple[str, ...]
    dest: Optional[str] = None


@dataclass(frozen=True)
class EngineSpec:
    name: str
    #: Directory under third_party/ holding the pinned inference source.
    vendor: str
    #: The interpreter this engine's venv needs — not the server's.
    py_version: str
    #: Requirements file at the repo root.
    requirements: str
    #: ``python -m <entrypoint>``, run with the install root as the cwd.
    entrypoint: str
    weights: Tuple[WeightSet, ...]
    #: Paths under the install root that must all exist for the install to count
    #: as done. Checked instead of "could be downloaded", so a compose job never
    #: starts a 5 GB fetch.
    ready_markers: Tuple[str, ...]
    #: One import per line, used to decide whether the venv is usable.
    probe_imports: Tuple[str, ...]
    #: Extra directories (relative to the install root) to put on PYTHONPATH.
    pythonpath: Tuple[str, ...] = ()
    #: Builds the argv tail and returns (argv, finish) where `finish` moves the
    #: engine's output to the requested path. Set below, after the builders.
    argv: Optional[Callable] = None


def _env_dir(name: str, var: str) -> Optional[Path]:
    explicit = os.environ.get(var)
    return Path(explicit) if explicit else None


def install_dir(name: str) -> Path:
    """Where this engine's working copy and weights live.

    One directory per engine under the coderai config root — the same place
    :func:`codai.compose.presenter._managed_engine_dir` looks, because that is
    what makes a managed install discoverable without this module being imported.
    """
    override = _env_dir(name, f"CODERAI_{name.upper()}_DIR")
    if override:
        return override
    try:
        from codai.platform_paths import legacy_style_config_dir
        return legacy_style_config_dir() / name
    except Exception:
        return Path(os.path.expanduser(f"~/.coderai/{name}"))


def venv_dir(name: str) -> Path:
    override = _env_dir(name, f"CODERAI_{name.upper()}_VENV")
    if override:
        return override
    for baked in (Path(f"/opt/coderai/{name}_venv"),
                  Path(f"/opt/coderai/venvs/{name}")):
        if (baked / ("Scripts" if os.name == "nt" else "bin")).exists():
            return baked
    return Path(os.path.expanduser(f"~/.coderai/{name}_venv"))


# --------------------------------------------------------------------------- #
# Per-engine invocation
# --------------------------------------------------------------------------- #

def _latentsync_argv(root: Path, video: str, audio: str, out: str,
                     options: dict) -> Tuple[List[str], Callable[[], str]]:
    """LatentSync takes the three paths directly, plus its checkpoint and config.

    Both of those are *required* arguments upstream, which is why the engine could
    never have worked from the generic three-flag call presenter used to make. The
    config choice is not cosmetic: ``cross_attention_dim`` in it decides whether
    the pipeline looks for ``checkpoints/whisper/tiny.pt`` (384) or ``small.pt``
    (768), and only tiny.pt is in the weights repo.
    """
    config = options.get("unet_config") or "configs/unet/stage2_512.yaml"
    argv = ["--unet_config_path", config,
            "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
            "--video_path", os.path.abspath(video),
            "--audio_path", os.path.abspath(audio),
            "--video_out_path", os.path.abspath(out),
            "--inference_steps", str(int(options.get("steps") or 20)),
            "--guidance_scale", str(float(options.get("guidance_scale") or 1.5))]
    if options.get("deepcache", True):
        argv.append("--enable_deepcache")
    if options.get("seed") is not None:
        argv += ["--seed", str(int(options["seed"]))]

    def _finish() -> str:
        if not os.path.isfile(out):
            raise FileNotFoundError(out)
        return out

    return argv, _finish


def _musetalk_argv(root: Path, video: str, audio: str, out: str,
                   options: dict) -> Tuple[List[str], Callable[[], str]]:
    """MuseTalk takes a YAML task list, not paths, and writes into a result dir.

    So one is written per run and the single result is moved where the caller
    asked for it. ``version`` picks which checkpoint pair is used; v15 is the
    better one and the default, and the two live in different directories under
    models/.
    """
    import uuid

    version = str(options.get("version") or "v15").lower()
    if version not in ("v1", "v15"):
        raise ValueError(f"MuseTalk version must be v1 or v15, not {version!r}")
    model_dir = "models/musetalkV15" if version == "v15" else "models/musetalk"
    unet = f"{model_dir}/unet.pth" if version == "v15" else f"{model_dir}/pytorch_model.bin"

    run_id = uuid.uuid4().hex[:10]
    work = root / "coderai-runs" / run_id
    work.mkdir(parents=True, exist_ok=True)
    task = work / "task.yaml"
    # The loader is OmegaConf, so a quoted scalar is enough; abspath keeps it
    # independent of the cwd the subprocess runs in.
    task.write_text(
        "task_0:\n"
        f" video_path: \"{os.path.abspath(video)}\"\n"
        f" audio_path: \"{os.path.abspath(audio)}\"\n"
        + (f" bbox_shift: {int(options['bbox_shift'])}\n"
           if options.get("bbox_shift") is not None else ""),
        encoding="utf-8")

    argv = ["--inference_config", str(task),
            "--result_dir", str(work / "out"),
            "--unet_model_path", unet,
            "--unet_config", f"{model_dir}/musetalk.json",
            "--whisper_dir", "models/whisper",
            "--version", version,
            "--fps", str(int(options.get("fps") or 25))]
    if options.get("use_float16", True):
        argv.append("--use_float16")
    if options.get("batch_size"):
        argv += ["--batch_size", str(int(options["batch_size"]))]

    def _finish() -> str:
        made = sorted((work / "out").rglob("*.mp4"), key=os.path.getmtime)
        if not made:
            raise FileNotFoundError(f"no mp4 under {work / 'out'}")
        shutil.move(str(made[-1]), out)
        shutil.rmtree(work, ignore_errors=True)
        return out

    return argv, _finish


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

ENGINES: Dict[str, EngineSpec] = {
    "latentsync": EngineSpec(
        name="latentsync",
        vendor="latentsync",
        # mediapipe and decord publish nothing for 3.13, and the pinned torch is
        # a cp311 build. 3.11 is the newest interpreter the whole set agrees on.
        py_version="3.11",
        requirements="requirements-latentsync.txt",
        entrypoint="scripts.inference",
        weights=(
            WeightSet("ByteDance/LatentSync-1.6",
                      ("latentsync_unet.pt", "whisper/tiny.pt"),
                      dest="checkpoints"),
            # Pulled by name from inside the pipeline, so it has to be cached.
            WeightSet("stabilityai/sd-vae-ft-mse",
                      ("config.json", "diffusion_pytorch_model.safetensors")),
        ),
        ready_markers=("scripts/inference.py",
                       "configs/unet/stage2_512.yaml",
                       "checkpoints/latentsync_unet.pt",
                       "checkpoints/whisper/tiny.pt"),
        probe_imports=("torch", "diffusers", "decord", "insightface", "DeepCache"),
        argv=_latentsync_argv,
    ),
    "musetalk": EngineSpec(
        name="musetalk",
        vendor="musetalk",
        py_version="3.11",
        requirements="requirements-musetalk.txt",
        entrypoint="scripts.inference",
        weights=(
            WeightSet("TMElyralab/MuseTalk",
                      ("musetalk/musetalk.json", "musetalk/pytorch_model.bin",
                       "musetalkV15/musetalk.json", "musetalkV15/unet.pth"),
                      dest="models"),
            WeightSet("stabilityai/sd-vae-ft-mse",
                      ("config.json", "diffusion_pytorch_model.bin"),
                      dest="models/sd-vae"),
            WeightSet("openai/whisper-tiny",
                      ("config.json", "pytorch_model.bin",
                       "preprocessor_config.json"),
                      dest="models/whisper"),
            WeightSet("yzd-v/DWPose", ("dw-ll_ucoco_384.pth",),
                      dest="models/dwpose"),
            # Upstream's download_weights.sh takes these two off a Google Drive
            # id and a pytorch.org URL. A mirror on the Hub means the whole set
            # has one provenance and one auth story.
            WeightSet("camenduru/MuseTalk",
                      ("face-parse-bisent/79999_iter.pth",
                       "face-parse-bisent/resnet18-5c106cde.pth"),
                      dest="models"),
        ),
        ready_markers=("scripts/inference.py",
                       "models/musetalkV15/unet.pth",
                       "models/sd-vae/config.json",
                       "models/whisper/config.json",
                       "models/dwpose/dw-ll_ucoco_384.pth",
                       "models/face-parse-bisent/79999_iter.pth"),
        probe_imports=("torch", "diffusers", "mmpose", "face_detection"),
        # preprocessing.py does `from face_detection import …` — a top-level
        # import of a package that lives under musetalk/utils.
        pythonpath=(".", "musetalk/utils"),
        argv=_musetalk_argv,
    ),
}

#: Engine names this module can install, in no particular order.
MANAGED = tuple(ENGINES)

_lock = threading.RLock()
_ready: Dict[str, Path] = {}


def spec(name: str) -> EngineSpec:
    try:
        return ENGINES[(name or "").strip().lower()]
    except KeyError:
        raise ValueError(
            f"{name!r} is not a managed lip-sync engine; known: "
            + ", ".join(sorted(ENGINES))) from None


# --------------------------------------------------------------------------- #
# Code: seeding the working copy
# --------------------------------------------------------------------------- #

def vendor_dir(name: str) -> Path:
    return _VENDOR_ROOT / spec(name).vendor


def _seed_stamp(root: Path) -> Path:
    return root / ".coderai-seeded"


def _seed_code(name: str, log=None) -> Path:
    """Copy the vendored tree into the install root, once.

    The stamp records the vendored tree's own fingerprint rather than a date, so
    upgrading coderai re-seeds the code while leaving the weights — which are the
    multi-GB part and are not versioned with it — exactly where they are.
    """
    sp = spec(name)
    src = vendor_dir(name)
    if not src.is_dir():
        raise RuntimeError(
            f"{name}'s vendored source is missing at {src}. It ships in the "
            f"repo under third_party/; see third_party/{sp.vendor.upper()}-PINNED.")
    root = install_dir(name)
    fingerprint = _vendor_fingerprint(src)
    stamp = _seed_stamp(root)
    if stamp.exists():
        try:
            if stamp.read_text(encoding="utf-8").strip() == fingerprint:
                return root
        except OSError:
            pass
    root.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir()):
        target = root / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    stamp.write_text(fingerprint + "\n", encoding="utf-8")
    if log:
        log(f"[{name}] seeded the inference source into {root}")
    return root


def _vendor_fingerprint(src: Path) -> str:
    """Cheap content fingerprint: every file's relative path, size and mtime."""
    import hashlib
    h = hashlib.sha256()
    for p in sorted(src.rglob("*")):
        if p.is_file():
            st = p.stat()
            h.update(str(p.relative_to(src)).encode())
            h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:32]


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #

def _fetch_weights(name: str, log=None) -> None:
    sp = spec(name)
    root = install_dir(name)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"installing {name} needs huggingface_hub to fetch its weights") from exc
    for ws in sp.weights:
        kwargs = {"repo_id": ws.repo_id, "allow_patterns": list(ws.patterns)}
        where = "the HF cache"
        if ws.dest:
            target = root / ws.dest
            target.mkdir(parents=True, exist_ok=True)
            kwargs["local_dir"] = str(target)
            where = str(target)
        if log:
            log(f"[{name}] fetching {ws.repo_id} into {where}")
        try:
            snapshot_download(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"could not download {name}'s weights from {ws.repo_id}: {exc}. "
                f"Fetch {', '.join(ws.patterns)} into {where} by hand, or set "
                f"CODERAI_{name.upper()}_DIR to an install that already has "
                f"them.") from exc


# --------------------------------------------------------------------------- #
# The venv
# --------------------------------------------------------------------------- #

def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")


def _pip_ok(name: str, py: Path, quiet: bool = False) -> bool:
    """Does this venv import what the engine needs?

    clean_py_env() is load-bearing rather than hygiene — see the module docstring.
    """
    sp = spec(name)
    code = "import " + ", ".join(sp.probe_imports)
    try:
        r = subprocess.run([str(py), "-c", code], capture_output=True,
                           env=clean_py_env())
        if r.returncode != 0 and not quiet:
            why = (r.stderr or r.stdout or b"").decode(errors="replace").strip()
            print(f"[{name}] venv at {py} is not usable: {why[-400:]}", flush=True)
        return r.returncode == 0
    except Exception as exc:
        if not quiet:
            print(f"[{name}] could not probe the venv at {py}: {exc}", flush=True)
        return False


def _uv() -> Optional[str]:
    return shutil.which("uv") or next(
        (p for p in (os.path.expanduser("~/.local/bin/uv"), "/usr/local/bin/uv")
         if os.path.isfile(p)), None)


def _create_venv(name: str, venv: Path) -> None:
    """Make the venv on the interpreter this engine needs.

    ``pythonX.Y -m venv`` first, uv second: on a Debian host the interpreter is
    often present without its ``-venv`` package, so ensurepip is missing and that
    route fails. uv builds it anyway, and will fetch the interpreter if the host
    has none — which is better than an error telling the user to apt-install
    something.
    """
    sp = spec(name)
    venv.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    base = shutil.which(f"python{sp.py_version}")
    if base:
        r = subprocess.run([base, "-m", "venv", str(venv)], capture_output=True,
                           text=True, env=clean_py_env())
        if r.returncode == 0:
            return
        errors.append(f"python{sp.py_version} -m venv: "
                      f"{(r.stderr or r.stdout).strip()[:300]}")
    else:
        errors.append(f"no python{sp.py_version} on PATH")
    uv = _uv()
    if uv:
        r = subprocess.run([uv, "venv", "--python", sp.py_version, str(venv)],
                           capture_output=True, text=True, env=clean_py_env())
        if r.returncode == 0:
            return
        errors.append(f"uv venv: {(r.stderr or r.stdout).strip()[:300]}")
    else:
        errors.append("uv not found")
    raise RuntimeError(f"could not create a python {sp.py_version} venv for "
                       f"{name} at {venv}: " + "; ".join(errors))


def _has_pip(py: Path) -> bool:
    try:
        return subprocess.run([str(py), "-m", "pip", "--version"],
                              capture_output=True,
                              env=clean_py_env()).returncode == 0
    except Exception:
        return False


def _install_requirements(name: str, py: Path, log=None) -> None:
    sp = spec(name)
    req = _REPO_ROOT / sp.requirements
    if not req.exists():
        raise RuntimeError(f"{name} requirements missing: {req}")
    if log:
        log(f"[{name}] installing {sp.requirements} (first run downloads "
            f"several GB)")
    if _has_pip(py):
        subprocess.run([str(py), "-m", "pip", "install", "-U", "pip"],
                       check=True, env=clean_py_env())
        subprocess.run([str(py), "-m", "pip", "install", "-r", str(req)],
                       check=True, env=clean_py_env())
    else:
        # A venv uv created has no pip, and on this host uv is the route that
        # works (Debian ships the interpreter without its -venv package, so
        # ensurepip is missing and `pythonX.Y -m venv` fails). Falling back to
        # `python -m pip` there fails with "No module named pip" at the end of a
        # multi-GB download, which reads as a broken requirements file.
        uv = _uv()
        if not uv:
            raise RuntimeError(
                f"the {name} venv at {py.parent.parent} has no pip and uv is "
                f"not installed, so its requirements cannot be installed")
        # --index-strategy unsafe-best-match is required, not a tuning knob:
        # uv's default is first-index-wins, so with the PyTorch index in these
        # requirements it resolves unrelated packages off it and pins the wrong
        # versions of things like requests.
        subprocess.run([uv, "pip", "install", "--python", str(py),
                        "--index-strategy", "unsafe-best-match", "-r", str(req)],
                       check=True, env=clean_py_env())
    if not _pip_ok(name, py):
        raise RuntimeError(
            f"the {name} venv installed but does not import "
            f"{', '.join(sp.probe_imports)}. Its pins need a wheel this host "
            f"has no build for; see {sp.requirements}.")


def _bootstrap_venv(name: str, log=None) -> Path:
    venv = venv_dir(name)
    py = _venv_python(venv)
    if py.exists() and _pip_ok(name, py, quiet=True):
        return py
    if not py.exists():
        if log:
            log(f"[{name}] creating isolated venv at {venv}")
        _create_venv(name, venv)
        py = _venv_python(venv)
    _install_requirements(name, py, log=log)
    return py


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #

def is_installed(name: str) -> bool:
    """Code and weights present — WITHOUT downloading anything or probing a venv.

    This is the question a compose job is allowed to ask: the answer has to be
    cheap, and "could be downloaded" is not the same as "ready".
    """
    try:
        sp = spec(name)
    except ValueError:
        return False
    root = install_dir(name)
    try:
        return all((root / m).exists() for m in sp.ready_markers)
    except OSError:
        return False


def is_available(name: str) -> bool:
    """Installed AND the venv can import the engine. Still no downloads."""
    if not is_installed(name):
        return False
    py = _venv_python(venv_dir(name))
    return py.exists() and _pip_ok(name, py, quiet=True)


def missing_pieces(name: str) -> List[str]:
    """What an install is short of, for an error message worth reading."""
    sp = spec(name)
    root = install_dir(name)
    out = [m for m in sp.ready_markers if not (root / m).exists()]
    py = _venv_python(venv_dir(name))
    if not py.exists():
        out.append(f"a python {sp.py_version} venv at {venv_dir(name)}")
    elif not _pip_ok(name, py, quiet=True):
        out.append(f"working deps in {venv_dir(name)}")
    return out


def ensure_installed(name: str, log=None) -> Tuple[Path, Path]:
    """Install the engine if it is not already. Returns ``(root, venv_python)``.

    Downloads and a pip install, so never call it from a render — it is the
    explicit "install this engine" path. :func:`run` calls it only once
    :func:`is_installed` is already true, which costs nothing.
    """
    sp = spec(name)
    with _lock:
        cached = _ready.get(sp.name)
        if cached is not None and cached.exists():
            return install_dir(sp.name), cached
        root = _seed_code(sp.name, log=log)
        if not is_installed(sp.name):
            _fetch_weights(sp.name, log=log)
        missing = [m for m in sp.ready_markers if not (root / m).exists()]
        if missing:
            raise RuntimeError(
                f"{sp.name} is still incomplete after installing; missing: "
                + ", ".join(missing))
        py = _bootstrap_venv(sp.name, log=log)
        _ready[sp.name] = py
        if log:
            log(f"[{sp.name}] ready: {root} on {py}")
        return root, py


def run(name: str, video_path: str, audio_path: str, out_path: str,
        timeout: int = 3600, options: dict = None, log=None) -> str:
    """Drive the engine over ``video_path`` + ``audio_path`` into ``out_path``.

    Raises RuntimeError on failure, so the caller reports it rather than quietly
    handing back an unsynchronised video. Both engines replace the mouth in a
    clip, so ``video_path`` is a base clip and not a still.
    """
    sp = spec(name)
    options = dict(options or {})
    if not is_installed(sp.name):
        raise RuntimeError(
            f"{sp.name} is not installed. Missing: "
            + ", ".join(missing_pieces(sp.name))
            + f". Install it with codai.api.lipsync_engines.ensure_installed"
              f"({sp.name!r}) — it downloads several GB, which is why a render "
              f"will not do it for you.")
    root, py = ensure_installed(sp.name, log=log)

    tail, finish = sp.argv(root, video_path, audio_path, out_path, options)
    cmd = [str(py), "-m", sp.entrypoint] + tail
    env = clean_py_env()
    paths = [str((root / p).resolve()) for p in (sp.pythonpath or (".",))]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if log:
        log(f"[{sp.name}] {' '.join(cmd[1:])}")
    try:
        proc = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{sp.name} did not finish within {timeout}s") from exc
    if proc.returncode != 0:
        why = (proc.stderr or proc.stdout or "").strip()[-900:]
        raise RuntimeError(f"{sp.name} failed (rc={proc.returncode}): {why}")
    try:
        return finish()
    except FileNotFoundError:
        why = (proc.stdout or proc.stderr or "").strip()[-600:]
        raise RuntimeError(
            f"{sp.name} exited 0 but produced no video. Last output: {why}") from None
