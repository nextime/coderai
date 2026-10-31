# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fully-managed isolated venv for embedders whose code predates transformers 5.

Two models live here, and they are here for the same reason: their remote code
was written against transformers 4 and will never be ported, so the server's
transformers 5.12 cannot run them — not "runs them worse", cannot.

* **jina-clip-v2** imports at all only with two shims (``clip_loss``, deleted
  from modeling_clip; the ``torch_dtype``/``dtype`` rename) and then returns
  1024 NaNs from every image. Measured on zeiss at fp32, bf16 and fp16, with
  xattn on and off, inside coderai and outside it. On transformers 4.49, same
  weights and images: finite unit-norm vectors, cos(text, image) ~0.26, on CUDA
  and CPU and on python 3.11 and 3.13 alike.
* **MM-Embed** needed three shims in a row and then asked for a fourth. Its
  ``modeling_nvmmembed`` registers ``LlavaNextConfig`` against its own class at
  import time — process-wide, so every LLaVA-NeXT checkpoint loaded afterwards
  came back as MM-Embed and e5-v started answering ``'LlavaNextConfig' object
  has no attribute 'retriever'`` — and once that was contained it failed on
  ``cannot import name 'MISTRAL_INPUTS_DOCSTRING'``. Shimming one deleted name
  at a time is a losing game: each fix reveals the next and the model is
  unusable until the last one. Out of the process, none of them applies.

So: ONE venv pinned to transformers 4.49 (nothing else is pinned), built with
the SERVER'S OWN interpreter — the measurements above hold on 3.11 and 3.13, so
there is no second python to find and an install here is a `python -m venv` and
a pip run. Each model runs in it as a subprocess
(:mod:`codai.api.tf4_embed_server`) whose VRAM the model manager reclaims by
killing it, and whose quantization policy is decided HERE, by the parent, from
models.json — the child cannot import coderai to ask.

coderai owns the lifecycle the way it owns pyannote's and NeMo's: baked into the
image when it is there, built on demand when it is not, and with a card on the
models page so the minutes of pip do not land on somebody's first request.
"""

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from codai.api import venv_env

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVER_SCRIPT = Path(__file__).resolve().parent / "tf4_embed_server.py"
_REQUIREMENTS = _REPO_ROOT / "requirements-embedders-tf4.txt"
_BAKED_VENV = Path("/opt/coderai/tf4_embedders_venv")

@dataclass(frozen=True)
class Tf4Model:
    """One model, and the transformers it needs."""
    #: The repo id, as models.json carries it.
    repo: str
    #: Which adapter the child drives it with -- the two have different APIs:
    #: jina exposes encode_text/encode_image, MM-Embed one encode() over dicts.
    family: str
    #: The EXACT transformers version its remote code was written against. One
    #: venv per distinct pin, because these are not interchangeable: MM-Embed's
    #: retriever runs its own decoder loop and calls each Mistral layer without
    #: `position_embeddings`, which transformers computed inside the attention
    #: layer up to 4.42 and requires from the caller after the rotary refactor.
    #: On 4.49 it dies in transformers' own code --
    #:
    #:     modeling_mistral.py:166, in forward
    #:         cos, sin = position_embeddings
    #:     TypeError: cannot unpack non-iterable NoneType object
    #:
    #: -- and NV-Embed-v1's model card says `pip install transformers==4.42.4`.
    transformers: str
    #: The interpreter that transformers can be installed on, when it is not
    #: the server's. Empty means "the server's own python", which is the normal
    #: case and needs no interpreter to be found. transformers 4.42.4 pins
    #: `tokenizers<0.20`, and tokenizers 0.19.1 publishes wheels up to cp312
    #: only -- so on the server's 3.13 pip builds it from source and PyO3 0.21
    #: refuses: "the configured Python interpreter version (3.13) is newer than
    #: PyO3's maximum supported version (3.12)".
    python: str = ""
    #: What transformers 5 does to it, for the models page and for anyone
    #: wondering why a model is not simply loaded in-process.
    why: str = ""


MODELS = {
    "jinaai/jina-clip-v2": Tf4Model(
        repo="jinaai/jina-clip-v2", family="jinaclip", transformers="4.49.0",
        why="on transformers 5 its image tower returns NaNs (and its remote "
            "code needs two shims just to import)"),
    "nvidia/mm-embed": Tf4Model(
        repo="nvidia/MM-Embed", family="mmembed", transformers="4.42.4",
        python="3.12",
        why="its remote code registers LlavaNextConfig process-wide (which "
            "broke e5-v), wants names transformers 5 deleted, and its "
            "retriever predates the rotary refactor of 4.43"),
}


#: Pins that belong to ONE transformers generation rather than to all of them.
#:
#: transformers 4.42.4 pins ``numpy<2``, and the current releases of these two
#: cannot live with that. Measured on zeiss 2026-10-31, in a venv where
#: everything else imported cleanly:
#:
#:     peft     ImportError: cannot import name 'EncoderDecoderCache' from
#:              'transformers'                 (peft 0.21 needs >= 4.43)
#:     datasets ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4
#:                                             (datasets 5 pulls pyarrow 26)
#:
#: They are also needed by exactly one model -- MM-Embed imports peft, and its
#: retriever imports datasets -- so they are not in the shared requirements at
#: all. jina's venv has no business carrying either.
PIN_EXTRAS = {
    # pyarrow as well, and for the same reason one step further out: datasets
    # 2.20 accepts `pyarrow>=15`, so pip happily kept pyarrow 26 -- and pyarrow
    # is the package that refuses numpy 1.x ("pyarrow requires NumPy 2.0 or
    # newer, found 1.26.4"). Pinning datasets without it fixes the version that
    # was named in the error and not the one that caused it.
    "4.42.4": ("peft==0.11.1", "datasets==2.20.0", "pyarrow==16.1.0",
               # accelerate too, and this one is not about numpy. Its 1.x
               # `dispatch_model` takes a `model.to(device)` shortcut for a
               # single-device map, which transformers 4.42 forbids for a
               # bitsandbytes model -- so a quantized GPU load died in
               # accelerate's own frame:
               #
               #   transformers/modeling_utils.py:3904 in from_pretrained
               #     dispatch_model(model, **device_map_kwargs)
               #   accelerate/big_modeling.py:514 in dispatch_model
               #     model.to(device)
               #   transformers/modeling_utils.py:2774 in to
               #     ValueError: `.to` is not supported for `4-bit` or `8-bit`
               #     bitsandbytes models.
               #
               # and the model fell back to the CPU: 30x slower and 33 GB of
               # host RAM for what fits in 5 GB of card.
               "accelerate==0.32.1",
               "bitsandbytes>=0.45"),
}

#: …and what each of those adds to the import probe, so a venv missing one is
#: repaired rather than kept (see _PROBE_IMPORTS).
PIN_PROBE_IMPORTS = {
    # pyarrow directly as well as through datasets: it is the one that refuses
    # numpy 1.x, and probing it by name repairs a venv whose pyarrow drifted
    # without relying on `import datasets` to notice.
    "4.42.4": ("peft", "datasets", "pyarrow", "accelerate", "bitsandbytes"),
}


def probe_imports(pin: str) -> tuple:
    """What this pin's venv must import for its models to run."""
    return tuple(_PROBE_IMPORTS) + tuple(PIN_PROBE_IMPORTS.get(pin, ()))


def python_for(pin: str) -> str:
    """The interpreter version this pin's venv must be built on, or "".

    From the registry rather than from a table of its own: it is a property of
    the transformers version, and every model sharing a pin shares its python.
    """
    for m in MODELS.values():
        if m.transformers == pin and m.python:
            return m.python
    return ""


def pins() -> list:
    """Every transformers version this install needs a venv for."""
    seen = []
    for m in MODELS.values():
        if m.transformers not in seen:
            seen.append(m.transformers)
    return seen


def model_for(name) -> Optional[Tf4Model]:
    """The registry entry for a requested id, or None when it is not ours.

    Matched on the full id and on its basename, case-insensitively, because a
    request names a model however the operator configured it -- 'MM-Embed',
    'nvidia/MM-Embed' and 'mm-embed' are one model.
    """
    low = str(name or "").strip().lower()
    if not low:
        return None
    if low in MODELS:
        return MODELS[low]
    base = low.rsplit("/", 1)[-1]
    for key, spec in MODELS.items():
        if base == key.rsplit("/", 1)[-1]:
            return spec
    return None


#: What the venv must be able to import for the server to run at all.
#:
#: Every one of these, not just the obvious four: this list is also what decides
#: whether an EXISTING venv is usable, so a dependency added to the
#: requirements reaches an installed venv only if its absence is noticed here.
#: Measured 2026-10-31 -- bitsandbytes, accelerate and peft were added for
#: MM-Embed, `ensure_installed` saw a venv that imported torch/transformers/
#: timm/einops/PIL, returned early, and the model failed with "This modeling
#: file requires the following packages that were not found in your
#: environment: peft".
_PROBE_IMPORTS = ("torch", "transformers", "timm", "einops", "PIL")

#: The same pins as requirements-embedders-tf4.txt, for installs that cannot see
#: it. The container bind-mounts codai/, tools/ and third_party/ over the image,
#: so a repo-ROOT file added after the running image was built is simply absent
#: until the container is recreated -- and this file was added in exactly that
#: state, which is how a venv that builds perfectly failed with "requirements
#: file missing". A test keeps the two copies equal, because two copies of a pin
#: drift.
#: transformers is absent on purpose: the version is per model and is passed
#: on the pip command line from the registry.
_FALLBACK_PINS = (
    "torch>=2.6",
    "torchvision>=0.21",
    "timm>=1.0.9",
    "einops>=0.8",
    "pillow>=10.0",
    "numpy>=1.26",
    "sentencepiece>=0.2",
    "protobuf>=4.25",
    # MM-Embed's chain (peft, datasets) is NOT here: those two are pinned per
    # transformers generation, see PIN_EXTRAS.
)

_lock = threading.RLock()
#: pin -> whether this process has already checked that venv.
_bootstrapped = {}


def _pointer_file(pin: str) -> Path:
    """Where the chosen venv path for this pin is recorded.

    In the CONFIG root, which is the one directory the whole fleet shares and
    which does not vary per process -- unlike the cache root, see below.
    """
    name = f"embedders-tf-{pin}-venv.path"
    try:
        from codai.platform_paths import legacy_style_config_dir
        return legacy_style_config_dir() / name
    except Exception:
        return Path(os.path.expanduser(f"~/.coderai/{name}"))


def _candidate_dirs(pin: str) -> list:
    """Every place a venv for this pin may already be, preferred name first.

    More than one, because the cache root is NOT the same directory in every
    process: the engines run with ``CODERAI_CACHE_DIR`` pointing at the big
    disk while the front and an admin request do not. Measured on zeiss
    2026-10-31 -- the engine built ``/AI/guffcache/jinaclip-venv`` (5.6 GB) and
    a second process built ``/cache/jinaclip-venv`` (6.0 GB), 11 GB of the same
    thing, and the models page would have reported a venv the serving engine
    was not using. So the resolution looks in all of them and converges on
    whichever usable one exists.

    The two unversioned names are only candidates for 4.49: that is what this
    venv held while it served one model, and an install that already has those
    several GB keeps them.
    """
    roots = []
    cache = os.environ.get("CODERAI_CACHE_DIR")
    if cache:
        roots.append(Path(cache))
    try:
        from codai.platform_paths import legacy_style_cache_root
        roots.append(legacy_style_cache_root() / "coderai")
    except Exception:
        roots.append(Path(os.path.expanduser("~/.cache/coderai")))
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        roots.append(Path(xdg))
    names = [f"embedders-tf-{pin}-venv"]
    if pin == "4.49.0":
        names += ["tf4-embedders-venv", "jinaclip-venv"]
    # Roomiest root first for a venv that does not exist yet. Six GB has to go
    # somewhere, and the cache roots are not equivalent: on zeiss the engines'
    # CODERAI_CACHE_DIR is /AI/guffcache with 197 GB free while the default
    # cache root had 15 GB, and building there took that partition to 11 MB
    # free -- at which point HF_HOME, also on it, failed every download with
    # "No space left on device" and MM-Embed fell back to the CPU. An EXISTING
    # venv still wins over a roomier empty root: the order below is only
    # consulted after the caller has looked for one.
    def _free(d: Path) -> int:
        try:
            st = os.statvfs(d if d.exists() else d.parent)
            return st.f_bavail * st.f_frsize
        except Exception:
            return 0

    roots.sort(key=_free, reverse=True)
    out = []
    for root in roots:
        for name in names:
            d = root / name
            if d not in out:
                out.append(d)
    return out


def resolve_venv_dir(pin: str = "4.49.0") -> Path:
    """Where this pin's venv lives, resolved the same way in every process.

    An explicit path, then the path a previous build recorded, then one baked
    into the image, then any usable venv among the candidates.

    The cache root rather than the config root for a NEW one, because this is
    rebuildable derived state and several GB of it -- the same choice the other
    isolated venvs make, and the same directory the SALAD server's venv uses.
    """
    explicit = (os.environ.get(f"CODERAI_EMBEDDERS_TF_{pin.replace('.', '_')}_VENV")
                or (os.environ.get("CODERAI_TF4_EMBEDDERS_VENV")
                    or os.environ.get("CODERAI_JINACLIP_VENV")
                    if pin == "4.49.0" else None))
    if explicit:
        return Path(explicit)
    try:
        recorded = _pointer_file(pin).read_text(encoding="utf-8").strip()
    except Exception:
        recorded = ""
    if recorded and venv_env.venv_python(Path(recorded)).is_file():
        return Path(recorded)
    baked = Path(f"/opt/coderai/embedders_tf_{pin.replace('.', '_')}_venv")
    if (baked / ("Scripts" if os.name == "nt" else "bin")).exists():
        return baked
    if pin == "4.49.0" and (_BAKED_VENV / (
            "Scripts" if os.name == "nt" else "bin")).exists():
        return _BAKED_VENV
    candidates = _candidate_dirs(pin)
    for d in candidates:
        if venv_env.venv_python(d).is_file():
            return d
    return candidates[0]


def venv_python(pin: str = "4.49.0", venv: Path = None) -> Path:
    return venv_env.venv_python(venv or resolve_venv_dir(pin))


def _transformers_ok(py: Path, pin: str) -> bool:
    """Imports everything, AND is EXACTLY the transformers this pin asks for.

    Exactly, not "a 4.x": the two pins are not interchangeable. MM-Embed's
    retriever dies inside transformers' own Mistral code on 4.49 (it predates
    the rotary refactor), and a venv silently carrying the other pin's version
    would fail there with every file in place and nothing to see. The same
    check catches a venv that --system-site-packages or a `pip install -U` has
    moved to transformers 5, which is what these venvs exist to avoid.
    """
    ok, _why = venv_env.import_ok(py, probe_imports(pin))
    if not ok:
        return False
    try:
        r = venv_env.run([str(py), "-c",
                          "import transformers,sys;"
                          f"sys.exit(0 if transformers.__version__ == {pin!r}"
                          " else 1)"],
                         capture_output=True)
        return r.returncode == 0
    except Exception:
        return False


def is_installed(pin: str = "4.49.0") -> bool:
    """Cheap: a venv whose python exists. Says nothing about its packages."""
    try:
        return venv_python(pin).is_file()
    except Exception:
        return False


def is_available(pin: str = "4.49.0") -> bool:
    """Installed AND importable AND on exactly this pin. No downloads."""
    try:
        py = venv_python(pin)
        return py.is_file() and _transformers_ok(py, pin)
    except Exception:
        return False


def ensure_installed(pin: str = "4.49.0", log=None) -> Path:
    """Build this pin's venv if it is not usable; return its python.

    Several GB of pip, so this is the explicit install path. It is also reached
    from the first embedding request, as pyannote's is, because refusing for
    ever is worse than one slow request -- but an operator can press the button
    on the models page instead and keep that request fast.
    """
    def _say(msg):
        print(f"[tf4-embed] {msg}", flush=True)
        if log:
            log(msg)

    with _lock:
        if _bootstrapped.get(pin) and venv_python(pin).is_file():
            return venv_python(pin)
        py = venv_python(pin)
        if py.is_file() and _transformers_ok(py, pin):
            _bootstrapped[pin] = True
            return py
        venv = resolve_venv_dir(pin)
        want_py = python_for(pin)
        # create_venv REBUILDS what cannot be used rather than running
        # `python -m venv` over the top of it, which fails on the very thing
        # that broke it. It builds with THIS interpreter -- right for a pin
        # that installs on the server's python, and wrong for one that does
        # not, hence the explicit route below.
        _say(f"building the venv for transformers {pin} at {venv}"
             + (f" on python {want_py}" if want_py else ""))
        py = (_create_venv_on(want_py, venv, _say) if want_py
              else venv_env.create_venv(venv))
        if not _transformers_ok(py, pin):
            _say(f"installing transformers {pin} and torch into it (first run "
                 f"downloads several GB)")
            venv_env.run([str(py), "-m", "pip", "install", "-U", "pip"],
                         check=True)
            # The pin comes from the registry, not from the requirements file:
            # one file serves every pin, because what differs between them is
            # only this one line.
            venv_env.run([str(py), "-m", "pip", "install",
                          f"transformers=={pin}"]
                         + list(PIN_EXTRAS.get(pin, ()))
                         + venv_env.pip_install_args(_REQUIREMENTS,
                                                     _FALLBACK_PINS),
                         check=True)
            if not _transformers_ok(py, pin):
                raise RuntimeError(
                    f"the venv for transformers {pin} installed but does not "
                    f"import {', '.join(probe_imports(pin))} on that version — see "
                    f"{_REQUIREMENTS.name}")
        _bootstrapped[pin] = True
        # Record it, so a process whose CODERAI_CACHE_DIR differs resolves to
        # THIS venv instead of building its own copy of the same 6 GB.
        try:
            pf = _pointer_file(pin)
            pf.parent.mkdir(parents=True, exist_ok=True)
            pf.write_text(str(venv) + "\n", encoding="utf-8")
        except Exception as exc:
            _say(f"could not record the venv path ({exc}); other processes may "
                 f"build their own copy")
        _say(f"ready: {py}")
        return py


def _interpreters(version: str) -> list:
    """Every python of exactly ``version`` this host can offer, best first.

    A LIST, not one: an interpreter that runs is not necessarily one that can
    build a venv. The shipped image carries /usr/bin/python3.12 without its
    `-venv` package, so `python3.12 -m venv` ends in "You may need to use sudo
    with that command. After installing the python3-venv package…" while the
    interpreter itself works perfectly -- measured on zeiss 2026-10-31. So the
    caller tries them in order and falls back to fetching one, which is the
    same ladder :mod:`codai.api.lipsync_engines` climbs for its 3.11.
    """
    def _runs_as(cand: Path) -> bool:
        try:
            r = venv_env.run([str(cand), "-c",
                              "import sys;print('%d.%d' % sys.version_info[:2])"],
                             capture_output=True, text=True, timeout=60)
            return r.returncode == 0 and r.stdout.strip() == version
        except Exception:
            return False

    out = []
    found = shutil.which(f"python{version}")
    if found and _runs_as(Path(found)):
        out.append(Path(found))
    # An interpreter provisioned for another isolated venv is an interpreter
    # for this one: the bundle builds /opt/coderai/py310 for the wav2lip venv
    # and a lip-sync install provisions <config>/py311.
    bundled = Path(f"/opt/coderai/py{version.replace('.', '')}/bin/python{version}")
    if bundled.is_file() and _runs_as(bundled):
        out.append(bundled)
    shared = venv_env.standalone_python(version)
    if shared is not None and shared not in out:
        out.append(shared)
    return out


def _create_venv_on(version: str, venv: Path, say) -> Path:
    """Build ``venv`` with a python of ``version``, rebuilding a stale one."""
    py = venv_env.venv_python(venv)
    if py.is_file() and venv_env.interpreter_version(py) == tuple(
            int(x) for x in version.split(".")[:2]):
        return py
    if venv.exists():
        say(f"the venv at {venv} is not a python {version} one; rebuilding it")
        shutil.rmtree(venv, ignore_errors=True)
    venv.parent.mkdir(parents=True, exist_ok=True)
    def _usable(made: Path) -> bool:
        """A venv is usable when it has PIP, not when `-m venv` exited 0.

        Measured on zeiss 2026-10-31: the image's /usr/bin/python3.12 has no
        `-venv` package, and `python3.12 -m venv` prints "After installing the
        python3-venv package, recreate your virtual environment" and still
        exits 0, leaving bin/python, python3, python3.12 and nothing else. The
        next step then failed with "No module named pip" -- after the exit code
        had already been taken as success.
        """
        try:
            r = venv_env.run([str(made), "-m", "pip", "--version"],
                             capture_output=True, text=True, timeout=120)
            return r.returncode == 0
        except Exception:
            return False

    errors = []
    for base in _interpreters(version):
        r = venv_env.run([str(base), "-m", "venv", str(venv)],
                         capture_output=True, text=True, timeout=600)
        made = venv_env.venv_python(venv)
        if r.returncode == 0 and made.is_file() and _usable(made):
            return made
        why = (r.stderr or r.stdout or "").strip()[-200:]
        if r.returncode == 0:
            why = why or "the venv it made has no pip (no ensurepip)"
        errors.append(f"{base}: {why}")
        shutil.rmtree(venv, ignore_errors=True)
    # Nothing here can build it, so fetch an interpreter that can. Last,
    # because every route above costs nothing and this one is a download; and
    # not optional, because otherwise this model is uninstallable on an image
    # whose distribution python ships without ensurepip -- which is the image
    # coderai ships.
    say(f"no python {version} on this host can build a venv"
        + (f" ({'; '.join(errors)})" if errors else "")
        + "; fetching a standalone one")
    fetched = venv_env.fetch_standalone_python(version, log=say)
    venv_env.run([str(fetched), "-m", "venv", str(venv)], check=True,
                 timeout=600)
    made = venv_env.venv_python(venv)
    if not _usable(made):
        raise RuntimeError(
            f"the python {version} fetched for this pin made a venv with no "
            f"pip at {venv}")
    return made


def _pump_stderr(proc: subprocess.Popen, model_id: str) -> None:
    """Copy the child's stderr into this process's log until it closes."""
    def _run():
        try:
            for line in proc.stderr:
                line = line.rstrip()
                if line:
                    print(f"[tf4-embed:{model_id}] {line}", flush=True)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True,
                     name=f"tf4-stderr-{model_id}").start()


def _die_with_parent():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, 9)   # PR_SET_PDEATHSIG, SIGKILL
    except Exception:
        pass


def spawn(model_id: str, device: str = "cuda", load_timeout: float = 900.0,
          env_extra: dict = None, quantization: dict = None) -> subprocess.Popen:
    """Start the server for ``model_id`` and return it once it is ready.

    Raises rather than returning a process that is still loading: the caller
    treats a returned handle as usable, and a half-loaded one would turn the
    first embedding request into the load error.

    ``quantization`` is decided by the CALLER, from models.json, because the
    child cannot import coderai to ask: ``{"load_in_4bit": bool,
    "load_in_8bit": bool, "skip_modules": [...], "compute_dtype": "bf16"}``.
    The skip list matters more than the bits do -- for a vision-language
    EMBEDDER the tower turns pixels into the vector being compared and the
    projector is one small matrix whose error lands on every image, so they
    stay at full precision while the language layers are quantized.
    """
    spec = model_for(model_id)
    if spec is None:
        raise ValueError(
            f"{model_id!r} is not served out of a pinned-transformers venv; "
            f"known: {', '.join(sorted(MODELS))}")
    py = ensure_installed(spec.transformers)
    if not _SERVER_SCRIPT.is_file():
        raise RuntimeError(f"tf4 embedding server script missing: {_SERVER_SCRIPT}")
    # own_libs: this venv is built by the server's interpreter but ships its own
    # torch and its own CUDA/cuDNN wheels, so it must not inherit the server's
    # LD_LIBRARY_PATH. With it, the text tower answered normally on the GPU and
    # every image failed in cuDNN (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED) --
    # only the vision tower runs convolutions, so half of the model worked.
    env = venv_env.env_for_venv(resolve_venv_dir(spec.transformers),
                                extra=dict(env_extra or {}), own_libs=True)
    env["TF4_MODEL"] = str(model_id)
    env["TF4_FAMILY"] = spec.family
    if str(device) == "cpu":
        env["TF4_FORCE_CPU"] = "1"
    if quantization:
        env["TF4_QUANTIZATION"] = json.dumps(quantization)
    proc = subprocess.Popen(
        [str(py), str(_SERVER_SCRIPT)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env, text=True, bufsize=1,
        preexec_fn=_die_with_parent)
    # Pump stderr into the engine log, in a thread, for the whole life of the
    # child. Not only on a silent exit, which is all this used to do: a child
    # that falls back from the GPU to the CPU says so on stderr, and MM-Embed
    # did exactly that on zeiss 2026-10-31 -- "ready on cpu" in the log, 33 GB
    # of host RAM, 1100% CPU, and the reason sitting unread in a pipe. A pipe
    # nobody drains is also a pipe that fills: 64 KB of warnings and the child
    # blocks writing to it, which looks like a model that hangs.
    _pump_stderr(proc, model_id)
    deadline = time.time() + float(load_timeout)
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            # stderr is being pumped to the log by the thread above, so the
            # reason is already there rather than read from the pipe here.
            raise RuntimeError(
                f"the {spec.family} server for {model_id} exited while "
                f"loading; its output is in the engine log")
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if msg.get("ready"):
            print(f"[tf4-embed] {model_id} ready on {msg.get('device')} "
                  f"({msg.get('dim')}-d, transformers "
                  f"{msg.get('transformers')}"
                  + (", quantized" if msg.get("quantized") else "")
                  + ")", flush=True)
            if msg.get("gpu_error"):
                # Loud, because the difference is 30x and tens of GB of host
                # RAM: a model on the CPU because it was pinned there and a
                # model on the CPU because the GPU load failed look identical
                # in the log otherwise.
                print(f"[tf4-embed] {model_id} IS ON THE CPU because the GPU "
                      f"load failed: {msg['gpu_error']}", flush=True)
            return proc
        if msg.get("error"):
            proc.kill()
            if msg.get("traceback"):
                # Printed, not raised: the request needs one line, the operator
                # needs the frame it happened in.
                print(f"[tf4-embed] {model_id} failed to load:\n"
                      f"{msg['traceback']}", flush=True)
            raise RuntimeError(f"{model_id}: {msg['error']}")
    proc.kill()
    raise RuntimeError(f"{model_id} did not load within "
                       f"{int(load_timeout)}s")


def request(proc: subprocess.Popen, payload: dict) -> list:
    """Send one request, return its embeddings. Raises on an error reply."""
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("the embedding server died mid-request")
        line = line.strip()
        if not line.startswith("{"):
            continue
        msg = json.loads(line)
        if "error" in msg:
            raise ValueError(f"tf4-embed: {msg['error']}")
        if "embeddings" in msg:
            return msg["embeddings"]


def state() -> list:
    """What the models page shows, without touching the network.

    One entry per transformers pin, each listing the models that need it. The
    two are not interchangeable: MM-Embed's retriever predates the rotary
    refactor of 4.43 and dies inside transformers' own Mistral code on 4.49, so
    "which venv does this model use" is a question the page has to answer.
    """
    out = []
    for pin in pins():
        venv = resolve_venv_dir(pin)
        installed = is_installed(pin)
        out.append({
            "name": f"transformers {pin} (embedders)",
            "pin": pin,
            "venv": str(venv),
            "requirements": _REQUIREMENTS.name,
            "installed": installed,
            "available": is_available(pin) if installed else False,
            "models": [{"repo": m.repo, "why": m.why}
                       for m in MODELS.values() if m.transformers == pin],
            "why": ("These models' remote code was written against a specific "
                    "transformers 4 and is not maintained: on the server's "
                    "transformers 5 one returns NaN image vectors and the "
                    "other asks for names that were deleted. This venv pins "
                    f"transformers {pin} and nothing else."),
        })
    return out
