# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fully-managed isolated venv for jina-clip-v2, the multilingual text+image embedder.

The reason it cannot share the server's environment is one package. The model's
remote code is pinned by its own config to revision 39e6a55 (July 2025), and
jinaai's jina-clip-implementation has had no commit since, so it will never be
transformers-5 code. On transformers 5.12 it needs two shims just to import
(``clip_loss``, deleted from modeling_clip; ``dtype=`` for ``torch_dtype=``), and
with both applied it loads and then returns 1024 NaNs from every image --
measured on zeiss at fp32, bf16 and fp16, with xattn on and off, and outside
coderai entirely. A NaN vector is worse than an error: it scores every
comparison equally, silently.

Measured the same day, same weights, same images, on transformers 4.49:

    text : shape=(2, 1024) finite=True nan=0 norm=0.9992
    image: shape=(1, 1024) finite=True nan=0 norm=1.0007
    cos(text, image): 0.258 / 0.260

on CUDA and on CPU, and on python 3.11 and 3.13 alike -- which is why this venv
is built with the SERVER'S OWN interpreter and pins nothing but transformers.
There is no second python to find, so an install here is a `python -m venv` and
a pip run, not a bring-up.

coderai owns the lifecycle, the way it does for pyannote and NeMo: the venv is
baked into the image when it is there, built on demand when it is not, and the
model runs as a subprocess (``codai/api/jina_clip_server.py``) whose VRAM the
model manager can reclaim by killing it.
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from codai.api import venv_env

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVER_SCRIPT = Path(__file__).resolve().parent / "jina_clip_server.py"
_REQUIREMENTS = _REPO_ROOT / "requirements-jinaclip.txt"
_BAKED_VENV = Path("/opt/coderai/jinaclip_venv")

#: What the venv must be able to import for the server to run at all.
_PROBE_IMPORTS = ("torch", "transformers", "timm", "einops", "PIL")

#: The same pins as requirements-jinaclip.txt, for the installs that cannot see
#: it. The container bind-mounts codai/, tools/ and third_party/ over the image,
#: so a repo-ROOT file added after the running image was built is simply absent
#: until the container is recreated -- and this file was added in exactly that
#: state, which is how a venv that builds perfectly failed with "requirements
#: file missing". A test keeps the two copies equal, because two copies of a pin
#: drift.
_FALLBACK_PINS = (
    "transformers==4.49.0",
    "torch>=2.6",
    "torchvision>=0.21",
    "timm>=1.0.9",
    "einops>=0.8",
    "pillow>=10.0",
    "numpy>=1.26",
    "sentencepiece>=0.2",
    "protobuf>=4.25",
)

_lock = threading.RLock()
_bootstrapped = False


def resolve_venv_dir() -> Path:
    """Where the venv lives: an explicit path, the baked one, or the cache root.

    The cache root rather than the config root, because this is rebuildable
    derived state and several GB of it -- the same choice the other isolated
    venvs make, and the same directory the SALAD server's venv uses.
    """
    explicit = os.environ.get("CODERAI_JINACLIP_VENV")
    if explicit:
        return Path(explicit)
    if (_BAKED_VENV / ("Scripts" if os.name == "nt" else "bin")).exists():
        return _BAKED_VENV
    cache = os.environ.get("CODERAI_CACHE_DIR")
    if cache:
        return Path(cache) / "jinaclip-venv"
    try:
        from codai.platform_paths import legacy_style_cache_root
        return legacy_style_cache_root() / "coderai" / "jinaclip-venv"
    except Exception:
        return Path(os.path.expanduser("~/.cache/coderai/jinaclip-venv"))


def venv_python(venv: Path = None) -> Path:
    return venv_env.venv_python(venv or resolve_venv_dir())


def _transformers_ok(py: Path) -> bool:
    """Imports AND is transformers 4.

    The import test alone would pass on a venv that --system-site-packages or a
    later `pip install -U` has moved to transformers 5, which is the one thing
    this venv exists to avoid. It would then serve NaNs again, with every file
    in place and nothing to see.
    """
    ok, _why = venv_env.import_ok(py, _PROBE_IMPORTS)
    if not ok:
        return False
    try:
        r = venv_env.run([str(py), "-c",
                          "import transformers,sys;"
                          "sys.exit(0 if transformers.__version__.split('.')[0]"
                          " == '4' else 1)"],
                         capture_output=True)
        return r.returncode == 0
    except Exception:
        return False


def is_installed() -> bool:
    """Cheap: a venv whose python exists. Says nothing about its packages."""
    try:
        return venv_python().is_file()
    except Exception:
        return False


def is_available() -> bool:
    """Installed AND importable AND still on transformers 4. No downloads."""
    try:
        py = venv_python()
        return py.is_file() and _transformers_ok(py)
    except Exception:
        return False


def ensure_installed(log=None) -> Path:
    """Build the venv if it is not usable; return its python.

    Several GB of pip, so this is the explicit install path. It is also reached
    from the first embedding request, as pyannote's is, because refusing for
    ever is worse than one slow request -- but an operator can press the button
    on the models page instead and keep that request fast.
    """
    global _bootstrapped
    def _say(msg):
        print(f"[jina-clip] {msg}", flush=True)
        if log:
            log(msg)

    with _lock:
        py = venv_python()
        if _bootstrapped and py.is_file():
            return py
        if py.is_file() and _transformers_ok(py):
            _bootstrapped = True
            return py
        venv = resolve_venv_dir()
        # create_venv REBUILDS what cannot be used rather than running
        # `python -m venv` over the top of it, which fails on the very thing
        # that broke it. It builds with THIS interpreter, which is correct here
        # and would not be for a worker pinning another python version.
        _say(f"building the isolated venv at {venv}")
        py = venv_env.create_venv(venv)
        if not _transformers_ok(py):
            _say("installing transformers 4.49 and torch into it (first run "
                 "downloads several GB)")
            venv_env.run([str(py), "-m", "pip", "install", "-U", "pip"],
                         check=True)
            venv_env.run([str(py), "-m", "pip", "install"]
                         + venv_env.pip_install_args(_REQUIREMENTS,
                                                     _FALLBACK_PINS),
                         check=True)
            if not _transformers_ok(py):
                raise RuntimeError(
                    "the jina-clip venv installed but does not import "
                    f"{', '.join(_PROBE_IMPORTS)} on transformers 4 — see "
                    f"{_REQUIREMENTS.name}")
        _bootstrapped = True
        _say(f"ready: {py}")
        return py


def _die_with_parent():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, 9)   # PR_SET_PDEATHSIG, SIGKILL
    except Exception:
        pass


def spawn(model_id: str, device: str = "cuda", load_timeout: float = 900.0,
          env_extra: dict = None) -> subprocess.Popen:
    """Start the server and return it once it has answered ``{"ready": true}``.

    Raises rather than returning a process that is still loading: the caller
    treats a returned handle as usable, and a half-loaded one would turn the
    first embedding request into the load error.
    """
    py = ensure_installed()
    if not _SERVER_SCRIPT.is_file():
        raise RuntimeError(f"jina-clip server script missing: {_SERVER_SCRIPT}")
    env = venv_env.env_for_venv(resolve_venv_dir(), extra=dict(env_extra or {}))
    env["JINACLIP_MODEL"] = str(model_id)
    if str(device) == "cpu":
        env["JINACLIP_FORCE_CPU"] = "1"
    proc = subprocess.Popen(
        [str(py), str(_SERVER_SCRIPT)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env, text=True, bufsize=1,
        preexec_fn=_die_with_parent)
    deadline = time.time() + float(load_timeout)
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            err = ""
            try:
                err = (proc.stderr.read() or "")[-800:]
            except Exception:
                pass
            raise RuntimeError(
                "the jina-clip server exited while loading"
                + (f": {err.strip()}" if err.strip() else ""))
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if msg.get("ready"):
            print(f"[jina-clip] server ready on {msg.get('device')} "
                  f"({msg.get('dim')}-d, transformers "
                  f"{msg.get('transformers')})", flush=True)
            return proc
        if msg.get("error"):
            proc.kill()
            raise RuntimeError(f"jina-clip server: {msg['error']}")
    proc.kill()
    raise RuntimeError(f"the jina-clip server did not load within "
                       f"{int(load_timeout)}s")


def request(proc: subprocess.Popen, payload: dict) -> list:
    """Send one request, return its embeddings. Raises on an error reply."""
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("the jina-clip server died mid-request")
        line = line.strip()
        if not line.startswith("{"):
            continue
        msg = json.loads(line)
        if "error" in msg:
            raise ValueError(f"jina-clip: {msg['error']}")
        if "embeddings" in msg:
            return msg["embeddings"]


def state() -> dict:
    """What the models page shows, without touching the network."""
    venv = resolve_venv_dir()
    installed = is_installed()
    return {
        "name": "jina-clip-v2",
        "venv": str(venv),
        "requirements": _REQUIREMENTS.name,
        "installed": installed,
        "available": is_available() if installed else False,
        "why": ("transformers 5 breaks the model's pinned remote code: it "
                "returns NaN image vectors. This venv holds transformers 4.49 "
                "and nothing else differs."),
    }
