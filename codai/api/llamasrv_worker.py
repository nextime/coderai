# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Fully-managed ``llama-server`` worker -- llama.cpp's own HTTP server, as a
subprocess, alongside the in-process llama-cpp-python path rather than instead
of it.

Why both. coderai serves GGUF through llama-cpp-python: ``libllama`` in-process,
which is the right default -- no second process, no port, no copy of the weights
in another address space. What it cannot reach is anything llama.cpp implements
*outside* the library. Speculative decoding is the one that matters: the
draft-and-verify loop lives in ``common/speculative.cpp``, and ``common`` is a
static library linked by the CLI tools, not part of ``libllama``. So a tree whose
headline feature is drafting can be driven only by running its server. The second
thing the server has is real slots (``-np``) with continuous batching, where the
in-process path serialises on a per-instance generation lock.

Lifecycle, mirroring :mod:`codai.api.ds4_worker`:

* :func:`ensure_built` materialises the chosen llama.cpp source tree and cmake-builds
  the ``llama-server`` target (idempotent -- it returns at once once the binary is there).
* :func:`ensure_service` launches it on a free loopback port with a per-launch bearer,
  waits for ``/health``, and returns the base URL.
* :func:`stop_service` is called from ``LlamaSrvBackend.cleanup()``, so the model
  manager's normal eviction tears the process down.

Which llama.cpp gets built is :attr:`LlamaSrvConfig.tree`:

``bindings``
    The tree vendored inside the installed llama-cpp-python sdist -- byte-identical
    to what the in-process path runs. This is deliberately the default: the server
    and the bindings are then one engine with two front ends, a GGUF behaves the
    same through either, and the only difference is batching. It is also what
    ``packaging/build-rpc-server.sh`` does, for the same no-version-skew reason.
``repo``
    ``repo_url`` at ``git_ref`` -- a fork. Tuned forks exist (and are the reason this
    backend has a ``tree`` knob at all), but they carry two hazards this module
    refuses to paper over: an undocumented upstream base, and a tuning target that
    is one GPU architecture. Hence the pinned ref and the architecture gate below.

Deliberately NOT done here: this binary is never registered as a ggml RPC peer.
The RPC protocol is versioned and a client and server from different llama.cpp
commits refuse each other, so :mod:`codai.backends.ggml_rpc` must keep talking to
the ``rpc-server`` built from the bindings' own tree -- see the header of
``packaging/build-rpc-server.sh``. A fork's tree is a different tree.
"""

import collections
import os
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

_lock = threading.RLock()
#: svc_key -> {"proc", "port", "url", "key"}. Keyed by the GGUF being served, so
#: two models pinned here each get their own server rather than fighting over one.
_services: dict = {}

#: The cmake target and the binary it produces.
_TARGET = "llama-server"


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def default_install_dir() -> Path:
    return Path(os.environ.get("CODERAI_LLAMASRV_DIR")
                or os.path.expanduser("~/.coderai/llamasrv"))


def _install_dir(cfg) -> Path:
    """Where this tree's source and build live.

    Per-tree, so switching ``tree`` (or bumping ``git_ref``) builds beside the old
    one instead of over it: a half-overwritten build directory is the kind of
    breakage that looks like a broken fork rather than a dirty tree.
    """
    base = (Path(cfg.install_dir).expanduser() if getattr(cfg, "install_dir", None)
            else default_install_dir())
    return base / _tree_slug(cfg)


def _tree_slug(cfg) -> str:
    tree = (getattr(cfg, "tree", "bindings") or "bindings").strip().lower()
    if tree != "repo":
        return "bindings"
    ref = (getattr(cfg, "git_ref", "") or "").strip() or "unpinned"
    url = (getattr(cfg, "repo_url", "") or "").strip()
    name = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git") or "repo"
    safe = "".join(c if (c.isalnum() or c in "-._") else "-" for c in f"{name}-{ref}")
    return safe[:80]


def bin_env_var(cfg) -> str:
    """The env var naming a PREBUILT binary for this engine.

    Per engine: an image bakes stock llama.cpp and a fork as two binaries, and
    one variable for both would silently serve whichever was baked last to
    every pinned model, defeating the split."""
    return f"CODERAI_{engine_key(cfg).upper()}_BIN"


def bin_env_vars(cfg) -> tuple:
    """Every env var that may name this engine's prebuilt binary, best first.

    An engine that was renamed keeps answering to its old variable: the name is
    derived from the engine key, so a rename silently stops an IMAGE from being
    found -- the pod would try to build the fork from source at runtime, in a
    runtime image with no CUDA toolchain, and fail long after the rename looked
    successful everywhere else.
    """
    names = [bin_env_var(cfg)]
    for old in getattr(cfg, "LEGACY_ENGINE_KEYS", ()) or ():
        names.append(f"CODERAI_{str(old).upper()}_BIN")
    return tuple(names)


def _env_bin_for(cfg) -> str:
    """The prebuilt binary this engine was given, from any of its env vars."""
    if cfg is None:
        return ""
    for var in bin_env_vars(cfg):
        val = (os.environ.get(var) or "").strip()
        if val:
            return val
    return ""


def _server_bin(install_dir: Path, cfg=None) -> Path:
    """Where the binary is.

    ``CODERAI_LLAMASRV_BIN`` wins: a capability image compiles ``llama-server``
    at image build time (with the CUDA toolchain, which the runtime image does
    not carry), so a pod must be told where it landed rather than deriving a
    path from a tree slug the image's build args chose. With it set, nothing is
    ever fetched or built on the pod.
    """
    env = _env_bin_for(cfg)
    if not env:
        # Back-compat: the single-engine variable, which named the stock server.
        env = (os.environ.get("CODERAI_LLAMASRV_BIN") or "").strip() \
            if (cfg is None or engine_key(cfg) == "llamasrv") else ""
    if env:
        return Path(env).expanduser()
    return install_dir / "build" / "bin" / _TARGET


# ---------------------------------------------------------------------------
# CUDA architecture
# ---------------------------------------------------------------------------

def resolve_architectures(cfg) -> Tuple[str, List[str]]:
    """``(cmake_arch_string, host_capabilities)`` for this build.

    Blank ``cuda_architectures`` means "whatever this box has", which is what you
    want on a workstation and on a rented pod alike. An explicit value is honoured
    as given -- and then checked against the host, because a tree tuned for one
    architecture aimed at a card it was never tested on does not fail cleanly: it
    builds, it loads, and it answers. ``require_matching_arch`` is on by default
    for exactly that reason.

    Returns ``("", caps)`` when there is no CUDA at all, which the caller turns
    into a CPU build rather than an error.
    """
    from codai.backends import gpu_probe
    caps = gpu_probe.compute_capabilities()
    want = (getattr(cfg, "cuda_architectures", "") or "").strip()

    if not want:
        if not caps:
            return "", caps
        # Collapse duplicates, keep order: two 3090s are one SASS target.
        seen, uniq = set(), []
        for c in caps:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return ";".join(uniq), caps

    asked = [a.strip() for a in want.replace(",", ";").split(";") if a.strip()]
    if getattr(cfg, "require_matching_arch", True):
        if not caps:
            raise RuntimeError(
                f"llamasrv.cuda_architectures is {want!r} but no CUDA card could be "
                "probed on this host, so the build cannot be checked against one. "
                "Install/repair the driver, or set require_matching_arch false to "
                "build anyway.")
        if not (set(asked) & set(caps)):
            raise RuntimeError(
                f"llamasrv.cuda_architectures is {want!r} but this host's cards are "
                f"{', '.join('sm_' + c for c in caps)}. A tree tuned for another "
                "architecture can load and answer with degraded or wrong output "
                "rather than failing, so this is refused. Set the right "
                "architecture, or require_matching_arch false to override.")
    return ";".join(asked), caps


# ---------------------------------------------------------------------------
# source trees
# ---------------------------------------------------------------------------

def _run_logged(cmd, cwd, label, tail, env=None):
    """Run a subprocess, streaming its output with a ``[llamasrv]`` prefix into ``tail``."""
    print(f"[llamasrv] $ {' '.join(str(c) for c in cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            print(f"[llamasrv] {line}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        joined = " | ".join(l.strip()[:300] for l in list(tail) if l.strip())
        raise RuntimeError(f"{label} failed (exit {proc.returncode}). {joined}")


def _bindings_version() -> str:
    from importlib.metadata import version
    return version("llama-cpp-python")


def _fetch_bindings_tree(dest: Path, tail) -> Path:
    """Extract the llama.cpp vendored in the installed llama-cpp-python's sdist.

    The wheel ships only the compiled library, so the C++ sources come from the
    sdist of the *same version*. That version pin is the whole point: the server
    built here then matches the library the in-process path already loaded.
    """
    ver = _bindings_version()
    print(f"[llamasrv] fetching the llama.cpp vendored in llama-cpp-python {ver} …",
          flush=True)
    with tempfile.TemporaryDirectory() as work:
        base = [sys.executable, "-m", "pip", "download", "--no-deps"]
        try:
            _run_logged(base + ["--no-binary", ":all:", "--no-build-isolation",
                                f"llama-cpp-python=={ver}", "-d", work],
                        cwd=work, label="pip download (sdist)", tail=tail)
        except RuntimeError:
            # --no-build-isolation fails on hosts without the build deps present;
            # the sdist itself is all we need, so retry without it.
            _run_logged(base + ["--no-binary", ":all:",
                                f"llama-cpp-python=={ver}", "-d", work],
                        cwd=work, label="pip download (sdist)", tail=tail)
        sdists = sorted(Path(work).glob("llama_cpp_python-*.tar.gz"))
        if not sdists:
            raise RuntimeError(
                f"no llama-cpp-python sdist downloaded for {ver}; cannot get the "
                "vendored llama.cpp. Set llamasrv.tree=repo to build from a clone "
                "instead.")
        with tarfile.open(sdists[0]) as tf:
            # Python 3.12+ warns without a filter; 'data' refuses paths that
            # escape the destination, which is what we want for a downloaded
            # archive regardless.
            try:
                tf.extractall(work, filter="data")
            except TypeError:        # Python < 3.12
                tf.extractall(work)
        found = sorted(Path(work).glob("llama_cpp_python-*/vendor/llama.cpp"))
        if not found or not (found[0] / "CMakeLists.txt").is_file():
            raise RuntimeError(
                f"the llama-cpp-python {ver} sdist has no vendor/llama.cpp tree to "
                "build (an sdist built without its submodules). Set "
                "llamasrv.tree=repo to build from a clone instead.")
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(found[0]), str(dest))
    return dest


def _fetch_repo_tree(cfg, dest: Path, tail) -> Path:
    url = (getattr(cfg, "repo_url", "") or "").strip()
    ref = (getattr(cfg, "git_ref", "") or "").strip()
    if not url:
        raise RuntimeError("llamasrv.tree is 'repo' but llamasrv.repo_url is empty.")
    if not ref:
        # Not fatal -- but say so loudly, because the build that comes out is not
        # reproducible and the next one may differ with no config change.
        print("[llamasrv] WARNING: llamasrv.git_ref is empty, so this builds "
              f"{url}'s default branch as it is right now. Pin a tag or a SHA: a "
              "fork's branch moves, and the build that breaks later will look like "
              "a broken fork rather than a moved ref.", flush=True)
    if not (dest / ".git").exists():
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # No --depth: a pinned SHA may not be the tip, and a shallow clone cannot
        # check it out. A tag-only pin would allow --branch, but one code path is
        # worth more here than one saved gigabyte.
        _run_logged(["git", "clone", url, str(dest)],
                    cwd=dest.parent, label="git clone", tail=tail)
    if ref:
        _run_logged(["git", "fetch", "--tags", "origin"], cwd=dest,
                    label="git fetch", tail=tail)
        _run_logged(["git", "checkout", "--detach", ref], cwd=dest,
                    label=f"git checkout {ref}", tail=tail)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(dest),
                          capture_output=True, text=True)
    if head.returncode == 0:
        print(f"[llamasrv] {url} at {head.stdout.strip()[:12]}"
              f"{f' ({ref})' if ref else ''}", flush=True)
    return dest


def ensure_source(cfg, tail=None) -> Path:
    tail = tail if tail is not None else collections.deque(maxlen=40)
    src = _install_dir(cfg) / "src"
    tree = (getattr(cfg, "tree", "bindings") or "bindings").strip().lower()
    if tree == "repo":
        return _fetch_repo_tree(cfg, src, tail)
    if (src / "CMakeLists.txt").is_file():
        return src
    return _fetch_bindings_tree(src, tail)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

_PREBUILT_LOGGED = set()


def _log_prebuilt_tree(binary: Path, cfg=None) -> None:
    """Say which llama.cpp a PREBUILT binary came from, once per path.

    When ``CODERAI_LLAMASRV_BIN`` is set, the ``tree``/``repo_url``/``git_ref``
    config describes a build that did not happen here: the binary was compiled
    into an image and nothing checks that it is the tree the config names. That
    gap is invisible at runtime -- a stock llama.cpp and an Ampere fork both
    answer, one of them without the speculative decoding the pin was for -- so
    the marker written beside the binary at image build time is logged instead
    of being left for someone to go and read.
    """
    var = bin_env_var(cfg) if cfg is not None else "CODERAI_LLAMASRV_BIN"
    if not ((os.environ.get(var) or "").strip()
            or (os.environ.get("CODERAI_LLAMASRV_BIN") or "").strip()):
        return
    key = str(binary)
    if key in _PREBUILT_LOGGED:
        return
    _PREBUILT_LOGGED.add(key)
    info = binary.parent.parent / "BUILD_INFO"
    try:
        detail = " ".join(info.read_text().split())
    except Exception:
        detail = "no BUILD_INFO beside it — which tree this is cannot be confirmed"
    print(f"[{engine_key(cfg) if cfg is not None else 'llamasrv'}] using the prebuilt "
          f"{_TARGET} at {binary} ({detail}); the tree/git_ref config is NOT what "
          "is serving", flush=True)


def ensure_built(cfg) -> Path:
    """cmake-build ``llama-server`` if it is missing. Returns the binary's path."""
    install_dir = _install_dir(cfg)
    binary = _server_bin(install_dir, cfg)
    if binary.is_file():
        _log_prebuilt_tree(binary, cfg)
        return binary
    env_bin = _env_bin_for(cfg) or (
        (os.environ.get("CODERAI_LLAMASRV_BIN") or "").strip()
        if engine_key(cfg) == "llamasrv" else "")
    if env_bin:
        # A prebuilt binary was promised and is not there. Building one here
        # would need a CUDA toolchain the runtime image does not carry, so say
        # what is actually wrong instead of failing later inside cmake.
        raise RuntimeError(
            f"CODERAI_LLAMASRV_BIN points at {env_bin}, which does not exist. "
            "This image is built with llama-server compiled in; if the path "
            "moved, fix the variable rather than building on the pod.")
    if not bool(getattr(cfg, "auto_build", True)):
        raise RuntimeError(
            f"{_TARGET} not found at {binary} and llamasrv.auto_build is disabled. "
            f"Build it by hand (cmake -S <llama.cpp> -B {install_dir / 'build'} "
            f"--target {_TARGET}) or enable auto_build.")
    if not shutil.which("cmake"):
        raise RuntimeError("llamasrv needs cmake to build llama-server, and cmake is "
                           "not on PATH.")

    # Resolve the architecture BEFORE fetching a tree: when the gate refuses, it
    # should refuse in a second rather than after a clone and a download.
    arch, caps = resolve_architectures(cfg)
    tail = collections.deque(maxlen=60)
    src = ensure_source(cfg, tail)
    build = install_dir / "build"

    cm = ["cmake", "-S", str(src), "-B", str(build),
          "-DCMAKE_BUILD_TYPE=Release",
          f"-DLLAMA_BUILD_SERVER=ON",
          "-DLLAMA_BUILD_TESTS=OFF",
          "-DLLAMA_BUILD_EXAMPLES=OFF",
          # GGML_NATIVE bakes in the build host's -march. On a workstation that is
          # free speed; in an image it is a binary that SIGILLs on an older CPU.
          "-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON"]
    if arch:
        cm += ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={arch}"]
        print(f"[llamasrv] building {_TARGET} with CUDA for sm_{arch.replace(';', ', sm_')}"
              f" (host: {', '.join('sm_' + c for c in caps) or 'no card probed'})",
              flush=True)
    else:
        print(f"[llamasrv] no CUDA card probed — building a CPU-only {_TARGET}. "
              "Generation will be slow; this is a fallback, not a configuration.",
              flush=True)
    extra = (getattr(cfg, "extra_cmake_args", "") or "").strip()
    if extra:
        cm += shlex.split(extra)

    _run_logged(cm, cwd=install_dir, label="cmake configure", tail=tail)
    jobs = str(max(1, (os.cpu_count() or 4) - 1))
    _run_logged(["cmake", "--build", str(build), "-j", jobs, "--target", _TARGET],
                cwd=install_dir, label="cmake build", tail=tail)

    if not binary.is_file():
        # Some trees put tools elsewhere; look before giving up.
        for cand in build.rglob(_TARGET):
            if cand.is_file() and os.access(cand, os.X_OK):
                print(f"[llamasrv] built {cand}", flush=True)
                return cand
        raise RuntimeError(
            f"the build finished but {binary} is missing. Last output: "
            + " | ".join(l.strip()[:300] for l in list(tail) if l.strip()))
    print(f"[llamasrv] built {binary}", flush=True)
    return binary


# ---------------------------------------------------------------------------
# service
# ---------------------------------------------------------------------------

def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _pump_logs(proc, tail):
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            print(f"[llamasrv] {line}", flush=True)


def _health_ok(url: str, key: str = "") -> bool:
    import requests
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        r = requests.get(url + "/health", headers=headers, timeout=3)
        # llama-server answers 503 with {"status":"loading model"} while it loads:
        # reachable but not ready, which is not the same as not there.
        return r.status_code == 200
    except Exception:
        return False


def engine_key(cfg) -> str:
    """Which llama-server engine this config object configures.

    ``llamasrv`` (stock llama.cpp) and ``llamampere`` are separate engines
    sharing this worker, so everything keyed per running server has to be keyed
    per engine as well."""
    return getattr(cfg, "ENGINE_KEY", "llamasrv") or "llamasrv"


def resolve_service_key(cfg, model_file: Optional[str] = None):
    """``(gguf_path, svc_key)`` -- which GGUF to serve and what to cache it under.

    The key carries the ENGINE, not just the file. The whole point of a second
    engine is serving one GGUF on stock llama.cpp and the same GGUF on a fork at
    the same time; keyed by path alone the second request would be handed the
    first engine's running server, and a comparison between them would silently
    be a comparison of one with itself."""
    resolved = ""
    cand = os.path.expanduser((model_file or "").strip())
    if cand and os.path.isfile(cand):
        resolved = cand
    base = resolved or (getattr(cfg, "model_id", "") or "llamasrv")
    return resolved, f"{engine_key(cfg)}::{base}"


def build_argv(cfg, binary, model_file: str, port: int, host: str, key: str,
               ctx: Optional[int] = None, mmproj: Optional[str] = None) -> List[str]:
    """The ``llama-server`` command line. Split out so a test can read it without
    building or launching anything."""
    cmd = [str(binary), "--host", host, "--port", str(port), "-m", model_file]
    if key:
        cmd += ["--api-key", key]

    try:
        ctx_val = int(ctx) if ctx else 0
    except (TypeError, ValueError):
        ctx_val = 0
    if ctx_val <= 0:
        ctx_val = int(getattr(cfg, "ctx", 0) or 0)
    if ctx_val > 0:
        cmd += ["-c", str(ctx_val)]

    ngl = int(getattr(cfg, "n_gpu_layers", -1) or -1)
    cmd += ["-ngl", "999" if ngl < 0 else str(ngl)]

    par = int(getattr(cfg, "parallel", 0) or 0)
    if par > 0:
        cmd += ["-np", str(par)]

    fa = (getattr(cfg, "flash_attn", "on") or "").strip().lower()
    if fa in ("on", "off", "auto"):
        cmd += ["-fa", fa]

    for flag, attr in (("-ctk", "cache_type_k"), ("-ctv", "cache_type_v")):
        val = (getattr(cfg, attr, "") or "").strip()
        if val:
            cmd += [flag, val]

    spec = (getattr(cfg, "spec_type", "") or "").strip()
    if spec:
        cmd += ["--spec-type", spec]
    n_max = int(getattr(cfg, "spec_draft_n_max", 0) or 0)
    if n_max > 0:
        cmd += ["--spec-draft-n-max", str(n_max)]

    extra = (getattr(cfg, "extra_args", "") or "").strip()
    extra_argv = shlex.split(extra) if extra else []

    # The vision projector. A multimodal GGUF without it is a text model that
    # accepts an image and describes nothing: the tower is a separate file, and
    # llama-server loads it only when told. The model entry already names the
    # file for the in-process path, so the same entry drives both and a model
    # does not lose the ability to see by being pinned to this engine.
    #
    # Skipped when extra_args already names one, so an operator pointing at a
    # different projector is not handed the flag twice -- llama-server takes the
    # last one, which would silently beat the explicit override.
    mm = (mmproj or "").strip()
    if mm and not any(t in ("--mmproj", "-mm") for t in extra_argv):
        cmd += ["--mmproj", mm]

    if extra_argv:
        cmd += extra_argv
    return cmd


def service_bearer(svc_key: str) -> str:
    """The bearer the running server for ``svc_key`` requires, or ''."""
    with _lock:
        svc = _services.get(svc_key)
        return (svc or {}).get("key", "")


def ensure_service(cfg, model_file: Optional[str] = None,
                   ctx: Optional[int] = None,
                   model_name: Optional[str] = None,
                   ready_timeout: Optional[float] = None,
                   mmproj: Optional[str] = None) -> Tuple[str, str]:
    """Build as needed, then start (or reuse) ``llama-server`` for this GGUF.

    Returns ``(base_url, bearer)``. The bearer is minted per launch unless one is
    configured: the port is loopback, but a port is not an authorisation, and
    another local process should not be able to drive the model by finding it.
    """
    remote = (str(getattr(cfg, "service_url", "") or "")
              or os.environ.get("CODERAI_LLAMASRV_SERVICE_URL") or "").strip()
    if remote:
        remote = remote.rstrip("/")
        key = (getattr(cfg, "api_key", "") or "").strip()
        if not _health_ok(remote, key):
            raise RuntimeError(
                f"llamasrv.service_url {remote} is not answering /health")
        print(f"[llamasrv] using the configured remote server at {remote}", flush=True)
        return remote, key

    resolved, svc_key = resolve_service_key(cfg, model_file)
    if not resolved:
        raise RuntimeError(
            "llamasrv serves a local GGUF and none resolved for this request. "
            "Point the model's path at a .gguf file. Unlike the in-process path "
            "this backend does not download weights.")

    timeout = float(ready_timeout if ready_timeout is not None
                    else (getattr(cfg, "ready_timeout_s", 1800) or 1800))
    with _lock:
        svc = _services.get(svc_key)
        if svc and svc["proc"].poll() is None and _health_ok(svc["url"], svc["key"]):
            return svc["url"], svc["key"]
        if svc and svc["proc"].poll() is not None:
            _services.pop(svc_key, None)      # died — relaunch below

        binary = ensure_built(cfg)
        host = (getattr(cfg, "host", "127.0.0.1") or "127.0.0.1").strip()
        port = int(getattr(cfg, "port", 0) or 0) or _free_port()
        connect_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        url = f"http://{connect_host}:{port}"
        key = (getattr(cfg, "api_key", "") or "").strip() or secrets.token_urlsafe(24)

        cmd = build_argv(cfg, binary, resolved, port, host, key, ctx, mmproj)
        env = os.environ.copy()
        applied = {}
        extra_env = (getattr(cfg, "extra_env", "") or "").strip()
        if extra_env:
            for tok in shlex.split(extra_env):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    if k.strip():
                        env[k.strip()] = v
                        applied[k.strip()] = v
        note = ("  (" + " ".join(f"{k}={v}" for k, v in applied.items()) + ")"
                if applied else "")
        # The bearer is redacted: this line goes to the engine log, which is
        # tailed on the Tasks page and shipped in support bundles.
        shown = " ".join("***" if i and cmd[i - 1] == "--api-key" else c
                         for i, c in enumerate(cmd))
        print(f"[llamasrv] launching: {shown}{note}", flush=True)
        # PR_SET_PDEATHSIG: the kernel kills this server if the engine that owns
        # it dies. VRAM eviction works through the owning engine -- it stops the
        # service and the subprocess frees the card -- so an engine that is
        # SIGKILLed (OOM killer, crash, `pkill`) would otherwise leave a
        # llama-server holding the whole model's VRAM with nothing left that
        # knows how to evict it, and the next load OOMs against a ghost.
        # atexit covers a clean exit; only this covers the unclean one.
        def _die_with_parent():
            try:
                import ctypes
                ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 9)  # PR_SET_PDEATHSIG, SIGKILL
            except Exception:
                pass

        proc = subprocess.Popen(cmd, cwd=str(_install_dir(cfg)),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env,
                                preexec_fn=_die_with_parent)
        tail = collections.deque(maxlen=20)
        threading.Thread(target=_pump_logs, args=(proc, tail), daemon=True).start()
        _services[svc_key] = {"proc": proc, "port": port, "url": url, "key": key}

    def _tail_msg():
        joined = " | ".join(l.strip()[:300] for l in list(tail) if l.strip()).strip()
        return f". Last output: {joined}" if joined else ""

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            _services.pop(svc_key, None)
            raise RuntimeError(
                f"{_TARGET} exited (code {proc.returncode}) before becoming ready"
                + _tail_msg())
        if _health_ok(url, key):
            print(f"[llamasrv] ready for {os.path.basename(svc_key)} at {url}",
                  flush=True)
            return url, key
        time.sleep(2)
    stop_service(svc_key)
    raise RuntimeError(f"{_TARGET} for {svc_key} did not become ready within "
                       f"{timeout:.0f}s" + _tail_msg())


def stop_service(svc_key: str) -> bool:
    """Stop the server for ``svc_key``; True if one was running.

    This is the VRAM eviction path. The manager evicts a llamasrv-served model
    by dropping the backend and calling its ``cleanup()``, which lands here, and
    the card is only free once this process is GONE -- not once it has been
    signalled. So the wait after kill is not a courtesy: eviction measures what
    it freed and the next load starts immediately after, and both would race a
    process that is still dying with the weights still mapped.
    """
    with _lock:
        svc = _services.pop(svc_key, None)
    if not svc:
        # Not nothing-to-do: the caller believes it evicted a model. A key that
        # matches no service means the VRAM is still held by a server nobody is
        # now tracking, which is worth a line in the log rather than silence.
        if _services:
            print(f"[llamasrv] stop_service({svc_key!r}) matched no running server; "
                  f"running: {sorted(_services)}", flush=True)
        return False
    proc = svc["proc"]
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
            # Wait for the kill, too. Returning here with the process still
            # dying is what makes an eviction look like it freed nothing.
            proc.wait(timeout=10)
        except Exception:
            pass
    if proc.poll() is None:
        print(f"[llamasrv] WARNING: server for {os.path.basename(svc_key)} did not exit; "
              "its VRAM is still held", flush=True)
        return False
    print(f"[llamasrv] server for {os.path.basename(svc_key)} stopped", flush=True)
    return True


def stop_all() -> None:
    for key in list(_services.keys()):
        stop_service(key)


import atexit as _atexit
_atexit.register(stop_all)
