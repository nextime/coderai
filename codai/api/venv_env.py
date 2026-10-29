# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Spawning a DIFFERENT Python than this process's: its environment, its venv.

Every isolated-venv worker runs an interpreter that is not the server's —
3.10 for LongCat and the lip-sync tools, 3.12 for ACE-Step, against a server
on 3.13 — or the server's own, in a venv whose packages must not be shadowed
by it (coqui-tts, parler-tts, NeMo, pyannote, CrisperWhisper). Anything in the
environment that points a Python at another installation's files makes that
child die at startup, or quietly import the wrong packages, before it can say
anything useful.

Both halves of the problem live here: what environment a child gets
(:func:`clean_py_env`, :func:`env_for_venv`) and whether the venv it is about
to run in can be used at all (:func:`venv_usable`, :func:`create_venv`). They
were separate once, and the three failures in
``docs/isolated-worker-environments.md`` are what that cost.

This is not hypothetical and it is not only about venvs the server builds:
in the shipped container the per-engine launchers set
``PYTHONHOME=/opt/coderai/python`` (the standalone 3.13), while PID 1 and an
interactive ``docker exec`` do not. A worker spawned from an engine process
therefore inherits it and a 3.12 interpreter cannot find its own stdlib,
while the same command run by hand in the same container works perfectly.
The observed symptom was a probe failing in milliseconds and the caller
reporting something else entirely.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

#: Variables that aim a Python at another installation's files. PYTHONHOME is
#: the one that is actually set in the container; the rest are stripped because
#: the failure they cause is fatal, silent in the parent, and costs nothing to
#: prevent.
_LEAKY = ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE",
          "VIRTUAL_ENV", "PYTHONNOUSERSITE", "PYTHONUSERBASE",
          "__PYVENV_LAUNCHER__")


#: Extra library directories that belong to THIS interpreter even though they
#: are not under its prefix. The container launcher puts the ldd-discovered
#: natives here, and they are built against this python's ABI.
#: `/opt/coderai/python` is named explicitly as well: it is the standalone
#: interpreter the image ships, and nothing on a different python may load from
#: it. (Only foreign interpreters go through this helper — the 3.13 venvs
#: pyannote/canary/crisperwhisper run on build their own env, and they do need
#: that directory for libpython.)
_OWN_LIB_DIRS = ("/opt/coderai/local-libs", "/opt/coderai/python")


#: Prefixes that are the WHOLE system rather than one installation. When
#: coderai runs from a distribution python (a pip install rather than the
#: image), ``sys.prefix`` is one of these, and treating it as "ours" would
#: strip ``/usr/local/nvidia/lib`` from every worker -- taking the GPU away
#: from the engines this helper exists to keep working.
_SYSTEM_PREFIXES = ("/", "/usr", "/usr/local")


def _is_our_lib_dir(entry: str) -> bool:
    """Is this LD_LIBRARY_PATH entry part of the server's own install?"""
    try:
        real = os.path.realpath(entry)
    except Exception:
        return False
    roots = [os.path.realpath(p) for p in (sys.prefix, sys.base_prefix)
             if p and os.path.realpath(p) not in _SYSTEM_PREFIXES]
    roots += [os.path.realpath(d) for d in _OWN_LIB_DIRS]
    return any(real == r or real.startswith(r + os.sep) for r in roots)


def clean_py_env(base: dict = None) -> dict:
    """Return ``base`` (default: this process's environment) without them.

    ``LD_LIBRARY_PATH`` is filtered rather than dropped: the container launcher
    exports ``/opt/coderai/python/lib:/opt/coderai/local-libs:<the server
    venv's nvidia wheels>``, and a foreign interpreter that inherits it loads
    THIS python's extension modules. Measured: SadTalker's python 3.10 picked up
    the 3.13 build and died with

        ImportError: /opt/coderai/local-libs/libtorch_python.so:
        undefined symbol: PyObject_GetOptionalAttrString

    -- a CPython 3.13 API. Entries outside the server's install (a system CUDA,
    anything the operator added) are left alone: an isolated venv ships its own
    torch and its own CUDA wheels, and those are the ones it must find.
    """
    env = dict(os.environ if base is None else base)
    for var in _LEAKY:
        env.pop(var, None)
    libs = env.get("LD_LIBRARY_PATH")
    if libs:
        keep = [e for e in libs.split(os.pathsep) if e and not _is_our_lib_dir(e)]
        if keep:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(keep)
        else:
            env.pop("LD_LIBRARY_PATH", None)
    return env


def clean_py_env_with(extra: Optional[dict] = None,
                       base: Optional[dict] = None) -> dict:
    """:func:`clean_py_env` with ``extra`` applied on top.

    ``extra`` is last, so a caller can still set what it needs (a service's
    ``PYTHONUNBUFFERED``, a licence acceptance, ``CUDA_VISIBLE_DEVICES``).
    Passing ``PYTHONPATH`` there is honoured -- a worker that genuinely needs
    one said so on purpose.
    """
    env = clean_py_env(base)
    if extra:
        env.update({k: str(v) for k, v in extra.items() if v is not None})
    return env


def hf_child_env(extra: Optional[dict] = None) -> dict:
    """:func:`child_env` plus coderai's HF cache variables.

    A worker never imports ``codai``, so without these it downloads a second
    copy of every model into whatever directory the image happened to set.
    """
    env = clean_py_env_with(extra)
    try:
        from codai.hf_cache import child_env as _hf
        env = _hf(env)
    except Exception:
        pass
    if extra:                       # the caller's values still win
        env.update({k: str(v) for k, v in extra.items() if v is not None})
    return env


def built_by_this_interpreter(venv_dir) -> bool:
    """Whether ``venv_dir`` was built by the interpreter running this process.

    It decides one thing: whether the child still needs the server's own
    library directories. A venv built from ``/opt/coderai/python`` runs that
    very interpreter and must find its ``libpython``, so stripping those
    directories would stop it starting at all -- the opposite of the problem
    they are stripped for, which is a FOREIGN python (3.10, 3.11, 3.12)
    loading this one's extension modules.
    """
    cfg = _cfg_values(venv_dir)
    home = cfg.get("home") or ""
    if not home:
        return False
    try:
        home_real = os.path.realpath(home)
        for base in (sys.prefix, sys.base_prefix):
            if not base:
                continue
            base_real = os.path.realpath(base)
            if home_real == base_real or home_real.startswith(base_real + os.sep):
                return True
    except Exception:
        return False
    return False


def env_for_venv(venv_dir, extra: Optional[dict] = None,
                 hf: bool = True) -> dict:
    """The environment to run something inside ``venv_dir`` with.

    Always drops what redirects an interpreter (``PYTHONHOME`` above all: it is
    exported by the engine processes, and under it a venv python takes its
    stdlib and site-packages from the SERVER's installation, so the venv's own
    packages are not on ``sys.path`` at all).

    ``LD_LIBRARY_PATH`` depends on whose interpreter the venv holds -- see
    :func:`built_by_this_interpreter`. Keeping it for a foreign python breaks
    it; stripping it for one of ours breaks it. Deciding per venv is the only
    way both work.
    """
    if built_by_this_interpreter(venv_dir):
        base = dict(os.environ)
        for var in _LEAKY:
            base.pop(var, None)
        env = base
        if extra:
            env.update({k: str(v) for k, v in extra.items() if v is not None})
    else:
        env = clean_py_env_with(extra)
    if hf:
        try:
            from codai.hf_cache import child_env as _hf
            env = _hf(env)
        except Exception:
            pass
        if extra:
            env.update({k: str(v) for k, v in extra.items() if v is not None})
    return env


def env_for_python(py, extra: Optional[dict] = None, hf: bool = True) -> dict:
    """:func:`env_for_venv`, given the venv's interpreter instead of its root."""
    return env_for_venv(Path(py).parent.parent, extra, hf)


def venv_python(venv_dir) -> Path:
    """The interpreter inside ``venv_dir`` (platform-correct)."""
    d = Path(venv_dir)
    return d / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")


def run(cmd: Iterable[str], **kwargs) -> subprocess.CompletedProcess:
    """``subprocess.run`` for a command that runs in a venv.

    The environment is built for the venv the command's interpreter belongs to
    (:func:`env_for_python`), which is what decides whether the server's own
    library directories are kept. ``python -m venv`` is the exception: its
    interpreter is THIS one, so it is given this process's cleaned
    environment.
    """
    cmd = [str(c) for c in cmd]
    extra = kwargs.pop("env_extra", None)
    if "env" not in kwargs:
        first = cmd[0] if cmd else ""
        if first == sys.executable or "venv" in cmd[1:3]:
            kwargs["env"] = clean_py_env_with(extra)
        else:
            kwargs["env"] = env_for_python(first, extra, hf=False)
    return subprocess.run(cmd, **kwargs)


def interpreter_version(py) -> Optional[Tuple[int, int]]:
    """``(major, minor)`` of the interpreter at ``py``, or None if it will not run."""
    try:
        out = run([str(py), "-c",
                   "import sys;print('%d.%d' % sys.version_info[:2])"],
                  capture_output=True, text=True, timeout=120)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    try:
        major, minor = (out.stdout or "").strip().split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return None


def _cfg_values(venv_dir) -> dict:
    """``pyvenv.cfg`` as a dict (lowercased keys), or empty.

    ``version_info`` is read as ``version``: both spellings appear in the
    wild (which one depends on the python that wrote the file), and a venv
    whose version cannot be read is judged as though it had none.
    """
    out = {}
    try:
        for line in (Path(venv_dir) / "pyvenv.cfg").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip().lower()] = v.strip()
    except Exception:
        return {}
    if "version" not in out and out.get("version_info"):
        out["version"] = out["version_info"]
    return out


def venv_usable(venv_dir, system_site_ok: bool = False,
                probe: bool = True) -> Tuple[bool, str]:
    """``(usable, reason)`` for a venv this process is about to use.

    Four ways a venv that *looks* present is not:

    * its ``bin/python`` is a symlink to an interpreter that no longer exists
      (a venv built on the host, used in the container, or built before the
      image moved its interpreter). ``Path.exists()`` follows the link and
      answers False, so the caller concludes "missing" and runs ``python -m
      venv`` over the top, which trips on the same dangling link and dies;
    * the interpreter is there but it is a *different minor version* than the
      one its packages were installed for. ``lib/python3.13/site-packages`` is
      invisible to a 3.12 interpreter, so every import fails while 875 MB of
      packages sit right there -- measured on zeiss, 2026-10-29, on an
      ``xtts_venv`` built by the host's 3.13 against the container's
      ``/usr/bin/python3``;
    * it was built with ``--system-site-packages`` while the caller wanted
      isolation, which defeats the only reason these venvs exist -- the
      server's incompatible stack leaks straight back in. ``system_site_ok``
      is for the workers that inherit on purpose (the MiniMax-H3 overlay takes
      the server's torch rather than downloading another one);
    * it runs nothing at all, which only running it can prove.

    Each of those wants the venv rebuilt, which is why they come back as one
    answer with a reason to log.

    ``probe=False`` decides from the filesystem alone, for callers that must
    not spend a subprocess -- a catalogue route answering "is this engine
    ready" is called often and must stay cheap.
    """
    d = Path(venv_dir)
    py = venv_python(d)
    if not d.exists():
        return False, "it does not exist"

    cfg = _cfg_values(d)
    # The recorded interpreter, checked without running it: a dangling symlink
    # chain, or a `home` that no longer holds the version this venv was built
    # with (the container's /usr/bin has 3.12 where the host's had 3.13).
    if not py.is_file():
        return False, ("its python does not run here (built by a different "
                       "interpreter)")
    cfg_ver = ".".join((cfg.get("version") or "").split(".")[:2])
    if cfg_ver and cfg.get("home"):
        home_exe = Path(cfg["home"]) / f"python{cfg_ver}"
        if not home_exe.exists() and not Path(cfg["home"]).joinpath("python").exists():
            return False, (f"the interpreter it was built with "
                           f"(python{cfg_ver} in {cfg['home']}) is not here")

    packages = sorted(p.name for p in (d / "lib").glob("python*"))         if (d / "lib").is_dir() else []
    if cfg_ver and packages and f"python{cfg_ver}" not in packages:
        return False, (f"its python is {cfg_ver} but its packages are installed "
                       f"for {', '.join(packages)}")

    shared = (cfg.get("include-system-site-packages") or "").lower() == "true"
    if shared and not system_site_ok:
        return False, "it was built with --system-site-packages"
    if not shared and system_site_ok:
        return False, ("it was built isolated but this worker overlays the "
                       "server's packages")

    if not probe:
        return True, ""

    ver = interpreter_version(py)
    if ver is None:
        return False, ("its python does not run here (built by a different "
                       "interpreter)")
    want = d / "lib" / f"python{ver[0]}.{ver[1]}" / "site-packages"
    if os.name != "nt" and not want.is_dir() and packages:
        return False, (f"its python is {ver[0]}.{ver[1]} but its packages "
                       f"are installed for {', '.join(packages)}")
    return True, ""


def create_venv(venv_dir, system_site: bool = False,
                timeout: float = 600) -> Path:
    """Create ``venv_dir`` with a clean environment and return its python.

    Rebuilds from scratch when what is there cannot be used -- see
    :func:`venv_usable`. A venv is never repaired in place: ``python -m venv``
    over a broken one fails on the very thing that broke it.

    The rebuild uses THIS interpreter, so a worker that needs a particular
    python version must not come through here -- ACE-Step pins 3.12 (spacy
    3.8.4), LatentSync and MuseTalk 3.11, the lipsync engine 3.10 -- because it
    would be rebuilt with the server's version and its packages would not
    install. Each of those builds its venv with an explicit interpreter of its
    own, and none of them calls this. A venv built by a deliberately different
    python is otherwise left alone, as long as that python is still where its
    ``pyvenv.cfg`` says it is.
    """
    import shutil
    d = Path(venv_dir)
    if d.exists():
        ok, why = venv_usable(d, system_site_ok=system_site)
        if not ok:
            print(f"[worker-env] rebuilding {d}: {why}", flush=True)
            shutil.rmtree(d, ignore_errors=True)
    if not venv_python(d).is_file():
        d.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "venv"]
        if system_site:
            cmd.append("--system-site-packages")
        run(cmd + [str(d)], check=True, timeout=timeout)
    return venv_python(d)

def pip_install_args(requirements, fallback_pins) -> list:
    """Pip arguments for a worker's install: the requirements file, or its pins.

    The file is the source of truth and is preferred whenever it is there. It
    is not always there: the container bind-mounts ``codai/``, ``tools/`` and
    ``third_party/`` over the image, so a repo-ROOT file added after the
    running image was built is simply absent until the container is recreated
    — and a self-healing rebuild that stops with "requirements file missing"
    is a worker that stays broken for a reason that has nothing to do with it.
    Measured twice on zeiss: once as ACE-Step reporting a missing
    ``requirements-acestep.txt`` for a venv that was fine, and once as a
    Parler rebuild refusing to run.

    ``fallback_pins`` must mirror the file. Each worker that uses this keeps a
    test that parses its file and compares, because two copies of a pin drift.
    """
    from pathlib import Path as _P
    req = _P(requirements)
    if req.is_file():
        return ["-r", str(req)]
    print(f"[worker-env] {req} is not in this install (the container mounts "
          f"only codai/, tools/ and third_party/ over the image); installing "
          f"the same pins inline", flush=True)
    return list(fallback_pins)
