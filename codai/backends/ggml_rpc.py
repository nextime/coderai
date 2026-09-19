# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Cards on other machines as ggml devices, for the in-process GGUF backend.

llama.cpp's RPC backend turns a ``host:port`` where ``rpc-server`` listens
into a device like any CUDA or Vulkan card: register it and the layer split
spreads a model over it. ``llama-server --rpc`` does exactly that; the
llama-cpp-python binding this backend loads through never exposed it, so this
module does the same three calls through ctypes against the ggml libraries
the binding ships:

1. ``ggml_backend_reg_by_name("RPC")`` — present only when the binding was
   built with ``-DGGML_RPC=ON`` (build.sh / the images do that);
2. ``ggml_backend_rpc_add_server(endpoint)`` (or the older ``add_device``),
   fetched through ``ggml_backend_reg_get_proc_address`` like llama.cpp's own
   CLI does, so the symbol need not be exported;
3. ``ggml_backend_device_register`` for each device the server offers.

Registration is process-wide and permanent. A model loaded with no device
list would then spread over every RPC device ever registered, so from the
first registration on, every load gets an explicit list: the local cards,
plus exactly the RPC devices the model asked for (:func:`load_with_devices`).
The split order llama.cpp sees is that list's order — local cards first,
then RPC servers as listed — which is how a model's ``tensor_split`` reads.
"""

import ctypes
import glob
import os
import threading
from typing import Dict, List, Optional, Tuple

_lock = threading.RLock()
_libs = None                       # (base, ggml)
_rpc_devs: Dict[str, List[int]] = {}      # endpoint -> device handles
_dev_names: Dict[int, str] = {}
_hooked = False
_tls = threading.local()


class _Ggml:
    """The ggml entry points we use, each found in whichever of the shipped
    libraries exports it (llama-cpp-python splits them between libggml and
    libggml-base) — and typed once."""

    _PROTOS = {
        "ggml_backend_reg_by_name": (ctypes.c_void_p, [ctypes.c_char_p]),
        "ggml_backend_reg_get_proc_address": (ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_char_p]),
        "ggml_backend_reg_dev_count": (ctypes.c_size_t, [ctypes.c_void_p]),
        "ggml_backend_reg_dev_get": (ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_size_t]),
        "ggml_backend_device_register": (None, [ctypes.c_void_p]),
        "ggml_backend_dev_count": (ctypes.c_size_t, []),
        "ggml_backend_dev_get": (ctypes.c_void_p, [ctypes.c_size_t]),
        "ggml_backend_dev_name": (ctypes.c_char_p, [ctypes.c_void_p]),
        "ggml_backend_dev_memory": (None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                          ctypes.POINTER(ctypes.c_size_t)]),
        "ggml_backend_load_all": (None, []),
    }

    def __init__(self, libs):
        self._libs = libs
        self._fns = {}
        for name, (res, args) in self._PROTOS.items():
            for lib in libs:
                try:
                    fn = getattr(lib, name)
                except AttributeError:
                    continue
                fn.restype, fn.argtypes = res, args
                self._fns[name] = fn
                break
        missing = [n for n in self._PROTOS if n not in self._fns and n != "ggml_backend_load_all"]
        if missing:
            raise RuntimeError(f"ggml symbols not found in the bundled libraries: {missing}")

    def __getattr__(self, name):
        try:
            return self._fns[name]
        except KeyError:
            raise AttributeError(name)


def _load_libs():
    global _libs
    if _libs is not None:
        return _libs
    import llama_cpp
    libdir = os.path.join(os.path.dirname(llama_cpp.__file__), "lib")
    libs = []
    for pat in ("libggml.so*", "libggml-base.so*", "libggml.dylib", "libggml-base.dylib"):
        for cand in sorted(glob.glob(os.path.join(libdir, pat))):
            try:
                libs.append(ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL))
                break
            except OSError:
                continue
    if not libs:
        raise RuntimeError(f"libggml not found under {libdir}")
    g = _Ggml(libs)
    # Make sure every linked backend has registered itself (no-op when static).
    try:
        g.ggml_backend_load_all()
    except Exception:
        pass
    _libs = (g, g)
    return _libs


def available() -> Tuple[bool, str]:
    """(True, "") when the bundled llama.cpp has the RPC backend; else why not."""
    try:
        base, _ = _load_libs()
    except Exception as exc:
        return False, str(exc)
    try:
        reg = base.ggml_backend_reg_by_name(b"RPC")
    except Exception as exc:
        return False, str(exc)
    if not reg:
        return False, ("the bundled llama-cpp-python was built without GGML_RPC — "
                       "rebuild it with CMAKE_ARGS including -DGGML_RPC=ON "
                       "(build.sh does)")
    return True, ""


def normalize_endpoints(raw) -> List[str]:
    """``"host:port, host2:port"`` / a list → clean, de-duplicated list."""
    if not raw:
        return []
    items = raw if isinstance(raw, (list, tuple)) else str(raw).replace("\n", ",").split(",")
    out = []
    for x in items:
        x = str(x).strip()
        if x and ":" in x and x not in out:
            out.append(x)
    return out


def _dev_name(dev: int) -> str:
    base, _ = _load_libs()
    try:
        return (base.ggml_backend_dev_name(dev) or b"").decode("utf-8", "replace")
    except Exception:
        return ""


def register_servers(endpoints) -> Dict[str, List[str]]:
    """Register RPC endpoints (idempotent); returns endpoint -> device names.

    Raises with a plain message when the backend is missing or a server does
    not answer — a load must fail loudly here rather than silently run local.
    """
    eps = normalize_endpoints(endpoints)
    if not eps:
        return {}
    ok, why = available()
    if not ok:
        raise RuntimeError(f"rpc_servers configured but unusable: {why}")
    out = {}
    base, _ = _load_libs()
    with _lock:
        reg = base.ggml_backend_reg_by_name(b"RPC")
        add_server = base.ggml_backend_reg_get_proc_address(reg, b"ggml_backend_rpc_add_server")
        add_device = base.ggml_backend_reg_get_proc_address(reg, b"ggml_backend_rpc_add_device")
        for ep in eps:
            if ep in _rpc_devs:
                out[ep] = [_dev_names.get(d, "") for d in _rpc_devs[ep]]
                continue
            devs = []
            if add_server:
                fn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)(add_server)
                sreg = fn(ep.encode())
                if sreg:
                    n = base.ggml_backend_reg_dev_count(sreg)
                    for i in range(n):
                        dev = base.ggml_backend_reg_dev_get(sreg, i)
                        if dev:
                            base.ggml_backend_device_register(dev)
                            devs.append(dev)
            elif add_device:
                fn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)(add_device)
                dev = fn(ep.encode())
                if dev:
                    base.ggml_backend_device_register(dev)
                    devs.append(dev)
            else:
                raise RuntimeError("the RPC backend exposes neither add_server nor "
                                   "add_device — llama.cpp too old or too new for this shim")
            if not devs:
                raise RuntimeError(f"rpc-server at {ep} did not answer (is it running, "
                                   f"is the port reachable, do the llama.cpp versions match?)")
            for d in devs:
                _dev_names[d] = _dev_name(d)
            _rpc_devs[ep] = devs
            out[ep] = [_dev_names[d] for d in devs]
            print(f"[rpc] registered {ep}: {', '.join(out[ep])}", flush=True)
    _install_hook()
    return out


def registered() -> Dict[str, List[str]]:
    with _lock:
        return {ep: [_dev_names.get(d, "") for d in devs] for ep, devs in _rpc_devs.items()}


def device_memory(endpoint: str) -> List[Tuple[int, int]]:
    """(free, total) bytes of each device an endpoint offers."""
    base, _ = _load_libs()
    out = []
    for dev in _rpc_devs.get(endpoint, []):
        free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
        try:
            base.ggml_backend_dev_memory(dev, ctypes.byref(free), ctypes.byref(total))
            out.append((int(free.value), int(total.value)))
        except Exception:
            out.append((0, 0))
    return out


def free_gb(endpoints) -> List[float]:
    """Free memory in GB, one number per RPC device, in endpoint order."""
    out = []
    for ep in normalize_endpoints(endpoints):
        for free, _total in device_memory(ep):
            out.append(free / (1024 ** 3))
    return out


def _all_rpc_handles() -> set:
    return {d for devs in _rpc_devs.values() for d in devs}


def _local_gpu_devices() -> List[int]:
    """Every registered device that is a card of this machine: not CPU, not an
    accelerator library, not an RPC peer."""
    base, _ = _load_libs()
    rpc = _all_rpc_handles()
    out = []
    for i in range(base.ggml_backend_dev_count()):
        dev = base.ggml_backend_dev_get(i)
        if not dev or dev in rpc:
            continue
        name = _dev_name(dev)
        if name == "CPU" or name.startswith("BLAS") or name.startswith("RPC"):
            continue
        out.append(dev)
    return out


def device_list(endpoints) -> List[int]:
    """Local cards first, then the requested RPC devices in endpoint order."""
    devs = _local_gpu_devices()
    for ep in normalize_endpoints(endpoints):
        devs.extend(_rpc_devs.get(ep, []))
    return devs


def _install_hook() -> None:
    """Wrap llama-cpp-python's model load so ``params.devices`` is filled from
    the thread-local request set by :func:`load_with_devices` — or, when a
    load asks for nothing, from the local cards only (so registered RPC
    devices never leak into a model that did not ask for them)."""
    global _hooked
    if _hooked:
        return
    import llama_cpp.llama_cpp as _lc
    orig = _lc.llama_model_load_from_file

    def _wrapped(path, params):
        want = getattr(_tls, "endpoints", None)
        devs = device_list(want or [])
        arr = (ctypes.c_void_p * (len(devs) + 1))(*devs, None)
        _tls.keepalive = arr           # must outlive the call
        try:
            params.devices = ctypes.cast(arr, ctypes.c_void_p)
        except Exception as exc:
            print(f"[rpc] could not set params.devices: {exc}", flush=True)
        try:
            return orig(path, params)
        finally:
            _tls.keepalive = None

    _lc.llama_model_load_from_file = _wrapped
    _hooked = True


class load_with_devices:
    """``with load_with_devices(endpoints): Llama(...)`` — the load inside uses
    this machine's cards plus exactly these RPC servers."""

    def __init__(self, endpoints):
        self.endpoints = normalize_endpoints(endpoints)

    def __enter__(self):
        if self.endpoints:
            register_servers(self.endpoints)
        _tls.endpoints = list(self.endpoints)
        return self

    def __exit__(self, *exc):
        _tls.endpoints = None
        return False
