# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""The llama.cpp ``rpc-server`` processes this machine contributes.

llama.cpp's RPC backend makes a card on another machine look like one more
ggml device: a loading process registers ``host:port`` endpoints, and the
usual layer split (``tensor_split``) then spreads a GGUF over local and
remote cards alike. Per token, the traffic is the activations between the
layers on each side — small — so decode over a LAN is workable; loading and
prompt processing are bound by the wire, and 1 GbE is where it hurts.

Each entry of ``cluster.rpc_servers`` becomes one process here, restarted if
it dies. The endpoints are advertised on ``/cluster/state`` so a head lists
them on the Models page. The protocol is unauthenticated: bind it to a LAN
or a WireGuard address, never to the internet.
"""

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RpcServerSpec:
    name: str
    host: str = "0.0.0.0"
    port: int = 50052
    device: str = ""          # ggml device name: CUDA0, Vulkan0 … (blank = server default)
    mem_gb: float = 0.0       # advertised memory cap (0 = the device's)
    threads: int = 0          # CPU backend threads (0 = default)
    cache: bool = False       # --cache: keep tensors on local disk across loads
    enabled: bool = True
    env: dict = None


def parse_rpc_servers(raw) -> List[RpcServerSpec]:
    out, ports = [], set()
    for i, r in enumerate(raw or []):
        if not isinstance(r, dict):
            continue
        try:
            port = int(r.get("port") or 0)
        except (TypeError, ValueError):
            port = 0
        if port <= 0 or port in ports:
            continue
        ports.add(port)
        spec = RpcServerSpec(
            name=str(r.get("name") or "").strip() or f"rpc{i + 1}",
            host=str(r.get("host") or "0.0.0.0").strip() or "0.0.0.0",
            port=port,
            device=str(r.get("device") or "").strip(),
            enabled=bool(r.get("enabled", True)),
            cache=bool(r.get("cache", False)),
            env={str(k): str(v) for k, v in (r.get("env") or {}).items()}
            if isinstance(r.get("env"), dict) else {},
        )
        for k in ("mem_gb", "threads"):
            try:
                setattr(spec, k, type(getattr(spec, k))(r.get(k) or 0))
            except (TypeError, ValueError):
                pass
        out.append(spec)
    return out


def find_rpc_server(explicit: str = "") -> Optional[str]:
    """Where the rpc-server binary is: the configured path, the venv's bin
    (where packaging/build-rpc-server.sh installs it), a coderai local-bin,
    then PATH."""
    cands = []
    if explicit:
        cands.append(os.path.expanduser(explicit))
    env = os.environ.get("CODERAI_RPC_SERVER_BIN")
    if env:
        cands.append(env)
    cands.append(os.path.join(os.path.dirname(sys.executable), "rpc-server"))
    cands.append("/opt/coderai/local-bin/rpc-server")
    cands.append(os.path.expanduser("~/.coderai/bin/rpc-server"))
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return shutil.which("rpc-server")


def advertise_host(configured: str = "") -> str:
    """The address other machines reach this one at."""
    if configured:
        return configured
    env = os.environ.get("CODERAI_ADVERTISE_HOST")
    if env:
        return env
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))      # no packet is sent
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return socket.gethostname()


def supported_flags(binary: str) -> set:
    """Which optional flags this rpc-server build takes (its --help says):
    ``-d`` device selection and ``-m`` memory cap came and went across
    llama.cpp versions; passing one the build lacks makes it exit at once."""
    try:
        r = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=10)
        text = (r.stdout or "") + (r.stderr or "")
    except Exception:
        return {"-d", "-t", "-c", "-m"}
    flags = set()
    for f in ("-d", "-t", "-c", "-m"):
        if f"  {f}," in text or f" {f} " in text:
            flags.add(f)
    return flags


def build_cmd(binary: str, spec: RpcServerSpec, flags: Optional[set] = None) -> list:
    flags = {"-d", "-t", "-c", "-m"} if flags is None else flags
    cmd = [binary, "-H", spec.host, "-p", str(spec.port)]
    if spec.device and "-d" in flags:
        cmd += ["-d", spec.device]
    if spec.mem_gb and spec.mem_gb > 0 and "-m" in flags:
        cmd += ["-m", str(int(spec.mem_gb * 1024))]
    if spec.threads and spec.threads > 0 and "-t" in flags:
        cmd += ["-t", str(spec.threads)]
    if spec.cache and "-c" in flags:
        cmd += ["-c"]
    return cmd


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host if host not in ("0.0.0.0", "") else "127.0.0.1",
                                       port), timeout=timeout):
            return True
    except OSError:
        return False


class RpcServerManager:
    """Spawn, watch and stop this machine's rpc-server processes."""

    def __init__(self, specs: List[RpcServerSpec], binary: str = "",
                 advertise: str = ""):
        self.specs = [s for s in specs if s.enabled]
        self.binary = find_rpc_server(binary)
        self.advertise = advertise_host(advertise)
        self._procs = {}          # name -> Popen
        self._tails = {}          # name -> last lines
        self._starts = {}         # name -> [epoch, ...]
        self._stopped = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._flags = supported_flags(self.binary) if self.binary else set()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if not self.specs:
            return
        if not self.binary:
            print("[rpc] cluster.rpc_servers configured but no rpc-server binary "
                  "found — run packaging/build-rpc-server.sh (or set "
                  "cluster.rpc_bin)", flush=True)
            return
        for spec in self.specs:
            self._spawn(spec)
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def _spawn(self, spec: RpcServerSpec) -> None:
        cmd = build_cmd(self.binary, spec, self._flags)
        if spec.mem_gb and "-m" not in self._flags:
            print(f"[rpc] {spec.name}: this rpc-server has no memory cap flag; "
                  f"mem_gb ignored", flush=True)
        env = dict(os.environ)
        env.update(spec.env or {})
        print(f"[rpc] starting {spec.name} on {spec.host}:{spec.port}"
              f"{' device ' + spec.device if spec.device else ''}: {' '.join(cmd)}",
              flush=True)
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as exc:
            print(f"[rpc] {spec.name}: cannot start: {exc}", flush=True)
            return
        with self._lock:
            self._procs[spec.name] = proc
            self._starts.setdefault(spec.name, []).append(time.time())
        tail = self._tails.setdefault(spec.name, [])
        threading.Thread(target=self._pump, args=(spec.name, proc, tail),
                         daemon=True).start()

    def _pump(self, name, proc, tail):
        try:
            for line in proc.stdout:
                line = line.rstrip()
                tail.append(line)
                del tail[:-20]
                if "error" in line.lower() or "failed" in line.lower():
                    print(f"[rpc:{name}] {line}", flush=True)
        except Exception:
            pass

    def _watch(self) -> None:
        while not self._stopped.is_set():
            for spec in self.specs:
                proc = self._procs.get(spec.name)
                if proc is not None and proc.poll() is not None:
                    hist = [t for t in self._starts.get(spec.name, [])
                            if time.time() - t < 120]
                    if len(hist) >= 5:
                        continue        # crash loop: leave it down, keep the log
                    tail = " | ".join(self._tails.get(spec.name, [])[-3:])
                    print(f"[rpc] {spec.name} exited (code {proc.returncode}); "
                          f"restarting. {tail}", flush=True)
                    time.sleep(2.0)
                    self._spawn(spec)
            self._stopped.wait(5.0)

    def stop(self) -> None:
        self._stopped.set()
        for name, proc in list(self._procs.items()):
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    # --------------------------------------------------------------- status
    def endpoints(self) -> List[dict]:
        """What this machine advertises: one entry per server, with liveness."""
        out = []
        for spec in self.specs:
            proc = self._procs.get(spec.name)
            alive = proc is not None and proc.poll() is None
            out.append({
                "name": spec.name,
                "endpoint": f"{self.advertise}:{spec.port}",
                "device": spec.device,
                "mem_gb": spec.mem_gb,
                "alive": alive,
                "listening": alive and _port_open(spec.host, spec.port),
                "pid": proc.pid if alive else None,
                "last_log": (self._tails.get(spec.name) or [""])[-1],
            })
        return out
