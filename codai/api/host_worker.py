# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""The ``host`` backend: a capability image on a machine you already have.

The capability images are ordinary containers. Nothing about them needs RunPod
— a pod is just a machine that pulls one and runs it. So the same image can run
on a box you own, a server you rent by the month, a second GPU in the next room,
and coderai should be able to use it exactly as it uses a pod: per model, with
an endpoint and a token, and nothing else.

Two shapes, chosen by whether ``start_cmd`` is set:

  always on — the container is already running; coderai just talks to it.
      {"backend": "host", "host": {"url": "http://gpubox:8000", "api_key": "..."}}

  on demand — coderai runs a command to start it, waits for /healthz, uses it,
      and runs another command to stop it after an idle timeout. The commands
      are whatever starts the thing: docker run, ssh box docker start, a systemd
      unit, a script.
      {"backend": "host", "host": {
          "url": "http://gpubox:8000", "api_key": "...",
          "start_cmd": "ssh gpubox docker start coderai-images",
          "stop_cmd":  "ssh gpubox docker stop coderai-images",
          "boot_timeout_s": 120, "idle_timeout_s": 600}}

What this deliberately does NOT have: GPU search, price ranking, budgets, spot
bids, capacity fallback, a reaper for orphaned machines. Those exist because a
cloud rents you an anonymous card by the second and can lose it. A host is a
named machine you are responsible for; the machinery would be wrong, not just
unnecessary.

The pool wears the same acquire()/release() shape as RunpodPodPool so the
gateway and the test run treat both identically.
"""

from codai.api import pod_http
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class HostConfig:
    """Parsed per-model ``host`` block."""
    url: str = ""
    api_key: str = ""
    start_cmd: str = ""                  # blank = always on
    stop_cmd: str = ""
    health_path: str = "/healthz"
    boot_timeout_s: int = 300            # after start_cmd, until health answers
    idle_timeout_s: int = 600            # after the last request, run stop_cmd
    env: dict = field(default_factory=dict)


def parse_host(block: dict) -> HostConfig:
    b = block or {}
    cfg = HostConfig()
    cfg.url = str(b.get("url") or "").strip().rstrip("/")
    cfg.api_key = str(b.get("api_key") or "").strip()
    cfg.start_cmd = str(b.get("start_cmd") or "").strip()
    cfg.stop_cmd = str(b.get("stop_cmd") or "").strip()
    cfg.health_path = str(b.get("health_path") or "/healthz").strip() or "/healthz"
    try:
        cfg.boot_timeout_s = max(5, int(b.get("boot_timeout_s", cfg.boot_timeout_s)))
    except (TypeError, ValueError):
        pass
    try:
        cfg.idle_timeout_s = max(0, int(b.get("idle_timeout_s", cfg.idle_timeout_s)))
    except (TypeError, ValueError):
        pass
    if isinstance(b.get("env"), dict):
        cfg.env = {str(k): str(v) for k, v in b["env"].items()}
    return cfg


def _health_ok(url: str, path: str, api_key: str, timeout: float = 5.0) -> bool:
    import requests
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = pod_http.get(url.rstrip("/") + path, headers=headers, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


class HostError(RuntimeError):
    pass


class HostPool:
    """One host, used by one model. acquire()/release() like a pod pool."""

    def __init__(self, model_key: str, cfg: HostConfig):
        self.model_key = model_key
        self.cfg = cfg
        self.api_key = cfg.api_key
        self._lock = threading.RLock()
        self._inflight = 0
        self._last_used = 0.0
        self._started_by_us = False
        self._stopper = None

    # -- the shape the gateway expects ------------------------------------ #
    def acquire(self, timeout: float = 1200.0, affinity: str = ""):
        with self._lock:
            if not _health_ok(self.cfg.url, self.cfg.health_path, self.api_key):
                if not self.cfg.start_cmd:
                    raise HostError(
                        f"host for {self.model_key!r} at {self.cfg.url} is not "
                        f"answering {self.cfg.health_path}, and it has no start_cmd "
                        f"— it is configured as always-on, so it should be up.")
                self._start()
            self._inflight += 1
            self._last_used = time.time()
            return self, self.cfg.url

    def release(self, handle) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self._last_used = time.time()
            if self.cfg.stop_cmd and self._started_by_us and self.cfg.idle_timeout_s:
                self._arm_stopper()

    # -- lifecycle ---------------------------------------------------------- #
    def _start(self) -> None:
        print(f"[host] starting {self.model_key!r}: {self.cfg.start_cmd}", flush=True)
        self._run(self.cfg.start_cmd)
        deadline = time.time() + self.cfg.boot_timeout_s
        t0 = time.time()
        while time.time() < deadline:
            if _health_ok(self.cfg.url, self.cfg.health_path, self.api_key):
                self._started_by_us = True
                print(f"[host] {self.model_key!r} up after {time.time() - t0:.0f}s "
                      f"at {self.cfg.url}", flush=True)
                return
            time.sleep(3)
        raise HostError(
            f"host for {self.model_key!r} did not answer {self.cfg.health_path} "
            f"within {self.cfg.boot_timeout_s}s of running start_cmd. Check the "
            f"command, the URL, and that the token matches CODERAI_API_TOKEN.")

    def _arm_stopper(self) -> None:
        if self._stopper is not None:
            self._stopper.cancel()

        def _maybe_stop():
            with self._lock:
                idle = time.time() - self._last_used
                if self._inflight > 0 or idle < self.cfg.idle_timeout_s:
                    self._arm_stopper()          # busy again, or not idle enough yet
                    return
                print(f"[host] {self.model_key!r} idle {idle:.0f}s — "
                      f"{self.cfg.stop_cmd}", flush=True)
                try:
                    self._run(self.cfg.stop_cmd)
                finally:
                    self._started_by_us = False
                    self._stopper = None

        self._stopper = threading.Timer(self.cfg.idle_timeout_s, _maybe_stop)
        self._stopper.daemon = True
        self._stopper.start()

    def _run(self, cmd: str) -> None:
        import os
        env = {**os.environ, **self.cfg.env}
        try:
            r = subprocess.run(shlex.split(cmd), env=env, capture_output=True,
                               text=True, timeout=self.cfg.boot_timeout_s)
        except subprocess.TimeoutExpired:
            raise HostError(f"command timed out after {self.cfg.boot_timeout_s}s: {cmd}")
        if r.returncode != 0:
            raise HostError(f"command failed ({r.returncode}): {cmd}\n"
                            f"{(r.stderr or r.stdout or '').strip()[:400]}")

    def stop(self) -> None:
        """Explicit stop — the cleanup path, not the idle one."""
        with self._lock:
            if self._stopper is not None:
                self._stopper.cancel()
                self._stopper = None
            if self.cfg.stop_cmd and self._started_by_us:
                try:
                    self._run(self.cfg.stop_cmd)
                finally:
                    self._started_by_us = False


_pools: dict = {}
_pools_lock = threading.Lock()


def get_host_pool(model_key: str, block: dict) -> HostPool:
    with _pools_lock:
        pool = _pools.get(model_key)
        if pool is None:
            pool = HostPool(model_key, parse_host(block))
            _pools[model_key] = pool
        return pool
