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
from typing import Optional
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


def parse_hosts(block: dict) -> list:
    """Every machine a ``host`` block names, as one HostConfig each.

    One machine: the flat fields (``url``, ``api_key``, ``start_cmd`` …).
    Several: ``hosts`` — a list of blocks with the same fields, each falling
    back to the flat ones — or ``urls``/lines of ``url | api_key | start_cmd |
    stop_cmd``. The model then has a pool: the first healthy, least-busy host
    takes each request, an on-demand host is started only when no host is up,
    and one that stops answering is skipped until it is back.
    """
    b = block or {}
    base = parse_host(b)
    out = []
    extra = b.get("hosts")
    if isinstance(extra, str):
        lines = [ln for ln in extra.replace("\r", "").split("\n") if ln.strip()]
        extra = []
        for ln in lines:
            parts = [p.strip() for p in ln.split("|")]
            d = {"url": parts[0]}
            for i, k in enumerate(("api_key", "start_cmd", "stop_cmd"), 1):
                if len(parts) > i and parts[i]:
                    d[k] = parts[i]
            extra.append(d)
    if base.url:
        out.append(base)
    for h in (extra or []):
        if not isinstance(h, dict) or not str(h.get("url") or "").strip():
            continue
        # Token, health path, timeouts and env carry over; a start/stop command
        # names ONE machine and never applies to another.
        merged = {k: v for k, v in b.items()
                  if k not in ("hosts", "urls", "url", "start_cmd", "stop_cmd")}
        merged.update({k: v for k, v in h.items() if v not in (None, "")})
        cfg = parse_host(merged)
        if cfg.url and all(cfg.url != o.url for o in out):
            out.append(cfg)
    return out


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
    """The host(s) behind one model. acquire()/release() like a pod pool.

    With one host this is what it always was: check it, start it if it has a
    start_cmd, use it, stop it after idling. With several, each request goes
    to the healthy host with the fewest requests in flight; when none answers,
    the first host with a start_cmd is started. ``failover`` lets the gateway
    move a request whose host died mid-way.
    """
    _HEALTH_TTL = 10.0
    _DOWN_FOR = 30.0

    def __init__(self, model_key: str, cfgs):
        self.model_key = model_key
        cfgs = list(cfgs) if isinstance(cfgs, (list, tuple)) else [cfgs]
        self.cfgs = cfgs
        self.cfg = cfgs[0]
        self.api_key = self.cfg.api_key
        self._lock = threading.RLock()
        self._inflight = {c.url: 0 for c in cfgs}
        self._last_used = 0.0
        self._started = {c.url: False for c in cfgs}
        self._stoppers = {}
        self._healthy_until = {}
        self._down_until = {}

    @property
    def _started_by_us(self) -> bool:
        """True while any host of this pool is up because we started it."""
        return any(self._started.values())

    def _by_url(self, url: str) -> HostConfig:
        for c in self.cfgs:
            if c.url == url:
                return c
        return self.cfg

    def _healthy(self, cfg: HostConfig, probe: bool = True) -> bool:
        now = time.time()
        if self._down_until.get(cfg.url, 0) > now:
            return False
        if self._healthy_until.get(cfg.url, 0) > now:
            return True
        if not probe:
            return False
        ok = _health_ok(cfg.url, cfg.health_path, cfg.api_key)
        if ok:
            self._healthy_until[cfg.url] = now + self._HEALTH_TTL
        return ok

    # -- the shape the gateway expects ------------------------------------ #
    def acquire(self, timeout: float = 1200.0, affinity: str = ""):
        with self._lock:
            order = sorted(self.cfgs, key=lambda c: self._inflight.get(c.url, 0))
            chosen = next((c for c in order if self._healthy(c)), None)
            if chosen is None:
                # A host that just failed is not the one to start again now.
                now = time.time()
                startable = [c for c in order
                             if c.start_cmd and self._down_until.get(c.url, 0) <= now]
                if not startable:
                    urls = ", ".join(c.url for c in self.cfgs)
                    raise HostError(
                        f"host for {self.model_key!r} at {urls} is not answering "
                        f"{self.cfg.health_path}, and it has no start_cmd — it is "
                        f"configured as always-on, so it should be up.")
                chosen = startable[0]
                self._start(chosen)
            self._inflight[chosen.url] = self._inflight.get(chosen.url, 0) + 1
            self._last_used = time.time()
            # The token of the host that was picked (they may differ).
            self.api_key = chosen.api_key
            return chosen.url, chosen.url

    def release(self, handle) -> None:
        with self._lock:
            url = handle if isinstance(handle, str) else self.cfg.url
            if self._inflight.get(url, 0) > 0:
                self._inflight[url] -= 1
            self._last_used = time.time()
            cfg = self._by_url(url)
            if cfg.stop_cmd and self._started.get(url) and cfg.idle_timeout_s:
                self._arm_stopper(cfg)

    def failover(self, failed_url: str):
        """Mark a host down and hand back another one, or None."""
        with self._lock:
            self._down_until[failed_url] = time.time() + self._DOWN_FOR
            self._healthy_until.pop(failed_url, None)
        self.release(failed_url)
        if len(self.cfgs) < 2:
            return None
        try:
            _h, url = self.acquire()
            return url
        except Exception:
            return None

    # -- lifecycle ---------------------------------------------------------- #
    def _start(self, cfg: HostConfig) -> None:
        print(f"[host] starting {self.model_key!r} on {cfg.url}: {cfg.start_cmd}",
              flush=True)
        self._run(cfg.start_cmd)
        deadline = time.time() + cfg.boot_timeout_s
        t0 = time.time()
        while time.time() < deadline:
            if _health_ok(cfg.url, cfg.health_path, cfg.api_key):
                self._started[cfg.url] = True
                self._healthy_until[cfg.url] = time.time() + self._HEALTH_TTL
                print(f"[host] {self.model_key!r} up after {time.time() - t0:.0f}s "
                      f"at {cfg.url}", flush=True)
                return
            time.sleep(3)
        raise HostError(
            f"host for {self.model_key!r} at {cfg.url} did not answer "
            f"{cfg.health_path} within {cfg.boot_timeout_s}s of running start_cmd. "
            f"Check the command, the URL, and that the token matches CODERAI_API_TOKEN.")

    def _arm_stopper(self, cfg: HostConfig) -> None:
        old = self._stoppers.get(cfg.url)
        if old is not None:
            old.cancel()

        def _maybe_stop():
            with self._lock:
                idle = time.time() - self._last_used
                if self._inflight.get(cfg.url, 0) > 0 or idle < cfg.idle_timeout_s:
                    self._arm_stopper(cfg)          # busy again, or not idle enough yet
                    return
                print(f"[host] {self.model_key!r} on {cfg.url} idle {idle:.0f}s — "
                      f"{cfg.stop_cmd}", flush=True)
                try:
                    self._run(cfg.stop_cmd)
                finally:
                    self._started[cfg.url] = False
                    self._healthy_until.pop(cfg.url, None)
                    self._stoppers.pop(cfg.url, None)

        t = threading.Timer(cfg.idle_timeout_s, _maybe_stop)
        t.daemon = True
        self._stoppers[cfg.url] = t
        t.start()

    def _run(self, cmd: str) -> None:
        import os
        # The env of the host whose command this is (several hosts may differ).
        cfg = next((c for c in self.cfgs if cmd in (c.start_cmd, c.stop_cmd)), self.cfg)
        env = {**os.environ, **cfg.env}
        try:
            r = subprocess.run(shlex.split(cmd), env=env, capture_output=True,
                               text=True, timeout=cfg.boot_timeout_s)
        except subprocess.TimeoutExpired:
            raise HostError(f"command timed out after {cfg.boot_timeout_s}s: {cmd}")
        if r.returncode != 0:
            raise HostError(f"command failed ({r.returncode}): {cmd}\n"
                            f"{(r.stderr or r.stdout or '').strip()[:400]}")

    def stop(self) -> None:
        """Explicit stop — the cleanup path, not the idle one."""
        with self._lock:
            for t in self._stoppers.values():
                t.cancel()
            self._stoppers.clear()
            for cfg in self.cfgs:
                if cfg.stop_cmd and self._started.get(cfg.url):
                    try:
                        self._run(cfg.stop_cmd)
                    finally:
                        self._started[cfg.url] = False

    def status(self) -> list:
        return [{"url": c.url, "on_demand": bool(c.start_cmd),
                 "inflight": self._inflight.get(c.url, 0),
                 "healthy": self._healthy(c, probe=False),
                 "started_by_us": self._started.get(c.url, False)}
                for c in self.cfgs]


_pools: dict = {}
_pools_lock = threading.Lock()


def get_host_pool(model_key: str, block: dict) -> HostPool:
    with _pools_lock:
        pool = _pools.get(model_key)
        if pool is None:
            pool = HostPool(model_key, parse_hosts(block))
            _pools[model_key] = pool
        return pool
