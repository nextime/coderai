# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""One model over several machines' GPUs, for the servers that can do it.

vLLM (tensor parallel inside a node, pipeline parallel across nodes, over a
Ray cluster) and SGLang (``--nnodes``/``--node-rank`` over torch.distributed)
both spread one model over machines on their own; what they need from
coderai is the choreography: the other machines must be running the same
software, be told where rank 0 is, and be up before rank 0 launches. This
module runs the per-node start/stop commands an operator configured — ssh,
docker, a systemd unit, whatever starts the same coderai-vllm image over
there — templated with the addresses the peers need, and waits for them.

Plain TCP between nodes: fine for pipeline parallel (activations cross the
wire once per layer boundary), slow for tensor parallel (all-reduce every
layer), so keep ``tensor_parallel_size`` inside a node and let
``pipeline_parallel_size`` be the number of nodes.
"""

import os
import shlex
import subprocess
import threading
import time
from typing import List, Optional


class NodeError(RuntimeError):
    pass


def render(cmd: str, **vars) -> str:
    """``{ray_address}`` and friends in a configured command."""
    out = cmd or ""
    for k, v in vars.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def run_cmd(cmd: str, timeout: float = 120.0, env: Optional[dict] = None) -> str:
    """Run a configured command through the shell (they are ssh/docker one-
    liners with quoting and pipes) and return its output; raise on failure."""
    if not (cmd or "").strip():
        return ""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, env={**os.environ, **(env or {})})
    except subprocess.TimeoutExpired:
        raise NodeError(f"command timed out after {timeout:.0f}s: {cmd}")
    if r.returncode != 0:
        raise NodeError(f"command failed ({r.returncode}): {cmd}\n"
                        f"{(r.stderr or r.stdout or '').strip()[:400]}")
    return (r.stdout or "").strip()


def parse_nodes(raw) -> List[dict]:
    """``[{"name","start_cmd","stop_cmd","gpus"}]`` or lines of
    ``name | start_cmd | stop_cmd | gpus``."""
    if isinstance(raw, str):
        out = []
        for ln in raw.replace("\r", "").split("\n"):
            if not ln.strip():
                continue
            parts = [p.strip() for p in ln.split("|")]
            d = {"name": parts[0]}
            for i, k in enumerate(("start_cmd", "stop_cmd", "gpus"), 1):
                if len(parts) > i and parts[i]:
                    d[k] = parts[i]
            out.append(d)
        return out
    out = []
    for i, n in enumerate(raw or []):
        if isinstance(n, dict) and (n.get("start_cmd") or n.get("name")):
            d = dict(n)
            d.setdefault("name", f"node{i + 1}")
            out.append(d)
    return out


class NodeSet:
    """The peers of one multi-node launch: started together, stopped together."""

    def __init__(self, nodes: List[dict], label: str = "nodes"):
        self.nodes = nodes
        self.label = label
        self._started: List[dict] = []
        self._lock = threading.Lock()

    def start(self, **vars) -> None:
        for n in self.nodes:
            cmd = render(str(n.get("start_cmd") or ""), **vars)
            if not cmd:
                continue
            print(f"[{self.label}] starting {n.get('name')}: {cmd}", flush=True)
            run_cmd(cmd, timeout=float(n.get("start_timeout_s") or 300))
            with self._lock:
                self._started.append(n)

    def stop(self, **vars) -> None:
        with self._lock:
            started, self._started = list(self._started), []
        for n in reversed(started):
            cmd = render(str(n.get("stop_cmd") or ""), **vars)
            if not cmd:
                continue
            print(f"[{self.label}] stopping {n.get('name')}: {cmd}", flush=True)
            try:
                run_cmd(cmd, timeout=float(n.get("stop_timeout_s") or 120))
            except Exception as exc:
                print(f"[{self.label}] stop of {n.get('name')} failed: {exc}", flush=True)

    def gpus_expected(self) -> int:
        n = 0
        for node in self.nodes:
            try:
                n += int(node.get("gpus") or 1)
            except (TypeError, ValueError):
                n += 1
        return n


# ------------------------------------------------------------------- ray
def ray_bin(py: str) -> str:
    """The ``ray`` CLI beside a venv's python."""
    cand = os.path.join(os.path.dirname(py), "ray")
    return cand if os.path.isfile(cand) else "ray"


def ray_cluster_gpus(py: str, address: str, timeout: float = 30.0) -> float:
    """How many GPUs a ray cluster reports, or -1 when it cannot be asked."""
    code = ("import ray,sys\n"
            f"ray.init(address={address!r}, ignore_reinit_error=True, log_to_driver=False)\n"
            "print(ray.cluster_resources().get('GPU', 0))\n")
    try:
        r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return float((r.stdout or "0").strip().splitlines()[-1])
    except Exception:
        pass
    return -1.0


class RayCluster:
    """A ray head started here (or an existing one joined), plus its workers."""

    def __init__(self, py: str, advertise: str, port: int = 6379,
                 address: str = "", nodes: Optional[List[dict]] = None,
                 ready_timeout_s: float = 600.0):
        self.py = py
        self.advertise = advertise
        self.port = int(port or 6379)
        self.address = (address or "").strip()          # existing cluster
        self.nodes = NodeSet(parse_nodes(nodes), label="ray")
        self.ready_timeout_s = float(ready_timeout_s or 600)
        self._started_head = False

    @property
    def ray_address(self) -> str:
        return self.address or f"{self.advertise}:{self.port}"

    def start(self, gpus_needed: int, local_gpus: int = 1) -> str:
        if not self.address:
            if ray_cluster_gpus(self.py, f"{self.advertise}:{self.port}", timeout=15) < 0:
                cmd = [ray_bin(self.py), "start", "--head", f"--port={self.port}",
                       f"--node-ip-address={self.advertise}", f"--num-gpus={local_gpus}",
                       "--disable-usage-stats"]
                print(f"[ray] starting head: {' '.join(cmd)}", flush=True)
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                if r.returncode != 0:
                    raise NodeError(f"ray head failed to start: "
                                    f"{(r.stderr or r.stdout or '').strip()[:400]}")
                self._started_head = True
        self.nodes.start(ray_address=self.ray_address, head=self.advertise,
                         port=self.port)
        deadline = time.time() + self.ready_timeout_s
        last = -1.0
        while time.time() < deadline:
            last = ray_cluster_gpus(self.py, self.ray_address)
            if last >= gpus_needed:
                print(f"[ray] cluster at {self.ray_address} has {last:.0f} GPU(s) "
                      f"(need {gpus_needed})", flush=True)
                return self.ray_address
            time.sleep(5)
        self.stop()
        raise NodeError(f"ray cluster at {self.ray_address} reached {last:.0f} GPU(s) "
                        f"of the {gpus_needed} needed within {self.ready_timeout_s:.0f}s "
                        f"— check the nodes' start commands and that they can reach "
                        f"{self.ray_address}")

    def stop(self) -> None:
        self.nodes.stop(ray_address=self.ray_address, head=self.advertise, port=self.port)
        if self._started_head:
            try:
                subprocess.run([ray_bin(self.py), "stop", "--force"], capture_output=True,
                               text=True, timeout=60)
            except Exception:
                pass
            self._started_head = False


# ---------------------------------------------------------------- sglang
class SglangNodes:
    """Ranks 1..n-1 of an SGLang launch, started before rank 0 here."""

    def __init__(self, nodes: Optional[List[dict]], nnodes: int, dist_init_addr: str):
        self.nodes = NodeSet(parse_nodes(nodes), label="sglang")
        self.nnodes = int(nnodes or 1)
        self.dist_init_addr = dist_init_addr

    def start(self, model: str, **extra) -> None:
        if self.nnodes <= 1:
            return
        if len(self.nodes.nodes) < self.nnodes - 1:
            raise NodeError(f"nnodes={self.nnodes} but only {len(self.nodes.nodes)} "
                            f"node command(s) are configured for ranks 1..{self.nnodes - 1}")
        for rank, n in enumerate(self.nodes.nodes[: self.nnodes - 1], 1):
            cmd = render(str(n.get("start_cmd") or ""), rank=rank, nnodes=self.nnodes,
                         dist_init_addr=self.dist_init_addr, model=model, **extra)
            if not cmd:
                continue
            print(f"[sglang] starting rank {rank} ({n.get('name')}): {cmd}", flush=True)
            run_cmd(cmd, timeout=float(n.get("start_timeout_s") or 300))
            self.nodes._started.append(n)

    def stop(self, model: str = "") -> None:
        self.nodes.stop(rank="", nnodes=self.nnodes, dist_init_addr=self.dist_init_addr,
                        model=model)

    def rank0_args(self) -> List[str]:
        if self.nnodes <= 1:
            return []
        return ["--nnodes", str(self.nnodes), "--node-rank", "0",
                "--dist-init-addr", self.dist_init_addr]
