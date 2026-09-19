# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""LoRA / QLoRA training over several machines — synchronous data parallel.

Every rank runs the SAME training loop on ITS card with ITS share of the
steps' samples; after each backward the gradients of the LoRA parameters
are averaged across ranks (an all-reduce) before the optimizer step, so all
ranks hold identical adapters throughout and rank 0's saved LoRA is the
result of the whole cluster's work. Only the adapter's gradients cross the
wire — a few MB per step — which is why this is fine on 1 GbE where tensor
parallelism is not.

This is plain ``torch.distributed`` with a hand-rolled all-reduce instead of
``DistributedDataParallel``: the trainers apply PEFT adapters to a frozen
base, and averaging ``p.grad`` of the trainable parameters is exactly what
DDP would do, minus the wrapper that PEFT-on-diffusers trips over.

The coordinator (rank 0) is the engine that received the training request;
it sends the same request, images inlined, to each node listed in
``nodes`` with a ``distributed`` block (rank, world size, rendezvous), then
trains itself. Peers save nothing that outlives them.
"""

import contextlib
import datetime
import os
import socket
import tempfile
import threading
from typing import List, Optional

_current = threading.local()


class TrainDist:
    def __init__(self, rank: int, world_size: int, master_addr: str, master_port: int,
                 backend: str = "", timeout_s: int = 600):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.master_addr = master_addr
        self.master_port = int(master_port)
        self.backend = backend or ""
        self.timeout_s = int(timeout_s or 600)
        self._pg = False
        self._tmp = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    # ---------------------------------------------------------------- setup
    def init(self) -> None:
        import torch
        import torch.distributed as dist
        if not self.backend:
            # gloo works everywhere (CUDA tensors included) and does not care
            # which interface carries it; nccl is faster on a fabric worth it.
            self.backend = "gloo"
        os.environ.setdefault("MASTER_ADDR", self.master_addr)
        os.environ.setdefault("MASTER_PORT", str(self.master_port))
        dist.init_process_group(
            backend=self.backend, init_method=f"tcp://{self.master_addr}:{self.master_port}",
            rank=self.rank, world_size=self.world_size,
            timeout=datetime.timedelta(seconds=self.timeout_s))
        self._pg = True
        print(f"[ddp] rank {self.rank}/{self.world_size} joined {self.master_addr}:"
              f"{self.master_port} ({self.backend})", flush=True)

    def destroy(self) -> None:
        if self._pg:
            try:
                import torch.distributed as dist
                dist.destroy_process_group()
            except Exception:
                pass
            self._pg = False
        if self._tmp:
            import shutil
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    # ------------------------------------------------------------ the loop
    def index(self, step: int, n: int) -> int:
        """Which sample this rank takes at ``step`` — ranks interleave."""
        return (step * self.world_size + self.rank) % max(1, n)

    def broadcast_params(self, params: List) -> None:
        """Rank 0's freshly initialized adapter to everyone, so all ranks start
        from the same point (PEFT's random init differs per process)."""
        import torch.distributed as dist
        for p in params:
            dist.broadcast(p.data, src=0)

    def sync_grads(self, params: List) -> None:
        """Average the LoRA gradients over all ranks."""
        import torch.distributed as dist
        for p in params:
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(self.world_size)

    def barrier(self) -> None:
        import torch.distributed as dist
        dist.barrier()

    def scratch_dir(self) -> str:
        """Where a peer writes what it must write (the trainers always save);
        removed with the context."""
        if self._tmp is None:
            self._tmp = tempfile.mkdtemp(prefix="coderai-ddp-peer-")
        return self._tmp


# ------------------------------------------------------------ module API
def current() -> Optional[TrainDist]:
    return getattr(_current, "ctx", None)


def is_main() -> bool:
    c = current()
    return c is None or c.is_main


def index(step: int, n: int) -> int:
    c = current()
    return (step % max(1, n)) if c is None else c.index(step, n)


def sync_grads(params) -> None:
    c = current()
    if c is not None and c.world_size > 1:
        c.sync_grads(list(params))


def broadcast_params(params) -> None:
    c = current()
    if c is not None and c.world_size > 1:
        c.broadcast_params(list(params))


@contextlib.contextmanager
def context(block: Optional[dict]):
    """``with ddp.context(req.distributed):`` around one training run."""
    if not block or int(block.get("world_size") or 1) < 2:
        yield None
        return
    ctx = TrainDist(rank=int(block.get("rank") or 0),
                    world_size=int(block["world_size"]),
                    master_addr=str(block.get("master_addr") or "127.0.0.1"),
                    master_port=int(block.get("master_port") or 29500),
                    backend=str(block.get("backend") or ""),
                    timeout_s=int(block.get("timeout_s") or 600))
    ctx.init()
    _current.ctx = ctx
    try:
        yield ctx
    finally:
        _current.ctx = None
        ctx.destroy()


def free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def parse_nodes(raw) -> List[dict]:
    """Peers for a training job: names of cluster nodes, or blocks with a url
    and token — ``"box2, box3"`` / ``[{"name": "box2"}, {"url": …, "api_key": …}]``."""
    if isinstance(raw, str):
        return [{"name": x.strip()} for x in raw.split(",") if x.strip()]
    out = []
    for n in raw or []:
        if isinstance(n, str) and n.strip():
            out.append({"name": n.strip()})
        elif isinstance(n, dict) and (n.get("name") or n.get("url")):
            out.append(dict(n))
    return out


def resolve_peers(nodes: List[dict]) -> List[dict]:
    """Each peer as {name, url, api_key, verify, ca_pem}, names looked up in
    cluster.nodes of this install."""
    known = {}
    try:
        from codai.admin.routes import config_manager
        for n in getattr(getattr(config_manager.config, "cluster", None), "nodes", None) or []:
            if isinstance(n, dict) and n.get("name"):
                known[str(n["name"]).lower()] = n
    except Exception:
        pass
    out = []
    for n in nodes:
        if n.get("url"):
            out.append({"name": n.get("name") or n["url"], "url": str(n["url"]).rstrip("/"),
                        "api_key": n.get("api_key", ""), "verify": n.get("verify", "system"),
                        "ca_pem": n.get("ca_pem", "")})
            continue
        k = known.get(str(n.get("name") or "").lower())
        if not k:
            raise ValueError(f"training node {n.get('name')!r} is not in cluster.nodes "
                             f"(Settings → Cluster) and has no url")
        out.append({"name": k["name"], "url": str(k["url"]).rstrip("/"),
                    "api_key": k.get("api_key", ""), "verify": k.get("verify", "system"),
                    "ca_pem": k.get("ca_pem", "")})
    return out
