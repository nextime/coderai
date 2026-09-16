#!/usr/bin/env python3
# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Find (and, with the daemon stopped, remove) overlay2 layers Docker has lost.

Twice now the build disk filled up while `docker system df` claimed ~90 GB of
images: 500+ GB of overlay2 directories belonged to layers that no image and no
container referenced — old flattened coderai:base images, 26 GB each, pinned by
container mount records (`image/overlay2/layerdb/mounts/<id>`) for containers
the daemon no longer knows about. `docker image prune`, `docker system prune`
and `docker builder prune` cannot see them: to the daemon those layers are
in use.

This walks the on-disk metadata the way the daemon would on a fresh start:

  live layers  = every chain id reachable from an image in imagedb, plus the
                 parent chain of every mount that belongs to a container
                 `docker ps -a` knows about
  stale layers = every layerdb entry not in that set
  stale mounts = every mounts/<id> whose id is not a known container

    sudo tools/docker_leak_audit.py            # report, nothing touched
    sudo tools/docker_leak_audit.py --fix      # remove them; REFUSES while dockerd runs

A build in progress shows a few small stale layers (created, not yet attached
to an image) — ignore those; the leak looks like dozens of 26 GB ones.

--fix needs the daemon stopped (`/etc/init.d/docker stop`; sysv here, not
systemd): dockerd caches the layer store in memory and would keep believing in
directories that are gone. Stopping it stops the production container, so
plan the relaunch.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.environ.get("DOCKER_ROOT", "/storage/docker")
IMG = f"{ROOT}/image/overlay2"
OVL = f"{ROOT}/overlay2"


def _chain(diff_ids):
    c = diff_ids[0]
    yield c
    for d in diff_ids[1:]:
        c = "sha256:" + hashlib.sha256(f"{c} {d}".encode()).hexdigest()
        yield c


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _known_containers():
    try:
        out = subprocess.check_output(["docker", "ps", "-aq", "--no-trunc"],
                                      text=True, stderr=subprocess.DEVNULL)
        return set(l.strip() for l in out.splitlines() if l.strip()), True
    except Exception:
        # Daemon down: every container the DB knows about is still listed on
        # disk, so read those instead of treating them all as phantoms.
        cdir = f"{ROOT}/containers"
        return (set(os.listdir(cdir)) if os.path.isdir(cdir) else set()), False


def audit():
    imgdb, layerdb, mounts = f"{IMG}/imagedb/content/sha256", f"{IMG}/layerdb/sha256", f"{IMG}/layerdb/mounts"
    live = set()
    for img in os.listdir(imgdb):
        try:
            ids = json.load(open(f"{imgdb}/{img}")).get("rootfs", {}).get("diff_ids") or []
        except Exception:
            continue
        if ids:
            live.update(_chain(ids))
    known, daemon_up = _known_containers()
    stale_mounts = []
    for m in os.listdir(mounts) if os.path.isdir(mounts) else []:
        if m not in known:
            ids = [_read(f"{mounts}/{m}/{k}") for k in ("mount-id", "init-id")]
            stale_mounts.append((m, [i for i in ids if i]))
            continue
        c = _read(f"{mounts}/{m}/parent")
        while c:
            live.add(c)
            c = _read(f"{layerdb}/{c.split(':', 1)[1]}/parent")
    stale = []
    for d in os.listdir(layerdb):
        if "sha256:" + d in live:
            continue
        cid = _read(f"{layerdb}/{d}/cache-id")
        try:
            size = int(_read(f"{layerdb}/{d}/size") or 0)
        except ValueError:
            size = 0
        stale.append((size, d, cid))
    stale.sort(reverse=True)
    return stale, stale_mounts, daemon_up, len(os.listdir(imgdb)), len(os.listdir(layerdb))


def main():
    fix = "--fix" in sys.argv
    if os.geteuid() != 0:
        print("run as root: the docker root is not world-readable", file=sys.stderr)
        return 2
    stale, stale_mounts, daemon_up, n_img, n_layers = audit()
    total = sum(s for s, _, _ in stale)
    print(f"images {n_img}, layerdb entries {n_layers}, stale layers {len(stale)} "
          f"({total / 1e9:.1f} GB by layerdb size), phantom mounts {len(stale_mounts)}")
    for s, d, c in stale[:25]:
        print(f"  {s / 1e9:8.1f} GB  layer {d[:12]}  overlay2/{c}")
    if len(stale) > 25:
        print(f"  … and {len(stale) - 25} smaller ones")
    if not fix:
        return 0
    if daemon_up or subprocess.run(["pgrep", "-x", "dockerd"], capture_output=True).returncode == 0:
        print("REFUSING --fix while dockerd runs: it would keep believing in layers "
              "that are gone. Stop it first (/etc/init.d/docker stop).", file=sys.stderr)
        return 1
    layerdb, mounts = f"{IMG}/layerdb/sha256", f"{IMG}/layerdb/mounts"
    n = 0
    for _, d, c in stale:
        for p in (f"{OVL}/{c}" if c else "", f"{layerdb}/{d}"):
            if p and os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
        n += 1
    for m, ids in stale_mounts:
        for i in ids:
            if os.path.isdir(f"{OVL}/{i}"):
                shutil.rmtree(f"{OVL}/{i}", ignore_errors=True)
        if os.path.isdir(f"{mounts}/{m}"):
            shutil.rmtree(f"{mounts}/{m}", ignore_errors=True)
        n += 1
    print(f"removed {n} entries — start docker again (/etc/init.d/docker start)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
