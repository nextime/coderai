#!/usr/bin/env python3
"""Write a small build manifest for local CoderAI distribution artifacts."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def cmd(args: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def package_versions(python_bin: str | None) -> dict[str, str]:
    if not python_bin:
        return {}
    code = r'''
import importlib.metadata as md
names = ["torch", "torchvision", "torchaudio", "transformers", "diffusers", "accelerate", "llama-cpp-python", "stable-diffusion-cpp-python", "whispercpp", "bitsandbytes", "onnxruntime", "onnxruntime-gpu"]
out = {}
for name in names:
    try:
        out[name] = md.version(name)
    except Exception:
        pass
import json
print(json.dumps(out, sort_keys=True))
'''
    try:
        raw = subprocess.check_output([python_bin, "-c", code], text=True, stderr=subprocess.DEVNULL)
        return json.loads(raw)
    except Exception:
        return {}


def main() -> int:
    out = Path(os.environ.get("MANIFEST_OUT", "BUILD-MANIFEST.json"))
    root = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
    python_bin = os.environ.get("MANIFEST_PYTHON")
    local_bins = [p for p in os.environ.get("MANIFEST_LOCAL_BINS", "").split(os.pathsep) if p]

    data = {
        "artifact": os.environ.get("MANIFEST_ARTIFACT", "unknown"),
        "build_mode": os.environ.get("MANIFEST_BUILD_MODE", "unknown"),
        "build_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": cmd(["git", "rev-parse", "HEAD"], str(root)),
        "git_dirty": bool(cmd(["git", "status", "--porcelain"], str(root))),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python_version": os.environ.get("PYTHON_VERSION", ""),
        "python_build_standalone_release": os.environ.get("PBS_RELEASE", ""),
        "uv_version": os.environ.get("UV_VERSION", ""),
        "cuda_version": os.environ.get("CUDA_VERSION", ""),
        "ubuntu_version": os.environ.get("UBUNTU_VERSION", ""),
        "source_venv": os.environ.get("MANIFEST_VENV", ""),
        "included_local_binaries": local_bins,
        "package_versions": package_versions(python_bin),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
