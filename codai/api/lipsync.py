# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Self-contained Wav2Lip lip-sync for installs that don't ship the packaged shims.

**The OCI image already handles lip sync properly** — ``packaging/linux/launcher/``
installs ``wav2lip`` and ``sadtalker`` shims on PATH which run the baked repo code in a
dedicated ``lipsync_venv`` and download the checkpoints on first use. When those shims
are present :func:`codai.api.video._apply_lipsync` uses them and never reaches this
module.

This module covers the *other* case: a from-source install (``build.sh`` + ``venv_all``)
where nothing is on PATH, which is where lip sync silently degraded to a plain audio mux.
It owns the lifecycle the way :mod:`codai.api.ds4_worker` owns ds4 — fetching code *and*
weights from a HuggingFace mirror on first use, patching what no longer runs on modern
numpy/librosa/torch, and driving ``inference.py`` as a subprocess.

It deliberately uses its **own** directory (``CODERAI_WAV2LIP_MANAGED_DIR``) rather than
the packaged ``CODERAI_WAV2LIP_DIR``: that one is the shim's writable working copy, seeded
from the baked read-only source and run against the lipsync venv's pinned dependencies.
Patching it from here would corrupt a working packaged install.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

#: Mirror carrying the Wav2Lip source tree, the s3fd face detector and the
#: wav2lip/wav2lip_gan checkpoints in one repo.
WAV2LIP_REPO = "camenduru/Wav2Lip"

#: Only what inference needs — not the training scripts or the eval corpus.
_ALLOW = [
    "inference.py", "audio.py", "hparams.py",
    "models/*", "face_detection/*",
    "checkpoints/wav2lip_gan.pth",
]

_ready: Optional[Tuple[Path, Path]] = None


def default_install_dir() -> Path:
    # NOT CODERAI_WAV2LIP_DIR — that belongs to the packaged launcher shim. The
    # distinct basename also avoids colliding with the shim's ~/.coderai/Wav2Lip on a
    # case-insensitive filesystem.
    return Path(os.environ.get("CODERAI_WAV2LIP_MANAGED_DIR")
                or os.path.expanduser("~/.coderai/wav2lip-managed"))


# --------------------------------------------------------------------------- #
# Compatibility patches
# --------------------------------------------------------------------------- #
# Wav2Lip was last updated against numpy 1.x, librosa 0.8 and torch 1.x. Each of
# these is a hard failure on a current stack, and all of them are mechanical.
_PATCHES = (
    # numpy >= 1.24 removed the aliases entirely.
    (r"\bnp\.float\b(?!\d|_)", "float"),
    (r"\bnp\.int\b(?!\d|_|p|e)", "int"),
    (r"\bnp\.bool\b(?!\d|_)", "bool"),
    # librosa >= 0.10 made the mel arguments keyword-only. The `[^,()=]` classes
    # exclude '=', so an argument that is already in keyword form cannot match a
    # second time — without that guard the rewrite compounds into `sr=sr=...`.
    (r"librosa\.filters\.mel\(\s*([^,()=]+?)\s*,\s*([^,()=]+?)\s*,",
     r"librosa.filters.mel(sr=\1, n_fft=\2,"),
    (r"librosa\.core\.stft\(", "librosa.stft("),
    # torch >= 2.6 defaults weights_only=True, which refuses these checkpoints.
    (r"torch\.load\((?![^()]*weights_only)([^()]*)\)", r"torch.load(\1, weights_only=False)"),
)


def _patch_tree(root: Path) -> None:
    """Apply the compatibility rewrites in place.

    Every rewritten file is compiled before it is kept: a pattern that produces
    invalid Python is reverted and reported, so a bad rewrite surfaces here instead
    of as a SyntaxError from a subprocess much later.
    """
    stamp = root / ".coderai-patched"
    if stamp.exists():
        return
    failed = []
    for path in list(root.rglob("*.py")):
        try:
            src = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        out = src
        for pattern, repl in _PATCHES:
            out = re.sub(pattern, repl, out)
        if out == src:
            continue
        rel = path.relative_to(root)
        try:
            compile(out, str(path), "exec")
        except SyntaxError as exc:
            failed.append(f"{rel}: {exc}")
            continue          # leave the original in place
        path.write_text(out, encoding="utf-8")
        print(f"[wav2lip] patched {rel}", flush=True)
    if failed:
        raise RuntimeError("Wav2Lip compatibility patching produced invalid Python: "
                           + "; ".join(failed))
    stamp.write_text("ok\n", encoding="utf-8")


def ensure_wav2lip(install_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Return ``(repo_dir, checkpoint)``, downloading and patching on first use.

    Raises RuntimeError with an actionable message when the assets can't be obtained.
    """
    global _ready
    if _ready is not None:
        return _ready

    root = Path(install_dir) if install_dir else default_install_dir()
    ckpt = root / "checkpoints" / "wav2lip_gan.pth"

    if not (root / "inference.py").exists() or not ckpt.exists():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "lip sync needs huggingface_hub to fetch the Wav2Lip assets") from exc
        root.mkdir(parents=True, exist_ok=True)
        print(f"[wav2lip] fetching {WAV2LIP_REPO} into {root} (first use only)…",
              flush=True)
        try:
            snapshot_download(repo_id=WAV2LIP_REPO, allow_patterns=_ALLOW,
                              local_dir=str(root))
        except Exception as exc:
            raise RuntimeError(
                f"could not download Wav2Lip assets from {WAV2LIP_REPO}: {exc}. "
                f"Place the repo and checkpoints/wav2lip_gan.pth in {root} manually, "
                f"or set CODERAI_WAV2LIP_MANAGED_DIR.") from exc

    if not (root / "inference.py").exists():
        raise RuntimeError(f"Wav2Lip inference.py missing under {root}")
    if not ckpt.exists():
        raise RuntimeError(f"Wav2Lip checkpoint missing at {ckpt}")

    _patch_tree(root)
    # inference.py writes its intermediate to the relative path temp/result.avi and
    # then muxes that with ffmpeg. The directory is a .gitignore'd scratch dir, so it
    # isn't in the mirror — without it the run exits 0 having produced nothing.
    (root / "temp").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    _ready = (root, ckpt)
    return _ready


def run_wav2lip(video_path: str, audio_path: str, out_path: str,
                timeout: int = 1800) -> str:
    """Lip-sync ``video_path`` to ``audio_path``. Returns ``out_path``.

    Raises RuntimeError on failure so the caller can report it instead of quietly
    handing back an unsynchronised video.
    """
    root, ckpt = ensure_wav2lip()

    cmd = [sys.executable, "inference.py",
           "--checkpoint_path", str(ckpt),
           "--face", os.path.abspath(video_path),
           "--audio", os.path.abspath(audio_path),
           "--outfile", os.path.abspath(out_path)]
    env = dict(os.environ)
    # inference.py imports audio/hparams/face_detection as top-level modules.
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True,
                          text=True, timeout=timeout)
    if proc.returncode != 0 or not os.path.exists(out_path):
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"wav2lip failed (rc={proc.returncode}): {tail}")
    return out_path


def is_available() -> bool:
    """True when the assets are already present (no download attempted)."""
    root = default_install_dir()
    return (root / "inference.py").exists() and \
           (root / "checkpoints" / "wav2lip_gan.pth").exists()


def which_sadtalker() -> Optional[str]:
    return shutil.which("sadtalker")
