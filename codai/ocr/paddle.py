# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.

"""PaddleOCR engine — runs in an ISOLATED venv subprocess.

PaddleOCR pulls opencv-contrib-python (clashes with the main venv's opencv-python/cv2) and
its paddlepaddle-gpu wheel bundles its own CUDA runtime, so it lives in a dedicated venv
(``ocr.paddle_venv``) driven by codai/ocr/workers/ocr_worker.py. See
:mod:`codai.ocr.subprocess_engine`.
"""

import os

from codai.ocr.subprocess_engine import SubprocessOcrEngine, _resolve_venv_dir

# Repo-root requirements file for the paddle isolated venv (auto-build).
_REQ = os.path.join(os.path.dirname(__file__), "..", "..", "requirements-ocr-paddle.txt")


class PaddleEngine(SubprocessOcrEngine):
    name = "paddle"
    _worker_engine = "paddle"

    def _venv_dir(self) -> str:
        return _resolve_venv_dir(self.cfg.paddle_venv, "paddle_venv")

    def _requirements(self) -> str:
        return os.path.abspath(_REQ)

    def _auto_build(self) -> bool:
        return bool(getattr(self.cfg, "paddle_auto_build", False))

    def _worker_opts(self) -> dict:
        return {
            "lang": self.cfg.paddle_lang or self.cfg.lang or "it",
            "use_gpu": bool(self.cfg.paddle_use_gpu),
            "structure": bool(self.cfg.paddle_structure),
            "det_model_dir": self.cfg.paddle_det_model_dir or "",
            "rec_model_dir": self.cfg.paddle_rec_model_dir or "",
        }
