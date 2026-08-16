# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.

"""Surya OCR engine — runs in an ISOLATED venv subprocess.

Surya caps pillow<11, which conflicts with the main coderai venv (pillow>=12), so it lives
in a dedicated venv (``ocr.surya_venv``) driven by codai/ocr/workers/ocr_worker.py. See
:mod:`codai.ocr.subprocess_engine`.

LICENSE: GPL (compatible with coderai's GPLv3). Enabled only when
``ocr.surya_accept_license`` is True (enforced by the manager before this is constructed).
"""

import os

from codai.ocr.subprocess_engine import SubprocessOcrEngine, _resolve_venv_dir
from codai.ocr.base import OcrError

_REQ = os.path.join(os.path.dirname(__file__), "..", "..", "requirements-surya.txt")


class SuryaEngine(SubprocessOcrEngine):
    name = "surya"
    _worker_engine = "surya"

    def _venv_dir(self) -> str:
        return _resolve_venv_dir(self.cfg.surya_venv, "surya_venv")

    def _requirements(self) -> str:
        return os.path.abspath(_REQ)

    def _auto_build(self) -> bool:
        return bool(getattr(self.cfg, "surya_auto_build", False))

    def _worker_opts(self) -> dict:
        return {"langs": self.cfg.surya_langs or "it"}

    def load(self) -> None:
        if not self.cfg.surya_accept_license:
            raise OcrError(
                "Surya is license-gated. Set ocr.surya_accept_license = true to use it.",
                status=400,
            )
        super().load()
