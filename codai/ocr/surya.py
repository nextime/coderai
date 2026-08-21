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

    def _serve_mode(self) -> str:
        return (getattr(self.cfg, "surya_serve", "local") or "local").strip().lower()

    def _worker_env(self) -> dict:
        """For the served ("Surya2") modes, tell the worker to attach to an external
        OpenAI server instead of running detection+recognition locally."""
        mode = self._serve_mode()
        if mode not in ("vllm", "llamacpp"):
            return {}
        url = self._server_url
        if not url:
            return {}
        return {"SURYA_INFERENCE_BACKEND": mode, "SURYA_INFERENCE_URL": url}

    def load(self) -> None:
        if not self.cfg.surya_accept_license:
            raise OcrError(
                "Surya is license-gated. Set ocr.surya_accept_license = true to use it.",
                status=400,
            )
        self._server_url = ""
        mode = self._serve_mode()
        if mode == "vllm":
            # Serve the Surya2 VLM checkpoint via coderai's vLLM backend (continuous
            # batching) and attach Surya to it. Requires vllm enabled/auto-buildable.
            from codai.api import vllm_worker
            from codai.models.manager import get_active_vllm_config
            vcfg = get_active_vllm_config()
            if vcfg is None:
                raise OcrError("Surya vllm mode needs the vLLM backend configured", status=400)
            model = (getattr(self.cfg, "surya_model", "") or "datalab-to/surya-ocr-2").strip()
            base = vllm_worker.ensure_service(vcfg, model_path=model, served_name=model)
            self._server_url = base.rstrip("/") + "/v1"
        elif mode == "llamacpp":
            url = (getattr(self.cfg, "surya_server_url", "") or "").strip()
            if not url:
                raise OcrError(
                    "Surya llamacpp mode needs ocr.surya_server_url (a running llama-server "
                    "OpenAI endpoint).", status=400)
            self._server_url = url
        super().load()
