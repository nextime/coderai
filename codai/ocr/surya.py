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
from codai.ocr.base import OcrError, vlm_serve_limits

_REQ = os.path.join(os.path.dirname(__file__), "..", "..", "requirements-surya.txt")


def _cuda_present() -> bool:
    """Whether this machine has a usable CUDA device.

    Asked of torch first, because that is what Surya itself will ask. torch may
    not be importable in the main venv at all (Surya lives in its own), so
    nvidia-smi is the fallback — and if neither can answer, the answer is NO.
    Failing closed is the point: the mode exists to use a GPU, and guessing that
    one is present ends in a CPU run that looks healthy and never finishes.
    """
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        pass
    try:
        import shutil
        import subprocess
        smi = shutil.which("nvidia-smi")
        if not smi:
            return False
        out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=10)
        return out.returncode == 0 and "GPU 0" in (out.stdout or "")
    except Exception:
        return False


class SuryaEngine(SubprocessOcrEngine):
    name = "surya"
    _worker_engine = "surya"

    # Where the recognition VLM is, what the server calls it, and the token that
    # opens it. Empty in the local mode, and resolved by load() in the served ones.
    _server_url = ""
    _server_key = ""
    _served_name = ""

    def _venv_dir(self) -> str:
        return _resolve_venv_dir(self.cfg.surya_venv, "surya_venv", ("surya",))

    def _requirements(self) -> str:
        return os.path.abspath(_REQ)

    def _auto_build(self) -> bool:
        return bool(getattr(self.cfg, "surya_auto_build", False))

    def _worker_opts(self) -> dict:
        return {"langs": self.cfg.surya_langs or "it",
                # Sent at LOAD, like everything else here: the hook it switches on
                # patches surya's backend, which has to happen before the first page
                # rather than per request.
                "token_trace": bool(getattr(self.cfg, "surya_token_trace", False)),
                "repair_accents": bool(getattr(self.cfg, "surya_repair_accents", True)),
                # None hands the decision back to surya; anything else is a
                # ceiling on how often it may re-run a repeating page.
                "max_retries": (None if int(getattr(self.cfg, "surya_max_retries", 1)) < 0
                                else int(getattr(self.cfg, "surya_max_retries", 1))),
                # 0 leaves surya's own 2048 alone.
                "max_tokens": max(0, int(getattr(self.cfg, "surya_max_tokens", 0) or 0))}

    #: Modes where the recognition VLM lives in a model server we only talk to.
    SERVED_MODES = ("model", "vllm", "llamacpp", "server")

    def lanes(self) -> int:
        """Pages in flight per worker process.

        Only in the SERVED modes, and that is the whole point: there the
        recognition VLM is somewhere else, the predictor holds no weights (1.7 MB
        of tokenizer and config on disk) and this process rasterises a page and
        then waits on a socket. Several pages can wait at once for the price of
        the torch import that is already paid.

        In `local` mode a lane would be a second copy of the model on this
        machine's GPU, so there is exactly one.
        """
        if self._serve_mode() not in self.SERVED_MODES:
            return 1
        try:
            return max(1, int(getattr(self.cfg, "surya_lanes", 4) or 1))
        except (TypeError, ValueError):
            return 4

    def _serve_mode(self) -> str:
        mode = (getattr(self.cfg, "surya_serve", "model") or "model").strip().lower()
        return "llamacpp" if mode == "server" else mode

    def _local_env(self) -> dict:
        """Device for the LOCAL mode worker.

        Said out loud rather than left to Surya's own fallback: with the GPU
        required we have already refused if there is none, and when the operator
        has chosen the CPU the worker is told so explicitly instead of arriving
        there by accident.
        """
        if getattr(self.cfg, "surya_use_gpu", True):
            return {"TORCH_DEVICE": "cuda"}
        return {"TORCH_DEVICE": "cpu"}

    def _worker_env(self) -> dict:
        """For the served ("Surya2") modes, tell the worker to attach to an external
        OpenAI server instead of running detection+recognition locally.

        Surya reads these from its own settings, so the handshake is three values and
        no patching: which client to use, where it is, and what the server calls the
        model — it refuses to attach to a server whose /v1/models disagrees with the
        checkpoint it expects. The key matters as much: a pod URL is reachable by
        anyone who learns it, so ours are launched bearer-locked, and Surya's vLLM
        client takes that token from VLLM_API_KEY (its llama.cpp client hardcodes
        "EMPTY", which is why a locked endpoint is always attached to as "vllm")."""
        mode = self._serve_mode()
        if mode not in self.SERVED_MODES:
            return self._local_env()
        url = self._server_url
        if not url:
            return {}
        env = {
            "SURYA_INFERENCE_BACKEND": "llamacpp" if mode == "llamacpp" else "vllm",
            "SURYA_INFERENCE_URL": url,
        }
        # One page per call, because a LANE is our unit of concurrency. Left
        # unset, surya sizes its own thread pool from a built-in GPU table
        # (min(max_num_seqs, MAX_AUTO_PARALLEL)) and fans out inside the worker
        # — so instances x lanes would stop being the real number of pages in
        # flight the moment a call carried more than one page. Pinning it keeps
        # the arithmetic in the admin UI true.
        env["SURYA_INFERENCE_PARALLEL"] = "1"
        if self._served_name:
            env["SURYA_MODEL_CHECKPOINT"] = self._served_name
        if self._server_key:
            env["VLLM_API_KEY"] = self._server_key
        return env

    def prelaunch_vram_gb(self) -> float:
        """In the served modes Surya rides a vLLM instance that grabs
        gpu_memory_utilization × the whole card before it will start, so that much has to
        be free FIRST — plus the ceiling's headroom, since our share is translated against
        what others hold (see vllm_worker.effective_gmu). (A 3090 with 11.3 GB free against
        a 14.1 GB demand crash-looped Surya-2 for hours: vLLM died on startup, the next OCR
        request tried again. Then 3.6 GB of resident embedder did the same to a 0.35
        share.)"""
        if self._serve_mode() != "vllm":
            return 0.0
        try:
            from codai.api.vllm_worker import prelaunch_free_gb
            from codai.models.manager import get_active_vllm_config
            vcfg = get_active_vllm_config()
            return prelaunch_free_gb(vcfg, self._vlm_gmu()) if vcfg is not None else 0.0
        except Exception:
            return 0.0

    def _vlm_gmu(self):
        """The OCR-side share of the card for this engine's vLLM (0/None = the backend's)."""
        try:
            v = float(getattr(self.cfg, "vlm_gpu_memory_utilization", 0.0) or 0.0)
        except Exception:
            v = 0.0
        return v or None

    def vram_gb(self) -> float:
        """Per-instance footprint, for post-load eviction accounting and release reports.

        In the served modes a worker only speaks HTTP to the shared vLLM — its own VRAM is
        nil, and the instance's real cost is reported once through
        :meth:`prelaunch_vram_gb` / the vLLM stop estimate. Counting a whole GB per
        instance here asked to free 24 GB on a 24-instance pool."""
        return 0.05 if self._serve_mode() in self.SERVED_MODES else 1.0

    def load(self) -> None:
        if not self.cfg.surya_accept_license:
            raise OcrError(
                "Surya is license-gated. Set ocr.surya_accept_license = true to use it.",
                status=400,
            )
        self._server_url = ""
        self._server_key = ""
        self._served_name = ""
        mode = self._serve_mode()
        if mode == "model":
            # The recognition VLM is a MODEL this install already serves — on the
            # Digesta orchestrator a RunPod A40 pod it rents on demand. Ask whichever
            # backend owns it where it ended up, and attach there. This is the only
            # served mode that works on a box with no GPU of its own, which is also
            # the box whose entire job is to rent one.
            from codai.ocr.served_model import endpoint_for
            model_id = ((getattr(self.cfg, "surya_model_id", "") or "").strip()
                        or (getattr(self.cfg, "surya_model", "") or "").strip())
            if not model_id:
                raise OcrError(
                    "Surya 'model' mode needs ocr.surya_model_id — the id of the "
                    "Surya-2 checkpoint in models.json (it must be served over HTTP: "
                    "backend \"runpod\", a service_url, or the local vLLM backend).",
                    status=400)
            self._server_url, self._server_key, self._served_name = endpoint_for(model_id)
            print(f"[ocr] surya attaches to {self._server_url} "
                  f"(served as {self._served_name!r})", flush=True)
        elif mode == "vllm":
            # Serve the Surya2 VLM checkpoint via coderai's vLLM backend (continuous
            # batching) and attach Surya to it. Requires vllm enabled/auto-buildable.
            from codai.api import vllm_worker
            from codai.models.manager import get_active_vllm_config
            vcfg = get_active_vllm_config()
            if vcfg is None:
                raise OcrError("Surya vllm mode needs the vLLM backend configured", status=400)
            model = (getattr(self.cfg, "surya_model", "") or "datalab-to/surya-ocr-2").strip()
            base = vllm_worker.ensure_service(vcfg, model_path=model, served_name=model,
                                              gpu_memory_utilization=self._vlm_gmu(),
                                              **vlm_serve_limits(self.cfg))
            self._server_url = base.rstrip("/") + "/v1"
            self._served_name = model
        elif mode == "local":
            # Local means on this machine's GPU, or not at all. Surya would
            # otherwise pick its own device and quietly settle for the CPU, where
            # a page takes minutes instead of seconds and nothing in the result
            # says why. paddle and docTR refuse in the same situation; this used
            # to be the one engine that did not.
            if getattr(self.cfg, "surya_use_gpu", True) and not _cuda_present():
                raise OcrError(
                    "Surya 'local' mode runs the model on this machine and found "
                    "no CUDA device. It will NOT quietly run on the CPU, where a "
                    "page takes minutes. Use ocr.surya_serve = \"model\" to run it "
                    "on whatever serves the Surya-2 checkpoint (a RunPod pod, a "
                    "cluster node, a local vLLM), or set ocr.surya_use_gpu = false "
                    "to accept the CPU deliberately.",
                    status=503)
        elif mode == "llamacpp":
            url = (getattr(self.cfg, "surya_server_url", "") or "").strip()
            if not url:
                raise OcrError(
                    "Surya llamacpp mode needs ocr.surya_server_url (a running llama-server "
                    "OpenAI endpoint).", status=400)
            self._server_url = url
        super().load()

    # ------------------------------------------------------------------ leases

    def _endpoint_alive(self) -> bool:
        """Whether the address this worker was pinned to is still OURS to use.

        A `False` here is expensive: the caller kills this worker process and
        spawns a replacement, which costs ~600 MB and several seconds of torch
        import. So only a definite answer may return False.

        In `model` mode the address is always coderai's own OCR gateway, never a
        pod directly — the gateway is what holds the pool and drops a reaped pod.
        It answers 404 "unknown OCR engine lease" when the lease really is gone
        (the front restarted), and that is the one case re-resolving fixes. A 503
        means no upstream *right now*; the lease is fine.

        A probe that times out is not an answer at all, and it used to return
        False. On 2026-10-12, with 64 of these workers on a 16-core host, the
        5-second probe began timing out under the load the workers themselves
        were creating — so coderai killed and respawned them, which added load,
        which timed out more probes. The batch stalled for nine minutes with the
        GPU idle. Respawning cannot fix a front that is down (the replacement
        asks the same front for a lease), so a timeout now means "busy, keep
        working" and the page's real error is reported as it stands.
        """
        import requests
        from codai.ocr.served_model import health_url
        url = health_url(self._server_url) + "/health"
        last = None
        for timeout in (10, 20):
            try:
                r = requests.get(url, timeout=timeout)
            except Exception as exc:
                last = exc
                continue
            if r.status_code == 404:
                return False              # the lease is genuinely gone
            return True                   # 200, or 503 "no upstream yet"
        print(f"[ocr] surya: {url} did not answer the health probe "
              f"({type(last).__name__ if last else 'timeout'}) — treating it as "
              f"busy, not gone; the worker stays up", flush=True)
        return True

    def recognize_image(self, image):
        """One page, re-resolving the endpoint if the lease expired underneath us.

        In 'model' mode the address is a rented pod, and a pod that has been idle for
        its timeout is reaped — correctly, it costs money. The worker is then pinned to
        a URL nothing answers, and would fail every page from then on. So a failure is
        checked against the endpoint itself: if it is still healthy the error is real
        and is reported, and if it is gone the worker is restarted against a freshly
        provisioned one. Exactly once — a second failure is an outage, not a lease."""
        try:
            return super().recognize_image(image)
        except OcrError as first:
            if self._serve_mode() != "model" or self._endpoint_alive():
                raise
            print(f"[ocr] surya endpoint {self._server_url} is gone "
                  f"({first}) — re-resolving", flush=True)
            from codai.models import placement
            from codai.ocr import served_model
            served_model.forget()
            # The candidate set is discovered, so a vanished pod or node must be
            # re-discovered too; otherwise the re-resolve hands back the same
            # dead address it just rejected.
            placement.forget()
            self.cleanup()
            self.load()
            self._loaded = True
            return super().recognize_image(image)
