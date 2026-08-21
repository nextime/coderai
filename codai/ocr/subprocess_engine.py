# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.

"""Isolated-venv subprocess OCR engine.

PaddleOCR and Surya have dependencies that conflict with the main coderai venv
(PaddleOCR pulls opencv-contrib-python; Surya caps pillow<11). Each therefore runs in its
OWN virtualenv, driven by ``codai/ocr/workers/ocr_worker.py`` over a newline-delimited
JSON pipe — the same isolation pattern coderai uses for the DINOv2-SALAD embedder.

Isolation bonus for PaddleOCR: paddlepaddle-gpu wheels bundle their own CUDA runtime, so a
cu12x Paddle wheel runs on GPU under a newer driver (e.g. CUDA-13 host) regardless of the
main venv's torch CUDA.
"""

import json
import os
import subprocess
import sys
import threading

from codai.ocr.base import OcrEngine, OcrPage, OcrLine, OcrRegion, OcrError

_WORKER = os.path.join(os.path.dirname(__file__), "workers", "ocr_worker.py")


def _resolve_venv_dir(configured: str, name: str) -> str:
    """Resolve an isolated-venv directory for an OCR engine.

    Precedence: explicit config path > a venv BAKED into the image
    (/opt/coderai/<name>) > a build on the PERSISTENT cache mount
    (CODERAI_CACHE_DIR or /cache → <cache>/ocr/<name>, survives container restarts) >
    ~/.coderai/<name> for host/source installs.

    The order matters: a runtime-built venv must NOT land in the image's writable layer
    (lost on `coderai docker` restart) — it goes on the /cache mount, like the colibri/k3
    engines.
    """
    if configured:
        return configured
    baked = f"/opt/coderai/{name}"
    if os.path.isdir(baked):
        return baked
    cache = os.environ.get("CODERAI_CACHE_DIR") or ("/cache" if os.path.isdir("/cache") else "")
    if cache and os.path.isdir(cache):
        return os.path.join(cache, "ocr", name)
    return os.path.expanduser(f"~/.coderai/{name}")


class SubprocessOcrEngine(OcrEngine):
    """Base for engines that run in an isolated venv subprocess."""

    # Subclasses set these:
    name = "subprocess"
    _worker_engine = ""          # "paddle" | "surya"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._proc = None
        self._lock = threading.Lock()

    # -- subclass hooks ----------------------------------------------------

    def _venv_dir(self) -> str:
        raise NotImplementedError

    def _requirements(self) -> str:
        """Path to the requirements file for auto-build (or '' to disable auto-build)."""
        return ""

    def _auto_build(self) -> bool:
        return False

    def _worker_opts(self) -> dict:
        return {}

    def _worker_env(self) -> dict:
        """Extra environment variables for the worker subprocess (overridable)."""
        return {}

    def _pip_install_cmd(self, py: str, req: str):
        """Return the pip command list to build the venv. Overridable (e.g. paddle index)."""
        return [py, "-m", "pip", "install", "-r", req]

    # -- lifecycle ---------------------------------------------------------

    def _venv_python(self) -> str:
        return os.path.join(os.path.expanduser(self._venv_dir()), "bin", "python")

    def _ensure_venv(self) -> str:
        py = self._venv_python()
        if os.path.isfile(py):
            return py

        # Missing venv. If a build is already in flight (web trigger or a prior first-use
        # request), report that. Otherwise, if auto-build is on, kick off a NON-BLOCKING
        # background build and tell the caller to retry — never block the request for the
        # minutes a torch/paddle install takes. If auto-build is off, point at the UI/CLI.
        from codai.ocr import venv_build
        if venv_build.is_running(self.name):
            raise OcrError(f"{self.name}: isolated venv is still building — retry shortly",
                           status=503)
        if not self._auto_build():
            raise OcrError(
                f"{self.name}: isolated venv not found at {os.path.dirname(os.path.dirname(py))}. "
                f"Build it from Settings → OCR (\"Build {self.name} venv now\"), enable "
                f"auto-build, or run: python3 -m venv <dir> && <dir>/bin/pip install -r "
                f"{os.path.basename(self._requirements())}.",
                status=503,
            )
        req = self._requirements()
        if not req or not os.path.isfile(req):
            raise OcrError(f"{self.name}: requirements file not found ({req})", status=500)
        venv_build.build_async(self.cfg, self.name)
        raise OcrError(
            f"{self.name}: isolated venv build started (installing dependencies) — "
            f"retry in a few minutes; watch progress in Settings → OCR.",
            status=503,
        )

    def load(self) -> None:
        py = self._ensure_venv()
        env = dict(os.environ)
        env.update(self._worker_env())
        try:
            self._proc = subprocess.Popen(
                [py, _WORKER, self._worker_engine],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, text=True, bufsize=1, env=env,
            )
        except Exception as e:
            raise OcrError(f"{self.name}: failed to launch worker: {e}", status=500)
        # Handshake: wait for {"ready": true}
        ready = self._readline()
        if not ready.get("ready"):
            raise OcrError(f"{self.name}: worker did not signal ready", status=500)
        resp = self._rpc({"cmd": "load", "opts": self._worker_opts()})
        if not resp.get("ok"):
            self._kill()
            raise OcrError(resp.get("error") or f"{self.name}: load failed",
                           status=int(resp.get("status", 500)))

    def _readline(self) -> dict:
        line = self._proc.stdout.readline()
        if not line:
            raise OcrError(f"{self.name}: worker closed the pipe", status=500)
        try:
            return json.loads(line)
        except Exception as e:
            raise OcrError(f"{self.name}: bad worker output: {e}", status=500)

    def _rpc(self, msg: dict) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            raise OcrError(f"{self.name}: worker not running", status=500)
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()
        return self._readline()

    def _kill(self):
        try:
            if self._proc:
                self._proc.kill()
        except Exception:
            pass
        self._proc = None

    def cleanup(self) -> None:
        """Stop the worker subprocess (frees its VRAM). SYNC."""
        self._kill()
        self._loaded = False

    def vram_gb(self) -> float:
        # paddle ≈ 2.44 GB/inst measured; surya-local similar. (surya vLLM footprint is
        # the separate vLLM server, not per-worker.)
        return 2.5

    def recognize_image(self, image) -> OcrPage:
        import base64
        import io
        buf = io.BytesIO()
        image.convert("RGB").save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        with self._lock:                       # one request at a time per subprocess
            resp = self._rpc({"cmd": "ocr", "image": b64})
        if not resp.get("ok"):
            raise OcrError(resp.get("error") or f"{self.name}: ocr failed",
                           status=int(resp.get("status", 500)))
        return self._page_from_dict(resp.get("page") or {})

    @staticmethod
    def _page_from_dict(d: dict) -> OcrPage:
        page = OcrPage(index=int(d.get("index", 0)), text=d.get("text", ""),
                       width=int(d.get("width", 0)), height=int(d.get("height", 0)))
        page.lines = [OcrLine(text=l.get("text", ""), bbox=list(l.get("bbox", [0, 0, 0, 0])),
                              conf=float(l.get("conf", 0.0))) for l in d.get("lines", [])]
        page.regions = [OcrRegion(label=r.get("label", ""), bbox=list(r.get("bbox", [0, 0, 0, 0])),
                                  conf=float(r.get("conf", 0.0))) for r in d.get("regions", [])]
        page.tables = list(d.get("tables", []))
        return page
