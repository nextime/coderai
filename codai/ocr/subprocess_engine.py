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


def _resolve_venv_dir(configured: str, name: str, baked_aliases: tuple = ()) -> str:
    """Resolve an isolated-venv directory for an OCR engine.

    Precedence: explicit config path > a venv BAKED into the image
    (/opt/coderai/<name>, or /opt/coderai/venvs/<alias>) > a build on the
    PERSISTENT cache mount (CODERAI_CACHE_DIR or /cache → <cache>/ocr/<name>,
    survives container restarts) > ~/.coderai/<name> for host/source installs.

    The order matters: a runtime-built venv must NOT land in the image's writable layer
    (lost on `coderai docker` restart) — it goes on the /cache mount, like the colibri/k3
    engines.

    ``baked_aliases`` are the names the capability-image builder uses: a profile's
    ``<profile>.venv-<alias>.txt`` becomes ``/opt/coderai/venvs/<alias>``, which is
    NOT ``/opt/coderai/<engine>_venv``. Without them a baked venv is invisible and
    the engine rebuilds it on the cache mount at first use — paying minutes (or,
    on a pod, failing) for something already in the image.
    """
    if configured:
        return configured
    for baked in (f"/opt/coderai/{name}",
                  *(f"/opt/coderai/venvs/{a}" for a in baked_aliases)):
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
        self._lock = threading.Lock()        # guards _pending
        self._wlock = threading.Lock()       # one writer on the worker's stdin
        self._pending = {}                   # request id -> {"event", "reply"}
        self._next_id = 0
        self._mux = False                    # replies carry ids and arrive out of order
        self._reader = None

    # -- subclass hooks ----------------------------------------------------

    def _venv_dir(self) -> str:
        raise NotImplementedError

    def _requirements(self) -> str:
        """Path to the requirements file for auto-build (or '' to disable auto-build)."""
        return ""

    def _auto_build(self) -> bool:
        return False

    def lanes(self) -> int:
        """How many pages this engine's worker may have in flight at once.

        One, unless a subclass knows better. A lane costs a worker OBJECT inside
        the existing process, not a process — which is only cheap when the
        engine's heavy part lives somewhere else (see SuryaEngine.lanes).
        """
        return 1

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
        opts = dict(self._worker_opts())
        want = max(1, int(self.lanes() or 1))
        opts["lanes"] = want
        resp = self._rpc({"cmd": "load", "opts": opts})
        if not resp.get("ok"):
            self._kill()
            raise OcrError(resp.get("error") or f"{self.name}: load failed",
                           status=int(resp.get("status", 500)))
        # The worker echoes the lanes it will actually serve. Only then does the
        # parent start reading replies out of order: a worker that did not answer
        # with a lane count is driven exactly as before, one page at a time.
        got = int(resp.get("lanes") or 1)
        self._mux = got > 1
        if self._mux:
            self._reader = threading.Thread(target=self._read_replies, daemon=True,
                                            name=f"ocr-{self.name}-reader")
            self._reader.start()
            print(f"[ocr] {self.name}: worker serves {got} page(s) at once "
                  f"in one process", flush=True)

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

    def _read_replies(self) -> None:
        """Hand each reply to the page that asked for it, until the pipe closes.

        A reply with no id belongs to a sequential RPC (load, ping) and cannot
        reach here: those run before the reader starts. Anything unrecognised is
        dropped rather than guessed at — attributing a page's text to the wrong
        request is the one failure worse than losing it.
        """
        proc = self._proc
        while True:
            try:
                line = proc.stdout.readline()
            except Exception:
                break
            if not line:
                break
            try:
                msg = json.loads(line)
            except Exception:
                continue
            rid = msg.get("id")
            if rid is None:
                continue
            with self._lock:
                slot = self._pending.pop(rid, None)
            if slot is not None:
                slot["reply"] = msg
                slot["event"].set()
        # The worker is gone. Everyone still waiting would otherwise wait for ever.
        self._fail_pending(f"{self.name}: worker closed the pipe")

    def _fail_pending(self, why: str) -> None:
        with self._lock:
            waiting = list(self._pending.values())
            self._pending.clear()
        for slot in waiting:
            slot["reply"] = {"ok": False, "error": why, "status": 500}
            slot["event"].set()

    def _rpc_mux(self, msg: dict) -> dict:
        """One page, tagged, answered by the reader thread whenever it lands."""
        if self._proc is None or self._proc.poll() is not None:
            raise OcrError(f"{self.name}: worker not running", status=500)
        slot = {"event": threading.Event(), "reply": None}
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            self._pending[rid] = slot
        try:
            line = json.dumps(dict(msg, id=rid)) + "\n"
            with self._wlock:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
        except Exception as e:
            with self._lock:
                self._pending.pop(rid, None)
            raise OcrError(f"{self.name}: worker not running ({e})", status=500)
        # No deadline on the page itself — a dense scan legitimately takes
        # minutes, and the engine above has its own timeout. But a worker that
        # DIED must not read as a page still being worked on, so the wait is
        # broken up to check that it is still there.
        while not slot["event"].wait(5.0):
            if self._proc is None or self._proc.poll() is not None:
                with self._lock:
                    self._pending.pop(rid, None)
                raise OcrError(f"{self.name}: worker died while reading a page",
                               status=500)
        return slot["reply"] or {"ok": False, "error": f"{self.name}: no reply"}

    def _kill(self):
        try:
            if self._proc:
                self._proc.kill()
        except Exception:
            pass
        self._proc = None
        self._mux = False
        self._fail_pending(f"{self.name}: worker stopped")

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
        if self._mux:
            resp = self._rpc_mux({"cmd": "ocr", "image": b64})
        else:
            with self._lock:                   # one request at a time per subprocess
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
        # The worker's byte-level diagnosis of an unreadable page, when it made one.
        # Carried on meta rather than a field of its own so engines that cannot
        # produce it change nothing about the response shape.
        trace = d.get("unreadable_trace")
        if trace:
            page.meta["unreadable_trace"] = trace
        # What the worker had to rebuild. Carried for the same reason the trace is:
        # the text is no longer byte-for-byte what the server sent, and a caller
        # comparing the two needs to know why. Missing this is why the first
        # successful repair reported `accents_repaired: null` while visibly having
        # fixed the page -- the fix worked and only its receipt was dropped.
        repaired = d.get("accents_repaired")
        if repaired:
            page.meta["accents_repaired"] = repaired
        return page
