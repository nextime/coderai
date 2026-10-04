# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""OCR engine manager — per-engine instance pools for concurrent GPU OCR.

An "instance" is one loaded engine; for the VLM engine (olmOCR) it is one in-flight
request against the serving model, so the pool size still bounds that engine's
concurrency.

Each engine is loaded as N resident instances (``*_instances`` in
:class:`~codai.config.OcrConfig`); pages are fanned across the pool so many documents
OCR concurrently on one GPU. An engine instance handles one page at a time (OCR runtimes
are not reentrant), so pool size bounds that engine's concurrency; ``max_concurrency``
bounds the whole subsystem.
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional

from codai.ocr.base import OcrEngine, OcrPage, OcrError, load_pages


# Engine registry: name → factory(cfg) → OcrEngine. Imports are deferred inside each
# factory so a missing OCR dependency never breaks manager import.
def _paddle_factory(cfg):
    from codai.ocr.paddle import PaddleEngine
    return PaddleEngine(cfg)


def _doctr_factory(cfg):
    from codai.ocr.doctr import DoctrEngine
    return DoctrEngine(cfg)


def _surya_factory(cfg):
    from codai.ocr.surya import SuryaEngine
    return SuryaEngine(cfg)


def _olmocr_factory(cfg):
    from codai.ocr.olmocr import OlmOcrEngine
    return OlmOcrEngine(cfg)


_ENGINE_FACTORIES = {
    "paddle": _paddle_factory,
    "doctr": _doctr_factory,
    "surya": _surya_factory,
    "olmocr": _olmocr_factory,
}


def _engine_enabled(cfg, name: str) -> bool:
    if name == "paddle":
        return bool(cfg.paddle_enabled)
    if name == "doctr":
        return bool(cfg.doctr_enabled)
    if name == "surya":
        return bool(cfg.surya_enabled and cfg.surya_accept_license)
    if name == "olmocr":
        return bool(getattr(cfg, "olmocr_enabled", False))
    return False


def _engine_instances(cfg, name: str) -> int:
    n = getattr(cfg, f"{name}_instances", 1)
    try:
        return max(1, int(n))
    except Exception:
        return 1


def _evict_for_ocr(needed_gb: float) -> None:
    """Ask the model manager to free ``needed_gb`` of VRAM before an OCR pool loads,
    so OCR instances contend for VRAM on equal footing with LLM/diffusion models
    (evict LRU models rather than OOM). Best-effort: never breaks OCR if the manager
    is unavailable."""
    try:
        if needed_gb <= 0:
            return
        from codai.models.manager import multi_model_manager
        multi_model_manager._evict_models_for_vram(float(needed_gb))
    except Exception as e:
        print(f"[ocr] VRAM evict-before-build skipped: {e}")


def _wait_thermal_safe() -> None:
    """Block until GPU temps are within safe limits (shared thermal governor).
    Runs OCR through the same cooldown gate as every other GPU workload."""
    try:
        from codai.models import thermal
        thermal.wait_until_safe(context="ocr")
    except Exception as e:
        print(f"[ocr] thermal wait skipped: {e}")


class _Pool:
    """A fixed-size pool of loaded engine instances, served via an asyncio.Queue."""

    def __init__(self, name: str, cfg, size: int):
        self.name = name
        self.cfg = cfg
        self.size = size
        self._q: asyncio.Queue = asyncio.Queue()
        self._instances = []          # every created instance (for release_sync)
        self._built = False
        self._build_lock = asyncio.Lock()
        self._fail_at = 0.0           # when the last build attempt failed
        self._fail_err = None         # and why (re-raised during the cooldown)
        # In-flight page accounting, for the clean swap in drain_and_release(). Touched
        # from the event loop (recognize) and READ from the model manager's eviction
        # thread, so it is guarded by a threading primitive rather than an asyncio one.
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._draining = False

    def _enter_page(self) -> None:
        with self._inflight_lock:
            self._inflight += 1
            self._idle.clear()

    def _leave_page(self) -> None:
        with self._inflight_lock:
            self._inflight = max(0, self._inflight - 1)
            if self._inflight == 0:
                self._idle.set()

    def _drain_timeout_s(self) -> float:
        try:
            return max(0.0, float(getattr(self.cfg, "evict_drain_timeout_s", 60.0)))
        except Exception:
            return 60.0

    def drain_and_release(self) -> float:
        """Let in-flight pages finish, THEN tear the pool down. Returns GB freed.

        This is the OCR side of the clean swap the model manager already does for a busy
        model (``release_idle_vram`` waits for a request boundary rather than evicting
        mid-request). Tearing the pool down under a page that is being recognised loses
        that page — and the caller's whole document — which is exactly what must not
        happen when OCR and a model take turns on one card. Bounded by
        ``ocr.evict_drain_timeout_s`` so a wedged engine cannot block the load forever."""
        self._draining = True
        try:
            budget = self._drain_timeout_s()
            with self._inflight_lock:
                busy = self._inflight
            if busy and budget > 0:
                print(f"[ocr] engine '{self.name}': {busy} page(s) in flight — waiting up "
                      f"to {budget:.0f}s for a clean swap before releasing VRAM")
                if not self._idle.wait(budget):
                    with self._inflight_lock:
                        stuck = self._inflight
                    print(f"[ocr] engine '{self.name}': {stuck} page(s) still in flight "
                          f"after {budget:.0f}s — releasing anyway (those pages will be "
                          f"re-read by the fallback engine)")
            return self.release_sync()
        finally:
            self._draining = False

    def _cooldown_s(self) -> float:
        try:
            return max(0.0, float(getattr(self.cfg, "build_retry_cooldown_s", 60.0)))
        except Exception:
            return 60.0

    async def _ensure_built(self):
        if self._built:
            return
        async with self._build_lock:
            if self._built:
                return
            # A release is draining right now: let it finish before rebuilding, or we
            # would re-evict the model that just took its turn and load into VRAM the
            # drain is still about to free. Taking turns means waiting for the handover.
            _waited = 0.0
            while self._draining and _waited < self._drain_timeout_s() + 5.0:
                await asyncio.sleep(0.05)
                _waited += 0.05
            # A build that just failed is very unlikely to succeed on the next request a
            # second later, and retrying is not free: a vLLM-backed engine spends ~35 s
            # booting before it dies. Surya-2 did that 338 times in a row against a card
            # that was 11 GB short — every one of those requests hung for half a minute
            # and then failed anyway. Inside the cooldown, fail FAST with the same reason.
            if self._fail_err is not None and (time.time() - self._fail_at) < self._cooldown_s():
                left = int(self._cooldown_s() - (time.time() - self._fail_at))
                raise OcrError(
                    f"{self._fail_err} (retrying in {left}s — last attempt failed; "
                    f"OCR engine '{self.name}' is in its build cooldown)",
                    status=getattr(self._fail_err, "status", 503))
            factory = _ENGINE_FACTORIES[self.name]
            self._instances = []
            try:
                for _ in range(self.size):
                    eng = factory(self.cfg)
                    first = not self._instances
                    need = 0.0
                    if first:
                        # Engines that boot a server claiming a fixed share of the card
                        # (the VLM engines on vLLM) must have the room BEFORE they load —
                        # the load is what fails otherwise. The figure covers the whole
                        # pool, because those instances share one server.
                        try:
                            need = float(eng.prelaunch_vram_gb() or 0.0)
                        except Exception:
                            need = 0.0
                        if need > 0:
                            await asyncio.to_thread(_evict_for_ocr, need)
                    # Load the first instance synchronously-in-thread so a missing
                    # dependency surfaces as OcrError(503) before we spawn the rest.
                    await asyncio.to_thread(eng.ensure_loaded)
                    # First instance of an engine that did NOT reserve up front: evict for
                    # its real footprint now that we know it (paddle/docTR), per instance.
                    # Reserving again here would evict live models for memory the engine's
                    # server has already taken.
                    if first and need <= 0:
                        await asyncio.to_thread(_evict_for_ocr, self.size * eng.vram_gb())
                    self._instances.append(eng)
                    await self._q.put(eng)
            except Exception as e:
                self.release_sync()
                self._fail_at = time.time()
                self._fail_err = e if isinstance(e, OcrError) else OcrError(
                    f"OCR engine '{self.name}' failed to load: {e}", status=503)
                raise self._fail_err
            self._fail_err = None
            self._built = True
            print(f"[ocr] engine '{self.name}': {self.size} instance(s) ready")

    def release_sync(self) -> float:
        """Tear down all instances (free VRAM). SYNC — safe to call from the model
        manager's eviction thread. Returns estimated GB freed."""
        freed = 0.0
        for eng in list(self._instances):
            try:
                freed += eng.vram_gb()
                eng.cleanup()
            except Exception:
                pass
        self._instances = []
        # Drain the queue so a rebuild starts clean.
        try:
            while not self._q.empty():
                self._q.get_nowait()
        except Exception:
            pass
        self._built = False
        return freed

    async def recognize(self, image) -> OcrPage:
        await self._ensure_built()
        eng: OcrEngine = await self._q.get()
        self._enter_page()
        try:
            # Engines that call back into coderai's async API (olmOCR 'model' mode) run
            # in a worker thread and need a live loop to hand the coroutine to; give them
            # the app's own rather than letting them spin up a second one.
            try:
                eng._host_loop = asyncio.get_running_loop()
            except Exception:
                pass
            return await asyncio.to_thread(eng.recognize_image, image)
        finally:
            self._leave_page()
            await self._q.put(eng)


class OcrManager:
    """Holds per-engine pools; (re)configured from OcrConfig on demand."""

    def __init__(self):
        self._cfg = None
        self._pools: Dict[str, _Pool] = {}
        self._sem: Optional[asyncio.Semaphore] = None
        self._lock = asyncio.Lock()
        self._releaser_registered = False

    def configure(self, cfg) -> None:
        """Point the manager at the current OcrConfig. Rebuilds pools if params changed."""
        prev = self._cfg
        self._cfg = cfg
        if prev is None or self._pool_params(prev) != self._pool_params(cfg):
            # Drop stale pools. Tear down live instances first so their VRAM is freed
            # promptly (rather than waiting on GC of the subprocess workers / torch models).
            for pool in self._pools.values():
                try:
                    pool.release_sync()
                except Exception:
                    pass
            self._pools = {}
            self._sem = asyncio.Semaphore(max(1, int(getattr(cfg, "max_concurrency", 4))))
        self._register_releaser()

    def _register_releaser(self) -> None:
        """Register OCR VRAM as reclaimable by the model manager, so loading an
        LLM/diffusion model can evict OCR pools (not just the reverse)."""
        if self._releaser_registered:
            return
        try:
            from codai.models.manager import multi_model_manager
            multi_model_manager.register_external_vram_releaser(self._release_vram)
            self._releaser_registered = True
        except Exception as e:
            print(f"[ocr] could not register VRAM releaser: {e}")

    def _release_vram(self, needed_gb: float = 999.0) -> float:
        """External VRAM releaser (called as ``fn(needed_gb)`` from the model manager's
        eviction path, SYNC). Tears down every built OCR pool and returns the estimated
        GB freed. Pools rebuild lazily on the next OCR request, so OCR takes its turn on
        the card again like any other model rather than staying dead for the rest of the
        process' life. ``needed_gb`` is advisory — OCR pools are all-or-nothing per
        engine, so we release everything held.

        In-flight pages are allowed to FINISH first (see ``_Pool.drain_and_release``):
        the handover is a clean swap at a page boundary, not a kill."""
        freed = 0.0
        for pool in list(self._pools.values()):
            try:
                freed += pool.drain_and_release()
            except Exception:
                pass
        # The managed Surya-2 vLLM subprocess isn't a manager-tracked model and its VRAM
        # (gpu_memory_utilization × card) isn't reclaimed by tearing down the worker pool
        # above — it must be stopped explicitly. Do it here so on-request eviction can
        # reclaim it; the next OCR request re-boots it via ensure_service.
        freed += self._stop_surya_vllm()
        freed += self._stop_olmocr_vllm()
        if freed:
            print(f"[ocr] released ~{freed:.1f} GB (pools + Surya vLLM torn down for VRAM eviction)")
        return freed

    def _stop_surya_vllm(self) -> float:
        """Stop the managed Surya-2 vLLM subprocess (if this build serves Surya via vLLM).
        Returns estimated GB freed."""
        # NOT gated on surya_serve being "vllm" right now: if the setting was changed
        # while a service was up, the old subprocess is still holding its share of the
        # card and skipping the stop would leak it until a restart. stop_service_for() is
        # a no-op returning 0.0 when nothing is running, so asking is free.
        cfg = self._cfg
        if cfg is None:
            return 0.0
        try:
            from codai.api import vllm_worker
            from codai.models.manager import get_active_vllm_config
            vcfg = get_active_vllm_config()
            if vcfg is None:
                return 0.0
            model = (getattr(cfg, "surya_model", "") or "datalab-to/surya-ocr-2").strip()
            return vllm_worker.stop_service_for(
                vcfg, model_path=model, served_name=model,
                gpu_memory_utilization=float(getattr(cfg, "vlm_gpu_memory_utilization", 0.0) or 0.0) or None)
        except Exception as e:
            print(f"[ocr] Surya vLLM stop skipped: {e}")
            return 0.0

    def _stop_olmocr_vllm(self) -> float:
        """Stop the managed olmOCR vLLM instance (olmocr_serve = "vllm"), for the same
        reason Surya's needs stopping: its VRAM is the subprocess's, not a pool's."""
        cfg = self._cfg          # unconditional, for the reason in _stop_surya_vllm
        if cfg is None:
            return 0.0
        try:
            from codai.api import vllm_worker
            from codai.models.manager import get_active_vllm_config
            from codai.ocr.olmocr import DEFAULT_MODEL
            vcfg = get_active_vllm_config()
            if vcfg is None:
                return 0.0
            model = (getattr(cfg, "olmocr_model", "") or DEFAULT_MODEL).strip()
            return vllm_worker.stop_service_for(
                vcfg, model_path=model, served_name=model,
                gpu_memory_utilization=float(getattr(cfg, "vlm_gpu_memory_utilization", 0.0) or 0.0) or None)
        except Exception as e:
            print(f"[ocr] olmOCR vLLM stop skipped: {e}")
            return 0.0

    @staticmethod
    def _pool_params(cfg) -> tuple:
        return (
            cfg.max_concurrency,
            cfg.paddle_enabled, cfg.paddle_instances, cfg.paddle_use_gpu,
            cfg.paddle_structure, cfg.paddle_lang,
            cfg.paddle_det_model_dir, cfg.paddle_rec_model_dir,
            cfg.paddle_venv, cfg.paddle_auto_build,
            cfg.doctr_enabled, cfg.doctr_instances, cfg.doctr_use_gpu,
            cfg.doctr_det_arch, cfg.doctr_reco_arch,
            cfg.surya_enabled, cfg.surya_accept_license, cfg.surya_instances,
            cfg.surya_langs, cfg.surya_venv, cfg.surya_auto_build,
            getattr(cfg, "olmocr_enabled", False), getattr(cfg, "olmocr_instances", 1),
            getattr(cfg, "olmocr_serve", "model"), getattr(cfg, "olmocr_model", ""),
            getattr(cfg, "olmocr_model_id", ""), getattr(cfg, "olmocr_server_url", ""),
            getattr(cfg, "olmocr_longest_side", 1288),
            getattr(cfg, "vlm_gpu_memory_utilization", 0.0),
        )

    def resolve_engine(self, requested: Optional[str]) -> str:
        cfg = self._require_cfg()
        name = (requested or cfg.default_engine or "paddle").strip().lower()
        if name not in _ENGINE_FACTORIES:
            raise OcrError(f"unknown OCR engine '{name}'", status=400)
        if not _engine_enabled(cfg, name):
            raise OcrError(
                f"OCR engine '{name}' is not enabled (or license not accepted). "
                f"Enable it in Settings → OCR.",
                status=400,
            )
        return name

    async def _get_pool(self, name: str) -> _Pool:
        async with self._lock:
            pool = self._pools.get(name)
            if pool is None:
                cfg = self._require_cfg()
                pool = _Pool(name, cfg, _engine_instances(cfg, name))
                self._pools[name] = pool
            return pool

    def _fallback_order(self, failed: str) -> List[str]:
        """Enabled engines to try after ``failed`` could not serve the document.

        An OCR request that comes back 503 is a document nobody read — and the whole
        point of having four engines is that a page can still be read when one of them
        will not come up (a VLM engine whose vLLM cannot fit on the card, a venv that
        needs rebuilding, a license flag). ``auto`` prefers the engines with the fewest
        ways to fail over the ones with the best output, because the alternative to a
        worse transcription here is no transcription."""
        cfg = self._require_cfg()
        raw = str(getattr(cfg, "fallback_engines", "auto") or "auto").strip().lower()
        if raw in ("off", "none", "no", "false"):
            return []
        if raw in ("auto", "", "on", "true"):
            names = ["paddle", "doctr", "olmocr", "surya"]
        else:
            names = [n.strip() for n in raw.replace(",", " ").split() if n.strip()]
        return [n for n in names
                if n != failed and n in _ENGINE_FACTORIES and _engine_enabled(cfg, n)]

    async def _pages_with(self, name: str, images: List) -> List[OcrPage]:
        pool = await self._get_pool(name)
        sem = self._sem or asyncio.Semaphore(4)

        async def _one(idx, img):
            async with sem:
                page = await pool.recognize(img)
                page.index = idx
                return page

        pages = await asyncio.gather(*[_one(i, im) for i, im in enumerate(images)])
        return list(pages)

    async def ocr_pages(self, images: List, engine: Optional[str] = None) -> (str, List[OcrPage]):
        """OCR a list of PIL page images. Returns (engine_name, [OcrPage]).

        The returned name is the engine that ACTUALLY read the pages, which is not
        necessarily the one asked for: when an engine cannot serve, the remaining
        enabled engines are tried in turn (see :meth:`_fallback_order`) rather than
        failing the document. Callers surface the name, so a caller that cares can see
        it fell back."""
        name = self.resolve_engine(engine)
        try:
            return name, await self._pages_with(name, images)
        except Exception as first:
            chain = self._fallback_order(name)
            if not chain:
                raise
            print(f"[ocr] engine '{name}' could not serve ({first}); "
                  f"falling back to {' -> '.join(chain)}")
            for alt in chain:
                try:
                    pages = await self._pages_with(alt, images)
                except Exception as e:
                    print(f"[ocr] fallback engine '{alt}' also failed: {e}")
                    continue
                print(f"[ocr] served by fallback engine '{alt}' instead of '{name}'")
                return alt, pages
            # Everything enabled is down: the original failure is the useful one.
            raise first

    async def ocr_document(self, data: bytes, filename: str = "", content_type: str = "",
                           engine: Optional[str] = None, dpi: Optional[int] = None,
                           detect: Optional[str] = None, structured: bool = False,
                           schema: Optional[str] = None) -> dict:
        """OCR raw bytes (image/PDF) end-to-end → response dict.

        ``detect`` overrides the configured stamp/signature detection mode
        (off|layout|detector|both). ``structured`` triggers field extraction via the
        configured text model; ``schema`` overrides the extraction schema for this call.
        """
        cfg = self._require_cfg()
        use_dpi = int(dpi) if dpi else int(cfg.dpi)
        images = load_pages(data, filename=filename, content_type=content_type, dpi=use_dpi)
        # Share the GPU thermal governor with every other workload: block here until
        # temps are safe (wait_until_safe blocks, so run it off the event loop).
        await asyncio.to_thread(_wait_thermal_safe)
        name, pages = await self.ocr_pages(images, engine=engine)
        full_text = "\n\n".join(p.text for p in pages)

        stamps, signatures = await self._detect_all(images, pages, detect)

        structured_out = None
        if structured:
            from codai.ocr.extract import extract_fields
            structured_out = await extract_fields(full_text, cfg, schema=schema)

        return {
            "engine": name,
            "num_pages": len(pages),
            "text": full_text,
            "pages": [p.to_dict() for p in pages],
            "stamps": stamps,
            "signatures": signatures,
            "structured": structured_out,
        }

    async def _detect_all(self, images, pages, detect: Optional[str]):
        """Run stamp/signature detection across pages; returns (stamps, signatures)
        with each detection tagged by page index. Empty when mode is 'off'."""
        from codai.ocr.detect import resolve_detect_mode, detect_page

        cfg = self._require_cfg()
        mode = resolve_detect_mode(cfg, detect)
        if mode == "off":
            return [], []

        sem = self._sem or asyncio.Semaphore(4)

        async def _one(idx, img, page):
            async with sem:
                res = await asyncio.to_thread(detect_page, img, page, cfg, mode)
                for d in res["stamps"]:
                    d["page"] = idx
                for d in res["signatures"]:
                    d["page"] = idx
                return res

        results = await asyncio.gather(*[_one(i, im, pg) for i, (im, pg) in enumerate(zip(images, pages))])
        stamps, signatures = [], []
        for r in results:
            stamps.extend(r["stamps"])
            signatures.extend(r["signatures"])
        return stamps, signatures

    def _require_cfg(self):
        if self._cfg is None:
            raise OcrError("OCR subsystem is not configured", status=503)
        if not getattr(self._cfg, "enabled", False):
            raise OcrError("OCR subsystem is disabled (enable it in Settings → OCR)", status=400)
        return self._cfg


# Module-level singleton (mirrors multi_model_manager).
ocr_manager = OcrManager()
