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

Each engine is loaded as N resident instances (``*_instances`` in
:class:`~codai.config.OcrConfig`); pages are fanned across the pool so many documents
OCR concurrently on one GPU. An engine instance handles one page at a time (OCR runtimes
are not reentrant), so pool size bounds that engine's concurrency; ``max_concurrency``
bounds the whole subsystem.
"""

import asyncio
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


_ENGINE_FACTORIES = {
    "paddle": _paddle_factory,
    "doctr": _doctr_factory,
    "surya": _surya_factory,
}


def _engine_enabled(cfg, name: str) -> bool:
    if name == "paddle":
        return bool(cfg.paddle_enabled)
    if name == "doctr":
        return bool(cfg.doctr_enabled)
    if name == "surya":
        return bool(cfg.surya_enabled and cfg.surya_accept_license)
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

    async def _ensure_built(self):
        if self._built:
            return
        async with self._build_lock:
            if self._built:
                return
            factory = _ENGINE_FACTORIES[self.name]
            self._instances = []
            for _ in range(self.size):
                eng = factory(self.cfg)
                # Load the first instance synchronously-in-thread so a missing
                # dependency surfaces as OcrError(503) before we spawn the rest.
                await asyncio.to_thread(eng.ensure_loaded)
                # First instance: evict other models to make room (VRAM eviction), now
                # that we know its real footprint. Mirrors the engine-load path.
                if not self._instances:
                    await asyncio.to_thread(_evict_for_ocr, self.size * eng.vram_gb())
                self._instances.append(eng)
                await self._q.put(eng)
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
        try:
            return await asyncio.to_thread(eng.recognize_image, image)
        finally:
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
        GB freed. Pools rebuild lazily on the next OCR request. ``needed_gb`` is advisory
        — OCR pools are all-or-nothing per engine, so we release everything held."""
        freed = 0.0
        for pool in list(self._pools.values()):
            try:
                freed += pool.release_sync()
            except Exception:
                pass
        # The managed Surya-2 vLLM subprocess isn't a manager-tracked model and its VRAM
        # (gpu_memory_utilization × card) isn't reclaimed by tearing down the worker pool
        # above — it must be stopped explicitly. Do it here so on-request eviction can
        # reclaim it; the next OCR request re-boots it via ensure_service.
        freed += self._stop_surya_vllm()
        if freed:
            print(f"[ocr] released ~{freed:.1f} GB (pools + Surya vLLM torn down for VRAM eviction)")
        return freed

    def _stop_surya_vllm(self) -> float:
        """Stop the managed Surya-2 vLLM subprocess (if this build serves Surya via vLLM).
        Returns estimated GB freed."""
        cfg = self._cfg
        if cfg is None or (getattr(cfg, "surya_serve", "") or "").strip().lower() != "vllm":
            return 0.0
        try:
            from codai.api import vllm_worker
            from codai.models.manager import get_active_vllm_config
            vcfg = get_active_vllm_config()
            if vcfg is None:
                return 0.0
            model = (getattr(cfg, "surya_model", "") or "datalab-to/surya-ocr-2").strip()
            return vllm_worker.stop_service_for(vcfg, model_path=model, served_name=model)
        except Exception as e:
            print(f"[ocr] Surya vLLM stop skipped: {e}")
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

    async def ocr_pages(self, images: List, engine: Optional[str] = None) -> (str, List[OcrPage]):
        """OCR a list of PIL page images. Returns (engine_name, [OcrPage])."""
        name = self.resolve_engine(engine)
        pool = await self._get_pool(name)
        sem = self._sem or asyncio.Semaphore(4)

        async def _one(idx, img):
            async with sem:
                page = await pool.recognize(img)
                page.index = idx
                return page

        pages = await asyncio.gather(*[_one(i, im) for i, im in enumerate(images)])
        return name, list(pages)

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
