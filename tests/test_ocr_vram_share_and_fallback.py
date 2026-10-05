"""Two fixes for the surya-on-vLLM 503 loop.

1. A VLM OCR engine's ``vlm_gpu_memory_utilization`` is the share it wants for ITSELF,
   but vLLM's flag is a fraction of the card's TOTAL memory and counts every other
   process' allocation against it. With 3.6 GB of resident bge-m3 on a 24 GB 3090, a
   0.35 share left surya-2 ~5 GB, which after weights + activation peak + CUDA graphs
   came out at "Available KV cache memory: -2.81 GiB" — and vLLM died on boot every 20
   minutes for hours. The share is now translated against what others hold, and the
   evict-first pass asks for enough that the translation fits under the ceiling.

2. A document that nobody could read is worse than a document read by the second-best
   engine, so a failing engine falls back to the others instead of returning 503.

No GPU, no vLLM, no models: memory figures and engines are stubbed.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codai.config import OcrConfig
from codai.ocr.base import OcrError, OcrPage
from codai.ocr import manager as ocr_manager_mod
from codai.api import vllm_worker


def _page(w=64, h=64):
    from PIL import Image
    return Image.new("RGB", (w, h), (255, 255, 255))


def _card(monkeypatch, total_gb=24.0, free_gb=24.0):
    """Pretend the box has one card of ``total_gb`` with ``free_gb`` free."""
    from codai.models.manager import multi_model_manager
    monkeypatch.setattr(multi_model_manager, "_total_vram_gb", lambda: total_gb)
    monkeypatch.setattr(multi_model_manager, "_get_free_vram_gb", lambda: free_gb)


# ---------------------------------------------------- the share translation
def test_an_empty_card_passes_the_share_through(monkeypatch):
    _card(monkeypatch, total_gb=24.0, free_gb=24.0)
    assert vllm_worker.effective_gmu(0.35) == pytest.approx(0.35, abs=1e-3)


def test_other_residents_do_not_inflate_the_share(monkeypatch):
    """The share is what THIS instance may use, full stop.

    It used to have other residents' usage added back, on the theory that vLLM charges
    them against the budget. Two live boots said otherwise (the same 11.1 GB footprint at
    two different budgets), and the add-back made the figure depend on unrelated models
    while asking for share*total for this instance on top of what they held. The
    evict-first pass clears the card instead."""
    _card(monkeypatch, total_gb=24.0, free_gb=24.0 - 6.1)
    assert vllm_worker.effective_gmu(0.60) == pytest.approx(0.60, abs=1e-3)


def test_the_share_is_capped_so_the_driver_keeps_its_room(monkeypatch):
    _card(monkeypatch, total_gb=24.0, free_gb=24.0)
    assert vllm_worker.effective_gmu(0.99) == pytest.approx(vllm_worker.GMU_CEILING)


def test_an_unmeasurable_card_still_passes_the_share_through(monkeypatch):
    from codai.models.manager import multi_model_manager
    monkeypatch.setattr(multi_model_manager, "_total_vram_gb", lambda: 24.0)
    monkeypatch.setattr(multi_model_manager, "_get_free_vram_gb", lambda: 999.0)
    assert vllm_worker.effective_gmu(0.60) == pytest.approx(0.60, abs=1e-3)


def test_no_share_means_no_flag(monkeypatch):
    _card(monkeypatch)
    assert vllm_worker.effective_gmu(0.0) == 0.0
    assert vllm_worker.effective_gmu(None) == 0.0


# ---------------------------------------------------- footprint caps
def test_the_vlm_engine_caps_its_own_vllms_batch(monkeypatch):
    """The activation peak these cap IS what kept surya-ocr-2 from starting: 9.4 GB of an
    11.1 GB footprint, profiled for the LLM backend's ctx=18432."""
    from codai.ocr.base import vlm_serve_limits
    assert vlm_serve_limits(OcrConfig()) == {
        "max_num_batched_tokens": 4096, "max_num_seqs": 16}   # max_model_len 0 = inherit


def test_zero_means_inherit_the_backends_value():
    from codai.ocr.base import vlm_serve_limits
    cfg = OcrConfig(vlm_max_num_batched_tokens=0, vlm_max_num_seqs=0, vlm_max_model_len=0)
    assert vlm_serve_limits(cfg) == {}


def test_an_explicit_model_len_is_passed_too():
    from codai.ocr.base import vlm_serve_limits
    cfg = OcrConfig(vlm_max_model_len=8192)
    assert vlm_serve_limits(cfg)["max_model_len"] == 8192


def test_the_caps_reach_the_launch_command(monkeypatch):
    _card(monkeypatch)
    cfg = type("C", (), {"model_id": "surya", "ctx": 18432, "max_num_seqs": 0,
                         "gpu_memory_utilization": 0.9})()
    cmd = vllm_worker._launch_cmd(
        "py", cfg, "127.0.0.1", 1, "datalab-to/surya-ocr-2", "datalab-to/surya-ocr-2",
        gpu_memory_utilization=0.60, max_num_batched_tokens=4096, max_num_seqs=16)
    assert "--max-num-batched-tokens" in cmd
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "4096"
    assert cmd[cmd.index("--max-num-seqs") + 1] == "16"
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.6"
    assert cmd[cmd.index("--max-model-len") + 1] == "18432"    # inherited


def test_an_override_wins_over_the_backends_context(monkeypatch):
    _card(monkeypatch)
    cfg = type("C", (), {"model_id": "surya", "ctx": 18432, "gpu_memory_utilization": 0.9})()
    cmd = vllm_worker._launch_cmd("py", cfg, "127.0.0.1", 1, "m", "m",
                                  max_model_len=8192)
    assert cmd[cmd.index("--max-model-len") + 1] == "8192"


def test_the_shipped_share_clears_the_measured_footprint():
    """0.35 of a 24 GB card is 8.4 GB against an 11.1 GB floor — below it, nothing starts.
    0.60 is 14.4 GB, which left 3.18 GB of KV cache on the live card."""
    share = OcrConfig().vlm_gpu_memory_utilization
    assert share == 0.60
    assert share * 24.0 > 11.1 + 2.0


def test_the_backends_own_setting_is_already_absolute(monkeypatch):
    """Only a side job's share is translated; vllm.gpu_memory_utilization is not."""
    _card(monkeypatch, total_gb=24.0, free_gb=24.0 - 8.0)
    cfg = type("C", (), {"gpu_memory_utilization": 0.90})()
    assert vllm_worker.resolve_gmu(cfg, None) == pytest.approx(0.90)
    assert vllm_worker.resolve_gmu(cfg, 0.60) == pytest.approx(0.60, abs=1e-3)


# ---------------------------------------------------- evict-first demand
def test_the_evict_demand_covers_the_share_plus_the_ceilings_headroom(monkeypatch):
    _card(monkeypatch, total_gb=24.0, free_gb=24.0)
    cfg = type("C", (), {"gpu_memory_utilization": 0.0})()
    need = vllm_worker.prelaunch_free_gb(cfg, 0.60)
    # The share itself plus the headroom the ceiling reserves.
    assert need == pytest.approx((0.60 + (1.0 - vllm_worker.GMU_CEILING)) * 24.0, abs=0.05)
    assert need > vllm_worker.planned_vram_gb(cfg, 0.60)


def test_a_crowded_card_is_cleared_rather_than_clamped(monkeypatch):
    """When the demand is met, the translation fits — that is the point of the pairing."""
    _card(monkeypatch, total_gb=24.0, free_gb=24.0)
    cfg = type("C", (), {"gpu_memory_utilization": 0.0})()
    need = vllm_worker.prelaunch_free_gb(cfg, 0.60)
    # Evictor honoured the demand exactly: `need` free, the rest held by others.
    _card(monkeypatch, total_gb=24.0, free_gb=need)
    assert need <= 24.0 and vllm_worker.effective_gmu(0.60) * 24.0 <= need


# ---------------------------------------------------- release accounting
def test_stopping_reports_what_was_reserved_not_the_engines_share(monkeypatch):
    """Under-reporting the release sends the evictor hunting for memory already free."""
    _card(monkeypatch, total_gb=24.0, free_gb=24.0 - 3.6)

    class _Proc:
        def poll(self):
            return None

    stopped = []
    monkeypatch.setattr(vllm_worker, "stop_service", lambda k: stopped.append(k))
    launched = vllm_worker.effective_gmu(0.60)          # 0.60 -> a 14.4 GB reservation
    monkeypatch.setitem(vllm_worker._services, "m|m",
                        {"proc": _Proc(), "port": 1, "url": "u", "ray": None,
                         "gmu": launched})
    cfg = type("C", (), {"model_path": "m", "gpu_memory_utilization": 0.0})()
    freed = vllm_worker.stop_service_for(cfg, model_path="m", served_name="m",
                                         gpu_memory_utilization=0.10)
    assert stopped == ["m|m"]
    assert freed == pytest.approx(launched * 24.0, abs=0.05)     # ~14.4 GB, what it took
    assert freed > 0.10 * 24.0                      # not the (stale) share the caller passed


def test_a_leftover_service_is_stopped_even_after_the_mode_changed(monkeypatch):
    """Gating the stop on the CURRENT serve mode leaked the card: flip surya_serve away
    from "vllm" while its service is up and nothing would ever reclaim its share."""
    import types
    import codai.api
    stopped = []
    fake = types.SimpleNamespace(
        stop_service_for=lambda vcfg, model_path=None, served_name=None,
        gpu_memory_utilization=None: (stopped.append(model_path), 8.4)[1],
        planned_vram_gb=lambda vcfg, share=None: 8.4,
        prelaunch_free_gb=lambda vcfg, share=None: 0.0,
        ensure_service=lambda *a, **k: "http://127.0.0.1:1/",
    )
    monkeypatch.setitem(sys.modules, "codai.api.vllm_worker", fake)
    monkeypatch.setattr(codai.api, "vllm_worker", fake, raising=False)
    monkeypatch.setattr("codai.models.manager.get_active_vllm_config",
                        lambda: types.SimpleNamespace(gpu_memory_utilization=0.9))
    monkeypatch.setattr(
        "codai.models.manager.multi_model_manager.register_external_vram_releaser",
        lambda fn: None)

    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg(surya_serve="local", olmocr_serve="model"))
    assert m._release_vram(999.0) >= 16.0
    assert "datalab-to/surya-ocr-2" in stopped


def test_nothing_running_frees_nothing(monkeypatch):
    cfg = type("C", (), {"model_path": "gone", "gpu_memory_utilization": 0.0})()
    assert vllm_worker.stop_service_for(cfg, model_path="gone") == 0.0


# ---------------------------------------------------- engine fallback
def _cfg(**kw):
    c = OcrConfig(enabled=True, default_engine="surya",
                  surya_enabled=True, surya_accept_license=True,
                  paddle_enabled=True, doctr_enabled=True)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _stub_engines(monkeypatch, broken=("surya",), log=None):
    """Register engine factories that work, except the named ones which refuse to load."""
    def _make(name):
        class _Eng:
            def __init__(self, cfg):
                pass

            def prelaunch_vram_gb(self):
                return 0.0

            def vram_gb(self):
                return 1.0

            def ensure_loaded(self):
                if log is not None:
                    log.append(f"load:{name}")
                if name in broken:
                    raise OcrError(f"{name}: vLLM would not start", status=503)

            def recognize_image(self, image):
                return OcrPage(index=0, text=f"read by {name}")

            def cleanup(self):
                pass
        return _Eng

    for n in ("paddle", "doctr", "surya", "olmocr"):
        monkeypatch.setitem(ocr_manager_mod._ENGINE_FACTORIES, n, _make(n))
    monkeypatch.setattr(ocr_manager_mod, "_evict_for_ocr", lambda gb: None)
    monkeypatch.setattr(ocr_manager_mod, "_wait_thermal_safe", lambda: None)


def test_a_dead_engine_falls_back_instead_of_503(monkeypatch):
    """The whole complaint: an OCR request must not come back 503."""
    _stub_engines(monkeypatch, broken=("surya",))
    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg())
    name, pages = asyncio.run(m.ocr_pages([_page()], engine="surya"))
    assert name == "paddle"                       # auto order, fewest ways to fail first
    assert pages[0].text == "read by paddle"


def test_the_fallback_keeps_going_until_one_engine_answers(monkeypatch):
    _stub_engines(monkeypatch, broken=("surya", "paddle", "doctr"))
    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg(olmocr_enabled=True))
    name, pages = asyncio.run(m.ocr_pages([_page()], engine="surya"))
    assert name == "olmocr"
    assert pages[0].text == "read by olmocr"


def test_a_disabled_engine_is_never_a_fallback(monkeypatch):
    _stub_engines(monkeypatch, broken=("surya", "paddle"))
    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg(doctr_enabled=False, olmocr_enabled=False))
    with pytest.raises(OcrError):
        asyncio.run(m.ocr_pages([_page()], engine="surya"))


def test_an_explicit_order_is_honoured(monkeypatch):
    _stub_engines(monkeypatch, broken=("surya",))
    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg(fallback_engines="doctr, paddle"))
    name, _pages = asyncio.run(m.ocr_pages([_page()], engine="surya"))
    assert name == "doctr"


def test_fallback_can_be_turned_off_and_then_the_error_is_the_originals(monkeypatch):
    _stub_engines(monkeypatch, broken=("surya",))
    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg(fallback_engines="off"))
    with pytest.raises(OcrError) as e:
        asyncio.run(m.ocr_pages([_page()], engine="surya"))
    assert "vLLM would not start" in str(e.value)


def test_when_everything_is_down_the_first_failure_is_what_surfaces(monkeypatch):
    _stub_engines(monkeypatch, broken=("surya", "paddle", "doctr", "olmocr"))
    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg(olmocr_enabled=True))
    with pytest.raises(OcrError) as e:
        asyncio.run(m.ocr_pages([_page()], engine="surya"))
    assert "surya" in str(e.value)


def test_a_healthy_engine_is_not_second_guessed(monkeypatch):
    log = []
    _stub_engines(monkeypatch, broken=(), log=log)
    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg())
    name, _pages = asyncio.run(m.ocr_pages([_page()], engine="surya"))
    assert name == "surya"
    assert log == ["load:surya"]        # no other engine was even built


def test_the_fallback_is_on_by_default():
    assert OcrConfig().fallback_engines == "auto"
    assert OcrConfig().evict_drain_timeout_s == 60.0


# ---------------------------------------------------- taking turns on the card
def test_eviction_waits_for_an_in_flight_page_instead_of_killing_it(monkeypatch):
    """A model load must not take OCR's VRAM out from under a page being recognised —
    the handover happens at a page boundary, like the model manager's own clean swap."""
    import threading
    started = threading.Event()
    finish = threading.Event()
    torn = []

    class _Slow:
        def __init__(self, cfg):
            pass

        def prelaunch_vram_gb(self):
            return 0.0

        def vram_gb(self):
            return 8.4

        def ensure_loaded(self):
            pass

        def recognize_image(self, image):
            started.set()
            finish.wait(5.0)            # a page mid-recognition
            return OcrPage(index=0, text="finished anyway")

        def cleanup(self):
            torn.append(1)

    monkeypatch.setitem(ocr_manager_mod._ENGINE_FACTORIES, "paddle", _Slow)
    monkeypatch.setattr(ocr_manager_mod, "_evict_for_ocr", lambda gb: None)
    monkeypatch.setattr(
        "codai.models.manager.multi_model_manager.register_external_vram_releaser",
        lambda fn: None)

    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg(default_engine="paddle", evict_drain_timeout_s=5.0))

    async def _go():
        task = asyncio.ensure_future(m.ocr_pages([_page()], engine="paddle"))
        await asyncio.to_thread(started.wait, 5.0)
        # The model manager's eviction thread arrives mid-page.
        releaser = asyncio.ensure_future(asyncio.to_thread(m._release_vram, 999.0))
        await asyncio.sleep(0.2)
        assert not releaser.done()          # it is WAITING, not tearing down
        assert not torn                     # nothing released yet
        finish.set()                        # the page completes
        freed = await releaser
        name, pages = await task
        return freed, name, pages

    freed, name, pages = asyncio.run(_go())
    assert pages[0].text == "finished anyway"   # the page was NOT lost
    assert name == "paddle"                     # and no fallback was needed
    assert torn and freed > 0                   # VRAM released after the drain


def test_the_drain_has_a_budget_so_a_wedged_engine_cannot_block_a_load(monkeypatch):
    import threading
    started = threading.Event()
    stuck = threading.Event()

    class _Wedged:
        def __init__(self, cfg):
            pass

        def prelaunch_vram_gb(self):
            return 0.0

        def vram_gb(self):
            return 8.4

        def ensure_loaded(self):
            pass

        def recognize_image(self, image):
            started.set()
            stuck.wait(30.0)
            return OcrPage(index=0, text="late")

        def cleanup(self):
            pass

    monkeypatch.setitem(ocr_manager_mod._ENGINE_FACTORIES, "paddle", _Wedged)
    monkeypatch.setattr(ocr_manager_mod, "_evict_for_ocr", lambda gb: None)
    monkeypatch.setattr(
        "codai.models.manager.multi_model_manager.register_external_vram_releaser",
        lambda fn: None)

    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg(default_engine="paddle", evict_drain_timeout_s=0.3))

    async def _go():
        task = asyncio.ensure_future(m.ocr_pages([_page()], engine="paddle"))
        await asyncio.to_thread(started.wait, 5.0)
        freed = await asyncio.to_thread(m._release_vram, 999.0)
        stuck.set()
        try:
            await task
        except Exception:
            pass
        return freed

    assert asyncio.run(_go()) > 0        # gave up on the wedged page and freed the card


def test_ocr_rebuilds_and_takes_its_turn_again_after_being_evicted(monkeypatch):
    """Evicted is not dead: the next request rebuilds, so OCR and models can alternate
    as many times as the traffic asks for."""
    builds = []

    class _Eng:
        def __init__(self, cfg):
            pass

        def prelaunch_vram_gb(self):
            return 0.0

        def vram_gb(self):
            return 8.4

        def ensure_loaded(self):
            builds.append(1)

        def recognize_image(self, image):
            return OcrPage(index=0, text="ok")

        def cleanup(self):
            pass

    monkeypatch.setitem(ocr_manager_mod._ENGINE_FACTORIES, "paddle", _Eng)
    evicted_for = []
    monkeypatch.setattr(ocr_manager_mod, "_evict_for_ocr",
                        lambda gb: evicted_for.append(gb))
    monkeypatch.setattr(
        "codai.models.manager.multi_model_manager.register_external_vram_releaser",
        lambda fn: None)

    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg(default_engine="paddle", paddle_instances=1))

    for _turn in range(3):
        name, pages = asyncio.run(m.ocr_pages([_page()], engine="paddle"))
        assert (name, pages[0].text) == ("paddle", "ok")
        assert m._release_vram(999.0) > 0        # a model takes the card back

    assert len(builds) == 3          # rebuilt for every turn, never permanently dead
    assert len(evicted_for) == 3     # and asked for room each time, like any model
