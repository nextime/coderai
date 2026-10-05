"""olmOCR-2 as an alternative OCR engine, plus the two things the Surya crash loop
exposed in the pool: evict BEFORE a server-backed engine loads, and a cooldown after a
failed build instead of re-attempting it on every request.

No model, no GPU, no network: the engine's HTTP/model call is stubbed, images are tiny
PIL pages.
"""

import asyncio
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codai.config import OcrConfig
from codai.ocr.base import OcrError, OcrPage
from codai.ocr.olmocr import (
    DEFAULT_LONGEST_SIDE, OLMOCR_PROMPT, OlmOcrEngine, parse_front_matter, prepare_image,
)
from codai.ocr import manager as ocr_manager_mod


def _page(w=900, h=1200, color=(255, 255, 255)):
    from PIL import Image
    return Image.new("RGB", (w, h), color)


def _cfg(**kw):
    c = OcrConfig(enabled=True, olmocr_enabled=True, default_engine="olmocr")
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _patch_vllm_worker(monkeypatch, fake):
    """Swap in a fake vLLM worker for code that does ``from codai.api import vllm_worker``.

    Patching sys.modules alone is not enough: once the real module has been imported by
    anything (another test file, for instance), it is bound as an attribute of the
    ``codai.api`` package and the from-import takes that attribute without consulting
    sys.modules. Patch both, or the fake silently does not apply."""
    import codai.api
    monkeypatch.setitem(sys.modules, "codai.api.vllm_worker", fake)
    monkeypatch.setattr(codai.api, "vllm_worker", fake, raising=False)


# ------------------------------------------------------------ front matter
def test_front_matter_split():
    meta, body = parse_front_matter(
        "---\n"
        "primary_language: it\n"
        "is_rotation_valid: True\n"
        "rotation_correction: 0\n"
        "is_table: False\n"
        "is_diagram: False\n"
        "---\n"
        "TRIBUNALE DI TORINO\n\nSentenza n. 42\n"
    )
    assert meta == {
        "primary_language": "it", "is_rotation_valid": True,
        "rotation_correction": 0, "is_table": False, "is_diagram": False,
    }
    assert body.startswith("TRIBUNALE DI TORINO")
    assert "---" not in body


def test_front_matter_in_a_code_fence():
    meta, body = parse_front_matter(
        "```markdown\n---\nprimary_language: en\nis_table: True\n---\n<table></table>\n```")
    assert meta["primary_language"] == "en" and meta["is_table"] is True
    assert body == "<table></table>"


def test_front_matter_unfenced_and_absent():
    meta, body = parse_front_matter("primary_language: en\nis_table: False\n\nHello page")
    assert meta["primary_language"] == "en" and body == "Hello page"
    # No front matter at all: every byte stays in the body rather than being eaten.
    meta, body = parse_front_matter("Just the page text, no metadata")
    assert meta == {} and body == "Just the page text, no metadata"


# ---------------------------------------------------------------- image prep
def test_prepare_image_scales_down_to_1288_and_never_up():
    big = prepare_image(_page(4000, 2000))
    assert big.startswith("data:image/png;base64,")
    from PIL import Image
    import base64, io
    img = Image.open(io.BytesIO(base64.b64decode(big.split(",", 1)[1])))
    assert max(img.size) == DEFAULT_LONGEST_SIDE
    assert img.size == (1288, 644)

    small = prepare_image(_page(300, 200))
    img = Image.open(io.BytesIO(base64.b64decode(small.split(",", 1)[1])))
    assert img.size == (300, 200)


# ------------------------------------------------------------------- engine
class _Stub:
    """Records the requests the engine makes and replies with canned pages."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, max_tokens, temperature):
        self.calls.append({"messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature})
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def _engine(cfg, stub):
    eng = OlmOcrEngine(cfg)
    eng._ask_local = stub
    eng._ask_http = stub
    eng.ensure_loaded()
    return eng


def test_recognize_uses_the_trained_prompt_and_returns_text_plus_meta():
    stub = _Stub("---\nprimary_language: it\nis_table: True\n---\n# Atto\n\nTesto.")
    eng = _engine(_cfg(olmocr_model_id="olmocr-2"), stub)
    page = eng.recognize_image(_page(800, 1000))

    msg = stub.calls[0]["messages"][0]
    assert msg["role"] == "user"
    assert msg["content"][0]["text"] == OLMOCR_PROMPT        # verbatim, as trained
    assert msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    assert page.text == "# Atto\n\nTesto."
    assert page.meta["primary_language"] == "it" and page.meta["is_table"] is True
    assert page.width == 800 and page.height == 1000
    # A VLM has no detector: no boxes, and the response omits an empty meta for the
    # engines that report none.
    assert page.lines == [] and page.regions == []
    assert "meta" in page.to_dict()
    assert "meta" not in OcrPage(index=0, text="x").to_dict()


def test_rotated_page_is_asked_once_more_turned():
    stub = _Stub(
        "---\nis_rotation_valid: False\nrotation_correction: 90\n---\ngibberish",
        "---\nis_rotation_valid: True\nrotation_correction: 0\n---\nthe real text",
    )
    eng = _engine(_cfg(olmocr_model_id="olmocr-2"), stub)
    page = eng.recognize_image(_page(900, 1200))
    assert len(stub.calls) == 2
    assert page.text == "the real text"
    assert page.meta["rotation_applied"] == 90
    # The retry ran on the turned page, so the reported size is the turned one.
    assert (page.width, page.height) == (1200, 900)


def test_rotation_retry_can_be_turned_off():
    stub = _Stub("---\nis_rotation_valid: False\nrotation_correction: 90\n---\ngibberish")
    eng = _engine(_cfg(olmocr_model_id="olmocr-2", olmocr_retry_rotation=False), stub)
    page = eng.recognize_image(_page())
    assert len(stub.calls) == 1 and page.text == "gibberish"


def test_each_mode_refuses_to_load_without_what_it_needs():
    with pytest.raises(OcrError) as e:
        OlmOcrEngine(_cfg(olmocr_serve="model", olmocr_model_id="")).load()
    assert "olmocr_model_id" in str(e.value) and e.value.status == 400

    with pytest.raises(OcrError) as e:
        OlmOcrEngine(_cfg(olmocr_serve="server", olmocr_server_url="")).load()
    assert "olmocr_server_url" in str(e.value)

    with pytest.raises(OcrError) as e:
        OlmOcrEngine(_cfg(olmocr_serve="nope")).load()
    assert "olmocr_serve" in str(e.value)


def test_server_mode_normalises_the_base_url():
    eng = OlmOcrEngine(_cfg(olmocr_serve="server", olmocr_server_url="http://box:8000/"))
    eng.load()
    assert eng._base == "http://box:8000/v1"
    eng2 = OlmOcrEngine(_cfg(olmocr_serve="server", olmocr_server_url="http://box:8000/v1"))
    eng2.load()
    assert eng2._base == "http://box:8000/v1"


def test_http_failures_map_to_useful_statuses(monkeypatch):
    cfg = _cfg(olmocr_serve="server", olmocr_server_url="http://box:8000/v1")
    eng = OlmOcrEngine(cfg)
    eng.load()

    fake = types.SimpleNamespace()

    class _Resp:
        status_code = 500
        text = "boom"

        def json(self):
            return {}

    fake.post = lambda *a, **k: _Resp()
    monkeypatch.setitem(sys.modules, "requests", fake)
    with pytest.raises(OcrError) as e:
        eng.recognize_image(_page(64, 64))
    assert e.value.status == 502

    def _raise(*a, **k):
        raise OSError("connection refused")

    fake.post = _raise
    with pytest.raises(OcrError) as e:
        eng.recognize_image(_page(64, 64))
    assert e.value.status == 503


def test_only_the_vllm_mode_asks_for_room_up_front(monkeypatch):
    monkeypatch.setattr(OlmOcrEngine, "_vllm_need", staticmethod(lambda: 14.1))
    assert OlmOcrEngine(_cfg(olmocr_serve="model", olmocr_model_id="x")).prelaunch_vram_gb() == 0.0
    assert OlmOcrEngine(_cfg(olmocr_serve="server", olmocr_server_url="u")).prelaunch_vram_gb() == 0.0
    assert OlmOcrEngine(_cfg(olmocr_serve="vllm")).prelaunch_vram_gb() == 14.1


# --------------------------------------------------------------- the manager
def test_the_engine_is_registered_and_gated_by_its_own_flag():
    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg())
    assert m.resolve_engine(None) == "olmocr"      # it is this config's default
    assert m.resolve_engine("olmocr") == "olmocr"

    m.configure(_cfg(olmocr_enabled=False, default_engine="paddle"))
    with pytest.raises(OcrError) as e:
        m.resolve_engine("olmocr")
    assert "not enabled" in str(e.value)


def test_the_pool_frees_vram_before_a_server_backed_engine_loads(monkeypatch):
    """The Surya-2 bug: eviction ran AFTER the load, and the load was what failed."""
    order = []

    class _Eng:
        def __init__(self, cfg):
            pass

        def prelaunch_vram_gb(self):
            return 14.1

        def vram_gb(self):
            return 14.1

        def ensure_loaded(self):
            order.append("load")

        def recognize_image(self, image):
            return OcrPage(index=0, text="ok")

        def cleanup(self):
            pass

    monkeypatch.setitem(ocr_manager_mod._ENGINE_FACTORIES, "olmocr", _Eng)
    monkeypatch.setattr(ocr_manager_mod, "_evict_for_ocr",
                        lambda gb: order.append(f"evict:{gb:.1f}"))

    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg(olmocr_instances=1))

    async def _go():
        _name, pages = await m.ocr_pages([_page(64, 64)], engine="olmocr")
        return pages

    pages = asyncio.run(_go())
    assert pages[0].text == "ok"
    assert order[0] == "evict:14.1" and order[1] == "load"


def test_a_failed_build_fails_fast_for_the_cooldown_then_tries_again(monkeypatch):
    attempts = []

    class _Broken:
        def __init__(self, cfg):
            pass

        def prelaunch_vram_gb(self):
            return 0.0

        def vram_gb(self):
            return 0.0

        def ensure_loaded(self):
            attempts.append(time.time())
            raise OcrError("vLLM exited before becoming ready: 11.3 GB free, 14.1 wanted",
                           status=503)

        def cleanup(self):
            pass

    monkeypatch.setitem(ocr_manager_mod._ENGINE_FACTORIES, "olmocr", _Broken)
    monkeypatch.setattr(ocr_manager_mod, "_evict_for_ocr", lambda gb: None)

    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg(build_retry_cooldown_s=30.0))

    async def _one():
        return await m.ocr_pages([_page(64, 64)], engine="olmocr")

    for _ in range(3):
        with pytest.raises(OcrError) as e:
            asyncio.run(_one())
    # Only the FIRST request paid for a boot attempt; the other two were refused
    # immediately, with the reason and the time left.
    assert len(attempts) == 1
    assert "11.3 GB free" in str(e.value) and "cooldown" in str(e.value)
    assert e.value.status == 503

    # Once the cooldown is up the engine gets another chance.
    m2 = ocr_manager_mod.OcrManager()
    m2.configure(_cfg(build_retry_cooldown_s=0.0))
    for _ in range(2):
        with pytest.raises(OcrError):
            asyncio.run(m2.ocr_pages([_page(64, 64)], engine="olmocr"))
    assert len(attempts) == 3


def test_vllm_mode_serves_with_the_ocr_side_gpu_share(monkeypatch):
    """The OCR VLM gets its own share of the card, not the one sized for an LLM."""
    seen = {}

    from codai.api.vllm_worker import GMU_CEILING
    _headroom = 1.0 - GMU_CEILING      # the ceiling keeps room for the driver/fragmentation

    fake_worker = types.SimpleNamespace(
        ensure_service=lambda vcfg, model_path=None, served_name=None,
        gpu_memory_utilization=None, **kw: (
            seen.update(model=model_path, share=gpu_memory_utilization, limits=kw),
            "http://127.0.0.1:1/")[1],
        planned_vram_gb=lambda vcfg, share=None: (share or 0.9) * 24.0,
        prelaunch_free_gb=lambda vcfg, share=None: ((share or 0.9) + _headroom) * 24.0,
    )
    _patch_vllm_worker(monkeypatch, fake_worker)
    monkeypatch.setattr("codai.models.manager.get_active_vllm_config",
                        lambda: types.SimpleNamespace(gpu_memory_utilization=0.9))

    cfg = _cfg(olmocr_serve="vllm", vlm_gpu_memory_utilization=0.60)
    eng = OlmOcrEngine(cfg)
    # 0.60 of the card for itself (not the LLM's 0.9) PLUS the ceiling's headroom, which
    # is what must be FREE before launching — vLLM reserves it up front.
    assert eng.prelaunch_vram_gb() == pytest.approx((0.60 + _headroom) * 24.0)
    eng.load()
    assert seen["model"] == "allenai/olmOCR-2-7B-1025-FP8"
    assert seen["share"] == 0.60
    # …and it caps its own instance's batch, which is what sets the activation peak.
    assert seen["limits"]["max_num_batched_tokens"] == 4096
    assert seen["limits"]["max_num_seqs"] == 16
    assert eng._base == "http://127.0.0.1:1/v1"

    # 0.60 is the shipped default: measured live, surya-ocr-2 needs ~11.1 GB before a
    # single KV block, so the 0.35 this once shipped (8.4 GB of a 24 GB card) was under
    # the floor and could not start at all.
    assert OcrConfig().vlm_gpu_memory_utilization == 0.60
    # Set back to 0 it defers to whatever the vLLM backend itself is configured with.
    eng2 = OlmOcrEngine(_cfg(olmocr_serve="vllm", vlm_gpu_memory_utilization=0.0))
    assert eng2.prelaunch_vram_gb() == pytest.approx((0.9 + _headroom) * 24.0)


def test_a_shared_server_is_reserved_for_ONCE_not_per_instance(monkeypatch):
    """surya_instances was 24 in production: multiplying the shared vLLM's demand by the
    pool size would have asked the manager to free 200 GB of a 24 GB card."""
    asked = []

    class _Shared:
        def __init__(self, cfg):
            pass

        def prelaunch_vram_gb(self):
            return 8.4            # the whole pool's share of the card, not each instance's

        def vram_gb(self):
            return 0.05           # a worker that only speaks HTTP to that server

        def ensure_loaded(self):
            pass

        def recognize_image(self, image):
            return OcrPage(index=0, text="ok")

        def cleanup(self):
            pass

    monkeypatch.setitem(ocr_manager_mod._ENGINE_FACTORIES, "olmocr", _Shared)
    monkeypatch.setattr(ocr_manager_mod, "_evict_for_ocr", lambda gb: asked.append(gb))

    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg(olmocr_instances=24))
    asyncio.run(m.ocr_pages([_page(64, 64)], engine="olmocr"))
    # Once, for the server's share — and NOT again after the load, for memory that
    # server has already taken.
    assert asked == [8.4]


def test_an_engine_without_a_server_is_still_evicted_for_per_instance(monkeypatch):
    """The paddle/docTR path is unchanged: nothing is reserved up front, and the real
    footprint is evicted for once it is known, times the pool size."""
    asked = []

    class _Local:
        def __init__(self, cfg):
            pass

        def prelaunch_vram_gb(self):
            return 0.0

        def vram_gb(self):
            return 0.7

        def ensure_loaded(self):
            asked.append("load")

        def recognize_image(self, image):
            return OcrPage(index=0, text="ok")

        def cleanup(self):
            pass

    monkeypatch.setitem(ocr_manager_mod._ENGINE_FACTORIES, "doctr", _Local)
    monkeypatch.setattr(ocr_manager_mod, "_evict_for_ocr",
                        lambda gb: asked.append(round(gb, 2)))

    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg(doctr_enabled=True, doctr_instances=3, default_engine="doctr"))
    asyncio.run(m.ocr_pages([_page(64, 64)], engine="doctr"))
    assert asked == ["load", 2.1, "load", "load"]


def test_the_model_manager_can_evict_ocr_including_both_vlm_servers(monkeypatch):
    """The other direction: loading a model asks OCR to give its VRAM back. Pools are torn
    down AND the managed vLLM instance is stopped — tearing down the pool alone leaves
    gpu_memory_utilization × the card held by a subprocess the manager cannot see."""
    registered = []
    monkeypatch.setattr(
        "codai.models.manager.multi_model_manager.register_external_vram_releaser",
        lambda fn: registered.append(fn))

    stopped = []
    fake_worker = types.SimpleNamespace(
        stop_service_for=lambda vcfg, model_path=None, served_name=None,
        gpu_memory_utilization=None: (stopped.append((model_path, gpu_memory_utilization)), 8.4)[1],
        planned_vram_gb=lambda vcfg, share=None: 8.4,
        prelaunch_free_gb=lambda vcfg, share=None: 0.0,
        ensure_service=lambda *a, **k: "http://127.0.0.1:1/",
    )
    _patch_vllm_worker(monkeypatch, fake_worker)
    monkeypatch.setattr("codai.models.manager.get_active_vllm_config",
                        lambda: types.SimpleNamespace(gpu_memory_utilization=0.9))

    released = []

    class _Eng:
        def __init__(self, cfg):
            pass

        def prelaunch_vram_gb(self):
            return 0.0

        def vram_gb(self):
            return 0.05

        def ensure_loaded(self):
            pass

        def recognize_image(self, image):
            return OcrPage(index=0, text="ok")

        def cleanup(self):
            released.append(1)

    monkeypatch.setitem(ocr_manager_mod._ENGINE_FACTORIES, "olmocr", _Eng)
    monkeypatch.setattr(ocr_manager_mod, "_evict_for_ocr", lambda gb: None)

    m = ocr_manager_mod.OcrManager()
    m.configure(_cfg(olmocr_serve="vllm", olmocr_instances=2,
                     surya_enabled=True, surya_accept_license=True, surya_serve="vllm"))
    assert m._release_vram in registered        # the manager now knows how to reclaim OCR
    asyncio.run(m.ocr_pages([_page(64, 64)], engine="olmocr"))

    freed = m._release_vram(999.0)
    assert len(released) == 2                  # both instances torn down
    assert freed >= 16.0                       # pool + BOTH vLLM services
    assert ("datalab-to/surya-ocr-2", 0.60) in stopped
    assert ("allenai/olmOCR-2-7B-1025-FP8", 0.60) in stopped
    # And the pool rebuilds on the next request rather than staying dead.
    asyncio.run(m.ocr_pages([_page(64, 64)], engine="olmocr"))
    assert len(released) == 2
