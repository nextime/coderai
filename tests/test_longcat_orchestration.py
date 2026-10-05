"""Step 3: LongCat as a citizen of the VRAM and swap machinery.

The operator's requirement is that a newly supported model be "correctly orchestrated like
any other, with model swap, memory management, scalability if configured". Three gaps had
to be closed for that to be true of an isolated-venv video model:

1. the VRAM estimator had no idea what a `longcat:` key costs — it would have fallen
   through to scanning a ~83 GB checkpoint directory;
2. nothing measured the real footprint, and no official figure exists;
3. the manager's busy signal is permanently False for this model (acquire_stt_backend pops
   the model pool), so eviction would tear a multi-minute generation down mid-flight;
   and video bypassed the front queue entirely, so `max_instances` did nothing.
"""

import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import longcat_worker as W
from codai.config import LongcatConfig
from codai.frontproxy.app import FrontProxy
from codai.models.manager import multi_model_manager as MM


@pytest.fixture(autouse=True)
def _clean():
    """Each test starts with no services and no measurements of its own."""
    W._services.clear()
    W._inflight.clear()
    W._idle.set()
    for k in [k for k in MM._measured_vram_gb if str(k).startswith("longcat:")]:
        MM._measured_vram_gb.pop(k, None)
    yield
    W._services.clear()
    W._inflight.clear()
    W._idle.set()


# ---------------------------------------------------------------- the estimate
def test_an_unmeasured_model_reserves_the_starting_estimate():
    assert MM._get_model_used_vram_gb("longcat:/w/lc#bf16") == W.DEFAULT_VRAM_GB


def test_a_measurement_wins_over_the_estimate():
    MM._measured_vram_gb["longcat:/w/lc#fp8"] = 13.7
    assert MM._get_model_used_vram_gb("longcat:/w/lc#fp8") == 13.7


def test_the_checkpoint_on_disk_is_never_used_as_the_footprint(tmp_path):
    """The repo is ~83 GB across variants; reporting that would ask the manager to evict
    the whole box on every request."""
    big = tmp_path / "dit"
    big.mkdir()
    (big / "model.safetensors").write_bytes(b"0" * (4 * 1024 * 1024))
    est = MM._get_model_used_vram_gb(f"longcat:{tmp_path}#bf16")
    assert est == W.DEFAULT_VRAM_GB          # not derived from the files


def test_the_estimate_is_only_a_starting_point():
    """No official VRAM figure exists for LongCat and it moves with variant, stage and
    offload, so the constant must not pretend to be a measurement."""
    assert W.DEFAULT_VRAM_GB > 0
    src = (ROOT / "codai/api/longcat_worker.py").read_text()
    assert "STARTING POINT" in src or "starting estimate" in src


# ---------------------------------------------------------------- measurement
def test_a_reported_peak_is_recorded():
    W._record_measurement("longcat:/w/lc#bf16", 31.4)
    assert MM._measured_vram_gb["longcat:/w/lc#bf16"] == 31.4


def test_the_high_water_mark_is_kept():
    """A draft run touches far less of the card than a 720p refinement; reserving the
    smaller figure would under-evict for the bigger one later."""
    W._record_measurement("longcat:/w/lc#bf16", 14.0)
    W._record_measurement("longcat:/w/lc#bf16", 31.4)
    W._record_measurement("longcat:/w/lc#bf16", 12.0)
    assert MM._measured_vram_gb["longcat:/w/lc#bf16"] == 31.4


def test_a_zero_measurement_is_ignored():
    """A CPU-only or failed run reports 0.0; recording it would disable eviction."""
    W._record_measurement("longcat:/w/lc#bf16", 0.0)
    assert "longcat:/w/lc#bf16" not in MM._measured_vram_gb


def test_measurements_are_per_config_not_per_path():
    """Sibling configs are different residencies — an fp8 entry must not inherit a bf16
    measurement."""
    W._record_measurement("longcat:/w/lc#bf16", 31.4)
    assert MM._get_model_used_vram_gb("longcat:/w/lc#fp8") == W.DEFAULT_VRAM_GB


def test_the_service_reports_reserved_not_allocated():
    """The caching allocator's arenas are what the card cannot give to anything else."""
    svc = (ROOT / "tools/longcat_service.py").read_text()
    assert "max_memory_reserved" in svc


# ---------------------------------------------------------------- the releaser
def test_the_releaser_matches_the_manager_contract():
    """fn(needed_gb) -> float, safe with nothing to free. voice_clone's takes no argument
    and raises TypeError inside eviction, which is why it is never reclaimed."""
    import inspect
    sig = inspect.signature(W.release_vram)
    assert len(sig.parameters) == 1
    assert W.release_vram(999.0) == 0.0          # nothing running


def test_it_is_registered_with_the_manager(monkeypatch):
    registered = []
    monkeypatch.setattr(MM, "register_external_vram_releaser",
                        lambda fn: registered.append(fn))
    W._releaser_registered = False
    W.register_releaser()
    assert W.release_vram in registered
    W.register_releaser()                        # idempotent
    assert len(registered) == 1


def test_registration_is_both_directions(monkeypatch):
    """acquire() evicts others for us; the releaser lets others evict us. A model that
    only does the first is not orchestrated like any other."""
    src = (ROOT / "codai/api/longcat_worker.py").read_text()
    assert "register_releaser()" in src.split("def acquire(")[1][:800]


def test_an_idle_service_is_stopped_at_once(monkeypatch):
    stopped = []
    monkeypatch.setattr(W, "stop_service", lambda k: stopped.append(k))
    W._services["longcat:/w/lc#bf16"] = {"proc": None, "port": 1, "url": "u", "meta": {}}
    MM._measured_vram_gb["longcat:/w/lc#bf16"] = 20.0
    freed = W.release_vram(5.0)
    assert stopped == ["longcat:/w/lc#bf16"]
    assert freed == 20.0                         # the measured figure, not a guess


def test_an_in_flight_generation_is_allowed_to_finish(monkeypatch):
    """The manager cannot see this work: acquire_stt_backend pops the model pool, so
    _is_key_busy is permanently False and Pass 1 of release_idle_vram would tear the
    subprocess down mid-generation and lose the whole request."""
    stopped = []
    monkeypatch.setattr(W, "stop_service", lambda k: stopped.append(k))
    monkeypatch.setattr(W, "_drain_timeout_s", lambda: 5.0)
    key = "longcat:/w/lc#bf16"
    W._services[key] = {"proc": None, "port": 1, "url": "u", "meta": {}}
    W._inflight[key] = 1
    W._idle.clear()

    done = threading.Event()

    def _release():
        W.release_vram(999.0)
        done.set()

    t = threading.Thread(target=_release, daemon=True)
    t.start()
    assert not done.wait(0.5)                    # waiting, not tearing down
    assert stopped == []
    # the generation finishes
    W._inflight[key] = 0
    W._idle.set()
    assert done.wait(5.0)
    assert stopped == [key]


def test_the_drain_has_a_budget(monkeypatch):
    """A wedged generation must not hold the card for ever."""
    stopped = []
    monkeypatch.setattr(W, "stop_service", lambda k: stopped.append(k))
    monkeypatch.setattr(W, "_drain_timeout_s", lambda: 0.3)
    key = "longcat:/w/lc#bf16"
    W._services[key] = {"proc": None, "port": 1, "url": "u", "meta": {}}
    W._inflight[key] = 1
    W._idle.clear()
    assert W.release_vram(999.0) > 0
    assert stopped == [key]


def test_the_drain_budget_is_configurable():
    assert LongcatConfig().evict_drain_timeout_s == 300.0


# ---------------------------------------------------------------- front queue
def _front(info=None):
    f = types.SimpleNamespace()
    f._task_kind = FrontProxy._task_kind
    f._model_info = lambda m: (info or {})
    f._queues_at_front = lambda p, m: FrontProxy._queues_at_front(f, p, m)
    return f


def test_text_still_always_queues():
    assert _front()._queues_at_front("/v1/chat/completions", "lisa")


def test_video_queues_only_when_max_instances_is_set():
    """Queueing video unconditionally would change behaviour for every existing video
    model and start returning 503 at queue_max_size. Setting the field IS the ask."""
    assert not _front()._queues_at_front("/v1/video/generations", "lc")
    assert _front({"max_instances": 2})._queues_at_front("/v1/video/generations", "lc")


def test_other_kinds_still_pass_through_unqueued():
    f = _front({"max_instances": 2})
    assert not f._queues_at_front("/v1/images/generations", "sd")
    assert not f._queues_at_front("/v1/embeddings", "bge")
    assert not f._queues_at_front("/v1/audio/speech", "tts")


def test_a_broken_model_lookup_does_not_block_the_request():
    f = types.SimpleNamespace()
    f._task_kind = FrontProxy._task_kind
    f._model_info = lambda m: (_ for _ in ()).throw(RuntimeError("boom"))
    assert FrontProxy._queues_at_front(f, "/v1/video/generations", "lc") is False


def test_every_queue_site_uses_the_helper():
    """Four sites acquire a slot — direct, keepalive and both brokered. One left on the
    old text-only guard would make the behaviour depend on which path served you."""
    src = (ROOT / "codai/frontproxy/app.py").read_text()
    assert src.count("self._queues_at_front(path, model)") == 4
    # …and no site is left on the old text-only guard.
    assert 'self._task_kind(path) == "text"' not in src
