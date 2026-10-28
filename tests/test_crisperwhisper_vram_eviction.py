"""A CrisperWhisper worker lives in another process, so the card must be managed for it.

Three models were configured on the crisperwhisper backend and they stacked: each one
is a resident Whisper-large of its own, the reservation taken before loading was the
weights-only 3 GB rather than the inference peak, and nothing in the eviction ladder
could give a worker's VRAM back. On a 24 GB card already serving text and embeddings
that ended as `torch.OutOfMemoryError` inside the encoder's softmax — a transcription
that failed with the card full of models nobody had asked it to keep.

What is asserted here: the reservation is measured and not guessed, a worker with a
transcription in flight is never torn down under it, idle siblings are, and the manager
can reclaim the card through the registered releaser.
"""
import threading
import time

import pytest


@pytest.fixture
def cw(monkeypatch):
    """The worker module with its process table empty and nothing really spawned."""
    from codai.api import crisperwhisper_worker as mod
    monkeypatch.setattr(mod, "_services", {}, raising=False)
    monkeypatch.setattr(mod, "_inflight", {}, raising=False)
    monkeypatch.setattr(mod, "_last_used", {}, raising=False)
    mod._idle.set()
    stopped = []
    monkeypatch.setattr(mod, "stop_service", lambda name: stopped.append(name))
    mod._stopped = stopped
    return mod


class _FakeManager:
    def __init__(self, measured=None):
        self._measured_vram_gb = dict(measured or {})
        self._loading_reservations = {}
        self.releasers = []

    def register_external_vram_releaser(self, fn):
        if fn not in self.releasers:
            self.releasers.append(fn)

    def note_loading(self, key, gb=0.0):
        self._loading_reservations[key] = float(gb)

    def clear_loading(self, key):
        self._loading_reservations.pop(key, None)


@pytest.fixture
def mgr(monkeypatch):
    """Stand in for codai.models.manager.multi_model_manager."""
    import sys
    import types
    fake = _FakeManager()
    module = types.ModuleType("codai.models.manager")
    module.multi_model_manager = fake
    monkeypatch.setitem(sys.modules, "codai.models.manager", module)
    return fake


# ── the reservation ──────────────────────────────────────────────────────────

def test_an_unmeasured_model_reserves_the_inference_peak_not_the_weights(cw, mgr):
    """Whisper-large weighs ~3 GB at fp16; a verbatim pass peaks far above that."""
    assert cw.reserved_vram_gb("nyrahealth/CrisperWhisper", {}) == cw.DEFAULT_VRAM_GB
    assert cw.DEFAULT_VRAM_GB > 3.0


def test_the_operators_own_figure_wins(cw, mgr):
    mgr._measured_vram_gb["audio:m"] = 9.0
    assert cw.reserved_vram_gb("m", {"used_vram_gb": 5.5}) == 5.5


def test_a_measured_peak_beats_the_default(cw, mgr):
    mgr._measured_vram_gb["audio:m"] = 11.25
    assert cw.reserved_vram_gb("m", {}) == 11.25


def test_a_reported_peak_is_recorded_as_a_high_water_mark(cw, mgr):
    cw.record_measurement("m", 7.5)
    assert mgr._measured_vram_gb["audio:m"] == 7.5
    cw.record_measurement("m", 10.0)          # longer audio, bigger window
    assert mgr._measured_vram_gb["audio:m"] == 10.0
    cw.record_measurement("m", 2.0)           # one short clip must not shrink it
    assert mgr._measured_vram_gb["audio:m"] == 10.0


def test_a_worker_that_reports_nothing_leaves_the_figure_alone(cw, mgr):
    cw.record_measurement("m", 0.0)
    assert "audio:m" not in mgr._measured_vram_gb


# ── in flight ────────────────────────────────────────────────────────────────

def test_an_idle_sibling_is_stopped_but_a_busy_one_is_not(cw, mgr):
    cw._services.update({"a": {}, "b": {}, "c": {}})
    cw._inflight["b"] = 1
    cw.evict_idle_siblings(keep="c")
    assert cw._stopped == ["a"]


def test_the_model_being_asked_for_is_never_the_one_evicted(cw, mgr):
    cw._services.update({"a": {}, "b": {}})
    cw.evict_idle_siblings(keep="a")
    assert cw._stopped == ["b"]


def test_what_is_freed_is_the_sibling_footprint_not_a_count(cw, mgr):
    mgr._measured_vram_gb["audio:a"] = 9.5
    cw._services.update({"a": {}, "b": {}})
    freed = cw.evict_idle_siblings(keep="b")
    assert freed == pytest.approx(9.5)


def test_inflight_nests_and_only_the_last_exit_goes_idle(cw, mgr):
    assert cw._idle.is_set()
    with cw.inflight("m"):
        assert not cw._idle.is_set()
        with cw.inflight("m"):
            assert not cw._idle.is_set()
        assert not cw._idle.is_set()
    assert cw._idle.is_set()


def test_two_models_in_flight_keep_each_other_from_being_declared_idle(cw, mgr):
    with cw.inflight("a"):
        with cw.inflight("b"):
            pass
        assert not cw._idle.is_set()
    assert cw._idle.is_set()


# ── giving the card back ─────────────────────────────────────────────────────

def test_the_releaser_frees_idle_workers_at_once(cw, mgr):
    mgr._measured_vram_gb.update({"audio:a": 8.0, "audio:b": 4.0})
    cw._services.update({"a": {}, "b": {}})
    freed = cw.release_vram(999.0)
    assert sorted(cw._stopped) == ["a", "b"]
    assert freed == pytest.approx(12.0)


def test_the_releaser_waits_for_a_transcription_rather_than_losing_it(cw, mgr, monkeypatch):
    """A 30s window is seconds of work: waiting beats killing the request."""
    monkeypatch.setattr(cw, "_drain_timeout_s", lambda: 5.0)
    cw._services["a"] = {}
    ctx = cw.inflight("a")
    ctx.__enter__()

    def _finish():
        time.sleep(0.2)
        ctx.__exit__(None, None, None)

    threading.Thread(target=_finish, daemon=True).start()
    started = time.monotonic()
    freed = cw.release_vram(999.0)
    assert time.monotonic() - started >= 0.15      # it waited
    assert cw._stopped == ["a"]                    # and then took the card
    assert freed > 0


def test_a_worker_still_busy_past_the_budget_is_released_anyway(cw, mgr, monkeypatch):
    monkeypatch.setattr(cw, "_drain_timeout_s", lambda: 0.1)
    cw._services["a"] = {}
    with cw.inflight("a"):
        cw.release_vram(999.0)
    assert cw._stopped == ["a"]


def test_the_releaser_is_safe_with_nothing_to_free(cw, mgr):
    assert cw.release_vram(999.0) == 0.0
    assert cw._stopped == []


def test_the_releaser_is_registered_once_and_with_the_manager(cw, mgr):
    cw._releaser_registered = False
    cw.register_releaser()
    cw.register_releaser()
    assert mgr.releasers == [cw.release_vram]


def test_registration_survives_a_manager_that_is_not_there(cw, monkeypatch):
    import sys
    import types
    broken = types.ModuleType("codai.models.manager")   # no multi_model_manager
    monkeypatch.setitem(sys.modules, "codai.models.manager", broken)
    cw._releaser_registered = False
    cw.register_releaser()                               # must not raise
    assert cw._releaser_registered is False


# ── the load window ──────────────────────────────────────────────────────────

def test_the_cold_start_window_is_reserved_and_then_released(cw, mgr):
    """Free VRAM comes from the driver, so a worker that is still downloading its
    shards looks like room. The reservation is what makes that window visible."""
    key = cw.model_key("m")
    cw._note_loading(key, cw.reserved_vram_gb("m", {}))
    assert mgr._loading_reservations[key] == cw.DEFAULT_VRAM_GB
    cw._clear_loading(key)
    assert key not in mgr._loading_reservations


def test_a_zero_reservation_is_not_recorded(cw, mgr):
    cw._note_loading(cw.model_key("m"), 0.0)
    assert mgr._loading_reservations == {}
