"""Under `ocr.pod_pipeline: require`, a fallback chain may only name pod engines.

The defect this pins is a coupling, not a crash. `require` means the
orchestrator never rasterises and never reads -- on a GPU-less box that rule is
what stands between a batch and four 9 GB surya processes in swap. But only
surya, chandra and olmOCR have a pod-side pipeline, and `fallback_engines: auto`
orders the chain fewest-ways-to-fail first, which puts paddle and docTR at the
front: on such an install `auto`'s first two choices were the two engines that
can only run locally.

The documented way out was `fallback_engines: "off"` -- which ALSO switches off
the degraded-page recovery, because `retry_degraded: "auto"` means "the fallback
order". So an install protecting its orchestrator silently lost the mechanism
written for the run that came out ~97% truncated with no error anywhere. Found
live on a production install, where both settings were exactly that.
"""
import types

import pytest

from codai.ocr import manager as mgr


def _cfg(**kw):
    base = dict(pod_pipeline="require", pod_pipeline_wait_s=600.0,
                fallback_engines="auto", retry_degraded="auto",
                default_engine="surya",
                paddle_enabled=True, doctr_enabled=True,
                surya_enabled=True, surya_accept_license=True,
                chandra_enabled=True, chandra_accept_license=True,
                olmocr_enabled=True)
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.fixture
def m(monkeypatch):
    """The manager, with a config of the test's choosing and no real pods."""
    man = mgr.OcrManager() if hasattr(mgr, "OcrManager") else mgr.ocr_manager
    cfgs = {}

    def _use(cfg):
        cfgs["cfg"] = cfg
        monkeypatch.setattr(man, "_require_cfg", lambda: cfg, raising=False)
        # Only the three VLM engines have a pod-side pipeline.
        monkeypatch.setattr(mgr, "pod_pipeline_model",
                            lambda c, engine="": ("model-id" if engine in
                                                  ("surya", "chandra", "olmocr") else ""),
                            raising=True)
        return man
    return _use


def test_require_drops_the_engines_that_can_only_read_here(m):
    man = m(_cfg())
    order = man._fallback_order("surya")
    assert order, "require must not empty the chain, only narrow it"
    assert "paddle" not in order and "doctr" not in order
    assert set(order) <= {"chandra", "olmocr"}


def test_without_require_the_local_engines_stay(m):
    """`prefer` and `auto` may read here, so a local engine is a real fallback."""
    man = m(_cfg(pod_pipeline="prefer"))
    assert "paddle" in man._fallback_order("surya")


def test_an_explicitly_named_local_engine_is_dropped_too(m):
    """Naming it does not make this box allowed to read it."""
    man = m(_cfg(fallback_engines="paddle doctr chandra"))
    assert man._fallback_order("surya") == ["chandra"]


def test_the_degraded_chain_is_narrowed_the_same_way(m):
    man = m(_cfg())
    chain = man._retry_degraded_order("surya")
    assert chain and set(chain) <= {"chandra", "olmocr"}


def test_the_degraded_chain_can_be_named_outright(m):
    man = m(_cfg(retry_degraded="chandra olmocr"))
    assert man._retry_degraded_order("surya") == ["chandra", "olmocr"]


def test_off_still_means_off(m):
    man = m(_cfg(retry_degraded="off"))
    assert man._retry_degraded_order("surya") == []


def test_degraded_recovery_survives_fallback_being_off(m):
    """The whole point: the two settings are no longer one setting.

    An install may well want no engine substitution when an engine FAILS while
    still wanting a looping page re-read -- those are different events.
    """
    man = m(_cfg(fallback_engines="off", retry_degraded="chandra olmocr"))
    assert man._fallback_order("surya") == []
    assert man._retry_degraded_order("surya") == ["chandra", "olmocr"]


def test_the_failed_engine_is_never_its_own_fallback(m):
    man = m(_cfg())
    assert "surya" not in man._fallback_order("surya")
    assert "chandra" not in man._fallback_order("chandra")


def test_a_disabled_engine_is_not_offered(m):
    man = m(_cfg(chandra_enabled=False))
    assert man._fallback_order("surya") == ["olmocr"]
