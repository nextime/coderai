"""The configured vLLM share is a starting point, not a verdict.

An engine's real footprint is not knowable up front: surya-ocr-2 on a 24 GB 3090 needs
~10.9 GiB before a single KV block, almost all of it the vision encoder's profiling peak,
and another model or another card is another number entirely. Guessing a constant is what
produced the 0.35 that could never boot — so when vLLM reports the budget cannot hold the
cache, the launcher escalates, re-evicts for the larger figure, and remembers what worked
keyed by model+card+limits.

No vLLM, no GPU: the single-attempt launcher is stubbed and the learned-share file is
redirected into tmp_path.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codai.api import vllm_worker


KV_ERR = ("vLLM exited (code 1) before becoming ready. Last output: "
          "ValueError: No available memory for the cache blocks. Try increasing "
          "`gpu_memory_utilization` when initializing the engine")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Learned shares go to tmp, the card is a 24 GB one, eviction is a no-op."""
    monkeypatch.setattr(vllm_worker, "_learned_gmu_path",
                        lambda: tmp_path / "vllm_gmu_learned.json")
    from codai.models.manager import multi_model_manager
    monkeypatch.setattr(multi_model_manager, "_total_vram_gb", lambda: 24.0)
    monkeypatch.setattr(multi_model_manager, "_get_free_vram_gb", lambda: 24.0)
    monkeypatch.setattr(multi_model_manager, "_evict_models_for_vram", lambda gb: None)


def _cfg(**kw):
    base = {"model_path": "m", "ctx": 18432, "gpu_memory_utilization": 0.0}
    base.update(kw)
    return type("C", (), base)()


def _launcher(monkeypatch, succeed_at: float = None, error=KV_ERR):
    """Stub the one-shot launcher: record every share tried, fail until `succeed_at`."""
    tried = []

    def _once(cfg, model_path=None, served_name=None, ready_timeout=3600.0,
              gmu_absolute=0.0, max_model_len=None, max_num_batched_tokens=None,
              max_num_seqs=None):
        tried.append(round(float(gmu_absolute), 3))
        if succeed_at is not None and gmu_absolute >= succeed_at - 1e-6:
            return "http://127.0.0.1:1/"
        raise RuntimeError(error)

    monkeypatch.setattr(vllm_worker, "_ensure_service_once", _once)
    return tried


# ---------------------------------------------------------------- escalation
def test_a_budget_too_small_for_the_cache_is_retried_larger(monkeypatch):
    tried = _launcher(monkeypatch, succeed_at=0.72)
    url = vllm_worker.ensure_service(_cfg(), model_path="m", served_name="m",
                                     gpu_memory_utilization=0.60)
    assert url == "http://127.0.0.1:1/"
    assert tried[0] == 0.60                      # the configured opening bid
    assert tried == sorted(tried) and len(tried) >= 2
    assert tried[-1] >= 0.72                     # climbed until it fit


def test_it_escalates_by_the_configured_step(monkeypatch):
    tried = _launcher(monkeypatch, succeed_at=0.99)   # never fits below the ceiling
    with pytest.raises(RuntimeError):
        vllm_worker.ensure_service(_cfg(), model_path="m", served_name="m",
                                   gpu_memory_utilization=0.50)
    assert tried[0] == 0.50
    assert tried[1] == pytest.approx(0.50 + vllm_worker._GMU_ESCALATION_STEP, abs=1e-3)


def test_escalation_stops_at_the_ceiling_and_reraises(monkeypatch):
    tried = _launcher(monkeypatch, succeed_at=None)
    with pytest.raises(RuntimeError) as e:
        vllm_worker.ensure_service(_cfg(), model_path="m", served_name="m",
                                   gpu_memory_utilization=0.60)
    assert "No available memory" in str(e.value)
    assert max(tried) <= vllm_worker.GMU_CEILING + 1e-6
    assert len(tried) <= vllm_worker._GMU_MAX_ATTEMPTS


def test_an_unrelated_failure_is_not_retried(monkeypatch):
    """Escalating past a missing model or a broken venv would just waste boots."""
    tried = _launcher(monkeypatch, succeed_at=None,
                      error="vLLM exited (code 1). Last output: No such file: /nope")
    with pytest.raises(RuntimeError):
        vllm_worker.ensure_service(_cfg(), model_path="m", served_name="m",
                                   gpu_memory_utilization=0.60)
    assert tried == [0.60]                       # one attempt, then give up


def test_a_first_attempt_that_works_is_left_alone(monkeypatch):
    tried = _launcher(monkeypatch, succeed_at=0.0)
    vllm_worker.ensure_service(_cfg(), model_path="m", served_name="m",
                               gpu_memory_utilization=0.60)
    assert tried == [0.60]


def test_no_share_anywhere_means_one_attempt_on_vllms_own_default(monkeypatch):
    tried = _launcher(monkeypatch, succeed_at=0.0)
    vllm_worker.ensure_service(_cfg(), model_path="m", served_name="m")
    assert tried == [0.0]


# ---------------------------------------------------------------- learning
def test_what_booted_is_remembered_and_reused(monkeypatch, tmp_path):
    tried = _launcher(monkeypatch, succeed_at=0.72)
    vllm_worker.ensure_service(_cfg(), model_path="m", served_name="m",
                               gpu_memory_utilization=0.60)
    assert (tmp_path / "vllm_gmu_learned.json").exists()

    # Second start: no rediscovery, straight to the share that worked.
    tried2 = _launcher(monkeypatch, succeed_at=0.72)
    vllm_worker.ensure_service(_cfg(), model_path="m", served_name="m",
                               gpu_memory_utilization=0.60)
    assert len(tried2) == 1 and tried2[0] >= 0.72


def test_a_learned_share_is_scoped_to_model_card_and_limits(monkeypatch):
    """A figure learned for one model on one card says nothing about another."""
    _launcher(monkeypatch, succeed_at=0.72)
    vllm_worker.ensure_service(_cfg(), model_path="m", served_name="surya",
                               gpu_memory_utilization=0.60)

    tried = _launcher(monkeypatch, succeed_at=0.72)
    vllm_worker.ensure_service(_cfg(), model_path="m", served_name="olmocr",
                               gpu_memory_utilization=0.60)
    assert tried[0] == 0.60          # a different model rediscovers from the opening bid

    k_surya = vllm_worker._learned_gmu_key(_cfg(), "surya", 18432, 4096)
    k_ctx = vllm_worker._learned_gmu_key(_cfg(), "surya", 8192, 4096)
    k_batch = vllm_worker._learned_gmu_key(_cfg(), "surya", 18432, 2048)
    assert len({k_surya, k_ctx, k_batch}) == 3    # context and batch size are part of it


def test_a_lower_configured_share_never_undoes_what_was_learned(monkeypatch):
    """Turning the knob down must not reintroduce a budget already proven too small."""
    _launcher(monkeypatch, succeed_at=0.72)
    vllm_worker.ensure_service(_cfg(), model_path="m", served_name="m",
                               gpu_memory_utilization=0.60)
    tried = _launcher(monkeypatch, succeed_at=0.72)
    vllm_worker.ensure_service(_cfg(), model_path="m", served_name="m",
                               gpu_memory_utilization=0.20)
    assert tried[0] >= 0.72


def test_an_unwritable_store_costs_rediscovery_not_a_failure(monkeypatch):
    monkeypatch.setattr(vllm_worker, "_learned_gmu_path",
                        lambda: Path("/proc/nope/vllm_gmu_learned.json"))
    tried = _launcher(monkeypatch, succeed_at=0.72)
    url = vllm_worker.ensure_service(_cfg(), model_path="m", served_name="m",
                                     gpu_memory_utilization=0.60)
    assert url == "http://127.0.0.1:1/" and len(tried) >= 2
