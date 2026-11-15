"""Averaging one adapter across pods that cannot reach each other.

The per-step all-reduce this repo already has needs a LAN. Rented pods in
different datacenters cannot address one another, so the exchange is periodic:
train a shard for a round, send the adapter, restart from everyone's average.

Four things in here are silent when wrong, which is why each has a test:

* the average must be weighted by work actually done, or a slow card gets an
  equal vote and quietly steers the adapter;
* the accumulation must be float32. Adapters are bf16 with 8 mantissa bits, so
  summing sixteen of them in bf16 drops every small contribution and produces a
  slightly lobotomised adapter that still loads and still answers;
* shards must partition the dataset exactly -- an overlap trains examples twice
  and a gap never trains them, and neither shows up anywhere;
* a straggler's adapter from an already-averaged round must be refused, not
  folded in, because it is a round of training behind and moves everyone back.
"""

import pytest

torch = pytest.importorskip("torch")

from codai.api import text_lora_rounds as tlr


def _sd(value, shape=(2, 3), dtype=torch.float32, key="lora_A.weight"):
    return {key: torch.full(shape, float(value), dtype=dtype)}


# ------------------------------------------------------------- the average ---

def test_the_average_is_weighted_by_work_done():
    """A pod that managed 300 examples must count three times one that managed
    100 -- otherwise a slow card has an equal say in the adapter."""
    out = tlr.average_adapters([_sd(0.0), _sd(4.0)], weights=[100, 300])
    assert out["lora_A.weight"].flatten()[0].item() == pytest.approx(3.0)


def test_equal_weights_are_the_default():
    out = tlr.average_adapters([_sd(0.0), _sd(2.0)])
    assert out["lora_A.weight"].flatten()[0].item() == pytest.approx(1.0)


def test_a_single_shard_is_itself():
    out = tlr.average_adapters([_sd(1.5)], weights=[7])
    assert out["lora_A.weight"].flatten()[0].item() == pytest.approx(1.5)


def test_bf16_adapters_are_accumulated_in_float32():
    """THE numerical test. bf16 has an 8-bit mantissa: near 1.0 its spacing is
    ~0.0039, so adding 0.002 to 1.0 in bf16 rounds straight back to 1.0. Sum
    sixteen shards that way and fifteen of them vanish.

    float32 keeps them, and the difference survives the final cast back to
    bf16 -- so this distinguishes the two implementations rather than measuring
    rounding noise.
    """
    shards = [_sd(1.0, dtype=torch.bfloat16)] + \
             [_sd(0.002, dtype=torch.bfloat16) for _ in range(15)]
    out = tlr.average_adapters(shards)
    got = out["lora_A.weight"].flatten()[0].to(torch.float32).item()

    naive = torch.zeros((), dtype=torch.bfloat16)
    for s in shards:                      # what bf16 accumulation would give
        naive = naive + s["lora_A.weight"].flatten()[0] / 16
    exact = (1.0 + 15 * 0.002) / 16       # 0.064375

    assert out["lora_A.weight"].dtype == torch.bfloat16, "cast back to the adapter's dtype"
    assert got == pytest.approx(exact, rel=0.02), f"got {got}, exact {exact}"
    assert abs(got - exact) < abs(naive.to(torch.float32).item() - exact)


def test_the_result_keeps_the_adapters_dtype():
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        out = tlr.average_adapters([_sd(1.0, dtype=dt), _sd(3.0, dtype=dt)])
        assert out["lora_A.weight"].dtype == dt


def test_a_shard_with_a_different_parameter_set_is_refused():
    """It trained a different target_modules. Averaging the overlap would make
    an adapter that loads, serves, and is not what any shard trained."""
    a = {"lora_A.weight": torch.zeros(2), "lora_B.weight": torch.zeros(2)}
    b = {"lora_A.weight": torch.zeros(2)}
    with pytest.raises(tlr.RoundError) as e:
        tlr.average_adapters([a, b])
    assert "different parameter set" in str(e.value)
    assert "lora_B.weight" in str(e.value), "name what is missing"


def test_a_shard_with_a_different_rank_is_refused():
    a = {"lora_A.weight": torch.zeros(16, 8)}
    b = {"lora_A.weight": torch.zeros(32, 8)}
    with pytest.raises(tlr.RoundError, match="different rank"):
        tlr.average_adapters([a, b])


def test_weights_that_sum_to_zero_are_refused():
    """Dividing by it would be a NaN adapter; saying so names the real
    situation -- no shard did any work this round."""
    with pytest.raises(tlr.RoundError, match="sum to zero"):
        tlr.average_adapters([_sd(1.0), _sd(2.0)], weights=[0, 0])


def test_a_weight_per_shard_is_required():
    with pytest.raises(tlr.RoundError, match="2 adapters but 1 weights"):
        tlr.average_adapters([_sd(1.0), _sd(2.0)], weights=[1])


def test_averaging_nothing_is_refused():
    with pytest.raises(tlr.RoundError, match="nothing to average"):
        tlr.average_adapters([])


def test_safetensors_files_round_trip(tmp_path):
    from safetensors.torch import load_file, save_file
    paths = []
    for i, v in enumerate((0.0, 4.0)):
        p = str(tmp_path / f"s{i}.safetensors")
        save_file(_sd(v), p)
        paths.append(p)
    out = str(tmp_path / "avg.safetensors")
    tlr.average_adapter_files(paths, [100, 300], out)
    assert load_file(out)["lora_A.weight"].flatten()[0].item() == pytest.approx(3.0)


# --------------------------------------------------------------- the shards --

def test_the_shards_partition_the_dataset_exactly():
    """An overlap trains an example twice, a gap never trains it, and neither
    is visible in any log or metric."""
    plan = tlr.RoundPlan(n_shards=4, seed=7)
    n = 101
    parts = [plan.indices_for(s, n) for s in range(4)]
    flat = [i for p in parts for i in p]
    assert sorted(flat) == list(range(n)), "every example exactly once"
    assert len(flat) == len(set(flat))


def test_the_split_is_the_same_every_time():
    """A pod that dies and is replaced must get its predecessor's shard, and a
    resumed job must not reshuffle the data under a half-trained adapter."""
    a = tlr.RoundPlan(n_shards=3, seed=11).indices_for(1, 50)
    b = tlr.RoundPlan(n_shards=3, seed=11).indices_for(1, 50)
    assert a == b


def test_a_different_seed_splits_differently():
    a = tlr.RoundPlan(n_shards=3, seed=11).indices_for(0, 50)
    b = tlr.RoundPlan(n_shards=3, seed=12).indices_for(0, 50)
    assert a != b


def test_shards_are_balanced_within_one():
    plan = tlr.RoundPlan(n_shards=5, seed=3)
    sizes = [len(plan.indices_for(s, 103)) for s in range(5)]
    assert max(sizes) - min(sizes) <= 1


# -------------------------------------------------------------- the rounds ---

def _tracker(n=4, **kw):
    return tlr.RoundTracker(tlr.RoundPlan(n_shards=n, seed=1), **kw)


def _sub(shard, rnd=0, examples=100):
    return tlr.Submission(shard=shard, round=rnd, path=f"/s{shard}.safetensors",
                          examples=examples)


def test_a_round_is_ready_when_every_shard_arrives():
    t = _tracker(3)
    assert not t.ready()
    for s in range(3):
        t.submit(_sub(s))
    assert t.ready()


def test_a_round_is_ready_at_the_minimum_fraction():
    t = _tracker(4, min_fraction=0.75)
    for s in range(3):
        t.submit(_sub(s))
    assert t.ready(), "3 of 4 is the configured minimum"


def test_a_round_is_never_ready_with_nothing():
    t = _tracker(4, min_fraction=0.0, deadline_s=1)
    assert not t.ready(now=1e9), "a deadline must not average zero shards"


def test_the_deadline_lets_a_round_proceed_without_a_straggler():
    """Waiting for every shard is what turned one dead pod into a stalled
    fleet: the round goes on with the survivors."""
    t = _tracker(4, min_fraction=0.99, deadline_s=600)
    t.open_round(0, now=1000.0)
    t.submit(_sub(0))
    assert not t.ready(now=1100.0)
    assert t.ready(now=1700.0)


def test_a_stale_submission_is_refused_not_folded_in():
    """It is a round of training behind; mixing it into the current average
    moves every pod backwards."""
    t = _tracker(2)
    t.submit(_sub(0, rnd=0))
    t.submit(_sub(1, rnd=0))
    t.close_round()
    with pytest.raises(tlr.RoundError) as e:
        t.submit(_sub(0, rnd=0))
    assert "stale" in str(e.value)


def test_a_submission_from_the_future_is_refused():
    t = _tracker(2)
    with pytest.raises(tlr.RoundError, match="ahead of"):
        t.submit(_sub(0, rnd=5))


def test_closing_a_round_reports_who_was_absent():
    t = _tracker(3)
    t.submit(_sub(0))
    t.submit(_sub(2))
    subs, absent = t.close_round()
    assert [s.shard for s in subs] == [0, 2]
    assert absent == [1]
    assert t.round == 1


def test_a_pod_that_keeps_missing_rounds_is_declared_dead():
    """The round upload IS the heartbeat: a training pod serves nothing and
    answers no health probe, so finishing rounds is its only honest signal."""
    t = _tracker(2)
    for _ in range(2):
        t.submit(_sub(0, rnd=t.round))     # each round, shard 1 never arrives
        t.close_round()
    assert t.dead_shards(after_misses=2) == [1]
    assert t.dead_shards(after_misses=3) == []


def test_a_pod_that_comes_back_is_not_dead():
    t = _tracker(2)
    t.submit(_sub(0, rnd=t.round)); t.close_round()
    assert t.dead_shards(after_misses=1) == [1]
    t.submit(_sub(0, rnd=t.round)); t.submit(_sub(1, rnd=t.round)); t.close_round()
    assert t.dead_shards(after_misses=1) == [], "a completed round clears the miss"


def test_the_averaging_weights_come_from_examples_trained():
    t = _tracker(2)
    t.submit(_sub(0, examples=40))
    t.submit(_sub(1, examples=400))
    subs, _ = t.close_round()
    assert t.weights_for(subs) == [40.0, 400.0]
