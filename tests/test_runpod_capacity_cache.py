"""A card refused for no capacity must not be re-asked by every boot thread.

RunPod's answer to "can I have this card in this region" is the same for every
caller while the shortage lasts, but each boot thread asked independently. On a
live run 32 pods were booting in parallel, each sweeping ~70 candidates: 406
rejections in 120 seconds, every GPU type re-probed 33 to 72 times, one core
saturated (coderai-cpu at 60%).

That mattered because of WHERE the cost landed. CPython gives the GIL to
whoever is running, so the provisioning threads starved the event loop that the
page forwards live on: 1,754 pages counted as in flight, not one socket opened
to a pod, zero documents completed -- and the undelivered queue read as demand,
so the scaler rented more pods, which blocked the loop harder. The fleet grew
because it could not serve, and could not serve because it was growing.

The verdict is now shared and short-lived: shared because asking again is what
cost the run, short because capacity comes back without telling us.
"""

import time

import pytest

from codai.api import runpod_worker as rw


@pytest.fixture(autouse=True)
def _clean():
    rw._unavail.clear()
    yield
    rw._unavail.clear()


def test_an_unmarked_card_is_available():
    assert rw._is_unavailable("NVIDIA A100-SXM4-80GB", "EU-SE-1") is False


def test_a_refusal_is_remembered():
    rw._mark_unavailable("NVIDIA A100-SXM4-80GB", "EU-SE-1")
    assert rw._is_unavailable("NVIDIA A100-SXM4-80GB", "EU-SE-1") is True


def test_it_is_remembered_per_region():
    """A shortage in one region says nothing about another."""
    rw._mark_unavailable("NVIDIA A40", "EU-SE-1")
    assert rw._is_unavailable("NVIDIA A40", "EU-SE-1") is True
    assert rw._is_unavailable("NVIDIA A40", "EU-FR-1") is False


def test_a_regionless_refusal_covers_every_region():
    """RunPod's choice of region: the refusal was not about a place."""
    rw._mark_unavailable("NVIDIA A40", "")
    assert rw._is_unavailable("NVIDIA A40", "EU-FR-1") is True
    assert rw._is_unavailable("NVIDIA A40", "") is True


def test_the_verdict_expires(monkeypatch):
    """Short on purpose: a freed card must be picked up on the next rent."""
    monkeypatch.setattr(rw, "UNAVAIL_TTL_S", 0.05)
    rw._mark_unavailable("NVIDIA L40S", "EU-NL-1")
    assert rw._is_unavailable("NVIDIA L40S", "EU-NL-1") is True
    time.sleep(0.1)
    assert rw._is_unavailable("NVIDIA L40S", "EU-NL-1") is False


def test_a_successful_deploy_clears_the_verdict():
    rw._mark_unavailable("NVIDIA A40", "EU-SE-1")
    rw._forget_unavailable("NVIDIA A40", "EU-SE-1")
    assert rw._is_unavailable("NVIDIA A40", "EU-SE-1") is False


def test_a_deploy_clears_the_regionless_verdict_too():
    rw._mark_unavailable("NVIDIA A40", "")
    rw._forget_unavailable("NVIDIA A40", "EU-SE-1")
    assert rw._is_unavailable("NVIDIA A40", "EU-SE-1") is False


def test_an_empty_card_id_is_never_cached():
    rw._mark_unavailable("", "EU-SE-1")
    assert rw._unavail == {}
    assert rw._is_unavailable("", "EU-SE-1") is False


def test_the_record_does_not_grow_without_bound(monkeypatch):
    monkeypatch.setattr(rw, "UNAVAIL_TTL_S", -1.0)      # expire immediately
    for i in range(5000):
        rw._mark_unavailable(f"card{i}", "r")
    assert len(rw._unavail) <= 4096


def test_the_ttl_is_short_enough_to_not_strand_a_fleet():
    """Too long and a shortage that cleared keeps the pool off a good card."""
    assert 10.0 <= rw.UNAVAIL_TTL_S <= 300.0


def test_one_thread_learning_spares_the_others():
    """The saving, stated as the arithmetic that caused the incident."""
    types = [f"gpu{i}" for i in range(10)]
    regions = [f"r{i}" for i in range(7)]
    probes = []

    def probe(t, r):
        if rw._is_unavailable(t, r):
            return False
        probes.append((t, r))
        rw._mark_unavailable(t, r)      # what the refusal does
        return False

    for _thread in range(32):           # 32 parallel boots
        for t in types:
            for r in regions:
                probe(t, r)
    # 32 x 70 = 2240 without the cache; one pass with it.
    assert len(probes) == 70, len(probes)


# ------------------------------------------------ the catalogue behind it #
# `_rank_gpus` asks for the GPU catalogue, and it runs once per pod being
# provisioned. 32 parallel boots fetched and parsed 32 identical price lists
# within seconds, in Python, holding the GIL the event loop needed.

from codai.api import runpod_client as rc


@pytest.fixture(autouse=True)
def _clean_catalog():
    rc._GPU_CATALOG = (0.0, None)
    yield
    rc._GPU_CATALOG = (0.0, None)


class _FakeClient(rc.RunpodClient):
    def __init__(self):
        self.fetches = 0

    def _gql(self, q, *a, **k):
        self.fetches += 1
        return {"gpuTypes": [
            {"id": "g1", "displayName": "NVIDIA A40", "memoryInGb": 48,
             "securePrice": 1.0, "communityPrice": 0.9,
             "lowestPrice": {"minimumBidPrice": 0.5, "uninterruptablePrice": 1.0}},
        ]}


def test_the_catalogue_is_fetched_once_for_many_callers():
    c = _FakeClient()
    for _ in range(32):                 # 32 boot threads
        rows = c.list_gpu_types()
        assert rows and rows[0]["display_name"] == "NVIDIA A40"
    assert c.fetches == 1, f"{c.fetches} catalogue fetches for 32 boots"


def test_each_caller_gets_its_own_copy():
    """A caller that mutates its rows must not corrupt the cache."""
    c = _FakeClient()
    first = c.list_gpu_types()
    first.append({"id": "junk"})
    assert len(c.list_gpu_types()) == 1


def test_fresh_forces_a_fetch():
    """For a caller showing prices to a person, not picking in a storm."""
    c = _FakeClient()
    c.list_gpu_types()
    c.list_gpu_types(fresh=True)
    assert c.fetches == 2


def test_the_catalogue_cache_expires(monkeypatch):
    c = _FakeClient()
    monkeypatch.setattr(rc, "GPU_CATALOG_TTL_S", 0.05)
    c.list_gpu_types()
    time.sleep(0.1)
    c.list_gpu_types()
    assert c.fetches == 2


def test_the_catalogue_ttl_is_not_so_long_that_prices_go_stale():
    assert 10.0 <= rc.GPU_CATALOG_TTL_S <= 600.0
