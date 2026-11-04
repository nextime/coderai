"""A pod rented on a fallback card must be adoptable after a restart.

`allow_other_gpus` lets a pool take whatever is free when none of its preferred
cards are -- that is the whole point of the setting, and it is how a pool whose
`gpu_types` is "NVIDIA A40" comes to be running an RTX A6000.

Adoption then refused that pod, because it filtered candidates on the pool's
PREFERRED list. For renting, "is this the card we prefer?" is the right
question. For adoption it is the wrong one: the pod is already rented and
already billing, and the question that matters is "would this configuration
have taken it?" -- to which the answer is yes, demonstrably, because this
configuration is what rented it.

The cost of getting that wrong is paid twice on every restart. The survivor
keeps billing outside the pool until the reaper's 20-minute grace expires, and
a pool with a warm floor rents a FRESH pod in the meantime -- another 6-14
minute boot, at double rate through the overlap.

Measured on the Digesta host, 2026-11-04, across three restarts in one hour:

    [runpod] not adopting pod lc4uzl4h8ujhw5: it runs on RTX A6000,
             this pool asks for NVIDIA A40

repeating every 15 seconds, for a pod that pool had rented itself, with
`gpu_types: "NVIDIA A40"`, `allow_other_gpus: true`, `min_pods: 1`.

`_adopt_survivors` already exempts survivors from the floor and the cap -- "they
are already paid for, so they are adopted whatever the floor or the cap say".
The card is the same kind of fact and was the one exemption nobody made.

With `allow_other_gpus` OFF the filter stands: the operator said they would
rather wait than take another card, and adopting one would quietly serve from
hardware they refused.
"""
import inspect

import pytest

from codai.api import runpod_worker as W

POOL = "digesta-qwen38"
POD = "lc4uzl4h8ujhw5"


@pytest.fixture
def registry(monkeypatch):
    """One surviving pod, on a fallback card, left by a dead owner."""
    entry = {
        POD: {"pool": POOL, "url": "https://pod.example/v1", "key": "tok",
              "gpu": "RTX A6000", "image": "ghcr.io/nextime/coderai-vllm:latest",
              "hourly_usd": 0.82, "at": 0.0, "pid": 1},
    }
    monkeypatch.setattr(W, "_registry_update", lambda fn: dict(entry))
    monkeypatch.setattr(W, "_is_this_process", lambda info: False)
    monkeypatch.setattr(W, "_pod_health_ok",
                        lambda url, path=None, api_key=None, timeout=0: True)
    return entry


def _find(gpus):
    return W.find_shared_pod(POOL, "/v1/models", "poolkey",
                             image="ghcr.io/nextime/coderai-vllm:latest",
                             gpus=gpus, exclude=set())


# ── the refusal, and its absence ─────────────────────────────────────────────

def test_the_preferred_card_filter_is_what_refused_it(registry, capsys):
    """The bug, pinned: with the preferred list passed in, the survivor is
    refused and the message is the one seen in production."""
    pod_id, url, _key, _rate = _find(["NVIDIA A40"])
    assert pod_id is None and not url
    out = capsys.readouterr().out
    assert "not adopting pod" in out and "RTX A6000" in out


def test_without_a_card_filter_the_survivor_is_adopted(registry):
    """What the fix passes instead."""
    pod_id, url, key, rate = _find([])
    assert pod_id == POD
    assert url == "https://pod.example/v1"
    assert key == "tok", "the pod answers only to the token it booted with"
    assert rate == pytest.approx(0.82)


def test_the_card_filter_still_works_when_it_is_asked_for(registry):
    """Not disabled outright -- a pool that may NOT take other cards keeps it."""
    assert _find(["RTX A6000"])[0] == POD, "a matching card must still adopt"
    assert _find(["NVIDIA L40S"])[0] is None, "a foreign card must still refuse"


def test_the_image_check_is_untouched(registry, capsys):
    """Different image is different code; that refusal is correct and stays."""
    pod_id, _u, _k, _r = W.find_shared_pod(
        POOL, "/v1/models", "poolkey", image="ghcr.io/other/image:latest",
        gpus=[], exclude=set())
    assert pod_id is None
    assert "it runs" in capsys.readouterr().out


# ── where the decision is made ───────────────────────────────────────────────

def test_adoption_drops_the_card_filter_when_other_cards_are_allowed():
    src = inspect.getsource(W.RunpodPodPool._adopt_shared_pod)
    assert "allow_other" in src, "the setting is not consulted at all"
    i = src.index("allow_other and want_gpus")
    assert "want_gpus = []" in src[i:i + 2200], (
        "the card filter must be dropped, not merely noted")
    # And only then: with the setting off, the filter has to survive.
    assert src.index("want_gpus = list(narrow") < i


def test_the_reason_is_logged_not_silent():
    """A pod adopted on a card the pool does not prefer is a surprise worth
    explaining the first time somebody reads the log for it."""
    src = inspect.getsource(W.RunpodPodPool._adopt_shared_pod)
    i = src.index("allow_other and want_gpus")
    body = src[i:i + 2200]
    assert "print(" in body
    assert "already paying for" in body


def test_the_fallback_path_reads_the_setting_too():
    """narrow_gpu_choice can raise; the except branch must not silently
    reinstate the filter it is meant to drop."""
    src = inspect.getsource(W.RunpodPodPool._adopt_shared_pod)
    head = src[:src.index("allow_other and want_gpus")]
    assert head.count("allow_other") >= 2, (
        "the except branch does not set allow_other, so a narrow_gpu_choice "
        "failure would fall back to filtering on the preferred card")


def test_narrow_gpu_choice_reports_the_setting():
    """The fix reads `allow_other_gpus` off that result, so it has to be there."""
    out = W.narrow_gpu_choice(
        type("M", (), {"gpu_types": ["NVIDIA A40"], "allow_other_gpus": True})())
    assert out.get("allow_other_gpus") is True
    out = W.narrow_gpu_choice(
        type("M", (), {"gpu_types": ["NVIDIA A40"], "allow_other_gpus": False})())
    assert out.get("allow_other_gpus") is False
