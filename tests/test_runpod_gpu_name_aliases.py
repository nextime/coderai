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

"""The same card, named at two lengths, is still the same card.

Measured live 2026-11-02 19:22, on a pool that had just been restarted:

    [runpod] not adopting pod zpadszianqgtzr: it runs on RTX 6000 Ada, this
    pool asks for NVIDIA A40, NVIDIA L40, NVIDIA L40S, NVIDIA RTX A6000,
    NVIDIA RTX PRO 5000 Blackwell, NVIDIA RTX 6000 Ada Generation

The card it was refused for is in the list it was compared against. The pod was
healthy, had a routable port, was 55 minutes old at $0.74/hr, and was refused
every few seconds -- billing outside any pool and serving nothing, while its
pool reported 1 pod against a floor of 2.

The two strings come from different columns of RunPod's catalogue:
``register_pod`` records the DISPLAY name ("RTX 6000 Ada") and an operator
writes ``gpu_types`` with the id ("NVIDIA RTX 6000 Ada Generation").
``_gpu_matches`` is exact after stripping "NVIDIA ", and exactness is not the
mistake -- a substring match would let "A40" adopt an "RTX A4000", a 16 GB card
where 48 GB was asked for. A40, L40 and L40S hid the bug for months by having a
display name equal to their id minus the prefix.

So the names are resolved through the catalogue, and -- because that catalogue
is a cached file that can be stale, short or missing (one stale row, on the
machine this was written on) -- also on WHOLE WORDS: one name's words being a
leading run of the other's. The three cases that decide whether that is safe
are all real cards in these configurations, and they are the first tests here.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.api import runpod_worker as rw


# ---------------------------------------------------------------------------
# the cases that decide whether this is safe at all
# ---------------------------------------------------------------------------

def test_the_pod_that_was_refused_is_now_adopted():
    """The exact pair from the log."""
    assert rw._gpu_name_matches("RTX 6000 Ada",
                                "NVIDIA RTX 6000 Ada Generation") is True


def test_l40_and_l40s_stay_different_cards():
    """One word apart in spelling, a different card in fact. A substring match
    would conflate them, and a pool asking for L40S would inherit an L40."""
    assert rw._gpu_name_matches("L40", "L40S") is False
    assert rw._gpu_name_matches("L40S", "L40") is False
    assert rw._gpu_name_matches("L40", "NVIDIA L40S") is False


def test_a40_never_matches_an_rtx_a4000():
    """The documented reason the comparison was exact: 16 GB where someone
    asked for 48 GB."""
    assert rw._gpu_name_matches("A40", "NVIDIA RTX A4000") is False
    assert rw._gpu_name_matches("RTX A4000", "NVIDIA A40") is False


# ---------------------------------------------------------------------------
# the names these pools actually carry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("display,configured", [
    ("A40", "NVIDIA A40"),
    ("L40", "NVIDIA L40"),
    ("L40S", "NVIDIA L40S"),
    ("RTX A6000", "NVIDIA RTX A6000"),
    ("RTX 6000 Ada", "NVIDIA RTX 6000 Ada Generation"),
    ("RTX PRO 5000", "NVIDIA RTX PRO 5000 Blackwell"),
    ("RTX PRO 6000", "NVIDIA RTX PRO 6000 Blackwell Server Edition"),
])
def test_every_card_in_the_live_pools_matches_both_ways(display, configured):
    assert rw._gpu_name_matches(display, configured) is True, display
    assert rw._gpu_name_matches(configured, display) is True, configured


@pytest.mark.parametrize("a,b", [
    ("A40", "NVIDIA L40"),
    ("RTX A6000", "NVIDIA RTX 6000 Ada Generation"),
    ("RTX PRO 5000", "NVIDIA RTX PRO 6000 Blackwell Server Edition"),
    ("L40", "NVIDIA A40"),
])
def test_genuinely_different_cards_do_not_match(a, b):
    assert rw._gpu_name_matches(a, b) is False
    assert rw._gpu_name_matches(b, a) is False


@pytest.mark.parametrize("a,b", [("", "NVIDIA A40"), ("A40", ""),
                                 ("", ""), (None, "NVIDIA A40"),
                                 ("A40", None)])
def test_a_missing_name_matches_nothing(a, b):
    """An entry recording no card is refused rather than trusted -- the
    pre-existing rule, which this must not relax."""
    assert rw._gpu_name_matches(a, b) is False


# ---------------------------------------------------------------------------
# how it decides
# ---------------------------------------------------------------------------

def test_the_catalogue_resolves_two_spellings_of_one_card(monkeypatch):
    """The authoritative route: both names name the same catalogue row."""
    monkeypatch.setattr(
        "codai.api.runpod_client._load_saved_catalog_quiet",
        lambda: [{"id": "NVIDIA GeForce RTX 4090",
                  "display_name": "RTX 4090", "memory_gb": 24}])
    assert rw._gpu_name_matches("RTX 4090", "NVIDIA GeForce RTX 4090") is True


def test_it_still_works_with_no_catalogue_at_all(monkeypatch):
    """A cached file that is stale, short or missing must not turn adoption
    off: that is how the pod ended up billing outside its pool."""
    monkeypatch.setattr(
        "codai.api.runpod_client._load_saved_catalog_quiet", lambda: [])
    assert rw._gpu_name_matches("RTX 6000 Ada",
                                "NVIDIA RTX 6000 Ada Generation") is True
    assert rw._gpu_name_matches("L40", "L40S") is False


def test_an_unreadable_catalogue_does_not_raise(monkeypatch):
    def _boom():
        raise OSError("no such file")
    monkeypatch.setattr(
        "codai.api.runpod_client._load_saved_catalog_quiet", _boom)
    assert rw._gpu_name_matches("A40", "NVIDIA A40") is True


def test_a_leading_run_of_words_is_required_not_a_substring(monkeypatch):
    """"Ada Generation" shares words with "RTX 6000 Ada Generation" but is not
    a leading run of it, so it is not the same card named shorter."""
    monkeypatch.setattr(
        "codai.api.runpod_client._load_saved_catalog_quiet", lambda: [])
    assert rw._gpu_name_matches("Ada Generation",
                                "NVIDIA RTX 6000 Ada Generation") is False
    assert rw._gpu_name_matches("6000", "NVIDIA RTX 6000 Ada Generation") is False


def test_the_prefix_is_stripped_on_both_sides(monkeypatch):
    monkeypatch.setattr(
        "codai.api.runpod_client._load_saved_catalog_quiet", lambda: [])
    assert rw._gpu_name_matches("NVIDIA RTX 6000 Ada",
                                "RTX 6000 Ada Generation") is True


def test_case_does_not_matter(monkeypatch):
    monkeypatch.setattr(
        "codai.api.runpod_client._load_saved_catalog_quiet", lambda: [])
    assert rw._gpu_name_matches("rtx 6000 ADA",
                                "NVIDIA RTX 6000 Ada Generation") is True


# ---------------------------------------------------------------------------
# and it has to reach the decision that refused the pod
# ---------------------------------------------------------------------------

def test_the_refused_pod_is_now_handed_back_by_find_shared_pod(monkeypatch):
    """End to end on the live shape: the registry records the display name,
    the pool asks with the id, and the pod must come back."""
    monkeypatch.setattr(rw, "_registry_update", lambda fn: {
        "zpadszianqgtzr": {"pool": "datalab-to/chandra-ocr-2",
                           "url": "https://107.150.186.62:12640/v1",
                           "key": "tok",
                           "image": "ghcr.io/nextime/coderai-vllm-ocr:latest",
                           "gpu": "RTX 6000 Ada", "hourly_usd": 0.74}})
    monkeypatch.setattr(rw, "_is_this_process", lambda info: False)
    monkeypatch.setattr(rw, "_pod_health_ok", lambda *a, **k: True)
    pid, url, key, rate = rw.find_shared_pod(
        "datalab-to/chandra-ocr-2", "/v1/models", "",
        image="ghcr.io/nextime/coderai-vllm-ocr:latest",
        gpus=["NVIDIA A40", "NVIDIA L40", "NVIDIA L40S", "NVIDIA RTX A6000",
              "NVIDIA RTX PRO 5000 Blackwell",
              "NVIDIA RTX 6000 Ada Generation"])
    assert pid == "zpadszianqgtzr"
    assert rate == 0.74


def test_a_pod_on_a_card_the_pool_did_not_ask_for_is_still_refused(monkeypatch):
    """The guard's actual job: a pod rented as a fallback must not be
    inherited by a lease that asked for one card and nothing else."""
    monkeypatch.setattr(rw, "_registry_update", lambda fn: {
        "p1": {"pool": "m", "url": "https://x/v1", "key": "t", "image": "img",
               "gpu": "RTX A4000", "hourly_usd": 0.2}})
    monkeypatch.setattr(rw, "_is_this_process", lambda info: False)
    monkeypatch.setattr(rw, "_pod_health_ok", lambda *a, **k: True)
    pid, url, _k, _r = rw.find_shared_pod("m", "/v1/models", "", image="img",
                                          gpus=["NVIDIA A40"])
    assert pid is None and not url
