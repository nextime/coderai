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

"""A pod must receive the adapter it was seeded to answer for.

A pooled pod is seeded with an adapter's CONTENT ID and the bytes are sent
separately -- `seed_model_env` says it outright: "the pod is told the id when
it is created and sent the bytes before the first request". Skip the sending
and the pod registers the NAME, answers requests for it, and serves the BASE
model's weights under it. Nothing fails anywhere.

Measured on a pool whose only member was the adapter itself. The pod came
ready, answered `digesta-extraction-a4v8` in 5.4s, and returned output
byte-identical to the base -- the same md5 over 400 characters -- because it
had never received the 159 MB it was named after. There was no `sent adapter`
line for that pod at all. The teach path does push adapters, but only when a
pod 404s on the name, and a pod seeded with the id does not 404, so that
safety net never fires.

The decision also sat under a bare `except Exception: pass`, which is the
worst possible place for it: every reason an adapter might not be sent became
silence, and the symptom was a pod quietly serving untuned weights.
"""

from __future__ import annotations

import json

import pytest

import codai.api.remote_gateway as rg


BASE_REPO = "vendor/Some-27B-AWQ-INT4"
BLOB = "sha256:" + "f7" * 32

ADAPTER_ENTRY = {
    "id": "tuned-a4v8", "enabled": True,
    "loras": [{"name": "tuned-a4v8", "path": BLOB, "weight": 1.0}],
    "runpod": {"pool": "tuned-only", "served_model": "tuned-a4v8",
               "hf_repo": BASE_REPO, "engine": "coderai-vllm"},
}


class _CM:
    def __init__(self, data):
        self.models_data = data


def mcfg_for(entry):
    from codai.api.runpod_worker import parse_model_runpod
    return parse_model_runpod(entry.get("runpod") or {})


def decide(entry):
    """The decision under test, called directly."""
    return rg.adapter_entry_to_send(entry, mcfg_for(entry), "tuned-a4v8")


# --------------------------------------------------------------------------- #

def test_the_models_own_adapter_is_queued_for_sending():
    """THE ONE THAT SERVED BASE WEIGHTS UNDER AN ADAPTER'S NAME.

    A content id is not something a pod can fetch, so the entry must be
    queued for `ensure_text_loras`. Before the fix this was reached only for
    pool SIBLINGS, and a pool whose sole member is the adapter has none.
    """
    got = decide(ADAPTER_ENTRY)
    assert got is not None, (
        "the pod is seeded with the adapter's content id; without the bytes "
        "it serves the base model under the adapter's name")
    assert str(got.get("id")) == "tuned-a4v8"


def test_a_coderai_vllm_pod_counts_as_able_to_receive_it():
    """`coderai-vllm` is a coderai pod running vLLM, not a bare vLLM image.

    `resolve_pod_engine` maps it to "coderai", and the upload endpoint is the
    coderai one -- the 159 MB pushes that DID work went to such a pod.
    """
    from codai.api.runpod_worker import resolve_pod_engine
    assert resolve_pod_engine(mcfg_for(ADAPTER_ENTRY), "tuned-a4v8", "",
                              ADAPTER_ENTRY) == "coderai"


def test_an_adapter_the_pod_can_fetch_itself_is_not_pushed():
    """A HuggingFace repo id needs no upload; queueing one would waste a push."""
    e = json.loads(json.dumps(ADAPTER_ENTRY))
    e["loras"] = [{"name": "tuned-a4v8", "path": "vendor/some-lora",
                   "weight": 1.0}]
    assert decide(e) is None


def test_a_model_with_no_adapter_queues_nothing():
    e = json.loads(json.dumps(ADAPTER_ENTRY))
    e.pop("loras")
    assert decide(e) is None


def test_an_image_that_cannot_receive_it_says_so(capsys):
    """A bare vLLM pod has no upload endpoint, and silence there is the bug.

    The pod would answer under the adapter's name with untuned weights, so the
    one clue a human gets must be printed.
    """
    e = json.loads(json.dumps(ADAPTER_ENTRY))
    e["runpod"]["engine"] = "vllm"
    assert decide(e) is None
    out = capsys.readouterr().out
    assert "cannot be sent" in out and "BASE" in out, out


def test_a_failure_to_decide_is_reported_not_swallowed(monkeypatch, capsys):
    """The bare `except Exception: pass` is what made this unfindable."""
    def boom(*a, **k):
        raise RuntimeError("catalogue exploded")
    monkeypatch.setattr("codai.models.text_loras.configured_specs", boom,
                        raising=False)
    assert decide(ADAPTER_ENTRY) is None
    out = capsys.readouterr().out
    assert "could not work out whether" in out, out
    assert "catalogue exploded" in out, out
