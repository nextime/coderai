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

"""An adapter registered after a pool exists must still reach its pods.

A pool object outlives every pod in it. `seed_entries` -- the list of names a
pooled pod must answer for -- was filled the first time it came back non-empty
and then kept for the life of the PROCESS, so an adapter registered through
`/v1/loras/register` afterwards was never seeded into any pod rented later.

Measured live. An adapter was registered at 17:34; a pod rented at 18:18 came
ready at 18:29 and was sent the two adapters the pool had known about since
boot and not that one. The pod then answered under the adapter's name with
BASE weights, which is the failure mode this file exists to prevent: vLLM
registers adapters when its engine STARTS, so the name resolved, the gateway
reported teaching the pod, generation succeeded, and the output was the base
model's. Nothing failed. Recycling the pod could not fix it, because the fresh
pod was seeded from the same stale list.
"""

from __future__ import annotations

import pytest

import codai.api.runpod_worker as rw


BASE_REPO = "vendor/Some-27B-AWQ-INT4"

BASE = {"id": "base-awq", "alias": "base", "path": BASE_REPO,
        "runpod": {"pool": "shared", "served_model": "base-awq",
                   "hf_repo": BASE_REPO, "engine": "coderai-vllm"}}
FIRST = {"id": "tuned-alpha", "enabled": True,
         "loras": [{"name": "tuned-alpha", "path": "sha256:" + "a" * 64,
                    "weight": 1.0}],
         "runpod": {"pool": "shared", "served_model": "tuned-alpha",
                    "hf_repo": BASE_REPO, "engine": "coderai-vllm"}}
LATER = {"id": "trained-later", "enabled": True,
         "loras": [{"name": "trained-later", "path": "sha256:" + "b" * 64,
                    "weight": 1.0}],
         "runpod": {"pool": "shared", "served_model": "trained-later",
                    "hf_repo": BASE_REPO, "engine": "coderai-vllm"}}


class _CM:
    def __init__(self, data):
        self.models_data = data


@pytest.fixture
def catalogue(monkeypatch):
    """A catalogue holding the base and ONE adapter, as at pool creation."""
    import codai.admin.routes as routes
    data = {"text_models": [dict(BASE), dict(FIRST)]}
    monkeypatch.setattr(routes, "config_manager", _CM(data), raising=False)
    return data


@pytest.fixture
def pool(monkeypatch):
    """A real pool object for `pool:shared`, with nothing rented."""
    monkeypatch.setattr(rw, "_ensure_scaler", lambda: None, raising=False)
    monkeypatch.setattr(rw, "_pools", {}, raising=False)
    created = {}

    class _Pool:
        def __init__(self, key, account, mcfg, served, entry):
            self.model_key, self.account = key, account
            self.mcfg, self.served, self.entry = mcfg, served, entry
            self.seed_entries = []
            created["pool"] = self

    monkeypatch.setattr(rw, "RunpodPodPool", _Pool, raising=False)
    return created


def fetch(entry):
    """Ask for the shared pool the way a load does."""
    return rw.get_pod_pool("pool:shared", object(), object(),
                           str(entry.get("id") or ""), shared=True,
                           entry=entry)


def seeded(p):
    return sorted(str(e.get("id") or "") for e in (p.seed_entries or []))


# --------------------------------------------------------------------------- #

def test_the_pool_seeds_the_adapter_that_existed_at_creation(catalogue, pool):
    p = fetch(BASE)
    assert seeded(p) == ["tuned-alpha"]


def test_AN_ADAPTER_REGISTERED_LATER_REACHES_THE_POOL(catalogue, pool):
    """THE ONE THAT COST THE EVENING.

    The pool is created, then an adapter is registered, then another pod is
    rented. Before the fix the second fetch returned the list built at
    creation and the new adapter was invisible until the container restarted.
    """
    p = fetch(BASE)
    assert seeded(p) == ["tuned-alpha"]

    # `/v1/loras/register` appends to the live catalogue and saves it.
    catalogue["text_models"].append(dict(LATER))

    p2 = fetch(BASE)
    assert p2 is p, "the same pool object is reused, which is the trap"
    assert seeded(p2) == ["trained-later", "tuned-alpha"], (
        "an adapter registered after the pool existed must be seeded into "
        "pods rented afterwards, or the pod answers under its name with "
        "base weights")


def test_an_adapter_removed_from_the_catalogue_stops_being_seeded(catalogue,
                                                                  pool):
    """Recomputing has to work in both directions, or it is still a cache."""
    catalogue["text_models"].append(dict(LATER))
    p = fetch(BASE)
    assert seeded(p) == ["trained-later", "tuned-alpha"]
    catalogue["text_models"] = [e for e in catalogue["text_models"]
                                if e.get("id") != "trained-later"]
    p2 = fetch(BASE)
    assert seeded(p2) == ["tuned-alpha"]


def test_an_empty_catalogue_does_not_erase_a_good_list(catalogue, pool):
    """A catalogue mid-reload must not strip a pod's adapters.

    A pod seeded with nothing launches vLLM without `--enable-lora`, and that
    cannot be turned on afterwards: the base has to be reloaded on a card
    being paid for by the hour. So an empty answer is ignored rather than
    believed.
    """
    p = fetch(BASE)
    assert seeded(p) == ["tuned-alpha"]
    catalogue["text_models"] = []
    p2 = fetch(BASE)
    assert seeded(p2) == ["tuned-alpha"], (
        "an empty recompute must not clear adapters a pod needs")


def test_a_disabled_adapter_is_not_seeded(catalogue, pool):
    later = dict(LATER)
    later["enabled"] = False
    catalogue["text_models"].append(later)
    p = fetch(BASE)
    assert seeded(p) == ["tuned-alpha"]


def test_the_model_asking_is_never_seeded_to_itself(catalogue, pool):
    """`exclude_id` still holds: a pod does not register its own renter twice."""
    p = fetch(FIRST)
    assert "tuned-alpha" not in seeded(p)


def test_an_unshared_pool_is_left_alone(catalogue, pool, monkeypatch):
    """Only named pools carry siblings; a private pool must not grow any."""
    rw._pools.clear()
    p = rw.get_pod_pool("base-awq", object(), object(), "base-awq",
                        shared=False, entry=BASE)
    assert seeded(p) == []
