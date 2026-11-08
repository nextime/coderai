# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""A pooled model must reach its pod with the repo id INTACT.

A pod is told what to serve through `CODERAI_SEED_MODELS`, and the seeder has a
branch that takes the BASENAME of a path: an engine model named by a local file
(`/AI/x.gguf`) travels as `x.gguf`, because the pod resolves it against its own
cache. A HuggingFace repo id is not a file name and that branch destroys it --
`Qwen/Qwen3-VL-32B-Instruct-FP8` became `Qwen3-VL-32B-Instruct-FP8`, which the
Hub answers 404 for.

It reached that branch only for a model in a SHARED POOL. `_plan` seeds the
pool's own model with the source it resolved, but a sibling with `source=""`:
the sibling has no `RunpodModelConfig` of its own to resolve from, so
`shared_pool_seed_entries` puts the fetchable repo into its `path` instead --
and the basename branch then truncated exactly the value that was placed there
to make the seed work.

Measured on the Digesta orchestrator, 2026-11-08. `qwen3vl32-fp8`
(`pool: qwen3vl32`, `engine: coderai-vllm`, `hf_repo:
Qwen/Qwen3-VL-32B-Instruct-FP8`) rented a healthy L40S that could never load
the model:

    OSError: Qwen3-VL-32B-Instruct-FP8 is not a local folder and is not a
    valid model identifier listed on 'https://huggingface.co/models'
    -> 503 on POST /v1/chat/completions: Model 'qwen3vl32-fp8' failed to load

Nothing was wrong with the config, the image, the card or the token, and the
only symptom arrived minutes into a machine billed by the hour. The sibling
seed is the ONLY path that carries a repo id in `path`, which is why no
un-pooled model ever showed it -- `qwen38-vision`, same image and same kind of
`hf_repo`, has always worked.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codai.api import runpod_worker as rw

REPO = "Qwen/Qwen3-VL-32B-Instruct-FP8"


def _entry(**over) -> dict:
    """The real failing entry, reduced to what seeding reads."""
    e = {"id": "qwen3vl32-fp8", "backend": "runpod", "enabled": True,
         "model_type": "vision", "n_ctx": 32768,
         "runpod": {"engine": "coderai-vllm", "hf_repo": REPO,
                    "pool": "qwen3vl32", "served_model": "qwen3vl32-fp8",
                    "mode": "pods"}}
    e.update(over)
    return e


def _sibling_seed(entry: dict) -> dict:
    """What `_plan` builds for a model that JOINED a pool, verbatim: the served
    name from the entry, no source, and the POOL's engine."""
    mcfg = rw.parse_model_runpod(entry["runpod"])
    out = rw.seed_model_env(entry, rw.seed_served_name(entry), "",
                            engine=rw.engine_pod_backend(entry, mcfg))
    return json.loads(out)[0] if out else {}


# --- the regression -------------------------------------------------------

def test_a_pooled_hf_model_is_seeded_with_the_whole_repo_id():
    """The org prefix is half the id: without it the pod fetches nothing."""
    entry = _entry()
    # `shared_pool_seed_entries` fills `path` with what the pod can fetch,
    # because these entries are described by `runpod.hf_repo` and carry no path.
    entry["path"] = rw.resolve_model_source(entry, rw.parse_model_runpod(
        entry["runpod"]))[1]
    assert entry["path"] == REPO, "the source resolver itself must not truncate"

    seed = _sibling_seed(entry)
    assert seed["path"] == REPO
    assert "/" in seed["path"], "a repo id with no owner is a 404"


def test_the_pool_s_own_model_was_never_affected():
    """The renter resolves its own source, so it took the `if source` branch --
    which is why this only ever bit the SECOND name on a pod."""
    entry = _entry()
    mcfg = rw.parse_model_runpod(entry["runpod"])
    kind, value = rw.resolve_model_source(entry, mcfg)
    assert (kind, value) == ("hf", REPO)
    own = json.loads(rw.seed_model_env(
        entry, "qwen3vl32-fp8", value,
        engine=rw.engine_pod_backend(entry, mcfg)))[0]
    assert own["path"] == REPO


def test_the_pod_is_told_the_whole_repo_through_the_plan():
    """Through `_plan`, which is what actually writes the pod's environment --
    the seed above is only correct if it survives into CODERAI_SEED_MODELS."""
    entry = _entry()
    entry["path"] = REPO
    # The POOL's own config, which is what `_plan` resolves the sibling's engine
    # from (`engine_pod_backend(entry or {}, mcfg)`): a stand-in answering None
    # would report no engine and skip the branch under test entirely.
    mcfg = rw.parse_model_runpod(entry["runpod"])
    assert rw.engine_pod_backend({}, mcfg) == "vllm"

    env = rw._plan(engine="", image="coderai-vllm", args={}, mcfg=mcfg,
                   api_key="k", entry=None, served="",
                   seed_entries=[entry]).get("env", {})
    seeded = json.loads(env["CODERAI_SEED_MODELS"])
    assert [s["path"] for s in seeded] == [REPO]
    # ...and under the name the client asks for, or the pod 404s on the alias
    # instead of on the weights.
    assert seeded[0].get("alias") == "qwen3vl32-fp8"


def test_the_alias_still_travels_so_the_pod_answers_the_configured_name():
    seed = _sibling_seed(_entry(path=REPO))
    assert seed["alias"] == "qwen3vl32-fp8"
    assert seed["backend"] == "vllm"
    assert seed["model_type"] == "vision"


# --- what the basename branch is FOR, kept working ------------------------

def test_a_local_engine_file_still_travels_as_its_basename():
    """The behaviour the guard must not break: ds4/colibri/k3 resolve a GGUF
    against the pod's own cache, so only the file name is meaningful there."""
    entry = {"id": "ds4", "path": "/AI/guffcache/DeepSeek-V4-Q4.gguf",
             "backend": "ds4", "model_type": "text"}
    seed = json.loads(rw.seed_model_env(entry, "ds4", "", engine="ds4"))[0]
    assert seed["path"] == "DeepSeek-V4-Q4.gguf"


def test_a_bare_file_name_is_untouched():
    entry = {"id": "k3", "path": "Kimi-K3-Q4.gguf", "backend": "k3",
             "model_type": "text"}
    seed = json.loads(rw.seed_model_env(entry, "k3", "", engine="k3"))[0]
    assert seed["path"] == "Kimi-K3-Q4.gguf"


def test_a_local_directory_is_still_refused_rather_than_basenamed():
    """A local path on a non-engine pod means nothing on a rented machine, and
    a seed naming one would turn "not available" into a load failure on a path
    nobody can see. The guard must not turn it into a plausible-looking name."""
    entry = {"id": "x", "path": "/AI/models/Something", "model_type": "text"}
    assert rw.seed_model_env(entry, "x", "") == ""


def test_a_repo_id_is_told_apart_from_a_file_name():
    """The discriminator itself: owner/name yes, paths and bare files no."""
    assert rw._looks_like_repo_id(REPO) is True
    assert rw._looks_like_repo_id("BAAI/bge-m3") is True
    assert rw._looks_like_repo_id("/AI/x.gguf") is False
    assert rw._looks_like_repo_id("x.gguf") is False
    assert rw._looks_like_repo_id("a/b/c") is False
    assert rw._looks_like_repo_id("./rel/path") is False
