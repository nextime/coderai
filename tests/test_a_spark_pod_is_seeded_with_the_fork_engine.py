"""A spark2_5 pod must be SEEDED to serve with the llamAmpere-fork engine.

Picking the right image is only half of serving a spark2_5 GGUF on a pod: the
pod boots with an EMPTY catalogue and is told about the one model it was rented
for via `seed_model_env`. If that seed does not carry `backend: llamampere` and
the `llamampere` tuning block, the pod registers the model with no engine (the
`backend: runpod` placement slot is popped on the far side) and falls through to
its default -- a 503 minutes into a rented machine, the exact failure the
seeding code warns about. It also must carry the per-model `--jinja` extra_args,
without which the spark2_5 chat template never loads.

These guard the four touchpoints that mirror `llamasrv` for `llamampere`:
engine_pod_backend, _ENGINE_POD_BACKENDS/_IMAGE, the seed keep-list.
"""

import json

from codai.api import runpod_worker as rw


def _entry():
    return {
        "alias": "rizzo-flow-1.7b-pod", "backend": "runpod",
        "engine": "llamampere", "model_type": "text_models",
        "path": "/AI/guffcache/spark-x2.5-1.7b-rizzo-flow-lora-q8_0.gguf",
        "llamampere": {"ctx": 32768, "flash_attn": "on",
                       "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                       "extra_args": "-b 2048 -ub 512 --jinja --no-context-shift"},
        "runpod": {
            "engine": "coderai-llamampere",
            "image": "ghcr.io/nextime/coderai-llamampere:latest",
            "source": "url",
            "model_url": "https://huggingface.co/rizzoaiacademy/rizzo-flow/"
                         "resolve/main/spark-x2.5-1.7b-rizzo-flow-lora-q8_0.gguf",
        },
    }


def test_coderai_llamampere_names_the_fork_engine():
    mcfg = rw.parse_model_runpod(_entry()["runpod"])
    assert rw.engine_pod_backend(_entry(), mcfg) == "llamampere"
    assert "llamampere" in rw._ENGINE_POD_BACKENDS
    assert rw._ENGINE_POD_IMAGE.get("llamampere") == "llamampere"


def test_a_bare_llamampere_pin_is_a_native_engine_too():
    """`engine: llamampere` (not the coderai- prefix) resolves the same way."""
    mcfg = rw.parse_model_runpod({"engine": "llamampere"})
    assert rw.engine_pod_backend({"backend": "llamampere"}, mcfg) == "llamampere"


def test_the_spark_pod_runs_a_coderai_server_not_stock_llamacpp():
    e = _entry()
    mcfg = rw.parse_model_runpod(e["runpod"])
    # A .gguf path alone would say "llamacpp"; the explicit fork engine wins and
    # keeps it on a coderai pod (which can be seeded the model after boot).
    assert rw.resolve_pod_engine(mcfg, e["alias"], e["path"], e) == "coderai"


def test_the_seed_carries_the_fork_engine_and_the_jinja_block():
    e = _entry()
    mcfg = rw.parse_model_runpod(e["runpod"])
    kind, value = rw.resolve_model_source(e, mcfg)
    assert kind == "url" and value.endswith("-q8_0.gguf")
    seed = rw.seed_model_env(e, e["alias"], source=value,
                             engine=rw.engine_pod_backend(e, mcfg))
    one = json.loads(seed)[0]
    # The pod serves it with the fork engine, not its default.
    assert one["backend"] == "llamampere"
    # The single GGUF the pod downloads, not a bare repo id (5 files, ambiguous).
    assert one["path"] == value
    # The per-model tuning block travelled, carrying the spark2_5 template flag.
    assert "--jinja" in one["llamampere"]["extra_args"]
