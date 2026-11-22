"""A spark2_5 model served on a pod must land on the llamAmpere-fork image.

Serving a spark2_5 GGUF (Spark-X2.5 / rizzo-flow) on RunPod needs the fork
`llama-server`, because the stock `coderai-llamasrv` / upstream llama.cpp images
know neither the `spark2_5` graph nor the `sjkvarn*` KV cache the local
`llamampere` entries use. The runpod block names `engine: coderai-llamampere`,
and the resolver must turn that into the published `coderai-llamampere` image --
landing on the stock llama image instead would boot a pod that cannot load the
model (or loads it with the wrong cache) and bills while it fails.
"""

from codai.api import runpod_worker as rw


def test_the_fork_image_is_a_published_capability():
    assert "llamampere" in rw.PUBLISHED_CAPABILITY_IMAGES
    assert rw._CODERAI_SERVER_ENGINES.get("coderai-llamampere") == "llamampere"


def test_the_engine_resolves_to_the_fork_image():
    img = rw.default_capability_image("llamampere")
    assert img.endswith("coderai-llamampere:latest")


def test_a_spark_entry_lands_on_the_fork_image():
    entry = {"runpod": {"engine": "coderai-llamampere"}}
    mcfg = rw.parse_model_runpod(entry["runpod"])
    img = rw.pod_image_for_entry(entry, mcfg, "spark-x2.5-4b",
                                 "/AI/guffcache/spark-x2.5-4b-q8_0.gguf")
    assert img.endswith("coderai-llamampere:latest")


def test_an_explicit_image_still_wins():
    """`image` on the runpod block is honoured first, as for any engine."""
    entry = {"runpod": {"engine": "coderai-llamampere",
                        "image": "ghcr.io/me/coderai-llamampere:ga102"}}
    mcfg = rw.parse_model_runpod(entry["runpod"])
    img = rw.pod_image_for_entry(entry, mcfg, "spark-x2.5-4b", "")
    assert img == "ghcr.io/me/coderai-llamampere:ga102"


def test_the_stock_llamasrv_engine_is_unchanged():
    """The fork wiring must not disturb the stock llamasrv mapping."""
    assert rw._CODERAI_SERVER_ENGINES.get("coderai-llamasrv") == "llamasrv"
    assert rw.default_capability_image("llamasrv").endswith("coderai-llamasrv:latest")
