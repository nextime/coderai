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

"""An embedder has to be loaded the way its entry says, and refuse early.

Two faults reported from production on 2026-10-29.

**A local GGUF embedder was fetched from the Hub.** `gme-Qwen2-VL-2B-Instruct-Q4_K_M`
and `dinov2-large-Q8_0` answered

    500 Failed to load embedding model: 404 Client Error …
    huggingface.co/sentence-transformers/<id>/resolve/main/

for files sitting in `/AI/guffcache`. The loader dispatches on the NAME:
`.gguf` goes to llama.cpp (or, for a ViT, to the dinov2-embed server) and
anything else goes to sentence-transformers, which prepends
`sentence-transformers/` to a name with no slash and asks the Hub for it. These
entries are registered under three ids — the path, the basename, and the
basename without the extension — and the request arrived as the last of those.

The config lookup missed for the same reason, which is worse than it looks: it
carries `mmproj`, so the vision GGUF would have lost its image tower even if
the dispatch had been right.

**An impossible request paid for a model load before being refused.** Four
texts and four images to a text-only 8 GB embedder: forty seconds of loading,
then the 400 it could have had immediately. A caller cannot tell that from a
hang.
"""

import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from codai.api import embeddings as E

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = io.open(os.path.join(_ROOT, "codai/api/embeddings.py"), encoding="utf-8").read()


# --------------------------------------------------------------------------
# What the entry names is what gets loaded
# --------------------------------------------------------------------------

def _run_embeddings_body():
    """The whole of `_run_embeddings`, not a fixed slice of it.

    A `[:4000]` window broke when a pre-load refusal was added ahead of the
    load: `_load_embedding_locked` moved past the cutoff and the test failed
    with "substring not found" on code that was correct. The window was never
    the thing being asserted — ordering within the function is.
    """
    after = _SRC.split("async def _run_embeddings", 1)[1]
    # Up to the next top-level def, so the body is bounded by structure.
    nxt = re.search(r"\n(?:async def |def )", after)
    return after[:nxt.start()] if nxt else after


def test_the_entry_path_decides_what_is_loaded():
    body = _run_embeddings_body()
    assert "_cfg_path" in body, "the entry's path is never consulted"
    assert body.index("_cfg_path") < body.index("_load_embedding_locked"), (
        "the name is resolved after the load, which is too late to change it")


def test_the_config_lookup_falls_back_to_the_resolver():
    """`self.config` is keyed by the registration id (a local file's full
    path); a request arrives as the basename. A miss drops every per-model
    setting, `mmproj` included."""
    body = _SRC.split("async def _run_embeddings", 1)[1][:4000]
    assert "_config_for_model(model_name)" in body


def test_a_gguf_embedder_is_not_looked_up_on_the_hub():
    """The dispatch test is on the name, so a bare id reaches
    sentence-transformers and becomes a Hub request for
    `sentence-transformers/<id>`."""
    loader = _SRC.split("def _load_embedding_model", 1)[1]
    gguf_at = loader.index("endswith('.gguf')")
    st_at = loader.index("from sentence_transformers import SentenceTransformer")
    assert gguf_at < st_at, (
        "sentence-transformers is tried before the GGUF branch")


# --------------------------------------------------------------------------
# The pre-load refusal
# --------------------------------------------------------------------------

def test_a_gguf_without_an_mmproj_cannot_take_images():
    assert E._may_embed_images("/AI/guffcache/Qwen3-Embedding-4B-Q4_K_M.gguf",
                               {"mmproj": None}) is False


def test_a_gguf_with_an_mmproj_can():
    assert E._may_embed_images(
        "/AI/guffcache/gme-Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
        {"mmproj": "/AI/guffcache/gme-mmproj-Q8_0.gguf"}) is True


def test_a_vit_gguf_can():
    """dinov2 GGUF goes to the dinov2-embed server, which is image-only."""
    assert E._may_embed_images("/AI/guffcache/dinov2-large-Q8_0.gguf", {}) is True


def test_the_coordinate_half_of_geoclip_cannot():
    assert E._may_embed_images("geoclip", {}) is True
    assert E._may_embed_images("geoclip-location", {}) is False


def test_an_unknown_model_is_not_refused():
    """The check is conservative: what it cannot determine goes to the load,
    and the loaded-object check decides. Refusing on a guess would break a
    model that works."""
    assert E._may_embed_images("definitely/not-a-real-model-anywhere", {}) is True


def test_the_refusal_happens_before_the_load():
    body = _SRC.split("async def _run_embeddings", 1)[1][:6000]
    assert "_may_embed_images(" in body, "nothing is checked before loading"
    assert body.index("_may_embed_images(") < body.index("_load_embedding_locked"), (
        "the check runs after the load, which is the forty seconds being fixed")


def test_the_loaded_object_check_is_still_there():
    """The pre-check is conservative, so the exact answer -- from the model
    itself -- has to stay."""
    body = _SRC.split("async def _run_embeddings", 1)[1][:9000]
    assert "_supports_images(model_obj)" in body
    assert body.index("_load_embedding_locked") < body.index("_supports_images(model_obj)")


def test_both_directions_are_refused_with_400():
    """Images to a text-only model, and text to an image-only one. The second
    already answered properly; the first is what sent a server into a
    forty-second load for nothing."""
    assert "is text-only; image embedding needs a" in _SRC
    assert "this embedding model is image-only" in _SRC


# --------------------------------------------------------------------------
# A validation must not cost what it is protecting against
# --------------------------------------------------------------------------

def test_the_precheck_does_not_touch_the_network_or_run_repo_code():
    """Measured while building this: AutoConfig took 20 s on the first call
    (network) and 2.7-5.6 s after that, because `trust_remote_code` executes
    the repo's own config code — to read two plain JSON fields. A validation
    that can hang is not a validation."""
    src = _SRC.split("def _may_embed_images", 1)[1].split("\ndef ", 1)[0]
    # Code only: the comment above the call explains why AutoConfig is NOT
    # used, and naming it there must not fail this.
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "AutoConfig" not in code, (
        "the pre-check builds an AutoConfig, which reaches the network and "
        "runs remote code")
    assert "_local_config_json(" in code


def test_the_config_is_read_from_local_files_only():
    src = _SRC.split("def _local_config_json", 1)[1].split("\ndef ", 1)[0]
    assert "try_to_load_from_cache" in src, "it can fetch from the Hub"
    assert "hf_hub_download" not in src and "snapshot_download" not in src


def test_a_config_that_is_not_here_is_not_judged(monkeypatch):
    monkeypatch.setattr(E, "_local_config_json", lambda name: None)
    assert E._may_embed_images("some/model-we-have-never-seen", {}) is True


def test_a_vision_tower_in_the_config_is_enough(monkeypatch):
    monkeypatch.setattr(E, "_local_config_json",
                        lambda name: {"model_type": "qwen2_vl",
                                      "vision_config": {"depth": 32}})
    assert E._may_embed_images("any/model", {}) is True


def test_a_text_config_is_refused(monkeypatch):
    monkeypatch.setattr(E, "_local_config_json",
                        lambda name: {"model_type": "qwen3",
                                      "architectures": ["Qwen3Model"]})
    assert E._may_embed_images("any/model", {}) is False


def test_a_clip_architecture_without_a_vision_config_still_counts(monkeypatch):
    monkeypatch.setattr(E, "_local_config_json",
                        lambda name: {"model_type": "clip_text_model",
                                      "architectures": ["CLIPModel"]})
    assert E._may_embed_images("any/model", {}) is True
