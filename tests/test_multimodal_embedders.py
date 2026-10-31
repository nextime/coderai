"""Four embedders that take text AND images, in one shared space.

The footage checker needs to score a photo against a text description, and of
everything configured only `gme-Qwen2-VL-2B` could do it: the rest are text-only
(Qwen3-Embedding, bge-m3), image-only (dinov2, salad, vpr) or coordinates
(geoclip-location). So three were added beside a 7B upgrade of the one that
works.

Three of the four need no new embedding code, which is the point of checking
their documented API before writing any:

    gme-Qwen2-VL-7B-Instruct   Qwen2-VL weights -> the existing `qwenvl` backend
    royokong/e5-v              SentenceTransformer.encode(texts | images)
    BAAI/BGE-VL-MLLM-S2        SentenceTransformer.encode({'text':…, 'image':…})

`_embed_images` already routes a sentence_transformers model through
`model.encode(pil_images)` with a `[{'image': im}]` fallback, which is exactly
the two shapes those two models document.

The fourth does need its own backend, because its API resembles nothing else
here:

    model.encode([{'txt': …, 'img': …}], max_length=4096,
                 is_query=True, instruction=…)['hidden_states']

Note what is NOT claimed: coderai embeds texts and images as separate items, so
MM-Embed's combined text+image QUERY form is not reachable through
`/v1/embeddings`. Text-vs-image similarity in one space — what the footage check
needs — works regardless.
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.api import embeddings as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = io.open(os.path.join(ROOT, 'codai/api/embeddings.py'), encoding='utf-8').read()


# ── identification ───────────────────────────────────────────────────────────

@pytest.mark.parametrize('name', ['nvidia/MM-Embed', 'mm-embed', 'MM-Embed',
                                  'mmembed', 'nvidia/mm-embed'])
def test_mmembed_is_recognised(name):
    assert E._is_mmembed(name) is True


@pytest.mark.parametrize('name', ['royokong/e5-v', 'BAAI/BGE-VL-MLLM-S2',
                                  'jinaai/jina-clip-v2', 'bge-m3', '', None])
def test_other_models_are_not_mistaken_for_it(name):
    assert E._is_mmembed(name) is False


# ── it must be allowed to take images ────────────────────────────────────────

def test_it_is_image_capable_before_loading():
    """Otherwise the pre-load guard refuses the pictures it exists to embed."""
    assert E._may_embed_images('nvidia/MM-Embed', {}) is True


def test_it_is_image_capable_after_loading():
    i = SRC.index("'vpr-server', 'mmembed'")
    assert i > 0, 'the loaded-object check would refuse images'


def test_geoclip_is_still_not_in_that_backend_list():
    """Its two modes SHARE the backend name, so listing it there would claim
    that geoclip-location accepts images. It takes coordinates."""
    i = SRC.index("'vpr-server', 'mmembed'")
    line = SRC[i - 200:i + 60]
    assert "'geoclip'" not in line


def test_the_location_half_of_geoclip_is_unaffected():
    assert E._may_embed_images('geoclip-location', {}) is False
    assert E._may_embed_images('geoclip', {}) is True


# ── the new backend is wired on both paths ───────────────────────────────────

def test_text_goes_through_the_backend():
    i = SRC.index("elif backend == 'mmembed':")
    assert "{'txt': t}" in SRC[i:i + 220]


def test_images_go_through_the_backend():
    hits = [i for i in range(len(SRC)) if SRC.startswith("elif backend == 'mmembed':", i)]
    assert len(hits) == 2, f'expected a text and an image dispatch, found {len(hits)}'
    assert "{'img': im}" in SRC[hits[1]:hits[1] + 220]


def test_the_loader_is_dispatched_by_name_before_the_hf_path():
    """It is a logical id with a bespoke API; letting it fall through to
    sentence-transformers would ask the Hub for the wrong thing."""
    i = SRC.index('if _is_mmembed(model_name):')
    j = SRC.index('from sentence_transformers import SentenceTransformer')
    assert i < j


def test_trust_remote_code_is_not_left_to_the_operator():
    """The checkpoint's encode() lives in its repo: without remote code it
    cannot load at all, so a missing flag would be a silent failure."""
    i = SRC.index('def _load_mmembed')
    body = SRC[i:SRC.index('\ndef ', i + 10)]
    assert "'trust_remote_code': True" in body


def test_the_vectors_are_taken_from_hidden_states():
    i = SRC.index('def _mmembed_embed')
    body = SRC[i:SRC.index('\ndef ', i + 10)]
    assert "out['hidden_states']" in body


def test_an_instruction_switches_it_to_query_mode():
    """Without one the model embeds the item as a PASSAGE. That is the right
    default for "embed this" and the wrong one for a query, so the distinction
    is kept rather than hardcoded."""
    i = SRC.index('def _mmembed_embed')
    body = SRC[i:SRC.index('\ndef ', i + 10)]
    assert "kw['is_query'] = True" in body
    assert 'instruction' in body


def test_dimensions_truncation_is_applied():
    i = SRC.index('def _mmembed_embed')
    body = SRC[i:SRC.index('\ndef ', i + 10)]
    assert '_truncate_dims(vecs, dimensions)' in body


def test_it_runs_without_grad():
    i = SRC.index('def _mmembed_embed')
    body = SRC[i:SRC.index('\ndef ', i + 10)]
    assert 'torch.no_grad()' in body


# ── the three that need no new code ──────────────────────────────────────────

def test_sentence_transformers_already_takes_images_both_ways():
    """e5-v documents `encode(images)`; BGE-VL documents
    `encode({'text':…,'image':…})`. Both shapes are already handled."""
    # Anchored inside _embed_images: the same `if backend ==` line also exists
    # in _embed_texts, and matching that one tests the wrong function.
    f = SRC.index('def _embed_images')
    i = SRC.index("if backend == 'sentence_transformers':", f)
    block = SRC[i:i + 700]
    assert 'model.encode(pil_images' in block
    assert "{'image': im}" in block


def test_the_qwenvl_backend_covers_the_7b_by_architecture():
    """gme-7B is Qwen2-VL weights, like the 2B that works, so it needs no new
    path — the detection is on model_type, not on the repo name."""
    assert "_qwen_mt in ('qwen2_vl', 'qwen2_5_vl')" in SRC
