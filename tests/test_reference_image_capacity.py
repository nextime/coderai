"""How many reference images an edit may send is declared, never guessed.

Measured 2026-11-01 on zeiss. Krea-2-Edit, two images, prompt "put the man
from picture 2 next to the woman":

    Sizes of tensors must match except in dimension 1. Expected size 1 but
    got size 0 for tensor number 1 in the list.
    "POST /v1/images/edits HTTP/1.1" 500 Internal Server Error

The edit path classes a pipeline as instruction-editing from its signature
(takes ``image``, has no ``strength``) and handed both images over. Krea-2
cannot use two, and said so as an internal shape error.

It cannot be detected. Every diffusers edit pipeline annotates ``image`` as
accepting ``list[PIL.Image.Image]`` -- verified against Flux2Klein,
FluxKontext, FluxKontextInpaint, ChronoEdit and BriaFiboEdit -- but for some
that list is several references of different subjects and for others a batch
of independent sources. So the card declares it, and a card that has not
declared it is read as single-reference and refused with a 400 rather than
being allowed to fail inside the pipeline.
"""
import pytest
from fastapi import HTTPException

import codai.api.images as images


def test_an_undeclared_card_reads_as_one_reference():
    assert images._declared_max_reference_images({}) is None
    assert images._declared_max_reference_images(None) is None


def test_a_declared_cap_is_read():
    assert images._declared_max_reference_images({"max_reference_images": 10}) == 10


def test_a_nonsense_cap_is_ignored_not_trusted():
    assert images._declared_max_reference_images(
        {"max_reference_images": "lots"}) is None
    assert images._declared_max_reference_images(
        {"max_reference_images": 0}) is None


def test_one_reference_is_always_allowed():
    """Whatever the card says, a plain single-image edit must not be refused."""
    images._check_reference_image_count(1, "anything", {})
    images._check_reference_image_count(0, "anything", {})


def test_an_undeclared_card_with_no_pipeline_is_refused():
    with pytest.raises(HTTPException) as e:
        images._check_reference_image_count(2, "Krea-2-Edit", {})
    assert e.value.status_code == 400
    assert "max_reference_images" in e.value.detail, (
        "the message must say how to fix it, not just that it is wrong"
    )


def test_an_undeclared_card_defers_until_the_pipeline_can_be_asked():
    """The pre-load check must not refuse what the pipeline may well support."""
    images._check_reference_image_count(2, "anything", {}, allow_defer=True)


def test_a_declared_card_allows_up_to_its_cap():
    cfg = {"max_reference_images": 3}
    images._check_reference_image_count(3, "Qwen-Image-Edit-2511", cfg)
    with pytest.raises(HTTPException) as e:
        images._check_reference_image_count(4, "Qwen-Image-Edit-2511", cfg)
    assert e.value.status_code == 400
    assert "at most 3" in e.value.detail


def test_the_refusal_happens_before_the_model_is_loaded():
    import inspect
    src = inspect.getsource(images.create_image_edit)
    i_check = src.index("_check_reference_image_count")
    i_load = src.index("request_model")
    assert i_check < i_load, (
        "refusing a request is not worth loading the model for first"
    )


def test_the_edit_headroom_sizes_the_requested_card():
    import inspect
    src = inspect.getsource(images.create_image_edit)
    seg = src[src.index("_evict_headroom_gb"):]
    assert "requested_id=request.model" in seg[:600], (
        "siblings share a checkpoint; without requested_id this sized "
        "Krea-2-Turbo (9.7 GB) for a resident Krea-2-RealDream (17.22 GB)"
    )


# --------------------------------------------------------------------------- #
#  the pipeline's own code is the source of truth when the card is silent
# --------------------------------------------------------------------------- #

def test_the_detector_reads_real_diffusers_pipelines():
    """Pinned against the installed diffusers, so an upstream change is caught.

    These three are the evidence that the question is answerable at all:
    identical signatures and identical docstrings, different behaviour.
    """
    diffusers = pytest.importorskip("diffusers")
    expected = {
        "Flux2KleinPipeline": "many",
        "QwenImageEditPlusPipeline": "many",
        "FluxKontextPipeline": "one",
    }
    seen = {}
    for name, want in expected.items():
        cls = getattr(diffusers, name, None)
        if cls is None:
            continue
        seen[name] = images._pipeline_reference_semantics(cls)
        assert seen[name] == want, f"{name}: got {seen[name]!r}, want {want!r}"
    assert seen, "none of the reference pipelines were importable"


def test_a_multi_reference_pipeline_is_allowed_without_a_declaration():
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "Flux2KleinPipeline", None)
    if cls is None:
        pytest.skip("Flux2KleinPipeline not in this diffusers")
    images._check_reference_image_count(3, "FLUX.2-klein-4B", {}, pipeline=cls)


def test_a_single_reference_pipeline_is_refused_with_its_reason():
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "FluxKontextPipeline", None)
    if cls is None:
        pytest.skip("FluxKontextPipeline not in this diffusers")
    with pytest.raises(HTTPException) as e:
        images._check_reference_image_count(3, "kontext", {}, pipeline=cls)
    assert e.value.status_code == 400
    assert "ignores the rest" in e.value.detail, (
        "the caller must learn that the extra references would be discarded, "
        "not merely that the count is wrong"
    )


def test_a_declaration_overrides_the_detector():
    """A source read is evidence, not a guarantee; the operator wins."""
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "Flux2KleinPipeline", None)
    if cls is None:
        pytest.skip("Flux2KleinPipeline not in this diffusers")
    with pytest.raises(HTTPException) as e:
        images._check_reference_image_count(
            3, "FLUX.2-klein-4B", {"max_reference_images": 2}, pipeline=cls)
    assert "at most 2" in e.value.detail


def test_an_unreadable_pipeline_is_refused_not_assumed():
    class _Opaque:
        __call__ = None
    assert images._pipeline_reference_semantics(_Opaque()) is None
    with pytest.raises(HTTPException):
        images._check_reference_image_count(2, "x", {}, pipeline=_Opaque())
