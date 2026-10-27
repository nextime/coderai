"""A 2-tuple from encode_prompt is not always (embeds, negative).

Krea 2 and the Qwen-Image family return ``(prompt_embeds, prompt_embeds_mask)``
— the second element is a mask, not a negative prompt. Handing it over as
``negative_prompt_embeds`` makes the pipeline answer

    If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be
    passed. Make sure to generate `prompt_embeds_mask` from the same text
    encoder that was used to generate `prompt_embeds`.

which reads like a missing argument rather than a mislabelled one, and it
arrives wrapped in "Failed to load image model", minutes into a 7.4B load.
"""
import inspect

from codai.api import images as images_mod


def _source():
    return inspect.getsource(images_mod)


def test_the_mask_families_are_told_apart_by_signature_not_by_name():
    """A name list would have to grow for every new family; the pipelines that
    need a mask say so in their own __call__ signature."""
    src = _source()
    assert "'prompt_embeds_mask' in _call_params" in src
    # And the discriminator comes from the pipeline itself.
    assert "_inspect.signature(pipeline.__call__).parameters" in src


def test_a_mask_pipeline_falls_back_to_its_own_encoder():
    """Weighted 75-token windows are a CLIP-pair feature and mean nothing to
    these pipelines, so the right answer is to pass the raw prompt."""
    src = _source()
    i = src.index("'prompt_embeds_mask' in _call_params")
    branch = src[i:i + 1200]
    assert "embed_kwargs = {}" in branch
    # The SD 1.x reading must still exist, just after it.
    assert "elif len(enc) == 2:" in branch
    assert "'negative_prompt_embeds': enc[1]," in branch


def test_sd_one_x_still_gets_its_negative_embeds():
    """The ordinary pair is untouched: enc[1] is a negative there."""
    src = _source()
    sd = src.index("# SD 1.x: (prompt_embeds, negative_prompt_embeds)")
    after = src[sd:sd + 300]
    assert "'prompt_embeds': enc[0]," in after
    assert "'negative_prompt_embeds': enc[1]," in after
