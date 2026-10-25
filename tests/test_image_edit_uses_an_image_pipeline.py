"""`/v1/images/edits` has to load a pipeline that can take an image.

Reported from a client on 2026-10-25 against `nesaorg/PonyRealism-Uncensored`
as "the edit result is garbage", "`size` is ignored" and a 500 — three
symptoms of two defects, neither of which announced itself.

**The edit was never an edit.** The img2img loader picked its pipeline class by
looking for substrings in the model's name — 'xl', 'sdxl', 'stable-diffusion',
'sd' — and that name contains none of them, so it fell through to the generic
`DiffusionPipeline`. That faithfully honours the checkpoint's own
`model_index.json`, which for this model declares `StableDiffusionXLPipeline`:
the **text-to-image** class. Every diffusers `__call__` ends in `**kwargs` (it
carries the deprecated `callback` pair), so `image=` and `strength=` were
accepted, silently dropped, and the pipeline rendered a fresh picture from the
prompt alone at its default 1024x1024. Hence "garbage" (a prompt saying "add X,
keep everything else" rendered from scratch), hence "`size` ignored" (1024x1024
is the text-to-image default, not the source's shape), and hence 200 OK
throughout.

**The 500 was a key used as a name.** These endpoints keep their task-specific
pipelines in `MultiModelManager.models` under `img2img:<repo>` and
`inpaint:<repo>`. The manager's fuzzy "is this already loaded?" pass matched
one of those keys by its last path segment and then assigned the whole key to
`resolved_name`, so the endpoint went on to ask HuggingFace for a repo called
`img2img:nesaorg/PonyRealism-Uncensored` — "Repo id must use alphanumeric
chars" — on a model that loads perfectly well.
"""
import inspect

import pytest

from codai.api import images as images_mod


# --------------------------------------------------------------------------- #
#  a pipeline that cannot take an image
# --------------------------------------------------------------------------- #

def _with_signature_of(func):
    """An object whose ``__call__`` is ``func``, for signature checks only."""
    return type("Pipe", (), {"__call__": func})()


def test_a_text2img_pipeline_is_recognised_as_unable_to_edit():
    from diffusers import StableDiffusionXLPipeline
    pipe = _with_signature_of(StableDiffusionXLPipeline.__call__)
    assert images_mod._accepts_image(pipe) is False


@pytest.mark.parametrize("cls_name", [
    "StableDiffusionXLImg2ImgPipeline",
    "StableDiffusionXLInpaintPipeline",
    "StableDiffusionImg2ImgPipeline",
])
def test_an_image_pipeline_is_recognised_as_able(cls_name):
    import diffusers
    cls = getattr(diffusers, cls_name)
    assert images_mod._accepts_image(_with_signature_of(cls.__call__)) is True


def test_kwargs_is_why_this_failed_silently():
    """The reason the wrong class answered 200 instead of raising: `image` and
    `strength` land in **kwargs and are dropped, not rejected."""
    from diffusers import StableDiffusionXLPipeline
    params = inspect.signature(StableDiffusionXLPipeline.__call__).parameters
    assert "image" not in params and "strength" not in params
    assert any(p.kind is p.VAR_KEYWORD for p in params.values()), \
        "without **kwargs this would have been a TypeError and never shipped"


def test_a_pipeline_with_nothing_to_judge_is_given_the_benefit_of_the_doubt():
    """A ``(*args, **kwargs)`` signature carries no parameter list, so "image
    is not in it" means nothing — refusing on that would turn an odd but
    working pipeline into a 501."""
    class Odd:
        __call__ = staticmethod(lambda *a, **k: None)
    assert images_mod._accepts_image(Odd()) is True


def test_a_real_parameter_list_without_image_is_a_refusal():
    """Whereas a signature that names its inputs and does not name `image` is
    evidence, and the only safe reading of it is "cannot edit"."""
    class Text2Img:
        def __call__(self, prompt=None, height=None, width=None, **kwargs):
            return None
    assert images_mod._accepts_image(Text2Img()) is False


# --------------------------------------------------------------------------- #
#  the class comes from the checkpoint, not from its name
# --------------------------------------------------------------------------- #

def test_the_loaders_no_longer_guess_from_the_model_name():
    """The regression in one line: a name with no 'xl' in it got a
    text-to-image pipeline."""
    src = open("codai/api/images.py").read()
    head = src.index("def _load_img2img_pipeline")
    body = src[head:head + 2500]
    assert "AutoPipelineForImage2Image" in body
    for guess in ("'xl' in name_lower", "'sdxl' in name_lower",
                  "'sd' in name_lower"):
        assert guess not in body, f"still guessing: {guess}"

    head = src.index("def _load_inpaint_pipeline")
    body = src[head:head + 2000]
    assert "AutoPipelineForInpainting" in body
    assert "'xl' in n" not in body


def test_the_reported_checkpoint_maps_to_an_img2img_class():
    """What AutoPipelineFor* does that the name heuristics could not: read the
    declared class and swap in the one for this task."""
    from diffusers.pipelines.auto_pipeline import (
        AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
        AUTO_INPAINT_PIPELINES_MAPPING,
        _get_task_class,
    )
    declared = "StableDiffusionXLPipeline"      # PonyRealism's model_index.json
    assert _get_task_class(AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, declared).__name__ \
        == "StableDiffusionXLImg2ImgPipeline"
    assert _get_task_class(AUTO_INPAINT_PIPELINES_MAPPING, declared).__name__ \
        == "StableDiffusionXLInpaintPipeline"


def test_conversion_reuses_the_weights_rather_than_loading_them_twice():
    """A text-to-image pipeline already in VRAM is recovered with from_pipe,
    which swaps the task class around the same components."""
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting
    assert hasattr(AutoPipelineForImage2Image, "from_pipe")
    assert hasattr(AutoPipelineForInpainting, "from_pipe")
    src = open("codai/api/images.py").read()
    assert "from_pipe" in src


def test_an_unconvertible_pipeline_is_refused_not_served():
    """501, not a 200 carrying a picture of something else."""
    src = open("codai/api/images.py").read()
    assert "cannot \nedit an image" in src or "cannot " in src
    assert src.count("status_code=501") >= 2, \
        "both edits and inpaint must refuse rather than silently generate"
    assert "has no `image` input" in src


@pytest.mark.parametrize("endpoint", ["create_image_edit", "create_image_inpaint"])
def test_the_guard_runs_on_a_cached_pipeline_too(endpoint):
    """A pipeline loaded by an earlier version is already in the manager's
    models dict, so checking only after a fresh load would leave every running
    install serving text-to-image until it restarted."""
    src = open("codai/api/images.py").read()
    head = src.index(f"async def {endpoint}")
    body = src[head:head + 4000]
    guard = body.index("_accepts_image(pipe)")
    cached_lookup = body.index("multi_model_manager.models.get(")
    assert guard > cached_lookup
    # and it is not nested inside the `if pipe is None:` load branch
    line_start = body.rindex("\n", 0, guard) + 1
    indent = len(body[line_start:guard]) - len(body[line_start:guard].lstrip())
    assert indent <= 4, "the guard is inside the load branch, so a cached " \
                        "text-to-image pipeline would still be used"


# --------------------------------------------------------------------------- #
#  a task key is not a model name
# --------------------------------------------------------------------------- #

def test_the_task_namespaces_are_declared():
    from codai.models.manager import _TASK_NAMESPACES
    assert "img2img:" in _TASK_NAMESPACES
    assert "inpaint:" in _TASK_NAMESPACES


def test_every_namespace_the_endpoints_invent_is_declared():
    """The invariant that keeps this from coming back: if a future endpoint
    keys a pipeline by task, name resolution has to know to skip it, or that
    key becomes a repo id again."""
    import re
    from codai.models.manager import _TASK_NAMESPACES
    src = open("codai/api/images.py").read()
    used = set(re.findall(r'f"([a-z0-9_]+):\{model_name\}"', src))
    # The manager owns these and strips them itself during resolution; every
    # other prefix is one an endpoint made up for a task.
    manager_own = {"image:", "audio:", "tts:", "vision:", "video:",
                   "audio_gen:", "embedding:", "spatial:", "text:"}
    invented = {f"{u}:" for u in used} - manager_own
    assert invented, "expected the edit endpoints to key pipelines by task"
    missing = invented - set(_TASK_NAMESPACES)
    assert not missing, f"not declared in _TASK_NAMESPACES: {sorted(missing)}"


def test_name_resolution_skips_a_foreign_task_key():
    """The 500, as the guard that prevents it: the fuzzy already-loaded pass
    must leave img2img:/inpaint: keys alone, because it assigns the matched key
    to resolved_name and that value goes to from_pretrained."""
    src = open("codai/models/manager.py").read()
    head = src.index("Fuzzy fallback:")
    body = src[head:head + 2200]
    assert "_TASK_NAMESPACES" in body
    # the skip has to come before the match, not after it
    assert body.index("_TASK_NAMESPACES") < body.index("resolved_name = k_name")
    assert "continue" in body[body.index("_TASK_NAMESPACES"):
                              body.index("resolved_name = k_name")]
