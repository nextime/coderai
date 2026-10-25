"""The image edit endpoints, executed rather than read.

Written after shipping `NameError: name 'model_key' is not defined` into
`/v1/images/edits` in 0.4.94 — a line defining it was removed with the load
block it belonged to, while a later line still used it. Editing was broken for
every request, instantly, and a client reported it from production.

There were 187 tests over this code at the time and not one of them caught it,
because every one of them read the source as text: "is this guard present",
"does this call come before that one". Structural tests cannot see an undefined
name on the first line of the request path. These tests call the endpoint
functions with their collaborators stubbed, so the body runs.

The stubs are deliberately thin. The point is not to verify diffusion — it is
to execute every line between the request arriving and the pipeline being
called, which is where wiring breaks.
"""
import asyncio
import types

import pytest
from PIL import Image

from codai.api import images as images_mod


class _RecordingPipe:
    """An img2img-shaped pipeline that records how it was called."""

    def __init__(self):
        self.calls = []

    def __call__(self, image=None, strength=None, prompt=None, **kw):
        self.calls.append(dict(image=image, strength=strength, prompt=prompt, **kw))
        return types.SimpleNamespace(images=[Image.new("RGB", (8, 8))])


@pytest.fixture
def wired(monkeypatch):
    """Every collaborator of the edit endpoints, stubbed to a known answer."""
    pipe = _RecordingPipe()
    source = Image.new("RGB", (472, 1024), (20, 40, 60))

    monkeypatch.setattr(images_mod, "_progress_loading", lambda *a, **k: None)
    monkeypatch.setattr(images_mod, "_decode_b64_image", lambda ref: source.copy())
    monkeypatch.setattr(images_mod, "_image_bytes", lambda *a, **k: b"", raising=False)
    monkeypatch.setattr(images_mod, "save_image_response",
                        lambda img, fmt, req: {"url": "/v1/files/out.png"})
    # one model, already loaded, so nothing tries to reach HuggingFace
    monkeypatch.setattr(
        images_mod.multi_model_manager, "request_model",
        lambda *a, **k: {"model_name": "org/Model", "model_key": "image:org/Model",
                         "model_object": pipe, "config": {}, "already_loaded": True})
    monkeypatch.setattr(images_mod, "_pipeline_for_task",
                        _async(lambda info, task: pipe))
    monkeypatch.setattr(images_mod, "global_args", types.SimpleNamespace(), raising=False)
    return types.SimpleNamespace(pipe=pipe, source=source)


def _async(fn):
    async def inner(*a, **k):
        return fn(*a, **k)
    return inner


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
#  /v1/images/edits
# --------------------------------------------------------------------------- #

def test_an_edit_request_runs_to_the_pipeline(wired):
    """The regression: this raised NameError before reaching the pipeline."""
    req = images_mod.ImageEditRequest(
        model="PonyRealism", prompt="score_9, golden sunset sky",
        image="/v1/files/up-abc.jpg", strength=0.5, size="576x1024",
        clip_skip=2, negative_prompt="score_4")
    out = _run(images_mod.create_image_edit(req, None))

    assert out["data"] == [{"url": "/v1/files/out.png"}]
    assert len(wired.pipe.calls) == 1
    call = wired.pipe.calls[0]
    # the SOURCE IMAGE reached the pipeline, resized to the size asked for
    assert call["image"] is not None
    assert call["image"].size == (576, 1024)
    assert call["strength"] == 0.5


def test_the_source_keeps_its_shape_when_no_size_is_asked_for(wired):
    req = images_mod.ImageEditRequest(
        model="m", prompt="p", image="/v1/files/up-abc.jpg")
    _run(images_mod.create_image_edit(req, None))
    w, h = wired.pipe.calls[0]["image"].size
    # 472x1024 kept on the 64-pixel grid at ~1 MP, not squashed to a square
    assert w != h
    assert abs((w / h) - (472 / 1024)) < 0.08


def test_a_malformed_size_is_a_400_from_the_endpoint(wired):
    from fastapi import HTTPException
    req = images_mod.ImageEditRequest(
        model="m", prompt="p", image="/v1/files/up-abc.jpg", size="1k x 1k")
    with pytest.raises(HTTPException) as exc:
        _run(images_mod.create_image_edit(req, None))
    assert exc.value.status_code == 400


def test_a_missing_image_is_a_400(wired):
    from fastapi import HTTPException
    req = images_mod.ImageEditRequest(model="m", prompt="p", image="")
    with pytest.raises(HTTPException) as exc:
        _run(images_mod.create_image_edit(req, None))
    assert exc.value.status_code == 400


def test_the_prompt_and_negative_both_reach_the_pipeline(wired):
    """Either as text or as embeddings -- the stub pipeline is not a CLIP pair,
    so this takes the plain-text path."""
    req = images_mod.ImageEditRequest(
        model="m", prompt="a cat", image="/v1/files/up-abc.jpg",
        negative_prompt="blurry")
    _run(images_mod.create_image_edit(req, None))
    call = wired.pipe.calls[0]
    assert call.get("prompt") == "a cat"
    assert call.get("negative_prompt") == "blurry"


def test_a_model_prefix_is_applied_through_the_endpoint(wired, monkeypatch):
    """End to end rather than through the helper: the prefix has to survive
    every layer between the request and the pipeline call."""
    monkeypatch.setattr(
        images_mod.multi_model_manager, "config",
        {"image:org/Model": {"prompt_prefix": "score_9, score_8_up"}})
    req = images_mod.ImageEditRequest(
        model="m", prompt="1girl", image="/v1/files/up-abc.jpg")
    _run(images_mod.create_image_edit(req, None))
    assert wired.pipe.calls[0]["prompt"] == "score_9, score_8_up, 1girl"


# --------------------------------------------------------------------------- #
#  /v1/images/inpaint
# --------------------------------------------------------------------------- #

def test_an_inpaint_request_runs_to_the_pipeline(wired):
    req = images_mod.ImageInpaintRequest(
        model="m", prompt="p", image="/v1/files/up-a.png",
        mask="/v1/files/up-b.png", negative_prompt="score_4")
    out = _run(images_mod.create_image_inpaint(req, None))
    assert out["data"] == [{"url": "/v1/files/out.png"}]
    call = wired.pipe.calls[0]
    assert call["image"] is not None
    assert call["mask_image"] is not None
    # the mask follows the image's shape, or it would not line up with it
    assert call["mask_image"].size == call["image"].size


def test_inpaint_without_a_mask_is_a_400(wired):
    from fastapi import HTTPException
    req = images_mod.ImageInpaintRequest(
        model="m", prompt="p", image="/v1/files/up-a.png", mask="")
    with pytest.raises(HTTPException) as exc:
        _run(images_mod.create_image_inpaint(req, None))
    assert exc.value.status_code == 400


# --------------------------------------------------------------------------- #
#  the lesson, as a check
# --------------------------------------------------------------------------- #

def test_every_edit_endpoint_has_a_test_that_executes_it():
    """A guard against the habit that let this through: these endpoints are
    the request path, and a test that only greps their source cannot tell
    whether they run at all."""
    executed = {"create_image_edit", "create_image_inpaint"}
    src = open(__file__).read()
    for name in executed:
        assert f"images_mod.{name}(" in src, f"{name} is never called here"
