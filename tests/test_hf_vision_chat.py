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

"""A model advertised as able to read images must read them, or refuse.

Measured on 2026-10-29. ``Qwen/Qwen3.5-9B`` was listed with capabilities
``["text_generation", "image_to_text"]`` and answered every photograph with
"a blank white image with no visible content" -- nine web pictures of one
video, all of them real images (the footage scorer rated them 0.53-0.59).

Two defects, one symptom:

* the engine loaded the checkpoint with ``AutoModelForCausalLM``, which for
  ``Qwen3_5Config`` resolves to ``Qwen3_5ForCausalLM`` -- the text decoder on
  its own. ``AutoModelForImageTextToText`` resolves to
  ``Qwen3_5ForConditionalGeneration``, the one that holds the vision tower.
* with no projector loaded, the chat path flattened the picture to the literal
  text ``[image_url content]`` and answered that. The model was right; the
  prompt was blank.

So: load the tower when the checkpoint has one, and where it has none, refuse
the request instead of answering it blind.
"""

import ast
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel):
    return io.open(os.path.join(_ROOT, rel), encoding="utf-8").read()


# --------------------------------------------------------------------------
# The request shaping, which is pure and can be exercised directly
# --------------------------------------------------------------------------

def _png_data_uri(size=(8, 8), colour=(10, 20, 30)):
    import base64
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_an_image_part_is_recognised_in_any_of_its_spellings():
    from codai.backends import hf_vision
    for spelling in ("image_url", "input_image", "image"):
        msgs = [{"role": "user", "content": [{"type": spelling, "image_url": {"url": "x"}}]}]
        assert hf_vision.has_images(msgs), spelling
    assert not hf_vision.has_images(
        [{"role": "user", "content": [{"type": "text", "text": "no picture here"}]}])
    assert not hf_vision.has_images([{"role": "user", "content": "plain string"}])


def test_a_data_uri_image_is_decoded_to_rgb():
    from codai.backends import hf_vision
    img = hf_vision.load_image(_png_data_uri(size=(12, 7)))
    assert img.size == (12, 7)
    assert img.mode == "RGB", "a palette/RGBA image reaches the tower with the wrong channels"


def test_a_corrupt_data_uri_is_an_error_not_a_blank_image():
    from codai.backends import hf_vision
    try:
        hf_vision.load_image("data:image/png;base64,!!!not base64!!!")
    except Exception:
        return
    raise AssertionError("a broken image decoded to something, which would be "
                         "answered as though it were a picture")


def test_images_are_handed_over_in_the_order_they_appear():
    """The processor fills placeholders from a list, so order IS the binding.
    Two pictures swapped means the answer describes the wrong one."""
    from codai.backends import hf_vision
    a = _png_data_uri(size=(4, 4))
    b = _png_data_uri(size=(9, 9))
    msgs = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": a}},
                                     {"type": "text", "text": "and"},
                                     {"type": "image_url", "image_url": {"url": b}}]},
    ]
    out, images = hf_vision.to_processor_messages(msgs)
    assert [i.size for i in images] == [(4, 4), (9, 9)]
    kinds = [p["type"] for p in out[0]["content"]]
    assert kinds == ["image", "text", "image"], kinds
    # markers carry no pixels on the two-step path: the pictures travel in the
    # separate list, and a marker that also held one would be counted twice
    assert all(set(p) == {"type"} for p in out[0]["content"] if p["type"] == "image")


def test_the_one_step_fallback_puts_each_picture_back_in_its_marker():
    from codai.backends import hf_vision
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _png_data_uri()}},
                                         {"type": "image_url", "image_url": {"url": _png_data_uri((5, 5))}}]}]
    out, images = hf_vision.to_processor_messages(msgs)
    embedded = hf_vision._embed_images(out, images)
    got = [p.get("image") for p in embedded[0]["content"]]
    assert got == images, "the fallback template would render markers with no pixels"


def test_a_plain_text_turn_survives_the_reshaping():
    from codai.backends import hf_vision
    out, images = hf_vision.to_processor_messages(
        [{"role": "system", "content": "be brief"},
         {"role": "user", "content": [{"type": "text", "text": "hello"}]}])
    assert images == []
    assert out[0] == {"role": "system", "content": [{"type": "text", "text": "be brief"}]}
    assert out[1]["content"] == [{"type": "text", "text": "hello"}]


def test_a_processor_without_an_image_processor_is_not_vision_capable():
    """transformers returns a tokenizer-shaped 'processor' for some text
    checkpoints. It passes every duck-type check until the first picture."""
    from codai.backends import hf_vision

    class _TokenizerOnly:
        pass

    import transformers
    _real = transformers.AutoProcessor
    try:
        class _Auto:
            @staticmethod
            def from_pretrained(*a, **k):
                return _TokenizerOnly()
        transformers.AutoProcessor = _Auto
        assert hf_vision.load_processor("some/text-model") is None
    finally:
        transformers.AutoProcessor = _real


# --------------------------------------------------------------------------
# The load path
# --------------------------------------------------------------------------

def test_no_load_path_asks_for_the_text_class_directly():
    """Every from_pretrained in the backend has to go through the resolver;
    one forgotten call site is one model that silently loses its tower."""
    src = _src("codai/backends/cuda.py")
    assert "AutoModelForCausalLM.from_pretrained" not in src, (
        "a load path still hardcodes the text-only class")


def test_the_resolver_falls_back_to_text_on_any_failure():
    from codai.backends import hf_vision
    assert hf_vision.auto_model_class("this/does-not-exist-at-all")[1] is False
    assert hf_vision.is_vision_checkpoint("this/does-not-exist-at-all") is False


def test_vision_is_only_claimed_when_both_halves_loaded():
    """`supports_vision` is what the API layer trusts to pass an image
    through. A tower with no processor, or a processor with no tower, still
    cannot answer about a picture."""
    src = _src("codai/backends/cuda.py")
    body = src.split("def _init_vision", 1)[1].split("\n    def ", 1)[0]
    assert "self.supports_vision = False" in body.split("try:", 1)[0], (
        "_init_vision does not start from 'cannot see'")
    assert body.index("load_processor") < body.index("self.supports_vision = True"), (
        "vision is claimed before the processor is known to exist")
    assert "is_vision_checkpoint" in body


def test_unloading_forgets_the_processor():
    src = _src("codai/backends/cuda.py")
    assert "self.supports_vision = False" in src.split("del self.tokenizer", 1)[1][:400], (
        "a swap would leave the next model claiming the old one's vision")


def test_the_vision_path_does_not_reuse_a_text_prefix_cache():
    """Image tokens are expanded from the pixels by the processor, so a KV
    prefix cached from another request would be reused against the wrong
    picture. The vision path must not touch the prefix cache at all."""
    src = _src("codai/backends/cuda.py")
    body = src.split("def _generate_vision_chat", 1)[1].split("\n    async def ", 1)[0]
    assert "_reuse_or_seed_prefix" not in body
    assert "past_key_values" not in body


def test_both_chat_entry_points_take_the_vision_branch():
    src = _src("codai/backends/cuda.py")
    tree = ast.parse(src)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name in ("generate_chat", "generate_chat_stream"):
            found[node.name] = any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "_vision_request" for n in ast.walk(node))
    assert found.get("generate_chat"), "non-streaming chat ignores images"
    assert found.get("generate_chat_stream"), "streaming chat ignores images"


# --------------------------------------------------------------------------
# The refusal
# --------------------------------------------------------------------------

def test_an_image_to_a_blind_model_is_refused_before_it_is_flattened():
    src = _src("codai/api/text.py")
    guard = src.index("if not _vision_ok:")
    flatten = src.index("[{item.get('type', 'unknown')} content]")
    assert guard < flatten, (
        "the placeholder is built before the refusal, so a blind model can "
        "still answer a picture")
    # The whole guard block, not a fixed slice: a slice breaks the moment the
    # message grows, which says nothing about the behaviour under test.
    body = src[guard:src.index("# Inject system prompt", guard)]
    assert "status_code=400" in body
    assert "_release_instance()" in body, (
        "refusing without releasing the instance slot leaks it")
    assert "vision_report(" in body, (
        "the refusal does not say what was actually found, which is how a "
        "broken vision loader looked like a misconfigured model")


def test_a_gguf_without_an_mmproj_does_not_advertise_vision():
    src = _src("codai/models/manager.py")
    block = src.split('resolved_type = model_type or', 1)[1][:1600]
    assert "caps.image_to_text = False" in block
    assert '.gguf' in block and 'mmproj' in block
    # a remote/pod-served model is the remote's business, not this listing's
    assert "vllm" in block and "runpod" in block


# --------------------------------------------------------------------------
# The mapping call itself
# --------------------------------------------------------------------------
#
# Shipped broken and found in production the same day: transformers'
# `_LazyAutoMapping.get` is NOT dict.get — its signature is
# `get(self, key, default)` with `default` REQUIRED. Called with one argument
# it raises
#
#   TypeError: _LazyAutoMapping.get() missing 1 required positional argument
#
# which this module caught and reported as "no vision class" — for every
# checkpoint, always. So every vision model loaded as text and then had its
# pictures refused by the guard added for exactly that case.
#
# The probe that "verified" it passed two arguments, so it tested a call the
# code does not make. These tests call what the code calls.


def test_the_mapping_is_queried_the_way_transformers_defines_it():
    """A stub whose `get` mirrors _LazyAutoMapping's required `default`."""
    import transformers
    from codai.backends import hf_vision

    class _Cfg:
        vision_config = {"depth": 2}
        image_token_id = 7

    class _Cls:
        pass

    class _StrictMapping:
        def get(self, key, default):          # default is REQUIRED, as upstream
            return {_Cfg: _Cls}.get(key, default)

    class _Auto:
        _model_mapping = _StrictMapping()

    real = transformers.AutoModelForImageTextToText
    try:
        transformers.AutoModelForImageTextToText = _Auto
        report = hf_vision.vision_report("x/y", config=_Cfg())
    finally:
        transformers.AutoModelForImageTextToText = real
    assert report["cls"] is _Cls, report["reason"]


def test_a_real_installed_vision_config_resolves_to_a_class():
    """No stub, no network, no weights: a config class this transformers
    actually ships, through the real mapping."""
    import transformers
    from codai.backends import hf_vision

    cfg = None
    for name in ("LlavaConfig", "Qwen2VLConfig", "Blip2Config", "PaliGemmaConfig"):
        klass = getattr(transformers, name, None)
        if klass is None:
            continue
        try:
            cfg = klass()
        except Exception:
            continue
        break
    if cfg is None:
        pytest.skip("this transformers ships none of the probe configs")
    report = hf_vision.vision_report("x/y", config=cfg)
    assert report["cls"] is not None, report["reason"]


def test_a_failure_to_decide_is_never_silent():
    """The 400 that refuses a picture was the only trace of this. A decision
    that consequential has to leave one of its own."""
    src = _src("codai/backends/hf_vision.py")
    rep = src.split("def vision_report", 1)[1].split("\ndef ", 1)[0]
    assert '"reason"' in rep
    auto = src.split("def auto_model_class", 1)[1].split("\ndef ", 1)[0]
    assert "print(" in auto, (
        "a vision checkpoint loaded as text says nothing in the log")
