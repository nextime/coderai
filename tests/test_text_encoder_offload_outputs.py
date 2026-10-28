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

"""The embeddings must be paired with the right mask.

``encode_prompt`` returns a bare tuple whose order differs per family and is
not discoverable from the signature. Read from the wrong slot, the pipeline
gets a negative-embeddings tensor where its attention mask belongs.

Measured on a live LTX-2.3 render: the request failed with "Must provide
`prompt_attention_mask` when specifying `prompt_embeds`" because the code
assumed (embeds, neg_embeds, mask, neg_mask) while diffusers returns
(embeds, mask, neg_embeds, neg_mask). The orders asserted here are copied
from the installed diffusers sources, so a family added later has to state
its own.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from codai.models import text_encoder_offload as teo


def _embeds(n=3):
    return torch.zeros((1, n, 8), dtype=torch.float32)


def _mask(n=3):
    return torch.ones((1, n), dtype=torch.int64)


LTX_PARAMS = {"prompt_embeds", "prompt_attention_mask",
              "negative_prompt_embeds", "negative_prompt_attention_mask"}
KREA_PARAMS = {"prompt_embeds", "prompt_embeds_mask",
               "negative_prompt_embeds", "negative_prompt_embeds_mask"}


def test_ltx2_order_is_embeds_mask_negembeds_negmask():
    """diffusers/pipelines/ltx2/pipeline_ltx2.py returns exactly this."""
    pe, pm, ne, nm = _embeds(), _mask(), _embeds(5), _mask(5)
    got = teo._assign_outputs((pe, pm, ne, nm), LTX_PARAMS,
                              "prompt_attention_mask")
    assert got["prompt_embeds"] is pe
    assert got["prompt_attention_mask"] is pm, \
        "the mask must come from slot 1, not slot 2"
    assert got["negative_prompt_embeds"] is ne
    assert got["negative_prompt_attention_mask"] is nm


def test_krea2_order_is_embeds_mask():
    """diffusers/pipelines/krea2/pipeline_krea2.py returns a 2-tuple."""
    pe, pm = _embeds(), _mask()
    got = teo._assign_outputs((pe, pm), KREA_PARAMS, "prompt_embeds_mask")
    assert got["prompt_embeds"] is pe
    assert got["prompt_embeds_mask"] is pm
    assert "negative_prompt_embeds" not in got


def test_a_family_with_no_mask_is_not_handed_embeddings_as_one():
    """Two float tensors = positive and negative embeddings, no mask."""
    pe, ne = _embeds(), _embeds(5)
    got = teo._assign_outputs((pe, ne), LTX_PARAMS, "prompt_attention_mask")
    assert got["prompt_embeds"] is pe
    assert got.get("negative_prompt_embeds") is ne
    assert "prompt_attention_mask" not in got, \
        "a float tensor was misfiled as an attention mask"


def test_none_entries_are_skipped():
    """LTX returns None negatives when guidance is off."""
    pe, pm = _embeds(), _mask()
    got = teo._assign_outputs((pe, pm, None, None), LTX_PARAMS,
                              "prompt_attention_mask")
    assert got["prompt_attention_mask"] is pm
    assert "negative_prompt_embeds" not in got


def test_unsupported_kwargs_are_not_passed_on():
    """A name the pipeline does not accept must not reach the call."""
    pe, pm = _embeds(), _mask()
    got = teo._assign_outputs((pe, pm), {"prompt_embeds"},
                              "prompt_embeds_mask")
    assert got == {"prompt_embeds": pe}


def test_missing_mask_aborts_rather_than_failing_the_request():
    """If the mask cannot be paired, generate the ordinary way instead of
    handing the pipeline embeddings it will reject."""

    class _Pipe:
        text_encoder = object()

        def __call__(self, prompt=None, prompt_embeds=None,
                     prompt_attention_mask=None):
            raise AssertionError("must not be called")

        def encode_prompt(self, prompt=None, negative_prompt=None):
            # A family that returns embeddings only -- no mask to pair.
            return (_embeds(),)

    pipe = _Pipe()
    kwargs = {"prompt": "a cat"}
    out, freed = teo.encode_then_free(pipe, kwargs)
    assert out is kwargs or out == kwargs, \
        "the original kwargs must come back untouched"
    assert freed == 0.0


def test_sdxl_shape_is_named_in_full_not_partially():
    """SDXL returns (embeds, neg_embeds, pooled, neg_pooled) -- four floats and
    no mask. Naming only the first two and dropping the pooled pair makes the
    pipeline raise "Must provide pooled_prompt_embeds", so the whole shape has
    to be named or nothing is."""
    sdxl_params = {"prompt_embeds", "negative_prompt_embeds",
                   "pooled_prompt_embeds", "negative_pooled_prompt_embeds"}
    pe, ne, po, npo = _embeds(), _embeds(5), _embeds(1), _embeds(1)
    got = teo._assign_outputs((pe, ne, po, npo), sdxl_params, None)
    assert got["prompt_embeds"] is pe
    assert got["negative_prompt_embeds"] is ne
    assert got["pooled_prompt_embeds"] is po
    assert got["negative_pooled_prompt_embeds"] is npo


def test_an_unnameable_shape_refuses_rather_than_dropping_tensors():
    """A family returning more than we can name must abandon the optimisation,
    not hand the pipeline an incomplete set it will reject."""
    many = tuple(_embeds() for _ in range(7))
    with pytest.raises(ValueError, match="could be named"):
        teo._assign_outputs(many, {"prompt_embeds"}, None)


def test_an_unnameable_shape_falls_back_to_the_ordinary_path():
    """End to end: the request still generates, via prompt rather than
    embeddings."""

    class _Pipe:
        text_encoder = object()

        def __call__(self, prompt=None, prompt_embeds=None):
            raise AssertionError("must not be called by the helper")

        def encode_prompt(self, prompt=None):
            return tuple(_embeds() for _ in range(7))

    kwargs = {"prompt": "a cat"}
    out, freed = teo.encode_then_free(_Pipe(), kwargs)
    assert out == kwargs, "the prompt must survive for the ordinary path"
    assert "prompt_embeds" not in out
    assert freed == 0.0


class _KreaLike:
    """A separate-negative family: encode_prompt has no `negative_prompt`, so
    the negative needs its own call. Mirrors Krea 2, whose __call__ otherwise
    encodes "" itself at denoise time -- against an encoder we just moved."""

    text_encoder = object()

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt=None, prompt_embeds=None, prompt_embeds_mask=None,
                 negative_prompt=None, negative_prompt_embeds=None,
                 negative_prompt_embeds_mask=None):
        raise AssertionError("must not be called by the helper")

    def encode_prompt(self, prompt=None, device=None, max_sequence_length=None):
        self.prompts.append(prompt)
        return _embeds(), _mask()


def test_separate_negative_family_gets_its_negative_encoded():
    pipe = _KreaLike()
    out, _ = teo.encode_then_free(pipe, {"prompt": "a cat",
                                         "negative_prompt": "blurry"})
    assert pipe.prompts == ["a cat", "blurry"], \
        f"the negative prompt was not encoded: {pipe.prompts}"
    assert "negative_prompt_embeds" in out
    assert "negative_prompt_embeds_mask" in out
    assert "negative_prompt" not in out and "prompt" not in out


def test_separate_negative_family_encodes_empty_when_none_given():
    """With no negative supplied the pipeline would encode "" itself, at
    denoise time, on an encoder that is no longer on the GPU."""
    pipe = _KreaLike()
    out, _ = teo.encode_then_free(pipe, {"prompt": "a cat"})
    assert pipe.prompts == ["a cat", ""], \
        f"empty negative was not encoded up front: {pipe.prompts}"
    assert out.get("negative_prompt_embeds") is not None


def test_both_in_one_family_is_not_called_twice():
    """LTX-2 encodes both in a single call; a second one would be waste."""

    calls = []

    class _LtxLike:
        text_encoder = object()

        def __call__(self, prompt=None, prompt_embeds=None,
                     prompt_attention_mask=None,
                     negative_prompt_embeds=None,
                     negative_prompt_attention_mask=None):
            raise AssertionError("must not be called by the helper")

        def encode_prompt(self, prompt=None, negative_prompt=None,
                          do_classifier_free_guidance=True, device=None):
            calls.append(prompt)
            return _embeds(), _mask(), _embeds(5), _mask(5)

    out, _ = teo.encode_then_free(_LtxLike(), {"prompt": "a cat",
                                               "negative_prompt": "blurry",
                                               "guidance_scale": 3.0})
    assert calls == ["a cat"], f"encoded more than once: {calls}"
    assert out["negative_prompt_attention_mask"] is not None


class _StatefulPipe:
    """A cached pipeline that really refuses to encode off-device.

    Models the production failure: the pipeline is cached between requests, so
    the previous request's offload has left the encoder on the CPU, and
    encode_prompt indexes cuda ids against cpu weights.
    """

    def __init__(self):
        self.device = "cuda"          # where the encoder currently is
        self._execution_device = "cuda"
        self.encodes = 0

        class _Enc:
            def __init__(self, outer):
                self._outer = outer

            def parameters(self):
                class _P:
                    pass
                p = _P()
                p.device = type("D", (), {"type": self._outer.device})()
                return iter([p])

            def to(self, dev):
                self._outer.device = "cuda" if "cuda" in str(dev) else "cpu"
                return self

        self.text_encoder = _Enc(self)

    def __call__(self, prompt=None, prompt_embeds=None,
                 prompt_attention_mask=None, negative_prompt_embeds=None,
                 negative_prompt_attention_mask=None):
        raise AssertionError("must not be called by the helper")

    def encode_prompt(self, prompt=None, negative_prompt=None,
                      do_classifier_free_guidance=True, device=None):
        if self.device != "cuda":
            raise RuntimeError(
                "Expected all tensors to be on the same device, but got index "
                "is on cuda:0, different from other tensors on cpu")
        self.encodes += 1
        return _embeds(), _mask(), _embeds(5), _mask(5)


def test_a_second_request_on_a_cached_pipeline_still_works():
    """The offload is a per-request CYCLE. Without bringing the encoder back,
    request 2 raised a device mismatch, the helper fell back to "the ordinary
    way", and the pipeline's OWN encode_prompt then hit the same mismatch --
    so the fallback produced a 500 rather than a slow path."""
    pipe = _StatefulPipe()

    out1, freed1 = teo.encode_then_free(pipe, {"prompt": "first"})
    assert "prompt_embeds" in out1, "request 1 should use the offload"
    assert pipe.device == "cpu", "the encoder should have been moved off"

    out2, _ = teo.encode_then_free(pipe, {"prompt": "second"})
    assert "prompt_embeds" in out2, (
        "request 2 fell back instead of bringing the encoder back")
    assert pipe.encodes == 2, "the second request never encoded"


def test_a_bail_out_leaves_the_encoder_where_the_pipeline_can_use_it():
    """Whatever goes wrong, the ordinary path must still be able to generate:
    it calls the same encoder."""

    class _BadShape(_StatefulPipe):
        def encode_prompt(self, prompt=None, negative_prompt=None,
                          do_classifier_free_guidance=True, device=None):
            if self.device != "cuda":
                raise RuntimeError("Expected all tensors to be on the same device")
            # More tensors than can be named -> the helper abandons the attempt.
            return tuple(_embeds() for _ in range(7))

    pipe = _BadShape()
    pipe.device = "cpu"               # as a previous request left it
    out, freed = teo.encode_then_free(pipe, {"prompt": "x"})
    assert out == {"prompt": "x"}, "the ordinary path needs its prompt back"
    assert freed == 0.0
    assert pipe.device == "cuda", (
        "the encoder was left on the CPU, so the ordinary path will 500 too")
