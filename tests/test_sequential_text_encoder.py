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

"""C5: encode the prompt, then give the card back before denoising.

A text encoder runs ONCE per request and is dead weight for every step after,
yet keeps its VRAM: 7.5 GiB of Z-Image's 19.2 GiB pipeline (measured on the
cached snapshot), ~6.5 GiB of 4-bit Gemma-3-12B in front of LTX-2.3's 15 GiB
transformer. That sum is why PinkCherry declares 24.0 GB on a 24 GiB card and
every LTX render was pushed to CPU offload.

It existed only as a side effect before -- diffusers' enable_model_cpu_offload
happens to move each component out after use -- so the resident path had no
equivalent.

The embedding argument names differ per family with no common base class to
ask, so they are discovered from the signature: Krea 2 takes
prompt_embeds + prompt_embeds_mask, LTX-2 takes prompt_embeds +
prompt_attention_mask.
"""

import inspect

import pytest

from codai.models import text_encoder_offload as T


class _T:
    """A stand-in for a returned tensor.

    It carries a dtype because that is how the embeddings are told apart from
    the attention mask -- the ORDER differs per family and cannot be assumed.
    A fake that returned bare strings could not exercise that at all, which is
    how the real tuple order went unchecked: this file's _LtxLike used to
    return (embeds, neg_embeds, mask, neg_mask), the live pipeline returns
    (embeds, mask, neg_embeds, neg_mask), and the live render failed with
    "Must provide `prompt_attention_mask` when specifying `prompt_embeds`".
    """

    class _D:
        def __init__(self, f):
            self.is_floating_point = f

    def __init__(self, tag, mask=False):
        self.tag = tag
        self.dtype = _T._D(not mask)

    def __eq__(self, other):
        return isinstance(other, str) and other == self.tag or self is other

    def __repr__(self):
        return f"<{self.tag}>"


class _KreaLike:
    """Two-call family: the negative prompt needs its own encode."""
    text_encoder = object()

    def __init__(self):
        self.calls = []
        self.moved = []

    def encode_prompt(self, prompt, device=None, max_sequence_length=512):
        self.calls.append(prompt)
        return (_T(f"E:{prompt}"), _T(f"M:{prompt}", mask=True))

    def __call__(self, prompt=None, negative_prompt=None, prompt_embeds=None,
                 prompt_embeds_mask=None, negative_prompt_embeds=None,
                 negative_prompt_embeds_mask=None, num_inference_steps=8):
        pass


class _LtxLike:
    """One-call family: encodes both prompts together."""
    text_encoder = object()

    def __init__(self):
        self.calls = []

    def encode_prompt(self, prompt, negative_prompt=None,
                      do_classifier_free_guidance=False,
                      num_videos_per_prompt=1):
        self.calls.append((prompt, negative_prompt,
                           do_classifier_free_guidance))
        # The real order, copied from diffusers' pipeline_ltx2.py:
        #   prompt_embeds, prompt_attention_mask,
        #   negative_prompt_embeds, negative_prompt_attention_mask
        return (_T("E"), _T("M", mask=True),
                _T("NE"), _T("NM", mask=True))

    def __call__(self, prompt=None, negative_prompt=None, prompt_embeds=None,
                 prompt_attention_mask=None, negative_prompt_embeds=None,
                 negative_prompt_attention_mask=None, guidance_scale=1.0):
        pass


# ------------------------------------------------------------- what qualifies

def test_a_pipeline_with_no_encoder_is_left_alone():
    class _P:
        def __call__(self, prompt=None):
            pass
    assert T.supported(_P()) is False


def test_a_pipeline_that_cannot_take_embeddings_is_left_alone():
    """Freeing the encoder without a way to pass embeddings back would mean
    generating from an empty prompt."""
    class _P:
        text_encoder = object()

        def encode_prompt(self, prompt):
            return ("E",)

        def __call__(self, prompt=None):
            pass
    assert T.supported(_P()) is False


def test_both_real_families_qualify():
    assert T.supported(_KreaLike()) is True
    assert T.supported(_LtxLike()) is True


# ------------------------------------------------- the two encode_prompt shapes

def test_the_two_call_family_encodes_the_negative_separately():
    p = _KreaLike()
    kw, _ = T.encode_then_free(p, {"prompt": "a fox", "negative_prompt": "blur",
                                   "num_inference_steps": 8})
    assert p.calls == ["a fox", "blur"], p.calls
    assert "prompt" not in kw and "negative_prompt" not in kw
    assert kw["prompt_embeds"] == "E:a fox"
    assert kw["prompt_embeds_mask"] == "M:a fox"
    assert kw["negative_prompt_embeds"] == "E:blur"
    assert kw["negative_prompt_embeds_mask"] == "M:blur"
    assert kw["num_inference_steps"] == 8, "unrelated kwargs must survive"


def test_the_one_call_family_encodes_both_together():
    p = _LtxLike()
    kw, _ = T.encode_then_free(p, {"prompt": "a sea", "negative_prompt": "ugly",
                                   "guidance_scale": 3.0})
    assert len(p.calls) == 1, "the negative was encoded twice"
    assert p.calls[0] == ("a sea", "ugly", True), p.calls
    assert kw["prompt_embeds"] == "E"
    assert kw["negative_prompt_embeds"] == "NE"
    assert kw["prompt_attention_mask"] == "M"
    assert kw["negative_prompt_attention_mask"] == "NM"


def test_cfg_is_derived_from_the_guidance_scale():
    """At guidance 1.0 there is no classifier-free guidance to encode for."""
    p = _LtxLike()
    T.encode_then_free(p, {"prompt": "x", "guidance_scale": 1.0})
    assert p.calls[0][2] is False


# ------------------------------------------------------------------- failures

def test_a_failing_encode_leaves_the_request_untouched():
    """A VRAM optimisation must never be why a generation fails."""
    class _P:
        text_encoder = object()

        def encode_prompt(self, prompt, device=None):
            raise RuntimeError("encoder exploded")

        def __call__(self, prompt=None, prompt_embeds=None):
            pass

    original = {"prompt": "a fox", "num_inference_steps": 8}
    kw, freed = T.encode_then_free(_P(), dict(original))
    assert kw == original, "the request was mangled by a failed optimisation"
    assert freed == 0.0


def test_no_prompt_means_nothing_to_do():
    p = _KreaLike()
    kw, _ = T.encode_then_free(p, {"num_inference_steps": 8})
    assert p.calls == [] and "prompt_embeds" not in kw


# ------------------------------------------------------------- the wiring

def test_both_generation_paths_consult_the_per_model_flag():
    from codai.api import images, video
    isrc = inspect.getsource(images._generate_with_diffusers)
    assert "sequential_text_encoder" in isrc and "encode_then_free" in isrc
    vsrc = inspect.getsource(video._generate_video)
    assert "sequential_text_encoder" in vsrc and "encode_then_free" in vsrc


def test_the_setting_is_configurable_from_the_model_page():
    import io, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    routes = io.open(os.path.join(root, "codai/admin/routes.py"),
                     encoding="utf-8").read()
    assert '"sequential_text_encoder"' in routes, (
        "the setting cannot be saved from the model page")
    html = io.open(os.path.join(root, "codai/admin/templates/models.html"),
                   encoding="utf-8").read()
    assert "cfg-seqtext" in html
    assert "sequential_text_encoder:" in html, "the page never sends it"


def test_it_is_opt_in():
    """Freeing the encoder costs a host-RAM copy; it should not be imposed on
    models that already fit."""
    from codai.api import images
    src = inspect.getsource(images._generate_with_diffusers)
    assert "(model_config or {}).get('sequential_text_encoder')" in src, (
        "the offload is no longer gated on the per-model flag")
