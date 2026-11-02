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

"""The edit pipeline must accept pre-computed embeddings, or its encoder
cannot be freed before the denoiser runs.

Ground truth, zeiss, a 24 GiB card with nothing else resident, 2026-11-02:

    06:00:54  Edit needs about 2.0 GB of working room and the card has less
              — evicting idle models (keeping image:Krea-2-RealDream)
    06:00:54  [text-encoder-offload] Krea2EditPipeline cannot be driven from
              embeddings (encode_prompt=True, text_encoder=True,
              prompt_embeds=False) — skipped
    06:00:55  torch.OutOfMemoryError: Tried to allocate 48.00 MiB. GPU 0 has a
              total capacity of 23.56 GiB of which 52.00 MiB is free
    06:00:55  503 on POST /v1/images/edits

The OOM is inside ``encode_prompt``: the encoder ran with the whole denoiser
resident because the offload had declined one second earlier. It declined for
one reason only — ``text_encoder_offload.supported`` asks the pipeline's own
signature for ``prompt_embeds``, and this pipeline did not have it. The
generation path of the same checkpoint has done the offload all along and logs
"3.0 GiB back for the denoiser".

So the contract under test is the signature itself, and then that supplied
embeddings are USED rather than silently re-encoded.
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.models import text_encoder_offload as T


torch = pytest.importorskip("torch")


def _pipeline_class():
    """The real ``Krea2EditPipeline``, built against a stand-in base.

    The module defines the real class only when diffusers exports
    ``Krea2Pipeline``, and falls back to a placeholder whose ``__init__``
    raises when it does not. The interpreter the suite runs under is one of
    those builds (diffusers 0.37.0.dev0 here, the container's venv has the
    Krea 2 one), so asking for the class directly would skip every test on
    the very box where the regression happened — and a permanently skipped
    test is not a test.

    Nothing in the section under examination touches the base class: the text
    section reads ``self.encode_prompt``, ``self.do_classifier_free_guidance``
    and the geometry attributes, all of which the fake pipeline below
    supplies. So the base is stood in for and the module reloaded, which
    yields the SAME ``__call__`` the container runs.
    """
    import importlib
    import diffusers

    from codai.pipelines import krea2_edit as K

    if getattr(diffusers, "Krea2Pipeline", None) is not None:
        return K.Krea2EditPipeline        # a build that really has it

    class _StandInBase:
        pass

    class _StandInOutput:
        def __init__(self, images=None):
            self.images = images

    had = {k: getattr(diffusers, k, None)
           for k in ("Krea2Pipeline", "Krea2PipelineOutput")}
    diffusers.Krea2Pipeline = _StandInBase
    diffusers.Krea2PipelineOutput = _StandInOutput
    try:
        reloaded = importlib.reload(K)
        cls = reloaded.Krea2EditPipeline
        assert "__call__" in cls.__dict__, (
            "the stand-in base did not produce the real class")
        return cls
    finally:
        for k, v in had.items():
            if v is None:
                delattr(diffusers, k)
            else:
                setattr(diffusers, k, v)


# ---------------------------------------------------------------------------
# the signature, which is what the offload reads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "prompt_embeds",
    "prompt_embeds_mask",
    "negative_prompt_embeds",
    "negative_prompt_embeds_mask",
])
def test_the_edit_call_accepts_the_embedding_arguments(name):
    params = set(inspect.signature(_pipeline_class().__call__).parameters)
    assert name in params, (
        f"{name} missing: text_encoder_offload reads this signature, so the "
        f"encoder stays resident through every denoising step without it")


def test_the_offload_helper_considers_the_edit_pipeline_supported():
    """supported() is the exact gate that refused on 2026-11-02."""
    cls = _pipeline_class()

    class _Fake:
        __call__ = cls.__call__
        text_encoder = object()

        def encode_prompt(self, **kw):
            raise AssertionError("not called by supported()")

    assert T.supported(_Fake()) is True


def test_the_mask_argument_is_the_one_krea_names():
    """_mask_arg picks prompt_embeds_mask over prompt_attention_mask; naming
    the LTX one here would hand Krea a keyword it does not take."""
    params = set(inspect.signature(_pipeline_class().__call__).parameters)
    assert T._mask_arg(params) == "prompt_embeds_mask"


# ---------------------------------------------------------------------------
# and that the embeddings are honoured once accepted
# ---------------------------------------------------------------------------

class _Recorder:
    """Enough of the pipeline for __call__ to reach the text section.

    The geometry, VAE and transformer are not under test: the run is stopped
    as soon as the text section has decided, which is the only part this
    change touches.
    """

    class _Stop(Exception):
        pass

    def __init__(self, cfg=False):
        self.encode_calls = []
        self._cfg = cfg
        self.vae_scale_factor = 8
        self.patch_size = 2
        self.seen = {}

        class _T:
            dtype = torch.float32

            class config:
                in_channels = 16
                is_distilled = True
        self.transformer = _T()
        self.text_encoder = object()

    _execution_device = torch.device("cpu")

    @property
    def do_classifier_free_guidance(self):
        return self._cfg

    def encode_prompt(self, prompt=None, device=None, num_images_per_prompt=1,
                      max_sequence_length=512):
        self.encode_calls.append(prompt)
        return (torch.zeros(num_images_per_prompt, 4, 8),
                torch.ones(num_images_per_prompt, 4, dtype=torch.long))

    def prepare_latents(self, *a, **k):
        # The text section is done by the time the canvas is prepared.
        raise _Recorder._Stop()


def _run(rec, **kwargs):
    cls = _pipeline_class()
    kwargs.setdefault("image", object())
    kwargs.setdefault("height", 64)
    kwargs.setdefault("width", 64)
    with pytest.raises(_Recorder._Stop):
        cls.__call__(rec, **kwargs)


def test_supplied_embeddings_are_not_re_encoded():
    """The point of the offload: the encoder is on the CPU by now, so calling
    it here is the device error the helper documents, not a slow path."""
    rec = _Recorder()
    _run(rec, prompt_embeds=torch.zeros(1, 4, 8),
         prompt_embeds_mask=torch.ones(1, 4, dtype=torch.long))
    assert rec.encode_calls == []


def test_without_embeddings_the_pipeline_still_encodes_its_own_prompt():
    rec = _Recorder()
    _run(rec, prompt="make it snow")
    assert rec.encode_calls == ["make it snow"]


def test_embeddings_without_their_mask_are_refused():
    """Krea 2 attends over a padded sequence: unmasked padding reads as
    prompt, so a caller that forgets the mask is told, not guessed at."""
    rec = _Recorder()
    cls = _pipeline_class()
    with pytest.raises(ValueError, match="prompt_embeds_mask"):
        cls.__call__(rec, image=object(), height=64, width=64,
                     prompt_embeds=torch.zeros(1, 4, 8))


def test_negative_embeddings_without_their_mask_are_refused():
    rec = _Recorder(cfg=True)
    cls = _pipeline_class()
    with pytest.raises(ValueError, match="negative_prompt_embeds_mask"):
        cls.__call__(rec, image=object(), height=64, width=64,
                     prompt_embeds=torch.zeros(1, 4, 8),
                     prompt_embeds_mask=torch.ones(1, 4, dtype=torch.long),
                     negative_prompt_embeds=torch.zeros(1, 4, 8))


def test_with_guidance_a_supplied_negative_is_not_re_encoded():
    rec = _Recorder(cfg=True)
    _run(rec, prompt_embeds=torch.zeros(1, 4, 8),
         prompt_embeds_mask=torch.ones(1, 4, dtype=torch.long),
         negative_prompt_embeds=torch.zeros(1, 4, 8),
         negative_prompt_embeds_mask=torch.ones(1, 4, dtype=torch.long))
    assert rec.encode_calls == []


def test_with_guidance_a_missing_negative_is_encoded_here():
    """The helper supplies one for this family, but a direct caller may not,
    and running CFG against a negative of None is a crash."""
    rec = _Recorder(cfg=True)
    _run(rec, prompt_embeds=torch.zeros(1, 4, 8),
         prompt_embeds_mask=torch.ones(1, 4, dtype=torch.long))
    assert rec.encode_calls == [""]


def test_without_guidance_no_negative_is_encoded():
    rec = _Recorder(cfg=False)
    _run(rec, prompt="x")
    assert rec.encode_calls == ["x"]


# ---------------------------------------------------------------------------
# n > 1: the batch the helper cannot know about
# ---------------------------------------------------------------------------

def test_embeddings_encoded_once_are_repeated_for_n_images():
    """encode_then_free calls encode_prompt WITHOUT num_images_per_prompt, so
    a batch of one arrives for a batch of n latents. encoder_hidden_states and
    hidden_states go into the transformer together and must agree."""
    rec = _Recorder()
    captured = {}

    def _prepare(batch, *a, **k):
        captured["latent_batch"] = batch
        captured["embeds"] = rec.seen.get("prompt_embeds")
        raise _Recorder._Stop()

    cls = _pipeline_class()
    # Read what the text section produced by stopping at prepare_latents and
    # reaching into the frame it was called from.
    import sys as _sys

    def _spy(batch, *a, **k):
        captured["latent_batch"] = batch
        frame = _sys._getframe(1)
        captured["pe"] = frame.f_locals.get("prompt_embeds")
        captured["mask"] = frame.f_locals.get("prompt_embeds_mask")
        raise _Recorder._Stop()

    rec.prepare_latents = _spy
    with pytest.raises(_Recorder._Stop):
        cls.__call__(rec, image=object(), height=64, width=64,
                     num_images_per_prompt=3,
                     prompt_embeds=torch.zeros(1, 4, 8),
                     prompt_embeds_mask=torch.ones(1, 4, dtype=torch.long))
    assert captured["latent_batch"] == 3
    assert captured["pe"].shape[0] == 3
    assert captured["mask"].shape[0] == 3


def test_a_batch_that_already_matches_is_left_alone():
    rec = _Recorder()
    captured = {}
    import sys as _sys

    def _spy(batch, *a, **k):
        captured["pe"] = _sys._getframe(1).f_locals.get("prompt_embeds")
        raise _Recorder._Stop()

    rec.prepare_latents = _spy
    given = torch.zeros(2, 4, 8)
    cls = _pipeline_class()
    with pytest.raises(_Recorder._Stop):
        cls.__call__(rec, image=object(), height=64, width=64,
                     num_images_per_prompt=2,
                     prompt_embeds=given,
                     prompt_embeds_mask=torch.ones(2, 4, dtype=torch.long))
    assert captured["pe"] is given
