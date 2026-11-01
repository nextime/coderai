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

"""Where a seeded request's random generator lives.

Sequential CPU offload leaves every module on the META device until its own
forward pulls it in, and a pipeline whose modules are all on meta reports meta
as its device. ``torch.Generator(device=pipeline.device)`` then reached torch's
accelerator hooks with a meta device:

    RuntimeError: META device type not an accelerator.

Measured on Z-Image-Turbo (bf16, sequential_text_encoder=true). The image route
reported it as "Failed to load image model 'Tongyi-MAI/Z-Image-Turbo'" -- naming
the load, which had in fact succeeded (footprint measured, 1.56 GB offloaded) --
and only SEEDED requests failed, so the model looked broken to anyone who
passes a seed and perfectly fine to anyone who does not.

Following ``_execution_device`` fixed the meta crash and left a second one, for
the same reason: that is the device the pipeline COMPUTES on, not necessarily
the one it allocates its initial latents on. diffusers' randn_tensor tolerates
a CPU generator (draw on the host, copy over) and REFUSES a CUDA one whenever
the target is CPU --

    Cannot generate a cpu tensor from a generator of type cuda.

-- measured again on Z-Image-Turbo, 2026-11-01. So the generator is built on
the host unconditionally, which also makes a seed reproduce the same image
whether the model is resident, offloaded or sharded.
"""

import io
import os

import torch

from codai.models.utils import generator_device

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel):
    return io.open(os.path.join(_ROOT, rel), encoding="utf-8").read()


class _Pipe:
    def __init__(self, device=None, execution_device=None):
        if device is not None:
            self.device = torch.device(device)
        if execution_device is not None:
            self._execution_device = torch.device(execution_device)


def test_a_meta_device_is_never_handed_to_a_generator():
    """The exact failure: every module offloaded, so .device is meta."""
    dev = generator_device(_Pipe(device="meta"))
    assert dev.type != "meta"
    # And it is usable, which is the only thing that matters here.
    torch.Generator(device=dev).manual_seed(1234)


def test_an_offloaded_pipeline_gets_a_host_generator():
    """_execution_device no longer decides. It is where the pipeline COMPUTES,
    and an offloaded pipeline allocates its first latents elsewhere -- which is
    how the cuda-generator/cpu-tensor refusal got in after the meta fix."""
    dev = generator_device(_Pipe(device="meta", execution_device="cpu"))
    assert dev == torch.device("cpu")


def test_a_cuda_pipeline_still_gets_a_host_generator():
    """THE new guarantee, and the one that would be lost by "optimising" this
    back to the pipeline's device. A CUDA generator is fatal the moment the
    pipeline wants a CPU tensor; a CPU generator is never fatal, because
    randn_tensor draws on the host and copies.

        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"          # fine
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(...)        # the measured failure
    """
    assert generator_device(_Pipe(device="cuda")) == torch.device("cpu")
    assert generator_device(
        _Pipe(device="cuda", execution_device="cuda")) == torch.device("cpu")


def test_the_device_is_always_one_a_generator_can_be_built_on():
    """Whatever the pipeline claims -- including nothing at all."""
    for pipe in (_Pipe(device="meta"), _Pipe(device="cuda"), _Pipe(device="cpu"),
                 object()):
        torch.Generator(device=generator_device(pipe)).manual_seed(7)


def test_a_seed_no_longer_depends_on_where_the_model_sits():
    """Two placements of the same model must draw the same noise: that is what
    a seed is for, and it was not true while the generator followed the
    pipeline."""
    a = generator_device(_Pipe(device="cuda"))
    b = generator_device(_Pipe(device="meta", execution_device="cpu"))
    assert a == b
    first = torch.Generator(device=a).manual_seed(42)
    second = torch.Generator(device=b).manual_seed(42)
    assert torch.equal(torch.randn(8, generator=first),
                       torch.randn(8, generator=second))


def test_an_ordinary_pipeline_is_left_exactly_alone():
    assert generator_device(_Pipe(device="cpu")) == torch.device("cpu")


def test_a_pipeline_with_no_device_still_yields_one():
    dev = generator_device(object())
    assert dev.type in ("cuda", "cpu")
    torch.Generator(device=dev).manual_seed(1)


def test_a_nonsense_device_does_not_raise():
    class _Bad:
        device = "not-a-device"
    dev = generator_device(_Bad())
    assert dev.type in ("cuda", "cpu")


def test_a_lazy_device_is_refused_too():
    """Same class of device as meta: no accelerator hooks behind it."""
    class _Lazy:
        device = torch.device("lazy")
    assert generator_device(_Lazy()).type != "lazy"


def test_every_pipeline_generator_goes_through_it():
    """Four sites constructed a Generator from a pipeline's own device: the
    text-to-image path, img2img, edit, and audio generation. One of them being
    fixed would leave the same failure reachable by three other routes."""
    for rel in ("codai/api/images.py", "codai/api/audio_gen.py"):
        src = _src(rel)
        assert "torch.Generator(device=pipeline.device)" not in src
        assert "torch.Generator(device=pipe.device)" not in src
        assert "torch.Generator(pipe.device)" not in src
    assert _src("codai/api/images.py").count(
        "model_utils.generator_device(") == 3
    assert _src("codai/api/audio_gen.py").count(
        "model_utils.generator_device(") == 1
