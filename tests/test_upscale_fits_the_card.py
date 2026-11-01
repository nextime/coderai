"""An upscale must not need more memory than the card it was admitted to.

Measured on a 24 GB card 2026-11-01 with stable-diffusion-x4-upscaler:

    500 on POST /v1/images/upscale: 'Upscaling failed: CUDA out of memory.
    Tried to allocate 9.00 GiB. GPU 0 has a total capacity of 23.56 GiB of
    which 4.29 GiB is free.'

Eviction was not at fault — it ran first and reported the 18.1 GB the
catalogue holds for this card as satisfied. The checkpoint is about 1.7 GB;
the rest is the float32 VAE decode at the OUTPUT resolution, which scales
with the picture and not with the weights. Reserving it twice over cannot
help, because the sum exceeds the whole card. Tiling the decode is what
makes the model usable, and it also makes the footprint worth reserving.

The same request asked for scale=2 and got an x4 attempt, because the
pipeline's ratio is fixed and `scale` was dropped on the floor: four times
the memory for a quarter of the pixels wanted.
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import codai.api.images as images

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = io.open(os.path.join(ROOT, 'codai/api/images.py'), encoding='utf-8').read()


def _loader_body():
    i = SRC.index('def _load_upscaler(model_name: str, global_args):')
    return SRC[i:SRC.index('\ndef ', i + 10)]


def test_every_diffusers_upscaler_is_bounded_before_it_is_returned():
    """A pipeline handed back unbounded is cached and stays unbounded: the
    cache is keyed by model id and the load happens once.

    The GAN backend is excluded because it is built tiled already (tile=512
    in _build_realesrgan) and exposes none of these methods; 'pil' is not a
    model at all.
    """
    body = _loader_body()
    returns = [l.strip() for l in body.splitlines()
               if l.strip().startswith('return (')
               and 'pil' not in l and 'realesrgan' not in l]
    assert len(returns) >= 3, ('no diffusers upscaler returns found — did the '
                               'loader move? ' + repr(returns))
    for r in returns:
        assert '_bound_upscaler_activations' in r, r


def test_the_bounding_tiles_the_vae_and_slices_attention():
    i = SRC.index('def _bound_upscaler_activations(pipe):')
    body = SRC[i:SRC.index('\ndef ', i + 10)]
    assert 'enable_vae_tiling' in body
    assert 'enable_attention_slicing' in body
    assert '9.00 GiB' in body and '4.29 GiB' in body, (
        'the measurement is what stops the tiling being removed as needless')


def test_a_missing_tiling_method_is_not_fatal():
    """Not every upscaler class offers all three; a GAN upscaler offers none.
    An upscaler that cannot be tiled must still load."""
    class _Bare:
        pass
    obj = _Bare()
    assert images._bound_upscaler_activations(obj) is obj


def test_the_bounding_returns_the_pipeline_it_was_given():
    """It is used inline in a return statement, so it must pass the object
    through rather than answer None."""
    calls = []

    class _Pipe:
        def enable_vae_tiling(self):
            calls.append('tile')

        def enable_attention_slicing(self):
            calls.append('slice')
    p = _Pipe()
    assert images._bound_upscaler_activations(p) is p
    assert calls == ['tile', 'slice']


# --- the requested scale ---------------------------------------------------

class _Img:
    def __init__(self, size):
        self.size = size
        self.resized_to = None

    def resize(self, size, *a, **kw):
        out = _Img(size)
        self.resized_to = size
        return out


def test_a_fixed_x4_result_is_brought_to_the_requested_x2():
    src = _Img((512, 512))
    out = images._to_requested_scale(_Img((2048, 2048)), src, 2)
    assert out.size == (1024, 1024)


def test_a_result_already_at_the_requested_scale_is_untouched():
    src = _Img((512, 512))
    native = _Img((1024, 1024))
    assert images._to_requested_scale(native, src, 2) is native


def test_no_scale_asked_for_keeps_the_native_result():
    """The field is optional; a caller who omits it wants what the model does."""
    src = _Img((512, 512))
    native = _Img((2048, 2048))
    for s in (None, 0):
        assert images._to_requested_scale(native, src, s) is native


def test_a_non_square_picture_keeps_its_aspect():
    src = _Img((800, 600))
    out = images._to_requested_scale(_Img((3200, 2400)), src, 2)
    assert out.size == (1600, 1200)


def test_both_diffusers_backends_honour_the_scale():
    i = SRC.index('def _run_upscale(upscaler, image_bytes: bytes, scale: int):')
    body = SRC[i:SRC.index('\ndef ', i + 10)]
    for backend in ("backend == 'diffusers_latent'", "backend == 'diffusers'"):
        j = body.index(backend)
        branch = body[j:body.index('return', j) + 80]
        assert '_to_requested_scale' in branch, backend


def test_the_gan_backend_still_scales_through_the_model():
    """Real-ESRGAN takes the factor natively via outscale, which is better
    than resampling after the fact — that branch must not be rerouted."""
    i = SRC.index('def _run_upscale(upscaler, image_bytes: bytes, scale: int):')
    body = SRC[i:SRC.index('\ndef ', i + 10)]
    j = body.index("backend == 'realesrgan'")
    branch = body[j:body.index("backend == 'diffusers_latent'", j)]
    assert 'outscale=scale' in branch
