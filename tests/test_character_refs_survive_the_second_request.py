"""An attached IP-Adapter must survive the next request on a cached pipeline.

Measured 2026-11-01 on zeiss. pornmaster-pro-pony, character_profiles=
['motus_test_char']: the first request loaded the adapter and generated; the
second, on the same cached pipeline, returned 500 with

    AttributeError: 'tuple' object has no attribute 'shape'
      hidden_states.shape if encoder_hidden_states is None
                          else encoder_hidden_states.shape

Mechanism: diffusers' ``Attention.set_attention_slice()`` REPLACES the
processor on every attention module, so the per-request
``enable_attention_slicing(slice_size="auto")`` wiped all 72
IPAdapterAttnProcessors. ``_ensure_ip_adapter_loaded`` then saw
``_coderai_ip_state == 'loaded'`` and returned early without reinstalling
them, while the UNet config still carried
``encoder_hid_dim_type='ip_image_proj'`` -- which makes diffusers hand the
processor a (prompt_embeds, image_embeds) tuple only an IP-Adapter processor
can unpack. So character consistency worked once per pipeline and never again.
"""
import inspect

import codai.api.images as images


class _Proc:
    pass


class _IPProc:
    def __init__(self):
        self.to_k_ip = object()
        self.to_v_ip = object()


class _Denoiser:
    def __init__(self, procs):
        self.attn_processors = procs


class _Pipe:
    def __init__(self, procs):
        self.unet = _Denoiser(procs)


def test_installed_processors_are_detected():
    assert images._ip_processors_installed(_Pipe({"a": _IPProc()})) is True


def test_wiped_processors_are_detected():
    assert images._ip_processors_installed(_Pipe({"a": _Proc()})) is False


def test_a_pipeline_without_a_denoiser_is_not_claimed():
    class _Bare:
        pass
    assert images._ip_processors_installed(_Bare()) is False


def test_slicing_is_not_re_enabled_while_an_adapter_is_attached():
    src = inspect.getsource(images)
    i = src.index('pipeline.enable_attention_slicing(slice_size="auto")')
    window = src[i - 400:i + 120]
    assert "_ip_attached" in window, (
        "the per-request memory block must not enable attention slicing on a "
        "pipeline that carries an IP-Adapter; doing so replaces its attention "
        "processors"
    )
    assert "not _ip_attached" in window


def test_the_loader_rechecks_the_processors_before_trusting_the_flag():
    src = inspect.getsource(images._ensure_ip_adapter_loaded)
    head = src[:src.index("if flag == 'unsupported'")]
    assert "_ip_processors_installed" in head, (
        "_coderai_ip_state records that we loaded an adapter, not that its "
        "processors are still installed; returning True on the flag alone is "
        "what produced the 500"
    )
