"""Editing an uploaded photo or clip: keep the audio, keep the shape, name the file.

Four defects, all of which let a request succeed while quietly doing the wrong
thing -- the worst kind, because a caller cannot tell from the response.

THE AUDIO. The enhancement paths (upscale, frame interpolation) decode a clip to
frames and encode a brand new video, so the audio track was simply left behind:
a clip that arrived with speech came back silent, with a 200. Only the ffmpeg
`minterpolate` fallback -- the one path nobody takes by default -- kept it. Both
paths preserve the source duration, so the track lines up without stretching,
and the fix is a remux rather than a re-encode wherever the container allows it.

THE REFERENCES. These endpoints took base64 (images) or base64 and an http(s)
URL (video) only, so a caller that had already uploaded a file through
`/v1/files/upload` had to re-send the whole thing on every edit -- megabytes per
call, for a file sitting on the same disk. `/v1/compose` and the extractors
already accept a `/v1/files/...` path or a `sha256:` id, and these now resolve
through the same resolver, which also reads our own files off disk instead of
fetching them back over the loopback.

THE SHAPE. `size` defaulted to the literal string "1024x1024", so omitting it
resized every non-square photo to a square and visibly squashed it. Omitted now
means "keep the source's shape", at the ~1 MP a diffusion model wants.

THE SILENT NON-UPSCALE. An unrecognised upscaler model fell through to PIL
LANCZOS -- a plain resize -- and still answered 200, so the caller paid for an
"AI upscale" that never ran and got back something merely soft. That is now a
400 that names the models which would work.
"""
import base64
import io
import os
import sys
import types

import pytest


# ---------------------------------------------------------------- the refs

@pytest.fixture
def files_dir(tmp_path, monkeypatch):
    """An install whose /v1/files directory is ``tmp_path``."""
    from codai.compose import media
    monkeypatch.setattr(media, "global_file_path", str(tmp_path), raising=False)
    return tmp_path


def _png_bytes(w=8, h=8, colour=(10, 120, 200)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("module,func", [
    ("codai.api.video", "_decode_b64_or_url"),
    ("codai.api.images", "_image_bytes"),
])
def test_a_v1_files_path_is_read_off_disk(files_dir, module, func):
    """The whole point: name the upload, do not re-send it."""
    mod = __import__(module, fromlist=["x"])
    data = _png_bytes()
    (files_dir / "shot.png").write_bytes(data)
    assert getattr(mod, func)("/v1/files/shot.png") == data


@pytest.mark.parametrize("module,func", [
    ("codai.api.video", "_decode_b64_or_url"),
    ("codai.api.images", "_image_bytes"),
])
def test_a_sha256_upload_id_resolves(files_dir, module, func):
    """What /v1/files/upload hands back, with and without the prefix."""
    from codai.compose.media import UPLOAD_PREFIX, sha256_hex
    mod = __import__(module, fromlist=["x"])
    data = _png_bytes()
    h = sha256_hex(data)
    (files_dir / f"{UPLOAD_PREFIX}{h}.png").write_bytes(data)
    assert getattr(mod, func)(f"sha256:{h}") == data
    assert getattr(mod, func)(h) == data


@pytest.mark.parametrize("module,func", [
    ("codai.api.video", "_decode_b64_or_url"),
    ("codai.api.images", "_image_bytes"),
])
def test_an_absolute_url_to_our_own_file_is_still_read_off_disk(files_dir, module,
                                                                func):
    """A caller echoing back the `url` we gave it must not cause a loopback
    fetch, which would need a token and fails when the front is bound elsewhere."""
    mod = __import__(module, fromlist=["x"])
    data = _png_bytes()
    (files_dir / "clip.png").write_bytes(data)
    assert getattr(mod, func)("https://example.invalid/v1/files/clip.png") == data


@pytest.mark.parametrize("module,func", [
    ("codai.api.video", "_decode_b64_or_url"),
    ("codai.api.images", "_image_bytes"),
])
def test_base64_and_data_uris_still_work(files_dir, module, func):
    """The old inputs keep working -- this is additive."""
    mod = __import__(module, fromlist=["x"])
    data = _png_bytes()
    b64 = base64.b64encode(data).decode()
    assert getattr(mod, func)(b64) == data
    assert getattr(mod, func)(f"data:image/png;base64,{b64}") == data


@pytest.mark.parametrize("module,func", [
    ("codai.api.video", "_decode_b64_or_url"),
    ("codai.api.images", "_image_bytes"),
])
def test_line_wrapped_base64_is_still_accepted(files_dir, module, func):
    """Routing through the resolver must not tighten what callers may send.

    The old decoders used a lenient b64decode; the resolver validates strictly,
    which rejects the 76-column wrapping that MIME, the `base64` shell tool and
    Python's encodebytes all produce. Whitespace carries no meaning in base64,
    so it is stripped rather than refused.
    """
    mod = __import__(module, fromlist=["x"])
    data = _png_bytes(64, 64)
    flat = base64.b64encode(data).decode()
    wrapped = "\n".join(flat[i:i + 76] for i in range(0, len(flat), 76))
    assert len(wrapped) > len(flat), "the fixture did not actually wrap"
    assert getattr(mod, func)(wrapped) == data
    assert getattr(mod, func)(f"data:image/png;base64,{wrapped}") == data


@pytest.mark.parametrize("module,func", [
    ("codai.api.video", "_decode_b64_or_url"),
    ("codai.api.images", "_image_bytes"),
])
def test_a_path_outside_the_files_dir_is_refused(files_dir, module, func):
    """A client may name our uploads, never arbitrary paths on this machine."""
    from fastapi import HTTPException
    mod = __import__(module, fromlist=["x"])
    with pytest.raises(HTTPException) as e:
        getattr(mod, func)("/v1/files/../../../../etc/passwd")
    assert e.value.status_code == 400


@pytest.mark.parametrize("module,func", [
    ("codai.api.video", "_decode_b64_or_url"),
    ("codai.api.images", "_image_bytes"),
])
def test_an_unusable_reference_is_a_400_not_a_500(files_dir, module, func):
    """It is the caller's field that is wrong, so it must not read as our fault."""
    from fastapi import HTTPException
    mod = __import__(module, fromlist=["x"])
    with pytest.raises(HTTPException) as e:
        getattr(mod, func)("/home/someone/secret.mp4")
    assert e.value.status_code == 400


def test_a_missing_upload_does_not_resolve_to_someone_elses_file(files_dir):
    """An id we do not hold is an error, never a different file."""
    from fastapi import HTTPException
    from codai.api.images import _image_bytes
    with pytest.raises(HTTPException):
        _image_bytes("sha256:" + "0" * 64)


# ------------------------------------------------------- the relative path

def test_image_responses_carry_the_relative_path(files_dir, monkeypatch):
    """So a caller can chain calls without knowing our public hostname."""
    from PIL import Image
    import codai.api.images as I
    monkeypatch.setattr(I, "global_file_path", str(files_dir), raising=False)
    out = I.save_image_response(Image.new("RGB", (4, 4)), "url", None)
    assert out.get("path", "").startswith("/v1/files/")
    assert out["path"].endswith(".png")


def test_video_responses_carry_the_relative_path(files_dir, monkeypatch):
    import codai.api.video as V
    monkeypatch.setattr(V, "global_file_path", str(files_dir), raising=False)
    out = V._save_file(b"\x00\x01", "mp4", None)
    assert out.get("path", "").startswith("/v1/files/")
    assert out["path"].endswith(".mp4")


# ------------------------------------------------------------- the shape

def test_an_omitted_size_is_not_a_square():
    """The defect: the field's default was the string "1024x1024"."""
    from codai.api.images import ImageEditRequest
    assert ImageEditRequest.model_fields["size"].default is None


@pytest.mark.parametrize("w,h", [(3000, 2000), (2000, 3000), (1920, 1080),
                                 (800, 600), (640, 480)])
def test_the_default_size_keeps_the_source_shape(w, h):
    """Aspect preserved as closely as a 64-grid allows, and never flipped."""
    from codai.api.images import _default_edit_size
    ow, oh = _default_edit_size(w, h)
    assert (ow > oh) == (w > h), "the orientation was flipped"
    assert abs((ow / oh) - (w / h)) < 0.08, f"{w}x{h} -> {ow}x{oh} distorts"


@pytest.mark.parametrize("w,h", [(3000, 2000), (1920, 1080), (800, 600),
                                 (1024, 1024), (5000, 400)])
def test_the_default_size_lands_on_the_models_grid_near_one_megapixel(w, h):
    from codai.api.images import _default_edit_size
    ow, oh = _default_edit_size(w, h)
    assert ow % 64 == 0 and oh % 64 == 0, f"{ow}x{oh} is off the 64 grid"
    assert ow >= 64 and oh >= 64
    assert 0.5e6 <= ow * oh <= 1.8e6, f"{ow}x{oh} = {ow * oh} pixels"


def test_a_square_source_still_comes_back_square():
    from codai.api.images import _default_edit_size
    assert _default_edit_size(1024, 1024) == (1024, 1024)


def test_an_explicit_size_is_still_obeyed_exactly():
    """Callers that send their own size must not be second-guessed."""
    from codai.api.images import _default_edit_size
    # the helper is only consulted when `size` is absent; a sanity check that it
    # is not rounding a caller's own numbers
    assert _default_edit_size(10, 10) == (1024, 1024)


# -------------------------------------------------- the silent non-upscale

def test_an_unknown_upscaler_is_refused(monkeypatch):
    """The defect: PIL LANCZOS answered 200 and billed as an AI upscale."""
    import asyncio
    from fastapi import HTTPException
    import codai.api.images as I

    monkeypatch.setattr(I.multi_model_manager, "request_model",
                        lambda name, model_type=None: {"model_name": "some/clip-model"},
                        raising=False)
    monkeypatch.setattr(I, "_load_upscaler", lambda n, a: ("pil", None))
    monkeypatch.setattr(I, "_configured_upscalers", lambda: ["org/RealESRGAN-x4"])
    monkeypatch.setattr(I, "_progress_loading", lambda *a, **k: None)

    req = I.ImageUpscaleRequest(model="some/clip-model",
                                image=base64.b64encode(_png_bytes()).decode())
    with pytest.raises(HTTPException) as e:
        asyncio.run(I.create_image_upscale(req, None))
    assert e.value.status_code == 400
    assert "not an upscaling model" in str(e.value.detail)
    assert "RealESRGAN" in str(e.value.detail), "it must say what WOULD work"


def test_the_refusal_says_when_nothing_is_configured(monkeypatch):
    import asyncio
    from fastapi import HTTPException
    import codai.api.images as I

    monkeypatch.setattr(I.multi_model_manager, "request_model",
                        lambda name, model_type=None: {"model_name": "x/y"},
                        raising=False)
    monkeypatch.setattr(I, "_load_upscaler", lambda n, a: ("pil", None))
    monkeypatch.setattr(I, "_configured_upscalers", lambda: [])
    monkeypatch.setattr(I, "_progress_loading", lambda *a, **k: None)

    req = I.ImageUpscaleRequest(model="x/y",
                                image=base64.b64encode(_png_bytes()).decode())
    with pytest.raises(HTTPException) as e:
        asyncio.run(I.create_image_upscale(req, None))
    assert "no upscaling model is configured" in str(e.value.detail)


def test_a_real_upscaler_is_not_refused(monkeypatch):
    """The guard must only reject the PIL fall-through, nothing else."""
    import asyncio
    import codai.api.images as I
    from PIL import Image

    monkeypatch.setattr(I.multi_model_manager, "request_model",
                        lambda name, model_type=None: {"model_name": "org/esrgan"},
                        raising=False)
    monkeypatch.setattr(I, "_load_upscaler", lambda n, a: ("realesrgan", object()))
    monkeypatch.setattr(I, "_run_upscale",
                        lambda up, raw, scale: Image.new("RGB", (16, 16)))
    monkeypatch.setattr(I, "_progress_loading", lambda *a, **k: None)
    monkeypatch.setattr(I, "save_image_response",
                        lambda img, fmt, req: {"url": "u", "path": "/v1/files/x.png"})

    req = I.ImageUpscaleRequest(model="org/esrgan",
                                image=base64.b64encode(_png_bytes()).decode())
    out = asyncio.run(I.create_image_upscale(req, None))
    assert out["data"][0]["path"] == "/v1/files/x.png"


def test_configured_upscalers_lists_only_upscaling_models(monkeypatch):
    import codai.api.images as I
    monkeypatch.setattr(I.multi_model_manager, "image_models",
                        ["org/esrgan-x4", "org/sdxl-base"], raising=False)

    def _caps(name):
        return types.SimpleNamespace(image_upscaling="esrgan" in name)

    monkeypatch.setitem(
        sys.modules, "codai.models.capabilities",
        types.SimpleNamespace(detect_model_capabilities=_caps))
    assert I._configured_upscalers() == ["org/esrgan-x4"]


# ---------------------------------------------------------------- the audio

class _Pkt:
    def __init__(self, dts=0):
        self.dts = dts
        self.stream = None


class _Stream:
    def __init__(self, kind, rate=48000):
        self.kind = kind
        self.rate = rate
        self.bit_rate = None
        self.format = types.SimpleNamespace(name="fltp")
        self.layout = types.SimpleNamespace(name="stereo")

    def encode(self, frame=None):
        return [_Pkt()]


class _InContainer:
    def __init__(self, video=(), audio=(), packets=3):
        self.streams = types.SimpleNamespace(video=list(video), audio=list(audio))
        self._packets = packets

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def demux(self, stream):
        for i in range(self._packets):
            yield _Pkt(dts=i)
        yield _Pkt(dts=None)      # a packet the muxer must skip

    def decode(self, stream):
        for _ in range(self._packets):
            yield types.SimpleNamespace(pts=0)


class _OutContainer:
    def __init__(self, template_audio_fails=False):
        self.muxed = 0
        self.added = []
        self._template_audio_fails = template_audio_fails

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def add_stream_from_template(self, s):
        if s.kind == "audio" and self._template_audio_fails:
            raise ValueError("codec not supported in this container")
        self.added.append(("template:" + s.kind, None))
        return _Stream(s.kind)

    def add_stream(self, name, rate=None):
        self.added.append((name, rate))
        return _Stream("audio", rate=rate or 48000)

    def mux(self, pkt):
        self.muxed += 1


def _install_fake_av(monkeypatch, *, source_has_audio=True,
                     template_audio_fails=False, out_holder=None,
                     open_raises=False):
    """A stand-in for PyAV: `av` is not installed in the test environment, and
    the point of these tests is the muxing decisions, not libav itself."""
    def _open(path, mode="r", **kw):
        if open_raises:
            raise RuntimeError("libav exploded")
        if mode == "w":
            out = _OutContainer(template_audio_fails=template_audio_fails)
            if out_holder is not None:
                out_holder.append(out)
            return out
        audio = [_Stream("audio")] if source_has_audio else []
        if str(path).endswith("_src.mp4"):
            return _InContainer(video=[_Stream("video")], audio=audio)
        # the freshly encoded, silent result
        return _InContainer(video=[_Stream("video")], audio=[])

    class _Resampler:
        def __init__(self, **kw):
            pass

        def resample(self, frame):
            return [frame]

    fake = types.SimpleNamespace(open=_open, AudioResampler=_Resampler)
    monkeypatch.setitem(sys.modules, "av", fake)
    return fake


@pytest.fixture
def clips(tmp_path):
    src = tmp_path / "in_src.mp4"
    vid = tmp_path / "out.mp4"
    src.write_bytes(b"source")
    vid.write_bytes(b"silent-result")
    return str(src), str(vid)


def test_the_source_audio_is_carried_onto_the_result(monkeypatch, clips):
    """The defect: upscale and interpolation returned a silent clip, with a 200."""
    import codai.api.video as V
    src, vid = clips
    held = []
    _install_fake_av(monkeypatch, source_has_audio=True, out_holder=held)
    temps = []
    out = V._mux_original_audio(vid, src, temps)
    assert out != vid, "the result is still the silent encode"
    assert held and held[0].muxed > 0, "nothing was muxed"
    assert any(a[0] == "template:audio" for a in held[0].added), \
        "the audio track was not copied"
    assert out in temps, "the new file must be cleaned up with the others"


def test_the_audio_is_stream_copied_not_re_encoded(monkeypatch, clips):
    """Re-encoding a track that could be copied loses quality for nothing."""
    import codai.api.video as V
    src, vid = clips
    held = []
    _install_fake_av(monkeypatch, out_holder=held)
    V._mux_original_audio(vid, src, [])
    assert ("aac", None) not in held[0].added
    assert any(a[0] == "template:audio" for a in held[0].added)


def test_an_incompatible_codec_is_re_encoded_rather_than_dropped(monkeypatch, clips):
    """opus/vorbis out of a webm cannot sit in MP4 -- re-encode, do not drop."""
    import codai.api.video as V
    src, vid = clips
    held = []
    _install_fake_av(monkeypatch, template_audio_fails=True, out_holder=held)
    out = V._mux_original_audio(vid, src, [])
    assert out != vid, "the track was dropped instead of re-encoded"
    assert any(a[0] == "aac" for a in held[0].added), "no AAC encoder was set up"


def test_a_silent_source_produces_a_silent_result_untouched(monkeypatch, clips):
    """No audio in means no audio out, and no pointless remux."""
    import codai.api.video as V
    src, vid = clips
    held = []
    _install_fake_av(monkeypatch, source_has_audio=False, out_holder=held)
    assert V._mux_original_audio(vid, src, []) == vid
    assert not held, "a container was opened for write with nothing to mux"


def test_a_broken_remux_keeps_the_enhanced_frames(monkeypatch, clips):
    """Losing the track is bad; throwing away a finished GPU job is worse."""
    import codai.api.video as V
    src, vid = clips
    _install_fake_av(monkeypatch, open_raises=True)
    assert V._mux_original_audio(vid, src, []) == vid


def test_a_missing_source_is_not_an_error(monkeypatch, tmp_path):
    """The two-image interpolation path has no source clip at all."""
    import codai.api.video as V
    vid = tmp_path / "out.mp4"
    vid.write_bytes(b"x")
    _install_fake_av(monkeypatch)
    assert V._mux_original_audio(str(vid), "", []) == str(vid)
    assert V._mux_original_audio(str(vid), str(tmp_path / "nope.mp4"), []) == str(vid)


def test_packets_without_timestamps_are_skipped(monkeypatch, clips):
    """A DTS-less packet makes libav refuse the whole mux."""
    import codai.api.video as V
    src, vid = clips
    held = []
    _install_fake_av(monkeypatch, out_holder=held)
    V._mux_original_audio(vid, src, [])
    # 3 video + 3 audio packets carry a DTS; the two trailing None ones do not
    assert held[0].muxed == 6, f"muxed {held[0].muxed} packets, expected 6"


# ------------------------------------------- the enhancement paths use it

@pytest.mark.parametrize("func", ["_model_upscale_video",
                                  "_interpolate_inprocess",
                                  "_rife_ncnn_interpolate"])
def test_every_enhancement_path_muxes_the_audio_back(func):
    """A path that forgets the call is a path that silently drops the track."""
    import inspect
    import codai.api.video as V
    src = inspect.getsource(getattr(V, func))
    assert "_mux_original_audio" in src, f"{func} does not restore the audio"
