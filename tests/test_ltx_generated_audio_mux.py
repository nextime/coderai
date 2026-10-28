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

"""C3: the soundtrack LTX generates with the picture must reach the MP4.

LTX-2.x is the one video family here with an audio VAE and a vocoder, so its
output is `LTX2PipelineOutput(frames=..., audio=...)` -- audio generated
jointly with the frames and therefore actually in step with them. Only
`frames` was read, so every LTX render came out silent and the only way to get
sound was to generate some separately with another model and mux that on top,
which does not match the picture.

`add_audio` is tri-state so that "say nothing" and "say no" differ: unset keeps
what the model generated, false gives a silent video, true runs the older
separate-generation path.
"""

import os
import subprocess

import pytest

from codai.api import video as V


def _ffmpeg_ok():
    try:
        from codai.compose import media
        subprocess.run([media.ffmpeg_bin(), "-version"],
                       capture_output=True, timeout=30)
        return True
    except Exception:
        return False


needs_ffmpeg = pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")
np = pytest.importorskip("numpy")


# ------------------------------------------------------- reading the output

class _Pipe:
    audio_sampling_rate = 44100


class _Out:
    def __init__(self, frames, audio):
        self.frames = frames
        self.audio = audio


def test_a_pipeline_that_generated_audio_hands_it_over():
    got = V._pipeline_audio(_Out("frames", np.zeros(1000, dtype="float32")),
                            _Pipe())
    assert got is not None
    audio, rate = got
    assert rate == 44100 and audio.shape == (1000,)


def test_a_tuple_result_works_too():
    """`return_dict=False` gives (video, audio)."""
    got = V._pipeline_audio(("frames", np.zeros(8, dtype="float32")), _Pipe())
    assert got is not None and got[1] == 44100


def test_a_pipeline_with_no_audio_hands_over_nothing():
    """Every other video family: there is no soundtrack to carry."""
    class _NoAudio:
        frames = "frames"
    assert V._pipeline_audio(_NoAudio(), _Pipe()) is None


def test_an_unknown_sample_rate_is_refused_rather_than_guessed():
    """A wrong rate plays the soundtrack at the wrong speed, which is worse
    than no soundtrack."""
    class _NoRate:
        pass
    assert V._pipeline_audio(_Out("f", np.zeros(8, dtype="float32")),
                             _NoRate()) is None


def test_the_rate_falls_back_to_the_audio_vae_config():
    class _Vae:
        class config:
            sample_rate = 16000
    class _P:
        audio_vae = _Vae()
    got = V._pipeline_audio(_Out("f", np.zeros(4, dtype="float32")), _P())
    assert got is not None and got[1] == 16000


def test_batch_dimensions_are_dropped():
    audio = np.zeros((1, 2, 500), dtype="float32")
    got = V._pipeline_audio(_Out("f", audio), _Pipe())
    assert got is not None and got[0].shape == (2, 500)


def test_an_empty_waveform_is_not_muxed():
    got = V._pipeline_audio(_Out("f", np.zeros((0,), dtype="float32")), _Pipe())
    assert got is None


# --------------------------------------------------------------- the wav

def test_the_waveform_is_written_as_16_bit_pcm(tmp_path):
    import wave
    temps = []
    a = np.sin(np.linspace(0, 20, 4410)).astype("float32")
    path = V._write_wav(a, 44100, temps)
    assert path and os.path.getsize(path) > 44
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == 44100
        assert wf.getsampwidth() == 2
        assert wf.getnchannels() == 1
        assert wf.getnframes() == 4410
    for t in temps:
        try:
            os.unlink(t)
        except OSError:
            pass


def test_a_stereo_waveform_keeps_both_channels():
    import wave
    temps = []
    a = np.zeros((2, 1000), dtype="float32")       # (channels, samples)
    path = V._write_wav(a, 48000, temps)
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 2
        assert wf.getnframes() == 1000
    for t in temps:
        try:
            os.unlink(t)
        except OSError:
            pass


def test_a_waveform_louder_than_full_scale_is_normalised_not_clipped():
    """A vocoder is not bound to [-1, 1]; clipping it would be audible."""
    import wave
    temps = []
    a = (np.ones(100, dtype="float32") * 4.0)
    path = V._write_wav(a, 16000, temps)
    with wave.open(path, "rb") as wf:
        raw = np.frombuffer(wf.readframes(100), dtype="<i2")
    assert raw.max() <= 32767
    # Normalised to full scale rather than every sample pinned by clipping.
    assert raw.max() >= 32000
    for t in temps:
        try:
            os.unlink(t)
        except OSError:
            pass


# --------------------------------------------------------------- the mux

@needs_ffmpeg
def test_the_generated_audio_ends_up_on_the_video(tmp_path):
    from codai.compose import media
    silent = str(tmp_path / "silent.mp4")
    subprocess.run([media.ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
                    "-f", "lavfi", "-i",
                    "testsrc=size=160x120:rate=10:duration=2",
                    "-pix_fmt", "yuv420p", silent],
                   check=True, timeout=180)
    assert not V._has_audio(silent), "fixture already has audio"

    temps = []
    a = np.sin(np.linspace(0, 50, 44100 * 2)).astype("float32")
    out = V._mux_generated_audio(silent, a, 44100, temps)
    try:
        assert out != silent, "nothing was muxed"
        assert V._has_audio(out), "the soundtrack did not reach the file"
    finally:
        for t in temps:
            try:
                os.unlink(t)
            except OSError:
                pass


@needs_ffmpeg
def test_a_failed_mux_returns_the_video_unchanged(tmp_path):
    """A silent video beats no video: the frames are what was asked for."""
    missing = str(tmp_path / "not-a-video.mp4")
    temps = []
    out = V._mux_generated_audio(missing, np.zeros(10, dtype="float32"),
                                 44100, temps)
    assert out == missing
    for t in temps:
        try:
            os.unlink(t)
        except OSError:
            pass


# ------------------------------------------------------------- the switch

def test_add_audio_is_tri_state():
    """Unset must differ from false, or the default would silence LTX."""
    from codai.pydantic.videorequest import VideoGenerationRequest
    f = VideoGenerationRequest.model_fields["add_audio"]
    assert f.default is None, (
        "add_audio defaulting to False cannot be told from an explicit false, "
        "so a generated soundtrack would be dropped by default")


def test_only_an_unset_add_audio_keeps_the_generated_track():
    import inspect
    src = inspect.getsource(V.video_generations)
    assert "request.add_audio is None" in src, (
        "the generated soundtrack is no longer gated on an UNSET add_audio")


def test_the_generated_track_is_muxed_before_the_other_steps():
    """Every later step re-encodes the frames, and the track belongs to
    these exact ones."""
    import inspect
    src = inspect.getsource(V.video_generations)
    # Compared on the CALLS, not on any mention: a comment naming the other
    # step made this pass or fail on prose.
    i_gen = src.index("None, _mux_generated_audio_bytes")
    i_post = src.index("None, _postprocess_video")
    assert i_gen < i_post, "the generated soundtrack is muxed after post-processing"


def test_the_capture_is_threaded_through_generation():
    import inspect
    assert "audio_out" in inspect.signature(V._run_pipeline).parameters
    assert "audio_out" in inspect.signature(V._run_pipeline_split).parameters
    src = inspect.getsource(V._generate_video)
    assert "_gen_audio" in src, "generation no longer collects the soundtrack"


# ------------------------------------------- telling audio from "cannot tell"

@needs_ffmpeg
def test_audio_detection_does_not_depend_on_pyav(tmp_path, monkeypatch):
    """PyAV is in no main requirements file, only the LongCat venv's.

    `_has_audio` tried `import av` under a bare `except: return False`, so
    without PyAV EVERY file answered "no audio" -- and because it gates
    `_mux_original_audio`, a clip's own sound was silently dropped from every
    upscale and interpolate. The audio was there, nothing could see it, and
    nothing said so. ffprobe cannot be missing the way PyAV can: ffmpeg is
    what does the muxing.
    """
    from codai.compose import media
    loud = str(tmp_path / "loud.mp4")
    silent = str(tmp_path / "silent.mp4")
    subprocess.run([media.ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
                    "-f", "lavfi", "-i",
                    "testsrc=size=160x120:rate=10:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-pix_fmt", "yuv420p", "-shortest", loud],
                   check=True, timeout=180)
    subprocess.run([media.ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
                    "-f", "lavfi", "-i",
                    "testsrc=size=160x120:rate=10:duration=1",
                    "-pix_fmt", "yuv420p", silent],
                   check=True, timeout=180)

    # Force the PyAV path to fail, exactly as it does on an install without it.
    import builtins
    real_import = builtins.__import__

    def _no_av(name, *a, **kw):
        if name == "av":
            raise ImportError("No module named 'av'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_av)
    assert V._has_audio(loud) is True, (
        "a file WITH audio was reported silent without PyAV")
    assert V._has_audio(silent) is False


def test_a_file_that_cannot_be_read_at_all_is_reported_not_swallowed(caplog):
    """"Cannot tell" must not look identical to "no audio" in the log."""
    import logging
    with caplog.at_level(logging.WARNING):
        assert V._has_audio("/nonexistent/path/to/nothing.mp4") is False
    assert any("audio" in r.message.lower() for r in caplog.records), (
        "a total failure to inspect the file was not logged")
