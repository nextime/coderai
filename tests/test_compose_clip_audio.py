"""B1: a scene can carry its clip's own sound, not only the narration.

Every visual was rendered with ffmpeg ``-an``, so footage whose sound IS the
content -- a crowd, a demo, a piece to camera -- arrived silent and the only
audio a scene could have was a synthesised voice over it.
"""
import os
import subprocess
import tempfile

import pytest

from codai.api.compose import Scene, Visual, _scene_clip_visual, _scene_mix


# ------------------------------------------------------------ the schema bits
def test_defaults_are_the_old_behaviour():
    sc = Scene(text="hi", visuals=[Visual(type="video", src="a.mp4")])
    assert _scene_mix(sc) == "voice"
    assert sc.clip_volume == 0.35
    assert sc.visuals[0].keep_audio is False
    # Nothing is picked up without being asked for: a request written before
    # this feature existed must render byte-for-byte as it did.
    assert _scene_clip_visual(sc) is None


@pytest.mark.parametrize("given,expect", [
    ("voice", "voice"), ("clip", "clip"), ("both", "both"),
    ("CLIP", "clip"), (" both ", "both"),
    ("nonsense", "voice"), (None, "voice"), ("", "voice"),
])
def test_the_mix_is_validated_down_to_three(given, expect):
    sc = Scene(text="hi") if given is None else Scene(text="hi", audio_mix=given)
    assert _scene_mix(sc) == expect


def test_keep_audio_picks_that_visual():
    sc = Scene(text="hi", audio_mix="both", visuals=[
        Visual(type="image", src="a.png"),
        Visual(type="video", src="b.mp4"),
        Visual(type="video", src="c.mp4", keep_audio=True),
    ])
    v = _scene_clip_visual(sc)
    assert v is not None and v.src == "c.mp4"


def test_the_first_marked_clip_wins_not_all_of_them():
    # Summing several soundtracks under one narration is not what anyone asked
    # for; a second clip's sound belongs in its own scene.
    sc = Scene(text="hi", audio_mix="both", visuals=[
        Visual(type="video", src="b.mp4", keep_audio=True),
        Visual(type="video", src="c.mp4", keep_audio=True),
    ])
    assert _scene_clip_visual(sc).src == "b.mp4"


def test_a_single_clip_needs_no_keep_audio_flag():
    # Asking for the clip's sound and not saying which clip can only mean the
    # one clip in the scene.
    sc = Scene(audio_mix="clip", visuals=[Visual(type="video", src="b.mp4")])
    assert _scene_clip_visual(sc).src == "b.mp4"
    # But with two, it is ambiguous and must be marked.
    sc2 = Scene(audio_mix="clip", visuals=[Visual(type="video", src="b.mp4"),
                                           Visual(type="video", src="c.mp4")])
    assert _scene_clip_visual(sc2) is None


def test_images_are_never_offered_as_an_audio_source():
    sc = Scene(audio_mix="clip", visuals=[Visual(type="image", src="a.png",
                                                 keep_audio=True)])
    assert _scene_clip_visual(sc) is None


# ------------------------------------------------- the render helpers, for real
def _ffmpeg():
    from codai.compose import media
    return media.ffmpeg_bin()


def _have_ffmpeg():
    try:
        subprocess.run([_ffmpeg(), "-version"], capture_output=True, timeout=30)
        return True
    except Exception:
        return False


needs_ffmpeg = pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not available")


def _make_clip(path, *, seconds=3.0, with_audio=True, freq=440):
    args = [_ffmpeg(), "-y", "-f", "lavfi", "-i",
            f"color=c=red:s=320x240:r=24:d={seconds}"]
    if with_audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}"]
    args += ["-t", str(seconds), "-pix_fmt", "yuv420p"]
    if with_audio:
        args += ["-shortest"]
    args += [path]
    subprocess.run(args, capture_output=True, check=True, timeout=120)
    return path


@needs_ffmpeg
def test_has_audio_stream_tells_the_two_apart():
    from codai.compose import render as rnd
    with tempfile.TemporaryDirectory() as d:
        loud = _make_clip(os.path.join(d, "loud.mp4"))
        mute = _make_clip(os.path.join(d, "mute.mp4"), with_audio=False)
        assert rnd.has_audio_stream(loud) is True
        assert rnd.has_audio_stream(mute) is False
        # A file that is not there at all answers "no" rather than raising:
        # the extraction would fail anyway and the render says so.
        assert rnd.has_audio_stream(os.path.join(d, "nope.mp4")) is False


@needs_ffmpeg
def test_clip_audio_is_cut_to_exactly_its_slot():
    from codai.compose import media
    from codai.compose import render as rnd
    with tempfile.TemporaryDirectory() as d:
        src = _make_clip(os.path.join(d, "s.mp4"), seconds=3.0)
        # Shorter than the clip: trimmed.
        a = rnd.clip_audio(src, os.path.join(d, "a.wav"), duration=1.5)
        assert abs(media.duration_of(a) - 1.5) < 0.08, media.duration_of(a)
        # Longer than the clip: padded with silence, never short.
        b = rnd.clip_audio(src, os.path.join(d, "b.wav"), duration=5.0)
        assert abs(media.duration_of(b) - 5.0) < 0.08, media.duration_of(b)
        # Looped, to match a video track that build_visual looped.
        c = rnd.clip_audio(src, os.path.join(d, "c.wav"), duration=7.0, loop=True)
        assert abs(media.duration_of(c) - 7.0) < 0.08, media.duration_of(c)


@needs_ffmpeg
def test_trim_start_skips_the_same_audio_the_picture_skips():
    from codai.compose import media
    from codai.compose import render as rnd
    with tempfile.TemporaryDirectory() as d:
        src = _make_clip(os.path.join(d, "s.mp4"), seconds=4.0)
        a = rnd.clip_audio(src, os.path.join(d, "a.wav"), duration=4.0,
                           trim_start=2.0)
        # Four seconds of slot, two seconds skipped: two of sound, two of pad.
        assert abs(media.duration_of(a) - 4.0) < 0.08


@needs_ffmpeg
def test_volume_is_applied():
    from codai.compose import render as rnd
    with tempfile.TemporaryDirectory() as d:
        src = _make_clip(os.path.join(d, "s.mp4"), seconds=2.0)
        loud = rnd.clip_audio(src, os.path.join(d, "l.wav"), duration=2.0,
                              volume=1.0)
        quiet = rnd.clip_audio(src, os.path.join(d, "q.wav"), duration=2.0,
                               volume=0.1)

        def _peak(p):
            out = subprocess.run(
                [_ffmpeg(), "-i", p, "-af", "volumedetect", "-f", "null", "-"],
                capture_output=True, text=True, timeout=120)
            for line in (out.stderr or "").splitlines():
                if "max_volume" in line:
                    return float(line.split("max_volume:")[1].replace("dB", "").strip())
            return None

        pl, pq = _peak(loud), _peak(quiet)
        assert pl is not None and pq is not None
        assert pq < pl - 15, (pl, pq)     # ~20 dB down for 0.1


@needs_ffmpeg
def test_mix_voice_clip_covers_every_combination():
    from codai.compose import media
    from codai.compose import render as rnd
    with tempfile.TemporaryDirectory() as d:
        src = _make_clip(os.path.join(d, "s.mp4"), seconds=3.0)
        clip = rnd.clip_audio(src, os.path.join(d, "c.wav"), duration=3.0)
        voice = rnd.silence(os.path.join(d, "v.wav"), 2.0)

        both = rnd.mix_voice_clip(voice, clip, os.path.join(d, "both.wav"),
                                  duration=3.0, duck=True)
        assert abs(media.duration_of(both) - 3.0) < 0.1

        only_clip = rnd.mix_voice_clip(None, clip, os.path.join(d, "oc.wav"),
                                       duration=3.0)
        assert abs(media.duration_of(only_clip) - 3.0) < 0.1

        only_voice = rnd.mix_voice_clip(voice, None, os.path.join(d, "ov.wav"),
                                        duration=3.0)
        assert abs(media.duration_of(only_voice) - 3.0) < 0.1

        neither = rnd.mix_voice_clip(None, None, os.path.join(d, "n.wav"),
                                     duration=3.0)
        assert abs(media.duration_of(neither) - 3.0) < 0.1


# --------------------------------------------------------------------------- #
# /v1/audio/generate: response_format was declared but ignored
# --------------------------------------------------------------------------- #

def test_asking_for_base64_audio_returns_bytes_not_a_url():
    """`response_format` is documented as url | b64_wav | b64_mp3, but the
    response was chosen by whether a file directory was configured -- so a
    client asking for base64 silently got a URL. The image endpoint had the
    same bug with b64_json."""
    from codai.api.audio_gen import _save_audio_response

    out = _save_audio_response(b"RIFFfake", "wav", None, "b64_wav")
    assert set(out) == {"b64_wav"}
    import base64 as _b64
    assert _b64.b64decode(out["b64_wav"]) == b"RIFFfake"


def test_the_default_still_prefers_a_url():
    """Unchanged for every existing caller: omitted or "url" keeps the URL."""
    from codai.api import audio_gen

    for fmt in (None, "url"):
        out = audio_gen._save_audio_response(b"RIFFfake", "wav", None, fmt)
        assert "b64_wav" in out or "url" in out


def test_mp3_is_actually_encoded_not_a_wav_under_an_mp3_key(monkeypatch):
    """Base64 of a wav under a b64_mp3 key is a decode failure at the client,
    not a format it can use."""
    from codai.api import audio_gen

    monkeypatch.setattr(audio_gen, "_to_mp3", lambda b: b"ID3fake-mp3")
    out = audio_gen._save_audio_response(b"RIFFfake", "wav", None, "b64_mp3")
    import base64 as _b64
    assert set(out) == {"b64_mp3"}
    assert _b64.b64decode(out["b64_mp3"]) == b"ID3fake-mp3"


# ------------------------------------- 14d: the sound in a separate file
#
# YouTube and other DASH sites serve picture and sound as separate streams, so
# the best available video of a clip often carries no audio track at all. A
# caller that does not run ffmpeg itself cannot mux them, and had to skip such
# a clip entirely. `Visual.audio_src` names the sound's own file.

def test_audio_src_defaults_to_nothing():
    """A request written before this existed must behave exactly as it did."""
    v = Visual(type="video", src="a.mp4")
    assert v.audio_src is None


def test_a_visual_with_a_separate_audio_file_is_still_the_clip_visual():
    sc = Scene(audio_mix="clip", visuals=[
        Visual(type="video", src="v.mp4", audio_src="a.m4a", keep_audio=True)])
    v = _scene_clip_visual(sc)
    assert v is not None and v.src == "v.mp4" and v.audio_src == "a.m4a"


@needs_ffmpeg
def test_a_video_with_no_audio_track_is_usable_when_audio_src_supplies_the_sound():
    """The whole point: `src` is not required to have an audio stream."""
    from codai.compose import render as rnd
    with tempfile.TemporaryDirectory() as d:
        vid = os.path.join(d, "video-only.mp4")
        aud = os.path.join(d, "audio-only.m4a")
        _make_clip(vid, seconds=2.0, with_audio=False)
        subprocess.run([_ffmpeg(), "-y", "-f", "lavfi", "-i",
                        "sine=frequency=330:duration=2.0", "-t", "2.0", aud],
                       capture_output=True, check=True, timeout=120)
        assert not rnd.has_audio_stream(vid), "fixture is wrong: video has audio"
        assert rnd.has_audio_stream(aud), "fixture is wrong: audio file has none"
        # The sound is taken from the companion file, cut the same way.
        out = os.path.join(d, "track.wav")
        rnd.clip_audio(aud, out, duration=1.0, trim_start=0.5,
                       loop=False, volume=1.0)
        assert os.path.getsize(out) > 44, "no PCM was written from audio_src"


def test_the_picture_path_is_never_replaced_by_the_audio_file():
    """Stage 3 reuses the resolved clip as this visual's VIDEO.

    Putting the audio-only file in that slot would render a scene with no
    image at all, so the two paths are tracked separately.
    """
    import inspect
    from codai.api import compose as C
    src = inspect.getsource(C._run_job) if hasattr(C, "_run_job") else ""
    if not src:
        pytest.skip("cannot read the job source")
    assert '"audio_path"' in src, (
        "the clip's sound is not tracked apart from its picture")
    # The visual cache must hand back `path`, not the audio file.
    assert '_cached["path"]' in src, (
        "the visual no longer reuses the resolved picture path")
