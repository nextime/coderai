"""B2: two characters talking in one scene.

A scene had one voice and one presenter, which is a monologue. A dialogue
needs an order, a voice per turn, and somewhere for each character to be while
the other one talks.
"""
import os
import subprocess
import tempfile

import pytest
from fastapi import HTTPException

from codai.api.compose import (ComposeRequest, Scene, SpeakerLine, Visual,
                               _speaker_key, _speaker_layout, _speaker_order,
                               _validate)


def _sp(**kw):
    return SpeakerLine(**kw)


@pytest.fixture(autouse=True)
def _a_tts_model_exists(monkeypatch):
    """Pretend a TTS model is installed, for every test in this module.

    These tests are about how a dialogue scene is VALIDATED, and say so: the
    portraits are given inline "so the test does not depend on what is saved on
    this install". The TTS check is the same kind of install dependency and was
    missed -- `_validate` refuses a request whose lines need narration when no
    TTS model is configured, so on a bare checkout three of these failed with
    501 "narration needs a TTS model" without ever reaching the validation they
    exercise.

    Only the INVENTORY is faked, not the check: `_has_model` still does its own
    name matching, and `test_a_line_whose_voice_model_is_not_installed_is_refused`
    (if it ever exists) would still see a real miss for a name not listed here.
    """
    import codai.api.compose as C
    monkeypatch.setattr(C, "_model_ids",
                        lambda mtype: (["hexgrad/Kokoro-82M"]
                                       if mtype == "tts" else []))


# ----------------------------------------------------------------- identity
def test_a_character_is_identified_by_profile_then_portrait_then_position():
    assert _speaker_key(_sp(character="Alice"), 0) == "alice"
    assert _speaker_key(_sp(image="a.png"), 0) == "a.png"
    assert _speaker_key(_sp(), 3) == "_speaker3"


def test_the_same_character_speaking_twice_is_one_column():
    sc = Scene(speakers=[_sp(character="alice", text="hi"),
                         _sp(character="bob", text="hello"),
                         _sp(character="Alice", text="how are you")])
    order = _speaker_order(sc)
    assert [k for k, _ in order] == ["alice", "bob"]


def test_the_column_order_is_first_speaking_order():
    sc = Scene(speakers=[_sp(character="bob", text="x"),
                         _sp(character="alice", text="y")])
    assert [k for k, _ in _speaker_order(sc)] == ["bob", "alice"]


# ------------------------------------------------------------------ layout
def test_columns_are_the_default():
    sc = Scene(speakers=[_sp(character="a", text="x"), _sp(character="b", text="y")])
    assert _speaker_layout(sc) == "row"


@pytest.mark.parametrize("layouts,expect", [
    (["left", "right"], "row"),
    (["full", "full"], "cut"),
    ([None, None], "row"),
    (["left", None], "row"),
])
def test_the_layout_is_derived_from_what_the_speakers_ask_for(layouts, expect):
    sc = Scene(speakers=[_sp(character=f"c{i}", text="x", layout=l)
                         for i, l in enumerate(layouts)])
    assert _speaker_layout(sc) == expect


def test_mixing_a_column_and_full_is_refused():
    # A character cannot both hold a column and be off screen while the other
    # one speaks; rendering it would have to silently pick one.
    sc = Scene(speakers=[_sp(character="a", text="x", layout="left"),
                         _sp(character="b", text="y", layout="full")])
    with pytest.raises(ValueError) as e:
        _speaker_layout(sc)
    assert "column" in str(e.value)


# -------------------------------------------------------------- validation
def _req(scene):
    return ComposeRequest(scenes=[scene])


def test_a_dialogue_scene_needs_neither_text_nor_visuals():
    # Its lines ARE the narration and the presenter. Refusing it for having
    # neither is what the old validation did. Portraits given inline, so the
    # test does not depend on what is saved on this install.
    _validate(_req(Scene(speakers=[
        _sp(image="data:image/png;base64,AAAA", text="hi"),
        _sp(image="data:image/png;base64,BBBB", text="hello")])))


def test_a_line_with_nothing_to_say_is_refused():
    with pytest.raises(HTTPException) as e:
        _validate(_req(Scene(speakers=[_sp(image="data:image/png;base64,AAAA")])))
    assert e.value.status_code == 400
    assert "text" in str(e.value.detail)


def test_a_speaker_with_no_face_is_refused_up_front():
    # It cannot be lip-synced, and finding that out mid-render costs the caller
    # the narration they already paid to synthesise.
    with pytest.raises(HTTPException) as e:
        _validate(_req(Scene(speakers=[_sp(text="hi")])))
    assert e.value.status_code == 400
    assert "character" in str(e.value.detail)


def test_a_speaker_may_inherit_the_request_presenter_portrait():
    from codai.api.compose import PresenterSpec
    req = ComposeRequest(
        scenes=[Scene(speakers=[_sp(text="hi")])],
        presenter=PresenterSpec(image="data:image/png;base64,AAAA"))
    _validate(req)


def test_a_misspelt_character_profile_is_caught_before_the_render(monkeypatch):
    import codai.api.characters as C
    monkeypatch.setattr(C, "_load_character_meta",
                        lambda n: {"name": n} if n == "alice" else None)
    _validate(_req(Scene(speakers=[_sp(character="alice", text="hi")])))
    with pytest.raises(HTTPException) as e:
        _validate(_req(Scene(speakers=[_sp(character="alise", text="hi")])))
    assert e.value.status_code == 400
    assert "alise" in str(e.value.detail)


def test_a_bad_layout_mix_is_a_400_not_a_500():
    with pytest.raises(HTTPException) as e:
        _validate(_req(Scene(speakers=[
            _sp(image="data:image/png;base64,AAAA", text="x", layout="left"),
            _sp(image="data:image/png;base64,BBBB", text="y", layout="full")])))
    assert e.value.status_code == 400


def test_too_many_lines_in_one_scene_is_refused():
    from codai.api.compose import _MAX_SPEAKERS_PER_SCENE
    sc = Scene(speakers=[_sp(image="data:image/png;base64,AAAA", text="x")
                         for _ in range(_MAX_SPEAKERS_PER_SCENE + 1)])
    with pytest.raises(HTTPException) as e:
        _validate(_req(sc))
    assert "at most" in str(e.value.detail)


# --------------------------------------------------- captions, per speaker
def test_a_caption_never_mixes_two_speakers():
    from codai.compose.captions import break_lines
    words = ([{"word": f"a{i}", "start": i, "end": i + 1, "speaker": "alice"}
              for i in range(3)]
             + [{"word": f"b{i}", "start": 3 + i, "end": 4 + i, "speaker": "bob"}
                for i in range(3)])
    evs = break_lines(words, max_words_per_line=4, max_lines=2)
    # Without the speaker split these six words are one event of a question
    # and its answer, attributed to nobody.
    assert len(evs) == 2
    assert evs[0]["speaker"] == "alice"
    assert evs[1]["speaker"] == "bob"
    assert "b0" not in evs[0]["text"]


def test_a_long_run_by_one_speaker_still_splits_by_length():
    from codai.compose.captions import break_lines
    words = [{"word": f"w{i}", "start": i, "end": i + 1, "speaker": "alice"}
             for i in range(10)]
    evs = break_lines(words, max_words_per_line=2, max_lines=1)
    assert len(evs) == 5
    assert all(e["speaker"] == "alice" for e in evs)


def test_captions_without_speakers_are_unchanged():
    from codai.compose.captions import break_lines
    words = [{"word": f"w{i}", "start": i, "end": i + 1} for i in range(4)]
    evs = break_lines(words, max_words_per_line=4, max_lines=1)
    assert len(evs) == 1
    assert "speaker" not in evs[0]


# ----------------------------------------------- the render helpers, for real
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


@needs_ffmpeg
def test_place_at_puts_each_line_where_it_falls():
    from codai.compose import media
    from codai.compose import render as rnd
    with tempfile.TemporaryDirectory() as d:
        a = rnd.silence(os.path.join(d, "a.wav"), 1.0)
        b = rnd.silence(os.path.join(d, "b.wav"), 1.0)
        out = rnd.place_at([(a, 0.0), (b, 2.0)], os.path.join(d, "t.wav"),
                           duration=4.0)
        assert abs(media.duration_of(out) - 4.0) < 0.1


@needs_ffmpeg
def test_place_at_with_nothing_is_silence_not_an_error():
    # A character who says nothing in a scene still needs a track of the
    # scene's length, or their lip-sync run has nothing to drive.
    from codai.compose import media
    from codai.compose import render as rnd
    with tempfile.TemporaryDirectory() as d:
        out = rnd.place_at([], os.path.join(d, "t.wav"), duration=2.5)
        assert abs(media.duration_of(out) - 2.5) < 0.1


@needs_ffmpeg
def test_composite_row_puts_two_speakers_side_by_side():
    from codai.compose import media
    from codai.compose import render as rnd
    with tempfile.TemporaryDirectory() as d:
        clips = []
        for i, colour in enumerate(("red", "blue")):
            p = os.path.join(d, f"c{i}.mp4")
            subprocess.run([_ffmpeg(), "-y", "-f", "lavfi", "-i",
                            f"color=c={colour}:s=240x320:r=24:d=2",
                            "-pix_fmt", "yuv420p", p],
                           capture_output=True, check=True, timeout=120)
            clips.append(p)
        out = rnd.composite_row(clips, os.path.join(d, "row.mp4"),
                                width=480, height=320, fps=24)
        info = media.probe(out)
        assert info["width"] == 480 and info["height"] == 320


@needs_ffmpeg
def test_cut_segment_lands_on_the_asked_for_window():
    from codai.compose import media
    from codai.compose import render as rnd
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.mp4")
        subprocess.run([_ffmpeg(), "-y", "-f", "lavfi", "-i",
                        "color=c=green:s=320x240:r=24:d=6",
                        "-pix_fmt", "yuv420p", src],
                       capture_output=True, check=True, timeout=120)
        out = rnd.cut_segment(src, os.path.join(d, "cut.mp4"), start=2.0,
                              duration=1.5, fps=24, width=320, height=240)
        assert abs(media.duration_of(out) - 1.5) < 0.12
