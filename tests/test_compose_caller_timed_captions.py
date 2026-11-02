"""Captions timed by the caller (``scenes[i].words``).

A caller that synthesised a scene's voice elsewhere often already HAS the word
timings, or can get them from the provider that spoke the line. Compose used to
transcribe that audio again to recover timings it was never told about: a
speech model, a GPU and a load, for a worse answer -- STT guesses at the audio
while the caller knows what was said.

``scenes[i].words`` are seconds from the start of that scene's audio, the same
clock ``_stt_words`` reports on, so both reach ``align_words`` unchanged.
Absent or empty, nothing changes.
"""
import inspect

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from codai.api import compose as C


# ---------------------------------------------------------------- the model

def test_a_scene_accepts_timed_words():
    sc = C.Scene(text="hello world",
                 words=[{"word": "hello", "start": 0.0, "end": 0.4},
                        {"word": "world", "start": 0.45, "end": 0.9}])
    assert [w.word for w in sc.words] == ["hello", "world"]
    assert sc.words[1].start == 0.45


def test_a_scene_without_words_is_unchanged():
    """The whole feature must be invisible to every existing caller."""
    assert C.Scene(text="hi").words == []


def test_the_caption_spec_still_defaults_to_auto():
    assert C.CaptionSpec().timing == "auto"


# ------------------------------------------------- normalising a scene's words

def test_blank_words_are_dropped():
    """A blank word would consume a caller word and give it no time."""
    sc = C.Scene(words=[{"word": "a", "start": 0, "end": 1},
                        {"word": "   ", "start": 1, "end": 2},
                        {"word": "", "start": 2, "end": 3}])
    assert [w["word"] for w in C._caller_words(sc)] == ["a"]


def test_a_reversed_span_is_squared_up_not_refused():
    """These come from a provider, not a person: one odd row costs that row's
    precision, not the scene's captions."""
    sc = C.Scene(words=[{"word": "x", "start": 2.0, "end": 1.0}])
    got = C._caller_words(sc)
    assert got == [{"word": "x", "start": 2.0, "end": 2.0}]


def test_a_scene_with_no_words_normalises_to_nothing():
    assert C._caller_words(C.Scene(text="hi")) == []


def test_an_unparseable_time_is_refused_at_the_boundary():
    """Better than dropping the word quietly: the caller is told which row of
    which scene it got wrong, as a 422, before any rendering starts."""
    with pytest.raises(ValidationError) as err:
        C.Scene(words=[{"word": "good", "start": 0.0, "end": 0.5},
                       {"word": "bad", "start": "later", "end": 9.0}])
    assert "words.1.start" in str(err.value)


def test_the_normaliser_is_still_defensive_about_times():
    """It reads with getattr, so it is reachable with objects pydantic never
    vetted; a bad row must cost its own word and nothing else."""
    class _W:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end

    class _S:
        words = [_W("good", 0.0, 0.5), _W("bad", "later", 9.0)]

    assert [w["word"] for w in C._caller_words(_S())] == ["good"]


# --------------------------------------------------- slicing a dialogue line

SCENE_WORDS = [
    {"word": "who", "start": 0.10, "end": 0.30},
    {"word": "goes", "start": 0.35, "end": 0.60},
    {"word": "there", "start": 0.65, "end": 0.95},
    {"word": "it", "start": 2.10, "end": 2.25},
    {"word": "is", "start": 2.30, "end": 2.50},
    {"word": "me", "start": 2.55, "end": 2.80},
]


def test_a_line_takes_only_its_own_window():
    got = C._caller_words_for_line(SCENE_WORDS, 0.0, 1.2)
    assert [w["word"] for w in got] == ["who", "goes", "there"]


def test_a_line_is_rebased_onto_its_own_clock():
    """align_words offsets by the line's start, so the words it is handed must
    be line-relative -- exactly what _stt_words would have returned for that
    line's own audio file."""
    got = C._caller_words_for_line(SCENE_WORDS, 2.0, 1.0)
    assert [w["word"] for w in got] == ["it", "is", "me"]
    assert got[0]["start"] == pytest.approx(0.10)
    assert got[0]["end"] == pytest.approx(0.25)


def test_a_word_is_placed_by_its_start_alone():
    """A word running past the line's end still belongs to the line it began
    in. Testing both ends would drop the last word of every line whose speaker
    ran into the gap after it."""
    words = [{"word": "traaails", "start": 0.9, "end": 2.4}]
    got = C._caller_words_for_line(words, 0.0, 1.0)
    assert [w["word"] for w in got] == ["traaails"]
    assert got[0]["end"] == pytest.approx(2.4)


def test_a_line_with_nothing_in_its_window_gets_nothing():
    assert C._caller_words_for_line(SCENE_WORDS, 5.0, 1.0) == []


def test_the_window_is_half_open_so_no_word_lands_in_two_lines():
    """A word starting exactly on the boundary belongs to the later line, and
    to it only -- otherwise it is captioned twice."""
    words = [{"word": "edge", "start": 1.0, "end": 1.2}]
    first = C._caller_words_for_line(words, 0.0, 1.0)
    second = C._caller_words_for_line(words, 1.0, 1.0)
    assert first == []
    assert [w["word"] for w in second] == ["edge"]


def test_slicing_nothing_is_safe():
    assert C._caller_words_for_line([], 0.0, 1.0) == []
    assert C._caller_words_for_line(None, 0.0, 1.0) == []


# ------------------------------------------------------------- the validation

def _req(timing, words=None):
    """A request that is valid apart from what each test is about."""
    return C.ComposeRequest(
        # `audio` given, not `text` alone: a caller with its own word timings
        # synthesised the voice elsewhere, which is the whole situation this
        # feature is for -- and it means the request needs no TTS on this box.
        scenes=[C.Scene(text="hello world", duration=2.0,
                        audio="data:audio/wav;base64,AAAA",
                        words=words or [],
                        visuals=[{"type": "color", "color": "#000000"}])],
        captions=C.CaptionSpec(timing=timing))


@pytest.mark.parametrize("timing", ["auto", "caller", "estimate"])
def test_every_documented_timing_passes_validation(timing):
    """`caller` is new; the others must not have been broken by adding it.

    `stt` is left out: it legitimately 501s on an install with no speech
    model, which is a property of the box, not of this change.
    """
    C._validate(_req(timing))        # must not raise


def test_an_unknown_timing_is_refused_and_the_error_names_caller():
    """The error has to list the value a client is meant to use."""
    with pytest.raises(HTTPException) as err:
        C._validate(_req("whenever"))
    assert err.value.status_code == 400
    assert "caller" in str(err.value.detail)


def test_timing_caller_needs_no_speech_model():
    """The whole point on an install that has none: exact timings without one.

    `stt` is what 501s here; `caller` must not, whatever this box has.
    """
    C._validate(_req("caller", words=[{"word": "hello", "start": 0.0, "end": 0.4},
                                      {"word": "world", "start": 0.5, "end": 0.9}]))


# ----------------------------------------------------- what the job does with it

def test_the_job_skips_speech_to_text_when_every_scene_brought_words():
    """The point of the feature: no model is picked when nothing needs one.

    Read as source because reaching that line means ffmpeg and a real render.
    """
    src = inspect.getsource(C._run_job)
    assert "_needs_stt" in src, "the job must work out which scenes need STT"
    assert "and _needs_stt else None" in src, (
        "_pick_stt_model must be conditional on a scene actually needing it")


def test_the_job_reports_how_each_scene_was_timed():
    src = inspect.getsource(C._run_job)
    assert "caption_timing" in src
    for value in ('"caller"', '"stt"', '"estimated"'):
        assert value in src, f"{value} must be a reportable timing"


def test_the_job_prefers_caller_words_over_transcribing():
    src = inspect.getsource(C._run_job)
    assert "if caller_words.get(i):" in src, (
        "the narration path must take caller words before reaching _stt_words")
