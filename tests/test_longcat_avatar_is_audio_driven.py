"""An avatar checkpoint cannot generate from a prompt, and must say so.

LongCat's avatar weights are a different pipeline class upstream: a
transformer plus an audio encoder, borrowing tokenizer/text_encoder/vae from
the base model, exposing only the audio-driven passes. A t2v or i2v request
reached base methods that class does not have, and the client got
"500 Internal Server Error" with no body at all -- measured 2026-11-03 on
'longcat-avatar', 11 requests (t2v, i2v and nine style clips), every one bare.

Two defects, one symptom: the mode could not work on these weights (so it is a
400, answerable before anything loads), and the worker's own account of a
failure was being dropped on the floor (so every other LongCat failure was
bare too).
"""
import inspect

from codai.api import video as V


SRC = inspect.getsource(V._generate_longcat)


def test_the_avatar_family_is_read_before_anything_loads():
    """The check must precede the payload and the service call -- the point is
    not to pay a multi-GB load to learn the mode was impossible."""
    i = SRC.index("family_of")
    assert i < SRC.index("payload = {")
    assert i < SRC.index("longcat_worker.generate")


def test_a_prompt_only_request_on_avatar_weights_is_a_client_error():
    i = SRC.index("family_of")
    window = SRC[i:i + 1400]
    assert "status_code=400" in window
    assert "at2v" in window and "ai2v" in window, (
        "the refusal must name the modes that DO work")
    assert "audio_file" in window, "and the field that carries the audio"


def test_the_base_checkpoint_refuses_the_audio_modes_for_the_same_reason():
    """The mirror image: a base entry asked for at2v has no audio encoder.
    Left unchecked it is the same bare 500 from the other direction."""
    i = SRC.index('if not _family and _task in ("at2v", "ai2v")')
    assert "status_code=400" in SRC[i:i + 500]


def test_the_family_check_cannot_itself_break_generation():
    """Reading the family is a string match on a path; if it ever raises, a
    working t2v request must not fail because of it."""
    i = SRC.index("_family = lc.family_of")
    assert "try:" in SRC[max(0, i - 200):i]


def test_a_worker_failure_reports_what_the_worker_said():
    i = SRC.index('task_registry.finish(_tid, "error", "generation failed")')
    window = SRC[i:i + 600]
    assert "_exc" in window, "the worker's text is the only account there is"
    assert "status_code=500" in window
    assert "raise\n" not in window.split("finally:")[0], (
        "a bare re-raise is what produced the empty 500")


def test_cancellation_still_answers_499():
    assert "status_code=499" in SRC
