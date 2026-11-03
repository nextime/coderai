"""A distilled video checkpoint must not be served at the generic defaults.

The image path has consulted the model's NAME for a step/guidance pair since
0.5.x (``distilled_defaults_for``); the video path never did. A Lightning or
distilled checkpoint whose models.json entry carries no ``defaults`` therefore
ran at 25 steps and CFG 7.5 -- three times the work, and a distilled
transformer at CFG 7.5 produces exactly the fault reported 2026-11-03 across
the text-to-video catalogue: subjects merged into each other, a dog fused into
the girl, "nothing recognisable, just a mix of colors".

The order is the point: the request wins, then a fused distill preset, then the
card, then the name, then the floor.
"""
import inspect

from codai.api import video as V
from codai.models.acceleration import distilled_defaults_for


SRC = inspect.getsource(V._generate_video)


def test_the_video_path_consults_the_name():
    assert "distilled_defaults_for" in SRC


def test_it_sits_below_the_card_and_above_the_floor():
    i_card = SRC.index("_card = model_defaults")
    i_dist = SRC.index("distilled_defaults_for")
    i_floor = SRC.index("kw.setdefault('num_inference_steps', 25)")
    assert i_card < i_dist < i_floor


def test_it_cannot_override_what_the_request_asked_for():
    """setdefault, not assignment -- _build_call_kwargs has already put the
    request's own steps in kw."""
    i = SRC.index("distilled_defaults_for")
    window = SRC[i:i + 900]
    assert "kw.setdefault('num_inference_steps'" in window
    assert "kw['num_inference_steps'] =" not in window


def test_a_failed_lookup_cannot_fail_the_generation():
    i = SRC.index("distilled_defaults_for")
    assert "except Exception" in SRC[i:i + 600]


def test_the_video_families_are_actually_recognised():
    """The table is shared with the image path; these are the ids from the
    report that must resolve."""
    assert distilled_defaults_for("Wan2.2-I2V-A14B-Lightning")["steps"] == 8
    assert distilled_defaults_for("Wan2.2-I2V-A14B-Lightning")["guidance_scale"] == 1.0
    assert distilled_defaults_for("Wan2.2-TI2V-5B") is None, (
        "a model that is not distilled must keep the generic defaults")
