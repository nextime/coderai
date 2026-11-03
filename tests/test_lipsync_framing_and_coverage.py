"""Two lip-sync defects that are both one missing argument.

Measured 2026-11-03 on a talking-presenter run:

* **sadtalker** animated the whole face well (5/5) but "crops the picture to a
  close-up of the face". ``--preprocess`` defaults to ``crop``, which renders
  and returns only the crop; ``full`` pastes the animated face back into the
  frame the caller sent.
* **wav2lip**: "the lip sync is fantastic, but the generated mouth doesn't
  fully cover the original static mouth, so both show" (4/5). Upstream's
  default ``--pads 0 10 0 0`` leaves ten pixels of chin, and the composite then
  shows the synced mouth over the still one underneath.

Both call sites pass the arguments through to the same ``inference.py``, so the
fix is one helper per engine and the tests are about what reaches the command
line.
"""
import inspect

from codai.api import lipsync as L


# ------------------------------------------------------------------ wav2lip

def test_the_mouth_gets_enough_chin_to_cover_the_old_one(monkeypatch):
    monkeypatch.delenv("CODERAI_WAV2LIP_PADS", raising=False)
    assert L.wav2lip_extra_args() == ["--pads", "0", "20", "0", "0"]


def test_an_operator_can_retune_the_padding(monkeypatch):
    monkeypatch.setenv("CODERAI_WAV2LIP_PADS", "0 30 5 5")
    assert L.wav2lip_extra_args() == ["--pads", "0", "30", "5", "5"]
    monkeypatch.setenv("CODERAI_WAV2LIP_PADS", "0,25,0,0")
    assert L.wav2lip_extra_args() == ["--pads", "0", "25", "0", "0"]


def test_an_empty_override_leaves_upstreams_own_default(monkeypatch):
    """"Pass nothing" has to be expressible, or an install where 20 is wrong
    has no way back."""
    monkeypatch.setenv("CODERAI_WAV2LIP_PADS", "")
    assert L.wav2lip_extra_args() == []


def test_a_malformed_override_is_ignored_rather_than_passed_on(monkeypatch):
    """Four numbers or nothing: three would be a usage error from inference.py
    minutes into a render."""
    monkeypatch.setenv("CODERAI_WAV2LIP_PADS", "0 20 0")
    assert L.wav2lip_extra_args() == []


def test_the_managed_runner_passes_them():
    src = inspect.getsource(L.run_wav2lip)
    assert "wav2lip_extra_args()" in src


# ---------------------------------------------------------------- sadtalker

def test_the_portrait_is_pasted_back_into_its_own_frame(monkeypatch):
    monkeypatch.delenv("CODERAI_SADTALKER_ARGS", raising=False)
    assert L.sadtalker_extra_args() == ["--preprocess", "full"]


def test_head_motion_is_not_suppressed(monkeypatch):
    """--still would freeze the head, and moving it is the one thing this
    engine does that the video-driven ones do not."""
    monkeypatch.delenv("CODERAI_SADTALKER_ARGS", raising=False)
    assert "--still" not in L.sadtalker_extra_args()


def test_the_whole_argument_list_can_be_replaced(monkeypatch):
    monkeypatch.setenv("CODERAI_SADTALKER_ARGS", "--preprocess full --still")
    assert L.sadtalker_extra_args() == ["--preprocess", "full", "--still"]


# ------------------------------------------------------------- both callers

def test_the_video_endpoint_passes_both():
    from codai.api import video as V
    src = inspect.getsource(V._apply_lipsync)
    assert "wav2lip_extra_args()" in src
    assert "sadtalker_extra_args()" in src


def test_the_presenter_passes_both():
    from codai.compose import presenter as P
    src = inspect.getsource(P)
    assert "wav2lip_extra_args()" in src
    assert "sadtalker_extra_args()" in src
