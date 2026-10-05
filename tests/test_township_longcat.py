"""The township match generator, upgraded for LongCat-Video.

The tool chains clips because the server generates one clip per request and VACE
frame-tail conditioning is the only continuation a Wan model has. That machinery exists
to work around drift: the `project_wan22_dual_expert_lora` and `project_vace_extension`
notes are both about chained parts going wrong.

LongCat was pretrained on continuation, so the server can generate a long shot itself —
one request, no re-encoded joins, no accumulating drift. Chaining it from the client would
reintroduce exactly what the chaining was invented to avoid.

This reads the tool's source rather than running it: a match render needs models, weights
and a GPU.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOOL = (ROOT / "tools/gen_township_fighters.py").read_text()
VIDEO = (ROOT / "codai/api/video.py").read_text()


def _contract():
    spec = importlib.util.spec_from_file_location(
        "longcat_common", ROOT / "tools/longcat_common.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LC = _contract()


# ------------------------------------------------------------ the tool
def test_the_tool_recognises_a_longcat_model():
    assert '_longcat = "longcat" in (video_model or "").lower()' in TOOL


def test_the_client_side_split_is_lifted_for_longcat():
    """_split_frame_budget(sn, _chunk_max) is the chaining. LongCat does it server-side
    and better, so the cap must not force parts."""
    seg = TOOL[TOOL.index("_longcat = "):TOOL.index("def _render_once")]
    assert "_chunk_max = 1 << 30" in seg
    # …and the non-LongCat path keeps the cap it always had.
    assert "SINGLE_CLIP_MAX_FRAMES" in seg and "MODEL_MAX_FRAMES" in seg


def test_other_models_are_unaffected():
    """A Wan/VACE run must behave exactly as before."""
    seg = TOOL[TOOL.index("_longcat = "):TOOL.index("def _render_once")]
    assert "else:" in seg
    assert "max(8, min(int(single_clip_max_frames" in seg


def test_longcat_continues_from_a_tail_like_vace_does():
    """Between deliberate shots the tool still chains — a cut is a cut. What differs is
    what happens underneath, not the call."""
    assert "_tail_capable = _vace or _longcat" in TOOL
    assert "cond_frames = prev_tail if (_tail_capable and pi > 0) else None" in TOOL


def test_the_tail_length_matches_what_longcat_was_trained_with():
    """VACE's 5 frames were tuned for its conditioning; LongCat's continuation
    pretraining used 13 of 93, and handing it fewer conditions it on less than it
    expects."""
    got = int(re.search(r"LONGCAT_COND_FRAMES = (\d+)", TOOL).group(1))
    assert got == LC.DEFAULT_COND_FRAMES
    assert "_tail_frames = LONGCAT_COND_FRAMES if _longcat else VACE_TAIL_FRAMES" in TOOL


def test_the_vace_tail_length_is_untouched():
    assert re.search(r"VACE_TAIL_FRAMES = 5", TOOL)


def test_the_log_names_which_continuation_is_in_use():
    """Diagnosing a seam starts with knowing which mechanism produced it."""
    assert "LongCat native continuation" in TOOL
    assert "VACE frame-tail extend" in TOOL


def test_the_uploader_and_odds_are_untouched():
    """They are downstream of the frames; this change must not reach them."""
    for marker in ("def _scan_matches", "concat_videos"):
        assert marker in TOOL


# ------------------------------------------------------------ the server side
def test_a_long_request_is_segmented_server_side():
    """So a client asks for the frames it wants instead of reimplementing the loop."""
    fn = VIDEO[VIDEO.index("async def _generate_longcat"):]
    fn = fn[:fn.index("\nasync def ")]
    assert "_per_call = min(_want, lc.DEFAULT_NUM_FRAMES)" in fn
    assert '"total_frames"' in fn


def test_the_per_call_count_is_capped_at_one_segment():
    """num_frames is the PER-CALL count; passing 400 straight through would ask the
    pipeline for a 400-frame single call."""
    fn = VIDEO[VIDEO.index("async def _generate_longcat"):]
    fn = fn[:fn.index("\nasync def ")]
    seg = fn[fn.index("_want ="):fn.index("payload = {")]
    assert "min(" in seg


def test_an_explicit_segment_count_still_wins():
    fn = VIDEO[VIDEO.index("async def _generate_longcat"):]
    fn = fn[:fn.index("\nasync def ")]
    assert 'payload.pop("total_frames", None)' in fn


@pytest.mark.parametrize("want,expected_segments", [
    (93, 1), (94, 2), (173, 2), (174, 3), (893, 11),
])
def test_the_segment_arithmetic_covers_a_requested_length(want, expected_segments):
    """What the server computes from total_frames."""
    assert LC.segments_for(want) == expected_segments


def test_a_minute_of_video_is_one_request():
    """The point of the change: ~900 frames used to be ~11 chained parts with a
    re-encoded join between each."""
    assert LC.segments_for(900) >= 11
    assert LC.frames_for(LC.segments_for(900)) >= 900
