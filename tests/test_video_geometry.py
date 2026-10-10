"""What a video model can emit, published per model on /v1/models.

Three modules knew a piece of this and none of them could tell a client: the Wan VAE's
4k+1 grid lived in codai/api/video.py, MiniMax-H3's 17n+5 in tools/h3_common.py, and
LongCat's 93-frame segment in tools/longcat_common.py. A client planning video therefore
guessed from the model NAME and kept its own table — which is how the township tool ended
up tuned for a model it was no longer using.

codai/models/video_geometry.py is that knowledge once. Because the LongCat and H3 workers
run in their own 3.10 venvs and cannot be imported from here, the numbers are literals,
so the first tests below pin each one against the module that owns it: duplication is
tolerable, silent drift is not.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.models import video_geometry as G


def _by_path(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


LC = _by_path("lc_common_geom", "tools/longcat_common.py")
H3 = _by_path("h3_common_geom", "tools/h3_common.py")


# --------------------------------------------------------------- pinned to source
def test_longcat_numbers_come_from_longcat_common():
    g = G.FAMILIES["longcat"]
    assert g["native_fps"] == LC.DEFAULT_FPS
    assert g["segment_frames"] == LC.DEFAULT_NUM_FRAMES
    assert g["cond_frames"] == LC.DEFAULT_COND_FRAMES
    assert g["frame_base"] == LC.DEFAULT_NUM_FRAMES
    # A continuation re-renders the tail, so a segment only ADDS num - cond.
    assert g["frame_step"] == LC.DEFAULT_NUM_FRAMES - LC.DEFAULT_COND_FRAMES


def test_the_longcat_grid_is_what_the_segment_loop_actually_yields():
    g = G.FAMILIES["longcat"]
    for segments in (1, 2, 3, 4, 7):
        assert G.frames_for(segments, g) == LC.frames_for(segments)
        assert G.snap_frames(LC.frames_for(segments), g) == LC.frames_for(segments)


def test_every_longcat_grid_point_is_a_count_bsa_can_tile():
    """80 divides by 16, so 93 + 80k is always 16n+13 — the counts bsa_problems accepts.

    This is the whole reason snapping to the segment grid is worth doing: a count off
    the grid drops the pass onto dense attention, which is slower and OOMed a 24GB card.
    """
    g = G.FAMILIES["longcat"]
    for k in range(8):
        frames = g["frame_base"] + g["frame_step"] * k
        assert not LC.bsa_problems(frames, 512, 832), frames


def test_h3_numbers_come_from_h3_common():
    g = G.FAMILIES["h3"]
    assert g["native_fps"] == H3.H3_FPS
    assert g["frame_step"] == H3.H3_FRAMES_CHUNK
    assert g["frame_base"] == H3.H3_FRAMES_REMAINDER
    assert g["side_multiple"] == H3.H3_CANVAS_MULTIPLE
    assert g["min_frames"] == H3.snap_frames(H3.H3_MIN_SECONDS * H3.H3_FPS)
    assert g["max_frames"] == H3.snap_frames(H3.H3_MAX_SECONDS * H3.H3_FPS)


@pytest.mark.parametrize("n", [1, 120, 150, 200, 345, 360, 400])
def test_h3_snapping_agrees_with_the_worker(n):
    """Up, not nearest: the worker snaps a request UP to what its VAE can decode."""
    assert G.snap_frames(n, G.FAMILIES["h3"], mode="up") == H3.snap_frames(n)


@pytest.mark.parametrize("n", [1, 5, 6, 48, 50, 81, 250])
def test_wan_snapping_agrees_with_the_render_path(n):
    from codai.api.video import _snap_wan_frames
    assert G.snap_frames(n, G.FAMILIES["wan"]) == _snap_wan_frames(n)


def test_wan_holds_eighty_one_frames_in_one_call():
    """Past ~81 the frames visibly jump, which is why longer clips are chained."""
    assert G.FAMILIES["wan"]["max_frames_per_render"] == 81
    assert G.FAMILIES["wan"]["continuation"] == "chained"


# ------------------------------------------------------------------ family match
@pytest.mark.parametrize("name,cfg,family", [
    ("longcat", {}, "longcat"),
    ("LongCat-Video-Avatar", {}, "longcat"),
    ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", {}, "wan"),
    ("MiniMax-H3", {}, "h3"),
    ("minimax_h3_fast", {}, "h3"),
    ("something-else", {}, "wan"),
])
def test_the_family_is_recognised_by_name(name, cfg, family):
    assert G.family_for(name, cfg) == family


def test_a_backend_pin_beats_the_name():
    """An entry aliased over an HF repo id is routed by its backend, so it must be
    described by its backend too — geometry and routing cannot be allowed to disagree."""
    assert G.family_for("my-video-model", {"backend": "longcat"}) == "longcat"
    assert G.family_for("my-video-model", {"backend": "h3"}) == "h3"


def test_an_alias_is_enough():
    assert G.family_for("org/checkpoint-42", {"alias": "longcat"}) == "longcat"


def test_the_raw_config_is_unwrapped():
    """build_runtime_kwargs drops every engine-specific key except inside _raw_cfg,
    which is exactly where a backend pin ends up on the video path."""
    geom = G.geometry_for("org/x", {"_raw_cfg": {"backend": "longcat"}})
    assert geom["family"] == "longcat"


def test_an_unknown_family_degrades_to_the_common_vae_rule():
    geom = G.geometry_for("brand-new-video-model", {})
    assert geom["family"] == G.DEFAULT_FAMILY
    assert (geom["frame_base"], geom["frame_step"]) == (1, 4)


# --------------------------------------------------------------- config overrides
def test_config_overrides_win():
    """Config is the source of truth: a checkpoint this table has never heard of is
    configurable rather than permanently mis-described."""
    geom = G.geometry_for("mystery", {
        "video_native_fps": 30, "video_frame_base": 9, "video_frame_step": 8,
        "video_max_frames_per_render": 200, "video_side_multiple": 64})
    assert geom["native_fps"] == 30
    assert G.snap_frames(100, geom) == 97      # 9 + 8k
    assert geom["max_frames_per_render"] == 200
    assert geom["side_multiple"] == 64


def test_a_junk_override_is_ignored_not_fatal():
    geom = G.geometry_for("longcat", {"video_native_fps": "soon"})
    assert geom["native_fps"] == G.FAMILIES["longcat"]["native_fps"]


def test_geometry_for_does_not_mutate_the_table():
    G.geometry_for("longcat", {"video_native_fps": 7})
    assert G.FAMILIES["longcat"]["native_fps"] == LC.DEFAULT_FPS


# ------------------------------------------------------------------- the maths
def test_snapping_never_lands_below_the_minimum():
    for fam in G.FAMILIES.values():
        assert G.snap_frames(0, fam) >= fam["min_frames"]
        assert G.snap_frames(-5, fam) >= fam["min_frames"]


def test_snapping_respects_a_ceiling():
    assert G.snap_frames(10_000, G.FAMILIES["h3"]) <= G.FAMILIES["h3"]["max_frames"]
    # LongCat has none: the server runs the segment loop as long as asked.
    assert G.FAMILIES["longcat"]["max_frames"] is None
    assert G.snap_frames(10_000, G.FAMILIES["longcat"]) > 1000


def test_nonsense_frames_do_not_raise():
    assert G.snap_frames(None, G.FAMILIES["wan"]) >= 5
    assert G.snap_frames("many", G.FAMILIES["wan"]) >= 5


def test_segments_and_frames_are_inverses():
    for fam in ("longcat", "wan"):
        g = G.FAMILIES[fam]
        for k in range(1, 6):
            assert G.segments_for(G.frames_for(k, g), g) == k


def test_seconds_become_a_legal_count_at_the_asked_rate():
    lc = G.FAMILIES["longcat"]
    assert G.seconds_to_frames(11.5, lc) == 173        # two segments at 15fps
    assert G.seconds_to_frames(22.0, lc) == 333        # four
    wan = G.FAMILIES["wan"]
    assert G.seconds_to_frames(6.25, wan, fps=8) == 49
    # The POINT of seconds: the same scene length at a different rate is more frames.
    assert G.seconds_to_frames(6.25, wan, fps=16) == 101


def test_seconds_default_to_the_models_own_rate():
    lc = G.FAMILIES["longcat"]
    assert G.seconds_to_frames(11.5, lc) == G.seconds_to_frames(11.5, lc, fps=15)


# ---------------------------------------------------------------- what's published
def test_the_listing_can_carry_it():
    from codai.pydantic.textrequest import ModelInfo
    geom = G.geometry_for("longcat", {})
    info = ModelInfo(id="longcat", type="video",
                     capabilities=["video_generation"], video=geom)
    assert info.model_dump()["video"]["frame_step"] == 80


def test_a_non_video_model_carries_nothing():
    from codai.pydantic.textrequest import ModelInfo
    assert ModelInfo(id="some-llm", type="text").video is None


def test_the_manager_publishes_geometry_for_video_models():
    """Read from the source: building a real manager needs a GPU and a config tree."""
    src = (ROOT / "codai/models/manager.py").read_text()
    assert "from codai.models import video_geometry" in src
    assert "video=video_geom," in src
    assert 'if resolved_type == "video" or "video_generation" in caps.to_list():' in src


def test_the_render_path_snaps_with_the_published_grid():
    """Otherwise what the server advertises and what it rounds to could drift."""
    src = (ROOT / "codai/api/video.py").read_text()
    body = src.split("def _snap_wan_frames", 1)[1].split("\ndef ", 1)[0]
    assert "video_geometry.snap_frames" in body
    assert "return max(5, k * 4 + 1)" in body, "the inline fallback must stay"
