"""Township clip lengths sized by the MODEL, not by a number typed once for Wan.

The tool was tuned when the only video model could hold ~81 frames in a single call, so
a 70-second match became a dozen short cuts. LongCat was pretrained on continuation and
the server renders a long shot in segments, so the same match wants a handful of long
takes — and the frame numbers that suit one are wrong for the other by a factor of four.
The templates knew this; nothing else did, and nothing snapped a picked count onto what
the model can actually emit, so `random.randint` handed the server lengths like 250 and
the server quietly rounded them.

So: 0 means AUTO (derive from the model's geometry and this tool's seconds band), an
explicit number still wins, and every picked count lands on the model's grid.
"""

import importlib.util
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _tool():
    """Import the tool. It does sys.modules[__name__] at import time, so it has to be
    registered before exec_module."""
    spec = importlib.util.spec_from_file_location(
        "township_geom_tool", ROOT / "tools" / "gen_township_fighters.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["township_geom_tool"] = m
    spec.loader.exec_module(m)
    return m


TW = _tool()
SRC = (ROOT / "tools/gen_township_fighters.py").read_text()

WAN = TW.video_geometry("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
LONGCAT = TW.video_geometry("longcat")
H3 = TW.video_geometry("MiniMax-H3")


class _FakeClient:
    """Stands in for CoderAIClient: /v1/models, and nothing else."""

    def __init__(self, models, boom=False):
        self._models = models
        self.boom = boom
        self.calls = 0

    def list_models(self):
        self.calls += 1
        if self.boom:
            raise RuntimeError("engine is loading")
        return self._models

    def list_video_models(self):
        return [m for m in self.list_models()
                if "video_generation" in (m.get("capabilities") or [])]


@pytest.fixture(autouse=True)
def _clear_caches():
    TW._GEOM_CACHE.clear()
    TW._AUTO_VIDEO_MODEL.clear()
    yield
    TW._GEOM_CACHE.clear()
    TW._AUTO_VIDEO_MODEL.clear()


# ------------------------------------------------------------------ where it comes from
def test_the_server_is_asked_first():
    """The server knows what it actually loaded; a name is a guess. A checkpoint whose
    alias this tool has never heard of is exactly the case that used to go wrong."""
    client = _FakeClient([{"id": "house-style-v3",
                           "capabilities": ["video_generation"],
                           "video": {"family": "longcat", "native_fps": 15,
                                     "frame_base": 93, "frame_step": 80,
                                     "min_frames": 93, "max_frames": None,
                                     "max_frames_per_render": 93, "cond_frames": 13,
                                     "side_multiple": 64, "continuation": "native"}}])
    geom = TW.video_geometry("house-style-v3", client)
    assert geom["frame_step"] == 80
    assert geom["continuation"] == "native"
    # Name matching alone would have called this a Wan model.
    assert TW._local_geometry_for("house-style-v3")["family"] == "wan"


def test_an_older_front_falls_back_to_the_local_table():
    """No published geometry, or no server at all: a plan must still be plannable."""
    client = _FakeClient([{"id": "longcat", "capabilities": ["video_generation"]}])
    assert TW.video_geometry("longcat", client)["frame_step"] == 80
    assert TW.video_geometry("longcat", _FakeClient([], boom=True))["frame_step"] == 80
    assert TW.video_geometry("longcat", None)["frame_step"] == 80


def test_a_published_geometry_missing_the_grid_is_not_trusted():
    client = _FakeClient([{"id": "longcat", "capabilities": ["video_generation"],
                           "video": {"native_fps": 15}}])
    assert TW.video_geometry("longcat", client)["frame_step"] == 80


def test_geometry_is_fetched_once_per_model():
    client = _FakeClient([{"id": "longcat", "capabilities": ["video_generation"]}])
    TW.video_geometry("longcat", client)
    TW.video_geometry("longcat", client)
    assert client.calls == 1, "the planner asks per match; /v1/models is one call"


def test_the_cached_geometry_cannot_be_mutated_by_a_caller():
    client = _FakeClient([{"id": "longcat", "capabilities": ["video_generation"]}])
    TW.video_geometry("longcat", client)["frame_step"] = 1
    assert TW.video_geometry("longcat", client)["frame_step"] == 80


def test_a_blank_model_on_the_run_page_resolves_to_what_the_server_would_pick():
    """The Run page leaves video_model blank to mean auto-select, and planning a blank
    model with Wan's geometry is the whole bug."""
    class _Args:
        video_model = ""
    client = _FakeClient([{"id": "longcat", "capabilities": ["video_generation"]}])
    assert TW.resolved_video_model(_Args(), client) == "longcat"
    assert TW.web_video_geometry(_Args(), client)["family"] == "longcat"


def test_an_explicit_model_is_not_second_guessed():
    class _Args:
        video_model = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    client = _FakeClient([{"id": "longcat", "capabilities": ["video_generation"]}])
    assert TW.resolved_video_model(_Args(), client).startswith("Wan-AI")


# --------------------------------------------------------------------- auto ranges
def test_zero_means_auto_and_a_number_still_wins():
    assert TW._clip_frame_range(0, 0, LONGCAT, "clip", 15) == (173, 333)
    assert TW._clip_frame_range(50, 70, LONGCAT, "clip", 15) == (50, 70)
    # A saved config from before auto existed keeps the exact budget it had.
    assert TW._clip_frame_range(50, 70, WAN, "clip", 8) == (50, 70)


def test_one_end_can_be_auto():
    lo, hi = TW._clip_frame_range(0, 200, LONGCAT, "clip", 15)
    assert (lo, hi) == (173, 200)


def test_wan_auto_lands_on_the_historical_budget():
    """Nothing should change shape for a run that was already tuned for Wan."""
    assert TW._clip_frame_range(0, 0, WAN, "clip", 8) == (49, 69)


def test_longcat_auto_is_the_segment_grid():
    lo, hi = TW._clip_frame_range(0, 0, LONGCAT, "clip", 15)
    assert (lo - 93) % 80 == 0 and (hi - 93) % 80 == 0
    assert lo == 173 and hi == 333


def test_the_intro_gets_its_own_shorter_budget():
    """Entrances and the face-off are not fight clips. Sharing the fight budget made
    every LongCat entrance an 11-22 s take."""
    fight = TW._clip_frame_range(0, 0, LONGCAT, "clip", 15)
    intro = TW._clip_frame_range(0, 0, LONGCAT, "intro", 15)
    assert intro[1] < fight[0]
    assert intro == (93, 93), "one segment is the shortest honest LongCat ask"
    w_fight = TW._clip_frame_range(0, 0, WAN, "clip", 8)
    w_intro = TW._clip_frame_range(0, 0, WAN, "intro", 8)
    assert w_intro[1] < w_fight[1]


def test_an_explicit_clip_range_still_governs_the_intro():
    """Changing that for someone who typed numbers in would be a surprise."""
    class _Args:
        clip_min_frames, clip_max_frames = 50, 70
    assert TW._intro_frame_range(_Args(), LONGCAT, 15, (50, 70)) == (50, 70)

    class _Auto:
        clip_min_frames, clip_max_frames = 0, 0
    assert TW._intro_frame_range(_Auto(), LONGCAT, 15, (173, 333)) == (93, 93)


def test_the_band_is_seconds_so_the_rate_moves_the_frames():
    """Pin frames instead and every scene silently changes duration when fps does."""
    lo8, hi8 = TW._clip_frame_range(0, 0, WAN, "clip", 8)
    lo16, hi16 = TW._clip_frame_range(0, 0, WAN, "clip", 16)
    assert lo16 > lo8 and hi16 > hi8
    assert abs(lo16 / 16 - lo8 / 8) < 0.6, "the same seconds, not the same frames"


def test_a_bad_range_cannot_break_the_planner():
    lo, hi = TW._clip_frame_range(300, 100, WAN, "clip", 8)
    assert lo <= hi, "random.randint requires lo <= hi"
    assert TW._clip_frame_range("x", None, WAN, "clip", 8) == (49, 69)
    assert TW._clip_frame_range(0, 0, None, "clip", 8) == (49, 69)


def test_auto_is_bounded_by_the_whole_plan_ceiling():
    for geom in (WAN, LONGCAT, H3):
        for kind in ("clip", "intro", "outcome"):
            lo, hi = TW._clip_frame_range(0, 0, geom, kind, None)
            assert 8 <= lo <= hi <= TW.MAX_PLANNED_FRAMES


# ------------------------------------------------------------------------ snapping
@pytest.mark.parametrize("n", [1, 100, 173, 250, 300, 333, 400])
def test_a_picked_count_lands_on_the_models_grid(n):
    out = TW._snap_frames(n, LONGCAT)
    assert (out - 93) % 80 == 0 and out >= 93
    assert TW._snap_frames(n, WAN) % 4 == 1


def test_snapping_is_idempotent():
    for geom in (WAN, LONGCAT, H3):
        once = TW._snap_frames(207, geom)
        assert TW._snap_frames(once, geom) == once


def test_snapping_respects_a_models_ceiling():
    assert TW._snap_frames(10_000, H3) <= H3["max_frames"]


def test_snapping_without_geometry_assumes_the_common_vae_rule():
    assert TW._snap_frames(250, None) % 4 == 1


def test_the_grid_is_also_what_keeps_block_sparse_attention_on():
    """93+80k is always 16n+13, the counts bsa_problems accepts. Landing off the grid
    is what silently drops a clip onto dense attention."""
    spec = importlib.util.spec_from_file_location(
        "lc_common_township", ROOT / "tools/longcat_common.py")
    lc = importlib.util.module_from_spec(spec)
    sys.modules["lc_common_township"] = lc
    spec.loader.exec_module(lc)
    for n in (100, 207, 250, 300):
        assert not lc.bsa_problems(TW._snap_frames(n, LONGCAT), 512, 832)


# ------------------------------------------------------------------- the clip plan
def _plan(geom, fps, target=70.0, seed=11):
    random.seed(seed)
    lo, hi = TW._clip_frame_range(0, 0, geom, "clip", fps)
    intro = TW._clip_frame_range(0, 0, geom, "intro", fps)
    return TW._build_match_clip_specs(fps, lo, hi, target, "A", "B",
                                      geom=geom, intro_range=intro)


def test_longcat_fills_the_same_match_with_far_fewer_clips():
    """The point of the whole change: fewer, longer takes for the same long cut."""
    wan = [c for c in _plan(WAN, 8) if c["role"] == "fight"]
    lc = [c for c in _plan(LONGCAT, 15) if c["role"] == "fight"]
    assert len(lc) * 2 <= len(wan), (len(lc), len(wan))
    assert len(lc) <= 6


def test_the_long_target_is_still_filled():
    for geom, fps in ((WAN, 8), (LONGCAT, 15), (H3, 24)):
        fight = [c for c in _plan(geom, fps) if c["role"] == "fight"]
        assert sum(c["clip_seconds"] for c in fight) >= 70.0


def test_every_planned_clip_is_a_count_the_model_can_emit():
    for geom, fps in ((WAN, 8), (LONGCAT, 15), (H3, 24)):
        for c in _plan(geom, fps):
            assert TW._snap_frames(c["nf"], geom) == c["nf"], (geom["family"], c)


def test_the_intro_is_three_clips_and_shorter_than_the_fight():
    specs = _plan(LONGCAT, 15)
    intro = [c for c in specs if c["role"] in ("entrance", "faceoff")]
    fight = [c for c in specs if c["role"] == "fight"]
    assert len(intro) == 3
    assert max(c["nf"] for c in intro) <= min(c["nf"] for c in fight)


def test_the_intro_does_not_count_toward_the_long_target():
    specs = _plan(LONGCAT, 15, target=30.0)
    fight = [c for c in specs if c["role"] == "fight"]
    assert sum(c["clip_seconds"] for c in fight) >= 30.0


def test_without_an_intro_range_the_clip_range_is_used():
    """Back-compat: the signature grew two optional arguments."""
    random.seed(3)
    specs = TW._build_match_clip_specs(8, 50, 70, 20.0, "A", "B")
    assert all(50 <= c["nf"] <= 70 for c in specs)


# ----------------------------------------------------------------- outcome videos
class _Prompter:
    def outcome_shot(self, fighter, outcome, env, role="", opponent=None):
        return f"{role} shot"


def test_each_outcome_shot_is_snapped_and_the_total_is_rewritten():
    """A 60/40 split of a legal total is two counts that usually are not, and each
    shot is its own render."""
    o = {"fighter": "A", "outcome": "win", "opponent": "B", "nf": 253,
         "target_s": round(253 / 15, 2)}
    TW._plan_outcome_shots(_Prompter(), o, {}, "B", geom=LONGCAT)
    assert [sh["nf"] for sh in o["shots"]] == [173, 93]
    assert o["nf"] == 266, "the plan says what will actually be rendered"
    assert o["target_s"] == pytest.approx(266 / 15, abs=0.2)


def test_without_geometry_the_split_is_untouched():
    o = {"fighter": "A", "outcome": "win", "opponent": "B", "nf": 100}
    TW._plan_outcome_shots(_Prompter(), o, {}, "B")
    assert sum(sh["nf"] for sh in o["shots"]) == 100
    assert o["nf"] == 100


def test_a_draw_is_still_two_shots():
    o = {"fighter": "A", "outcome": "draw", "opponent": "B", "nf": 173}
    TW._plan_outcome_shots(_Prompter(), o, {}, "B", geom=LONGCAT)
    assert [sh["role"] for sh in o["shots"]] == ["final_exchange", "draw_decision"]
    assert all(TW._snap_frames(sh["nf"], LONGCAT) == sh["nf"] for sh in o["shots"])


# -------------------------------------------------------------- the per-render cap
def test_the_cap_comes_from_the_model():
    assert TW._single_render_cap(0, H3) == H3["max_frames_per_render"]
    assert TW._single_render_cap(0, LONGCAT) == 93


def test_wan_keeps_chaining_at_fifty():
    """81 is where coherence breaks; 50 is where it still looks good, and that has
    been the shipped behaviour since the beginning."""
    assert TW._single_render_cap(0, WAN) == TW.SINGLE_CLIP_MAX_FRAMES == 50


def test_an_explicit_cap_is_bounded_by_one_render():
    assert TW._single_render_cap(120, WAN) == TW.MODEL_MAX_FRAMES
    assert TW._single_render_cap(120, H3) == 120
    assert TW._single_render_cap(30, WAN) == 30


def test_a_natively_continuing_model_is_never_chained_from_here():
    """Chaining LongCat would reintroduce exactly the drift its continuation avoids."""
    body = SRC.split("# Max frames per SINGLE model generation", 1)[1][:1600]
    assert '_geom.get("continuation") == "native"' in body
    assert "_chunk_max = 1 << 30" in body


def test_the_tail_length_comes_from_the_geometry():
    assert 'int(_geom.get("cond_frames") or 0) or LONGCAT_COND_FRAMES' in SRC


# ------------------------------------------------------------------ size sanity
def test_a_size_the_model_cannot_tile_is_called_out():
    warn = TW._size_warning("832x480", LONGCAT)
    assert "64" in warn and "832x512" in warn, warn
    assert TW._size_warning("832x512", LONGCAT) == ""
    assert TW._size_warning("832x480", WAN) == ""


def test_a_junk_size_is_not_an_error():
    assert TW._size_warning("", LONGCAT) == ""
    assert TW._size_warning("wide", LONGCAT) == ""


def test_the_size_is_never_changed_behind_the_operators_back():
    """Resolution has cost and aspect consequences; it is an explicit choice."""
    assert "video_size =" not in SRC.split("def _size_warning", 1)[1][:900]


# ----------------------------------------------------------- defaults and plumbing
@pytest.mark.parametrize("field", [
    "clip_min_frames", "clip_max_frames", "single_clip_max_frames",
    "outcome_min_frames", "outcome_max_frames"])
def test_the_cli_defaults_to_auto(field):
    flag = "--" + field.replace("_", "-")
    block = SRC.split(f'"{flag}"', 1)[1][:200]
    assert "default=0" in block, block


@pytest.mark.parametrize("field", [
    "clip_min_frames", "clip_max_frames", "single_clip_max_frames",
    "outcome_min_frames", "outcome_max_frames"])
def test_the_run_page_defaults_to_auto(field):
    assert re.search(rf"name={field} type=number min=0[^>]*_v\('{field}', 0\)", SRC), field


@pytest.mark.parametrize("field", [
    "clip_min_frames", "clip_max_frames", "single_clip_max_frames",
    "outcome_min_frames", "outcome_max_frames"])
def test_both_form_parsers_default_to_auto(field):
    """Two of them, and a literal left in either one pins the old Wan number."""
    for m in re.finditer(rf'_fv\("{field}", "(\d+)"\)', SRC):
        assert m.group(1) == "0", (field, m.group(0))
    assert len(re.findall(rf'_fv\("{field}", "0"\)', SRC)) == 2, field


def test_the_run_page_explains_what_zero_does():
    assert "to size clips from the video" in SRC


def test_the_frame_fields_are_still_saved_in_a_config():
    for field in ("clip_min_frames", "clip_max_frames", "single_clip_max_frames",
                  "outcome_min_frames", "outcome_max_frames"):
        assert field in TW.CONFIG_FIELDS


def test_the_builtin_templates_no_longer_pin_frames():
    """They were the only model-aware thing in the tool; the model is now."""
    for name, tpl in TW.BUILTIN_TEMPLATES.items():
        for field in ("clip_min_frames", "clip_max_frames", "single_clip_max_frames",
                      "outcome_min_frames", "outcome_max_frames"):
            assert tpl[field] == 0, (name, field)
    # fps and size stay: they are editorial, and a model's native rate is not a
    # reason to re-encode an existing show at a different one.
    assert TW.BUILTIN_TEMPLATES["LongCat - few long scenes"]["fps"] == 15
    assert TW.BUILTIN_TEMPLATES["Wan - many short scenes"]["fps"] == 8


def test_every_planner_entry_point_is_model_aware():
    """Four places plan clips: the CLI run (which has its own inline loop), and the
    new-match, replan and full-re-render web jobs. One left name-blind plans the wrong
    lengths for that job only, which is invisible until the output looks wrong."""
    calls = [m for m in re.finditer(r"(def )?_build_match_clip_specs\(([^)]*)\)",
                                    SRC, re.S) if not m.group(1)]
    assert len(calls) == 3, f"{len(calls)} web call sites — did one appear or vanish?"
    for call in calls:
        assert "geom=" in call.group(2), call.group(2)
    # The CLI run plans inline rather than through the helper; both of its picks
    # (intro and fight) must still be snapped onto the model's grid.
    inline = SRC.split("def stage_videos", 1)[1].split("\ndef ", 1)[0]
    assert inline.count("_snap_frames(random.randint(") == 3, "intro, fight, outcome"
    assert "_geom = video_geometry(video_model, client)" in inline
    shots = [m for m in re.finditer(r"(def )?_plan_outcome_shots\(", SRC)
             if not m.group(1)]
    assert len(shots) == 5, f"{len(shots)} call sites"
    for call in shots:
        # Nested parens (o.get("opponent")) make the argument list awkward to match,
        # so look at the statement's first two lines.
        stmt = SRC[call.end():call.end() + 220].split("\n")[:2]
        assert "geom=" in " ".join(stmt), stmt


def test_the_render_phase_resolves_its_own_geometry():
    """It is a separate function from the planner and decides the chaining: a geometry
    resolved only in the planner is a NameError on the first render."""
    body = SRC.split("def _stage_videos_render", 1)[1].split("\ndef ", 1)[0]
    assert "_geom = video_geometry(video_model, client)" in body
    assert "_single_render_cap(single_clip_max_frames, _geom)" in body
    assert "single_clip_max_frames=0," in SRC, "the render default is auto too"


def test_the_web_render_path_does_not_force_the_wan_chunk():
    """Coercing a blank cap to 50 there would chain a model that never needs it."""
    assert 'scm = int(getattr(default_args, "single_clip_max_frames", 0) or 0)' in SRC


def test_the_clip_budget_is_logged():
    """An operator who does not know what auto chose cannot sanity-check a run."""
    assert "clip budget:" in SRC
