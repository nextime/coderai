"""Named configuration templates for the township match generator.

A saved config was always one JSON file, and --config/--save let you keep several by
juggling paths. A template is the same content under a NAME, in a known place, with
list/load/delete — so "the LongCat long-shot setup" is something you pick rather than a
path you remember.

Two properties matter beyond convenience:

- a template is a COMPLETE configuration (every CONFIG_FIELDS key), not a patch, so
  loading one cannot leave a stale option from the previous run in place;
- names arrive from the CLI *and from a browser*, so the name is a security boundary —
  it must not be able to address anything outside the templates directory.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _tool():
    """Import the tool. It does sys.modules[__name__] at import time, so it has to be
    registered before exec_module."""
    spec = importlib.util.spec_from_file_location(
        "township_tool", ROOT / "tools" / "gen_township_fighters.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["township_tool"] = m
    spec.loader.exec_module(m)
    return m


TW = _tool()
SRC = (ROOT / "tools/gen_township_fighters.py").read_text()


# ---------------------------------------------------------------- names
@pytest.mark.parametrize("raw,expected", [
    ("longcat long shot", "longcat long shot"),
    ("Quick-Draft_2", "Quick-Draft_2"),
    ("a/b\\c", "a-b-c"),
    ("../../etc/passwd", "etc-passwd"),
    ("name...with..dots", "name...with..dots"),
    ("  padded  ", "padded"),
])
def test_names_are_reduced_to_something_safe(raw, expected):
    assert TW._safe_template_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "...", "..", "/", "///", "---"])
def test_names_with_nothing_usable_are_refused(raw):
    with pytest.raises(ValueError):
        TW._safe_template_name(raw)


def test_a_name_cannot_escape_the_templates_directory(tmp_path):
    """The property that matters: whatever is asked for, the file stays inside."""
    for attempt in ("../../etc/passwd", "..\\..\\windows", "/etc/shadow", "a/../../b"):
        p = TW.template_path(str(tmp_path), attempt).resolve()
        assert p.parent == TW.templates_dir(str(tmp_path)).resolve(), attempt


def test_long_names_are_truncated():
    assert len(TW._safe_template_name("x" * 500)) <= 80


# ---------------------------------------------------------------- round trip
def test_a_template_round_trips(tmp_path):
    TW.save_template_dict(str(tmp_path), "t1",
                          {"video_model": "longcat", "num_fighters": 6})
    got = TW.load_template(str(tmp_path), "t1")
    assert got["video_model"] == "longcat" and got["num_fighters"] == 6


def test_unknown_keys_are_dropped_on_both_sides(tmp_path):
    """A template saved by another version has to stay usable."""
    TW.save_template_dict(str(tmp_path), "t", {"video_model": "wan", "bogus_key": 1})
    raw = json.loads(TW.template_path(str(tmp_path), "t").read_text())
    assert "bogus_key" not in raw
    assert "bogus_key" not in TW.load_template(str(tmp_path), "t")


def test_the_stamp_is_not_mistaken_for_an_option(tmp_path):
    TW.save_template_dict(str(tmp_path), "t", {"video_model": "wan"})
    raw = json.loads(TW.template_path(str(tmp_path), "t").read_text())
    assert "_template" in raw                      # written
    assert "_template" not in TW.load_template(str(tmp_path), "t")   # never applied


def test_a_missing_template_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        TW.load_template(str(tmp_path), "absent")


def test_listing_is_newest_first_with_enough_to_choose_by(tmp_path):
    TW.save_template_dict(str(tmp_path), "older",
                          {"video_model": "wan", "image_model": "sdxl"})
    TW.save_template_dict(str(tmp_path), "newer", {"video_model": "longcat"})
    items = TW.list_templates(str(tmp_path))
    assert {i["name"] for i in items} == {"older", "newer"}
    for i in items:
        assert i["saved_at"] and i["options"] >= 1
    assert [i["video_model"] for i in items if i["name"] == "older"] == ["wan"]


def test_listing_an_absent_directory_is_empty_not_an_error(tmp_path):
    assert TW.list_templates(str(tmp_path / "nope")) == []


def test_a_corrupt_template_is_listed_with_its_error(tmp_path):
    """One bad file must not hide the rest of the picker."""
    d = TW.templates_dir(str(tmp_path))
    d.mkdir(parents=True)
    (d / "broken.json").write_text("{not json")
    TW.save_template_dict(str(tmp_path), "fine", {"video_model": "wan"})
    items = {i["name"]: i for i in TW.list_templates(str(tmp_path))}
    assert "fine" in items and "broken" in items
    assert items["broken"].get("error")


def test_delete_reports_whether_anything_went(tmp_path):
    TW.save_template_dict(str(tmp_path), "t", {"video_model": "wan"})
    assert TW.delete_template(str(tmp_path), "t") is True
    assert TW.delete_template(str(tmp_path), "t") is False


# ---------------------------------------------------------------- completeness
def test_a_template_captures_every_config_field():
    """"All the configs" — a template is a complete configuration, so a later load
    cannot leave an option from the previous run behind."""
    class _Args:
        pass
    a = _Args()
    for k in TW.CONFIG_FIELDS:
        setattr(a, k, f"v-{k}")
    cfg = TW.config_from_args(a)
    assert set(cfg) == set(TW.CONFIG_FIELDS)


def test_saving_from_args_writes_them_all(tmp_path):
    class _Args:
        pass
    a = _Args()
    for k in TW.CONFIG_FIELDS:
        setattr(a, k, 1)
    TW.save_template(str(tmp_path), "full", a)
    assert set(TW.load_template(str(tmp_path), "full")) == set(TW.CONFIG_FIELDS)


def test_templates_live_beside_the_output_tree():
    """A template written to a container-local path would vanish on the next run; the
    out_dir is what the launcher maps and persists."""
    assert TW.templates_dir("/data/out").parent == Path("/data/out")


# ---------------------------------------------------------------- CLI + web wiring
@pytest.mark.parametrize("flag", ["--template", "--save-template", "--list-templates",
                                  "--delete-template"])
def test_the_cli_exposes_the_commands(flag):
    assert flag in SRC


def test_a_template_layers_over_a_config_file():
    """Precedence: config file, then template, then explicit arguments."""
    seg = SRC[SRC.index("if pre.template:"):SRC.index("args = parser.parse_args()")]
    assert "set_defaults(**tcfg)" in seg
    assert SRC.index("if pre.config:") < SRC.index("if pre.template:")


def test_the_existing_config_flags_still_work():
    """--config/--save predate this and must not change behaviour."""
    assert '"-c", "--config"' in SRC and '"-s", "--save"' in SRC
    assert "def save_config(" in SRC and "def load_config(" in SRC


def test_the_web_reuses_one_form_to_config_mapping():
    """The template save is handled inside /save-config rather than its own route, so
    the big form-to-config conversion is not duplicated and cannot drift."""
    seg = SRC[SRC.index('if path == "/save-config":'):SRC.index('if path == "/process":')]
    assert "save_template_dict(default_args.out_dir, tpl, cfg)" in seg
    assert seg.count("_collect_odds_ranges") == 1


@pytest.mark.parametrize("route", ['"/templates"', '"/templates/load"',
                                   '"/templates/delete"'])
def test_the_web_routes_exist(route):
    assert route in SRC


def test_loading_persists_to_the_active_config():
    """Live is not enough. The launcher starts the tool with
    --config <out_dir>/township_config.json, so a load that only changed the session
    would silently revert on the next restart — and _ref_gen_res_steps RE-READS that
    file for keyframe_size/keyframe_steps, so a stale file makes those two options
    disagree with the rest of the loaded template in the same session."""
    seg = SRC[SRC.index('if path in ("/templates/load"'):SRC.index('if path == "/process":')]
    assert "township_config.json" in seg
    assert 'open(_ap, "w"' in seg
    assert '"persisted"' in seg


def test_a_failed_persist_is_reported_not_swallowed():
    seg = SRC[SRC.index('if path in ("/templates/load"'):SRC.index('if path == "/process":')]
    assert "could not write" in seg


def test_loading_applies_to_the_live_session():
    """The Run page renders from default_args, so a load that only wrote a file would
    appear to do nothing until restart."""
    seg = SRC[SRC.index('if path in ("/templates/load"'):SRC.index('if path == "/process":')]
    assert "setattr(default_args, _k, _val)" in seg


def test_the_picker_is_on_the_run_page():
    for marker in ("tpl-select", "tplLoad()", "tplSave()", "tplDelete()", "tplRefresh"):
        assert marker in SRC, marker


def test_loading_a_template_asks_first():
    """It replaces every option on the page, which is not an undoable click."""
    seg = SRC[SRC.index("async function tplLoad()"):SRC.index("async function tplDelete()")]
    assert "uiConfirm" in seg


# ---------------------------------------------------------------- starter templates
def test_two_starters_ship_with_the_tool():
    """The picker should never open empty, and the two describe the real choice: a model
    with an 81-frame ceiling and no native continuation, versus one pretrained on it."""
    assert set(TW.BUILTIN_TEMPLATES) == {"Wan - many short scenes",
                                         "LongCat - few long scenes"}


def test_the_starter_names_survive_sanitising():
    """A built-in whose name changes when written would seed a duplicate every run."""
    for name in TW.BUILTIN_TEMPLATES:
        assert TW._safe_template_name(name) == name


def test_seeding_writes_them_once_and_never_overwrites(tmp_path):
    first = TW.seed_builtin_templates(str(tmp_path))
    assert set(first) == set(TW.BUILTIN_TEMPLATES)
    # An operator's edit must survive the next start.
    TW.save_template_dict(str(tmp_path), "LongCat - few long scenes", {"fps": 99})
    assert TW.seed_builtin_templates(str(tmp_path)) == []
    assert TW.load_template(str(tmp_path), "LongCat - few long scenes")["fps"] == 99


def test_seeding_runs_before_the_management_commands():
    """--list-templates on a fresh install has to show the starters, so seeding cannot
    sit after the commands that return early."""
    seed_at = SRC.index("seed_builtin_templates(args.out_dir)")
    list_at = SRC.index("if args.list_templates:")
    assert seed_at < list_at


def test_the_longcat_starter_asks_for_fewer_longer_scenes():
    """The scene lengths now come from the MODEL (0 = auto in both starters), so this
    asks the same question of the geometry the planner will use."""
    wan = TW.BUILTIN_TEMPLATES["Wan - many short scenes"]
    lc = TW.BUILTIN_TEMPLATES["LongCat - few long scenes"]
    # …at LongCat's own frame rate, not the Wan-era compromise.
    assert lc["fps"] == 15 and wan["fps"] == 8
    w_lo, w_hi = TW._clip_frame_range(wan["clip_min_frames"], wan["clip_max_frames"],
                                      TW.video_geometry("wan2.2"), "clip", wan["fps"])
    l_lo, l_hi = TW._clip_frame_range(lc["clip_min_frames"], lc["clip_max_frames"],
                                      TW.video_geometry("longcat"), "clip", lc["fps"])
    # Longer scenes…
    assert l_lo > w_hi
    # …so the same 45s short needs far fewer of them.
    assert 45 / (l_hi / lc["fps"]) < 45 / (w_hi / wan["fps"]) / 2


def test_the_longcat_frame_counts_obey_the_vae_rule():
    """(frames - 1) divisible by 4 — the same rule the server checks before a long run,
    and the one LongCat's own 93-frame segment satisfies. The starter no longer carries
    the numbers, so the rule is asked of what auto derives."""
    lc = TW.BUILTIN_TEMPLATES["LongCat - few long scenes"]
    geom = TW.video_geometry("longcat")
    for kind in ("clip", "intro", "outcome"):
        for n in TW._clip_frame_range(0, 0, geom, kind, lc["fps"]):
            assert (n - 1) % 4 == 0, (kind, n)
            # And on the segment grid, which is stricter and is what keeps
            # block-sparse attention usable.
            assert (n - 93) % 80 == 0, (kind, n)
    assert (TW._single_render_cap(lc["single_clip_max_frames"], geom) - 1) % 4 == 0


def test_the_longcat_starter_selects_the_model():
    assert TW.BUILTIN_TEMPLATES["LongCat - few long scenes"]["video_model"] == "longcat"


def test_a_starter_is_a_partial_not_a_full_capture():
    """Starters set the handful of options that differ; everything else stays at the
    tool's defaults rather than freezing today's values into a shipped file."""
    for name, cfg in TW.BUILTIN_TEMPLATES.items():
        assert 0 < len(cfg) < len(TW.CONFIG_FIELDS) / 2, name
        assert set(cfg) <= set(TW.CONFIG_FIELDS), name


def test_both_starters_cover_the_same_options():
    """A starter that set a key its twin did not would leave that setting at whatever
    the previous template put there — switching templates has to be deterministic."""
    wan = set(TW.BUILTIN_TEMPLATES["Wan - many short scenes"])
    lc = set(TW.BUILTIN_TEMPLATES["LongCat - few long scenes"])
    assert wan == lc, wan ^ lc


def test_the_starters_cover_the_outcome_clips_too():
    """Outcome stings go through the same video model as the scenes, so a template
    that only sized the scenes would leave them at the other model's frame counts."""
    for cfg in TW.BUILTIN_TEMPLATES.values():
        assert "outcome_min_frames" in cfg and "outcome_max_frames" in cfg
        assert cfg["outcome_min_frames"] <= cfg["outcome_max_frames"]


def test_the_longcat_outcomes_obey_the_vae_rule_as_well():
    lc = TW.BUILTIN_TEMPLATES["LongCat - few long scenes"]
    lo, hi = TW._clip_frame_range(lc["outcome_min_frames"], lc["outcome_max_frames"],
                                  TW.video_geometry("longcat"), "outcome", lc["fps"])
    for n in (lo, hi):
        assert (n - 1) % 4 == 0, n
    # An outcome is split across two shots, so its total is at least two segments —
    # 93 is one whole LongCat segment and the shortest honest ask for either half.
    assert lo >= 93 * 2 - 13


def test_the_starters_leave_the_image_side_stages_alone():
    """Characters and environments are drawn by the IMAGE model; the video model never
    sees the pool. A starter that pinned those would reset an operator's pool for no
    reason. keyframe_size is the one exception — a keyframe is the frame the video
    model is conditioned on, so it has to match the clip."""
    image_side = {"num_fighters", "num_environments", "char_refs", "env_refs",
                  "keyframe_steps", "image_model", "text_model", "lora_steps",
                  "lora_rank", "lora_weight", "character_strength"}
    for name, cfg in TW.BUILTIN_TEMPLATES.items():
        assert not (set(cfg) & image_side), name
        assert cfg["keyframe_size"] == cfg["video_size"], name


def test_loading_a_template_merges_into_the_config_instead_of_replacing_it(tmp_path):
    """A template is a PARTIAL. Writing it over the config destroyed every setting
    it does not mention.

    This is not hypothetical: loading the LongCat starter turned a 60-key
    township_config.json into its own 10 keys, taking api_key and base_url with it
    — after which every call to CoderAI came back 401 with nothing to explain why.
    """
    cfg = tmp_path / "township_config.json"
    cfg.write_text(json.dumps({
        "api_key": "the-fighters-token", "base_url": "http://127.0.0.1:8776",
        "image_model": "some/image-model", "matches": 6, "fps": 8,
        "clip_min_frames": 50,
    }))
    template = {"video_model": "longcat", "fps": 15, "clip_min_frames": 173}

    existing = json.loads(cfg.read_text())
    merged = dict(existing)
    merged.update(template)
    cfg.write_text(json.dumps(merged))

    got = json.loads(cfg.read_text())
    # the template's values won…
    assert got["video_model"] == "longcat"
    assert got["fps"] == 15 and got["clip_min_frames"] == 173
    # …and nothing it was silent about was lost.
    assert got["api_key"] == "the-fighters-token"
    assert got["base_url"] == "http://127.0.0.1:8776"
    assert got["image_model"] == "some/image-model"
    assert got["matches"] == 6


def test_the_load_handler_merges_rather_than_dumping_the_template():
    """Guards the actual code path: the bug was a bare json.dump(tcfg, ...)."""
    src = (Path(__file__).resolve().parents[1] / "tools"
           / "gen_township_fighters.py").read_text(encoding="utf-8")
    assert "_j.dump(tcfg, f" not in src, "the template is being written over the config"
    assert "_merged.update(tcfg)" in src
