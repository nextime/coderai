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
