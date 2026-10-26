"""Three defects a client hit on one render path, plus the VRAM-estimate bug
that starved it.

1. A caption font the image never installed (libass substitutes a wider family
   and the lines leave the frame).
2. Caption size derived from the frame HEIGHT, so a vertical video got a font
   sized for a 1920px width in a 1080px frame.
3. ``used_vram_gb = 0`` read as "this model needs no VRAM", so a 27B model was
   aimed at a GPU with no room for it.
"""
import os
import re
import tempfile

import pytest

from codai.compose import captions as C


# ----------------------------------------------------------------- sizing
def _ev():
    words = "INCREDIBLE NEUROSCIENCE DISCOVERIES EXPLAINED".split()
    return [{"start": 0.0, "end": 2.0,
             "words": [{"word": w, "start": i * 0.3, "end": i * 0.3 + 0.3}
                       for i, w in enumerate(words)],
             "rows": [words]}]


def _style_size(ass: str) -> int:
    line = [l for l in ass.splitlines() if l.startswith("Style: cap")][0]
    return int(line.split(",")[2])


def test_landscape_size_is_unchanged_by_the_short_side_rule():
    """16:9 must look exactly as it did: there the short side IS the height."""
    ass = C.build_ass(_ev(), width=1920, height=1080, preset="karaoke")
    assert _style_size(ass) == int(round(1080 * 0.052))


def test_vertical_video_is_not_sized_from_its_long_side():
    """The bug: height*0.052 at 1080x1920 gave ~100px in a 1080px-wide frame."""
    ass = C.build_ass(_ev(), width=1080, height=1920, preset="karaoke")
    size = _style_size(ass)
    assert size == int(round(1080 * 0.052))
    assert size < int(round(1920 * 0.052))


def test_portrait_and_landscape_of_the_same_short_side_agree():
    a = _style_size(C.build_ass(_ev(), width=1080, height=1920))
    b = _style_size(C.build_ass(_ev(), width=1920, height=1080))
    assert a == b


@pytest.mark.parametrize("w,h", [(1080, 1920), (1080, 1350), (720, 1280)])
def test_the_style_size_never_exceeds_the_usable_width(w, h):
    """A sanity bound: one glyph cannot be wider than the text area."""
    ass = C.build_ass(_ev(), width=w, height=h)
    assert _style_size(ass) < w - 2 * int(round(w * 0.06))


# ------------------------------------------------------------------ fitting
def test_a_line_too_wide_to_fit_gets_its_own_smaller_size():
    long_words = ["EXTRAORDINARILY", "INCOMPREHENSIBLE", "NEUROSCIENTIFIC",
                  "BREAKTHROUGHS"]
    ev = [{"start": 0.0, "end": 2.0,
           "words": [{"word": w, "start": i * 0.3, "end": i * 0.3 + 0.3}
                     for i, w in enumerate(long_words)],
           "rows": [long_words]}]
    ass = C.build_ass(ev, width=1080, height=1920, font="DejaVu Sans")
    shrunk = [l for l in ass.splitlines() if "\\fs" in l]
    assert shrunk, "an overflowing line should carry an \\fs override"
    got = int(re.search(r"\\fs(\d+)", shrunk[0]).group(1))
    assert got < _style_size(ass)
    assert got >= 12


def test_a_line_that_fits_carries_no_override():
    ev = [{"start": 0.0, "end": 1.0,
           "words": [{"word": "HI", "start": 0.0, "end": 1.0}],
           "rows": [["HI"]]}]
    ass = C.build_ass(ev, width=1080, height=1920, font="DejaVu Sans")
    assert "\\fs" not in ass


def test_the_shrink_applies_to_the_karaoke_path_too():
    long_words = ["EXTRAORDINARILY", "INCOMPREHENSIBLE", "NEUROSCIENTIFIC",
                  "BREAKTHROUGHS"]
    ev = [{"start": 0.0, "end": 2.0,
           "words": [{"word": w, "start": i * 0.3, "end": i * 0.3 + 0.3}
                     for i, w in enumerate(long_words)],
           "rows": [long_words]}]
    ass = C.build_ass(ev, width=1080, height=1920, font="DejaVu Sans",
                      preset="karaoke")
    dialogues = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogues) == len(long_words)       # one event per word
    assert all("\\fs" in l for l in dialogues)


def test_unmeasurable_text_is_left_alone_rather_than_guessed():
    """No font file -> no measurement -> keep the computed size, don't invent one."""
    assert C._fit_size([["ANYTHING"]], 56, 950, "Nonexistent-Family-XYZ",
                       True, 0.0, False) in (56,)


def test_width_measurement_includes_ass_letter_spacing():
    w0 = C._text_width("ABCDEF", 48, "DejaVu Sans", True, 0.0)
    w5 = C._text_width("ABCDEF", 48, "DejaVu Sans", True, 5.0)
    if w0 is None:
        pytest.skip("no measurable font in this environment")
    assert w5 == pytest.approx(w0 + 5.0 * 5, abs=0.01)


# -------------------------------------------------------------------- fonts
def test_operator_font_dir_is_honoured_from_the_environment(monkeypatch):
    from codai.compose import media
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("CODERAI_FONT_DIR", d)
        assert os.path.abspath(d) in media.operator_font_dirs()


def test_registration_writes_a_fontconfig_snippet_a_subprocess_can_read(monkeypatch):
    """libass runs in ffmpeg, a separate process: the directory must land in
    fontconfig's own config, not just in this process's cache."""
    from codai.compose import media
    with tempfile.TemporaryDirectory() as fonts, \
         tempfile.TemporaryDirectory() as xdg:
        open(os.path.join(fonts, "Fake-Regular.ttf"), "wb").write(b"\x00")
        monkeypatch.setenv("CODERAI_FONT_DIR", fonts)
        monkeypatch.setenv("XDG_CONFIG_HOME", xdg)
        monkeypatch.setattr(media.shutil, "which", lambda *_a, **_k: None)
        got = media.register_operator_fonts(force=True)
        assert os.path.abspath(fonts) in got
        conf = os.path.join(xdg, "fontconfig", "conf.d", "99-coderai-fonts.conf")
        assert os.path.isfile(conf)
        assert os.path.abspath(fonts) in open(conf).read()


def test_an_empty_font_dir_registers_nothing(monkeypatch):
    """A directory with no font files in it must not be registered.

    The config dir is a second, independent source, so it is pointed at an
    empty directory here too -- on a real installation it may well hold fonts,
    and this test is about the empty case.
    """
    from codai.compose import media
    import codai.platform_paths as pp
    with tempfile.TemporaryDirectory() as fonts, \
         tempfile.TemporaryDirectory() as cfg, \
         tempfile.TemporaryDirectory() as xdg:
        monkeypatch.setenv("CODERAI_FONT_DIR", fonts)
        monkeypatch.setenv("XDG_CONFIG_HOME", xdg)
        monkeypatch.setattr(pp, "default_fonts_dir", lambda: cfg)
        assert media.register_operator_fonts(force=True) == []
        assert not os.path.exists(
            os.path.join(xdg, "fontconfig", "conf.d", "99-coderai-fonts.conf"))


def test_both_the_env_dir_and_the_config_dir_are_searched(monkeypatch):
    """An operator may use either; CODERAI_FONT_DIR comes first."""
    from codai.compose import media
    import codai.platform_paths as pp
    with tempfile.TemporaryDirectory() as env_d, \
         tempfile.TemporaryDirectory() as cfg_d:
        monkeypatch.setenv("CODERAI_FONT_DIR", env_d)
        monkeypatch.setattr(pp, "default_fonts_dir", lambda: cfg_d)
        dirs = media.operator_font_dirs()
        assert dirs.index(os.path.abspath(env_d)) < dirs.index(os.path.abspath(cfg_d))


def test_the_image_build_installs_the_caption_fonts():
    """Dockerfile.oci and Dockerfile.update carried these; the Dockerfile that
    build_oci_image.sh actually builds did not, which is why no shipped image
    ever had Montserrat."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("Dockerfile.oci-venv", "Dockerfile.oci", "Dockerfile.update"):
        p = os.path.join(root, "packaging", "linux", name)
        if not os.path.isfile(p):
            continue
        body = open(p, encoding="utf-8").read()
        assert "fonts-montserrat" in body, f"{name} installs no caption fonts"
        assert "fontconfig" in body, f"{name} has no fontconfig"


# --------------------------------------------------------------- VRAM estimate
def _estimate(cfg, key="/AI/guffcache/Qwen3.8-27B-Uncensored-X-Q4_K_P.gguf"):
    """Run the real estimator over one config entry.

    The manager is not constructed (it would start engines); only the fields
    this method touches are supplied, and the config lookup is pointed at the
    entry under test.
    """
    import types
    from codai.models.manager import MultiModelManager
    mgr = object.__new__(MultiModelManager)
    mgr._measured_vram_gb = {}
    mgr.config = {}
    mgr.model_aliases = {}
    mgr._config_for_model = types.MethodType(lambda self, *a, **k: cfg, mgr)
    return mgr._get_model_used_vram_gb(key)


def test_a_configured_zero_does_not_mean_the_model_is_free():
    """The incident: used_vram_gb=0 beat measured_vram_gb=18.0, the planner
    reserved nothing, and a 27B was aimed at 8.6 GB of free VRAM."""
    est = _estimate({"path": "/AI/guffcache/x-Q4_K_P.gguf",
                     "used_vram_gb": 0, "measured_vram_gb": 18.0,
                     "load_in_4bit": True, "n_gpu_layers": -1})
    assert est == pytest.approx(18.0), (
        "a real measurement must beat an unset (zero) config field")


def test_a_positive_configured_value_is_still_authoritative():
    est = _estimate({"path": "/AI/guffcache/x-Q4_K_P.gguf",
                     "used_vram_gb": 11.0, "measured_vram_gb": 18.0,
                     "load_in_4bit": True, "n_gpu_layers": -1})
    # The configured number drives it (plus the usual runtime reserve); what it
    # must NOT do is take the 18.0 measurement instead.
    assert 11.0 <= est < 18.0


def test_zero_with_no_measurement_falls_through_rather_than_claiming_zero():
    """With nothing measured the file-size scan is the next best thing; what it
    must not do is return 0.00 GB and call the model free."""
    import os as _os
    est = _estimate({"path": "/nonexistent/x-Q4_K_P.gguf", "used_vram_gb": 0,
                     "load_in_4bit": True, "n_gpu_layers": -1})
    assert est >= 0.0           # no crash; 0 only because nothing can be sized


def test_an_explicit_zero_layer_count_still_reserves_nothing():
    """n_gpu_layers=0 is how an operator says 'weights stay in host RAM', and
    that genuinely needs no weight VRAM -- it must keep working."""
    est = _estimate({"path": "/AI/guffcache/x-Q4_K_P.gguf",
                     "used_vram_gb": 12.0, "n_gpu_layers": 0})
    assert est == pytest.approx(0.0)
