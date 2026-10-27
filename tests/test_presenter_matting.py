"""`presenter.remove_background`: the person, not the room they were filmed in.

The mask is the whole feature — everything downstream is ffmpeg — so these
tests pin the three things that break it silently: a mask video that does not
line up with the clip it came from, a filter graph that ignores it, and a
matting failure that takes the presenter down with it.
"""
import os
import subprocess

import pytest

from codai.api import compose as compose_mod
from codai.compose import matting
from codai.compose import render as rnd

ffmpeg = pytest.importorskip("shutil").which("ffmpeg")
pytestmark = pytest.mark.skipif(not ffmpeg, reason="ffmpeg is not installed")


def _clip(path, *, w=160, h=160, seconds=1, fps=12, colour="red"):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c={colour}:s={w}x{h}:r={fps}:d={seconds}",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def _size(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height,nb_read_frames",
                          "-count_frames", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    w, h, n = out.split(",")[:3]
    return int(w), int(h), int(n)


# --------------------------------------------------------------------------- #
# which engine
# --------------------------------------------------------------------------- #

def test_a_configured_torchscript_model_wins_over_the_built_in_one(monkeypatch, tmp_path):
    """An install that has RVM exported should use it: it is recurrent, so it
    does not flicker the way a per-frame segmenter does."""
    f = tmp_path / "rvm_mobilenetv3.torchscript"
    f.write_bytes(b"not really a module, only its presence is read here")
    monkeypatch.setenv("CODERAI_MATTING_MODEL", str(f))
    assert matting.engine_name() == "torchscript:rvm_mobilenetv3.torchscript"


def test_a_configured_path_that_is_not_there_does_not_silently_fall_back(monkeypatch):
    """Naming a model and getting the built-in one anyway is how you ship a
    reel matted by the wrong thing and never find out."""
    monkeypatch.setenv("CODERAI_MATTING_MODEL", "/nonexistent/rvm.torchscript")
    with pytest.raises(matting.MattingError) as err:
        matting._load("cpu")
    assert "/nonexistent/rvm.torchscript" in str(err.value)


def test_torchvision_is_enough_on_its_own(monkeypatch):
    monkeypatch.delenv("CODERAI_MATTING_MODEL", raising=False)
    pytest.importorskip("torchvision")
    assert matting.engine_name() == "deeplabv3"
    assert matting.available()


# --------------------------------------------------------------------------- #
# the mask video
# --------------------------------------------------------------------------- #

def test_the_mask_matches_the_clip_frame_for_frame(monkeypatch, tmp_path):
    """One frame in, one frame out, at the clip's own size: the mask is used as
    an alpha stream against that clip, so a different length or size is a
    presenter that goes transparent part-way through.

    The segmenter is stubbed — what is under test is the frame plumbing, not
    torchvision's accuracy."""
    torch = pytest.importorskip("torch")
    src = _clip(tmp_path / "pres.mp4", w=160, h=96, seconds=1, fps=12)

    class _Half:
        """Left half of the frame is the 'person'."""
        def __call__(self, batch):
            a = torch.zeros((batch.shape[0], 1, batch.shape[2], batch.shape[3]))
            a[..., : batch.shape[3] // 2] = 1.0
            return a

    monkeypatch.setattr(matting, "_load", lambda device: _Half())
    out = matting.matte(src, str(tmp_path / "alpha.mp4"), fps=12)

    sw, sh, sn = _size(src)
    mw, mh, mn = _size(out)
    assert (mw, mh) == (sw, sh)
    assert abs(mn - sn) <= 1      # the smoothing filter may drop the first frame


def test_an_engine_that_cannot_load_raises_before_writing_anything(monkeypatch, tmp_path):
    """The caller turns a MattingError into a warning and composites the
    presenter as filmed; a half-written mask file would instead composite
    nothing at all."""
    pytest.importorskip("torch")
    src = _clip(tmp_path / "pres.mp4", seconds=1, fps=12)
    monkeypatch.setattr(matting, "_probe_size", lambda p: (160, 160))

    def _no_frames(device):
        raise matting.MattingError("no engine here")

    monkeypatch.setattr(matting, "_load", _no_frames)
    with pytest.raises(matting.MattingError):
        matting.matte(src, str(tmp_path / "alpha.mp4"), fps=12)


# --------------------------------------------------------------------------- #
# the composite
# --------------------------------------------------------------------------- #

def test_pip_with_an_alpha_video_renders_and_draws_no_border(tmp_path):
    """The bubble's shape mask is replaced by the person's; a border would
    outline a rectangle around someone's head."""
    base = _clip(tmp_path / "base.mp4", w=256, h=256, colour="blue")
    pres = _clip(tmp_path / "pres.mp4", w=128, h=128, colour="green")
    alpha = _clip(tmp_path / "alpha.mp4", w=128, h=128, colour="white")
    out = rnd.composite_pip(base, pres, str(tmp_path / "out.mp4"),
                            width=256, height=256, fps=12, size=0.4,
                            border={"width": 8, "color": "#FFFFFF"},
                            alpha_video=alpha, workdir=str(tmp_path))
    assert _size(out)[:2] == (256, 256)
    # The shape mask is what the bubble PNG is for: with an alpha video none is
    # written, so nothing can quietly fall back to a circle.
    assert not list(tmp_path.glob("bubble-*.mask.png"))


def test_pip_without_an_alpha_video_still_uses_the_shape(tmp_path):
    base = _clip(tmp_path / "base.mp4", w=256, h=256, colour="blue")
    pres = _clip(tmp_path / "pres.mp4", w=128, h=128, colour="green")
    out = rnd.composite_pip(base, pres, str(tmp_path / "out.mp4"),
                            width=256, height=256, fps=12, size=0.4,
                            shape="circle", workdir=str(tmp_path))
    assert _size(out)[:2] == (256, 256)
    assert list(tmp_path.glob("bubble-*.mask.png"))


def test_split_with_an_alpha_puts_the_presenter_in_front_of_the_b_roll(tmp_path):
    """Without matting the presenter's half is their own footage; with it, both
    halves are the scene and one has a person standing in it."""
    base = _clip(tmp_path / "base.mp4", w=256, h=256, colour="blue")
    pres = _clip(tmp_path / "pres.mp4", w=256, h=128, colour="green")
    alpha = _clip(tmp_path / "alpha.mp4", w=256, h=128, colour="white")
    out = rnd.composite_split(base, pres, str(tmp_path / "out.mp4"),
                              width=256, height=256, fps=12,
                              presenter_half="bottom", alpha_video=alpha)
    assert _size(out)[:2] == (256, 256)


# --------------------------------------------------------------------------- #
# failure is a warning, never a lost presenter
# --------------------------------------------------------------------------- #

def _matte(pres, monkeypatch, **kw):
    warnings = []
    compose_mod._matte_presenter(
        pres, index=0, work=kw.pop("work", "/tmp"), fps=24,
        warn=warnings.append, stage=lambda *a, **k: None, progress=30.0,
        scene_no="1/1", cancel=None, log=None, gpu_lease=None, **kw)
    return warnings


def test_a_full_screen_presenter_is_told_why_not(monkeypatch):
    pres = {"path": "/nonexistent.mp4", "layout": "full", "engine": "sadtalker"}
    warnings = _matte(pres, monkeypatch)
    assert "alpha" not in pres
    assert any("pip or split" in w for w in warnings)


def test_a_matting_failure_keeps_the_presenter(monkeypatch):
    """remove_background is optional; the presenter is not. A scene that cannot
    be cut out is still a scene with a presenter in it."""
    def _boom(*a, **kw):
        raise matting.MattingError("the segmenter fell over")

    monkeypatch.setattr(matting, "matte", _boom)
    monkeypatch.setattr(matting, "available", lambda: True)
    pres = {"path": "/tmp/pres.mp4", "layout": "pip", "engine": "wav2lip"}
    warnings = _matte(pres, monkeypatch)
    assert "alpha" not in pres and pres["engine"] == "wav2lip"
    assert any("composited as filmed" in w for w in warnings)


def test_a_cancelled_job_is_not_swallowed_as_a_warning(monkeypatch):
    def _cancelled(*a, **kw):
        raise rnd.Cancelled()

    monkeypatch.setattr(matting, "matte", _cancelled)
    monkeypatch.setattr(matting, "available", lambda: True)
    with pytest.raises(rnd.Cancelled):
        _matte({"path": "/tmp/p.mp4", "layout": "pip", "engine": "wav2lip"}, monkeypatch)


def test_a_successful_matte_names_its_engine_for_the_result(monkeypatch, tmp_path):
    monkeypatch.setattr(matting, "available", lambda: True)
    monkeypatch.setattr(matting, "engine_name", lambda: "deeplabv3")
    monkeypatch.setattr(matting, "matte",
                        lambda src, out, **kw: out)
    pres = {"path": "/tmp/pres.mp4", "layout": "pip", "engine": "wav2lip"}
    warnings = _matte(pres, monkeypatch, work=str(tmp_path))
    assert pres["matted"] == "deeplabv3" and pres["alpha"].endswith("pres0-alpha.mp4")
    assert warnings == []


def test_the_engines_endpoint_says_whether_a_client_may_offer_the_toggle():
    """A client that cannot ask has to discover it from a warning after the
    render."""
    import asyncio
    info = asyncio.run(compose_mod.presenter_engines())
    assert "matting_engine" in info and "remove_background" in info
    assert info["remove_background"] == bool(info["matting_engine"])
