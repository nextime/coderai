"""The LongCat source is vendored, not cloned.

`longcat_video` (the pipeline, DiT, VAE, scheduler and INT8 quantisation helpers)
lives inside the upstream repo and is not on PyPI. Resolving it by cloning from
GitHub at install time would mean an image someone else pulls cannot run LongCat
without network access to a third party — so the 26 MIT-licensed source files are
vendored under third_party/ and pinned to a commit.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party"


def test_the_package_is_present_in_the_repo():
    """Upstream ships no top-level __init__.py — it is a namespace package — so the
    modules that are actually imported are what to check for."""
    pkg = VENDOR / "longcat_video"
    assert pkg.is_dir()
    assert (pkg / "pipeline_longcat_video.py").is_file()
    assert (pkg / "pipeline_longcat_video_avatar.py").is_file()
    # the INT8 loader coderai calls for a quantised DiT
    assert (pkg / "modules" / "quantization.py").is_file()


def test_the_licence_travels_with_it():
    lic = VENDOR / "LONGCAT-VIDEO-LICENSE"
    assert lic.is_file(), "vendored MIT code must carry its licence"
    text = lic.read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Meituan" in text


def test_the_provenance_pins_a_commit():
    """'whatever main happened to be' is not a version."""
    readme = (VENDOR / "README.md").read_text(encoding="utf-8")
    assert "github.com/meituan-longcat/LongCat-Video" in readme
    assert re.search(r"\b[0-9a-f]{40}\b", readme), "no commit hash recorded"


def test_nothing_compiled_or_heavy_was_vendored():
    """Source only: no weights, no wheels, no .so — it must stay reviewable and
    small enough to live in the repo."""
    files = [p for p in (VENDOR / "longcat_video").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    assert files, "nothing vendored"
    non_py = [p for p in files if p.suffix != ".py"]
    assert not non_py, f"non-source files vendored: {non_py[:5]}"
    total = sum(p.stat().st_size for p in files)
    assert total < 5 * 1024 * 1024, f"{total/1e6:.1f} MB is too much to vendor"


def test_the_resolver_prefers_the_vendored_copy():
    lw = pytest.importorskip("codai.api.longcat_worker")
    got = lw.resolve_source_dir()
    assert (got / "longcat_video").is_dir(), f"{got} has no longcat_video"
    assert got == VENDOR, "the vendored copy should win over clone/baked fallbacks"


def test_an_explicit_setting_still_wins_over_the_vendored_copy(tmp_path):
    """Someone running their own checkout must still be able to point at it."""
    lw = pytest.importorskip("codai.api.longcat_worker")
    (tmp_path / "longcat_video").mkdir()
    assert lw.resolve_source_dir({"longcat_source": str(tmp_path)}) == tmp_path


def test_it_is_not_excluded_from_the_image():
    """COPY . /opt/coderai/app ships it — unless .dockerignore drops it."""
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert not any(l.strip().rstrip("/") == "third_party"
                   for l in ignore.splitlines()), "third_party is dockerignored"
