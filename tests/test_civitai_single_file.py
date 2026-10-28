"""C4: fetching a Civitai checkpoint.

A large part of the community's fine-tunes are published only on Civitai, as
one ComfyUI-style checkpoint file -- exactly the shape
build_single_file_pipeline already handles. The only thing missing was a way
to fetch them, so an operator had to download by hand and hard-code a path.
"""
import os

import pytest

from codai.models.hf_loading import (civitai_token, is_single_file_ref,
                                     resolve_civitai, resolve_single_file)


@pytest.mark.parametrize("ref,expect", [
    ("civitai:1609320", True),
    ("CIVITAI:153568", True),
    ("/AI/models/someMix_v18.safetensors", True),
    ("org/repo:someMix_v18.safetensors", True),
    ("org/repo", False),
    ("", False),
    (None, False),
])
def test_a_civitai_reference_is_recognised_as_a_single_file(ref, expect):
    # It names a version id, not a filename, so there is no suffix to test --
    # but it is unambiguously one checkpoint.
    assert is_single_file_ref(ref) is expect


def test_the_version_id_is_required_and_must_be_a_number():
    # A Civitai "model" is a family with many versions. A config naming the
    # family would silently change underneath the operator when a new version
    # was published, so the version is what gets named.
    for bad in ("civitai:", "civitai:abc", "civitai:1.2", "civitai:v3"):
        with pytest.raises(ValueError) as e:
            resolve_civitai(bad)
        assert "modelVersionId" in str(e.value)


def test_a_cached_checkpoint_is_not_downloaded_again(monkeypatch, tmp_path):
    import codai.models.hf_loading as H
    d = tmp_path / "civitai" / "999"
    d.mkdir(parents=True)
    f = d / "RealDream.safetensors"
    f.write_bytes(b"x" * 2_000_000)
    monkeypatch.setattr(H, "legacy_style_cache_root", lambda: tmp_path,
                        raising=False)
    import codai.platform_paths as PP
    monkeypatch.setattr(PP, "legacy_style_cache_root", lambda: tmp_path)
    # Nothing may reach the network: a cached file is returned before any
    # request is made, so a urlopen call here is a bug.
    import urllib.request
    monkeypatch.setattr(H, "_civitai_open",
                        lambda *a, **k: pytest.fail("re-downloaded a cached file"))
    assert resolve_civitai("civitai:999") == str(f)


def test_the_largest_cached_file_wins(monkeypatch, tmp_path):
    import codai.platform_paths as PP
    monkeypatch.setattr(PP, "legacy_style_cache_root", lambda: tmp_path)
    d = tmp_path / "civitai" / "888"
    d.mkdir(parents=True)
    (d / "small.safetensors").write_bytes(b"x" * 1_500_000)
    big = d / "big.safetensors"
    big.write_bytes(b"x" * 3_000_000)
    assert resolve_civitai("civitai:888") == str(big)


def test_an_html_error_page_is_not_saved_as_a_checkpoint(monkeypatch, tmp_path):
    import codai.models.hf_loading as H
    """The failure this guards against.

    Civitai answers an unauthenticated download with a web page, HTTP 200. A
    naive fetch writes it to a .safetensors name and the problem surfaces
    hours later as an unreadable tensor header.
    """
    import io
    import urllib.request

    import codai.platform_paths as PP
    monkeypatch.setattr(PP, "legacy_style_cache_root", lambda: tmp_path)

    class _Resp(io.BytesIO):
        headers = {"Content-Type": "text/html; charset=utf-8"}
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(H, "_civitai_open",
                        lambda *a, **k: _Resp(b"<html>login</html>"))
    with pytest.raises(RuntimeError) as e:
        resolve_civitai("civitai:777")
    msg = str(e.value)
    assert "web page" in msg
    assert "authentication" in msg
    # And nothing was left behind pretending to be a checkpoint.
    d = tmp_path / "civitai" / "777"
    assert not any(p.suffix == ".safetensors" for p in d.iterdir())


def test_a_truncated_download_is_rejected(monkeypatch, tmp_path):
    import codai.models.hf_loading as H
    import io
    import urllib.request

    import codai.platform_paths as PP
    monkeypatch.setattr(PP, "legacy_style_cache_root", lambda: tmp_path)

    class _Resp(io.BytesIO):
        headers = {"Content-Type": "application/octet-stream"}
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(H, "_civitai_open",
                        lambda *a, **k: _Resp(b"tiny"))
    with pytest.raises(RuntimeError) as e:
        resolve_civitai("civitai:666")
    assert "not a checkpoint" in str(e.value)


def test_the_missing_token_is_named_in_the_error(monkeypatch, tmp_path):
    import io
    import urllib.error
    import urllib.request

    import codai.models.hf_loading as H
    import codai.platform_paths as PP
    monkeypatch.setattr(PP, "legacy_style_cache_root", lambda: tmp_path)
    monkeypatch.setattr(H, "civitai_token", lambda: None)

    def _boom(*a, **k):
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)

    monkeypatch.setattr(H, "_civitai_open", _boom)
    with pytest.raises(RuntimeError) as e:
        resolve_civitai("civitai:555")
    msg = str(e.value)
    assert "401" in msg
    assert "CIVITAI_TOKEN" in msg


def test_the_token_comes_from_the_environment_or_the_config(monkeypatch, tmp_path):
    import codai.models.hf_loading as H
    import codai.platform_paths as PP
    monkeypatch.setenv("CIVITAI_TOKEN", "from-env")
    assert H.civitai_token() == "from-env"
    monkeypatch.delenv("CIVITAI_TOKEN", raising=False)
    monkeypatch.setattr(PP, "legacy_style_config_dir", lambda: tmp_path)
    assert H.civitai_token() is None
    (tmp_path / "civitai_token").write_text("from-file\n")
    assert H.civitai_token() == "from-file"


def test_resolve_single_file_routes_civitai_without_touching_hf(monkeypatch, tmp_path):
    import codai.models.hf_loading as H
    called = {}

    def _fake(ref):
        called["ref"] = ref
        return "/tmp/x.safetensors"

    monkeypatch.setattr(H, "resolve_civitai", _fake)
    assert H.resolve_single_file("civitai:4242") == "/tmp/x.safetensors"
    assert called["ref"] == "civitai:4242"


# ===================================== choosing WHICH file of a version to get
# A Civitai version is not one checkpoint. Real Dream's Krea 2 V3 publishes the
# same weights three times -- fp8 13.46 GB, int8 13.18 GB, nvfp4 8.12 GB -- and
# they are not interchangeable: nvfp4 needs Blackwell, so on an Ampere card it
# is 8 GB downloaded to produce a load error. Taking the largest, as this used
# to, lands on fp8 by a 0.28 GB margin nobody chose.

REAL_DREAM = [
    {"id": 3286566, "name": "fp8.safetensors", "sizeKB": 13.46e6,
     "metadata": {"fp": "fp8"}},
    {"id": 3289346, "name": "nvfp4.safetensors", "sizeKB": 8.12e6,
     "metadata": {"fp": "nvfp4"}},
    {"id": 3304483, "name": "int8.safetensors", "sizeKB": 13.18e6,
     "metadata": {"fp": "int8"}},
]


@pytest.fixture
def ampere(monkeypatch):
    """A card that is not Blackwell — which is every card this runs on today."""
    import codai.models.hf_loading as H
    monkeypatch.setattr(H, "_fp_is_runnable_here",
                        lambda fp: (fp or "").lower() not in ("nvfp4", "fp4"))


def test_an_explicit_file_id_is_obeyed(ampere):
    from codai.models.hf_loading import _pick_civitai_file
    got = _pick_civitai_file(REAL_DREAM, "3396368", "3286566")
    assert got["name"] == "fp8.safetensors"


def test_without_a_file_id_precision_decides_rather_than_size(ampere):
    from codai.models.hf_loading import _pick_civitai_file
    got = _pick_civitai_file(REAL_DREAM, "3396368", None)
    assert got["name"] == "int8.safetensors"


def test_a_build_this_gpu_cannot_run_is_never_chosen(ampere):
    from codai.models.hf_loading import _pick_civitai_file
    only_nvfp4 = [f for f in REAL_DREAM if f["metadata"]["fp"] == "nvfp4"]
    with pytest.raises(RuntimeError) as e:
        _pick_civitai_file(only_nvfp4, "3396368", None)
    assert "nvfp4" in str(e.value)
    # ...but naming it explicitly still works, for the operator who knows better.
    assert _pick_civitai_file(only_nvfp4, "3396368", "3289346")["name"] \
        == "nvfp4.safetensors"


def test_a_file_id_that_is_not_there_lists_what_is(ampere):
    from codai.models.hf_loading import _pick_civitai_file
    with pytest.raises(RuntimeError) as e:
        _pick_civitai_file(REAL_DREAM, "3396368", "123")
    msg = str(e.value)
    assert "3304483" in msg and "int8" in msg


def test_an_unlabelled_file_is_not_rejected(ampere):
    from codai.models.hf_loading import _pick_civitai_file
    plain = [{"id": 1, "name": "x.safetensors", "sizeKB": 1e6, "metadata": {}}]
    assert _pick_civitai_file(plain, "1", None)["name"] == "x.safetensors"


@pytest.mark.parametrize("ref,vid,fid", [
    ("civitai:3396368", "3396368", None),
    ("civitai:3396368:3304483", "3396368", "3304483"),
])
def test_a_reference_may_name_the_file_as_well_as_the_version(ref, vid, fid):
    from codai.models.hf_loading import is_single_file_ref
    assert is_single_file_ref(ref) is True


def test_a_non_numeric_file_id_is_refused():
    with pytest.raises(ValueError) as e:
        resolve_civitai("civitai:3396368:abc")
    assert "fileId" in str(e.value)


def test_two_precisions_of_one_version_cache_side_by_side(monkeypatch, tmp_path):
    """Without a per-file directory the "already downloaded?" check answers with
    whichever build happens to be biggest — the bug one level down."""
    import codai.models.hf_loading as H
    monkeypatch.setattr(H, "civitai_cache_root", lambda: tmp_path)
    for fid, name, size in (("3304483", "int8.safetensors", 2_000_000),
                            ("3286566", "fp8.safetensors", 3_000_000)):
        d = tmp_path / "3396368" / fid
        d.mkdir(parents=True)
        (d / name).write_bytes(b"x" * size)
    import urllib.request
    monkeypatch.setattr(H, "_civitai_open",
                        lambda *a, **k: pytest.fail("re-downloaded a cached file"))
    assert resolve_civitai("civitai:3396368:3304483").endswith("int8.safetensors")
    assert resolve_civitai("civitai:3396368:3286566").endswith("fp8.safetensors")


def test_the_cache_root_can_be_pointed_at_the_weights_disk(monkeypatch):
    from codai.models.hf_loading import civitai_cache_root
    monkeypatch.setenv("CODERAI_CIVITAI_CACHE", "/AI/civitaicache")
    assert str(civitai_cache_root()) == "/AI/civitaicache"


def test_the_cache_root_is_not_derived_from_the_hf_cache(monkeypatch):
    """Putting multi-GB checkpoints under the HF cache root is how a second GGUF
    directory grew inside one."""
    from codai.models.hf_loading import civitai_cache_root
    monkeypatch.delenv("CODERAI_CIVITAI_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", "/AI/hfcache/huggingface")
    assert "hfcache" not in str(civitai_cache_root())
