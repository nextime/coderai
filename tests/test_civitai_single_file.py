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
    monkeypatch.setattr(urllib.request, "urlopen",
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

    monkeypatch.setattr(urllib.request, "urlopen",
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
    import io
    import urllib.request

    import codai.platform_paths as PP
    monkeypatch.setattr(PP, "legacy_style_cache_root", lambda: tmp_path)

    class _Resp(io.BytesIO):
        headers = {"Content-Type": "application/octet-stream"}
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(urllib.request, "urlopen",
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

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
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
