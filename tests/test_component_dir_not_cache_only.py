"""The component probe must not answer from the cache alone.

When a component's weights come from a GGUF, the base repo's copy of that
subfolder is never downloaded — there is no reason to fetch 20 GB of weights
the file replaces. A cache-only lookup therefore returns None, and every
caller that needs the component's `config.json` silently does nothing.

That is exactly how the LTX-2.5 key reconciliation came to be a no-op: the
repair was shipped, the retry failed identically, and the only difference
between "fixed" and "did nothing" was a few hundred bytes of config that
nobody had downloaded.
"""
import os

from codai.models import hf_loading


def test_a_local_directory_is_used_as_is(tmp_path):
    (tmp_path / "transformer").mkdir()
    got = hf_loading._component_dir({"path": str(tmp_path)}, "transformer")
    assert got == str(tmp_path / "transformer")


def test_a_component_missing_from_a_local_dir_is_none(tmp_path):
    assert hf_loading._component_dir({"path": str(tmp_path)}, "transformer") is None


def test_the_cache_is_tried_first_then_the_hub(monkeypatch, tmp_path):
    """Cache hit: no network call at all."""
    root = tmp_path / "snap"
    (root / "transformer").mkdir(parents=True)
    monkeypatch.setattr(hf_loading, "_norm", hf_loading._norm)  # keep module import
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo, **kw: str(root))

    def _boom(*a, **kw):
        raise AssertionError("the hub must not be touched on a cache hit")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)
    got = hf_loading._component_dir({"path": "some/repo"}, "transformer")
    assert got == str(root / "transformer")


def test_a_cache_miss_fetches_only_that_components_config(monkeypatch, tmp_path):
    """The fallback pulls one small file, never the component's weights."""
    import huggingface_hub

    asked = []

    def _no_snapshot(repo, **kw):
        raise OSError("not in the cache")

    def _one_file(repo, filename, **kw):
        asked.append(filename)
        p = tmp_path / "transformer" / "config.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
        return str(p)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _no_snapshot)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _one_file)
    got = hf_loading._component_dir({"path": "some/repo"}, "transformer")
    assert got == str(tmp_path / "transformer")
    assert asked == ["transformer/config.json"]
