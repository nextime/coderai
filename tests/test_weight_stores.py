"""Finding model weights that live outside the HF and GGUF caches.

The case this exists for: colibri's GLM-5.2 container sat in
/AI/offloads/colibri/models--mastouri--… at 400 GB while the HF cache held a 4 KB
shell of the same repo. "Free disk" deleted the shell and reported success.
"""
import os
import types

import pytest

from codai.models import weight_stores as WS


def _cfg(directory):
    return types.SimpleNamespace(offload=types.SimpleNamespace(directory=str(directory)))


def _repo(root, model_id, blob_bytes=4096):
    """An HF-cache-layout repo dir: a blob plus a snapshot symlink to it."""
    d = root / WS.hf_dir_name(model_id)
    (d / "blobs").mkdir(parents=True)
    (d / "snapshots" / "abc").mkdir(parents=True)
    blob = d / "blobs" / "deadbeef"
    blob.write_bytes(b"\x00" * blob_bytes)
    os.symlink(blob, d / "snapshots" / "abc" / "model.safetensors")
    return d


def test_the_hf_directory_name_matches_huggingface_hub():
    assert WS.hf_dir_name("mastouri/GLM-5.2-colibri") == "models--mastouri--GLM-5.2-colibri"


def test_a_sibling_of_the_offload_root_is_searched(tmp_path):
    """The exact shape that was missed: coderai offloads to .../offloads/nvidia and
    the engine put its weights in .../offloads/colibri."""
    (tmp_path / "offloads" / "nvidia").mkdir(parents=True)
    colibri = tmp_path / "offloads" / "colibri"
    colibri.mkdir()
    _repo(colibri, "mastouri/GLM-5.2", blob_bytes=5000)
    found = WS.find_external_weights("mastouri/GLM-5.2",
                                     _cfg(tmp_path / "offloads" / "nvidia"))
    assert len(found) == 1
    assert found[0][1] == 5000


def test_a_symlinked_snapshot_is_not_counted_twice(tmp_path):
    """An HF snapshot is a tree of links into blobs/; following them would report
    the weights twice and make 'freed' a lie in the generous direction."""
    (tmp_path / "offloads" / "nvidia").mkdir(parents=True)
    store = tmp_path / "offloads" / "engine"
    store.mkdir()
    d = _repo(store, "org/model", blob_bytes=10000)
    assert WS.dir_size(str(d)) == 10000


def test_only_an_exact_name_match_is_ever_returned(tmp_path):
    """The sibling scan widens where we look, never what we delete."""
    (tmp_path / "offloads" / "nvidia").mkdir(parents=True)
    store = tmp_path / "offloads" / "engine"
    store.mkdir()
    _repo(store, "org/model")
    _repo(store, "org/model-v2")
    (store / "some-unrelated-dir").mkdir()
    found = WS.find_external_weights("org/model", _cfg(tmp_path / "offloads" / "nvidia"))
    assert [os.path.basename(p) for p, _ in found] == ["models--org--model"]


def test_a_repo_one_level_down_is_found(tmp_path):
    """huggingface_hub puts repos under <root>/hub, so a root given without it
    still has to resolve."""
    (tmp_path / "offloads" / "nvidia").mkdir(parents=True)
    hub = tmp_path / "offloads" / "engine" / "hub"
    hub.mkdir(parents=True)
    _repo(hub, "org/model")
    found = WS.find_external_weights("org/model", _cfg(tmp_path / "offloads" / "nvidia"))
    assert len(found) == 1


def test_the_same_directory_is_not_reported_twice(tmp_path):
    """Registered roots and the sibling scan can name the same place."""
    (tmp_path / "offloads" / "nvidia").mkdir(parents=True)
    store = tmp_path / "offloads" / "engine"
    store.mkdir()
    _repo(store, "org/model")
    WS.register_weight_root(str(store), "engine")
    try:
        found = WS.find_external_weights("org/model", _cfg(tmp_path / "offloads" / "nvidia"))
        assert len(found) == 1
    finally:
        WS._REGISTERED.clear()


def test_purge_removes_the_files_and_reports_the_bytes(tmp_path):
    (tmp_path / "offloads" / "nvidia").mkdir(parents=True)
    store = tmp_path / "offloads" / "engine"
    store.mkdir()
    d = _repo(store, "org/model", blob_bytes=7777)
    freed, paths = WS.purge_external_weights("org/model",
                                             _cfg(tmp_path / "offloads" / "nvidia"))
    assert freed == 7777
    assert paths == [str(d)] or paths[0].endswith("models--org--model")
    assert not d.exists()


def test_a_dry_run_measures_without_deleting(tmp_path):
    (tmp_path / "offloads" / "nvidia").mkdir(parents=True)
    store = tmp_path / "offloads" / "engine"
    store.mkdir()
    d = _repo(store, "org/model", blob_bytes=1234)
    freed, paths = WS.purge_external_weights("org/model",
                                             _cfg(tmp_path / "offloads" / "nvidia"),
                                             dry_run=True)
    assert freed == 1234 and len(paths) == 1
    assert d.exists()


def test_nothing_anywhere_is_not_an_error(tmp_path):
    (tmp_path / "offloads" / "nvidia").mkdir(parents=True)
    freed, paths = WS.purge_external_weights("org/absent",
                                             _cfg(tmp_path / "offloads" / "nvidia"))
    assert freed == 0 and paths == []


def test_an_empty_model_id_matches_nothing(tmp_path):
    """Guards the degenerate 'models--' name against matching a directory."""
    (tmp_path / "offloads" / "nvidia").mkdir(parents=True)
    (tmp_path / "offloads" / "engine").mkdir()
    (tmp_path / "offloads" / "engine" / "models--").mkdir()
    assert WS.find_external_weights("", _cfg(tmp_path / "offloads" / "nvidia")) == []


@pytest.mark.parametrize("n,expected", [
    (0, "0 B"), (512, "512 B"), (4096, "4.0 KB"),
    (400 * 1024 ** 3, "400.0 GB"), (None, "0 B"),
])
def test_sizes_read_the_way_the_admin_ui_shows_them(n, expected):
    assert WS.human_bytes(n) == expected


def test_hardlinked_data_is_counted_once(tmp_path):
    """A cache that hardlinks rather than copies would otherwise have the weights
    counted once per link, overstating what a delete frees."""
    d = tmp_path / "repo"
    d.mkdir()
    real = d / "shard.safetensors"
    real.write_bytes(b"\x00" * 9000)
    os.link(real, d / "also-shard.safetensors")
    assert WS.dir_size(str(d)) == 9000


def test_a_cache_symlink_into_a_purged_store_is_removed(tmp_path):
    """The exact leftover: the HF cache held the model as a SYMLINK into the
    engine's directory, so purging the store left a dangling entry that still
    looks like a present model to anything that only stats the name."""
    (tmp_path / "offloads" / "nvidia").mkdir(parents=True)
    store = tmp_path / "offloads" / "colibri"
    store.mkdir()
    real = _repo(store, "org/model", blob_bytes=2048)
    hub = tmp_path / "offloads" / "hfcache" / "hub"
    hub.mkdir(parents=True)
    link = hub / WS.hf_dir_name("org/model")
    os.symlink(real, link)

    cfg = _cfg(tmp_path / "offloads" / "nvidia")
    # The link and the store are the same data — counted once, not twice.
    freed, _ = WS.purge_external_weights("org/model", cfg, dry_run=True)
    assert freed == 2048

    freed, paths = WS.purge_external_weights("org/model", cfg)
    assert freed == 2048
    assert not real.exists()
    assert not os.path.islink(link), "the dangling cache entry should be gone"
    assert str(link) in paths


def test_a_live_cache_symlink_is_left_alone(tmp_path):
    """Only a link whose target is GONE is pruned."""
    (tmp_path / "offloads" / "nvidia").mkdir(parents=True)
    store = tmp_path / "offloads" / "engine"
    store.mkdir()
    real = _repo(store, "org/keeper")
    hub = tmp_path / "offloads" / "hfcache" / "hub"
    hub.mkdir(parents=True)
    link = hub / WS.hf_dir_name("org/keeper")
    os.symlink(real, link)
    assert WS.prune_dangling_links("org/other", _cfg(tmp_path / "offloads" / "nvidia")) == []
    assert os.path.islink(link)
