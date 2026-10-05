"""The models page's scan cache, and the part of it that survives a restart.

The in-process cache was already stale-while-revalidate, but it started empty in
every new process — so the first read after a restart recomputed the whole scan
synchronously, which is exactly when someone is waiting for the page. Measured on
a real configuration: 3.69s cold, 0.001s once the result is read back from disk.
"""
import json
import types

import pytest

from codai.admin import routes


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "config_manager",
                        types.SimpleNamespace(config_dir=tmp_path, models_path=None))
    return {"data": None, "sig": None, "at": 0.0, "busy": False, "disk": "model-scan.json"}


def test_a_result_is_written_where_the_next_process_will_find_it(state, tmp_path):
    routes._save_disk_cache(state, {"hf": [{"id": "a/b"}], "gguf": []})
    path = tmp_path / "cache" / "model-scan.json"
    assert path.is_file()
    blob = json.loads(path.read_text())
    assert blob["version"] == routes._DISK_CACHE_VERSION
    assert blob["data"]["hf"][0]["id"] == "a/b"


def test_a_fresh_process_is_seeded_from_disk(state):
    routes._save_disk_cache(state, {"hf": [{"id": "a/b"}], "gguf": []})
    fresh = {"data": None, "sig": None, "at": 0.0, "busy": False, "disk": "model-scan.json"}
    routes._load_disk_cache(fresh)
    assert fresh["data"]["hf"][0]["id"] == "a/b"


def test_the_restored_entry_is_never_treated_as_verified(state):
    """The signature is a list of mtimes. Restoring it would serve a list that
    might be hours out of date WITHOUT revalidating — so it is deliberately left
    unset, which makes the first read return instantly and refresh."""
    state["sig"] = ("something",)
    routes._save_disk_cache(state, {"hf": [], "gguf": []})
    fresh = {"data": None, "sig": ("stale",), "at": 999.0, "busy": False,
             "disk": "model-scan.json"}
    routes._load_disk_cache(fresh)
    assert fresh["sig"] is None
    assert fresh["at"] == 0.0


def test_disk_is_read_once_per_process(state, tmp_path):
    routes._save_disk_cache(state, {"hf": [{"id": "first"}], "gguf": []})
    fresh = {"data": None, "sig": None, "at": 0.0, "busy": False, "disk": "model-scan.json"}
    routes._load_disk_cache(fresh)
    # A later write must not be picked up by a second seed call: the in-memory
    # cache is authoritative once the process is running.
    routes._save_disk_cache(state, {"hf": [{"id": "second"}], "gguf": []})
    routes._load_disk_cache(fresh)
    assert fresh["data"]["hf"][0]["id"] == "first"


@pytest.mark.parametrize("content", [
    "not json at all",
    '{"version": 999, "data": {"hf": []}}',   # a version this build does not know
    '{"version": 1}',                          # no data
    '[]',                                      # not an object
])
def test_an_unusable_cache_file_falls_back_to_scanning(state, tmp_path, content):
    path = tmp_path / "cache" / "model-scan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    fresh = {"data": None, "sig": None, "at": 0.0, "busy": False, "disk": "model-scan.json"}
    routes._load_disk_cache(fresh)
    assert fresh["data"] is None, "a bad cache must not become the answer"


def test_nothing_is_written_without_a_config_dir(monkeypatch):
    monkeypatch.setattr(routes, "config_manager", None)
    s = {"data": None, "disk": "model-scan.json"}
    assert routes._disk_cache_path(s) == ""
    routes._save_disk_cache(s, {"hf": []})      # must not raise


def test_a_state_with_no_disk_name_is_memory_only(state):
    s = dict(state)
    s["disk"] = None
    assert routes._disk_cache_path(s) == ""


def test_the_cached_reader_does_not_block_on_a_rescan(state, monkeypatch):
    """The point of persisting: a read after a restart returns the stored result
    straight away and revalidates behind it, instead of making the caller wait for
    the scan. Measured on a real configuration: 3.69s before, 0.001s after."""
    import time
    routes._save_disk_cache(state, {"hf": [{"id": "from-disk"}], "gguf": []})
    fresh = {"data": None, "sig": None, "at": 0.0, "busy": False, "disk": "model-scan.json"}
    started = []

    def _slow_scan():
        started.append(time.time())
        time.sleep(0.6)
        return {"hf": [{"id": "rescanned"}], "gguf": []}

    monkeypatch.setattr(routes, "_scan_signature", lambda: ("sig",))
    t0 = time.time()
    got = routes._cached(fresh, _slow_scan)
    elapsed = time.time() - t0
    assert elapsed < 0.2, f"the caller waited {elapsed:.2f}s for the scan"
    assert got["hf"][0]["id"] == "from-disk"
    # …and the rescan really was kicked off, so the list does not stay stale.
    assert started, "the background revalidation should have started"
    for _ in range(40):
        if fresh["data"]["hf"][0]["id"] == "rescanned":
            break
        time.sleep(0.05)
    assert fresh["data"]["hf"][0]["id"] == "rescanned"


def test_without_a_stored_result_the_first_read_still_scans(state, monkeypatch):
    """No cache file (a genuinely first run) must not serve an empty list."""
    monkeypatch.setattr(routes, "_scan_signature", lambda: ("sig",))
    fresh = {"data": None, "sig": None, "at": 0.0, "busy": False, "disk": "model-scan.json"}
    got = routes._cached(fresh, lambda: {"hf": [{"id": "scanned"}], "gguf": []})
    assert got["hf"][0]["id"] == "scanned"
