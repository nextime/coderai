"""A model served from a pod must be visible on the models page.

Reported on 2026-11-08 against a RunPod-only install: "there is the base model
as to be downloaded and no configs". Six configured, enabled, traffic-serving
models had NO row on /admin/models at all -- digesta-vision, qwen38-vision,
qwen38-awq, qwen38-extract, digesta-alpha, diogene-alpha -- while the single HF
repo they share appeared as a base model to download with zero configs.

The cause: `_scan_caches` files each config under `m.get("path") or m.get("id")`
and then builds rows from things that exist as BYTES -- GGUF files on disk, HF
cache entries, and configured repos whose files were evicted. That last branch
guards on `is_huggingface_model_id`, which rejects a name like
"digesta-vision" (no owner/repo slash), so a model whose weights live on a
rented pod matched nothing and vanished.

Such a row must NOT offer a download: the pod fetches the weights, and pulling
them here would put tens of GB on an orchestrator meant to hold none.
"""
import types

import pytest

from codai.admin import routes


def _catalogue():
    """A RunPod-only catalogue: SEVEN aliases over one base repo, as live."""
    REPO = "cyankiwi/Qwen3.8-27B-AWQ-INT4"
    def _rp(**over):
        base = {"mode": "pods", "hf_repo": REPO}
        base.update(over)
        return base
    return {
        "vision_models": [
            {"id": "digesta-vision", "enabled": True, "backend": "runpod",
             "n_ctx": 32768,
             "runpod": _rp(served_model="digesta-vision", pool="digesta-vision",
                           ctx=32768, pod_max_parallel_requests=12,
                           cloud_types=["SECURE", "COMMUNITY"]),
             "vllm": {"max_num_seqs": 12}},
            {"id": "qwen38-vision", "enabled": True, "backend": "runpod",
             "runpod": _rp(served_model="qwen38-vision", pool="digesta-qwen38",
                           ctx=122880)},
        ],
        "text_models": [
            {"id": "qwen38-awq", "enabled": True, "backend": "runpod",
             "runpod": _rp(served_model="qwen38-awq", pool="digesta-qwen38",
                           ctx=122880)},
            {"id": "digesta-extraction", "enabled": True, "backend": "runpod",
             "n_ctx": 16384,
             "runpod": _rp(served_model="digesta-extraction",
                           pool="digesta-extraction", ctx=16384,
                           pod_max_parallel_requests=16)},
            {"id": "digesta-alpha", "enabled": True, "backend": "runpod",
             "runpod": _rp(served_model="digesta-alpha", pool="digesta-qwen38",
                           ctx=122880)},
            {"id": "diogene-alpha", "enabled": True, "backend": "runpod",
             "runpod": _rp(served_model="diogene-alpha", pool="digesta-qwen38",
                           ctx=122880)},
            # An external OpenAI-compatible server: also no local bytes, and a
            # base of its own (its URL), so it must not be folded into the repo.
            {"id": "remote-host-model", "enabled": True, "backend": "host",
             "host": {"url": "https://example.invalid/v1"}},
            # A plain local HF model must keep going through the cache scan.
            {"id": "BAAI/bge-m3", "enabled": True},
        ],
    }


REPO = "cyankiwi/Qwen3.8-27B-AWQ-INT4"


@pytest.fixture
def scanned(monkeypatch):
    """`_scan_caches` on a weights-less orchestrator -- the case under test.

    The HF cache scan is stubbed EMPTY on purpose. A RunPod-only box holds no
    weights at all, which is exactly why the base model has no row to hang its
    configs on; leaving the real scan in place made this test pass or fail
    depending on whether the developer's own machine happened to have the repo
    cached (zeiss does, Aruba does not).
    """
    import huggingface_hub

    def _empty_scan(_d=None):
        return types.SimpleNamespace(repos=[], warnings=[])

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", _empty_scan)
    monkeypatch.setattr(routes, "config_manager",
                        types.SimpleNamespace(models_data=_catalogue()),
                        raising=False)
    return routes._scan_caches()


def _by_id(rows):
    return {r["id"]: r for r in rows}


def _cfg_ids(row):
    return sorted((c.get("settings") or {}).get("id") for c in row["configs"])


def test_the_base_model_gets_one_row_not_seven(scanned):
    rows = _by_id(scanned["hf"])
    assert REPO in rows, "the base model needs a row; got %s" % sorted(rows)
    for alias in ("digesta-extraction", "digesta-alpha", "diogene-alpha",
                  "qwen38-awq", "qwen38-vision", "digesta-vision"):
        assert alias not in rows, (
            "%s is a CONFIG of %s, not a model of its own" % (alias, REPO))


def test_every_alias_is_a_config_under_it(scanned):
    row = _by_id(scanned["hf"])[REPO]
    assert _cfg_ids(row) == sorted([
        "digesta-alpha", "digesta-extraction", "digesta-vision",
        "diogene-alpha", "qwen38-awq", "qwen38-vision"])


def test_each_config_keeps_its_own_settings(scanned):
    """Two configs of one base model, differing exactly as configured."""
    row = _by_id(scanned["hf"])[REPO]
    by = {(c["settings"] or {}).get("id"): c["settings"] for c in row["configs"]}
    assert by["digesta-extraction"]["runpod"]["ctx"] == 16384
    assert by["digesta-alpha"]["runpod"]["ctx"] == 122880
    assert by["digesta-extraction"]["runpod"]["pool"] == "digesta-extraction"
    assert by["digesta-alpha"]["runpod"]["pool"] == "digesta-qwen38"


def test_the_vision_config_is_grouped_with_the_text_ones(scanned):
    """Same weights, so same base model, whatever section it lives in."""
    row = _by_id(scanned["hf"])[REPO]
    cats = {c.get("cat") for c in row["configs"]}
    assert "vision_models" in cats and "text_models" in cats


def test_the_base_model_is_not_offered_as_a_download(scanned):
    """Nothing is missing: the pod fetches the weights."""
    row = _by_id(scanned["hf"])[REPO]
    assert row["missing"] is False
    assert row.get("to_download") is not True
    assert row["remote_only"] is True


def test_a_host_backed_model_is_not_folded_into_a_repo(scanned):
    rows = _by_id(scanned["hf"])
    assert "https://example.invalid/v1" in rows
    assert rows["https://example.invalid/v1"]["remote_only"] is True


def test_a_local_model_is_left_to_the_cache_scan(scanned):
    rows = [r for r in scanned["hf"] if r["id"] == "BAAI/bge-m3"]
    assert all(not r.get("remote_only") for r in rows)
    assert len(rows) <= 1, "no duplicate rows for a real repo id"
