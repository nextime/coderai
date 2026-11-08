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
    """A RunPod-only vision/text catalogue, shaped like the live one."""
    return {
        "vision_models": [
            {"id": "digesta-vision", "enabled": True, "backend": "runpod",
             "n_ctx": 32768,
             "runpod": {"mode": "pods", "hf_repo": "cyankiwi/Qwen3.8-27B-AWQ-INT4",
                        "served_model": "digesta-vision", "pool": "digesta-vision",
                        "pod_max_parallel_requests": 12, "cloud_types": ["SECURE"]},
             "vllm": {"max_num_seqs": 12}},
        ],
        "text_models": [
            {"id": "qwen38-awq", "enabled": True, "backend": "runpod",
             "runpod": {"mode": "pods", "hf_repo": "cyankiwi/Qwen3.8-27B-AWQ-INT4",
                        "served_model": "qwen38-awq", "pool": "digesta-qwen38"}},
            # An external OpenAI-compatible server: also no local bytes.
            {"id": "remote-host-model", "enabled": True, "backend": "host",
             "host": {"url": "https://example.invalid/v1"}},
            # A plain local HF model must keep going through the cache scan.
            {"id": "BAAI/bge-m3", "enabled": True},
        ],
    }


@pytest.fixture
def scanned(monkeypatch, tmp_path):
    """`_scan_caches` with the HF/GGUF scans empty, as on a weights-less box."""
    monkeypatch.setattr(routes, "config_manager",
                        types.SimpleNamespace(models_data=_catalogue()),
                        raising=False)
    return routes._scan_caches()


def _by_id(rows):
    return {r["id"]: r for r in rows}


def test_a_pod_served_model_gets_its_own_row(scanned):
    rows = _by_id(scanned["hf"])
    assert "digesta-vision" in rows, (
        "a configured, enabled, pod-served model must appear on the page; "
        "rows were: %s" % sorted(rows))
    row = rows["digesta-vision"]
    assert row["in_config"] is True
    assert row["model_type"] == "vision_models"


def test_the_row_carries_the_config_the_editor_needs(scanned):
    row = _by_id(scanned["hf"])["digesta-vision"]
    rp = (row["settings"] or {}).get("runpod") or {}
    # The exact fields the RunPod panel populates from.
    assert rp.get("pool") == "digesta-vision"
    assert rp.get("pod_max_parallel_requests") == 12
    assert rp.get("cloud_types") == ["SECURE"]
    assert (row["settings"].get("vllm") or {}).get("max_num_seqs") == 12
    # ...and the multi-config list the pills render from.
    assert row["configs"], "the config list must not be empty"


def test_it_is_not_offered_as_a_download(scanned):
    """Nothing is missing, so no download/re-download button."""
    row = _by_id(scanned["hf"])["digesta-vision"]
    assert row["missing"] is False
    assert row.get("to_download") is not True
    assert row["remote_only"] is True


def test_the_row_says_what_it_actually_serves(scanned):
    rows = _by_id(scanned["hf"])
    assert rows["digesta-vision"]["source_repo"] == "cyankiwi/Qwen3.8-27B-AWQ-INT4"
    assert rows["remote-host-model"]["source_repo"] == "https://example.invalid/v1"


def test_several_ids_sharing_one_repo_each_get_their_own_row(scanned):
    """The bug's visible symptom: one repo, many models, one row.

    digesta-vision and qwen38-awq pull the SAME repo. Both are separately
    configured models with different pools and limits, so both need a row.
    """
    rows = _by_id(scanned["hf"])
    assert {"digesta-vision", "qwen38-awq"} <= set(rows)
    assert rows["digesta-vision"]["settings"]["runpod"]["pool"] != \
        rows["qwen38-awq"]["settings"]["runpod"]["pool"]


def test_a_local_model_is_left_to_the_cache_scan(scanned):
    """An owner/repo id must not be duplicated by this branch.

    `BAAI/bge-m3` has no `runpod` block here, and even if it had, the evicted
    branch and the HF cache scan know its size and revisions. Only names that
    cannot be a repo id belong to the new path.
    """
    rows = [r for r in scanned["hf"] if r["id"] == "BAAI/bge-m3"]
    assert all(not r.get("remote_only") for r in rows)
    assert len(rows) <= 1, "no duplicate rows for a real repo id"
