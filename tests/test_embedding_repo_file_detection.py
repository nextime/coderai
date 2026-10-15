"""A feature detected by a FILE must be detectable before the file is there.

`_is_bge_m3` keys off `sparse_linear.pt` — the definitive signal that a repo is
a multi-vector embedder, since plain dense models do not ship one. It asked
`_resolve_repo_file`, which answers from the local disk and never the network.

That is correct for "give me the path" and wrong for "does this exist", and the
difference is invisible until the model has never been downloaded.

On a RENTED POD it has never been downloaded. The pod is disposable and its
cache starts empty, so on every pod, every time, bge-m3 loaded through the
DENSE sentence-transformers path — and the request for sparse was then refused
with a message telling the caller to serve the model on the coderai-embeddings
image, which is the image it was already running on.

Measured end to end on a real A40: 379 s, then HTTP 400 saying exactly that.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import embeddings as emb                        # noqa: E402


@pytest.fixture(autouse=True)
def _clear_cache():
    emb._REPO_FILE_CACHE.clear()
    yield
    emb._REPO_FILE_CACHE.clear()


@pytest.fixture
def no_local(monkeypatch):
    """Nothing in the cache — a freshly rented pod."""
    monkeypatch.setattr(emb, "_resolve_repo_file", lambda *a, **k: None)


def _hub(monkeypatch, files, calls=None):
    import huggingface_hub

    def _list(repo, *a, **k):
        if calls is not None:
            calls.append(repo)
        return list(files)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", _list, raising=False)


# ------------------------------------------------------ the cold pod case #
def test_the_head_is_found_on_a_pod_with_an_empty_cache(monkeypatch, no_local):
    """THE regression. Nothing on disk, and the repo does carry the head."""
    _hub(monkeypatch, ["config.json", "sparse_linear.pt", "colbert_linear.pt"])
    assert emb._repo_has_file("BAAI/bge-m3", "sparse_linear.pt") is True
    assert emb._is_bge_m3("BAAI/bge-m3", False) is True


def test_a_plain_dense_repo_is_still_dense(monkeypatch, no_local):
    _hub(monkeypatch, ["config.json", "model.safetensors"])
    assert emb._repo_has_file("some/plain-bert", "sparse_linear.pt") is False
    assert emb._is_bge_m3("some/plain-bert", False) is False


# --------------------------------------------------- what must not change #
def test_a_cached_file_needs_no_hub_call(monkeypatch):
    """The local answer still wins, and costs nothing."""
    monkeypatch.setattr(emb, "_resolve_repo_file", lambda *a, **k: "/cache/x.pt")
    calls = []
    _hub(monkeypatch, [], calls)
    assert emb._repo_has_file("BAAI/bge-m3", "sparse_linear.pt") is True
    assert calls == []


def test_a_local_directory_is_the_whole_answer(monkeypatch, tmp_path):
    """A path on disk is not a Hub repo: asking the Hub about it would look up
    someone else's model of the same name."""
    calls = []
    _hub(monkeypatch, ["sparse_linear.pt"], calls)
    monkeypatch.setattr(emb, "_resolve_repo_file", lambda *a, **k: None)
    assert emb._repo_has_file(str(tmp_path), "sparse_linear.pt") is False
    assert calls == []


def test_offline_installs_never_reach_the_network(monkeypatch, no_local):
    calls = []
    _hub(monkeypatch, ["sparse_linear.pt"], calls)
    for val in ("1", "true", "YES", "on"):
        emb._REPO_FILE_CACHE.clear()
        monkeypatch.setenv("HF_HUB_OFFLINE", val)
        assert emb._repo_has_file("BAAI/bge-m3", "sparse_linear.pt") is False
    assert calls == []


def test_an_unreachable_hub_answers_the_way_it_always_did(monkeypatch, no_local):
    """A metadata lookup must never be able to fail a load."""
    import huggingface_hub

    def _boom(*a, **k):
        raise OSError("no route to host")
    monkeypatch.setattr(huggingface_hub, "list_repo_files", _boom, raising=False)
    assert emb._repo_has_file("BAAI/bge-m3", "sparse_linear.pt") is False
    assert emb._is_bge_m3("BAAI/bge-m3", False) is False


def test_the_hub_is_asked_once_per_repo(monkeypatch, no_local):
    """A repo's file list does not change under a running server, and this sits
    on the load path of every embedding request."""
    calls = []
    _hub(monkeypatch, ["sparse_linear.pt"], calls)
    for _ in range(5):
        assert emb._repo_has_file("BAAI/bge-m3", "sparse_linear.pt") is True
    assert len(calls) == 1


def test_a_negative_answer_is_cached_too(monkeypatch, no_local):
    calls = []
    _hub(monkeypatch, ["config.json"], calls)
    for _ in range(4):
        assert emb._repo_has_file("some/plain", "sparse_linear.pt") is False
    assert len(calls) == 1


def test_two_repos_are_cached_separately(monkeypatch, no_local):
    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub, "list_repo_files",
        lambda repo, *a, **k: (["sparse_linear.pt"] if "bge-m3" in repo
                               else ["config.json"]), raising=False)
    assert emb._repo_has_file("BAAI/bge-m3", "sparse_linear.pt") is True
    assert emb._repo_has_file("other/model", "sparse_linear.pt") is False


def test_the_detector_uses_the_existence_check_not_the_path_one():
    """Pin the distinction, because the two read almost the same at the call
    site and only one of them works on a pod."""
    src = (ROOT / "codai/api/embeddings.py").read_text()
    blk = src.split("def _is_bge_m3")[1].split("\ndef ")[0]
    assert "_repo_has_file(" in blk
    assert "_resolve_repo_file(" not in blk


# ------------------------------------------- and then the heads must ARRIVE #
def test_the_head_is_downloaded_when_it_is_not_cached(monkeypatch, no_local):
    """Detecting the head is not having it. `from_pretrained` fetches the
    config and the weights the config NAMES — `sparse_linear.pt` is neither,
    so after the base model loads it is still absent on a fresh machine, and a
    cache-only lookup reports "This model has no sparse head." That reads as a
    property of the model rather than of the disk."""
    import huggingface_hub
    calls = []

    def _dl(repo, fname, *a, **k):
        calls.append((repo, fname, k.get("local_files_only")))
        return "/tmp/fetched.pt"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _dl, raising=False)
    monkeypatch.setattr(emb.os.path, "isfile", lambda p: p == "/tmp/fetched.pt")
    got = emb._fetch_repo_file("BAAI/bge-m3", "sparse_linear.pt")
    assert got == "/tmp/fetched.pt"
    # Downloaded for real: local_files_only must NOT be set on this call.
    assert calls and not calls[0][2]


def test_a_cached_head_is_not_downloaded_again(monkeypatch):
    monkeypatch.setattr(emb, "_resolve_repo_file", lambda *a, **k: "/cache/h.pt")
    import huggingface_hub

    def _boom(*a, **k):
        raise AssertionError("downloaded a file that was already here")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom, raising=False)
    assert emb._fetch_repo_file("BAAI/bge-m3", "sparse_linear.pt") == "/cache/h.pt"


def test_a_head_that_cannot_be_fetched_means_the_dense_path(monkeypatch, no_local):
    """None, not an exception: the caller already handles a model with no
    head, and a network blip must not turn into a failed load."""
    import huggingface_hub

    def _boom(*a, **k):
        raise OSError("gated repo")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom, raising=False)
    assert emb._fetch_repo_file("BAAI/bge-m3", "sparse_linear.pt") is None


def test_offline_does_not_fetch_heads(monkeypatch, no_local):
    import huggingface_hub

    def _boom(*a, **k):
        raise AssertionError("reached the network while offline")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert emb._fetch_repo_file("BAAI/bge-m3", "sparse_linear.pt") is None


def test_the_head_loader_fetches_rather_than_resolves():
    """The two read almost identically at the call site and only one of them
    works on a machine that has never seen the repo."""
    src = (ROOT / "codai/api/embeddings.py").read_text()
    blk = src.split("def _load_bge_m3")[1].split("\ndef ")[0]
    assert "_fetch_repo_file(" in blk
    assert "_resolve_repo_file(" not in blk
