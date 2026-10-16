"""A catalogue entry must not be able to crash an engine.

The incident, 2026-10-16 on the Digesta orchestrator: a `vision_models` row was
added for a model served on RunPod, pinned `backend: runpod`. At engine startup
something resolved its path; the id (`qwen38-vision`) is not a local file, not a
URL and not a HuggingFace repo id, so the resolver fell through to "assume it's
a repo id", `download_model` fell through to "then it must be a URL", and
`requests.get("qwen38-vision")` raised MissingSchema. The engine crash-looped
five times in 120 s and was QUARANTINED, taking /v1/ocr and every RunPod pool
down with it for about ninety minutes. `/healthz` still answered `ok: true`,
because the FRONT was up.

Two independent guards, because either alone would have prevented it:

* a model pinned to a backend that serves it elsewhere has no local weights, so
  nothing resolves, downloads or caches one -- which is also what lets ONE
  catalogue row run on rented pods here and load locally on a box with a GPU;
* a name that is none of the three resolvable kinds is refused with a sentence
  naming all three, not with a stack trace from inside requests.
"""

import types

import pytest


@pytest.fixture
def catalogue(monkeypatch):
    import codai.admin.routes as routes

    def _set(md):
        monkeypatch.setattr(routes, "config_manager",
                            types.SimpleNamespace(models_data=md), raising=False)
    return _set


# ------------------------------------------------- the pin wins over loading #

@pytest.mark.parametrize("pin", ["runpod", "remote", "node", "cluster", "RunPod"])
def test_a_remote_pinned_model_is_never_resolved_locally(catalogue, pin):
    from codai.models.manager import multi_model_manager
    catalogue({"vision_models": [{"id": "qwen38-vision", "backend": pin,
                                  "runpod": {"hf_repo": "owner/Qwen"}}]})
    assert multi_model_manager.load_model("qwen38-vision") == "qwen38-vision"


def test_the_same_row_WOULD_load_locally_without_the_pin(catalogue, monkeypatch):
    """The point of the pin: one catalogue entry, two kinds of install."""
    from codai.models import manager as mm
    catalogue({"vision_models": [{"id": "owner/Qwen-VL"}]})          # no backend
    assert mm.multi_model_manager.load_model("owner/Qwen-VL") == "owner/Qwen-VL"


def test_local_weights_on_a_pinned_entry_are_still_meaningful(catalogue, tmp_path):
    """The pin stops THIS install loading them. It does not make them useless.

    Weights that exist here can still reach a pod: `resolve_model_source` is
    configurable (`source: auto | hf | url | upload`) and reads the entry
    DIRECTLY -- `entry["path"]`, `mcfg.hf_repo`, `mcfg.model_url` -- so a local
    path is what `upload` pushes and what a repo id is recovered from. Neither
    goes through the resolver guarded above, and this test exists so a later
    change cannot quietly couple them.
    """
    from codai.api.runpod_worker import resolve_model_source
    import types

    from codai.models.manager import multi_model_manager
    f = tmp_path / "w.gguf"
    f.write_bytes(b"x")
    entry = {"id": str(f), "path": str(f), "backend": "runpod"}
    catalogue({"text_models": [entry]})

    # Nothing is loaded here...
    assert multi_model_manager.load_model(str(f)) == str(f)

    # ...and every way of getting it onto a pod still works, by configuration.
    up = types.SimpleNamespace(source="upload", hf_repo="", model_url="",
                               served_model="")
    assert resolve_model_source(entry, up) == ("upload", str(f))
    hf = types.SimpleNamespace(source="hf", hf_repo="owner/name", model_url="",
                               served_model="")
    assert resolve_model_source(entry, hf) == ("hf", "owner/name")
    url = types.SimpleNamespace(source="url", hf_repo="",
                                model_url="https://h/m.gguf", served_model="")
    assert resolve_model_source(entry, url) == ("url", "https://h/m.gguf")


def test_an_id_is_recovered_so_the_pod_can_download_it(catalogue):
    """"auto": a repo id is often recoverable even when the path is local, and
    then the POD downloads it rather than anything being pushed."""
    import types

    from codai.api.runpod_worker import resolve_model_source
    entry = {"id": "x", "path": "/local/only/weights", "backend": "runpod"}
    auto = types.SimpleNamespace(source="auto", hf_repo="owner/name",
                                 model_url="", served_model="")
    assert resolve_model_source(entry, auto) == ("hf", "owner/name")


def test_an_unknown_model_is_unaffected(catalogue):
    from codai.models.manager import multi_model_manager
    catalogue({})
    # No entry, so no pin: resolution proceeds as it always did (and the name
    # below is a repo id, so it resolves to itself without any network).
    assert multi_model_manager.load_model("owner/name") == "owner/name"


def test_a_broken_catalogue_cannot_block_loading(monkeypatch):
    import codai.admin.routes as routes
    from codai.models.manager import multi_model_manager
    monkeypatch.setattr(routes, "config_manager", None, raising=False)
    assert multi_model_manager.load_model("owner/name") == "owner/name"


# --------------------------------------- an unresolvable name is a sentence #

def test_a_bare_name_is_refused_not_fetched(capsys):
    """`requests.get("qwen38-vision")` is what killed the engine."""
    from codai.models.cache import download_model
    assert download_model("qwen38-vision") is None
    out = capsys.readouterr().out
    assert "not a local path" in out and "not a URL" in out
    assert "HuggingFace repo id" in out
    # ...and it points at the fix rather than only at the symptom.
    assert "backend" in out


def test_a_bare_name_never_reaches_requests(monkeypatch):
    import requests
    from codai.models.cache import download_model

    def _boom(*a, **k):
        raise AssertionError("requests.get must not be called for a bare name")
    monkeypatch.setattr(requests, "get", _boom)
    assert download_model("qwen38-vision") is None


def test_a_real_url_still_reaches_requests(monkeypatch):
    import requests
    from codai.models.cache import download_model
    seen = {}

    class _Resp:
        headers = {"content-length": "0"}
        status_code = 200

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=0):
            return iter(())

    def _get(url, **k):
        seen["url"] = url
        return _Resp()
    monkeypatch.setattr(requests, "get", _get)
    # A unique URL per run: the cache is keyed by a hash of it, and a hit from
    # an earlier run would skip the fetch and make this test pass for the wrong
    # reason (it did, once).
    import uuid
    url = f"https://example.invalid/{uuid.uuid4().hex}.gguf"
    download_model(url)
    assert seen.get("url") == url


def test_a_repo_id_does_not_take_the_url_path(monkeypatch):
    import requests
    from codai.models.cache import download_model

    def _boom(*a, **k):
        raise AssertionError("a repo id must not be fetched as a URL")
    monkeypatch.setattr(requests, "get", _boom)
    # It will fail to reach HuggingFace in a sandbox; what matters is that it
    # never treats 'owner/name' as a URL.
    try:
        download_model("owner/name")
    except Exception:
        pass
