"""A multimodal GGUF pinned to llama-server must not go blind.

``LlamaSrvBackend`` declared ``supports_vision = True`` and forwarded multipart
image requests faithfully -- while never passing ``--mmproj`` to the server it
launched. llama.cpp loads the vision tower only when told, and a multimodal GGUF
without its projector is a text model that accepts an image and describes
nothing: it answers, plausibly, about nothing it was shown. The claim was there
and the capability was not, which is the worst arrangement of the two -- a client
reads ``image_to_text`` in the catalogue and gets confident fiction.

The projector is named by the model's own models.json entry, the same field the
in-process path reads, so a model keeps its sight when it is pinned here.
"""

import os

import pytest

import codai.api.llamasrv_worker as worker
from codai.backends.llamasrv import LlamaSrvBackend
from codai.config import LlamaSrvConfig, LlamAmpereConfig


def _argv(cfg, mmproj=None):
    return worker.build_argv(cfg, "/bin/llama-server", "/AI/m.gguf",
                             8080, "127.0.0.1", "secret", None, mmproj)


# ------------------------------------------------------------- the argv ------

def test_the_projector_is_passed_when_there_is_one():
    argv = _argv(LlamaSrvConfig(), "/AI/m-mmproj.gguf")
    assert "--mmproj" in argv
    assert argv[argv.index("--mmproj") + 1] == "/AI/m-mmproj.gguf"


def test_nothing_is_passed_for_a_text_only_model():
    """Passing an empty value would be a flag with no argument, and the server
    would refuse to start -- turning 'no vision' into 'no model'."""
    for value in (None, "", "   "):
        assert "--mmproj" not in _argv(LlamaSrvConfig(), value)


def test_an_operators_own_projector_is_not_overridden():
    """llama-server takes the LAST --mmproj, so adding ours after an explicit one
    in extra_args would silently beat the override it was written to apply."""
    for flag in ("--mmproj", "-mm"):
        cfg = LlamaSrvConfig()
        cfg.extra_args = f"{flag} /AI/other-projector.gguf"
        argv = _argv(cfg, "/AI/entry-projector.gguf")
        assert argv.count("--mmproj") + argv.count("-mm") == 1
        assert "/AI/other-projector.gguf" in argv
        assert "/AI/entry-projector.gguf" not in argv


def test_extra_args_still_reach_the_server_alongside_a_projector():
    cfg = LlamaSrvConfig()
    cfg.extra_args = "--threads 8"
    argv = _argv(cfg, "/AI/m-mmproj.gguf")
    assert "--mmproj" in argv and "--threads" in argv and "8" in argv


# --------------------------------------------------- resolving it from the entry

@pytest.fixture
def catalogue(monkeypatch, tmp_path):
    """A models_data with one vision model and one text model."""
    gguf = tmp_path / "coder.gguf"
    gguf.write_bytes(b"GGUF")
    proj = tmp_path / "coder-mmproj.gguf"
    proj.write_bytes(b"GGUF")
    entries = {"text_models": [
        {"alias": "seer", "path": str(gguf), "mmproj": str(proj),
         "capabilities": ["text_generation", "image_to_text"]},
        {"alias": "blind", "path": str(tmp_path / "plain.gguf")},
        {"alias": "promised", "path": str(gguf),
         "mmproj": str(tmp_path / "gone-mmproj.gguf")},
    ]}

    class FakeCM:
        models_data = entries

    import codai.admin.routes as ar
    monkeypatch.setattr(ar, "config_manager", FakeCM, raising=False)
    return {"gguf": str(gguf), "proj": str(proj)}


def test_the_projector_comes_from_the_models_entry(catalogue):
    be = LlamaSrvBackend(LlamAmpereConfig())
    assert be._resolve_mmproj("seer") == os.path.abspath(catalogue["proj"])


def test_a_text_model_resolves_no_projector(catalogue):
    be = LlamaSrvBackend(LlamAmpereConfig())
    assert be._resolve_mmproj("blind") is None


def test_a_declared_but_missing_projector_is_skipped_not_passed(catalogue, capsys):
    """llama-server refuses to start when --mmproj names a file that is not
    there. Serving text is a degradation; refusing to serve is an outage."""
    be = LlamaSrvBackend(LlamAmpereConfig())
    assert be._resolve_mmproj("promised") is None
    out = capsys.readouterr().out
    assert "declared but not on disk" in out
    assert "without vision" in out


def test_an_unknown_model_resolves_nothing(catalogue):
    be = LlamaSrvBackend(LlamAmpereConfig())
    assert be._resolve_mmproj("nobody") is None


def test_the_gguf_and_the_projector_come_from_the_SAME_entry(catalogue):
    """Two independent lookups could match different entries -- one model's
    weights with another's tower, which loads and then answers nonsense."""
    be = LlamaSrvBackend(LlamAmpereConfig())
    entry = be._entry_for("seer")
    assert entry is not None
    assert be._resolve_gguf("seer") == os.path.abspath(catalogue["gguf"])
    assert be._resolve_mmproj("seer") == os.path.abspath(entry["mmproj"])


# ------------------------------------------------------------ the whole path --

def test_loading_a_vision_model_hands_the_projector_to_the_launch(catalogue, monkeypatch):
    """The end-to-end wiring: what the backend resolves has to arrive at
    ensure_service, or everything above this is decoration."""
    seen = {}

    def _fake_ensure(cfg, **kw):
        seen.update(kw)
        return "http://127.0.0.1:9", "bearer"

    monkeypatch.setattr(worker, "ensure_service", _fake_ensure)
    monkeypatch.setattr(worker, "resolve_service_key",
                        lambda cfg, path=None: (path, "llamampere::k"))
    be = LlamaSrvBackend(LlamAmpereConfig())
    be.load_model("seer")
    assert seen.get("mmproj") == os.path.abspath(catalogue["proj"])


def test_loading_a_text_model_hands_over_nothing(catalogue, monkeypatch):
    seen = {}
    monkeypatch.setattr(worker, "ensure_service",
                        lambda cfg, **kw: (seen.update(kw), ("http://x", "b"))[1])
    monkeypatch.setattr(worker, "resolve_service_key",
                        lambda cfg, path=None: (path, "llamasrv::k"))
    be = LlamaSrvBackend(LlamaSrvConfig())
    be.load_model("blind")
    assert seen.get("mmproj") is None
