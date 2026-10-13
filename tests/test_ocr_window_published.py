"""The window an attached engine reads must be the one its server serves.

An OCR engine attached through a lease reads `/v1/models` from the GATEWAY, not
from the pod and not from this install — and the gateway answered with just
`{id, object, owned_by}`. So coderai's own 0.3.53 work (publishing
`max_model_len` on `/v1/models`) was invisible on the one path that needs it,
and the worker said:

    surya-ocr-2 does not report a context window on /v1/models; resumption
    cannot stop before the server does — set ocr.surya_context_tokens to the
    served --max-model-len

which reads as the POD failing to advertise its window, and was reported that
way. The pod was never asked.

The second half: a pod was launched with THIS INSTALL's engine config, so a
per-model `ctx` on the runpod block did nothing — an entry asking for 18432 got
a pod serving 32768. Publishing a window while serving another is worse than
publishing none: resumption either stops early or overruns and gets a 400.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.ocr import gateway                                      # noqa: E402


def _reset():
    gateway.revoke()
    gateway._windows.clear()


def _models(monkeypatch, window):
    import asyncio
    monkeypatch.setattr(gateway, "_window_for", lambda m: window)
    token = gateway.issue("m", "served-name")
    reply = asyncio.run(gateway.models(token))
    return json.loads(bytes(reply.body).decode())


def test_the_window_is_published_to_the_engine(monkeypatch):
    _reset()
    body = _models(monkeypatch, 32768)
    assert body["data"][0]["max_model_len"] == 32768
    # The id still has to be the engine's own checkpoint name: surya refuses to
    # attach to a server whose /v1/models disagrees with it.
    assert body["data"][0]["id"] == "served-name"


def test_an_unknown_window_is_omitted_not_guessed(monkeypatch):
    # "Unknown" is a case the engine handles — it refuses to resume and says
    # which setting to set. A guessed number would make it overrun instead.
    _reset()
    assert "max_model_len" not in _models(monkeypatch, 0)["data"][0]


def test_a_learned_window_beats_the_configured_one():
    _reset()
    gateway.learn_window("m", 32768)
    assert gateway._window_for("m") == 32768


def test_learning_ignores_rubbish():
    _reset()
    for bad in (0, -1, None, "", "lots"):
        gateway.learn_window("m", bad)
    assert gateway._window_for("m") in (0, None) or isinstance(
        gateway._window_for("m"), int)
    assert "m" not in gateway._windows


def test_the_configured_window_is_read_from_the_model_entry(monkeypatch):
    _reset()
    import codai.models.manager as mm
    import codai.models.placement as pl
    monkeypatch.setattr(pl, "model_entry", lambda m: {"runpod": {"ctx": 18432}})
    monkeypatch.setattr(mm, "context_window_for", lambda mgr, m, meta:
                        (meta.get("runpod") or {}).get("ctx"))
    assert gateway._window_for("m") == 18432


def test_a_broken_lookup_is_unknown_not_an_error(monkeypatch):
    _reset()
    import codai.models.placement as pl
    monkeypatch.setattr(pl, "model_entry",
                        lambda m: (_ for _ in ()).throw(RuntimeError("no config")))
    assert gateway._window_for("m") == 0


# --- the pod has to SERVE what we publish ----------------------------------

def test_a_models_own_ctx_reaches_the_pod():
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    blk = src.split("forwarded = engine_config_for_pod(eng)")[1][:1200]
    assert '_own = int(getattr(mcfg, "ctx", 0) or 0)' in blk
    assert 'forwarded["ctx"] = _own' in blk


def test_the_override_is_announced():
    """Two different numbers for one pod must not be a silent difference."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    assert "pod context window" in src and "this model's own" in src
