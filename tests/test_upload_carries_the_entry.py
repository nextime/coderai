"""An uploaded model must reach the pod with its whole entry, engine first.

2026-11-25, a spark2_5 decision model served with `source: upload` on a
`coderai-llamampere` pod:

  [remote-gateway] uploading spark-x2.5-4b-...-q4_k_m.gguf (2.6 GB) to https://pod
  [remote-gateway] uploaded spark-x2.5-4b-...-q4_k_m.gguf to https://pod
  [runpod] pod did not know 'digesta-decisionale-4b' — taught it, retrying
  Model 'digesta-decisionale-4b' failed to load: Unknown backend: cpu

The upload registered `{path, model_type, alias}` and nothing else: no
`backend` (a RunPod model carries `backend: "runpod"`, which means nothing on a
pod, so it fell through to "cpu"), and none of the per-engine block -- not even
llamampere's `--jinja`, so a load that DID succeed would have rendered every
prompt without the model's chat template. The hf/url teach path had carried the
engine since diogene-alpha; upload never did. And the pod's catalogue counts a
second registration at the same path as "already known", so it could not be
corrected afterwards: the entry has to travel WITH the bytes.
"""
import json

import pytest

from codai.api import models_transfer as mt
from codai.api import remote_gateway as rg

ENTRY = {
    "id": "digesta-decisionale-4b",
    "alias": "digesta-decisionale-4b",
    "backend": "runpod",
    "thinking": "off",
    "llamampere": {"ctx": 4096, "parallel": 1, "flash_attn": "on",
                   "extra_args": "-b 2048 -ub 512 --jinja --no-context-shift"},
    "runpod": {"mode": "pods", "engine": "coderai-llamampere", "source": "upload",
               "pool": "digesta-decisionale",
               "served_model": "digesta-decisionale-4b", "ctx": 4096},
}


def _main_row(seed_json, name):
    rows = json.loads(seed_json)
    rows = rows if isinstance(rows, list) else [rows]
    return next(r for r in rows if str(r.get("path", "")).endswith(name))


def test_the_upload_seed_carries_engine_settings_and_name():
    seed = rg._upload_seed(ENTRY, "m.gguf")
    row = _main_row(seed, "m.gguf")
    assert row["backend"] == "llamampere"             # THE BUG: absent -> "cpu"
    assert row["llamampere"]["ctx"] == 4096
    assert row["llamampere"]["parallel"] == 1
    assert "--jinja" in row["llamampere"]["extra_args"]
    assert row["thinking"] == "off"
    assert row["alias"] == "digesta-decisionale-4b"
    assert "runpod" not in row                        # placement stays here


@pytest.fixture
def gguf(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"GGUF" + b"\0" * 64)
    entry = dict(ENTRY, path=str(p))
    rg._UPLOADED.clear()
    rg._UPLOADED_PATHS.clear()
    return entry


class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(self.status_code)


def test_a_fresh_upload_sends_the_entry(gguf, monkeypatch):
    sent = {}
    monkeypatch.setattr("codai.api.pod_http.get",
                        lambda *a, **k: _Resp(404))

    def _post(url, params=None, data=None, headers=None, timeout=None):
        sent["url"], sent["params"] = url, params
        return _Resp(200, {"path": "/pod/models/m.gguf", "registered": "entry"})
    monkeypatch.setattr("codai.api.pod_http.post", _post)

    assert rg.ensure_model_uploaded("https://pod:1", "k", gguf) == "/pod/models/m.gguf"
    assert sent["url"].endswith("/v1/models/upload")
    row = _main_row(sent["params"]["entry"], "m.gguf")
    assert row["backend"] == "llamampere"


def test_weights_already_there_still_get_their_entry(gguf, monkeypatch):
    """A warm or adopted pod holds the bytes but may not know the entry."""
    posted = {}
    monkeypatch.setattr("codai.api.pod_http.get",
                        lambda *a, **k: _Resp(200, {"path": "/pod/models/m.gguf"}))

    def _post(url, data=None, headers=None, timeout=None, **_):
        posted["url"], posted["rows"] = url, json.loads(data)
        return _Resp(200, {"added": ["/pod/models/m.gguf"]})
    monkeypatch.setattr("codai.api.pod_http.post", _post)

    assert rg.ensure_model_uploaded("https://pod:1", "k", gguf) == "/pod/models/m.gguf"
    assert posted["url"].endswith("/v1/models/register")
    row = next(r for r in posted["rows"] if r["path"] == "/pod/models/m.gguf")
    assert row["backend"] == "llamampere"


def test_the_pod_registers_the_sent_entry_at_its_own_path(monkeypatch):
    got = []
    monkeypatch.setattr(mt, "register_model_runtime",
                        lambda row, section: got.append((dict(row), section)) or "added")
    monkeypatch.setattr(mt, "_register", lambda *a, **k: pytest.fail("bare path used"))
    seed = rg._upload_seed(ENTRY, "m.gguf")

    assert mt._register_upload("/pod/models/m.gguf", "text_models", "", seed) == "entry"
    row, section = got[0]
    assert row["path"] == "/pod/models/m.gguf"
    assert row["backend"] == "llamampere"
    assert section == "text_models"


def test_without_an_entry_the_pod_keeps_the_bare_registration(monkeypatch):
    bare = []
    monkeypatch.setattr(mt, "_register", lambda *a: bare.append(a))
    monkeypatch.setattr(mt, "register_model_runtime",
                        lambda *a: pytest.fail("no entry was sent"))
    assert mt._register_upload("/pod/models/x.gguf", "text_models", "x", "") == "bare"
    assert bare == [("/pod/models/x.gguf", "text_models", "x")]
