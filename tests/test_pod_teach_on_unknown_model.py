"""A pod that has never heard of the model is taught, not written off.

A pool hands back whichever pod it holds. Adoption checks the image, the GPU
class and the bearer token — none of which say anything about the CATALOGUE —
so a pod rented for one name of a shared pool is handed to a request naming
another, and answers 404. The pool keeps it warm, so that 404 is permanent.

Two separate failures made it unreadable as well as fatal:

* `raise_for_status()` threw the pod's body away, so the only thing that
  reached the caller was "404 Client Error: Not Found for url: …" — which reads
  like a routing or network fault, not a catalogue one. The body says exactly
  which models the pod has.
* nothing acted on it. The remote GATEWAY teaches a pod that says "not
  available" and retries, but /v1/chat/completions is in its _SKIP_PATHS: text
  generation goes through the in-process RunPod backend, which had no such
  recovery.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.backends import runpod as rb                        # noqa: E402


class _Pod:
    pod_id = "pod-1"
    url = "https://203.0.113.5:1"


class _Pool:
    api_key = "cra-test"

    def acquire(self, timeout=1200.0, affinity=""):
        return _Pod(), _Pod.url

    def release(self, pod):
        pass

    def discard(self, pod, reason=""):
        pass


def _backend():
    b = rb.RunpodBackend.__new__(rb.RunpodBackend)
    b._mode = "pods"
    b._pool = _Pool()
    b._headers = {}
    b._url = ""
    b._model_id = "tuned-alpha"
    b._served_name = "tuned-alpha"
    b._last_usage = {}
    b._last_logprobs = None
    b._enter_request = lambda: None
    b._exit_request = lambda: None
    return b


class _Reply:
    def __init__(self, status, payload):
        self.status_code = status
        self.ok = status < 400
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


UNKNOWN = {"detail": "Model 'tuned-alpha' is not available. Use one of: base-awq"}


# --------------------------------------------------------------------------- #
def test_the_pods_own_words_reach_the_caller(monkeypatch):
    b = _backend()
    monkeypatch.setattr(rb.pod_http, "post",
                        lambda url, **kw: _Reply(400, {"detail": "context too long"}))
    with pytest.raises(Exception) as err:
        b.generate_chat([{"role": "user", "content": "x"}])
    assert "context too long" in str(err.value)
    assert "Client Error" not in str(err.value)


def test_an_unknown_model_is_taught_and_the_request_retried(monkeypatch):
    b = _backend()
    replies = iter([_Reply(404, UNKNOWN),
                    _Reply(200, {"choices": [{"message": {"content": "hi"}}],
                                 "usage": {}})])
    monkeypatch.setattr(rb.pod_http, "post", lambda url, **kw: next(replies))

    taught = []

    def _teach(base, key, model):
        taught.append((base, key, model))
        return True

    import codai.api.remote_gateway as gw
    monkeypatch.setattr(gw, "teach_pod", _teach)

    assert b.generate_chat([{"role": "user", "content": "x"}]) == "hi"
    assert len(taught) == 1
    base, key, model = taught[0]
    # The pod ROOT, not the /v1 the pool hands out: register lives at
    # <root>/v1/models/register and a doubled /v1 is a 404 of its own.
    assert base == _Pod.url
    assert key == "cra-test"
    assert model == "tuned-alpha"


def test_a_pod_that_cannot_be_taught_fails_with_its_own_message(monkeypatch):
    b = _backend()
    monkeypatch.setattr(rb.pod_http, "post", lambda url, **kw: _Reply(404, UNKNOWN))
    import codai.api.remote_gateway as gw
    monkeypatch.setattr(gw, "teach_pod", lambda *a, **k: False)

    with pytest.raises(rb.RemoteModelUnknown) as err:
        b.generate_chat([{"role": "user", "content": "x"}])
    assert "is not available" in str(err.value)


def test_teaching_is_tried_once_not_forever(monkeypatch):
    """A pod that still refuses after being taught is a real failure."""
    b = _backend()
    posts = []

    def _post(url, **kw):
        posts.append(url)
        return _Reply(404, UNKNOWN)

    monkeypatch.setattr(rb.pod_http, "post", _post)
    import codai.api.remote_gateway as gw
    monkeypatch.setattr(gw, "teach_pod", lambda *a, **k: True)

    with pytest.raises(rb.RemoteModelUnknown):
        b.generate_chat([{"role": "user", "content": "x"}])
    assert len(posts) == 2


def test_a_plain_404_that_is_not_about_the_model_is_not_retried(monkeypatch):
    b = _backend()
    posts = []

    def _post(url, **kw):
        posts.append(url)
        return _Reply(404, {"detail": "no such route"})

    monkeypatch.setattr(rb.pod_http, "post", _post)
    import codai.api.remote_gateway as gw
    monkeypatch.setattr(gw, "teach_pod",
                        lambda *a, **k: pytest.fail("should not teach"))

    with pytest.raises(Exception) as err:
        b.generate_chat([{"role": "user", "content": "x"}])
    assert "no such route" in str(err.value)
    assert len(posts) == 1


def test_an_openai_shaped_error_body_is_read_too():
    r = _Reply(401, {"error": {"message": "Invalid API key.",
                               "code": "invalid_api_key"}})
    assert rb._pod_detail(r) == "Invalid API key."


def test_teach_pod_sends_the_adapter_before_registering_the_entry(monkeypatch):
    """Registering alone names a content id the pod does not hold yet."""
    import codai.api.remote_gateway as gw

    order = []
    monkeypatch.setattr(gw, "ensure_text_loras",
                        lambda base, key, entry: order.append("loras"))
    monkeypatch.setattr(gw, "teach_model",
                        lambda base, key, model: order.append("register") or True)
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "_model_entry_for",
                        lambda name: {"id": name, "loras": [{"name": name}]},
                        raising=False)

    assert gw.teach_pod("https://pod", "k", "tuned-alpha") is True
    assert order == ["loras", "register"]


def test_teach_pod_does_not_register_when_the_adapter_cannot_be_sent(monkeypatch):
    """A pod with the entry but not the bytes fails at load, somewhere nobody looks."""
    import codai.api.remote_gateway as gw

    def _boom(*a, **k):
        raise RuntimeError("upload refused")

    monkeypatch.setattr(gw, "ensure_text_loras", _boom)
    monkeypatch.setattr(gw, "teach_model",
                        lambda *a, **k: pytest.fail("should not register"))
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "_model_entry_for", lambda name: {"id": name},
                        raising=False)

    assert gw.teach_pod("https://pod", "k", "tuned-alpha") is False
