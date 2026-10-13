"""olmOCR could not reach a pod: CERTIFICATE_VERIFY_FAILED.

A pod on the direct-TCP path serves HTTPS with a certificate signed by this
install's OWN CA (codai/api/pod_tls.py), addressed by a public IP that was not
known when the certificate was issued. Plain `requests` fails that twice over.

Surya never hit it because surya attaches through the fan-out gateway, which
learned to trust pod certificates in 0.3.16. olmOCR talks to the chosen upstream
directly -- it is the fallback engine, so the failure only showed up when surya
was the one being exercised and olmOCR was asked to cover for it. Same missing
trust, second place.

`pod_http` is the drop-in that already encodes the rule: https to a bare IP is
verified against our CA with the hostname check off; everything else goes to
`requests` untouched.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import pod_http                                   # noqa: E402
from codai.ocr import olmocr as om                               # noqa: E402


class _Reply:
    status_code = 200

    def json(self):
        return {"choices": [{"message": {"content": "testo"}}]}


class _Cfg:
    olmocr_api_key = ""
    olmocr_timeout = 30.0
    olmocr_model_id = "olmocr"


def _engine():
    e = om.OlmOcrEngine.__new__(om.OlmOcrEngine)
    e.cfg = _Cfg()
    e._base = "https://1.2.3.4:8000/v1"
    e._model_name = lambda: "olmocr"
    return e


def test_a_pod_url_goes_through_pod_http(monkeypatch):
    """The whole bug: this call used to go to requests and fail verification."""
    seen = {}

    def _post(url, **kw):
        seen["url"] = url
        return _Reply()
    monkeypatch.setattr(pod_http, "post", _post)
    # _ask_http returns a resume.Reply since v0.3.62 (it carries the finish
    # reason and token counts, so a cut-off page can be continued).
    reply = _engine()._ask_http([], 16, 0.0)
    assert reply.text == "testo"
    assert seen["url"] == "https://1.2.3.4:8000/v1/chat/completions"


def test_it_is_not_calling_requests_directly_any_more(monkeypatch):
    import requests
    called = []
    monkeypatch.setattr(requests, "post",
                        lambda *a, **kw: called.append(1) or _Reply())
    monkeypatch.setattr(pod_http, "post", lambda url, **kw: _Reply())
    _engine()._ask_http([], 16, 0.0)
    assert not called, "olmOCR bypassed pod_http and would fail a pod's TLS"


def test_the_per_page_base_override_is_honoured(monkeypatch):
    """Model mode passes the upstream the dispatcher chose for THIS page; a lease
    is not a fixed address, so the override is the one that must be used."""
    seen = {}
    monkeypatch.setattr(pod_http, "post",
                        lambda url, **kw: seen.update(url=url) or _Reply())
    _engine()._ask_http([], 16, 0.0, base="https://9.9.9.9:8000/v1")
    assert seen["url"].startswith("https://9.9.9.9:8000/v1")


def test_an_ordinary_host_is_untouched_by_the_pinning_rule():
    """Only https-to-bare-IP is a pod. A named server must keep normal
    verification -- this is what makes the drop-in safe everywhere else."""
    assert pod_http.is_pinned_url("https://1.2.3.4:8000/v1/chat/completions")
    assert not pod_http.is_pinned_url("https://ocr.example.com/v1/chat/completions")
    assert not pod_http.is_pinned_url("http://127.0.0.1:8000/v1/chat/completions")


def test_a_transport_failure_is_still_a_503_with_the_url_in_it(monkeypatch):
    from codai.ocr.base import OcrError

    def _boom(url, **kw):
        raise OSError("connection refused")
    monkeypatch.setattr(pod_http, "post", _boom)
    try:
        _engine()._ask_http([], 16, 0.0)
    except OcrError as e:
        assert e.status == 503
        assert "1.2.3.4" in str(e)
    else:
        raise AssertionError("an unreachable upstream must raise")
