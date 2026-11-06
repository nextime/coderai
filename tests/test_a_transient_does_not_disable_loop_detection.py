"""A blip while starting a stream must not turn early loop abort off for good.

`StreamUnsupported` answers one question: may this request be re-issued as an
ordinary blocking call? It is raised when nothing was consumed -- which
includes a connection reset, a read timeout, and a pod restarting mid-handshake.

Both OCR engines were reading it as the answer to a DIFFERENT question: does
this server support streaming? On any such failure they called
`mark_no_stream(url)`, which is process-lifetime and per server. So one
transient early in a run disabled early loop detection for every page after it,
and every looping page then ran to its full token cap (12,384 on the install
where this was found) instead of being cancelled after a few hundred
characters. The only notice was a single log line, printed once, that scrolled
away -- and the symptom that remained was a KV cache overflowing with runaway
pages, which reads as "too many requests in flight".

So the two questions are now separate, and these tests pin the separation:

  1. a transient carries `capability=False` and is NOT remembered
  2. a real refusal carries `capability=True` and IS
  3. either way the page itself still gets read -- degrading must not drop work
"""
import json

import pytest

from codai.ocr import loopguard


class _Resp:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code
        self.closed = False
        self.text = "no"

    def iter_lines(self):
        for line in self._lines:
            yield line.encode()

    def json(self):
        raise ValueError("not json")

    def close(self):
        self.closed = True


def _post(resp_or_exc):
    def _p(url, **kw):
        if isinstance(resp_or_exc, BaseException):
            raise resp_or_exc
        return resp_or_exc
    return _p


# --- 1. a transient says nothing about the server ------------------------

@pytest.mark.parametrize("exc", [
    OSError("connection reset by peer"),
    TimeoutError("read timed out"),
    ConnectionRefusedError("pod restarting"),
])
def test_a_connection_failure_is_not_a_capability_verdict(exc):
    with pytest.raises(loopguard.StreamUnsupported) as got:
        loopguard.stream_chat(_post(exc), "https://10.0.0.9:2222/v1/chat/completions",
                              {}, {}, 30.0, loopguard.LoopGuard())
    assert got.value.capability is False, (
        "a connection failure must not be remembered as 'this server cannot "
        "stream' -- that is what disabled loop detection for a whole run")


def test_a_transient_leaves_streaming_enabled_for_the_next_page():
    url = "https://10.0.0.7:2222/v1/chat/completions"
    loopguard._NO_STREAM.clear()
    try:
        with pytest.raises(loopguard.StreamUnsupported) as got:
            loopguard.stream_chat(_post(OSError("reset")), url, {}, {}, 30.0,
                                  loopguard.LoopGuard())
        # The caller only remembers a capability verdict, so this one is not
        # recorded and the next page is still streamed.
        if getattr(got.value, "capability", False):
            loopguard.mark_no_stream(url)
        assert loopguard.stream_supported(url) is True
    finally:
        loopguard._NO_STREAM.clear()


# --- 2. a refusal does say something about the server --------------------

def test_a_4xx_to_a_streaming_request_is_a_capability_verdict():
    """A server without `stream_options` answers 400. Nothing was generated, so
    it is safe to re-ask without streaming -- and worth remembering."""
    with pytest.raises(loopguard.StreamUnsupported) as got:
        loopguard.stream_chat(_post(_Resp([], status_code=400)), "u", {}, {},
                              30.0, loopguard.LoopGuard())
    assert got.value.capability is True


def test_a_body_that_is_not_a_stream_and_not_json_is_a_capability_verdict():
    with pytest.raises(loopguard.StreamUnsupported) as got:
        loopguard.stream_chat(_post(_Resp([""])), "u", {}, {}, 30.0,
                              loopguard.LoopGuard())
    assert got.value.capability is True


def test_a_capability_verdict_is_remembered_per_server():
    url = "https://10.0.0.6:2222/v1/chat/completions"
    loopguard._NO_STREAM.clear()
    try:
        with pytest.raises(loopguard.StreamUnsupported) as got:
            loopguard.stream_chat(_post(_Resp([], status_code=400)), url, {},
                                  {}, 30.0, loopguard.LoopGuard())
        assert got.value.capability is True
        loopguard.mark_no_stream(url)
        assert loopguard.stream_supported(url) is False
        assert loopguard.stream_supported("https://10.0.0.5:2222/v1/x") is True
    finally:
        loopguard._NO_STREAM.clear()


# --- 3. the exception still means what it meant --------------------------

def test_the_default_is_not_a_capability_verdict():
    """Anything raising it without saying so is a transient. Defaulting the
    other way would re-introduce the bug for every raise site added later."""
    assert loopguard.StreamUnsupported("whatever").capability is False


def test_a_server_that_ignores_stream_is_read_whole_and_not_penalised():
    """It generated the page and we are paying for it, so it is parsed rather
    than discarded -- and that is not a reason to stop streaming either."""
    class _Whole(_Resp):
        def json(self):
            return {"choices": [{"message": {"content": "una pagina"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3}}

    got = loopguard.stream_chat(_post(_Whole([""])), "u", {}, {}, 30.0,
                                loopguard.LoopGuard())
    assert got.text == "una pagina"
    assert got.aborted is False


# --- 4. the engines apply the distinction --------------------------------

@pytest.mark.parametrize("module,attr", [
    ("codai.ocr.chandra", "chandra"),
    ("codai.ocr.olmocr", "olmocr"),
])
def test_both_engines_only_remember_a_capability_verdict(module, attr):
    """Pinned as source, because the bug was a missing condition and a test
    that drives a whole page through either engine needs a model to do it."""
    import importlib, inspect
    src = inspect.getsource(importlib.import_module(module))
    i = src.find("except loopguard.StreamUnsupported")
    assert i > 0, f"{module} no longer handles StreamUnsupported"
    block = src[i:i + 1600]
    mark = block.find("mark_no_stream")
    guard = block.find('getattr(exc, "capability"')
    assert guard > 0, (f"{module} must check the capability flag before "
                       f"remembering a no-stream verdict")
    assert guard < mark, (f"{module} calls mark_no_stream before testing the "
                          f"capability flag, so a transient is remembered")
