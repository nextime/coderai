"""Stop a degenerate page as it happens, instead of paying for all of it.

A VLM that loses its place on a page repeats the last thing it saw until it runs
out of tokens, and the result arrives looking like a successful transcription.
Every engine here already runs `detect_repeat_token` over the FINISHED reply and
treats a looping one as retryable -- the waste is that the reply has to finish
first, so a page that starts looping on its tenth token still emits its whole
cap before anyone looks.

Measured on 100 pages of a real judicial corpus (surya, A40, cap 4096 / 1
continuation): 4 pages looped and accounted for **18.9% of the run's entire
generation time**. The worst spent 450 s producing 17,586 characters of one
repeated clause. Cutting them takes the run from 0.560 to 0.690 pages/s (+23%)
and the worst page from 450 s to 67 s.

THE REASON THIS CAN BE ON BY DEFAULT is that it is the same function with the
same thresholds that already judges the finished reply. A page aborted here is
a page the post-hoc check would have rejected anyway, reached sooner. Nothing
new decides anything.

What these tests pin, in order of what getting it wrong would cost:

  1. it never cuts a page that is not provably looping -- a truncated page
     answers 200 and gets indexed, which is this subsystem's worst failure
  2. an aborted reply must not look like one that merely ran long, or the
     resumption logic continues it, and continuing a loop is the whole problem
  3. the stream is CLOSED, or nothing is saved
  4. it degrades to the old behaviour on anything unexpected
"""
import json

import pytest

from codai.ocr import loopguard
from codai.ocr.chandra import detect_repeat_token


def _loopy(unit="the Court finds that ", times=400):
    return unit * times


def _real():
    return ("Nel giudizio di appello proposto dal ricorrente avverso la "
            "sentenza del Tribunale di Milano, la Corte osserva quanto segue. "
            "Il primo motivo di ricorso e infondato per le ragioni che "
            "seguono, atteso che la documentazione prodotta non consente di "
            "ritenere provata la circostanza dedotta. " * 12)


# --- 1. it only cuts what is provably looping ----------------------------

def test_it_cuts_a_looping_page():
    g = loopguard.LoopGuard()
    assert g.feed(_loopy()) is True
    assert g.tripped_at >= loopguard.DEFAULT_MIN_CHARS


def test_it_does_not_cut_real_text():
    g = loopguard.LoopGuard()
    assert g.feed(_real()) is False
    assert g.tripped is False


def test_nothing_is_judged_before_min_chars():
    """The largest genuine page in the measured corpus is 5,054 characters and
    the median ~1,900, so a verdict before 2,000 is a verdict on a page that
    has not yet shown its character."""
    g = loopguard.LoopGuard(min_chars=2000)
    short = "ab" * 200                       # 400 chars, maximally repetitive
    assert g.feed(short) is False
    assert loopguard.DEFAULT_MIN_CHARS >= 2000


def test_a_repetitive_passage_that_recovers_is_not_cut():
    """The measured false positive at the loosest setting: a page with a
    repetitive middle and genuine text after it. Aborting there is silent data
    loss -- the page answers 200 and gets indexed truncated."""
    g = loopguard.LoopGuard(min_chars=1200, persistence=2)
    g.feed("Totale ..... 1.200,00 EUR\n" * 60)     # a table of dotted leaders
    g.feed(_real())                                 # then real prose
    assert g.tripped is False


def test_persistence_is_what_separates_them():
    """With persistence 1 that same page IS cut -- which is why it is 2."""
    one = loopguard.LoopGuard(min_chars=1200, persistence=1)
    one.feed("Totale ..... 1.200,00 EUR\n" * 60)
    two = loopguard.LoopGuard(min_chars=1200, persistence=2)
    two.feed("Totale ..... 1.200,00 EUR\n" * 60)
    assert one.tripped is True
    assert two.tripped is False
    assert loopguard.DEFAULT_PERSISTENCE >= 2


def test_a_real_loop_survives_the_persistence_requirement():
    """It must not be possible to evade the check by looping: a real loop
    never recovers, so it fires on every subsequent check."""
    g = loopguard.LoopGuard(min_chars=2000, persistence=3)
    for _ in range(40):
        if g.feed("the Court finds that " * 30):
            break
    assert g.tripped is True


def test_the_verdict_is_the_same_function_that_judges_the_finished_reply():
    """The whole safety argument. A separate heuristic could abandon a page the
    post-hoc check would have accepted."""
    import inspect
    src = inspect.getsource(loopguard._default_detector)
    assert "detect_repeat_token" in src
    assert "from codai.ocr.chandra import detect_repeat_token" in src


def test_a_verdict_sticks_once_given():
    g = loopguard.LoopGuard()
    g.feed(_loopy())
    assert g.tripped is True
    assert g.feed("and then some perfectly ordinary prose arrived") is True


def test_a_detector_that_raises_never_cuts_a_page():
    def _boom(_t):
        raise RuntimeError("detector is confused")
    g = loopguard.LoopGuard(detector=_boom)
    assert g.feed(_loopy()) is False


def test_the_verdict_does_not_depend_on_how_the_server_chunks():
    """First written so that one check ran per CALL, which made a persistence
    above 1 unreachable for a caller that fed a whole reply at once -- and made
    every verdict a function of the server's chunk framing."""
    text = _loopy()
    whole = loopguard.LoopGuard()
    whole.feed(text)
    dribbled = loopguard.LoopGuard()
    for i in range(0, len(text), 17):
        if dribbled.feed(text[i:i + 17]):
            break
    assert whole.tripped is True
    assert dribbled.tripped is True
    # and they agree about WHERE, within one step
    assert abs(whole.tripped_at - dribbled.tripped_at) <= loopguard.DEFAULT_STEP


def test_a_clean_page_fed_whole_is_still_clean():
    whole = loopguard.LoopGuard()
    whole.feed(_real())
    assert whole.tripped is False


# --- 2. an aborted reply must not be resumed -----------------------------

def _chunks(pieces, usage=None):
    out = []
    for p in pieces:
        out.append("data: " + json.dumps(
            {"choices": [{"delta": {"content": p}}]}))
    if usage:
        out.append("data: " + json.dumps({"choices": [], "usage": usage}))
    out.append("data: [DONE]")
    return out


class _Resp:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def iter_lines(self):
        for line in self._lines:
            yield line.encode()

    def close(self):
        self.closed = True


def _post(resp):
    def _p(url, **kw):
        _p.kwargs = kw
        return resp
    return _p


def test_an_aborted_reply_is_not_marked_as_length():
    """`length` is what the resumption logic reads as "the page ran long, ask
    for the rest" -- and asking for the rest of a loop extends the loop."""
    resp = _Resp(_chunks(["the Court finds that " * 40] * 10))
    got = loopguard.stream_chat(_post(resp), "http://x/v1/chat/completions",
                                {}, {}, 30.0, loopguard.LoopGuard())
    assert got.aborted is True
    assert got.finish_reason == "abort_repeat"
    assert got.finish_reason != "length"


def test_an_aborted_reply_carries_the_text_it_did_produce():
    resp = _Resp(_chunks(["the Court finds that " * 40] * 10))
    got = loopguard.stream_chat(_post(resp), "u", {}, {}, 30.0,
                                loopguard.LoopGuard())
    assert len(got.text) >= loopguard.DEFAULT_MIN_CHARS
    assert got.reason and "abandoned" in got.reason


def test_a_clean_reply_passes_through_untouched():
    resp = _Resp(_chunks(["Nel giudizio ", "di appello ", "la Corte osserva."],
                         usage={"prompt_tokens": 1200, "completion_tokens": 42}))
    got = loopguard.stream_chat(_post(resp), "u", {}, {}, 30.0,
                                loopguard.LoopGuard())
    assert got.aborted is False
    assert got.text == "Nel giudizio di appello la Corte osserva."
    assert got.completion_tokens == 42
    assert got.prompt_tokens == 1200


def test_usage_is_requested_or_every_page_reads_as_complete():
    """Without include_usage a streamed reply carries no token counts, and
    completeness is decided by comparing completion_tokens against the cap --
    so a missing count silently reclassifies every page as whole."""
    resp = _Resp(_chunks(["hello"]))
    p = _post(resp)
    loopguard.stream_chat(p, "u", {"max_tokens": 10}, {}, 30.0, None)
    sent = p.kwargs["json"]
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}


# --- 3. the stream is closed, or nothing is saved ------------------------

def test_the_stream_is_closed_on_abort():
    """Closing it is what cancels the sequence on the server. Reading to the
    end and discarding the tail would save nothing at all."""
    resp = _Resp(_chunks(["the Court finds that " * 40] * 10))
    loopguard.stream_chat(_post(resp), "u", {}, {}, 30.0, loopguard.LoopGuard())
    assert resp.closed is True


def test_the_stream_is_closed_on_a_clean_reply_too():
    resp = _Resp(_chunks(["short"]))
    loopguard.stream_chat(_post(resp), "u", {}, {}, 30.0, loopguard.LoopGuard())
    assert resp.closed is True


def test_it_stops_reading_at_the_abort():
    """If it kept consuming, the tokens would still be generated."""
    seen = []

    class _Counting(_Resp):
        def iter_lines(self):
            for line in self._lines:
                seen.append(1)
                yield line.encode()

    resp = _Counting(_chunks(["the Court finds that " * 40] * 50))
    loopguard.stream_chat(_post(resp), "u", {}, {}, 30.0, loopguard.LoopGuard())
    assert len(seen) < 50, "read the whole stream instead of abandoning it"


def test_a_5xx_is_a_real_error_and_is_not_retried():
    """The server saying IT failed, not that the request was wrong. Falling
    back would ask a broken server the same question twice and report the
    second answer, hiding the first."""
    class _Bad(_Resp):
        status_code = 503
        text = "no model"
    from codai.ocr.base import OcrError
    with pytest.raises(OcrError):
        loopguard.stream_chat(_post(_Bad([])), "u", {}, {}, 30.0, None)


def test_a_4xx_means_this_server_cannot_stream():
    """Which is what a server without `stream_options` says. Nothing was
    generated, so the blocking path may answer it."""
    class _Bad(_Resp):
        status_code = 400
        text = "unknown field stream_options"
    with pytest.raises(loopguard.StreamUnsupported):
        loopguard.stream_chat(_post(_Bad([])), "u", {}, {}, 30.0, None)


def test_a_failure_after_text_has_arrived_is_not_retryable():
    """Once the server has begun answering, the generation HAS happened and is
    being paid for. Re-issuing it would run the page twice -- two generations
    billed, and on an engine with a resumption queue the second reply mistaken
    for the continuation of the first."""
    class _Broken(_Resp):
        def iter_lines(self):
            yield ("data: " + json.dumps(
                {"choices": [{"delta": {"content": "real text"}}]})).encode()
            raise OSError("connection reset mid-stream")

    with pytest.raises(OSError):
        loopguard.stream_chat(_post(_Broken([])), "u", {}, {}, 30.0,
                              loopguard.LoopGuard())


def test_a_server_that_cannot_stream_is_only_asked_once():
    loopguard._NO_STREAM.clear()
    assert loopguard.stream_supported("https://10.0.0.9:2222/v1/chat/completions")
    loopguard.mark_no_stream("https://10.0.0.9:2222/v1/chat/completions")
    assert not loopguard.stream_supported("https://10.0.0.9:2222/v1/other")
    # and it is per SERVER, not per URL or per pool
    assert loopguard.stream_supported("https://10.0.0.8:2222/v1/chat/completions")
    loopguard._NO_STREAM.clear()


# --- 4. it degrades, it does not fail ------------------------------------

def test_the_guard_can_be_switched_off():
    class _Cfg:
        early_loop_abort = False
    assert loopguard.settings(_Cfg())["enabled"] is False


def test_a_client_wrapper_is_not_even_built_when_it_is_off():
    class _Cfg:
        early_loop_abort = False
    sentinel = object()
    assert loopguard.guarded_client(sentinel, _Cfg()) is sentinel


def test_settings_survive_a_config_that_says_nonsense():
    class _Cfg:
        early_loop_abort = True
        early_loop_min_chars = 0
        early_loop_step = -5
        early_loop_persistence = None
    s = loopguard.settings(_Cfg())
    assert s["min_chars"] == loopguard.DEFAULT_MIN_CHARS
    assert s["step"] == loopguard.DEFAULT_STEP
    assert s["persistence"] == loopguard.DEFAULT_PERSISTENCE


# --- the engines are actually wired to it --------------------------------

@pytest.mark.parametrize("path,label", [
    ("codai/ocr/chandra.py", "chandra"),
    ("codai/ocr/olmocr.py", "olmocr"),
])
def test_the_engines_that_own_their_http_call_use_the_guard(path, label):
    src = open(path).read()
    assert "loopguard.stream_chat(" in src, label
    assert "loopguard.settings(self.cfg)" in src, label
    # and never on a continuation, where the prefix is text we already accepted
    assert 'guarded["enabled"] and not prefix' in src, label


def test_surya_wraps_the_client_it_cannot_patch():
    """surya sends its own request, so there is no payload of ours to add
    `stream` to -- but the client arrives as a keyword to the hook we own."""
    src = open("codai/ocr/workers/ocr_worker.py").read()
    assert "_guarded_client(kwargs[\"client\"], _drt)" in src
    assert "from surya.inference.util import detect_repeat_token as _drt" in src


def test_the_surya_worker_uses_suryas_own_detector():
    """Not ours: the function that decides the abort must be the same one that
    decides surya's retry, or a page could be abandoned and then accepted."""
    src = open("codai/ocr/workers/ocr_worker.py").read()
    assert "surya.inference.util import detect_repeat_token" in src
    assert "from codai.ocr" not in src, "the worker must not import codai.ocr"


def test_the_surya_hook_does_not_depend_on_an_unrelated_setting():
    """It rides the same hook as accent repair. Leaving it out of the install
    condition made a 19% throughput feature depend on `repair_accents`."""
    src = open("codai/ocr/workers/ocr_worker.py").read()
    assert "if not (self._trace or self._repair or _EARLY_LOOP[0]):" in src


def test_the_settings_reach_the_worker():
    src = open("codai/ocr/surya.py").read()
    for k in ("early_loop_abort", "early_loop_min_chars", "early_loop_step",
              "early_loop_persistence"):
        assert f'"{k}"' in src, k


def test_the_config_carries_them_with_an_env_override():
    src = open("codai/config.py").read()
    assert "early_loop_abort: bool = True" in src
    assert "CODERAI_OCR_EARLY_LOOP_ABORT" in src
    assert "CODERAI_OCR_EARLY_LOOP_PERSISTENCE" in src
