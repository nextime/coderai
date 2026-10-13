"""The resumption rules, once, for every engine that transcribes with a VLM.

surya learned these the hard way and chandra and olmocr had none of them: a
dense page simply stopped at `chandra_max_tokens` / `olmocr_max_tokens` and the
half-finished markup parsed to a fragment, which in the finished text is
indistinguishable from a page the model could not read.

These tests pin the rules themselves — the engine wiring is tested separately,
against each engine's own ask path.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.ocr import resume as R                                  # noqa: E402


def _reply(text="", finish="", prompt=0, done=0):
    return R.Reply(text=text, finish_reason=finish, prompt_tokens=prompt,
                   completion_tokens=done)


def _asker(*replies):
    seen = []

    def ask(prefix, seg_cap):
        seen.append((prefix, seg_cap))
        return replies[len(seen) - 1] if len(seen) <= len(replies) else _reply("")

    ask.seen = seen
    return ask


# --- when a reply is finished ----------------------------------------------

def test_finish_reason_length_is_authoritative():
    # A token count near the cap is a guess; this is the server saying it.
    assert R.at_cap(_reply(finish="length", done=10), 4096) is True


def test_a_reply_that_stopped_on_its_own_is_finished():
    assert R.at_cap(_reply(finish="stop", done=4096), 4096) is False


def test_a_count_at_the_cap_counts_without_a_finish_reason():
    assert R.at_cap(_reply(done=4096), 4096) is True
    assert R.at_cap(_reply(done=100), 4096) is False


def test_no_cap_means_nothing_is_at_it():
    assert R.at_cap(_reply(finish="length"), 0) is False


# --- the window is the ceiling ---------------------------------------------

def test_the_segment_shrinks_as_the_prefix_grows():
    assert R.segment_cap(2048, 32768, 4000) == 2048
    assert R.segment_cap(2048, 32768, 31000) == 1768
    assert R.segment_cap(2048, 32768, 33000) == 0


def test_an_unknown_window_leaves_the_cap_alone():
    # Guessing one would either waste it or overrun it.
    assert R.segment_cap(2048, 0, 999999) == 2048


def test_a_full_window_stops_the_loop_and_says_a_smaller_cap_helps():
    ask = _asker(_reply("more", finish="length", done=100))
    res = R.resume(ask, _reply("start", finish="length", prompt=18000, done=300),
                   cap=2048, budget=4, limit=18432, label="chandra")
    assert ask.seen == []                      # never even asked
    assert res.truncated and "SMALLER chandra_max_tokens" in res.reason


def test_a_context_error_from_the_server_ends_it_cleanly():
    def ask(prefix, seg_cap):
        raise RuntimeError("This model's maximum context length is 18432 tokens")

    res = R.resume(ask, _reply("start", finish="length", done=2048),
                   cap=2048, budget=4, limit=0)
    assert res.rounds == 0 and res.truncated
    assert "refused the next segment" in res.reason


# --- the loop ---------------------------------------------------------------

def test_segments_are_joined_and_counted():
    ask = _asker(_reply("b", finish="length", done=2048),
                 _reply("c", finish="stop", done=10))
    res = R.resume(ask, _reply("a", finish="length", done=2048), cap=2048,
                   budget=4)
    assert res.text == "abc"
    assert res.rounds == 2 and res.tokens == [2048, 2048, 10]
    assert not res.truncated


def test_the_prefix_sent_back_is_everything_so_far():
    ask = _asker(_reply("b", finish="length", done=2048),
                 _reply("c", finish="stop", done=5))
    R.resume(ask, _reply("a", finish="length", done=2048), cap=2048, budget=4)
    assert [p for p, _ in ask.seen] == ["a", "ab"]


def test_the_budget_is_a_ceiling_on_attempts():
    ask = _asker(*[_reply("seg%d" % i, finish="length", done=2048)
                   for i in range(9)])
    res = R.resume(ask, _reply("a", finish="length", done=2048), cap=2048,
                   budget=3)
    assert len(ask.seen) == 3 and res.rounds == 3
    assert res.truncated and "budget (3) ran out" in res.reason


def test_zero_budget_does_not_resume_and_says_so():
    ask = _asker(_reply("b"))
    res = R.resume(ask, _reply("a", finish="length", done=2048), cap=2048,
                   budget=0, label="olmocr")
    assert ask.seen == [] and res.rounds == 0 and res.truncated
    assert "olmocr_max_continuations" in res.reason


def test_a_finished_reply_is_never_resumed():
    ask = _asker(_reply("b"))
    res = R.resume(ask, _reply("a", finish="stop", done=10), cap=2048, budget=4)
    assert ask.seen == [] and res.text == "a" and not res.truncated


# --- progress has to be real ------------------------------------------------

def test_a_reply_already_repeating_itself_is_not_resumed():
    ask = _asker(_reply("b", finish="length", done=2048))
    res = R.resume(ask, _reply("la la la", finish="length", done=2048),
                   cap=2048, budget=4, repeats=lambda t: "la la" in t)
    assert ask.seen == []
    assert "LOOPING" in res.reason and res.rounds == 0


def test_a_segment_identical_to_the_previous_one_ends_it():
    ask = _asker(_reply("a", finish="length", done=2048))
    res = R.resume(ask, _reply("a", finish="length", done=2048), cap=2048,
                   budget=4)
    assert res.rounds == 0 and "verbatim" in res.reason


def test_a_resumption_that_comes_back_looping_is_kept_but_ends_it():
    ask = _asker(_reply("la la la", finish="length", done=2048),
                 _reply("more", finish="stop", done=5))
    res = R.resume(ask, _reply("start", finish="length", done=2048), cap=2048,
                   budget=4, repeats=lambda t: "la la" in t)
    # Kept: a looping tail usually follows correct text, and discarding it would
    # throw away the part that is right.
    assert res.text == "startla la la" and res.rounds == 1
    assert len(ask.seen) == 1 and "came back repeating" in res.reason


def test_an_empty_continuation_ends_it():
    ask = _asker(_reply("", finish="stop", done=0))
    res = R.resume(ask, _reply("a", finish="length", done=2048), cap=2048,
                   budget=4)
    assert res.rounds == 0 and "empty continuation" in res.reason


# --- budgets from config and from the request ------------------------------

class _Cfg:
    chandra_max_tokens = 4096
    chandra_max_continuations = 8
    chandra_context_tokens = 0


def test_the_configured_budgets_are_read_by_prefix():
    b = R.budgets(_Cfg(), "chandra", default_cap=8192)
    assert b == {"cap": 4096, "continuations": 8, "context_tokens": 0}


def test_a_missing_setting_falls_back_to_the_default_cap():
    class _Bare:
        pass

    assert R.budgets(_Bare(), "olmocr", default_cap=4096)["cap"] == 4096


def test_the_request_overrides_the_configuration(monkeypatch):
    from codai.ocr import base
    tok = base.PAGE_OPTIONS.set({"max_tokens": 3072, "max_continuations": 4})
    try:
        b = R.budgets(_Cfg(), "chandra")
    finally:
        base.PAGE_OPTIONS.reset(tok)
    assert b["cap"] == 3072 and b["continuations"] == 4


def test_a_request_may_turn_resumption_off_for_one_page():
    from codai.ocr import base
    tok = base.PAGE_OPTIONS.set({"max_continuations": 0})
    try:
        assert R.budgets(_Cfg(), "chandra")["continuations"] == 0
    finally:
        base.PAGE_OPTIONS.reset(tok)


def test_rubbish_in_the_request_does_not_break_the_page():
    from codai.ocr import base
    tok = base.PAGE_OPTIONS.set({"max_tokens": "lots"})
    try:
        assert R.budgets(_Cfg(), "chandra")["cap"] == 4096
    finally:
        base.PAGE_OPTIONS.reset(tok)


# --- the report -------------------------------------------------------------

def test_a_page_that_fitted_reports_nothing():
    res = R.Resumed(text="a", tokens=[100])
    assert R.report(res, cap=4096, budget=8) == {}


def test_a_resumed_page_says_it_is_whole():
    res = R.Resumed(text="ab", rounds=1, tokens=[4096, 50])
    info = R.report(res, cap=4096, budget=8)
    assert info["resumed"] == 1 and info["calls"] == 2
    assert "the page is whole" in info["why"]
    assert info["max_continuations"] == 8


def test_an_incomplete_page_carries_the_real_reason():
    res = R.Resumed(text="ab", rounds=2, tokens=[4096, 4096, 4096],
                    truncated=True, reason="the budget (2) ran out")
    info = R.report(res, cap=4096, budget=2)
    assert info["truncated"] is True and info["complete"] is False
    assert "the budget (2) ran out" in info["why"]


def test_a_capped_but_unresumed_page_is_reported_as_such():
    res = R.Resumed(text="a", tokens=[4096], reason="resumption is off")
    info = R.report(res, cap=4096, budget=0)
    assert info["truncated"] is True and "resumption is off" in info["why"]


def test_the_budget_in_force_is_always_published():
    """So a caller can tell a request that landed from one that was ignored."""
    res = R.Resumed(text="a", tokens=[4096], truncated=True)
    assert R.report(res, cap=3072, budget=4)["max_continuations"] == 4


def test_retries_alone_are_reported_too():
    res = R.Resumed(text="a", tokens=[10])
    info = R.report(res, cap=4096, budget=8, retries=2, label="chandra")
    assert info["calls"] == 3 and "chandra re-ran this page" in info["why"]
