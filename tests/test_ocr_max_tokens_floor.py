"""An operator may decline a self-harming page budget.

A per-request `max_tokens` wins over the engine's own default and over the
operator's configured budget. That is correct -- only the caller knows whether
it is sending a four-page contract or a fifty-page judgment -- and it stays the
default here.

But it cost a real run. A client sent `max_tokens: 3072` on every page of a
corpus whose pages need just over that, so each page generated to the cap,
stopped mid-text, and paid a continuation that RE-SENT the first 3072 tokens to
fetch the last few hundred. By the second cut-off the model was often repeating
itself, so the page was refused a resume and re-run whole at a higher
temperature. Measured across 594 continuations: **532 pages reached segment 2**,
44 segment 3, 11 segment 4, 7 segment 5 -- the signature of a budget
*marginally* too small, which is the most expensive kind, because nearly every
page pays double rather than a few pages paying a lot.

Meanwhile the cards sat at `GPU KV cache usage: 1.4%`, the fleet grew to 65
pods, and the operator paying for them had no way to decline.

So `<engine>_max_tokens_floor` lets them state a minimum. Off by default,
never lowers a budget, and says so once per engine rather than once per page.
"""

import types

import pytest

from codai.config import OcrConfig
from codai.ocr import manager as om


@pytest.fixture(autouse=True)
def _quiet():
    """The rate limiter is module state and outlives a test."""
    om._SAID_AT.clear()
    yield
    om._SAID_AT.clear()


def _cfg(**kw):
    c = OcrConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ------------------------------------------------------------ off by default #

def test_the_floor_is_off_by_default():
    """An install that silently overrode a caller would be lying about an API."""
    c = OcrConfig()
    assert c.surya_max_tokens_floor == 0
    assert c.chandra_max_tokens_floor == 0


def test_with_no_floor_the_caller_is_untouched():
    out = om.floor_page_opts(_cfg(), "surya", {"max_tokens": 3072})
    assert out == {"max_tokens": 3072}


# ---------------------------------------------------------------- the floor #

def test_a_budget_below_the_floor_is_raised():
    """THE CASE: 3072 against pages that need just over it."""
    out = om.floor_page_opts(_cfg(surya_max_tokens_floor=12288),
                             "surya", {"max_tokens": 3072})
    assert out["max_tokens"] == 12288


def test_a_budget_above_the_floor_is_left_alone():
    """A caller asking for more knows something the floor does not."""
    out = om.floor_page_opts(_cfg(surya_max_tokens_floor=8192),
                             "surya", {"max_tokens": 20000})
    assert out["max_tokens"] == 20000


def test_a_budget_exactly_at_the_floor_is_untouched():
    out = om.floor_page_opts(_cfg(surya_max_tokens_floor=8192),
                             "surya", {"max_tokens": 8192})
    assert out["max_tokens"] == 8192


def test_a_caller_that_said_nothing_is_not_given_a_budget():
    """Omitted means 'use the configured value', which is not the floor's job
    to override -- the configured value may well be higher."""
    out = om.floor_page_opts(_cfg(surya_max_tokens_floor=8192), "surya", {})
    assert "max_tokens" not in out


def test_an_explicit_zero_is_not_raised():
    """0 is a meaningful setting: unlimited. Raising it would cap it."""
    out = om.floor_page_opts(_cfg(surya_max_tokens_floor=8192),
                             "surya", {"max_tokens": 0})
    assert out.get("max_tokens") == 0


def test_other_options_survive():
    out = om.floor_page_opts(_cfg(surya_max_tokens_floor=8192), "surya",
                             {"max_tokens": 100, "max_continuations": 2})
    assert out["max_continuations"] == 2


def test_the_callers_dict_is_not_mutated():
    asked = {"max_tokens": 3072}
    om.floor_page_opts(_cfg(surya_max_tokens_floor=8192), "surya", asked)
    assert asked == {"max_tokens": 3072}


def test_none_opts_are_accepted():
    assert om.floor_page_opts(_cfg(surya_max_tokens_floor=8192),
                              "surya", None) == {}


# ------------------------------------------------------------- per engine #

def test_the_floor_is_per_engine():
    """chandra's budgets are not surya's; one floor must not set the other's."""
    cfg = _cfg(surya_max_tokens_floor=12288, chandra_max_tokens_floor=0)
    assert om.floor_page_opts(cfg, "chandra",
                              {"max_tokens": 3072})["max_tokens"] == 3072
    assert om.floor_page_opts(cfg, "surya",
                              {"max_tokens": 3072})["max_tokens"] == 12288


def test_an_unknown_engine_falls_back_without_raising():
    out = om.floor_page_opts(_cfg(surya_max_tokens_floor=8192),
                             "something-else", {"max_tokens": 10})
    assert out["max_tokens"] == 10


def test_a_config_without_the_setting_is_fine():
    """An install whose config predates this."""
    out = om.floor_page_opts(types.SimpleNamespace(), "surya",
                             {"max_tokens": 10})
    assert out["max_tokens"] == 10


def test_a_nonsense_floor_is_ignored():
    out = om.floor_page_opts(_cfg(surya_max_tokens_floor="not a number"),
                             "surya", {"max_tokens": 10})
    assert out["max_tokens"] == 10


# ------------------------------------------------------------- it says so #

def test_it_says_what_it_did_and_why(capsys):
    om.floor_page_opts(_cfg(surya_max_tokens_floor=12288),
                       "surya", {"max_tokens": 3072})
    out = capsys.readouterr().out
    assert "3072" in out and "12288" in out
    assert "surya_max_tokens_floor" in out
    # The reason, not just the fact: an operator should not have to find the
    # commit to learn why a small budget is expensive.
    assert "re-sends" in out


def test_it_is_said_once_per_engine_not_once_per_page(capsys):
    """A 5,000-document run must not write this 30,000 times."""
    cfg = _cfg(surya_max_tokens_floor=12288)
    for _ in range(500):
        om.floor_page_opts(cfg, "surya", {"max_tokens": 3072})
    assert capsys.readouterr().out.count("max_tokens_floor") == 1
