"""A decision is the model's renormalised probability over the answer letters.

The decision endpoint asks for one token with logprobs and reads the letters'
first-token probabilities back. The parsing has three jobs that are each a way
to be quietly wrong:

* read the FIRST generated position's candidates, not a later one;
* match a letter even when the tokenizer gave it a leading space;
* renormalise over the offered letters, and refuse (not zero-divide) when the
  model put none of its first-token mass on any of them.
"""

import math

import pytest

from codai.api.spark_decisions import (
    distribution_from_logprobs, summarize, _match_logprob,
)


def _lp(top):
    """An OpenAI chat-logprobs object whose first token offers `top`
    ({token: logprob})."""
    return {"content": [{"token": "x", "logprob": -0.1,
                         "top_logprobs": [{"token": t, "logprob": l}
                                          for t, l in top.items()]}]}


def test_probabilities_are_renormalised_over_the_letters():
    # exp(0)=1 for A, exp(ln(1/3))=1/3 for B -> 0.75 / 0.25 after renorm.
    lp = _lp({"A": 0.0, "B": math.log(1 / 3), "C": -50.0})
    dist = distribution_from_logprobs(lp, ["A", "B"])
    assert dist["A"] == pytest.approx(0.75, abs=1e-6)
    assert dist["B"] == pytest.approx(0.25, abs=1e-6)
    assert set(dist) == {"A", "B"}          # C was not asked for


def test_a_leading_space_token_still_matches_its_letter():
    lp = _lp({" A": math.log(0.8), " B": math.log(0.2)})
    dist = distribution_from_logprobs(lp, ["A", "B"])
    assert dist["A"] == pytest.approx(0.8, abs=1e-6)


def test_an_exact_token_wins_over_a_trimmed_one():
    cands = [{"token": " A", "logprob": math.log(0.2)},
             {"token": "A", "logprob": math.log(0.9)}]
    assert _match_logprob(cands, "A") == pytest.approx(math.log(0.9))


def test_a_letter_with_no_candidate_gets_zero_but_others_stand():
    lp = _lp({"A": math.log(0.6)})
    dist = distribution_from_logprobs(lp, ["A", "B"])
    assert dist["A"] == pytest.approx(1.0)      # only A found -> all mass
    assert dist["B"] == 0.0


def test_no_letter_found_is_an_error_not_an_empty_distribution():
    lp = _lp({"Z": 0.0, "Q": -1.0})
    with pytest.raises(ValueError):
        distribution_from_logprobs(lp, ["A", "B"])


def test_absent_logprobs_is_a_clear_error():
    with pytest.raises(ValueError):
        distribution_from_logprobs(None, ["A"])
    with pytest.raises(ValueError):
        distribution_from_logprobs({"content": []}, ["A"])


def test_summarize_returns_argmax_and_its_probability():
    argmax, conf = summarize({"A": 0.7, "B": 0.3})
    assert argmax == "A" and conf == pytest.approx(0.7)
    assert summarize({}) == (None, 0.0)


def test_the_endpoint_and_request_model_are_wired():
    """A light wiring check: the route exists and the request model imports."""
    from codai.pydantic.textrequest import DecisionRequest
    req = DecisionRequest(model="m", messages=[{"role": "user", "content": "q"}],
                          letters=["A", "B"])
    assert req.letters == ["A", "B"]
    from codai.api.text import router
    paths = {getattr(r, "path", None) for r in router.routes}
    assert "/v1/decision" in paths
