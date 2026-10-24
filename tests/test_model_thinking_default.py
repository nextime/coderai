"""A model can be configured not to think, without every client saying so.

A thinking model generates its thought FIRST and charges it against max_tokens.
On extraction and retrieval work that thought is most of the latency and none of
the answer, and on a tight budget it is worse than slow: the allowance runs out
inside the thought and the answer comes back empty, with the request reporting
that it merely stopped. Measured on a real one -- a 2,100-token budget, ~6,000
characters of thought, no visible answer at all.

Until now the only way to turn it off was `enable_thinking: false` on every
request, which an install does not control: its clients do. So the model itself
carries a default.

Three states, not two. "auto" has to mean "leave the model's own behaviour" and
has to be what an unconfigured model gets, because for a reasoning model the
thought is the answer's quality -- an unset field must never be read as "off".
An explicit request value always wins over the configured default, in both
directions: that is what makes the setting a default rather than a policy.
"""
import pytest


class _Req:
    """Only what the code under test reads off a request."""

    def __init__(self, model="m", enable_thinking=None, extra=None):
        self.model = model
        self.enable_thinking = enable_thinking
        self.model_extra = extra or {}


@pytest.fixture
def configured(monkeypatch):
    """Point the per-model lookup at a config this test chooses."""
    def _set(cfg):
        from codai.models import manager as mgr
        monkeypatch.setattr(mgr.multi_model_manager, "_config_for_model",
                            lambda name: dict(cfg), raising=False)
    return _set


def _default(name="m"):
    from codai.api.text import _model_thinking_default
    return _model_thinking_default(name)


# -- resolving the configured default ------------------------------------

def test_an_unconfigured_model_keeps_its_own_behaviour(configured):
    configured({})
    assert _default() is None, "unset must not be read as 'do not think'"


def test_auto_keeps_the_models_own_behaviour(configured):
    configured({"thinking": "auto"})
    assert _default() is None


@pytest.mark.parametrize("value", ["off", "false", "no", "never", "0", False])
def test_off_asks_the_model_not_to_think(configured, value):
    configured({"thinking": value})
    assert _default() is False


@pytest.mark.parametrize("value", ["on", "true", "yes", "always", "1", True])
def test_on_asks_the_model_to_think(configured, value):
    configured({"thinking": value})
    assert _default() is True


def test_an_unknown_value_is_not_guessed(configured):
    configured({"thinking": "sometimes"})
    assert _default() is None


def test_no_model_name_consults_nothing(configured):
    configured({"thinking": "off"})
    assert _default("") is None


def test_a_broken_config_lookup_leaves_the_model_alone(monkeypatch):
    from codai.models import manager as mgr

    def _boom(name):
        raise RuntimeError("models.json is being rewritten")

    monkeypatch.setattr(mgr.multi_model_manager, "_config_for_model", _boom,
                        raising=False)
    assert _default() is None


# -- what reaches the engine ---------------------------------------------

def _kwargs(request, want_think):
    from codai.api.text import _chat_passthrough
    return (_chat_passthrough(request, want_think) or {}).get("chat_template_kwargs")


def test_the_default_travels_as_a_template_argument():
    """Only the chat template can decide whether the model thinks."""
    assert _kwargs(_Req(), False) == {"enable_thinking": False}


def test_thinking_on_travels_too():
    assert _kwargs(_Req(), True) == {"enable_thinking": True}


def test_nothing_configured_sends_nothing():
    assert _kwargs(_Req(), None) is None


def test_an_explicit_request_value_wins_over_the_default():
    """A client asking to think must get it, whatever the model is set to."""
    assert _kwargs(_Req(enable_thinking=True), True) == {"enable_thinking": True}
    assert _kwargs(_Req(enable_thinking=False), False) == {"enable_thinking": False}


def test_a_callers_own_template_kwargs_are_not_overwritten():
    req = _Req(extra={"chat_template_kwargs": {"enable_thinking": True}})
    assert _kwargs(req, False) == {"enable_thinking": True}


def test_enable_thinking_is_not_forwarded_as_a_bare_field():
    """vLLM takes it as a template argument; as a top-level field it is a 400."""
    from codai.api.text import _chat_passthrough
    out = _chat_passthrough(_Req(enable_thinking=False), False) or {}
    assert "enable_thinking" not in out
