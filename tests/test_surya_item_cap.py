"""The configured max_tokens must beat surya's own per-item budget.

Reported from the Digesta side as "the pod ignores max_tokens and uses a fixed
12288 cap, loops, and silently drops dense pages", with the pod blamed. The pod
has nothing to do with it. 12288 appears nowhere in coderai: it is
`settings.SURYA_MAX_TOKENS_FULL_PAGE` in the surya library running on the
orchestrator, and `surya/recognition/__init__.py` builds every full-page item
with `max_tokens=` that value. surya's own client then resolves the budget as

    max_tokens = item.max_tokens or max_tokens_default

so `item.max_tokens` was always truthy and the `max_tokens_default` our hook
rewrote was never read. Every configured and per-request cap was discarded.

On a 32k window a 12288 budget leaves room for about two segments once the
image prefill is counted, so a dense page reached the ceiling mid-answer — which
is the "loops and drops pages" part of the same report.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "tests"))

from test_surya_retry_budget import (_fake_surya, _worker,           # noqa: E402
                                     _worker_module, _Result)


class _Item:
    """A surya request item, which carries its own token budget."""

    def __init__(self, max_tokens=12288):
        self.max_tokens = max_tokens


def _hooked(monkeypatch, cap, item, kw=None):
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    mod._CAP_NOTED[0] = False
    seen = {}

    def _gen(*a, **k):
        seen["args"], seen["kwargs"] = a, dict(k)
        return _Result()

    oc._generate_one = _gen
    w = _worker(mod, token_trace=True, max_tokens=cap)
    w._install_trace_hook()
    oc._generate_one(item, **(kw or {}))
    return seen


def test_the_configured_cap_replaces_suryas_12288(monkeypatch):
    item = _Item(12288)
    _hooked(monkeypatch, 3072, item, {"max_tokens_default": 2048})
    assert item.max_tokens == 3072


def test_it_wins_even_without_max_tokens_default_in_the_call(monkeypatch):
    # The guard on the default was `"max_tokens_default" in kwargs`; the item
    # must be capped whether or not that keyword is there.
    item = _Item(12288)
    _hooked(monkeypatch, 3072, item)
    assert item.max_tokens == 3072


def test_a_cap_larger_than_suryas_own_also_applies(monkeypatch):
    # Raising it is the documented way to fit a denser page in fewer segments.
    item = _Item(12288)
    _hooked(monkeypatch, 16384, item, {"max_tokens_default": 2048})
    assert item.max_tokens == 16384


def test_no_cap_leaves_suryas_own_budget_untouched(monkeypatch):
    """Opt-in: surya's per-page estimate is a reasonable default and changing
    it costs tokens on every page."""
    item = _Item(770)
    _hooked(monkeypatch, 0, item, {"max_tokens_default": 2048})
    assert item.max_tokens == 770


def test_the_per_page_override_reaches_the_item(monkeypatch):
    """A /v1/ocr request's own max_tokens, not just the configured one."""
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    mod._CAP_NOTED[0] = False
    oc._generate_one = lambda *a, **k: _Result()
    w = _worker(mod, token_trace=True, max_tokens=0)
    w._install_trace_hook()
    item = _Item(12288)
    # Per-page overrides arrive with the page, exactly as the worker installs
    # them before reading it.
    mod._trace_begin()
    mod._trace_opts({"max_tokens": 4096})
    oc._generate_one(item, max_tokens_default=2048)
    assert item.max_tokens == 4096


def test_the_default_is_still_rewritten_for_an_item_without_one(monkeypatch):
    # An item whose own budget is 0 falls through to the default in surya's
    # `or`, so that path has to keep working.
    item = _Item(0)
    seen = _hooked(monkeypatch, 3072, item, {"max_tokens_default": 2048})
    assert seen["kwargs"]["max_tokens_default"] == 3072
    assert item.max_tokens == 3072


def test_the_override_is_announced_once_not_once_per_page(monkeypatch, capsys):
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    mod._CAP_NOTED[0] = False
    oc._generate_one = lambda *a, **k: _Result()
    w = _worker(mod, token_trace=True, max_tokens=3072)
    w._install_trace_hook()
    for _ in range(3):
        oc._generate_one(_Item(12288), max_tokens_default=2048)
    err = capsys.readouterr().err
    # Eight lanes over hundreds of pages would bury the log in a line whose
    # numbers never change.
    assert err.count("page budget 3072 tokens") == 1
    assert "overriding surya's own 12288" in err


def test_an_unsettable_item_says_so_instead_of_pretending(monkeypatch, capsys):
    class _Frozen:
        max_tokens = 12288

        def __setattr__(self, k, v):
            raise AttributeError("frozen")

    _hooked(monkeypatch, 3072, _Frozen(), {"max_tokens_default": 2048})
    err = capsys.readouterr().err
    assert "could not apply the 3072-token cap" in err and "12288 stands" in err


def test_a_missing_item_is_not_an_error(monkeypatch):
    # _generate_one is also called with no positional item in some paths.
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    mod._CAP_NOTED[0] = False
    oc._generate_one = lambda *a, **k: _Result()
    w = _worker(mod, token_trace=True, max_tokens=3072)
    w._install_trace_hook()
    oc._generate_one(max_tokens_default=2048)      # must not raise
