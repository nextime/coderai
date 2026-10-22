"""A progress notice must not end up inside the answer the client saves.

2026-10-22, production: with auto-compaction on, a long conversation's streamed
reply began

    🗜 Compacting context (~45406+300 tok > 120% of 32768) using 'summarize'…
    ✂ A single message exceeds the target; slicing it…
      • summarizing a large user message (~45387 tok)…

...and only then the actual answer. Compaction can take a minute, so the client
does need to hear something -- but coderai's own progress log was being written
into `delta.content`, which is the text a client stores, displays, and builds
the next turn on.

The wait keepalive had already answered this question (`wait_status_mode`:
silent | invisible | visible, per model or globally), so compaction follows the
same setting rather than inventing a second one.
"""
import pytest

from codai.api.text import _status_mode


class _Cfg:
    def __init__(self, mode=None):
        self.models = type("M", (), {"wait_status_mode": mode})()


def _globally(monkeypatch, mode):
    import codai.admin.routes as routes
    monkeypatch.setattr(routes, "config_manager",
                        type("CM", (), {"config": _Cfg(mode)})(), raising=False)


def _per_model(monkeypatch, entry):
    from codai.models.manager import multi_model_manager as mmm
    monkeypatch.setattr(mmm, "_config_for_model", lambda name: dict(entry),
                        raising=False)


def test_the_default_keeps_notices_out_of_the_content(monkeypatch):
    """THE BUG: the default has to be the safe one. A client that never
    configured anything was getting coderai's progress log in its replies."""
    _globally(monkeypatch, None)
    _per_model(monkeypatch, {})
    assert _status_mode("qwen38") == "invisible"


def test_a_model_can_ask_for_visible_notices(monkeypatch):
    """Someone watching a chat UI wants to see why it is taking a minute."""
    _globally(monkeypatch, "invisible")
    _per_model(monkeypatch, {"wait_status_mode": "visible"})
    assert _status_mode("qwen38") == "visible"


def test_the_per_model_setting_beats_the_global_one(monkeypatch):
    _globally(monkeypatch, "visible")
    _per_model(monkeypatch, {"wait_status_mode": "silent"})
    assert _status_mode("qwen38") == "silent"


def test_the_setting_is_read_from_the_raw_entry_too(monkeypatch):
    """models.json keys survive on `_raw_cfg`, not on the runtime kwargs dict --
    the same place the auto_compact flags live, and the same reason they were
    silently lost once."""
    _globally(monkeypatch, None)
    _per_model(monkeypatch, {"_raw_cfg": {"wait_status_mode": "silent"}})
    assert _status_mode("qwen38") == "silent"


def test_the_global_setting_applies_when_the_model_says_nothing(monkeypatch):
    _globally(monkeypatch, "silent")
    _per_model(monkeypatch, {})
    assert _status_mode("qwen38") == "silent"


def test_a_nonsense_mode_falls_back_to_the_safe_one(monkeypatch):
    """A typo must not be read as "write it into the content"."""
    _globally(monkeypatch, "LOUD")
    _per_model(monkeypatch, {})
    assert _status_mode("qwen38") == "invisible"
    _per_model(monkeypatch, {"wait_status_mode": "  Visible  "})
    assert _status_mode("qwen38") == "visible"


def test_an_unresolvable_model_is_not_an_error(monkeypatch):
    """This runs inside the request path; it answers, it does not raise."""
    from codai.models.manager import multi_model_manager as mmm

    def _boom(name):
        raise RuntimeError("no config loaded")
    monkeypatch.setattr(mmm, "_config_for_model", _boom, raising=False)
    _globally(monkeypatch, None)
    assert _status_mode("whatever") in ("silent", "invisible", "visible")
