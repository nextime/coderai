"""A page surya gave up on must SAY it was looping, and the detector must be tunable.

A VLM that loses its place repeats the last thing it saw until it runs out of
tokens. surya retries such a page at rising temperature and, when the retries
run out, returns the looping reply anyway — which parses to a few blocks or to
none. Downstream that is a short or empty page, and it reads exactly like a scan
the model could not see. They are opposite conclusions: one should be routed to
another engine, the other means the document itself is bad.

chandra has flagged this since it shipped (`repeat_detected`). surya said
nothing, so a cascade had to infer it from emptiness — which is why dense pages
"fell through" with no explanation.

The detector's thresholds are an operator setting, not a per-request one: a
corpus of tables, dotted leaders or form fields repeats short strings
legitimately, and whether that is degeneration depends on the documents. A false
positive is expensive — several full generations, and usually less text than the
first reply already had.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_surya_retry_budget import (_fake_surya, _worker,           # noqa: E402
                                     _worker_module, _Result)


def _hooked(monkeypatch, **cfg):
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    mod._REPEAT_OPTS.clear()
    w = _worker(mod, token_trace=True, **cfg)
    w._install_trace_hook()
    mod._trace_begin()
    return mod, oc, w


# --- the flag ---------------------------------------------------------------

def test_a_page_still_looping_when_surya_gives_up_is_flagged(monkeypatch):
    mod, oc, _w = _hooked(monkeypatch)
    oc.detect_repeat_token = lambda text, **kw: "loop" in (text or "")
    res = _Result(raw="loop loop loop")
    # retries exhausted: surya's own decision function says no more
    oc._should_retry(res, 3, 3)
    assert mod._state()["looping"] is True
    assert mod._state()["loop_chars"] == len("loop loop loop")


def test_a_page_that_stopped_looping_is_not_flagged(monkeypatch):
    mod, oc, _w = _hooked(monkeypatch)
    oc.detect_repeat_token = lambda text, **kw: "loop" in (text or "")
    oc._should_retry(_Result(raw="clean text"), 3, 3)
    assert "looping" not in mod._state()


def test_a_loop_that_will_be_retried_is_not_yet_a_verdict(monkeypatch):
    """A loop surya then rescues is not a property of the page."""
    mod, oc, _w = _hooked(monkeypatch)
    oc.detect_repeat_token = lambda text, **kw: True
    oc._should_retry(_Result(raw="loop"), 0, 3)      # will retry
    assert "looping" not in mod._state()


def test_the_character_count_separates_garbage_from_nothing(monkeypatch):
    """"Looped and produced 4,000 characters" has a usable beginning;
    "looped and produced nothing" does not."""
    mod, oc, _w = _hooked(monkeypatch)
    oc.detect_repeat_token = lambda text, **kw: True
    oc._should_retry(_Result(raw="x" * 4000), 3, 3)
    assert mod._state()["loop_chars"] == 4000


def test_a_new_page_does_not_inherit_the_verdict(monkeypatch):
    mod, oc, _w = _hooked(monkeypatch)
    oc.detect_repeat_token = lambda text, **kw: True
    oc._should_retry(_Result(raw="loop"), 3, 3)
    mod._trace_begin()
    assert "looping" not in mod._state() and "loop_chars" not in mod._state()


def test_the_flag_reaches_the_page_dict():
    """The worker's page dict is copied wholesale into OcrPage.meta, so this is
    what a caller actually sees — the same key chandra reports."""
    src = (ROOT / "codai/ocr/workers/ocr_worker.py").read_text()
    assert 'page["repeat_detected"] = True' in src
    assert 'page["repeat_chars"]' in src


# --- the tunable detector ---------------------------------------------------

def _detector_probe(monkeypatch, **cfg):
    """A fresh surya whose detector records the arguments it is called with.

    Set up BEFORE the hook is installed: the patch happens once per process
    (`_coderai_traced`), so a detector swapped in afterwards is never wrapped.
    """
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    mod._REPEAT_OPTS.clear()
    seen = {}
    oc.detect_repeat_token = lambda text, **kw: seen.update(kw) or False
    _worker(mod, token_trace=True, **cfg)._install_trace_hook()
    return mod, oc, seen


def test_the_thresholds_are_applied_when_set(monkeypatch):
    _mod, oc, seen = _detector_probe(monkeypatch, repeat_max_repeats=8,
                                     repeat_window=200, repeat_scaling=1.5)
    oc.detect_repeat_token("anything")
    assert seen == {"base_max_repeats": 8, "window_size": 200,
                    "scaling_factor": 1.5}


def test_unset_thresholds_leave_suryas_own_alone(monkeypatch):
    _mod, oc, seen = _detector_probe(monkeypatch)
    oc.detect_repeat_token("anything")
    # Not even a keyword: surya's signature defaults are the whole point of 0.
    assert seen == {}


def test_an_explicit_argument_still_wins(monkeypatch):
    """`_should_retry` passes cut_from_end itself; our defaults must not fight
    the caller's own arguments."""
    _mod, oc, seen = _detector_probe(monkeypatch, repeat_max_repeats=8)
    oc.detect_repeat_token("anything", base_max_repeats=2, cut_from_end=50)
    assert seen["base_max_repeats"] == 2 and seen["cut_from_end"] == 50


def test_rubbish_thresholds_are_ignored_rather_than_crashing(monkeypatch):
    mod = _worker_module()
    _fake_surya(monkeypatch, mod)
    mod._REPEAT_OPTS.clear()
    _worker(mod, token_trace=True, repeat_max_repeats="lots", repeat_window=-5)
    assert mod._REPEAT_OPTS == {}


def test_the_settings_travel_from_the_engine_to_the_worker():
    """They are sent at LOAD, like the other hook settings — the detector is
    patched before the first page, not per request."""
    src = (ROOT / "codai/ocr/surya.py").read_text()
    for key in ("repeat_max_repeats", "repeat_window", "repeat_scaling"):
        assert f'"{key}"' in src
