"""The per-page budgets have to survive surya's own thread pool.

surya's `chat_completions_batch` runs every item through a ThreadPoolExecutor:

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_process, batch))

so the hook around `_generate_one` executes on an EXECUTOR thread, while
`_trace_opts` installed this page's options on the lane thread that submitted
it. Thread-local state cannot cross that, and nothing said so:

  * `max_tokens=3072` from a request was read as absent, and every page ran at
    surya's own `item.max_tokens` — 12288 for a full page, or an
    `image_token_budget` estimate like 1360. Reported from the Digesta side as
    "the cap is still wrong: 12288 and 1360, never the 3072 I send";
  * `max_continuations` and `context_tokens` went the same way;
  * the per-page trace — attempts, token counts, logprobs, accent repairs —
    was written on the executor thread and never read, so `model_attempts`
    could not report what it was built to report.

The item is the one object that crosses the boundary, so it is the key: the lane
registers its page state against the batch's items, and the executor thread
adopts it.
"""
import pathlib
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_surya_retry_budget import (_fake_surya, _worker,           # noqa: E402
                                     _worker_module, _Result)


class _Item:
    def __init__(self, max_tokens=12288):
        self.max_tokens = max_tokens


def _surya_with_batch(monkeypatch, mod):
    """A fake surya whose batch entry point dispatches like the real one."""
    oc = _fake_surya(monkeypatch, mod)

    def _batch(batch, **kw):
        out = []

        def _run(it):
            out.append(oc._generate_one(it, max_tokens_default=2048))

        for it in batch:
            t = threading.Thread(target=_run, args=(it,))
            t.start()
            t.join()
        return out

    oc.chat_completions_batch = _batch
    return oc


def test_the_cap_crosses_into_the_executor_thread(monkeypatch):
    mod = _worker_module()
    oc = _surya_with_batch(monkeypatch, mod)
    oc._generate_one = lambda *a, **k: _Result()
    w = _worker(mod, token_trace=True, max_tokens=0)
    w._install_trace_hook()

    mod._trace_begin()
    mod._trace_opts({"max_tokens": 3072})
    item = _Item(12288)
    oc.chat_completions_batch([item])
    # Previously 12288: the lane's options were invisible to the thread that ran.
    assert item.max_tokens == 3072


def test_the_trace_written_on_another_thread_is_read_by_the_lane(monkeypatch):
    mod = _worker_module()
    oc = _surya_with_batch(monkeypatch, mod)
    w = _worker(mod, token_trace=True, max_tokens=0)
    w._install_trace_hook()

    mod._trace_begin()
    mod._trace_opts({})
    oc.chat_completions_batch([_Item(0), _Item(0)])
    # Two attempts, counted on executor threads, visible here.
    assert mod._attempts() == 2
    assert len(mod._attempt_token_counts()) == 2


def test_two_lanes_do_not_share_a_page_state(monkeypatch):
    """The reason this was thread-local in the first place: one page's logprobs
    must never be attributed to another page."""
    mod = _worker_module()
    oc = _surya_with_batch(monkeypatch, mod)
    oc._generate_one = lambda *a, **k: _Result()
    w = _worker(mod, token_trace=True, max_tokens=0)
    w._install_trace_hook()

    seen = {}

    def lane(name, cap, item):
        mod._trace_begin()
        mod._trace_opts({"max_tokens": cap})
        oc.chat_completions_batch([item])
        seen[name] = (item.max_tokens, mod._attempts())

    a_item, b_item = _Item(12288), _Item(12288)
    ta = threading.Thread(target=lane, args=("a", 3072, a_item))
    tb = threading.Thread(target=lane, args=("b", 777, b_item))
    ta.start(); tb.start(); ta.join(); tb.join()
    assert seen["a"] == (3072, 1)
    assert seen["b"] == (777, 1)


def test_the_binding_is_released_so_items_are_not_kept_alive(monkeypatch):
    mod = _worker_module()
    oc = _surya_with_batch(monkeypatch, mod)
    oc._generate_one = lambda *a, **k: _Result()
    w = _worker(mod, token_trace=True, max_tokens=0)
    w._install_trace_hook()
    mod._trace_begin()
    mod._trace_opts({"max_tokens": 3072})
    oc.chat_completions_batch([_Item(0)])
    # A registry that grew per page would leak for the life of the worker, and
    # id() is reused — a stale entry would hand a new item another page's state.
    assert mod._ITEM_STATE == {}


def test_the_binding_is_released_even_when_the_batch_raises(monkeypatch):
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)

    def _boom(batch, **kw):
        raise RuntimeError("upstream died")

    oc.chat_completions_batch = _boom
    w = _worker(mod, token_trace=True, max_tokens=3072)
    w._install_trace_hook()
    mod._trace_begin()
    try:
        oc.chat_completions_batch([_Item(0)])
    except RuntimeError:
        pass
    assert mod._ITEM_STATE == {}


def test_an_unregistered_item_falls_back_to_this_threads_state(monkeypatch):
    """Not every path goes through the batch entry point; the direct call must
    keep working."""
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    oc._generate_one = lambda *a, **k: _Result()
    w = _worker(mod, token_trace=True, max_tokens=0)
    w._install_trace_hook()
    mod._trace_begin()
    mod._trace_opts({"max_tokens": 1234})
    item = _Item(12288)
    oc._generate_one(item, max_tokens_default=2048)
    assert item.max_tokens == 1234


def test_the_batch_hook_is_reported_when_surya_has_no_such_entry_point(monkeypatch, capsys):
    """Silence would mean the budgets are quietly unreachable on that surya."""
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    if hasattr(oc, "chat_completions_batch"):
        del oc.chat_completions_batch
    w = _worker(mod, token_trace=True, max_tokens=3072)
    w._install_trace_hook()
    assert "no chat_completions_batch to hook" in capsys.readouterr().err


# --- the report must name the real reason, not assert a config value --------

def test_a_looping_page_is_not_reported_as_resumption_being_off():
    """The message was a hardcoded string in a branch reached four ways.

    A caller read "resumption is off (ocr.surya_max_continuations = 0)" exactly
    as written and went looking for a setting that was never set; the real cause
    was a generation that had already started repeating itself, with a budget of
    4 in force. A diagnostic that names the wrong cause is worse than silence,
    because it is acted on.
    """
    mod = _worker_module()
    mod._trace_begin()
    mod._state()["budget"] = 4
    mod._no_resume("the generation was already repeating itself")
    assert mod._state()["no_resume"] == "the generation was already repeating itself"
    src = (ROOT / "codai/ocr/workers/ocr_worker.py").read_text()
    # The old text claimed a value it never read. It must not come back.
    assert "off (ocr.surya_max_continuations = 0)" not in src
    assert 'why = _state().get("no_resume")' in src


def test_the_budget_in_force_is_published():
    """So a caller can tell a request that landed from one that was ignored."""
    src = (ROOT / "codai/ocr/workers/ocr_worker.py").read_text()
    assert 'info["max_continuations"] = int(_budget)' in src


def test_a_new_page_does_not_inherit_the_previous_reason():
    mod = _worker_module()
    mod._trace_begin()
    mod._no_resume("looping")
    mod._trace_begin()
    assert "no_resume" not in mod._state()
    assert "budget" not in mod._state()


def test_zero_budget_still_says_resumption_is_off(monkeypatch):
    """That message was right in ONE of the four cases; it must survive there."""
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    def _at_the_cap(*a, **k):
        r = _Result()
        r.token_count = 3072        # the reply filled its budget
        return r

    oc._generate_one = _at_the_cap
    w = _worker(mod, token_trace=True, max_tokens=3072, max_continuations=0)
    w._install_trace_hook()
    mod._trace_begin()
    mod._trace_opts({"max_tokens": 3072, "max_continuations": 0})
    oc._generate_one(_Item(3072), max_tokens_default=3072)
    assert "max_continuations = 0" in (mod._state().get("no_resume") or "")
