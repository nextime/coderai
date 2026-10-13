"""The tokens a batch spends re-reading pages it has already failed to read.

Measured on the Digesta orchestrator, 2026-10-13 11:20-11:30, from the pod's own
drained log. Ten minutes of batch OCR at `lavoratori=4`:

    Running: 15-19 reqs, Waiting: 0 reqs, GPU KV cache usage: 5-7%
    Avg generation throughput: ~1,900-2,100 tokens/s, sustained
    524 x "POST /v1/chat/completions HTTP/1.1" 200 OK, no aborts
    2,644 generated tokens PER REQUEST (pod 2: 4,010)
    MM cache hit rate: 25-27%

Three things follow, and only the third is a defect:

1. The GPU was never the bottleneck. It held 5-7% of its KV cache and queued
   nothing for ten minutes — it could have taken several times the load. So
   "give Surya an A40 to itself" cannot be the fix: the card was ~95% idle on
   the dimension that limits a VLM, and the co-loaded pyannote cost nothing.
2. The pipeline never stalled. ~100,000 generated tokens per minute, every
   minute, with ~18 pages in flight.
3. It spent 2,644 generated tokens on pages whose text is ~400, and a quarter of
   its requests were images the pod had already read. That is surya's own
   `_should_retry`: a generation that repeats itself is re-run up to three more
   times at a rising temperature, each run able to reach max_tokens. On a page
   the model genuinely cannot read, that is four full generations to arrive at
   the same unreadable answer — and nothing anywhere said so.

`max_retries` is not reachable from surya's settings: its vLLM backend calls
`chat_completions_batch` without it, so the default 3 was the only value
available. Hooking the decision — in the same module, beside the token-trace
hook that is already there — is the only seam.
"""

import pathlib
import sys
import threading
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util                                                # noqa: E402

SRC = ROOT / "codai/ocr/workers/ocr_worker.py"


def _worker_module():
    """The real worker module, loaded by path (it is not importable as a package)."""
    spec = importlib.util.spec_from_file_location("ocr_worker_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Result:
    def __init__(self, raw="ciao", logprobs=None):
        self.raw = raw
        self.token_count = 4
        self.logprobs = logprobs
        self.error = False


def _fake_surya(monkeypatch, mod, retry_decision=True):
    """A stand-in for surya.inference.backends.openai_client."""
    oc = types.ModuleType("surya.inference.backends.openai_client")
    oc.calls = []
    oc.retry_asked = []

    def _generate_one(*a, **k):
        oc.calls.append(1)
        return _Result()

    def _should_retry(result, retries, max_retries, *a, **k):
        oc.retry_asked.append((retries, max_retries))
        return bool(retry_decision) and retries < max_retries

    oc._generate_one = _generate_one
    oc._should_retry = _should_retry
    pkg = types.ModuleType("surya.inference.backends")
    pkg.openai_client = oc
    monkeypatch.setitem(sys.modules, "surya", types.ModuleType("surya"))
    monkeypatch.setitem(sys.modules, "surya.inference", types.ModuleType("surya.inference"))
    monkeypatch.setitem(sys.modules, "surya.inference.backends", pkg)
    monkeypatch.setitem(sys.modules, "surya.inference.backends.openai_client", oc)
    return oc


def _worker(mod, **opts):
    w = mod.SuryaWorker.__new__(mod.SuryaWorker)
    o = {"langs": "it", "token_trace": False, "repair_accents": False}
    o.update(opts)
    mod.SuryaWorker.__init__(w, o)
    return w


# ------------------------------------------------------------- the ceiling #
def test_our_budget_clamps_suryas_own_number(monkeypatch):
    """surya asks for 3. We say 1, and 1 is what the decision sees."""
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    w = _worker(mod, max_retries=1, token_trace=True)
    w._install_trace_hook()
    oc._should_retry(_Result(), 0, 3)
    assert oc.retry_asked[-1] == (0, 1)


def test_zero_disables_the_retry_entirely(monkeypatch):
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    w = _worker(mod, max_retries=0, token_trace=True)
    w._install_trace_hook()
    assert oc._should_retry(_Result(), 0, 3) is False


def test_none_hands_the_decision_back_to_surya(monkeypatch):
    """-1 in the config means "whatever surya decides" — we must not silently
    become the policy for an install that did not ask us to be."""
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    w = _worker(mod, max_retries=None, token_trace=True)
    w._install_trace_hook()
    oc._should_retry(_Result(), 0, 3)
    assert oc.retry_asked[-1] == (0, 3)


def test_the_budget_never_raises_suryas_number(monkeypatch):
    """A ceiling, not a setting: asking for 10 when surya allows 3 gets 3."""
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    w = _worker(mod, max_retries=10, token_trace=True)
    w._install_trace_hook()
    oc._should_retry(_Result(), 0, 3)
    assert oc.retry_asked[-1] == (0, 3)


def test_a_retry_is_logged(capsys, monkeypatch):
    """The whole reason this was invisible: a retried page looked like a slow
    one, in a batch where slowness was expected."""
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    w = _worker(mod, max_retries=2, token_trace=True)
    w._install_trace_hook()
    oc._should_retry(_Result(), 0, 3)
    err = capsys.readouterr().err
    assert "repeated itself" in err and "another full generation" in err


# ------------------------------------------------------- counting attempts #
def test_attempts_are_counted_per_page(monkeypatch):
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    w = _worker(mod, token_trace=True)
    w._install_trace_hook()
    mod._trace_begin()
    for _ in range(3):
        oc._generate_one()
    assert mod._attempts() == 3


def test_each_page_starts_from_zero(monkeypatch):
    """Page 2 must not inherit page 1's retries — that is what made the
    per-worker trace wrong once lanes put several pages in one process."""
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    w = _worker(mod, token_trace=True)
    w._install_trace_hook()
    mod._trace_begin()
    oc._generate_one(); oc._generate_one()
    assert mod._attempts() == 2
    mod._trace_begin()
    assert mod._attempts() == 0


def test_attempts_are_per_thread(monkeypatch):
    """Two lanes, two pages, two counts. A shared counter would blame one page
    for another's retries."""
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    w = _worker(mod, token_trace=True)
    w._install_trace_hook()
    seen = {}

    def lane(name, n):
        mod._trace_begin()
        for _ in range(n):
            oc._generate_one()
        seen[name] = mod._attempts()

    a = threading.Thread(target=lane, args=("a", 1))
    b = threading.Thread(target=lane, args=("b", 4))
    a.start(); b.start(); a.join(); b.join()
    assert seen == {"a": 1, "b": 4}


# --------------------------------------------------- it reaches the caller #
def test_a_retried_page_says_so_in_its_result():
    """Reported always, not only under token_trace: this is the difference
    between a slow page and a page generated four times over."""
    src = SRC.read_text()
    assert '"model_attempts"' in src
    blk = src.split("tries = _attempts()")[1].split("return page")[0]
    assert "if tries > 1 or capped or resumed:" in blk   # clean pages untouched


def test_a_truncated_reply_is_distinguished_from_an_unreadable_page():
    """The two failure modes a live batch produced: ~98 pages of ~71 characters
    and 100 with no text at all. Both are what a reply cut off mid-markup parses
    to, and both look exactly like a page the model could not read."""
    src = SRC.read_text()
    blk = src.split("tries = _attempts()")[1].split("return page")[0]
    assert '"truncated"' in blk and '"complete"' in blk
    # Wrapped across f-string lines in the source, so match the fragments that
    # are contiguous rather than the sentence they form.
    assert "INCOMPLETE." in blk and "cut-off reply" in blk


def test_a_clean_page_carries_no_attempt_block():
    """One attempt, well under the cap: byte-identical to what it always
    returned."""
    src = SRC.read_text()
    blk = src.split("tries = _attempts()")[1].split("return page")[0]
    assert "if tries > 1 or capped or resumed:" in blk


def test_the_token_counts_are_per_thread():
    """Like the trace and the attempt count: one lane must not report another
    lane's token counts, or the truncation verdict lands on the wrong page."""
    mod = _worker_module()
    import threading as _th
    seen = {}

    def lane(name, counts):
        mod._trace_begin()
        for c in counts:
            mod._attempt_tokens(c)
        seen[name] = mod._attempt_token_counts()

    a = _th.Thread(target=lane, args=("a", [2048]))
    b = _th.Thread(target=lane, args=("b", [11, 12]))
    a.start(); b.start(); a.join(); b.join()
    assert seen == {"a": [2048], "b": [11, 12]}


def test_the_cap_can_be_raised_without_touching_surya(monkeypatch):
    """surya's vLLM backend never passes max_tokens_default, so 2048 was the
    only value available. It arrives at _generate_one as a keyword, so the hook
    can raise it."""
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    seen = {}

    def _gen(*a, **k):
        seen.update(k)
        return _Result()

    oc._generate_one = _gen
    w = _worker(mod, token_trace=True, max_tokens=8192)
    w._install_trace_hook()
    oc._generate_one(max_tokens_default=2048)
    assert seen["max_tokens_default"] == 8192


def test_zero_leaves_suryas_own_cap_alone(monkeypatch):
    """Raising it costs tokens on every page, so it is opt-in."""
    mod = _worker_module()
    oc = _fake_surya(monkeypatch, mod)
    seen = {}
    oc._generate_one = lambda *a, **k: (seen.update(k), _Result())[1]
    w = _worker(mod, token_trace=True, max_tokens=0)
    w._install_trace_hook()
    oc._generate_one(max_tokens_default=2048)
    assert seen["max_tokens_default"] == 2048
    from codai.config import OcrConfig
    assert OcrConfig().surya_max_tokens == 0


def test_the_engine_sends_the_token_cap_at_load():
    from codai.ocr.surya import SuryaEngine
    from codai.config import OcrConfig
    cfg = OcrConfig()
    cfg.surya_max_tokens = 6000
    assert SuryaEngine(cfg)._worker_opts()["max_tokens"] == 6000


# -------------------------------------------- the engine passes it through #
def test_the_engine_sends_the_budget_at_load():
    from codai.ocr.surya import SuryaEngine
    from codai.config import OcrConfig
    cfg = OcrConfig()
    cfg.surya_max_retries = 1
    eng = SuryaEngine(cfg)
    assert eng._worker_opts()["max_retries"] == 1
    cfg.surya_max_retries = -1
    assert SuryaEngine(cfg)._worker_opts()["max_retries"] is None


def test_the_default_keeps_one_rescue_attempt():
    """A retry does rescue pages; it is the long tail that was costing the
    batch. 1 keeps the rescue and drops the tail."""
    from codai.config import OcrConfig
    assert OcrConfig().surya_max_retries == 1


def test_suryas_own_fanout_is_pinned_to_one_page():
    """A LANE is our unit of concurrency. Left unset, surya sizes its own thread
    pool from a built-in GPU table and fans out inside the worker, so
    instances x lanes would stop being the real number of pages in flight."""
    from codai.ocr.surya import SuryaEngine
    from codai.config import OcrConfig
    cfg = OcrConfig()
    cfg.surya_serve = "model"
    eng = SuryaEngine(cfg)
    eng._server_url = "https://pod:8000/v1"
    eng._server_key = "k"
    eng._served_name = "datalab-to/surya-ocr-2"
    assert eng._worker_env()["SURYA_INFERENCE_PARALLEL"] == "1"


def test_local_mode_gets_no_client_parallel():
    """There is no HTTP client in local mode; the setting would be noise."""
    from codai.ocr.surya import SuryaEngine
    from codai.config import OcrConfig
    cfg = OcrConfig()
    cfg.surya_serve = "local"
    assert "SURYA_INFERENCE_PARALLEL" not in SuryaEngine(cfg)._worker_env()


# ------------------------------------------------- the oversubscription note #
def test_how_many_pods_this_many_pages_implies(capsys, monkeypatch):
    """The useful number. 64 pages at 16 per pod is 4 pods — and a pod takes
    ~12 minutes, so the first burst gets whatever is already warm, however
    high max_pods is."""
    from codai.ocr import manager as m
    monkeypatch.setattr(m, "_fleet_shape", lambda name: (16, 64))
    m._warn_if_oversubscribed("surya", 64)
    out = capsys.readouterr().out
    assert "64 pages at once spreads over 4 pods" in out
    assert "12 min" in out and "already warm" in out


def test_more_pods_than_max_pods_is_called_out(capsys, monkeypatch):
    """Here the surplus genuinely cannot be served: it queues until a pod frees
    up, and no amount of waiting grows the fleet past max_pods."""
    from codai.ocr import manager as m
    monkeypatch.setattr(m, "_fleet_shape", lambda name: (16, 2))
    m._warn_if_oversubscribed("surya", 64)
    out = capsys.readouterr().out
    assert "needs 4 pods" in out and "max_pods is 2" in out and "32 will be served" in out


def test_one_pod_is_worth_no_remark(capsys, monkeypatch):
    """8 pages at 16 per pod fits one pod. Saying so would be noise."""
    from codai.ocr import manager as m
    monkeypatch.setattr(m, "_fleet_shape", lambda name: (16, 64))
    m._warn_if_oversubscribed("surya", 8)
    assert capsys.readouterr().out == ""


def test_an_unknown_fleet_says_nothing(capsys, monkeypatch):
    """No RunPod config, a local engine, a lookup that failed — none of those
    are a fleet, and guessing would cry wolf on every build."""
    from codai.ocr import manager as m
    monkeypatch.setattr(m, "_fleet_shape", lambda name: None)
    m._warn_if_oversubscribed("surya", 9999)
    assert capsys.readouterr().out == ""


def test_the_estimate_reads_the_models_own_runpod_block(monkeypatch):
    """Per-pod concurrency is one knob per model, not an OCR setting."""
    from codai.ocr import manager as m
    import codai.models.manager as mm
    monkeypatch.setattr(mm, "_model_entry_for",
                        lambda name: {"runpod": {"mode": "pods", "max_pods": 4,
                                                 "scale_up_inflight_per_pod": 16}},
                        raising=False)
    assert m._fleet_shape("surya") == (16, 4)


def test_a_serverless_model_has_no_pod_shape(monkeypatch):
    """The arithmetic is about pods; serverless scales on its own terms."""
    from codai.ocr import manager as m
    import codai.models.manager as mm
    monkeypatch.setattr(mm, "_model_entry_for",
                        lambda name: {"runpod": {"mode": "serverless",
                                                 "max_pods": 4,
                                                 "scale_up_inflight_per_pod": 16}},
                        raising=False)
    assert m._fleet_shape("surya") is None


# ------------------------------------------------- resuming a cut-off reply #
#
# Raising the cap only moves the cliff: a dense enough page overflows any fixed
# number, and incomplete block markup parses to a fragment or to nothing. A live
# document (4 pages) came back as 47 characters with `truncated: false` — the
# other failure mode — which is exactly why the two must be distinguishable and
# why a page known to be incomplete must say so rather than pass as read.

class _Img:
    width = height = 100

    def save(self, *a, **k):
        pass


class _Item:
    prompt = "ocr"
    prompt_type = "ocr"
    max_tokens = 0
    image = _Img()
    request_logprobs = True
    temperature = None
    top_p = None
    guided_json = None
    guided_regex = None


class _Completion:
    def __init__(self, text, tokens, logprobs=None):
        lp = type("_LP", (), {"content": logprobs or []})()
        choice = type("_C", (), {"message": type("_M", (), {"content": text})(),
                                 "logprobs": lp, "finish_reason": "length"})()
        self.choices = [choice]
        self.usage = type("_U", (), {"completion_tokens": tokens})()


def _client(segments, reject_flag=False, reject_all=False):
    """A chat client that hands back `segments` in order."""
    sent = []

    class _Chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                sent.append(kwargs)
                if reject_all:
                    raise RuntimeError("no")
                if reject_flag and "extra_body" in kwargs:
                    raise RuntimeError("unknown field continue_final_message")
                if not segments:
                    raise AssertionError("asked for more than it was given")
                text, tokens = segments.pop(0)
                return _Completion(text, tokens)

    c = type("_Client", (), {"chat": _Chat()})()
    c.sent = sent
    return c


def _resume(mod, monkeypatch, first, segments, budget=2, cap=100, **kw):
    oc = _fake_surya(monkeypatch, mod)
    oc._build_messages = lambda img, prompt: [{"role": "user", "content": prompt}]
    oc.scale_to_fit = lambda img: img
    oc.PROMPT_MAPPING = {"ocr": "ocr"}
    res = _Result(raw=first[0])
    res.token_count = first[1]
    cl = _client(list(segments), **kw)
    out, rounds, still = mod._continue_reply(
        oc, res, _Item(), cl, "m", cap, 0.0, 0.1, 60.0, False, budget)
    return out, rounds, still, cl


def test_a_cut_off_reply_is_resumed_and_joined():
    mod = _worker_module()
    import _pytest.monkeypatch as _mp
    m = _mp.MonkeyPatch()
    try:
        out, rounds, still, _ = _resume(mod, m, ("<block>aaa", 100),
                                        [("bbb</block>", 20)])
    finally:
        m.undo()
    assert out.raw == "<block>aaabbb</block>"
    assert rounds == 1 and still is False
    assert out.token_count == 120


def test_it_keeps_going_while_each_segment_fills_the_cap():
    mod = _worker_module()
    import _pytest.monkeypatch as _mp
    m = _mp.MonkeyPatch()
    try:
        out, rounds, still, _ = _resume(mod, m, ("a", 100),
                                        [("b", 100), ("c", 5)], budget=3)
    finally:
        m.undo()
    assert out.raw == "abc" and rounds == 2 and still is False


def test_the_budget_stops_it_and_the_page_says_it_is_incomplete():
    """A pathological page must not cost without bound — and must not pass as
    read either."""
    mod = _worker_module()
    import _pytest.monkeypatch as _mp
    m = _mp.MonkeyPatch()
    try:
        out, rounds, still, _ = _resume(mod, m, ("a", 100),
                                        [("b", 100), ("c", 100)], budget=2)
    finally:
        m.undo()
    assert rounds == 2 and still is True


def test_resumption_prefills_the_assistant_turn():
    """It must CONTINUE the exact string, not answer afresh about the same
    image — otherwise the second reply restarts the page and the join is
    nonsense."""
    mod = _worker_module()
    import _pytest.monkeypatch as _mp
    m = _mp.MonkeyPatch()
    try:
        _, _, _, cl = _resume(mod, m, ("partial text", 100), [("rest", 5)])
    finally:
        m.undo()
    msgs = cl.sent[0]["messages"]
    assert msgs[-1]["role"] == "assistant" and msgs[-1]["content"] == "partial text"
    assert cl.sent[0]["extra_body"]["continue_final_message"] is True
    assert cl.sent[0]["extra_body"]["add_generation_prompt"] is False


def test_a_server_that_rejects_the_flag_gets_a_plain_instruction():
    """`continue_final_message` is a vLLM extension; llama.cpp rejects it."""
    mod = _worker_module()
    import _pytest.monkeypatch as _mp
    m = _mp.MonkeyPatch()
    try:
        out, rounds, _, cl = _resume(mod, m, ("a", 100), [("b", 5)],
                                     reject_flag=True)
    finally:
        m.undo()
    assert out.raw == "ab" and rounds == 1
    assert "extra_body" not in cl.sent[-1]
    assert cl.sent[-1]["messages"][-1]["role"] == "user"


def test_a_server_that_rejects_both_leaves_the_page_partial_not_broken():
    """A page that cannot be finished is still a page. Raising here would turn
    a degraded result into no result."""
    mod = _worker_module()
    import _pytest.monkeypatch as _mp
    m = _mp.MonkeyPatch()
    try:
        out, rounds, still, _ = _resume(mod, m, ("a", 100), [], reject_all=True)
    finally:
        m.undo()
    assert out.raw == "a" and rounds == 0 and still is True


def test_resumption_is_at_temperature_zero():
    """The retry path deliberately RAISES temperature to break a repetition.
    Resumption is the opposite: it must continue the same text deterministically."""
    mod = _worker_module()
    import _pytest.monkeypatch as _mp
    m = _mp.MonkeyPatch()
    try:
        _, _, _, cl = _resume(mod, m, ("a", 100), [("b", 5)])
    finally:
        m.undo()
    assert cl.sent[0]["temperature"] == 0.0


def test_at_cap_allows_for_a_server_stopping_a_token_short():
    mod = _worker_module()
    assert mod._at_cap(2048, 2048) is True
    assert mod._at_cap(2040, 2048) is True
    assert mod._at_cap(1000, 2048) is False
    assert mod._at_cap(2048, 0) is False      # no cap known = no verdict


def test_the_engine_sends_the_resumption_budget():
    from codai.ocr.surya import SuryaEngine
    from codai.config import OcrConfig
    cfg = OcrConfig()
    assert (SuryaEngine(cfg)._worker_opts()["max_continuations"]
            == cfg.surya_max_continuations)
    cfg.surya_max_continuations = 0
    assert SuryaEngine(cfg)._worker_opts()["max_continuations"] == 0


def test_resumption_is_on_by_default():
    """Returning a page KNOWN to be incomplete is a correctness failure, not a
    cost trade-off, so this is not opt-in."""
    from codai.config import OcrConfig
    assert OcrConfig().surya_max_continuations >= 1


# ------------------------------------------- per-request budgets, and safety #
#
# One global cap cannot serve a public deployment: a four-page contract and a
# fifty-page judgment need different budgets, and only the caller knows which it
# is sending. /v1/ocr took engine, dpi, lang, structured, schema and detect —
# anything else a caller sent was silently dropped by FastAPI, so a `max_tokens`
# in the request was discarded and the page ran at the configured value.

def test_the_request_can_set_the_budgets():
    from codai.api.ocr import _page_opts
    assert _page_opts(8000, 20) == {"max_tokens": 8000, "max_continuations": 20}


def test_omitting_them_is_not_the_same_as_zero():
    """0 is meaningful for both — no cap override, and resumption off. A caller
    that said nothing must get the configured value, not zero."""
    from codai.api.ocr import _page_opts
    assert _page_opts(None, None) == {}
    assert _page_opts(0, 0) == {"max_tokens": 0, "max_continuations": 0}


def test_the_endpoint_actually_accepts_them():
    """The bug this fixes: FastAPI drops unknown form fields silently, so a
    caller can send max_tokens for a week and never be told it did nothing."""
    import inspect
    from codai.api.ocr import create_ocr
    params = inspect.signature(create_ocr).parameters
    assert "max_tokens" in params and "max_continuations" in params


def test_the_page_carries_the_options_to_the_worker():
    src = (ROOT / "codai/ocr/subprocess_engine.py").read_text()
    blk = src.split("def recognize_image")[1].split("\n    @staticmethod")[0]
    assert '"cmd": "ocr"' in blk and 'msg["opts"] = opts' in blk


def test_overrides_are_per_thread_so_lanes_cannot_cross():
    """Several pages are in flight in one process, and the hooks they feed are
    module-level. A per-request value in a module cell would be overwritten by
    the next page to arrive."""
    mod = _worker_module()
    import threading as _th
    seen = {}

    def lane(name, opts, expect):
        mod._trace_opts(opts)
        mod._trace_begin()
        seen[name] = mod._page_override("max_tokens", 2048)

    a = _th.Thread(target=lane, args=("a", {"max_tokens": 9000}, 9000))
    b = _th.Thread(target=lane, args=("b", {}, 2048))
    a.start(); b.start(); a.join(); b.join()
    assert seen == {"a": 9000, "b": 2048}


def test_beginning_a_page_does_not_clear_its_options():
    """`_trace_begin()` resets the page's trace; the options are set just before
    it and have to survive."""
    mod = _worker_module()
    mod._trace_opts({"max_continuations": 5})
    mod._trace_begin()
    assert mod._page_override("max_continuations", 2) == 5


def test_only_known_options_are_accepted():
    mod = _worker_module()
    mod._trace_opts({"max_tokens": 10, "rm": "-rf", "lanes": 99})
    assert mod._page_override("lanes", 1) == 1
    assert mod._page_override("max_tokens", 0) == 10


def test_a_resumption_that_repeats_itself_stops_early():
    """A budget raised for the one enormous page must not also buy a degenerate
    page that many more full generations.

    It also catches the server ignoring `continue_final_message`: a model that
    starts the page over returns the text we already have, which is a verbatim
    repeat of the previous segment. Resuming eight times against a server that
    does not support resumption would otherwise cost eight full generations and
    produce the page eight times over.
    """
    mod = _worker_module()
    import _pytest.monkeypatch as _mp
    m = _mp.MonkeyPatch()
    try:
        out, rounds, still, cl = _resume(mod, m, ("aaa", 100),
                                         [("aaa", 100), ("aaa", 100)], budget=8)
    finally:
        m.undo()
    # Nothing was added, so nothing was resumed: reporting a successful
    # resumption here would be the claim the counter was fixed to stop making.
    assert rounds == 0
    assert len(cl.sent) == 1, "it kept paying after the repeat"
    assert out.raw == "aaa", "the repeat must not be appended"
    assert still is True, "an unfinished page must still say so"


def test_the_default_budget_covers_a_dense_page():
    """Per PAGE, not per document: a fifty-page document is fifty pages each
    with its own budget. What this has to cover is the densest SINGLE page."""
    from codai.config import OcrConfig
    assert OcrConfig().surya_max_continuations >= 8
