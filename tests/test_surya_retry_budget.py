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
    blk = src.split("tries = _attempts()")[1][:1600]
    assert "if tries > 1 or capped:" in blk      # a clean page stays untouched


def test_a_truncated_reply_is_distinguished_from_an_unreadable_page():
    """The two failure modes a live batch produced: ~98 pages of ~71 characters
    and 100 with no text at all. Both are what a reply cut off mid-markup parses
    to, and both look exactly like a page the model could not read."""
    src = SRC.read_text()
    blk = src.split("tries = _attempts()")[1][:1600]
    assert '"truncated"' in blk
    assert "cap" in blk and "NOT a page the model could not read" in blk


def test_a_clean_page_carries_no_attempt_block():
    """One attempt, well under the cap: byte-identical to what it always
    returned."""
    src = SRC.read_text()
    i = src.index("tries = _attempts()")
    assert "if tries > 1 or capped:" in src[i:i + 1600]


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
