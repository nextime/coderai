"""Keeping an over-long conversation inside the window, and saying so when it won't.

Measured on the Aruba production install 2026-10-22, which is why this exists:
`qwen38` is served from a rented A40 pod with `--max-model-len 32768`, clients
sent conversations longer than that, and the requests were simply refused --
auto-compaction has shipped for a while but defaults to OFF, so nothing shrank
them.

Three things are tested here, because all three were wrong in a way that only
showed up end to end:

* the SPLIT -- what survives. The head was reserved AFTER the tail was filled,
  so head+tail could overshoot the target by the whole head.
* the SUMMARY -- what gets summarized. It was built from "everything but the
  last two turns", which includes the messages kept verbatim, so the kept ends
  appeared twice in a window that was already too small.
* the REFUSAL -- what the client is told. The numbers were buried behind an
  apology, inside a pod's wrapper, inside the orchestrator's: a client reading
  a truncated prefix got three layers of prose and neither token count.
"""
import asyncio

import pytest

from codai.api import context_overflow as co
from codai.api.text import (_compact_after_refusal, _compact_messages,
                            _compact_strategy, _estimate_tokens,
                            _split_for_compaction)


def setup_function():
    co.forget()


# --------------------------------------------------------- reading a refusal #

#: Exactly as vLLM 0.27.1 refuses one, which is what a pod answers with.
VLLM = ("This model's maximum context length is 32768 tokens. However, you "
        "requested 35000 tokens (34000 in the messages, 1000 in the "
        "completion). Please reduce the length of the messages or completion.")


#: The refusal a production pod ACTUALLY sent, 2026-10-22, wrappers and all:
#: vLLM's words inside the pod's coderai inside the pod's HTTP error. Written
#: out in full because the first version of this parser was tested against a
#: hand-written fixture using the older OpenAI wording and matched nothing here
#: -- the real message names the OUTPUT tokens first and the prompt second.
PRODUCTION = (
    "The conversation is too long for this model's context window (400 Bad "
    "Request from http://127.0.0.1:41721/v1/chat/completions on "
    "chat/completions: This model's maximum context length is 32768 tokens. "
    "However, you requested 300 output tokens and your prompt contains at "
    "least 32469 input tokens, for a total of at least 32769 tokens. Please "
    "reduce the length of the input prompt or the number of requested output "
    "tokens. (parameter=input_tokens, value=32469)). Shorten the prompt or "
    "lower max_tokens, or increase the model's context size (n_ctx) in its "
    "configuration. (the pod answered 400)")


def test_the_refusal_production_really_sent_parses():
    got = co.parse(PRODUCTION)
    assert got["limit"] == 32768
    assert got["prompt"] == 32469
    assert got["completion"] == 300
    assert got["requested"] == 32769


def test_the_production_refusal_front_loads_both_numbers():
    """What the client keeps when it truncates: the first clause has to carry
    them, because the wrappers above push everything else out of reach."""
    msg = co.message(co.parse(PRODUCTION), model="qwen38")
    assert "32769" in msg[:90] and "32768" in msg[:90]


def test_the_prompt_size_is_recovered_from_the_parameter_clause():
    """A build that states the limit but not the split still appends
    "(parameter=input_tokens, value=N)", which is enough to plan a retry."""
    got = co.parse("This model's maximum context length is 8192 tokens. "
                   "However, the request is too long. "
                   "(parameter=input_tokens, value=9001)")
    assert got["limit"] == 8192 and got["prompt"] == 9001


def test_the_vllm_refusal_gives_up_all_four_numbers():
    got = co.parse(VLLM)
    assert got == {"limit": 32768, "requested": 35000,
                   "prompt": 34000, "completion": 1000}


def test_the_v1_engine_phrasing_parses_too():
    """Same condition, different words, and no total -- so the total is derived."""
    got = co.parse("The decoder prompt (length 41000) is longer than the "
                   "maximum model length of 32768.")
    assert got["limit"] == 32768
    assert got["prompt"] == 41000
    assert got["requested"] == 41000


def test_llama_cpp_phrasing_parses():
    got = co.parse("Requested tokens (5000) exceed context window of 4096")
    assert got["limit"] == 4096 and got["requested"] == 5000


def test_our_own_message_parses_back():
    """A downstream coderai's refusal reaches us as text; it must read as well as
    an engine's, or a pod in front of a pod loses the numbers."""
    first = co.message({"limit": 32768, "requested": 35000, "prompt": 34000},
                       model="qwen38")
    got = co.parse(first)
    assert got["limit"] == 32768 and got["requested"] == 35000


def test_an_unrelated_error_is_not_an_overflow():
    assert co.parse("401 Unauthorized") is None
    assert not co.is_overflow("the pod answered 502: bad gateway")
    assert not co.is_overflow("")


def test_an_overflow_without_numbers_is_still_recognised():
    """A retry needs numbers, but the CLASSIFICATION must not: calling this a
    plain 500 loses the one error a client can act on."""
    assert co.is_overflow("the request exceeds the available context size")
    assert co.is_overflow("KV cache is full and context shift is disabled")


# ------------------------------------------------------- answering with one #

def test_the_numbers_come_before_the_apology():
    """THE TRAP: this message is wrapped by the pod's coderai and then by the
    orchestrator, each adding a prefix and a 120-character proxy URL. A client
    truncating at 300 characters got the apologies and no figures, so it could
    not tell this from any other failure -- measured in production."""
    msg = co.message(co.parse(VLLM), model="qwen38")
    assert msg.startswith("context_length_exceeded:")
    head = msg[:90]
    assert "35000" in head and "32768" in head
    assert "qwen38" in msg


def test_the_engine_s_own_words_are_kept_after_the_numbers():
    msg = co.message(co.parse(VLLM), model="m", raw=VLLM)
    assert msg.index("32768") < msg.index("engine said")


def test_a_message_without_numbers_still_names_the_condition():
    msg = co.message(None, model="m")
    assert msg.startswith("context_length_exceeded:")


def test_the_window_an_engine_reports_is_remembered():
    """The configured ctx is what coderai ASKS for; the served window is what can
    refuse a request, and only a refusal reveals the two disagreeing."""
    assert co.learned_limit("qwen38") is None
    co.learn_from("qwen38", VLLM)
    assert co.learned_limit("qwen38") == 32768
    co.forget("qwen38")
    assert co.learned_limit("qwen38") is None


def test_a_refusal_with_no_limit_teaches_nothing():
    co.learn_from("m", "the request exceeds the available context size")
    assert co.learned_limit("m") is None


# ---------------------------------------------------------------- the split #

def _convo(n, chars=400):
    """n alternating turns, each ~chars/4 tokens, numbered so they can be told
    apart after compaction."""
    out = [{"role": "system", "content": "SYSTEM RULES"}]
    for i in range(n):
        out.append({"role": "user" if i % 2 == 0 else "assistant",
                    "content": f"msg{i} " + ("x" * chars)})
    return out


def test_the_middle_is_what_gets_dropped():
    msgs = _convo(20)
    sys_m, head, middle, tail = _split_for_compaction(
        msgs, target=600, strategy="summarize", head_messages=2)
    assert [m["content"].split()[0] for m in head] == ["msg0", "msg1"]
    assert tail and tail[-1] is msgs[-1]          # the live turn survives
    assert middle                                  # something was actually lost
    # Every message is accounted for exactly once.
    assert len(sys_m) + len(head) + len(middle) + len(tail) == len(msgs)


def test_the_head_is_reserved_before_the_tail_is_filled():
    """THE BUG: the tail used to be filled to the whole target and the head added
    afterwards, so the result overshot the target by the size of the head -- and
    was refused for the very length compaction had just been run to avoid."""
    msgs = _convo(30)
    target = 900
    sys_m, head, _, tail = _split_for_compaction(
        msgs, target=target, strategy="keep_head_tail", head_messages=3)
    assert head, "the head is the point of this strategy"
    kept = _estimate_tokens(sys_m + head + tail)
    assert kept <= target


def test_a_big_head_cannot_starve_the_recent_end():
    msgs = _convo(40)
    _, head, _, tail = _split_for_compaction(
        msgs, target=1000, strategy="keep_head_tail", head_messages=99)
    assert tail, "asking for a huge head must still leave recent turns"
    assert _estimate_tokens(head) <= _estimate_tokens(tail) + 200


def test_the_head_can_be_capped_in_tokens():
    msgs = _convo(40)
    _, head, _, _ = _split_for_compaction(
        msgs, target=4000, strategy="keep_head_tail",
        head_messages=10, head_tokens=250)
    assert head and _estimate_tokens(head) <= 250 + _estimate_tokens([msgs[1]])


def test_the_tail_can_be_capped_in_tokens():
    msgs = _convo(40)
    _, _, middle, tail = _split_for_compaction(
        msgs, target=8000, strategy="keep_head_tail", tail_tokens=300)
    assert _estimate_tokens(tail) <= 300 + _estimate_tokens([msgs[-1]])
    assert middle, "a capped tail must leave a middle to compact"


def test_drop_oldest_keeps_no_head():
    _, head, middle, tail = _split_for_compaction(
        _convo(20), target=600, strategy="drop_oldest", head_messages=5)
    assert head == []
    assert middle and tail


def test_the_kept_tail_never_starts_on_an_orphan_tool_result():
    """A tool result whose assistant tool_calls turn was dropped makes a chat
    template raise -- the request fails for the shape of the history, not its
    length."""
    msgs = [{"role": "system", "content": "s"}]
    for i in range(8):
        msgs.append({"role": "user", "content": f"q{i} " + "x" * 400})
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": "f", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": "result " + "y" * 400})
    msgs.append({"role": "user", "content": "now answer"})
    _, _, _, tail = _split_for_compaction(msgs, target=500, strategy="drop_oldest")
    assert tail[0]["role"] != "tool"


def test_the_head_never_ends_on_an_unanswered_tool_call():
    """The mirror of the rule above: an assistant tool_calls turn kept in the head
    whose results are in the dropped middle breaks the same templates."""
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "q " + "x" * 400},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "r " + "y" * 2000}]
    msgs += [{"role": "user", "content": f"m{i} " + "x" * 400} for i in range(10)]
    _, head, _, _ = _split_for_compaction(
        msgs, target=700, strategy="keep_head_tail", head_messages=2)
    assert not any(m.get("tool_calls") for m in head)


# ------------------------------------------------------------ the assembling #

def test_compaction_keeps_the_system_prompt_and_the_live_turn():
    msgs = _convo(30)
    out, info = _compact_messages(msgs, n_ctx=2000, pct=85, strategy="keep_head_tail")
    assert info and info["dropped"]
    assert out[0]["content"] == "SYSTEM RULES"
    assert out[-1] is msgs[-1]
    assert info["after_tokens"] < info["before_tokens"]


def test_a_conversation_that_fits_is_left_alone():
    msgs = _convo(2)
    out, info = _compact_messages(msgs, n_ctx=32768, pct=85, strategy="summarize")
    assert info is None and out is msgs


def test_the_summary_is_inserted_as_a_system_note():
    msgs = _convo(30)
    out, info = _compact_messages(msgs, n_ctx=2000, pct=85, strategy="summarize",
                                  summary_text="THE GIST")
    assert any("THE GIST" in str(m.get("content")) for m in out)
    assert info["strategy"] == "summarize"


def test_without_a_summary_the_loss_is_still_declared():
    """Silently shorter history makes a model look like it forgot; the note is
    what tells the model (and a reader of the log) that it was cut."""
    msgs = _convo(30)
    out, _ = _compact_messages(msgs, n_ctx=2000, pct=85, strategy="keep_head_tail")
    assert any("omitted to fit the context window" in str(m.get("content"))
               for m in out)


def test_the_target_percentage_is_honoured():
    msgs = _convo(60)
    tight, _ = _compact_messages(msgs, n_ctx=4000, pct=80,
                                 strategy="keep_head_tail", target_pct=25)
    loose, _ = _compact_messages(msgs, n_ctx=4000, pct=80,
                                 strategy="keep_head_tail", target_pct=80)
    assert _estimate_tokens(tight) < _estimate_tokens(loose)


def test_an_unknown_strategy_keeps_the_ends_rather_than_dropping_the_head():
    """A typo in a config must not quietly throw away the opening instructions:
    that looks like a model ignoring its brief, with nothing to point at."""
    assert _compact_strategy("summarize_middle") == "summarize"
    assert _compact_strategy("Keep-Head-Tail") == "keep_head_tail"
    assert _compact_strategy("summarise") == "summarize"
    assert _compact_strategy("nonsense") == "keep_head_tail"
    assert _compact_strategy("") == "drop_oldest"
    assert _compact_strategy(None) == "drop_oldest"


# ------------------------------------------------------- retry on a refusal #

class _FakeManager:
    """Enough of a manager for the compaction plan: a window and a summarizer."""

    def __init__(self, n_ctx=32768):
        self._n = n_ctx
        self.model_name = "fake"
        self.summarized = []

    def get_context_size(self):
        return self._n

    def generate_chat(self, messages=None, **kw):
        self.summarized.append(messages)
        return "SUMMARY OF THE MIDDLE"


def _plan(messages, **over):
    plan = {"pct": 85.0, "strategy": "summarize", "n_ctx": 32768,
            "compact_model": "", "current_manager": _FakeManager(),
            "model": "qwen38", "max_tokens": 1000, "tolerance": 1.2,
            "min_output": 512, "safety": 1.15, "head_messages": 1,
            "head_tokens": 0, "tail_tokens": 0, "target_pct": 65.0,
            "retry_on_overflow": True}
    plan.update(over)
    return plan


def test_a_refusal_plans_the_retry_from_its_own_numbers():
    """The estimate let this prompt through, so repeating it with the same
    estimate would repeat the refusal. The engine counted the prompt for real;
    that count is what the retry is planned against."""
    msgs = _convo(40, chars=800)
    plan = _plan(msgs)
    got = asyncio.run(_compact_after_refusal(plan, msgs, "qwen38", VLLM))
    assert got is not None
    out, eff, info = got
    assert info and info["dropped"]
    assert _estimate_tokens(out) < _estimate_tokens(msgs)
    assert co.learned_limit("qwen38") == 32768   # and remembered for next time


def test_the_retry_is_abandoned_when_the_refusal_taught_nothing():
    """No numbers means no better plan, and re-sending an identical request is
    worse than reporting the error."""
    msgs = _convo(40)
    got = asyncio.run(_compact_after_refusal(
        _plan(msgs), msgs, "m", "the request exceeds the available context size"))
    assert got is None


def test_the_retry_respects_a_window_smaller_than_the_configured_one():
    """A proxied model is launched elsewhere: `ctx: 32768` in models.json is a
    request, not a fact, and a pod started with half of it refuses accordingly."""
    msgs = _convo(60, chars=600)
    plan = _plan(msgs, n_ctx=131072)
    got = asyncio.run(_compact_after_refusal(
        plan, msgs, "qwen38",
        "This model's maximum context length is 8192 tokens. However, you "
        "requested 40000 tokens (39000 in the messages, 1000 in the completion)."))
    assert got is not None
    out, _eff, _info = got
    assert _estimate_tokens(out) < 8192
    assert co.learned_limit("qwen38") == 8192


def test_the_summarizer_is_given_only_the_messages_being_dropped():
    """THE BUG: the summary was built from everything but the last two turns, so
    the head and most of the kept tail were summarized AND kept verbatim --
    paying twice for the same words in a window that was already short, and
    presenting a paraphrase of the live conversation as earlier history."""
    msgs = _convo(40, chars=800)
    mgr = _FakeManager()
    plan = _plan(msgs, current_manager=mgr, strategy="summarize")
    out, _eff, _info = asyncio.run(
        _compact_after_refusal(plan, msgs, "qwen38", VLLM))
    assert mgr.summarized, "the summarize strategy must have summarized something"
    handed = "\n".join(str(m[-1].get("content")) for m in mgr.summarized)
    kept = [str(m.get("content")) for m in out if "msg" in str(m.get("content"))]
    first_kept = next((c.split()[0] for c in kept), None)
    last_kept = kept[-1].split()[0] if kept else None
    assert first_kept and last_kept
    # The messages that SURVIVED verbatim are not in the summarizer's input.
    assert f"{first_kept} " not in handed
    assert f"{last_kept} " not in handed
    # ...and something from the middle is.
    assert "msg20" in handed or "msg15" in handed
