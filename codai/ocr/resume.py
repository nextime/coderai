# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Resuming a transcription that stopped at its token cap — for any OCR engine.

A VLM asked to transcribe a dense page stops when it reaches `max_tokens`, and
what it has produced is half a sentence inside half a block of markup. Parsed,
that yields a few blocks or none at all, which in the finished text is
indistinguishable from a page the model could not read. The fix is to send the
text back as a prefilled assistant turn and ask for the rest — but the rules
for doing that safely were learned the hard way on surya, and they are the same
rules for every engine:

* **the flags have to arrive.** vLLM resumes a prefilled turn only with
  `continue_final_message: true` AND `add_generation_prompt: false`. With the
  second left at its default the chat template appends a fresh assistant turn
  and the model starts a NEW answer about the same image — which looks exactly
  like a model repeating itself, and was read that way for a whole day;
* **a reply that is already repeating itself must not be resumed.** It did not
  run out of room, it ran out of page. Resuming can only extend the loop, and
  the engine is about to re-run the page anyway;
* **the context window is the ceiling, not the budget.** Resumption re-sends
  everything so far as INPUT, so the room left shrinks by what was just
  produced. 2048-token segments in a 32k window give about thirteen; 12288 give
  two. A segment that no longer fits is not short of budget, it is out of
  context — and the fix is a SMALLER cap, not a bigger one;
* **progress must be real.** A segment identical to the previous one, or one
  that comes back looping, ends the loop. Otherwise a budget raised for the one
  genuinely enormous page also buys a degenerate page that many more full
  generations.

The engine supplies one callback that knows how to talk to its own server, and
gets back the joined text plus a `model_attempts` report in the shape every
engine uses, so a caller can read one decision table instead of three.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional

#: What the image and the instructions cost, before any server has told us. Only
#: the first round needs a guess: from then on `usage.prompt_tokens` is exact.
PROMPT_TOKENS_GUESS = 4096

#: A floor on the room LEFT, never on the cap. Comparing it against the cap
#: treated any deployment whose cap is below this as a permanently full window.
MIN_SEGMENT_TOKENS = 256


@dataclass
class Reply:
    """One generation, with the two things that say whether it is finished."""

    text: str = ""
    #: "length" is the server saying it hit the cap. Authoritative when present:
    #: a token count near the cap is a guess, this is not.
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    conf: Optional[float] = None


@dataclass
class Resumed:
    """The outcome of a resumption loop."""

    text: str = ""
    rounds: int = 0
    #: Still unfinished after everything we were allowed to do.
    truncated: bool = False
    tokens: List[int] = field(default_factory=list)
    #: Why it was not resumed (or not resumed further), when that is known.
    #: Never a hardcoded guess — a diagnostic that names the wrong cause is
    #: acted on, and one did send a whole investigation the wrong way.
    reason: str = ""
    conf: Optional[float] = None


def result_dict(result) -> dict:
    """A chat-completions result as a plain dict, whatever shape it arrived in.

    The in-process handler returns a pydantic model or a JSONResponse, an HTTP
    server returns parsed JSON, and a stub in a test returns a dict.
    """
    import json as _json

    r = None
    if isinstance(result, dict):
        r = result
    elif hasattr(result, "model_dump"):
        try:
            r = result.model_dump()
        except Exception:
            r = None
    if r is None and hasattr(result, "body"):
        try:
            r = _json.loads(result.body)
        except Exception:
            r = None
    return r if isinstance(r, dict) else {}


def finish_and_usage(result) -> tuple:
    """``(finish_reason, prompt_tokens, completion_tokens)``, 0 when unreported.

    The finish reason is what separates a transcription that ran out of budget
    from one that reached the end of the page — the single fact a cut-off reply
    can be recognised by, and every engine but surya was discarding it.
    """
    r = result_dict(result)
    choices = r.get("choices") or []
    reason = ""
    if choices and isinstance(choices[0], dict):
        reason = str(choices[0].get("finish_reason") or "")
    usage = r.get("usage") if isinstance(r.get("usage"), dict) else {}
    return (reason, int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0))


def at_cap(reply: "Reply", cap: int) -> bool:
    """Whether this reply stopped because it ran out of budget."""
    if not cap:
        return False
    # A finish reason the server gave is authoritative BOTH ways: "stop" means
    # the model finished and happened to use its whole budget, which is a
    # complete page and must not be resumed. Only when nothing was reported does
    # the token count have to stand in for it.
    reason = (reply.finish_reason or "").lower()
    if reason:
        return reason == "length"
    return bool(reply.completion_tokens) and reply.completion_tokens >= cap * 0.98


def segment_cap(cap: int, limit: int, prompt_tokens: int) -> int:
    """How many tokens the next segment may ask for.

    0 when the window is full. ``limit`` 0 means unknown, in which case the
    cap stands as configured — guessing a window would either waste it or
    overrun it.
    """
    if not limit:
        return max(0, int(cap))
    return max(0, min(int(cap), int(limit) - int(prompt_tokens)))


def is_context_error(exc: BaseException) -> bool:
    """Whether a server's refusal is "this no longer fits in the window"."""
    t = f"{exc}".lower()
    return (("context" in t and ("length" in t or "window" in t))
            or "maximum context" in t or "longer than the maximum" in t
            or "reduce the length" in t or "max_model_len" in t)


def budgets(cfg, prefix: str, default_cap: int = 0) -> dict:
    """``{cap, continuations, context_tokens}`` for the page being read.

    The configured values (``<prefix>_max_tokens``,
    ``<prefix>_max_continuations``, ``<prefix>_context_tokens``), overridden by
    anything this REQUEST asked for. One global cap cannot serve a public
    deployment: a four-page contract and a fifty-page judgment need different
    budgets and only the caller knows which it is sending.

    Omitted means "use the configured value" — not zero, which is a meaningful
    setting for both.
    """
    from codai.ocr.base import page_options

    out = {
        "cap": max(0, int(getattr(cfg, f"{prefix}_max_tokens", default_cap)
                          or default_cap or 0)),
        "continuations": max(0, int(getattr(cfg, f"{prefix}_max_continuations", 0)
                                    or 0)),
        "context_tokens": max(0, int(getattr(cfg, f"{prefix}_context_tokens", 0)
                                     or 0)),
    }
    opts = page_options() or {}
    for key, name in (("max_tokens", "cap"),
                      ("max_continuations", "continuations"),
                      ("context_tokens", "context_tokens")):
        if key in opts:
            try:
                out[name] = max(0, int(opts[key]))
            except (TypeError, ValueError):
                pass
    return out


def resume(ask: Callable[[str, int], "Reply"], first: "Reply", cap: int,
           budget: int, limit: int = 0, repeats: Optional[Callable] = None,
           label: str = "ocr", log: Optional[Callable] = None) -> "Resumed":
    """Keep asking for the rest of a cut-off reply, within the rules above.

    ``ask(prefix, seg_cap)`` sends the text so far back as a prefilled assistant
    turn and returns the next :class:`Reply`; the engine owns that call because
    only it knows its server, its auth and its logprob settings.

    ``repeats(text)`` is the engine's own degeneration detector, used both to
    refuse resuming a looping reply and to stop when one comes back.
    """
    say = log or (lambda msg: print(f"[{label}] {msg}", flush=True))
    out = Resumed(text=first.text or "", rounds=0,
                  tokens=[int(first.completion_tokens or 0)],
                  conf=first.conf)
    if not budget:
        out.truncated = at_cap(first, cap)
        if out.truncated:
            out.reason = ("resumption is off for this page "
                          f"(max_continuations = 0, from the request or "
                          f"{label}_max_continuations)")
        return out
    if not at_cap(first, cap):
        return out

    if repeats is not None:
        try:
            if repeats(first.text or ""):
                say(f"not resuming a reply that is already repeating itself — "
                    f"it stopped at the {cap}-token cap because it is looping, "
                    f"not because the page is long")
                out.reason = ("the generation was already repeating itself, so "
                              "it stopped at the cap because it is LOOPING, not "
                              "because the page is long; resuming would only "
                              "extend the loop")
                return out
        except Exception:
            pass

    segments = [first.text or ""]
    last = int(first.completion_tokens or 0)
    total = last
    # Everything generated so far comes back as input on the next round, so the
    # input grows by each segment. Exact from the second round, when the server
    # has reported what the prefix and the image actually cost.
    prompt_tokens = (int(first.prompt_tokens or 0) + last
                     if first.prompt_tokens else PROMPT_TOKENS_GUESS + total)

    attempts = 0
    # We are here because `first` hit its cap. From then on it is the cap each
    # SEGMENT was given that decides — a segment shortened by the context window
    # is finished when it fills that smaller number, not the configured one.
    hit = True
    while attempts < int(budget) and hit:
        attempts += 1
        seg = segment_cap(cap, limit, prompt_tokens)
        if limit and seg < min(int(cap), MIN_SEGMENT_TOKENS):
            say(f"page filled the model's {limit}-token context window after "
                f"{out.rounds} resumption(s) — returning it partial and marked "
                f"(a smaller {label}_max_tokens fits more text in the same "
                f"window)")
            out.reason = (f"the page filled the model's {limit}-token context "
                          f"window; a SMALLER {label}_max_tokens fits more text "
                          f"in the same window, a larger one cannot help")
            break
        try:
            more = ask("".join(segments), seg)
        except Exception as exc:
            if is_context_error(exc):
                say(f"page filled the model's context window after "
                    f"{out.rounds} resumption(s) ({exc}) — returning it partial "
                    f"and marked")
                out.reason = f"the server refused the next segment: {exc}"
            else:
                say(f"could not resume a cut-off page ({type(exc).__name__}: "
                    f"{exc}) — returning it partial and marked")
                out.reason = f"the next segment could not be fetched ({exc})"
            break
        text = (more.text or "")
        last = int(more.completion_tokens or 0)
        prompt_tokens = (int(more.prompt_tokens or 0) + last
                         if more.prompt_tokens else prompt_tokens + last)
        if not text:
            out.reason = "the server returned an empty continuation"
            break
        if text == segments[-1]:
            say("resumption repeated the previous segment verbatim — stopping")
            out.reason = ("the resumption repeated the previous segment "
                          "verbatim, so the page has given all it will")
            break
        looping = False
        if repeats is not None:
            try:
                looping = bool(repeats(text))
            except Exception:
                looping = False
        segments.append(text)
        out.rounds += 1
        out.tokens.append(last)
        total += last
        hit = at_cap(more, seg)
        if looping:
            say("resumption came back repeating itself — stopping rather than "
                "spending the rest of the budget on it")
            out.reason = ("a resumption came back repeating itself, so the rest "
                          "of the budget was not spent on it")
            break
        say(f"page resumed after the {cap}-token cap (segment {out.rounds + 1}, "
            f"{last} more tokens, "
            f"{max(0, limit - prompt_tokens) if limit else '?'} of the context "
            f"window left)")

    out.text = "".join(segments)
    out.truncated = bool(hit)
    if out.truncated and not out.reason and attempts >= int(budget):
        out.reason = (f"still unfinished after {out.rounds} resumption(s) — the "
                      f"budget ({budget}) ran out, so raise "
                      f"{label}_max_continuations or {label}_max_tokens")
    return out


def report(res: "Resumed", cap: int, budget: int, retries: int = 0,
           label: str = "ocr") -> dict:
    """The ``model_attempts`` block, in the shape every engine uses.

    A page whose reply fitted and needed no retry carries nothing: the field
    appears only when there is something to report, so an untouched page stays
    byte-identical to what it always returned.
    """
    capped = any(t >= cap * 0.98 for t in res.tokens if t) if cap else False
    if not (retries or capped or res.rounds):
        return {}
    info = {"calls": 1 + int(retries) + int(res.rounds),
            "tokens": list(res.tokens), "cap": int(cap),
            "max_continuations": int(budget)}
    if res.rounds:
        info["resumed"] = res.rounds
        info["why"] = (f"the reply ran past the {cap}-token cap and was resumed "
                       f"{res.rounds} time(s) from where it stopped, so the "
                       f"page is whole")
    if res.truncated:
        info["truncated"] = True
        info["complete"] = False
        info["why"] = (f"the reply stopped at the {cap}-token cap and was not "
                       f"finished: {res.reason}. So this page is INCOMPLETE. A "
                       f"cut-off reply, NOT a page the model could not read."
                       if res.reason else
                       f"the reply stopped at the {cap}-token cap and is "
                       f"INCOMPLETE. A cut-off reply, NOT a page the model "
                       f"could not read.")
    elif capped and not res.rounds:
        info["truncated"] = True
        info["complete"] = False
        info["why"] = (f"the reply stopped at the {cap}-token cap and was not "
                       f"resumed: {res.reason}. So this page is INCOMPLETE."
                       if res.reason else
                       f"the reply stopped at the {cap}-token cap and was not "
                       f"resumed. So this page is INCOMPLETE.")
    elif retries:
        info["why"] = (f"{label} re-ran this page because the generation "
                       f"repeated itself; each re-run is a full generation at a "
                       f"higher temperature")
    return info
