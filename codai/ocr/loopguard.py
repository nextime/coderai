# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Stop a degenerate page as it happens, instead of paying for all of it.

A VLM that loses its place on a page does not fail. It repeats the last thing
it saw until it runs out of tokens, and the result arrives looking like a
successful transcription -- which is why every engine here already runs
:func:`codai.ocr.chandra.detect_repeat_token` over the finished reply and
treats a looping one as retryable.

The waste is that the reply has to FINISH first. The budget bounds the worst
case, so a page that starts looping on its tenth token still emits the whole
cap before anyone looks at it, and then the retry does it again at a higher
temperature. Measured on a 100-page sample of a real judicial corpus (surya,
A40, cap 4096 / 1 continuation): 4 pages of 100 looped and accounted for
**18.9% of the run's entire generation time**. One of them took 450 s to
produce 17,586 characters of the same clause over and over.

So the detector runs on the growing output instead, and the generation is
abandoned the moment it is provably degenerate.

WHY THIS CANNOT REJECT A PAGE THAT WOULD HAVE BEEN KEPT
-------------------------------------------------------
It is the same function, with the same thresholds, that already judges the
finished reply. Nothing new decides anything: a page aborted here is a page the
existing post-hoc check would have rejected anyway, reached sooner. That is the
whole design, and it is what makes enabling this by default defensible.

Two further margins, both measured rather than guessed, against cutting a page
that only LOOKS repetitive for a while -- a form, a table of dotted leaders, a
column of repeated dates:

* nothing is judged before ``min_chars`` of output (2000; the largest genuine
  page in the corpus is 5,054 characters and the median is ~1,900, so this is
  comfortably past where a real page has shown its character);
* the detector must fire on ``persistence`` consecutive checks (2, i.e. the
  repetition survives another 400 characters). A real loop never recovers; a
  repetitive passage does.

Swept over 393 page reads from four measured runs:

    min_chars  persistence   loopers caught   false positives
    1200       1             12/12            1   <- a table, then real text
    1200       2             12/12            0
    2000       1             12/12            0
    2000       2             12/12            0   <- shipped
    3000       3             12/12            0

The single false positive at the loosest setting is the case this guards
against: a 1,444-character page with a repetitive middle and genuine text
after it. Aborting there would have been silent data loss, which is the worst
failure this system has -- a truncated page answers 200 and gets indexed. Both
margins cost 0.5 percentage points of the saving between them, and they are
kept for that reason.
"""

import json
from typing import Callable, Optional

from codai.ocr.base import OcrError

#: Nothing is judged before this much output exists. See the module docstring:
#: the largest genuine page in the measured corpus is 5,054 characters.
DEFAULT_MIN_CHARS = 2000
#: How often to re-run the detector, in characters of new output. The detector
#: is cheap but not free (it walks candidate sequence lengths backwards from the
#: tail), and running it per streamed chunk would be per TOKEN.
DEFAULT_STEP = 400
#: Consecutive checks that must all say "repeating" before the page is
#: abandoned. A real loop never recovers.
DEFAULT_PERSISTENCE = 2


def settings(cfg) -> dict:
    """``{enabled, min_chars, step, persistence}`` for this install."""
    def _int(name, default):
        try:
            v = int(getattr(cfg, name, default) or 0)
        except (TypeError, ValueError):
            return default
        return v if v > 0 else default

    return {
        "enabled": bool(getattr(cfg, "early_loop_abort", True)),
        "min_chars": _int("early_loop_min_chars", DEFAULT_MIN_CHARS),
        "step": _int("early_loop_step", DEFAULT_STEP),
        "persistence": _int("early_loop_persistence", DEFAULT_PERSISTENCE),
    }


class LoopGuard:
    """Decides, as output arrives, whether a generation has gone degenerate.

    Feed it the text so far (or each new chunk); it answers True once it is
    confident. Cheap: the detector runs once per ``step`` characters, not per
    chunk, and never before ``min_chars``.
    """

    def __init__(self, min_chars: int = DEFAULT_MIN_CHARS,
                 step: int = DEFAULT_STEP,
                 persistence: int = DEFAULT_PERSISTENCE,
                 detector: Optional[Callable[[str], bool]] = None):
        self.min_chars = max(1, int(min_chars))
        self.step = max(1, int(step))
        self.persistence = max(1, int(persistence))
        self._detect = detector or _default_detector()
        self._text: list = []
        self._len = 0
        self._next_at = self.min_chars
        self._run = 0
        self.tripped = False
        #: Characters of output at the moment it tripped, for the report.
        self.tripped_at = 0

    @property
    def text(self) -> str:
        return "".join(self._text)

    def feed(self, chunk: str) -> bool:
        """Add output; True once the page should be abandoned.

        Stays True once it has tripped: a caller that keeps feeding (a buffered
        reply replayed through the guard, say) must not see the verdict flip.
        """
        if chunk:
            self._text.append(chunk)
            self._len += len(chunk)
        if self.tripped or self._len < self._next_at:
            return self.tripped
        # ONE CHECK PER STEP BOUNDARY CROSSED, not one per call. A chunk that
        # arrives carrying several steps' worth of text must be judged as many
        # times as the same text would have been had it dribbled in, or the
        # verdict depends on how the server happens to frame its chunks -- and
        # with `persistence` above 1 a caller that fed a whole reply at once
        # could never reach it at all, which is exactly how this was first
        # written. The loop normally runs once.
        text = self.text
        while not self.tripped and self._len >= self._next_at:
            at = self._next_at
            self._next_at = at + self.step
            try:
                hit = bool(self._detect(text[:at]))
            except Exception:
                # A detector that raises must not take the page down with it.
                # The post-hoc check still runs on the finished reply.
                return False
            if not hit:
                self._run = 0
                continue
            self._run += 1
            if self._run >= self.persistence:
                self.tripped = True
                self.tripped_at = at
        return self.tripped

    def reason(self) -> str:
        return (f"the generation was repeating itself after {self.tripped_at} "
                f"characters and was abandoned there rather than running to the "
                f"token cap; the page is NOT readable from this reply")


def _default_detector():
    """chandra's port of its upstream's detector, imported lazily.

    Lazily because this module is imported by the engines at load time and
    `chandra` pulls in an HTML parser and the rest of that engine; and shared
    rather than reimplemented because the POINT is that the early check and the
    final check are the same judgement.
    """
    def _detect(text: str) -> bool:
        from codai.ocr.chandra import detect_repeat_token
        return detect_repeat_token(text)
    return _detect


# --------------------------------------------------------------------------- #
# the streaming transport
# --------------------------------------------------------------------------- #

class StreamUnsupported(Exception):
    """The stream produced nothing usable, and NOTHING WAS CONSUMED.

    The distinction a retry depends on. A caller may safely re-issue the
    request as an ordinary blocking call only if the server never began
    answering -- a connection error, a refusal, a body in a shape this cannot
    read before any text arrived. Once text has arrived the generation has
    HAPPENED and is being paid for, so re-issuing it would run the page twice:
    two generations billed, and on an engine with a resumption queue the second
    reply would be mistaken for the continuation of the first.

    Found by the existing resume tests, whose fake transport cannot stream: the
    stream attempt consumed reply one and the fallback consumed reply two, so
    the first segment's text vanished and the continuation was returned in its
    place.
    """


#: Servers that have already proved they cannot stream, by base URL. One wasted
#: request per server per process, not one per page: a server without streaming
#: does not acquire it, and paying a failed attempt on every page of a
#: million-page run to re-learn that would be its own defect.
_NO_STREAM: set = set()


def stream_supported(url: str) -> bool:
    return _server_of(url) not in _NO_STREAM


def mark_no_stream(url: str) -> None:
    """Stop asking this server to stream."""
    _NO_STREAM.add(_server_of(url))


def _server_of(url: str) -> str:
    try:
        from urllib.parse import urlsplit
        p = urlsplit(str(url or ""))
        return f"{p.scheme}://{p.netloc}"
    except Exception:
        return str(url or "")


class StreamedReply:
    """What a streamed chat-completion produced, abort or not."""

    def __init__(self):
        self.text = ""
        self.finish_reason = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.logprobs: list = []
        self.aborted = False
        self.reason = ""


def stream_chat(post, url: str, payload: dict, headers: dict, timeout: float,
                guard: Optional[LoopGuard] = None,
                want_logprobs: bool = False) -> StreamedReply:
    """POST a chat completion with ``stream: true`` and stop early if it loops.

    ``post`` is the caller's own HTTP verb (``codai.api.pod_http.post``), so a
    pod reached by pinned TLS keeps its session and its certificate.

    Closing the response is what makes this worth doing: the server sees the
    client go away and cancels the sequence, so the tokens we did not wait for
    are tokens that were never generated. Reading to the end and discarding the
    tail would save nothing at all.

    ``stream_options.include_usage`` is REQUIRED, not decorative. Without it a
    streamed reply carries no token counts, and the resumption logic decides
    whether a page is complete by comparing `completion_tokens` against the cap
    -- so a missing count silently turns every page into "not at the cap",
    i.e. complete. A truncated page reported as whole is exactly the failure
    this engine has had before.
    """
    body = dict(payload)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    out = StreamedReply()
    #: Has the server begun answering? Once it has, this request is being paid
    #: for and must not be re-issued. See StreamUnsupported.
    consumed = [False]
    try:
        resp = post(url, json=body, headers=headers, timeout=timeout,
                    stream=True)
    except Exception as exc:
        raise StreamUnsupported(f"{type(exc).__name__}: {exc}")
    try:
        code = int(getattr(resp, "status_code", 500) or 500)
        if code >= 400:
            text = ""
            try:
                text = resp.text[:300]
            except Exception:
                pass
            if code < 500:
                # "I do not understand this request" -- which is what a server
                # without `stream_options`, or without streaming at all, says.
                # Nothing was generated, so the blocking path can answer it.
                raise StreamUnsupported(
                    f"HTTP {code} to a streaming request: {text}")
            # A 5xx is the server saying IT failed, not that the request was
            # wrong. Falling back would ask a broken server the same question
            # twice and report the second answer, hiding the first.
            #
            # OcrError is imported at module scope. A local `from ... import`
            # here made the name local to the WHOLE function, so the
            # `except OcrError` below raised UnboundLocalError before ever
            # matching -- and swallowed a mid-stream transport failure as a
            # retryable one, which is the single case that must not be retried.
            raise OcrError(f"OCR server {url} returned HTTP {code}: {text}",
                           status=502)
        lines = getattr(resp, "iter_lines", None)
        if not callable(lines):
            # A transport with no line iterator, or a server that ignored
            # `stream` and answered with a complete body. Either way the
            # generation HAS happened and is being paid for, so it is read
            # rather than discarded -- discarding it and re-asking would run
            # the page twice. The guard still gets the text, which costs
            # nothing and keeps one verdict for every page.
            return _from_whole_body(resp, guard, want_logprobs)
        for line in lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", "replace")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                # vLLM sends usage in a final chunk with an empty choices list.
                out.prompt_tokens = int(usage.get("prompt_tokens") or 0)
                out.completion_tokens = int(usage.get("completion_tokens") or 0)
            for choice in (chunk.get("choices") or []):
                if not isinstance(choice, dict):
                    continue
                if choice.get("finish_reason"):
                    out.finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta") or {}
                piece = delta.get("content") if isinstance(delta, dict) else None
                if piece is None:
                    piece = choice.get("text")
                if want_logprobs:
                    lp = choice.get("logprobs")
                    if isinstance(lp, dict):
                        got = lp.get("content")
                        if isinstance(got, list):
                            out.logprobs.extend(got)
                if not piece:
                    continue
                consumed[0] = True
                if guard is not None and guard.feed(piece):
                    out.text = guard.text
                    out.aborted = True
                    out.reason = guard.reason()
                    # finish_reason is deliberately NOT "length". A page
                    # abandoned for looping must not look to the resumption
                    # logic like a page that merely ran long, or it would be
                    # continued -- and continuing a loop is the thing this
                    # exists to stop.
                    out.finish_reason = "abort_repeat"
                    return out
        if guard is not None:
            out.text = guard.text
        if not consumed[0]:
            # 200, a line iterator, and not one usable chunk: the server
            # answered something that is not an SSE stream. Read it whole for
            # the same reason as above, rather than paying for it twice.
            try:
                return _from_whole_body(resp, guard, want_logprobs)
            except StreamUnsupported:
                raise
            except Exception:
                raise StreamUnsupported("the stream produced no content")
    except StreamUnsupported:
        raise
    except OcrError:
        raise
    except Exception as exc:
        # A body this cannot read. Safe to fall back ONLY if the server had not
        # begun answering; otherwise the generation has happened and re-running
        # it would bill the page twice.
        if consumed[0]:
            raise
        raise StreamUnsupported(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            resp.close()
        except Exception:
            pass
    return out


def _from_whole_body(resp, guard, want_logprobs) -> "StreamedReply":
    """Read a NON-streamed reply that arrived where a stream was expected.

    A server that ignores `stream: true` still generated the page, and that
    generation is already paid for. Parsing it here is the difference between
    one call per page and two -- and on an engine with a resumption queue, two
    calls means the second reply is mistaken for the continuation of the first.
    Found by the existing resume tests, whose transport cannot stream.
    """
    from codai.ocr.base import logprobs_of
    try:
        body = resp.json()
    except Exception as exc:
        raise StreamUnsupported(f"not a stream and not JSON: {exc}")
    if not isinstance(body, dict):
        raise StreamUnsupported("not a stream and not a JSON object")
    out = StreamedReply()
    out.text = _text_of_body(body)
    reason, used_in, used_out = _usage_of_body(body)
    out.finish_reason = reason or "stop"
    out.prompt_tokens = used_in
    out.completion_tokens = used_out
    if want_logprobs:
        out.logprobs = logprobs_of(body) or []
    if guard is not None and out.text:
        # One verdict per page either way. It saves nothing here -- the tokens
        # are already spent -- but a page judged on one path and not the other
        # would report differently for no reason the operator could see.
        guard.feed(out.text)
        if guard.tripped:
            out.aborted = True
            out.reason = guard.reason()
            out.finish_reason = "abort_repeat"
    return out


def _text_of_body(body: dict) -> str:
    choices = body.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    first = choices[0]
    msg = first.get("message")
    if isinstance(msg, dict):
        c = msg.get("content")
        if isinstance(c, list):
            return "".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c or ""
    return first.get("text") or ""


def _usage_of_body(body: dict):
    choices = body.get("choices") or []
    reason = ""
    if choices and isinstance(choices[0], dict):
        reason = str(choices[0].get("finish_reason") or "")
    usage = body.get("usage") or {}
    return (reason, int((usage or {}).get("prompt_tokens") or 0),
            int((usage or {}).get("completion_tokens") or 0))


# --------------------------------------------------------------------------- #
# the OpenAI-SDK shape, for an engine whose HTTP call we do not own
# --------------------------------------------------------------------------- #
#
# surya makes its own call -- `client.chat.completions.create(**kwargs)` in
# `surya/inference/backends/openai_client.py:115` -- so there is no request of
# ours to add `stream` to. What there IS, is the `client` object, handed in as a
# keyword to the function the OCR worker already wraps.
#
# So the client is wrapped instead: `create()` streams, the guard watches the
# text as it arrives, and the result is handed back in the shape the caller
# expects (`.choices[0].message.content`, `.usage.completion_tokens`,
# `.choices[0].logprobs.content`). Nothing in surya is patched beyond the hook
# we already own, and anything unexpected falls back to the ordinary blocking
# call -- a page read slowly is always better than a page not read.


class _Msg:
    def __init__(self, content):
        self.content = content


class _LP:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason, logprobs):
        self.message = _Msg(content)
        self.text = content
        self.finish_reason = finish_reason
        self.logprobs = _LP(logprobs) if logprobs else None


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = int(prompt_tokens or 0)
        self.completion_tokens = int(completion_tokens or 0)
        self.total_tokens = self.prompt_tokens + self.completion_tokens


class GuardedCompletion:
    """A chat-completion result in the OpenAI SDK's shape.

    ``aborted`` is ours and additive: a caller that does not know about it sees
    an ordinary (short) completion, which is exactly what a looping page is.
    """

    def __init__(self, content, finish_reason="stop", logprobs=None,
                 prompt_tokens=0, completion_tokens=0, aborted=False,
                 aborted_at=0):
        self.choices = [_Choice(content, finish_reason, logprobs)]
        self.usage = _Usage(prompt_tokens, completion_tokens)
        self.aborted = bool(aborted)
        self.aborted_at = int(aborted_at or 0)


def guarded_create(create, kwargs: dict, guard: "LoopGuard",
                   on_abort=None) -> "GuardedCompletion":
    """Run one chat completion as a stream, stopping it if it goes degenerate.

    ``create`` is the SDK's own ``chat.completions.create``. Returns a
    GuardedCompletion either way, so the caller cannot tell a guarded call from
    an ordinary one except by looking at ``aborted``.

    Closing the stream is the entire point. The server sees the client go away
    and cancels the sequence, so the tokens we did not wait for were never
    generated. Reading to the end and discarding the tail would save nothing.

    `stream_options.include_usage` is required rather than decorative: without
    it a streamed reply carries no token counts, and surya decides whether a
    page was cut off by comparing `completion_tokens` against the cap -- so a
    missing count silently reclassifies every page as complete, which is the
    failure mode this engine has had before.
    """
    ask = dict(kwargs)
    ask["stream"] = True
    ask["stream_options"] = {"include_usage": True}
    want_lp = bool(ask.get("logprobs"))
    text: list = []
    logprobs: list = []
    finish = ""
    pin = pout = 0
    stream = create(**ask)
    try:
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                pin = int(getattr(usage, "prompt_tokens", 0) or 0)
                pout = int(getattr(usage, "completion_tokens", 0) or 0)
            choices = getattr(chunk, "choices", None) or []
            for ch in choices:
                fr = getattr(ch, "finish_reason", None)
                if fr:
                    finish = str(fr)
                delta = getattr(ch, "delta", None)
                piece = getattr(delta, "content", None) if delta is not None else None
                if piece is None:
                    piece = getattr(ch, "text", None)
                if want_lp:
                    lp = getattr(ch, "logprobs", None)
                    got = getattr(lp, "content", None) if lp is not None else None
                    if got:
                        for c in got:
                            logprobs.append(
                                c.model_dump() if hasattr(c, "model_dump") else c)
                if not piece:
                    continue
                text.append(piece)
                if guard.feed(piece):
                    if on_abort is not None:
                        try:
                            on_abort(guard)
                        except Exception:
                            pass
                    return GuardedCompletion(
                        "".join(text),
                        # NOT "length". A page abandoned for looping must not
                        # look to the resumption logic like one that merely ran
                        # long, or it would be continued -- and continuing a
                        # loop is the thing this exists to stop.
                        finish_reason="abort_repeat",
                        logprobs=logprobs or None,
                        prompt_tokens=pin,
                        # Whatever usage had arrived, which on an abort is
                        # usually NOTHING: the usage chunk comes last and we
                        # closed the stream before it. So an aborted page
                        # under-reports its token cost, and the character count
                        # (`aborted_at`) is the honest measure of what it spent.
                        # Deliberate: a count inflated by a guess would be worse
                        # than one visibly absent, and `token_count: 0` also
                        # keeps the resumption logic from reading the page as
                        # one that ran long.
                        completion_tokens=pout or 0,
                        aborted=True, aborted_at=guard.tripped_at)
    finally:
        for name in ("close", "response"):
            try:
                obj = getattr(stream, name, None)
                if name == "close" and callable(obj):
                    obj()
                elif name == "response" and obj is not None:
                    obj.close()
            except Exception:
                pass
    return GuardedCompletion("".join(text), finish_reason=finish or "stop",
                             logprobs=logprobs or None,
                             prompt_tokens=pin, completion_tokens=pout)


def guarded_client(client, cfg=None, on_abort=None, **override):
    """`client`, with `chat.completions.create` watched by a LoopGuard.

    Returns the client unchanged when the guard is switched off, so an install
    that does not want this pays nothing -- not even a wrapper.

    A fresh guard per call, because the verdict is about one page.
    """
    opts = dict(settings(cfg))
    opts.update({k: v for k, v in override.items() if v is not None})
    if not opts.get("enabled"):
        return client
    inner = client.chat.completions.create

    def _create(**kwargs):
        if kwargs.get("stream"):
            # Someone already wants the stream itself; do not take it from them.
            return inner(**kwargs)
        guard = LoopGuard(min_chars=opts["min_chars"], step=opts["step"],
                          persistence=opts["persistence"])
        try:
            return guarded_create(inner, kwargs, guard, on_abort=on_abort)
        except Exception as exc:
            # A server that cannot stream, an SDK whose chunks are shaped
            # differently, a transport that failed halfway: fall back to the
            # ordinary call. A page read slowly beats a page not read, and the
            # post-hoc detector still judges the result.
            print(f"[ocr] early loop detection unavailable for this call "
                  f"({type(exc).__name__}: {exc}) — falling back to a blocking "
                  f"read", flush=True)
            return inner(**kwargs)

    class _Completions:
        create = staticmethod(_create)

    class _Chat:
        completions = _Completions()

    class _Guarded:
        """Everything the real client has, with one method replaced."""

        chat = _Chat()

        def __getattr__(self, name):
            return getattr(client, name)

    return _Guarded()
