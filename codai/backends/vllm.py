# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""vLLM proxy backend.

vLLM serves an OpenAI-compatible HTTP API with continuous batching, so this backend is a
thin proxy (mirroring :class:`~codai.backends.ktransformers.KtransformersBackend`): it
forwards chat/completion requests to the managed vLLM subprocess (lifecycle in
:mod:`codai.api.vllm_worker`, which runs vLLM in an isolated venv).
"""

import asyncio
import threading
from typing import AsyncGenerator, Dict, List, Optional

from codai.backends.base import (ModelBackend, RemoteStreamError, call_sink,
                                 merge_tool_call_deltas, messages_for_the_wire,
                                 sink_get, sink_put)


class VllmBackend(ModelBackend):
    """Thin OpenAI proxy to a managed vLLM server."""

    #: Multipart (image) content goes to the server as it arrived — see the same flag
    #: on RunpodBackend for why a proxy must not flatten it.
    supports_vision = True

    _inflight = 0
    _inflight_lock = threading.Lock()

    @classmethod
    def _enter_request(cls):
        with cls._inflight_lock:
            cls._inflight += 1

    @classmethod
    def _exit_request(cls):
        with cls._inflight_lock:
            cls._inflight = max(0, cls._inflight - 1)

    @classmethod
    def any_request_active(cls) -> bool:
        with cls._inflight_lock:
            return cls._inflight > 0

    def __init__(self, cfg=None):
        if cfg is None:
            from codai.config import VllmConfig
            cfg = VllmConfig()
        self._cfg = cfg
        self._model_id = getattr(cfg, "model_id", "vllm") or "vllm"
        self._svc_key: Optional[str] = None
        self._url: Optional[str] = None
        self._ctx = int(getattr(cfg, "ctx", 32768) or 32768)
        self._served_name = self._model_id
        self._last_usage: Dict = {}
        self._last_logprobs: Optional[Dict] = None
        self._last_tool_calls: Optional[List[Dict]] = None
        self._last_reasoning: Optional[str] = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def load_model(self, model_name: str, **kwargs) -> None:
        from codai.api import vllm_worker
        if model_name:
            self._model_id = model_name
        _ctx = kwargs.get("n_ctx", kwargs.get("ctx"))
        if isinstance(_ctx, (list, tuple)):
            _ctx = _ctx[0] if _ctx else None
        try:
            _ctx = int(_ctx) if _ctx else 0
        except (TypeError, ValueError):
            _ctx = 0
        model_path = self._resolve_model_dir(model_name)
        # This model's own `vllm` block over the global template: parallelism,
        # the machines it spans, memory fraction, flags.
        from codai.backends.overrides import VLLM_FIELDS, apply, model_block
        _blk = model_block(model_name, "vllm") or {}
        self._cfg = apply(self._cfg, _blk, VLLM_FIELDS, "vllm")
        # The window this model is SERVED with, which is the only one worth
        # reporting: the model's own `vllm.ctx` when it sets one, else a
        # per-model `n_ctx`, else the global template. The generic kwarg must not
        # win over an explicit block, because it can be a node-wide fallback
        # nobody chose for this model -- `vulkan.n_ctx`, whose default is 2048 --
        # and a 2048 reported against a 32768 server makes auto-compaction crush
        # every conversation to a sixteenth of the window it has (measured on a
        # production install 2026-10-22).
        _blk_ctx = 0
        try:
            _blk_ctx = int((_blk or {}).get("ctx") or 0)
        except (TypeError, ValueError):
            _blk_ctx = 0
        if not _blk_ctx and _ctx > 0:
            # An explicit per-model n_ctx has to reach the LAUNCH too, or the
            # reported window and the served one disagree by design.
            try:
                self._cfg = apply(self._cfg, {"ctx": _ctx}, VLLM_FIELDS, "vllm")
            except Exception:
                pass
        try:
            self._ctx = int(getattr(self._cfg, "ctx", 0) or 0) or self._ctx
        except (TypeError, ValueError):
            pass
        # Per-model: serve each model-list entry under ITS OWN name (like nvidia/radeon),
        # not a single engine-wide model_id. Fall back to the config model_id only for the
        # single-model convenience (a model with no list entry).
        self._served_name = (model_name or getattr(self._cfg, "model_id", "") or "vllm")
        _resolved, self._svc_key = vllm_worker.resolve_service_key(self._cfg, model_path)
        self._url = vllm_worker.ensure_service(self._cfg, model_path=model_path,
                                               served_name=self._served_name)

    def _resolve_model_dir(self, model_name: str) -> Optional[str]:
        """Resolve the requested model to an HF model dir OR repo id for vLLM.

        vLLM's --model accepts both a local directory and a bare HF repo id
        (e.g. 'Qwen/Qwen2.5-0.5B-Instruct'), so we must NOT require a local dir."""
        import os
        cand = os.path.expanduser(model_name or "")
        if cand and os.path.isdir(cand):
            return os.path.abspath(cand)
        # Match the model list entry (by path/basename/alias/id) → use its path, which may
        # be a local dir OR a repo id. Prefer a local dir; else accept the repo id string.
        try:
            from codai.admin.routes import config_manager
            md = getattr(config_manager, "models_data", {}) or {}
            name_l = (model_name or "").strip().lower()
            for lst in md.values():
                if not isinstance(lst, list):
                    continue
                for m in lst:
                    if not isinstance(m, dict):
                        continue
                    path = str(m.get("path") or "").strip()
                    if not path:
                        continue
                    base = os.path.basename(path.rstrip("/")).lower()
                    cands = {path.lower(), base, str(m.get("alias") or "").lower(),
                             str(m.get("id") or "").lower()}
                    if name_l in cands:
                        exp = os.path.expanduser(path)
                        return os.path.abspath(exp) if os.path.isdir(exp) else path
        except Exception:
            pass
        # A bare repo id passed directly, or the single-model convenience path.
        if cand and ("/" in cand or cand):
            # model_name itself may be a repo id like "org/model".
            if "/" in cand and not cand.startswith(("/", "~", ".")):
                return cand
        mp = os.path.expanduser((getattr(self._cfg, "model_path", "") or "").strip())
        if mp and os.path.isdir(mp):
            return os.path.abspath(mp)
        return (getattr(self._cfg, "model_path", "") or "").strip() or None

    def get_model_name(self) -> str:
        return self._model_id

    def get_context_size(self) -> int:
        """The window this model is served with.

        The SERVED number when the server has already been asked for it (see
        context_length), because that is the only one that can refuse a request;
        the configured one otherwise. They disagree in practice, and a caller
        sizing a conversation against the larger of the two gets a 400.
        """
        served = getattr(self, "_ctx_len", None)
        if isinstance(served, int) and served > 0:
            return served
        return self._ctx

    def get_last_usage(self) -> dict:
        return dict(sink_get("usage", self._last_usage) or {})

    def get_last_logprobs(self):
        """The server's per-token log-probabilities for the last call, or None.

        ``generate_chat`` returns only the text, so a caller that asked for
        ``logprobs`` reads them here — the same shape as ``get_last_usage``, and
        for the same reason. Surya derives every line's confidence from these
        (mean token probability); with none it reports a flat 1.0, so dropping
        them does not merely lose a number, it invents a perfect one.
        """
        return sink_get("logprobs", self._last_logprobs)

    def get_last_tool_calls(self):
        """The tool calls the SERVER parsed out of the last answer, or None.

        vLLM applies the chat template and runs its own tool parser, so a
        recognised call comes back structurally — ``tool_calls`` set, ``content``
        null — and never appears in the text ``generate_chat`` returns. See the
        same method on RunpodBackend.
        """
        return sink_get("tool_calls", self._last_tool_calls)

    def get_last_reasoning(self):
        """The thought the SERVER separated out of the last answer, or None.

        It arrives in ``message.reasoning_content``, beside a content that has
        already had it removed, so the string ``generate_chat`` returns cannot
        carry it. See the same method on ModelBackend.
        """
        return sink_get("reasoning", self._last_reasoning)

    def cleanup(self) -> None:
        from codai.api import vllm_worker
        key = getattr(self, "_svc_key", None) or getattr(self._cfg, "model_id", self._model_id)
        vllm_worker.stop_service(key)
        self._url = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _base(self) -> str:
        if not self._url:
            raise RuntimeError("vLLM service not started")
        return self._url

    def context_length(self):
        """The context window the SERVER reports, in tokens, or None.

        Asked of the running server rather than read from config, because the
        two disagree and only one of them is true: an operator can set a
        per-model `ctx` that never reaches the launch arguments (measured on one
        install 2026-10-13 -- `ctx: 18432` configured, `--max-model-len 32768`
        actually served), and a client that trusted the config would plan around
        a window the model does not have.

        Cached: a served model cannot change shape under us, so one question per
        process is enough. Failure is cached as None and not retried hard, since
        not knowing is survivable and a probe on every listing is not.
        """
        if getattr(self, "_ctx_len", "unset") != "unset":
            return self._ctx_len
        self._ctx_len = None
        try:
            import requests
            r = requests.get(self._base() + "/v1/models", timeout=5)
            for m in (r.json().get("data") or []):
                if m.get("id") != getattr(self, "_served_name", None):
                    continue
                for k in ("max_model_len", "max_context_length", "context_length"):
                    if m.get(k):
                        self._ctx_len = int(m[k])
                        break
                break
        except Exception:
            pass
        return self._ctx_len

    def _store_usage(self, usage: dict) -> None:
        if usage:
            self._last_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            sink_put("usage", dict(self._last_usage))

    def _collect_logprobs(self, piece) -> None:
        """Accumulate a streamed chunk's logprobs into one response-shaped object.

        A stream carries them a few tokens at a time; a caller reading
        ``get_last_logprobs()`` after the stream wants the whole list, as the
        buffered path gives it.
        """
        if not isinstance(piece, dict):
            return
        content = piece.get("content")
        if not content:
            return
        if not isinstance(self._last_logprobs, dict):
            self._last_logprobs = {"content": []}
        self._last_logprobs.setdefault("content", []).extend(content)

    def format_messages(self, messages) -> str:
        parts = []
        for m in messages:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    @staticmethod
    def _raise_for_status(r, what="request"):
        """`raise_for_status`, but carrying what the server actually said.

        `requests` reports only "400 Client Error: Bad Request for url: ...",
        which names the symptom and nothing else. vLLM puts the reason -- which
        field it rejected, which template failed to render, by how much the
        context was overrun -- in the response BODY, and that body was being
        discarded on the way up. A whole diagnosis cycle was spent guessing at a
        400 whose explanation had already been sent and thrown away.
        """
        code = getattr(r, "status_code", None)
        if code is None:
            # An object that does not report a code is not something to crash
            # on inside the request path: fall back to whatever it does offer.
            r.raise_for_status()
            return
        if code < 400:
            return
        detail = ""
        try:
            body = r.json()
            detail = (body.get("message") or body.get("detail")
                      or body.get("error") or "")
            if isinstance(detail, dict):
                detail = detail.get("message") or str(detail)
            detail = str(detail) or str(body)
        except Exception:
            try:
                detail = (r.text or "").strip()
            except Exception:
                detail = ""
        if len(detail) > 2000:
            detail = detail[:2000] + " ...[truncated]"
        import requests as _rq
        # What the server SAID comes first, the status line and URL after it.
        # This error is wrapped again by every layer above -- a pod's coderai,
        # then the orchestrator -- and each prepends its own prefix, so putting
        # a 120-character proxy URL in front of the explanation pushes it past
        # wherever the client truncates. The numbers in "maximum context length
        # is 32768 ... you requested 35000" are the whole message, and a caller
        # that reads only a prefix was getting none of them.
        where = (f"{code} {getattr(r, 'reason', '')} from "
                 f"{getattr(r, 'url', '?')} on {what}")
        raise _rq.HTTPError(
            f"{detail} ({where})" if detail
            else f"{where} (the server sent no explanation)", response=r)

    #: Never overridable by a caller: these are the request's structure, not its
    #: sampling. Letting `passthrough` rewrite them would let a client retarget
    #: the server at another model or flip streaming under the response parser.
    _PAYLOAD_STRUCTURE = ("model", "messages", "stream")

    def _apply_passthrough(self, payload, passthrough):
        """Merge fields coderai does not itself understand into the request.

        A proxy that forwards only the fields it knows about decides, on the
        server's behalf, which of the server's own features exist. vLLM grows
        request fields faster than any allow-list here can be updated
        (`continue_final_message`, `add_generation_prompt`,
        `chat_template_kwargs`, `truncate_prompt_tokens`, guided-decoding
        parameters), and a dropped field fails in the worst possible way: the
        server answers something plausible for a DIFFERENT request, or rejects
        it with a 400 that names nothing.

        Measured 2026-10-13: OCR page resumption sends `continue_final_message`
        so the model carries on the exact string it was cut off in. It never
        arrived, so what reached vLLM was a conversation ending in an assistant
        turn with no instruction to continue it -- rejected 400, every resumed
        page lost, at every token cap low enough to need one.
        """
        if not passthrough:
            return payload
        for k, v in dict(passthrough).items():
            if k in self._PAYLOAD_STRUCTURE:
                continue
            payload[k] = v
        return payload

    def _chat_payload(self, messages, max_tokens, temperature, top_p, stop, stream,
                      logprobs=None, top_logprobs=None, passthrough=None):
        payload = {
            "model": getattr(self, "_served_name", self._model_id),
            "messages": messages_for_the_wire(messages),
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop
        if logprobs:
            payload["logprobs"] = True
            if top_logprobs is not None:
                payload["top_logprobs"] = int(top_logprobs)
            # Ask for the RAW VOCABULARY IDS rather than rendered strings, when the
            # operator has turned it on. This is the only lossless channel for a
            # model whose vocabulary is byte-level: measured on vLLM 0.27.1 serving
            # surya-ocr-2, `content` comes back with one U+FFFD per accented letter
            # and the per-token `bytes` come back EMPTY, so neither output carries
            # the character -- while the ids are intact and surya's own tokenizer
            # decodes them correctly. vLLM's own words for the field: "so that
            # tokens that are not JSON-encodable can be identified".
            #
            # Set here rather than relaunching the server with
            # --return-tokens-as-token-ids, because this is a per-request field and
            # a config change beats a pod rebuild. It only changes the `token`
            # STRING inside logprobs; nothing in coderai reads that except the OCR
            # worker's repair.
            if getattr(self._cfg, "return_token_ids", False):
                payload["return_tokens_as_token_ids"] = True
        # Last, so a caller that asks for a field coderai also computes gets the
        # value it asked for -- it is talking to this server deliberately.
        return self._apply_passthrough(payload, passthrough)

    # ------------------------------------------------------------------ #
    # chat-level generation
    # ------------------------------------------------------------------ #
    def generate_chat(self, messages: List[Dict], max_tokens=None, temperature=0.7,
                      top_p=1.0, stop=None, tools=None, response_format=None,
                      logprobs=None, top_logprobs=None, passthrough=None):
        import requests
        self._enter_request()
        self._last_logprobs = None
        self._last_tool_calls = None
        self._last_reasoning = None
        # The per-call store is cleared with them. A store is opened per request,
        # but a caller that reuses one context for several calls (a pipeline, a
        # test) would otherwise read the PREVIOUS call's tool call -- the exact
        # staleness the attribute resets exist to prevent.
        for _k in ("usage", "logprobs", "tool_calls", "reasoning"):
            sink_put(_k, None)
        try:
            payload = self._chat_payload(messages, max_tokens, temperature, top_p, stop, False,
                                         logprobs=logprobs, top_logprobs=top_logprobs,
                                         passthrough=passthrough)
            # Forward the structured-output request AS ASKED. This used to
            # keep only {"type":"json_object"}, so a json_schema — the strict
            # form every current client sends, Surya's guided decoding among
            # them — was dropped on the way out and the server answered free
            # text that the caller then failed to parse. The remote decides
            # what it supports; a proxy that edits the request decides for it.
            if response_format:
                payload["response_format"] = response_format
            if tools:
                payload["tools"] = tools
            r = requests.post(self._base() + "/v1/chat/completions", json=payload, timeout=3600)
            self._raise_for_status(r, "chat/completions")
            data = r.json()
            self._store_usage(data.get("usage", {}))
            choice = (data.get("choices") or [{}])[0]
            self._last_logprobs = choice.get("logprobs")
            message = choice.get("message") or {}
            self._last_tool_calls = message.get("tool_calls") or None
            self._last_reasoning = (message.get("reasoning_content")
                                    or message.get("reasoning") or None)
            # ...and in THIS call's store, which no concurrent request can reset
            # between the answer and the caller's read.
            sink_put("logprobs", self._last_logprobs)
            sink_put("tool_calls", self._last_tool_calls)
            sink_put("reasoning", self._last_reasoning)
            return message.get("content") or ""
        finally:
            self._exit_request()

    async def generate_chat_stream(self, messages: List[Dict], max_tokens=None,
                                   temperature=0.7, top_p=1.0, stop=None, tools=None,
                                   response_format=None, logprobs=None,
                                   top_logprobs=None,
                                   passthrough=None) -> AsyncGenerator[str, None]:
        self._enter_request()
        self._last_logprobs = None
        self._last_tool_calls = None
        self._last_reasoning = None
        # The per-call store is cleared with them. A store is opened per request,
        # but a caller that reuses one context for several calls (a pipeline, a
        # test) would otherwise read the PREVIOUS call's tool call -- the exact
        # staleness the attribute resets exist to prevent.
        for _k in ("usage", "logprobs", "tool_calls", "reasoning"):
            sink_put(_k, None)
        try:
            payload = self._chat_payload(messages, max_tokens, temperature, top_p, stop, True,
                                         logprobs=logprobs, top_logprobs=top_logprobs,
                                         passthrough=passthrough)
            # Same for the streamed path, which ignored it entirely.
            if response_format:
                payload["response_format"] = response_format
            if tools:
                payload["tools"] = tools
            async for chunk in self._stream(self._base() + "/v1/chat/completions", payload,
                                            delta_key="delta"):
                yield chunk
        finally:
            self._exit_request()

    # ------------------------------------------------------------------ #
    # plain completion (fallback path)
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, max_tokens=None, temperature: float = 0.7,
                 top_p: float = 1.0, stop=None, repeat_penalty: float = 1.0,
                 presence_penalty: float = 0.0, frequency_penalty: float = 0.0) -> str:
        return self.generate_chat([{"role": "user", "content": prompt}],
                                  max_tokens, temperature, top_p, stop)

    async def generate_stream(self, prompt: str, max_tokens=None, temperature: float = 0.7,
                              top_p: float = 1.0, stop=None, repeat_penalty: float = 1.0,
                              presence_penalty: float = 0.0,
                              frequency_penalty: float = 0.0) -> AsyncGenerator[str, None]:
        async for chunk in self.generate_chat_stream(
                [{"role": "user", "content": prompt}], max_tokens, temperature, top_p, stop):
            yield chunk

    # ------------------------------------------------------------------ #
    # SSE streaming
    # ------------------------------------------------------------------ #
    async def _stream(self, url: str, payload: dict, delta_key: str
                      ) -> AsyncGenerator[str, None]:
        import json
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()
        # Whether we are inside a reconstructed <think> block for THIS stream.
        _thought = {"open": False}
        # THIS stream's own accumulators, and the store for the call that opened
        # it -- captured here, in the caller's context, because the reader below
        # runs on a thread and a thread inherits no context variables. Keeping
        # them per stream also stops two concurrent streams on this one backend
        # object from appending their thoughts to each other.
        _acc = {"reasoning": None, "tool_calls": None, "logprobs": None}
        _sink = call_sink()

        def _publish(key, value):
            """Record one of the structured channels for this call."""
            _acc[key] = value
            if _sink is not None:
                _sink[key] = value
            setattr(self, "_last_" + key, value)      # the no-sink fallback

        def _handle_line(line: str) -> bool:
            if not line or not line.startswith("data:"):
                return False
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                return True
            try:
                obj = json.loads(data)
            except ValueError:
                return False
            # An error reported INSIDE the stream. A remote serving an OpenAI
            # API answers 200 and then says so in an SSE event (coderai does it
            # itself, deliberately, so a client treats it as an error instead of
            # as assistant text) -- so there is no status code to notice. This
            # event carries no `choices`, so reading only the delta dropped it
            # and the stream simply ended: the caller saw a clean finish with no
            # content and reported an empty answer. Measured 2026-10-22 on a
            # conversation past the pod's context window, which came back as a
            # successful, empty completion.
            # Present and not None, rather than truthy: `{"error": {}}` is a
            # reported failure (and must not pass as a success), while some
            # OpenAI-compatible servers put a literal `"error": null` in every
            # ordinary chunk, which must not fail the stream.
            err = obj.get("error", None) if isinstance(obj, dict) else None
            if err is not None:
                msg = err.get("message") if isinstance(err, dict) else err
                raise RemoteStreamError(str(msg or "") or "the remote reported "
                                        "an error with no message")
            choice = (obj.get("choices") or [{}])[0]
            piece = choice.get(delta_key) or {}
            text = piece.get("content") or ""
            # A remote that separates the thought sends it in its own delta
            # field, with the tags removed. This generator yields text, so the
            # tags go back on: the caller's reasoning gate then splits it out
            # again and emits it as reasoning AS IT ARRIVES, which is the whole
            # point of thinking — it should show while the model is thinking,
            # not land in one lump at the end.
            think = piece.get("reasoning_content") or piece.get("reasoning") or ""
            if think:
                _publish("reasoning", (_acc["reasoning"] or "") + think)
                if not _thought["open"]:
                    _thought["open"] = True
                    think = "<think>" + think
                loop.call_soon_threadsafe(queue.put_nowait, think)
            if text:
                if _thought["open"]:
                    _thought["open"] = False
                    text = "</think>" + text
                loop.call_soon_threadsafe(queue.put_nowait, text)
            # A streamed tool call arrives as deltas and carries no content at
            # all, so it cannot travel in the text this generator yields.
            _calls = merge_tool_call_deltas(_acc["tool_calls"],
                                            piece.get("tool_calls"))
            if _calls != _acc["tool_calls"]:
                _publish("tool_calls", _calls)
            if obj.get("usage"):
                self._store_usage(obj["usage"])
            _lp = choice.get("logprobs")
            if isinstance(_lp, dict) and _lp.get("content"):
                _cur = _acc["logprobs"] if isinstance(_acc["logprobs"], dict) else {}
                _merged = {**_cur, "content": list(_cur.get("content") or [])
                           + list(_lp["content"])}
                _publish("logprobs", _merged)
            return bool(choice.get("finish_reason"))

        def _worker():
            import requests
            try:
                with requests.post(url, json=payload, stream=True, timeout=3600) as r:
                    self._raise_for_status(r, "streamed chat/completions")
                    buf = b""
                    done = False
                    for bchunk in r.iter_content(chunk_size=8192):
                        if not bchunk:
                            continue
                        buf += bchunk
                        while b"\n" in buf:
                            raw, buf = buf.split(b"\n", 1)
                            if _handle_line(raw.decode("utf-8", "replace").strip()):
                                done = True
                                break
                        if done:
                            break
                    if not done and buf.strip():
                        _handle_line(buf.decode("utf-8", "replace").strip())
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        threading.Thread(target=_worker, daemon=True).start()
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item
