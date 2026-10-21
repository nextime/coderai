# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""RunPod remote-GPU proxy backend.

A thin OpenAI proxy (mirroring :class:`codai.backends.vllm.VllmBackend`) that
forwards chat/completion requests to a model running on RunPod — either a
serverless endpoint (RunPod autoscales) or a coderai-managed pod pool. It uses
NO local VRAM and downloads NOTHING: the remote worker holds the weights.

Lifecycle + URL resolution live in :mod:`codai.api.runpod_worker`; account
settings (API key, endpoints, caps) in :class:`codai.config.RunpodConfig`.

Phase 2: the serverless path is live. Pods mode raises a clear error until the
pod pool lands in the next phase.
"""

from codai.api import pod_http
import asyncio
import threading
from typing import AsyncGenerator, Dict, List, Optional

from codai.backends.base import ModelBackend, merge_tool_call_deltas


class RemoteModelUnknown(RuntimeError):
    """The pod answered that it does not know the model we asked for."""


def _pod_detail(r) -> str:
    """What the pod SAID, not what requests says about the status line.

    ``raise_for_status()`` produces "404 Client Error: Not Found for url: …" and
    throws the body away — and the body is the whole message: it names the models
    the pod actually has. A pod that had never been told about a model therefore
    reached the caller as something indistinguishable from a routing fault.
    """
    try:
        body = r.json()
    except Exception:
        return (getattr(r, "text", "") or "").strip()[:400]
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:400]
        for key in ("detail", "message", "error"):
            if body.get(key):
                return str(body[key])[:400]
    return str(body)[:400]


class RunpodBackend(ModelBackend):
    """Thin OpenAI proxy to a model hosted on RunPod."""

    # RunPod requests run on REMOTE GPUs, so they must not count against the local
    # global concurrency gate. This class-level inflight is informational only.
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

    # RunPod models never occupy local VRAM.
    uses_local_vram = False

    #: Multipart (image) content is forwarded to the remote server VERBATIM.
    #
    # This is not a claim that every model behind this backend reads images — it is a
    # statement about who gets to decide. The front flattens multipart content to the
    # literal text "[image_url content]" unless the backend says otherwise, which for a
    # proxy is the one unacceptable answer: the page is dropped and the model answers
    # confidently about nothing. Forwarded, a text-only model's server returns a plain
    # 400 and a VLM's server reads the image. (Found by olmOCR in "model" mode against
    # a RunPod pod: every page arrived as that placeholder string.)
    supports_vision = True

    def __init__(self, cfg=None):
        if cfg is None:
            from codai.config import RunpodConfig
            cfg = RunpodConfig()
        self._acct = cfg
        self._model_id = "runpod"
        self._served_name = "runpod"
        self._url: Optional[str] = None          # serverless: fixed base URL
        self._headers: Dict[str, str] = {}
        self._ctx = 32768
        self._mcfg = None
        self._mode = "serverless"
        self._pool = None                         # pods: RunpodPodPool
        self._last_usage: Dict = {}
        self._last_logprobs: Optional[Dict] = None
        self._last_tool_calls: Optional[List[Dict]] = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def load_model(self, model_name: str, **kwargs) -> None:
        """Resolve the remote endpoint for ``model_name``. Downloads nothing."""
        from codai.api import runpod_worker
        if model_name:
            self._model_id = model_name
        block = runpod_worker.model_runpod_block(model_name)
        self._mcfg = runpod_worker.parse_model_runpod(block)
        # served name: explicit served_model wins, else the model's own name.
        self._served_name = self._mcfg.served_model or model_name or "runpod"
        if self._mcfg.ctx:
            self._ctx = self._mcfg.ctx
        _ctx = kwargs.get("n_ctx", kwargs.get("ctx"))
        if isinstance(_ctx, (list, tuple)):
            _ctx = _ctx[0] if _ctx else None
        try:
            _ctx = int(_ctx) if _ctx else 0
        except (TypeError, ValueError):
            _ctx = 0
        if _ctx > 0:
            self._ctx = _ctx

        self._mode = (self._mcfg.mode or "pods").lower()
        if self._mode == "serverless":
            self._url = runpod_worker.serverless_base_url(self._acct, self._mcfg)
            self._headers = runpod_worker.auth_headers(self._acct)
            print(f"[runpod] '{model_name}' -> serverless endpoint "
                  f"{self._mcfg.endpoint_id} ({self._url})", flush=True)
        else:
            # pods / auto: coderai-managed pool of remote GPU pods. (auto currently
            # behaves as pods; serverless-vs-pods auto-arbitration is a later refinement.)
            # A model may join a shared pool rather than rent its own card; the
            # request carries its own model name, so one multi-model pod serves
            # them all until it saturates and the pool grows.
            _pkey = runpod_worker.shared_pool_key(
                self._mcfg, model_name, runpod_worker._model_path_for(model_name))
            self._pool = runpod_worker.get_pod_pool(
                _pkey, self._acct, self._mcfg, self._served_name,
                shared=(_pkey != model_name))
            # A RunPod proxy URL is reachable by anyone who learns it, so the pod
            # is launched locked to a bearer token (generated when none is
            # configured) and every request we send must carry it.
            _key = getattr(self._pool, "api_key", "")
            self._headers = {"Authorization": f"Bearer {_key}"} if _key else {}
            # NOTE: that snapshot is only a starting value. A pool can CHANGE its
            # token after this point — it adopts the token of a pod already
            # running rather than renting a second one — so every send reads the
            # pool again through _auth(). See _auth().
            # Eagerly warm a pod only when min_pods >= 1; otherwise provision lazily
            # on the first request (min_pods=0 = scale-to-zero when idle).
            if self._mcfg.min_pods >= 1:
                self._pool.ensure_ready()
            print(f"[runpod] '{model_name}' -> managed pod pool "
                  f"(min={self._mcfg.min_pods}, max={self._mcfg.max_pods})", flush=True)

    def get_model_name(self) -> str:
        return self._model_id

    def get_context_size(self) -> int:
        return self._ctx

    def get_last_usage(self) -> dict:
        return dict(self._last_usage)

    def get_last_logprobs(self):
        """The pod's per-token log-probabilities for the last call, or None.

        See the same method on VllmBackend: an OCR pipeline turns these into the
        confidence it reports per line, and a proxy that swallows them makes the
        pipeline fall back to a flat 1.0 — a confidence that was never measured.
        """
        return self._last_logprobs

    def get_last_tool_calls(self):
        """The tool calls the POD parsed out of the last answer, or None.

        A pod serving an OpenAI API owns the chat template, so it is the side
        that recognises a tool call and reports it structurally: the message
        comes back with ``tool_calls`` set and ``content`` null. ``generate_chat``
        returns a string, so those calls used to end here — the caller received
        "" , found no call to parse in it, and answered the client with an empty
        assistant message while the pod had generated a perfectly good call.
        """
        return self._last_tool_calls

    def cleanup(self) -> None:
        # Serverless: nothing to tear down (RunPod owns it). Pods: the pool
        # reaper handles teardown (next phase).
        self._url = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _base(self) -> str:
        if not self._url:
            raise RuntimeError("RunPod endpoint not resolved")
        return self._url

    def _pool_key(self, pod) -> str:
        """The bearer for ONE pod, from the pool. Pods in a pool can carry
        different tokens, so the key belongs to the pod, not to the send."""
        pool = getattr(self, "_pool", None)
        getter = getattr(pool, "pod_key", None)
        if callable(getter):
            try:
                return getter(pod) or ""
            except Exception:
                pass
        return getattr(pool, "api_key", "") or ""

    def _acquire_endpoint(self):
        """Return (base_url, release_callable, bearer) for one request.

        The bearer comes with the address because it belongs to the POD that was
        just handed out: `self._last_pod` is one attribute shared by every
        concurrent request, so reading the key from it afterwards can pick up
        another request's pod. Serverless yields the fixed endpoint and "" (its
        key is the configured header, not a pod's)."""
        if self._mode == "serverless":
            return self._base(), (lambda: None), ""
        if not self._pool:
            raise RuntimeError("RunPod pod pool not initialised")
        pod, url = self._pool.acquire()
        # The pod runs a vLLM OpenAI server at /v1; its proxy URL has no path, so
        # add /v1 to match the serverless base (which already ends in /openai/v1).
        self._last_pod = pod
        self._ensure_adapters(url.rstrip("/"))
        return (url.rstrip("/") + "/v1", (lambda: self._pool.release(pod)),
                self._pool_key(pod))

    #: Pods already sent this model's adapters, by (pod url, model). A pod keeps
    #: what it is given and a new pod has a new URL, so this never goes stale.
    _ADAPTERS_SENT: set = set()

    def _ensure_adapters(self, root: str) -> None:
        """Put this model's LOCAL adapters on the pod before anything loads.

        A pod is seeded with the adapter's CONTENT ID, not its bytes — the id is
        computed here so that the pod can be told about it before it exists. It
        still has to receive the bytes, and until it does, the engine launches
        without the adapter and answers under its name with BASE weights.

        Before the first request, not after a failure: vLLM registers adapters
        when it STARTS, so bytes that arrive late cost a full reload of the base
        model to take effect. This is the difference between a pod that is right
        on boot and one that reloads a 27B to become right.
        """
        # getattr: sending adapters must never be the thing that breaks a
        # request. Anything missing here means there is nothing to send.
        name = getattr(self, "_model_id", "") or getattr(self, "_served_name", "")
        if not name:
            return
        try:
            from codai.models.manager import _model_entry_for
            from codai.models.text_loras import configured_specs, portable_specs
            entry = _model_entry_for(name) or {}
            if not entry:
                return
            specs = configured_specs(entry)
            _, needs_sending = portable_specs(specs)
            if not needs_sending:
                return
            # Keyed on WHICH adapter, not just its name. A retrain keeps the
            # name and changes the bytes, so a cache that remembered only the
            # name never sent the new ones and the pod went on serving the old
            # adapter — with nothing in the answer to say which one replied.
            key = (root, name, tuple(sorted(str(s.get("source") or "")
                                            for s in specs)))
            if key in self._ADAPTERS_SENT:
                return
            self._ADAPTERS_SENT.add(key)        # once per version, even if it fails
            # Bytes AND entry. Pushing the bytes alone leaves the pod's
            # catalogue naming the adapter it was SEEDED with, which after a
            # re-register is a content id nobody holds — so the engine drops
            # the adapter and serves the base under its name. Measured live:
            # seeded sha256:2e820c…, pushed sha256:b8eac366…, and the pod went
            # on looking for the first.
            from codai.api.remote_gateway import teach_pod
            pool = getattr(self, "_pool", None)
            teach_pod(root, getattr(pool, "api_key", "") if pool else "", name)
        except Exception as exc:
            print(f"[runpod] could not send {name!r}'s adapters to {root}: {exc}",
                  flush=True)

    def _auth(self, bearer: str = "") -> dict:
        """The bearer header for THIS send: `bearer` when the caller holds the
        key of the pod it acquired, else read from the pool now.

        Not the snapshot taken in load_model. A pool may replace its token
        between loading the model and sending a request: when a pod for this
        pool is already running — a sibling engine's, or ours from before a
        restart — the pool adopts that pod's token instead of renting a second
        card, and `pool.api_key` changes under us.

        A request that read the header before that happened carried the old
        token and the pod answered 401. Observed live: 64 OCR pages in flight
        when a pool adopted a running pod all failed with "Invalid API key",
        every one of them lost, while the same lease worked seconds later. The
        snapshot stays for serverless, which has no pool and no such event.

        And it is THIS POD's token, not the pool's: one pool can hold pods with
        different ones (an adopted pod keeps the token it was launched with), so
        a pool-wide key 401s every pod that does not share it. 2026-10-20: a
        restart adopted one pod and rented another in the same minute, and
        qwen38 answered 401 to every request until both pods were destroyed.
        """
        pool = getattr(self, "_pool", None)
        if pool is None:
            return dict(self._headers or {})
        key = (bearer or "") or self._pool_key(getattr(self, "_last_pod", None))
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _check(self, r) -> None:
        """raise_for_status(), keeping what the pod said and naming the one case
        we can do something about."""
        if r.ok:
            return
        from codai.api.remote_gateway import _is_unknown_model
        import requests
        detail = _pod_detail(r)
        if _is_unknown_model(r.status_code, (detail or "").encode("utf-8", "replace")):
            raise RemoteModelUnknown(detail)
        raise requests.HTTPError(
            f"the pod answered {r.status_code}: {detail}", response=r)

    def _teach(self, base: str, bearer: str = "") -> bool:
        """Tell this pod about our model after it said it has never heard of it.

        ``bearer`` is THIS pod's token, handed out with its address. Pods in one
        pool can carry different ones (see _pool_key), so reading the pool's own
        key here would 401 against any pod that does not share it -- and the
        request would then be lost to a 404 that teaching was meant to cure.
        """
        try:
            from codai.api.remote_gateway import teach_pod
        except Exception:
            return False
        name = self._model_id or self._served_name
        pool = getattr(self, "_pool", None)
        key = bearer or (getattr(pool, "api_key", "") if pool is not None else "")
        # The pool's base ends in /v1; teach_pod posts to <root>/v1/models/register.
        root = base[:-3].rstrip("/") if base.endswith("/v1") else base
        return bool(teach_pod(root, key, name))

    def openai_endpoint(self):
        """``(base_url ending /v1, api_key)`` for a client that will speak to this
        model over HTTP ITSELF, instead of going through us.

        The OCR VLM engines work that way: Surya and olmOCR attach to an OpenAI server
        and drive it with their own prompts, guided-decoding schemas and logprobs — all
        things a proxy hop would quietly drop. So they get the address and the key.

        Provisioning happens here (a cold pool rents and boots a pod before returning),
        but the in-flight slot is released at once: a long-lived attachment must not
        hold a scaling slot open, and the pod's own idle reaper decides when it goes.
        The address is therefore a lease — the caller re-resolves when it stops
        answering rather than treating it as permanent."""
        base, release, bearer = self._acquire_endpoint()
        try:
            auth = (self._auth(bearer) or {}).get("Authorization", "")
            key = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
            return base, key
        finally:
            release()

    def _unreachable(self) -> None:
        """The pod refused the connection: set it aside and let the caller retry.

        A pod terminated behind the pool's back (the test harness reaping by
        API, a reclaimed spot machine) stays "healthy" until the next timed
        probe; without this, one request in that window fails for nothing.

        `discard` takes it out of rotation and does NOT terminate it: a refused
        connection is as likely to mean the pod's listen backlog is full as that
        the pod is gone, and the pool re-probes it before deciding. Reporting a
        refusal here is therefore cheap and safe, which it was not when this
        call destroyed the card.
        """
        pod = getattr(self, "_last_pod", None)
        if pod is not None and self._pool is not None:
            self._pool.discard(pod, "connection refused")

    def _store_usage(self, usage: dict) -> None:
        if usage:
            self._last_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

    def _collect_logprobs(self, piece) -> None:
        """Accumulate a streamed chunk's logprobs into one response-shaped object."""
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

    def _chat_payload(self, messages, max_tokens, temperature, top_p, stop, stream,
                      logprobs=None, top_logprobs=None):
        payload = {
            "model": self._served_name,
            "messages": messages,
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
        return payload

    # ------------------------------------------------------------------ #
    # chat-level generation
    # ------------------------------------------------------------------ #
    def generate_chat(self, messages: List[Dict], max_tokens=None, temperature=0.7,
                      top_p=1.0, stop=None, tools=None, response_format=None,
                      logprobs=None, top_logprobs=None):
        import requests
        self._enter_request()
        self._last_logprobs = None
        self._last_tool_calls = None
        try:
            for attempt in (1, 2):
                base, release, bearer = self._acquire_endpoint()
                try:
                    payload = self._chat_payload(messages, max_tokens, temperature, top_p,
                                                 stop, False, logprobs=logprobs,
                                                 top_logprobs=top_logprobs)
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
                    r = pod_http.post(base + "/chat/completions", json=payload,
                                      headers=self._auth(bearer), timeout=3600)
                except requests.exceptions.ConnectionError:
                    # Nothing answered at all: the pod is gone. Once, on a
                    # fresh pod — a second refusal is a real outage.
                    release()
                    self._unreachable()
                    if attempt == 2:
                        raise
                    continue
                try:
                    try:
                        self._check(r)
                    except RemoteModelUnknown as unknown:
                        # The pod does not know this model. That is recoverable
                        # exactly once: teach it and ask the SAME pod again,
                        # rather than pay for a second card to serve a name the
                        # running one can hold.
                        if attempt == 2 or not self._teach(base, bearer):
                            raise
                        print(f"[runpod] pod did not know "
                              f"{self._served_name!r} ({unknown}) — taught it, "
                              f"retrying", flush=True)
                        continue
                    data = r.json()
                    self._store_usage(data.get("usage", {}))
                    choice = (data.get("choices") or [{}])[0]
                    self._last_logprobs = choice.get("logprobs")
                    message = choice.get("message") or {}
                    self._last_tool_calls = message.get("tool_calls") or None
                    return message.get("content") or ""
                finally:
                    release()
        finally:
            self._exit_request()

    async def generate_chat_stream(self, messages: List[Dict], max_tokens=None,
                                   temperature=0.7, top_p=1.0, stop=None, tools=None,
                                   response_format=None, logprobs=None,
                                   top_logprobs=None) -> AsyncGenerator[str, None]:
        import requests
        self._enter_request()
        self._last_logprobs = None
        self._last_tool_calls = None
        try:
            for attempt in (1, 2):
                # Pod acquire may provision (blocking) — do it off the event loop.
                base, release, bearer = await asyncio.to_thread(self._acquire_endpoint)
                payload = self._chat_payload(messages, max_tokens, temperature, top_p, stop,
                                             True, logprobs=logprobs,
                                             top_logprobs=top_logprobs)
                # A streamed request asks for structure as often as a buffered one,
                # and dropped it here even for json_object: the parameter was
                # accepted and then never looked at again.
                if response_format:
                    payload["response_format"] = response_format
                if tools:
                    payload["tools"] = tools
                sent = False
                try:
                    async for chunk in self._stream(base + "/chat/completions", payload,
                                                    delta_key="delta", bearer=bearer):
                        sent = True
                        yield chunk
                except requests.exceptions.ConnectionError:
                    # Refused before a byte arrived: the pod is gone. Drop it
                    # and go again on a fresh one, once.
                    self._unreachable()
                    if attempt == 2:
                        raise
                    continue
                except RemoteModelUnknown as unknown:
                    # The refusal happens before the first byte, so going again
                    # replays nothing — but check rather than assume, because a
                    # retry after output has started would duplicate it.
                    if sent or attempt == 2 or not await asyncio.to_thread(
                            self._teach, base, bearer):
                        raise
                    print(f"[runpod] pod did not know {self._served_name!r} "
                          f"({unknown}) — taught it, retrying", flush=True)
                    continue
                finally:
                    release()
                return
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
    async def _stream(self, url: str, payload: dict, delta_key: str,
                      bearer: str = "") -> AsyncGenerator[str, None]:
        import json
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()
        headers = self._auth(bearer)

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
            choice = (obj.get("choices") or [{}])[0]
            piece = choice.get(delta_key) or {}
            text = piece.get("content") or ""
            if text:
                loop.call_soon_threadsafe(queue.put_nowait, text)
            # A streamed tool call arrives as deltas and carries no content at
            # all, so it cannot travel in the text this generator yields.
            self._last_tool_calls = merge_tool_call_deltas(
                self._last_tool_calls, piece.get("tool_calls"))
            if obj.get("usage"):
                self._store_usage(obj["usage"])
            self._collect_logprobs(choice.get("logprobs"))
            return bool(choice.get("finish_reason"))

        def _worker():
            import requests
            try:
                with pod_http.post(url, json=payload, headers=headers,
                                   stream=True, timeout=3600) as r:
                    # Not raise_for_status(): it discards the body, which on a
                    # 404 is the pod telling us which models it actually has.
                    self._check(r)
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
