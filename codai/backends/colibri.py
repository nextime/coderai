# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""colibri (GLM-5.2) backend — the C engine, driven in-process.

Where :class:`~codai.backends.ds4.Ds4Backend` proxies HTTP to a managed
``ds4-server``, colibri ships no server we keep running — so this backend owns the
full gateway that colibri's ``openai_server.py`` would otherwise provide: it renders
the GLM-5.2 chat template and speaks the engine's stdin/stdout "mux" protocol via the
:class:`~codai.api.colibri_worker.MuxEngine` (whose process lifecycle lives in
:mod:`codai.api.colibri_worker`). The GLM-5.2 chat template below byte-matches
colibri's ``render_chat`` (which itself matches the model's ``chat_template.jinja``).

Tool/think parsing is handled the same way as the other backends — by
``ModelParserAdapter`` over the returned text.
"""

import asyncio
import json
import threading
from typing import AsyncGenerator, Dict, List, Optional

from codai.backends.base import ModelBackend

# GLM-5.2 chat-template markers (from colibri openai_server.py — the model expresses
# tool calls as ordinary text, so we render them into the prompt and let the parser
# read them back).
BOX_START, BOX_END = "<tool_call>", "</tool_call>"
TR_OPEN, TR_CLOSE = "<tool_response>", "</tool_response>"


def _content_text(content) -> str:
    """Flatten OpenAI message content (string or list of text parts) to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                t = part.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return str(content)


def render_chat(messages, enable_thinking: bool = False, reasoning_effort: Optional[str] = None,
                tools=None, tool_choice=None) -> str:
    """Render the text subset of the official GLM-5.2 chat template.

    Byte-matches colibri's ``openai_server.render_chat`` so the engine sees exactly
    the prompt it was trained on (the engine tokenizes the returned string itself).
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("`messages` must be a non-empty array.")
    prompt = ["[gMASK]<sop>"]
    if enable_thinking:
        effort = "High" if reasoning_effort == "high" else "Max"
        prompt.append(f"<|system|>Reasoning Effort: {effort}")
    forced = None
    if isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name") or tool_choice.get("name"))
        if forced:
            tools = [t for t in (tools or [])
                     if ((t.get("function", t) if isinstance(t, dict) else {}).get("name") == forced)]
    elif tool_choice == "none":
        tools = None
    if tools:
        prompt.append("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the "
                      "user query.\n\nYou are provided with function signatures within <tools></tools> "
                      "XML tags:\n<tools>\n")
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            clean = {k: v for k, v in fn.items() if k not in ("defer_loading", "strict")}
            prompt.append(json.dumps(clean, ensure_ascii=False) + "\n")
        prompt.append("</tools>\n\nFor each function call, output the function name and arguments "
                      "within the following XML format:\n<tool_call>{function-name}"
                      "<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
                      "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>")
        if forced:
            prompt.append(f"\n\nYou must call the function `{forced}`. Do not answer directly.")
        elif tool_choice == "required":
            prompt.append("\n\nYou must call one of the functions above. Do not answer directly.")
    prev_tool = False
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object.")
        role = message.get("role")
        if role in ("system", "developer"):
            prompt.append(f"<|system|>{_content_text(message.get('content'))}")
        elif role == "user":
            prompt.append(f"<|user|>{_content_text(message.get('content'))}")
        elif role == "assistant":
            raw = message.get("content")
            text = _content_text(raw) if raw is not None else ""
            prompt.append(f"<|assistant|><think></think>{text.strip()}")
            for tc in (message.get("tool_calls") or []):
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                prompt.append(BOX_START + (fn.get("name") or ""))
                for key, value in (args or {}).items():
                    prompt.append(f"<arg_key>{key}</arg_key><arg_value>"
                                  + (value if isinstance(value, str)
                                     else json.dumps(value, ensure_ascii=False)) + "</arg_value>")
                prompt.append(BOX_END)
        elif role == "tool":
            if not prev_tool:
                prompt.append("<|observation|>")
            prompt.append(TR_OPEN + _content_text(message.get("content")) + TR_CLOSE)
        else:
            raise ValueError(f"Unsupported message role: {role!r}.")
        prev_tool = (role == "tool")
    prompt.append("<|assistant|><think>" if enable_thinking else "<|assistant|><think></think>")
    return "".join(prompt)


class ColibriBackend(ModelBackend):
    """In-process backend that drives a managed colibri C engine (GLM-5.2)."""

    # Process-wide count of in-flight colibri requests (across all backend instances).
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
            from codai.config import ColibriConfig
            cfg = ColibriConfig()
        self._cfg = cfg
        self._model_id = getattr(cfg, "model_id", "glm-5.2-colibri") or "glm-5.2-colibri"
        self._svc_key: Optional[str] = None
        self._engine = None
        self._ctx = int(getattr(cfg, "ctx", 100000) or 100000)
        self._enable_thinking = False
        self._last_usage: Dict = {}

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def load_model(self, model_name: str, **kwargs) -> None:
        from codai.api import colibri_worker
        if model_name:
            self._model_id = model_name
        _ctx = kwargs.get("n_ctx", kwargs.get("ctx"))
        if isinstance(_ctx, (list, tuple)):
            _ctx = _ctx[0] if _ctx else None
        try:
            _ctx = int(_ctx) if _ctx else 0
        except (TypeError, ValueError):
            _ctx = 0
        if _ctx > 0:
            self._ctx = _ctx
        model_dir = self._resolve_container(model_name)
        overrides = self._colibri_overrides(model_name, model_dir)
        if overrides:
            import dataclasses
            try:
                self._cfg = dataclasses.replace(self._cfg, **overrides)
                print(f"[colibri] per-model overrides for '{model_name}': "
                      + ", ".join(f"{k}={v!r}" for k, v in overrides.items()), flush=True)
            except Exception as exc:
                print(f"[colibri] failed to apply per-model overrides: {exc}", flush=True)
        _resolved, self._svc_key = colibri_worker.resolve_service_key(self._cfg, model_dir)
        self._engine = colibri_worker.ensure_engine(
            self._cfg, model_dir=model_dir, ctx=(self._ctx or None))

    @staticmethod
    def _hf_snapshot_dir(repo_id: str) -> Optional[str]:
        """Local snapshot DIRECTORY of a fully-cached HF repo, or None. colibri needs
        the directory (SNAP=<dir>), not a file inside it, so we can't reuse the
        file-returning cache resolver — ask huggingface_hub for the snapshot root."""
        import os
        try:
            from huggingface_hub import snapshot_download
            hf_dir = None
            try:
                from codai.models.cache import get_all_cache_dirs
                hf_dir = (get_all_cache_dirs() or {}).get("huggingface") or None
            except Exception:
                hf_dir = None
            d = snapshot_download(repo_id, local_files_only=True, cache_dir=hf_dir)
            return os.path.abspath(d) if d and os.path.isdir(d) else None
        except Exception:
            return None

    @classmethod
    def _resolve_container(cls, model_name: str) -> Optional[str]:
        """Map a requested model name/alias/path to its GLM-5.2 container directory.

        The colibri model is a directory (int4 container), not a file. Resolution
        order: a local directory given directly → the model's models.json entry
        ``path`` when it is a local directory → that entry's ``path`` (or the name
        itself) resolved as an HF repo id to its local snapshot directory. The last
        case is what makes a model downloaded the normal way (an HF repo → snapshot
        dir) Just Work.
        """
        import os
        cand = os.path.expanduser(model_name or "")
        if cand and os.path.isdir(cand):
            return os.path.abspath(cand)

        entry_path = None
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
                    path = str(m.get("path") or "")
                    base = os.path.basename(path.rstrip("/"))
                    cands = {path.lower(), base.lower(), str(m.get("alias") or "").lower(),
                             str(m.get("id") or "").lower()}
                    if name_l and name_l in cands:
                        entry_path = path
                        if path and os.path.isdir(os.path.expanduser(path)):
                            return os.path.abspath(os.path.expanduser(path))
                        break
                if entry_path is not None:
                    break
        except Exception:
            pass

        # HF repo id (from the entry's path, else the requested name) → snapshot dir.
        for rid in (entry_path, model_name):
            rid = (rid or "").strip()
            if rid and "/" in rid and not os.path.isabs(rid):
                snap = cls._hf_snapshot_dir(rid)
                if snap:
                    return snap
        return None

    @staticmethod
    def _colibri_overrides(model_name: str, model_dir: Optional[str]) -> Dict:
        """Per-model colibri overrides from the model's own models.json entry.

        Optional ``colibri`` block fields: ``kv_slots``, ``cap``, ``cuda_expert_gb``,
        ``extra_args``, ``extra_env``. Unset/blank fields inherit the global config.
        """
        import os
        out: Dict = {}
        try:
            from codai.admin.routes import config_manager
            md = getattr(config_manager, "models_data", {}) or {}
            target = os.path.basename(os.path.expanduser(model_dir or "").rstrip("/")) or None
            name_l = (model_name or "").strip().lower()
            entry = None
            for lst in md.values():
                if not isinstance(lst, list):
                    continue
                for m in lst:
                    if not isinstance(m, dict):
                        continue
                    path = str(m.get("path") or m.get("id") or "")
                    base = os.path.basename(path.rstrip("/"))
                    cands = {path.lower(), base.lower(), str(m.get("alias") or "").lower()}
                    if (target and base == target) or (name_l and name_l in cands):
                        entry = m
                        break
                if entry:
                    break
            co = entry.get("colibri") if entry and isinstance(entry.get("colibri"), dict) else None
            if not co:
                return out
            for k in ("kv_slots", "cap"):
                v = co.get(k)
                if v not in (None, "", 0, "0"):
                    try:
                        out[k] = max(1, int(v))
                    except (TypeError, ValueError):
                        pass
            for k in ("cuda_expert_gb", "extra_args", "extra_env"):
                v = co.get(k)
                if v and str(v).strip():
                    out[k] = str(v).strip()
        except Exception:
            pass
        return out

    def get_model_name(self) -> str:
        return self._model_id

    def get_context_size(self) -> int:
        return self._ctx

    def get_last_usage(self) -> dict:
        return dict(self._last_usage)

    def cleanup(self) -> None:
        from codai.api import colibri_worker
        key = getattr(self, "_svc_key", None) or getattr(self._cfg, "model_id", self._model_id)
        colibri_worker.stop_service(key)
        self._engine = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _need_engine(self):
        if self._engine is None or not self._engine.is_alive():
            # Re-establish (evicted or died) so a stale handle self-heals.
            self.load_model(self._model_id)
        return self._engine

    def _store_usage(self, stats: dict) -> None:
        if stats:
            pt = int(stats.get("prompt_tokens", 0) or 0)
            ct = int(stats.get("completion_tokens", 0) or 0)
            self._last_usage = {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
            }

    def format_messages(self, messages) -> str:
        return render_chat(messages, enable_thinking=self._enable_thinking)

    # ------------------------------------------------------------------ #
    # chat-level generation (preferred by the manager)
    # ------------------------------------------------------------------ #
    def generate_chat(self, messages: List[Dict], max_tokens=None, temperature=0.7,
                      top_p=1.0, stop=None, tools=None, response_format=None):
        self._enter_request()
        try:
            engine = self._need_engine()
            prompt = render_chat(messages, enable_thinking=self._enable_thinking, tools=tools)
            chunks: List[str] = []
            stats = engine.run(prompt, int(max_tokens or 1024), float(temperature),
                               float(top_p), on_text=chunks.append)
            self._store_usage(stats)
            return "".join(chunks)
        finally:
            self._exit_request()

    async def generate_chat_stream(self, messages: List[Dict], max_tokens=None,
                                   temperature=0.7, top_p=1.0, stop=None, tools=None,
                                   response_format=None) -> AsyncGenerator[str, None]:
        self._enter_request()
        try:
            engine = self._need_engine()
            prompt = render_chat(messages, enable_thinking=self._enable_thinking, tools=tools)
            async for chunk in self._stream(engine, prompt, int(max_tokens or 1024),
                                            float(temperature), float(top_p)):
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
    # SSE streaming: the engine's blocking run() streams tokens to a callback on a
    # worker thread; bridge them to the event loop through an asyncio.Queue.
    # ------------------------------------------------------------------ #
    async def _stream(self, engine, prompt: str, max_tokens: int, temperature: float,
                      top_p: float) -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        out_queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()

        def _on_text(text: str):
            if text:
                loop.call_soon_threadsafe(out_queue.put_nowait, text)

        def _worker():
            try:
                stats = engine.run(prompt, max_tokens, temperature, top_p, on_text=_on_text)
                self._store_usage(stats)
            except Exception as exc:  # surface to the consumer
                loop.call_soon_threadsafe(out_queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(out_queue.put_nowait, _SENTINEL)

        threading.Thread(target=_worker, daemon=True).start()
        while True:
            item = await out_queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item
