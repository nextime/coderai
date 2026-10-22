# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Base classes for model backends."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional


class ModelBackend(ABC):
    """Abstract base class for model backends."""
    
    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                 temperature: float = 0.7, top_p: float = 1.0,
                 stop: Optional[List[str]] = None,
                 repeat_penalty: float = 1.0,
                 presence_penalty: float = 0.0,
                 frequency_penalty: float = 0.0) -> str:
        """Generate text non-streaming."""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, max_tokens: Optional[int] = None,
                        temperature: float = 0.7, top_p: float = 1.0,
                        stop: Optional[List[str]] = None,
                        repeat_penalty: float = 1.0,
                        presence_penalty: float = 0.0,
                        frequency_penalty: float = 0.0) -> AsyncGenerator[str, None]:
        """Generate text in streaming fashion."""
        pass
    
    @abstractmethod
    def format_messages(self, messages) -> str:
        """Format messages into a prompt string."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the loaded model name."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
    
    def get_context_size(self) -> int:
        """Return the model's context window size."""
        return 2048  # Default fallback

    def get_last_tool_calls(self):
        """The tool calls the backend itself reported for the last call, or None.

        A backend that generates text has none: the call is still inside the
        generated string, and the caller's parser digs it out. A *proxy* backend
        is the opposite case — the remote server owns the chat template, so it
        parses the call and answers with structured ``tool_calls`` and a null
        ``content``. The text contract of ``generate_chat`` cannot carry that, so
        the proxy leaves them here, next to ``get_last_usage``.
        """
        return None


def messages_for_the_wire(messages):
    """``messages`` with every ``tool_call.function.arguments`` as a JSON STRING.

    The OpenAI schema says that field is a string, and a server validating
    against it refuses an object outright::

        400 Bad Request ... {'type': 'string_type',
          'loc': (..., 'ChatCompletionMessageFunctionToolCallParam',
                  'function', 'arguments')}

    But coderai hands its backends a MAPPING on purpose: several GGUF chat
    templates render arguments with ``|items``, and llama.cpp raises "Can only
    get item pairs from a mapping" on a string (see
    codai.api.text._normalize_tool_call_arguments). Both are right for their own
    side, so the conversion belongs here, at the edge where a request stops
    being a template's input and becomes another server's HTTP body.

    Only reached once a client replays an assistant turn that HAS tool calls,
    which is why it stayed invisible while those calls were being dropped
    (v0.4.60). Copies rather than mutates: the caller retries with the same list.
    """
    import json
    if not messages:
        return messages if isinstance(messages, list) else []
    out = []
    changed = False
    for m in messages:
        if not isinstance(m, dict) or not m.get("tool_calls"):
            out.append(m)
            continue
        calls = []
        for tc in (m.get("tool_calls") or []):
            if hasattr(tc, "model_dump"):
                tc = tc.model_dump()
            if not isinstance(tc, dict):
                calls.append(tc)
                continue
            fn = tc.get("function")
            if isinstance(fn, dict) and not isinstance(fn.get("arguments"), str):
                args = fn.get("arguments")
                fn = dict(fn)
                try:
                    # None/absent means "no arguments", which on the wire is {}
                    # — not the string "null", which a server would then try to
                    # read as an argument object and reject.
                    fn["arguments"] = "{}" if args is None else json.dumps(args)
                except Exception:
                    fn["arguments"] = "{}"
                tc = dict(tc)
                tc["function"] = fn
                changed = True
            calls.append(tc)
        if changed:
            m = dict(m)
            m["tool_calls"] = calls
        out.append(m)
    # Nothing to convert — hand back the list that came in. Every request goes
    # through here, OCR's page-per-request traffic included, and a conversation
    # with no tool calls is the common case.
    return out if changed else messages


def merge_tool_call_deltas(acc: Optional[List[dict]], delta) -> Optional[List[dict]]:
    """Fold one streamed ``delta.tool_calls`` fragment into ``acc``.

    OpenAI streams a tool call in pieces: the first fragment carries ``id`` and
    the function name, later ones append to ``function.arguments``, and each is
    tagged with the ``index`` of the call it belongs to. Concatenating the
    arguments of the wrong call produces valid JSON for a call nobody made, so
    fragments are keyed by that index and never by arrival order.
    """
    if not delta:
        return acc
    out = list(acc or [])
    for frag in delta:
        if not isinstance(frag, dict):
            continue
        # Absent index means a single call, which is what one-call servers send.
        idx = frag.get("index")
        idx = 0 if idx is None else int(idx)
        while len(out) <= idx:
            out.append({"id": "", "type": "function",
                        "function": {"name": "", "arguments": ""}})
        cur = out[idx]
        if frag.get("id"):
            cur["id"] = frag["id"]
        if frag.get("type"):
            cur["type"] = frag["type"]
        fn = frag.get("function") or {}
        if fn.get("name"):
            cur["function"]["name"] = fn["name"]
        piece = fn.get("arguments")
        if piece:
            cur["function"]["arguments"] += piece
    return out or None