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

"""Pydantic models for API."""

import time
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict] = None


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    cache_control: Optional[Dict] = None  # OpenAI-style: {"type": "ephemeral"}
    # A previous turn's thinking, sent back by a client that kept it. Qwen3's own
    # chat template reads `message.reasoning_content` and re-renders it inside
    # <think> tags, so dropping it changes the prompt the model is continuing
    # from. Pydantic ignores unknown fields, so it was being dropped silently.
    reasoning_content: Optional[str] = None
    reasoning: Optional[str] = None
    
    @field_validator('content', mode='before')
    @classmethod
    def convert_content_array_to_string(cls, v):
        """Flatten a TEXT-ONLY multipart content array to a string for clients that
        send text as parts (e.g. KiloCode). Multimodal arrays — anything with an
        ``image_url`` (or other non-text part) — are PRESERVED as a list so the
        vision backend (mmproj/CLIP) actually receives the image. Flattening those
        replaced the image with a "[image_url content]" placeholder, so a vision
        model saw only text and answered "no image attached"."""
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            # Preserve the array untouched when it carries any non-text part — the
            # downstream chat path (vulkan _fold_system + MTMDChatHandler) handles
            # list content and feeds image_url parts to the vision projector.
            has_non_text = any(
                isinstance(item, dict) and item.get('type') not in (None, 'text')
                for item in v)
            if has_non_text:
                return v
            # Pure text parts → join to a single string (the KiloCode case).
            parts = []
            for item in v:
                if isinstance(item, dict):
                    if item.get('type') == 'text' and 'text' in item:
                        parts.append(item['text'])
                    else:
                        parts.append(f"[{item.get('type', 'unknown')} content]")
                else:
                    parts.append(str(item))
            return '\n'.join(parts)
        return str(v)


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Text/chat model id to use.")
    messages: List[ChatMessage] = Field(..., description="Conversation messages (roles: system/user/assistant/tool). Content may include text and image parts for vision models.")
    temperature: float = Field(0.7, description="Sampling temperature; higher = more random.")
    top_p: float = Field(1.0, description="Nucleus sampling probability mass.")
    n: int = Field(1, description="Number of completions to generate.")
    max_tokens: Optional[int] = Field(None, description="Max tokens to generate (model default if omitted).")
    max_completion_tokens: Optional[int] = Field(None, description="Newer OpenAI alias for max_tokens; used when max_tokens is omitted.")
    stream: bool = Field(False, description="Stream the response as Server-Sent Events.")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequence(s) that end generation.")
    presence_penalty: float = Field(0.0, description="Penalize tokens already present (encourages new topics).")
    frequency_penalty: float = Field(0.0, description="Penalize frequent tokens (reduces repetition).")
    repeat_penalty: float = Field(1.0, description="llama.cpp repetition penalty.")
    tools: Optional[List[Tool]] = Field(None, description="Tool/function definitions the model may call.")
    tool_choice: Optional[Union[str, Dict]] = Field("auto", description="Tool selection: 'auto', 'none', or a specific tool.")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility.")
    logprobs: Optional[bool] = Field(None, description="Return token log-probabilities (if supported).")
    top_logprobs: Optional[int] = Field(None, description="Number of top log-probs to return per token.")
    response_format: Optional[Dict] = Field(None, description="Structured-output format, e.g. {'type': 'json_object'}.")
    user: Optional[str] = Field(None, description="Opaque end-user identifier (passthrough).")
    enable_thinking: Optional[bool] = Field(None, description="Thinking/reasoning mode. Unset leaves the model's own default, which for a thinking model means it thinks; false asks it not to think at all (forwarded to the chat template); true asks for it explicitly.")
    reasoning: Optional[Dict] = Field(None, description="Reasoning controls. OpenRouter-style; set {'exclude': true} to drop the model's thinking from the response.")
    reasoning_effort: Optional[str] = Field(None, description="OpenAI-style reasoning effort ('low'|'medium'|'high'); 'none'/'off' suppresses reasoning output.")
    suppress_reasoning: Optional[bool] = Field(None, description="Drop the model's reasoning/thinking from the response instead of returning it as a separate field.")

    model_config = ConfigDict(extra="allow")  # Allow extra fields to prevent 422 errors

    @model_validator(mode="after")
    def _alias_max_completion_tokens(self):
        """Accept the newer OpenAI ``max_completion_tokens`` as an alias for
        ``max_tokens``. ``max_tokens`` wins when both are sent."""
        if self.max_tokens is None and self.max_completion_tokens is not None:
            self.max_tokens = self.max_completion_tokens
        return self


class CompletionRequest(BaseModel):
    model: str = Field(..., description="Text model id to use.")
    prompt: Union[str, List[str]] = Field(..., description="Prompt text (or list of prompts) to complete.")
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repeat_penalty: float = 1.0
    # Extra fields that clients may send but we ignore
    seed: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    best_of: Optional[int] = None
    echo: Optional[bool] = None
    user: Optional[str] = None
    
    model_config = ConfigDict(extra="allow")  # Allow extra fields to prevent 422 errors


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "huggingface"
    type: Optional[str] = None          # e.g. "text", "image", "video", "audio", "tts", "vision", "embedding"
    capabilities: Optional[List[str]] = None  # list of capability strings
    backend: Optional[str] = None
    model_path: Optional[str] = None
    server_path: Optional[str] = None
    alias: Optional[str] = None
    port: Optional[int] = None
    gpu_device: Optional[int] = None
    load_mode: Optional[str] = None
    #: Why this model cannot currently serve, when that is knowable without
    #: loading it -- today only a missing python package. Two models were listed
    #: here as available for as long as they existed and answered every request
    #: with an import error, because the catalogue never asked. None means
    #: nothing known to be wrong, which includes "not checked".
    unavailable_reason: Optional[str] = None
    # Languages this model handles (empty/None = unknown, or accepts any):
    # for speech-to-text the ones it transcribes, for text-to-speech the ones it
    # can speak. `supports_translation` is STT-only (source→target_language,
    # e.g. NVIDIA Canary).
    languages: Optional[List[str]] = None
    supports_translation: Optional[bool] = None
    # Text-to-speech: the built-in voices, each with whatever is genuinely known
    # about it (id, name, gender, languages). Published so a client can offer a
    # voice picker without loading the model to ask it.
    voices: Optional[List[Dict]] = None
    speaker_diarization: Optional[bool] = None
    # The context window this model is being SERVED with, in tokens.
    #
    # Named as vLLM names it, so a client written against a vLLM server reads it
    # without changes. It is published because a client cannot otherwise plan:
    # anything that sends a growing conversation -- a continued generation, a
    # resumed OCR page, a long tool loop -- needs to know when the window runs
    # out, and the alternative is to find out from a 400 after the work is done.
    # coderai's own OCR page resumption was the first casualty: it probed
    # /v1/models for this field, coderai did not publish it, so the resumption
    # budget could not see the ceiling it was about to hit.
    #
    # The number is what the model is served with, not what the checkpoint
    # advertises: an operator may serve a shorter window deliberately. None means
    # "not established here", which a client must treat as unknown rather than
    # unlimited.
    max_model_len: Optional[int] = None
    # Video generation: what this model can actually emit — native frame rate, the
    # legal frame-count grid, how long ONE render may be and whether continuation is
    # native. Published so a client can PLAN around the model (how long a scene can
    # be, how many scenes fill a minute) instead of hardcoding a table per family.
    # See codai/models/video_geometry.py.
    video: Optional[Dict] = None
    # What this model recommends for itself, from its card: steps, guidance
    # scale, sampler. Published because a client cannot otherwise know: a
    # distilled model wants 8 steps at CFG 1 and its non-distilled sibling
    # wants 30 at CFG 4, and a client with no table of its own gets 30 from
    # the 8-step model -- four times the cost for a worse picture. The server
    # applies these when a request omits them; this field is so a UI can SHOW
    # them.
    defaults: Optional[Dict] = None
    # Which directions this model generates in: "t2v", "i2v", "v2v" for video,
    # "t2i", "i2i" for images. A caller otherwise discovers that a model has no
    # image-to-video path by sending one and getting a 400.
    modes: Optional[List[str]] = None
    # The licence the weights are under, as the operator recorded it. Some
    # models are free only below a revenue threshold or require a content
    # filter, and that is a fact about the model which belongs beside it
    # rather than in a document somewhere.
    license: Optional[str] = None
    # Whether this model is usable RIGHT NOW. Used by the built-in endpoints
    # whose models are not model-manager models (moderation): an unavailable
    # check has to be visible before content is sent for it, and an omitted
    # entry would read as "this server does not do that at all".
    available: Optional[bool] = None
    # The ONE name that identifies this model, and the other names it also
    # answers to.
    #
    # A model is published under more than one id on purpose: its alias, its
    # repo id or file path, and the bare filename. That is how an alias can
    # stand for a round-robin group, and how "image:name"-style ids coexist
    # with the underlying checkpoint. The cost is that a client walking
    # /v1/models sees 206 ids for 119 cards and has no way to tell which of
    # them are the same model -- so a capability sweep tests several of them
    # three times and reports a catalogue that grew.
    #
    # `canonical_id` is the same string on every id belonging to one model, so
    # grouping by it is enough. `alias_of` is set only on the NON-canonical
    # ids, so skipping the entries that have it leaves exactly one per model.
    # `aliases`, on the canonical entry, lists the others.
    #
    # Two cards that share a checkpoint but differ in configuration are NOT the
    # same model and never share a canonical_id: they are separate entries with
    # separate settings, and a client that wants both must be able to see both.
    canonical_id: Optional[str] = None
    alias_of: Optional[str] = None
    aliases: Optional[List[str]] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]
