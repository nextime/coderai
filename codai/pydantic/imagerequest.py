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

"""Pydantic models for image generation API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class LoraConfig(BaseModel):
    """A LoRA adapter for one image request. Weights may be supplied (resolved
    server-side, in priority) via `id` ("name:<registered>" or "sha256:<hex>"),
    inline `file`/`data` base64, a `url`, or the legacy `model`/`path` (local path
    / HF id) — so a remote client needn't share the server's filesystem."""
    model: Optional[str] = Field(None, description="Legacy: local path or HF id of the weights (shared-filesystem only).")
    path: Optional[str] = Field(None, description="Alias of `model` — local path to the .safetensors weights.")
    id: Optional[str] = Field(None, description='Registry/blob reference: "name:<registered-lora>" or "sha256:<hex>" (from /v1/loras/upload).')
    url: Optional[str] = Field(None, description="HTTP(S) URL the server downloads and caches.")
    file: Optional[str] = Field(None, description="Base64 of the .safetensors file (or a data: URI). Sent inline so no shared filesystem is needed.")
    data: Optional[str] = Field(None, description="Alias of `file` — inline base64 weights.")
    weight: float = Field(1.0, description="Adapter strength / scale.")
    name: Optional[str] = Field(None, description="Optional adapter name.")
    model_config = ConfigDict(extra="allow")


class ImageGenerationRequest(BaseModel):
    #: The caller's own name for this request, so it can poll
    #: ``/v1/images/progress?request_id=...`` for its OWN steps instead of a
    #: single global slot shared with everyone generating at the same time.
    #: An ``X-Request-Id`` header does the same; the body wins when both are
    #: given. Echoed back in the progress document.
    request_id: Optional[str] = None
    model: str = Field(..., description="Model id to generate with (must be a configured image model).")
    prompt: str = Field(..., description="Text prompt describing the image.")
    n: int = Field(1, description="Number of images to generate.")
    size: Optional[str] = Field("1024x1024", description="Output size as 'WIDTHxHEIGHT'.")
    steps: Optional[int] = Field(None, description="Denoising steps (model/acceleration default if omitted).")
    guidance_scale: Optional[float] = Field(None, description="Classifier-free guidance scale (model/acceleration default if omitted).")
    quality: Optional[str] = Field("standard", description="Quality hint: 'standard' or 'hd'.")
    style: Optional[str] = Field(None, description="Optional style hint passed through to the model.")
    response_format: Optional[str] = Field("url", description="How to return the result: 'url' or 'b64_json'.")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility.")
    user: Optional[str] = Field(None, description="Opaque end-user identifier (passthrough).")
    disable_safety_checker: Optional[bool] = Field(False, description=(
        "Null out the diffusers safety_checker so uncensored fine-tunes are not blocked. "
        "Only affects SD 1.x/2.x (SDXL/Flux ship no checker)."))
    negative_prompt: Optional[str] = Field(None, description="What to avoid in the output.")
    clip_skip: Optional[int] = Field(None, description=(
        "Take the text embedding this many CLIP layers from the end instead of the last. "
        "The booru-tag SDXL fine-tune families (Pony, Illustrious, NoobAI) are trained at 2. "
        "Omitted uses the model's configured default, then the pipeline's own."))
    prompt_prefix: Optional[str] = Field(None, description=(
        "Text prepended to the prompt, comma-joined. Omitted uses the model's own "
        "prompt_prefix (how a model fine-tuned on a quality ladder gets it on every "
        "request); an empty string switches it off for this call. Applied at most "
        "once -- a prompt that already starts with it is left alone."))
    negative_prompt_prefix: Optional[str] = Field(None, description=(
        "The same, for the negative prompt (e.g. 'score_4, score_5, score_6')."))
    embeddings: Optional[List[str]] = Field(None, description=(
        "Textual-inversion embeddings to load before generating, so their trigger "
        "words work in the prompt (e.g. [\"easynegative\"]). A bare name is looked "
        "up in image.embeddings_dir; a path or a HuggingFace repo id also works, and "
        "'ref:token' renames the trigger word. Everything in embeddings_dir, plus the "
        "model's own textual_inversion list, is loaded whether this is set or not."))
    prompt_weighting: Optional[bool] = Field(None, description=(
        "Parse the A1111 prompt grammar: '(tag:1.3)' to emphasise, '[tag]' to attenuate, "
        "'(tag)' for 1.1, nesting multiplies, '\\\\(' escapes, and BREAK starts a new "
        "75-token window. False keeps brackets literal. Either way the prompt is encoded "
        "in as many windows as it needs, so nothing is truncated at 77 tokens."))

    # Per-request component overrides
    sampler: Optional[str] = Field(None, description=(
        "Scheduler for this request (euler, euler_a, dpmpp_2m, ddim, unipc, "
        "lcm, flow_euler, ...). Defaults to the model's own recommendation "
        "where its card carries one, else the pipeline's. A sampler from the "
        "wrong family for the model is a 400, not a silently worse picture."))
    vae_model: Optional[str] = Field(None, description="Override the VAE for this request.")
    loras: Optional[List[LoraConfig]] = Field(None, description="Additional LoRA adapters to apply for this request.")

    # Character consistency
    characters: Optional[List[Any]] = Field(None, description=(
        'The characters in the picture: profile names (["alice", "bob"]) or '
        'slots ([{"name": "alice", "images": ["b64...", ...]}]). Several are '
        'kept as SEPARATE reference sets, so two named characters appear as two '
        'people on a model that takes multiple references (FLUX.2, '
        'Qwen-Image-Edit); IP-Adapter averages its references and would blend '
        'them into one.'))
    character_profiles: Optional[List[str]] = Field(None, description="Saved character profile names to apply (IP-Adapter).")
    character_references: Optional[List[str]] = Field(None, description="Inline base64 reference images for character consistency.")
    character_strength: Optional[float] = Field(0.6, description="IP-Adapter scale for character references.")
    environment_profiles: Optional[List[str]] = Field(None, description="Saved environment profile names to apply (IP-Adapter).")

    model_config = ConfigDict(extra="allow")


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[Dict]
    model_config = ConfigDict(extra="allow")
