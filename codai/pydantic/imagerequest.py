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

from typing import Dict, List, Optional

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

    # Per-request component overrides
    vae_model: Optional[str] = Field(None, description="Override the VAE for this request.")
    loras: Optional[List[LoraConfig]] = Field(None, description="Additional LoRA adapters to apply for this request.")

    # Character consistency
    character_profiles: Optional[List[str]] = Field(None, description="Saved character profile names to apply (IP-Adapter).")
    character_references: Optional[List[str]] = Field(None, description="Inline base64 reference images for character consistency.")
    character_strength: Optional[float] = Field(0.6, description="IP-Adapter scale for character references.")
    environment_profiles: Optional[List[str]] = Field(None, description="Saved environment profile names to apply (IP-Adapter).")

    model_config = ConfigDict(extra="allow")


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[Dict]
    model_config = ConfigDict(extra="allow")
