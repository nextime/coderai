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

from pydantic import BaseModel, ConfigDict


class LoraConfig(BaseModel):
    model: str
    weight: float = 1.0
    name: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class ImageGenerationRequest(BaseModel):
    model: str
    prompt: str
    n: int = 1
    size: Optional[str] = "1024x1024"
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    quality: Optional[str] = "standard"
    style: Optional[str] = None
    response_format: Optional[str] = "url"
    seed: Optional[int] = None
    user: Optional[str] = None
    disable_safety_checker: Optional[bool] = False
    negative_prompt: Optional[str] = None

    # Per-request component overrides
    vae_model: Optional[str] = None                 # Override the VAE for this request
    loras: Optional[List[LoraConfig]] = None        # Additional LoRA weights for this request

    # Character consistency
    character_profiles: Optional[List[str]] = None      # saved profile names
    character_references: Optional[List[str]] = None    # inline base64 images
    character_strength: Optional[float] = 0.6           # IP-Adapter scale
    environment_profiles: Optional[List[str]] = None    # saved environment profile names (IP-Adapter)

    model_config = ConfigDict(extra="allow")


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[Dict]
    model_config = ConfigDict(extra="allow")
