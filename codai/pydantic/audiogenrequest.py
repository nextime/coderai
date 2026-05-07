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

"""Pydantic models for audio generation API."""

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class AudioGenerationRequest(BaseModel):
    model: str
    prompt: str
    duration: Optional[float] = 10.0       # seconds
    top_k: Optional[int] = 250
    top_p: Optional[float] = 0.0
    temperature: Optional[float] = 1.0
    cfg_coef: Optional[float] = 3.0        # classifier-free guidance coefficient
    seed: Optional[int] = None
    # Reference audio for melody conditioning (MusicGen Melody)
    melody: Optional[str] = None           # base64/URL
    # Voice profile for singing/speech conditioning
    voice_profile: Optional[str] = None    # saved voice profile name
    # Output
    response_format: Optional[str] = "url"  # url | b64_wav | b64_mp3
    user: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class AudioGenerationResponse(BaseModel):
    created: int
    data: List[Dict]
    model_config = ConfigDict(extra="allow")