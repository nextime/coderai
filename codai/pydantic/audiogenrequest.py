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
    # Output
    response_format: Optional[str] = "url"  # url | b64_wav | b64_mp3
    user: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class AudioGenerationResponse(BaseModel):
    created: int
    data: List[Dict]
    model_config = ConfigDict(extra="allow")
