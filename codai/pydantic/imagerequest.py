"""Pydantic models for image generation API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class ImageGenerationRequest(BaseModel):
    model: str
    prompt: str
    n: int = 1
    size: Optional[str] = "1024x1024"
    steps: Optional[int] = None  # Number of inference steps (overrides quality-based default)
    guidance_scale: Optional[float] = None  # CFG scale (overrides quality-based default)
    quality: Optional[str] = "standard"
    style: Optional[str] = None
    response_format: Optional[str] = "url"
    seed: Optional[int] = None
    user: Optional[str] = None
    
    model_config = ConfigDict(extra="allow")


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[Dict]
    model_config = ConfigDict(extra="allow")
