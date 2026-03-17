"""Pydantic models for transcription API."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class TranscriptionRequest(BaseModel):
    model: str
    file: Optional[bytes] = None
    file_path: Optional[str] = None
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"
    temperature: Optional[float] = 0.0
    timestamp_granularities: Optional[List[str]] = None
    
    model_config = ConfigDict(extra="allow")


class TranscriptionResponse(BaseModel):
    text: str
    model_config = ConfigDict(extra="allow")
