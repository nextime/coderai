"""Pydantic models for embeddings API."""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict


class EmbeddingsRequest(BaseModel):
    model: str
    input: Union[str, List[str]]           # text(s) to embed
    image: Optional[Union[str, List[str]]] = None  # base64/URL image(s) for multimodal embed
    encoding_format: Optional[str] = "float"       # float | base64
    dimensions: Optional[int] = None               # truncate to N dims if supported
    user: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: Union[List[float], str]     # float list or base64


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: Dict
    model_config = ConfigDict(extra="allow")
