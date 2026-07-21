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

"""Pydantic models for embeddings API."""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class EmbeddingsRequest(BaseModel):
    model: str = Field(..., description="Embedding model id to use.")
    input: Optional[Union[str, List[str]]] = Field(None, description="Text or list of texts to embed. Optional only when 'image' is given.")
    image: Optional[Union[str, List[str]]] = Field(None, description="Image(s) for multimodal embedding models: http(s) URL, data URI, local path or bare base64. Vectors are returned after the text ones.")
    encoding_format: Optional[str] = Field("float", description="Return embeddings as 'float' arrays or 'base64'.")
    dimensions: Optional[int] = Field(None, description="Truncate embeddings to N dimensions (if the model supports it).")
    quantization: Optional[str] = Field(None, description="Optional TurboQuant vector quantization: 'turbo' (8-bit), 'turbo8', 'turbo6', 'turbo4' or 'turbo2'. With encoding_format='float' the (lossy) reconstructed vectors are returned; with 'base64' the compact packed bytes are returned plus a 'quantization' metadata block describing how to decode them.")
    user: Optional[str] = Field(None, description="Opaque end-user identifier (passthrough).")
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