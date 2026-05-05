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