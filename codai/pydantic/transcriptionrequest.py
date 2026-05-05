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