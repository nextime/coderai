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

"""
Text-to-speech endpoints for the codai API.
"""

import asyncio
import base64
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

# Import from codai modules
from codai.models.manager import multi_model_manager


# Global reference to be set by coderai
global_args = None


def set_global_args(args):
    """Set global args from coderai."""
    global global_args
    global_args = args


# =============================================================================
# Router and Endpoints
# =============================================================================

router = APIRouter()


class TTSRequest(BaseModel):
    model: str
    input: str
    voice: str = "af_sarah"
    response_format: str = "mp3"
    speed: float = 1.0
    voice_profile: Optional[str] = None   # saved voice profile name (uses F5-TTS cloning)

    model_config = ConfigDict(extra="allow")


class TTSResponse(BaseModel):
    audio: str  # base64 encoded audio
    
    model_config = ConfigDict(extra="allow")


@router.post("/v1/audio/speech", summary="Text-to-speech synthesis")
async def create_speech(request: TTSRequest, http_request: Request = None):
    """
    Text-to-speech endpoint (OpenAI-compatible).
    
    Supports:
    - Kokoro TTS models (when --tts-model is specified)
    """
    # If a voice profile is requested, delegate to voice cloning (F5-TTS)
    if request.voice_profile:
        from codai.api.voice_clone import _load_voice, _f5tts_clone
        meta = _load_voice(request.voice_profile)
        if not meta:
            raise HTTPException(status_code=404,
                detail=f"Voice profile '{request.voice_profile}' not found")
        ref_audio_path = meta['audio_file']
        ref_text = meta.get('transcript', '')
        if not ref_text:
            raise HTTPException(status_code=400,
                detail="Voice profile has no transcript; update it with PATCH /v1/audio/voices/{name}")
        try:
            audio_bytes = await asyncio.get_event_loop().run_in_executor(
                None, _f5tts_clone,
                ref_audio_path, ref_text, request.input,
                request.speed or 1.0, None,
            )
        except ImportError:
            raise HTTPException(status_code=501,
                detail="f5-tts not installed. Run: pip install f5-tts")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Voice cloning failed: {e}")
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return {"audio": audio_base64}

    # Use the manager to resolve the model and manage VRAM
    model_info = await asyncio.to_thread(
        multi_model_manager.request_model,
        requested_model=request.model,
        model_type="tts",
    )
    
    # Check if the model was rejected as not allowed
    if model_info.get('error'):
        raise HTTPException(status_code=404, detail=model_info['error'])
    
    model_name = model_info['model_name']
    model_key = model_info['model_key']
    kokoro_model = model_info['model_object']
    
    # If no TTS model configured, return an error
    if not model_name:
        raise HTTPException(
            status_code=400,
            detail="TTS not configured. Use --tts-model to specify a model."
        )
    
    # Try to use kokoro if available
    try:
        from kokoro import Kokoro
        
        if kokoro_model is None:
            print(f"Loading Kokoro TTS model: {model_name}")
            
            # Check if model_name is a URL - download it (with caching)
            model_path = None
            if model_name.startswith('http://') or model_name.startswith('https://'):
                print(f"Loading model from URL: {model_name}")
                from codai.models.cache import load_model
                model_path = load_model(model_name)
                if not model_path:
                    raise Exception(f"Failed to load model from {model_name}")
            else:
                # Use local path or model name
                model_path = model_name
            
            # Load the Kokoro model
            kokoro_model = Kokoro(model_path if model_path else model_name)
            multi_model_manager.add_model(model_key, kokoro_model)
            multi_model_manager.current_model_key = model_key
        
        # Generate speech
        voice = request.voice or "af_sarah"
        speed = request.speed or 1.0
        
        audio_bytes = kokoro_model.generate(request.input, voice=voice, speed=speed)

        try:
            from codai.api.archive import archive_manager
            asyncio.get_event_loop().create_task(asyncio.to_thread(
                archive_manager.save_generation,
                "tts", "/v1/audio/speech",
                model_name,
                request.input,
                {"voice": voice, "speed": speed, "response_format": request.response_format},
                [(audio_bytes, request.response_format or "mp3")],
            ))
        except Exception:
            pass

        # Convert to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        return {
            "audio": audio_base64
        }
        
    except ImportError as e:
        # kokoro not installed
        raise HTTPException(
            status_code=501,
            detail=f"TTS not available. Install kokoro: pip install kokoro. Error: {str(e)}"
        )
    except Exception as e:
        print(f"TTS error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")