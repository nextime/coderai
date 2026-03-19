"""
Text-to-speech endpoints for the codai API.
"""

import base64
import os

from fastapi import APIRouter, HTTPException
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
    
    model_config = ConfigDict(extra="allow")


class TTSResponse(BaseModel):
    audio: str  # base64 encoded audio
    
    model_config = ConfigDict(extra="allow")


@router.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    Text-to-speech endpoint (OpenAI-compatible).
    
    Supports:
    - Kokoro TTS models (when --tts-model is specified)
    """
    # Use the manager to resolve the model and manage VRAM
    model_info = multi_model_manager.request_model(
        requested_model=request.model,
        model_type="tts"
    )
    
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
