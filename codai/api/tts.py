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


def get_cached_model_path(url: str) -> str:
    """Get cached model path if available."""
    from codai.models.manager import multi_model_manager
    return multi_model_manager.get_cached_model_path(url)


def get_model_cache_dir() -> str:
    """Get model cache directory."""
    from codai.models.manager import multi_model_manager
    return multi_model_manager.get_model_cache_dir()


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
    tts_model = multi_model_manager.tts_model
    
    # If no TTS model configured, return an error
    if not tts_model:
        raise HTTPException(
            status_code=400,
            detail="TTS not configured. Use --tts-model to specify a model."
        )
    
    # Get load mode to determine if we need to unload other models first
    from codai.api.state import get_load_mode
    from codai.models.manager import model_manager
    load_mode = get_load_mode()
    
    # In ondemand mode, if ANY model is loaded and it's different from what we need, unload first
    if load_mode == "ondemand":
        has_any_model = len(multi_model_manager.models) > 0 or model_manager.backend is not None
        
        if has_any_model:
            # Resolve both the requested TTS model and currently loaded model to their canonical names
            requested_canonical = multi_model_manager.resolve_model_name(f"tts:{tts_model}")
            loaded_canonical = multi_model_manager.get_currently_loaded_model_name()
            
            # Also check legacy model_manager
            if not loaded_canonical and model_manager.backend is not None:
                loaded_canonical = "legacy_model_manager"
            
            # Compare: if they're different models, unload first
            already_loaded = (requested_canonical and loaded_canonical and 
                            requested_canonical == loaded_canonical)
            
            if not already_loaded:
                print(f"In ondemand mode - model switch detected:")
                print(f"  Requested: 'tts:{tts_model}' (resolved to: '{requested_canonical}')")
                print(f"  Loaded: '{loaded_canonical}'")
                print(f"  -> Fully unloading current model(s) before loading TTS model...")
                multi_model_manager.unload_all_models()
                if model_manager.backend is not None:
                    try:
                        model_manager.cleanup()
                    except:
                        pass
    
    # Determine model to use
    model_to_use = request.model
    if model_to_use.startswith("tts:"):
        model_to_use = tts_model
    
    # Try to use kokoro if available
    try:
        from kokoro import Kokoro
        
        # Determine model key
        model_key = f"tts:{model_to_use}"
        kokoro_model = multi_model_manager.get_model(model_key)
        
        if kokoro_model is None:
            print(f"Loading Kokoro TTS model: {model_to_use}")
            
            # Check if model_to_use is a URL - download it (with caching)
            model_path = None
            if model_to_use.startswith('http://') or model_to_use.startswith('https://'):
                # Check cache first
                cached_path = get_cached_model_path(model_to_use)
                if cached_path:
                    model_path = cached_path
                    print(f"Using cached model: {model_path}")
                else:
                    print(f"Downloading model from URL: {model_to_use}")
                    try:
                        import requests
                        import hashlib
                        
                        # Get cache directory
                        cache_dir = get_model_cache_dir()
                        
                        # Extract filename from URL
                        url_path = model_to_use.split('?')[0]
                        filename = os.path.basename(url_path)
                        
                        if not filename.endswith('.pt') and not filename.endswith('.bin'):
                            filename = "kokoro-model.pt"
                        
                        # Create safe filename in cache
                        url_hash = hashlib.sha256(model_to_use.encode()).hexdigest()
                        cached_filename = f"{url_hash}_{filename}"
                        model_path = os.path.join(cache_dir, cached_filename)
                        
                        # Download to cache
                        response = requests.get(model_to_use, stream=True)
                        response.raise_for_status()
                        
                        total_size = int(response.headers.get('content-length', 0))
                        downloaded = 0
                        
                        with open(model_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192*1024):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    if total_size > 0:
                                        percent = (downloaded / total_size) * 100
                                        print(f"Downloaded: {percent:.1f}%", end='\r')
                        
                        print(f"\nDownloaded and cached to: {model_path}")
                        
                    except Exception as e:
                        print(f"Error downloading model: {e}")
                        raise
            else:
                # Use local path or model name
                model_path = model_to_use
            
            # Load the Kokoro model
            kokoro_model = Kokoro(model_path if model_path else model_to_use)
            multi_model_manager.add_model(model_key, kokoro_model)
        
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
