"""
Audio transcription endpoint for the codai API.
"""

import io
import os
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

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


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    model: str = Form(...),
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
):
    """
    Audio transcription endpoint (OpenAI-compatible).
    """
    # Check if whisper-server is available FIRST
    if multi_model_manager.whisper_server and multi_model_manager.whisper_server.is_running():
        file_content = await file.read()
        result = multi_model_manager.whisper_server.transcribe(
            file_content,
            language=language,
            prompt=prompt
        )
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"text": result.get("text", "")}
    
    audio_model = multi_model_manager.audio_models[0] if multi_model_manager.audio_models else None
    if not audio_model:
        raise HTTPException(
            status_code=400,
            detail="Audio transcription not configured. Use --audio-model or --whisper-server."
        )
    
    # Determine model to use
    model_to_use = model
    if model_to_use.startswith("whisper:") or model_to_use.startswith("audio:"):
        model_to_use = audio_model
    
    # Read the uploaded file
    file_content = await file.read()
    
    # Save to temp file (needed for some backends)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name
    
    try:
        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel
            
            # Determine model key
            model_key = f"audio:{model_to_use}"
            whisper_model = multi_model_manager.get_model(model_key)
            
            if whisper_model is None:
                print(f"Loading faster-whisper model: {model_to_use}")
                
                # Determine compute type - always use int8 for CPU
                compute_type = "int8"
                
                # Load the model
                whisper_model = WhisperModel(
                    model_to_use,
                    device="cpu",  # Always use CPU - faster-whisper CUDA doesn't work with AMD
                    compute_type=compute_type,
                )
                
                # Cache the model
                multi_model_manager.add_model(model_key, whisper_model)
                print(f"Loaded faster-whisper model: {model_to_use}")
            
            # Run transcription
            segments, info = whisper_model.transcribe(
                tmp_path,
                language=language,
                initial_prompt=prompt,
                temperature=temperature,
            )
            
            # Collect all segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)
            
            full_text = "".join(text_parts)
            
            return {
                "text": full_text.strip()
            }
            
        except ImportError:
            pass
        
        # Try whispercpp as fallback
        try:
            import whispercpp
            
            # Determine model key
            model_key = f"audio:{model_to_use}"
            whisper_model = multi_model_manager.get_model(model_key)
            
            if whisper_model is None:
                print(f"Loading whispercpp model: {model_to_use}")
                
                # Check if it's a built-in model name
                if model_to_use in ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large']:
                    # It's a built-in model name
                    whisper_model = whispercpp.Whisper.from_pretrained(model_to_use)
                else:
                    # It's a path to a GGUF file
                    whisper_model = whispercpp.Whisper.from_pretrained(model_to_use)
                
                # Cache the model
                multi_model_manager.add_model(model_key, whisper_model)
                print(f"Loaded whispercpp model: {model_to_use}")
            
            # Run transcription
            result = whisper_model.transcribe(tmp_path)
            
            # Extract text from result
            text = ""
            if hasattr(result, 'text'):
                text = result.text
            elif isinstance(result, dict):
                text = result.get('text', '')
            elif isinstance(result, list):
                # Some versions return a list of segments
                for segment in result:
                    if hasattr(segment, 'text'):
                        text += segment.text
                    elif isinstance(segment, dict):
                        text += segment.get('text', '')
            
            return {
                "text": text.strip()
            }
            
        except ImportError as e:
            raise HTTPException(
                status_code=501,
                detail="Audio transcription not available. Install faster-whisper or whispercpp."
            )
            
    except Exception as e:
        print(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
