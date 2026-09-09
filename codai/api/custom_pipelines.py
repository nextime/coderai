"""
Custom pipeline executor.

GET  /v1/pipelines/custom          — list saved custom pipelines
POST /v1/pipelines/custom          — save a new custom pipeline definition
PUT  /v1/pipelines/custom/{id}     — update a pipeline
DELETE /v1/pipelines/custom/{id}   — delete a pipeline
POST /v1/pipelines/custom/{id}/run — execute a saved pipeline
POST /v1/pipelines/run             — execute an inline pipeline definition (no save)

Pipeline definition schema:
{
  "id": "my-pipeline",          # auto-generated if absent
  "name": "My Pipeline",
  "steps": [
    {
      "type": "text_gen",        # step type (see STEP_TYPES)
      "label": "Write script",   # optional display label
      "params": {                # static params merged with runtime context
        "model": "Qwen/Qwen3.5-9B",
        "prompt": "{{input}}",   # {{input}} = pipeline input text
                                 # {{stepN.output}} = output of step N
                                 # {{stepN.url}} = URL output of step N
      }
    },
    {
      "type": "image_gen",
      "params": {
        "model": "sd-model",
        "prompt": "{{step0.output}}"
      }
    }
  ]
}

Step types and their endpoint mapping:
  text_gen      → POST /v1/chat/completions
  image_gen     → POST /v1/images/generations
  image_edit    → POST /v1/images/edits
  image_inpaint → POST /v1/images/inpaint
  image_upscale → POST /v1/images/upscale
  image_deblur  → POST /v1/images/deblur
  image_unpix   → POST /v1/images/unpixelate
  image_outfit  → POST /v1/images/outfit
  image_faceswap→ POST /v1/images/faceswap
  video_gen     → POST /v1/video/generations
  video_upscale → POST /v1/video/upscale
  video_sub     → POST /v1/video/subtitle
  video_interp  → POST /v1/video/interpolate
  video_dub     → POST /v1/video/dub
  tts           → POST /v1/audio/speech
  stt           → POST /v1/audio/transcriptions (multipart)
  audio_gen     → POST /v1/audio/generate
  voice_clone   → POST /v1/audio/clone
  voice_convert → POST /v1/audio/convert
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

router = APIRouter()

# ---------------------------------------------------------------------------
# Step type → (handler_module, handler_fn, request_class)
# ---------------------------------------------------------------------------

STEP_TYPES = {
    "text_gen":       ("codai.api.text",         "chat_completions",       "codai.pydantic.textrequest.ChatCompletionRequest"),
    "image_gen":      ("codai.api.images",        "create_image_generation","codai.pydantic.imagerequest.ImageGenerationRequest"),
    "image_edit":     ("codai.api.images",        "create_image_edit",      None),
    "image_inpaint":  ("codai.api.images",        "create_image_inpaint",   None),
    "image_upscale":  ("codai.api.images",        "create_image_upscale",   None),
    "image_deblur":   ("codai.api.images",        "create_image_deblur",    None),
    "image_unpix":    ("codai.api.images",        "create_image_unpixelate",None),
    "image_outfit":   ("codai.api.images",        "create_image_outfit",    None),
    "image_faceswap": ("codai.api.faceswap",      "faceswap",               None),
    "video_gen":      ("codai.api.video",         "video_generations",       "codai.pydantic.videorequest.VideoGenerationRequest"),
    "video_upscale":  ("codai.api.video",         "video_upscale",           None),
    "video_sub":      ("codai.api.video",         "video_subtitle",          None),
    "video_interp":   ("codai.api.video",         "video_interpolate",       None),
    "video_dub":      ("codai.api.video",         "video_dub",               None),
    "tts":            ("codai.api.tts",           "create_speech",          None),
    "stt":            ("codai.api.custom_pipelines", "run_stt_step",       None),
    "audio_gen":      ("codai.api.audio_gen",     "audio_generate",         None),
    "voice_clone":    ("codai.api.voice_clone",   "clone_voice",            None),
    "voice_convert":  ("codai.api.voice_convert", "convert_voice",          None),
    "ocr":            ("codai.api.ocr",           "run_ocr_step",           None),
}

# Human-readable labels for the UI
STEP_TYPE_LABELS = {
    "text_gen":       "Text Generation (LLM)",
    "image_gen":      "Image Generation",
    "image_edit":     "Image Edit (i2i)",
    "image_inpaint":  "Image Inpaint",
    "image_upscale":  "Image Upscale",
    "image_deblur":   "Image Deblur",
    "image_unpix":    "Image Unpixelate",
    "image_outfit":   "Outfit Change",
    "image_faceswap": "Face Swap",
    "video_gen":      "Video Generation",
    "video_upscale":  "Video Upscale",
    "video_sub":      "Video Subtitles",
    "video_interp":   "Video Interpolate",
    "video_dub":      "Video Dub",
    "tts":            "Text-to-Speech",
    "stt":            "Speech-to-Text",
    "audio_gen":      "Audio/Music Generation",
    "voice_clone":    "Voice Clone (TTS)",
    "voice_convert":  "Voice Convert (SVC)",
    "ocr":            "OCR (document → text)",
}

# Which params each step type accepts (for the UI form builder)
STEP_PARAMS = {
    "text_gen":       [("model","text","Model ID"),("prompt","textarea","Prompt"),("system","textarea","System prompt (opt)")],
    "image_gen":      [("model","text","Model ID"),("prompt","textarea","Prompt"),("negative_prompt","text","Negative prompt"),("size","text","Size","1024x1024"),("steps","number","Steps"),("guidance_scale","number","CFG","7.5"),("seed","number","Seed")],
    "image_edit":     [("model","text","Model ID"),("prompt","textarea","Prompt"),("image","ref","Source image ({{stepN.url}})"),("strength","number","Strength","0.75"),("steps","number","Steps"),("seed","number","Seed")],
    "image_inpaint":  [("model","text","Model ID"),("prompt","textarea","Prompt"),("image","ref","Source image"),("mask","ref","Mask image"),("strength","number","Strength","0.99"),("steps","number","Steps"),("seed","number","Seed")],
    "image_upscale":  [("model","text","Model ID (opt)"),("image","ref","Source image"),("scale","number","Scale","4")],
    "image_deblur":   [("image","ref","Source image"),("strength","number","Strength","0.5")],
    "image_unpix":    [("image","ref","Source image"),("scale","number","Scale","4")],
    "image_outfit":   [("model","text","Inpaint model ID"),("image","ref","Source image"),("prompt","textarea","Outfit prompt"),("negative_prompt","text","Negative prompt"),("steps","number","Steps"),("seed","number","Seed")],
    "image_faceswap": [("source_face","ref","Source face image"),("target","ref","Target image/video"),("target_type","select:image|video","Target type","image")],
    "video_gen":      [("model","text","Model ID"),("prompt","textarea","Prompt"),("mode","select:t2v|i2v|v2v|ti2v","Mode","t2v"),("init_image","ref","Init image (i2v)"),("num_frames","number","Frames","16"),("fps","number","FPS","8"),("num_inference_steps","number","Steps","25"),("guidance_scale","number","CFG","7.5"),("seed","number","Seed")],
    "video_upscale":  [("model","text","Model ID"),("video","ref","Source video"),("upscale_factor","number","Scale","2")],
    "video_sub":      [("model","text","Model ID"),("video","ref","Source video"),("language","text","Language"),("burn","checkbox","Burn into video")],
    "video_dub":      [("model","text","Model ID"),("video","ref","Source video"),("target_lang","text","Target language"),("source_lang","text","Source language"),("burn_subtitles","checkbox","Burn subtitles")],
    "tts":            [("model","text","Model ID"),("input","textarea","Text ({{stepN.output}})"),("voice","text","Voice","af_sarah"),("speed","number","Speed","1.0")],
    "stt":            [("model","text","Model ID"),("audio","ref","Audio/video input"),("language","text","Language hint"),("prompt","text","Context hint"),("response_format","select:json|text|verbose_json|srt|vtt","Response format","json")],
    "audio_gen":      [("model","text","Model ID"),("prompt","textarea","Prompt"),("duration","number","Duration (s)","10"),("temperature","number","Temperature","1.0")],
    "audio_stems":    [("audio","ref","Source audio"),("stem_mode","select:vocals-instrumental|4-stem|drums-bass-other","Requested split","vocals-instrumental")],
    "audio_cleanup":  [("audio","ref","Source audio"),("noise_reduction","checkbox","Reduce background noise"),("normalize","checkbox","Normalize levels"),("remove_hum","checkbox","Remove hum"),("repair_clicks","checkbox","Repair clicks")],
    "voice_clone":    [("text","textarea","Text to synthesize"),("voice_name","text","Voice profile name"),("ref_text","text","Reference transcript"),("speed","number","Speed","1.0")],
    "voice_convert":  [("source_audio","ref","Source audio"),("voice_name","text","Voice profile name"),("f0_condition","checkbox","Singing mode"),("pitch_shift","number","Pitch shift","0"),("diffusion_steps","number","Steps","10")],
    "ocr":            [("image","ref","Document image/PDF ({{stepN.url}})"),("engine","select:paddle|doctr|surya","OCR engine","paddle"),("detect","select:off|layout|detector|both","Stamp/signature detect","off"),("structured","checkbox","Structured JSON extraction")],
}



def _resolve_template(value: Any, context: Dict) -> Any:
    """Replace {{input}}, {{stepN.output}}, {{stepN.url}} etc. in string values."""
    if not isinstance(value, str):
        return value
    import re
    def _replace(m):
        key = m.group(1).strip()
        # {{input}} → pipeline input
        if key == 'input':
            return str(context.get('input', ''))
        # {{stepN.field}}
        match = re.match(r'step(\d+)\.(\w+)', key)
        if match:
            n, field = int(match.group(1)), match.group(2)
            step_result = context.get(f'step{n}', {})
            return str(step_result.get(field, ''))
        return m.group(0)
    return re.sub(r'\{\{([^}]+)\}\}', _replace, value)


def _resolve_params(params: Dict, context: Dict) -> Dict:
    return {k: _resolve_template(v, context) for k, v in params.items()}


def _extract_output(step_type: str, result: Any) -> Dict:
    """Extract useful fields from a step result for use in subsequent steps."""
    if result is None:
        return {}
    r = result if isinstance(result, dict) else (result.__dict__ if hasattr(result, '__dict__') else {})
    out = {}
    if 'choices' in r:
        out['output'] = r['choices'][0].get('message', {}).get('content', '') if r['choices'] else ''
    if 'text' in r:
        out['text'] = r['text']
        out.setdefault('output', r['text'])
    if 'data' in r and r['data']:
        item = r['data'][0]
        if isinstance(item, dict):
            out['url'] = item.get('url', '')
            for k, v in item.items():
                out[k] = v
    if 'audio' in r:
        out['audio'] = r['audio']
        out['output'] = r['audio']
    return out


class STTStepRequest(BaseModel):
    model: str
    audio: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = 'json'
    model_config = ConfigDict(extra='allow')


async def run_stt_step(request: STTStepRequest):
    from starlette.datastructures import Headers, UploadFile
    from io import BytesIO
    import base64

    from codai.api.transcriptions import create_transcription

    audio_ref = request.audio or ''
    filename = 'input.wav'
    payload = b''
    if audio_ref.startswith('data:'):
        header, encoded = audio_ref.split(',', 1)
        payload = base64.b64decode(encoded)
        if 'audio/' in header:
            subtype = header.split('audio/', 1)[1].split(';', 1)[0]
            if subtype:
                filename = f'input.{subtype}'
        elif 'video/' in header:
            subtype = header.split('video/', 1)[1].split(';', 1)[0]
            if subtype:
                filename = f'input.{subtype}'
    else:
        payload = base64.b64decode(audio_ref)

    upload = UploadFile(file=BytesIO(payload), filename=filename, headers=Headers())
    return await create_transcription(
        model=request.model,
        file=upload,
        language=request.language,
        prompt=request.prompt,
        response_format=request.response_format or 'json',
        temperature=0.0,
    )


async def _run_step(step: Dict, context: Dict, http_request) -> Dict:
    """Execute a single pipeline step and return its output context."""
    step_type = step['type']
    if step_type not in STEP_TYPES:
        raise ValueError(f"Unknown step type: {step_type}")

    mod_name, fn_name, req_class_path = STEP_TYPES[step_type]
    params = _resolve_params(step.get('params', {}), context)

    # Import handler
    import importlib
    mod = importlib.import_module(mod_name)
    handler = getattr(mod, fn_name)

    # Build request object
    if req_class_path:
        req_mod, req_cls = req_class_path.rsplit('.', 1)
        req_class = getattr(importlib.import_module(req_mod), req_cls)
        # text_gen needs messages format
        if step_type == 'text_gen':
            messages = [{"role": "user", "content": params.pop('prompt', '')}]
            if 'system' in params and params['system']:
                messages.insert(0, {"role": "system", "content": params.pop('system')})
            else:
                params.pop('system', None)
            params['messages'] = messages
            params.setdefault('stream', False)
        req = req_class(**{k: v for k, v in params.items() if v != ''})
    else:
        # Find the request class from the handler's type hints
        import inspect
        sig = inspect.signature(handler)
        first_param = list(sig.parameters.values())[0]
        ann = first_param.annotation
        if ann != inspect.Parameter.empty:
            req = ann(**{k: v for k, v in params.items() if v != ''})
        else:
            req = type('Req', (), params)()

    result = await handler(req, http_request)
    return _extract_output(step_type, result)


def _infer_step_model_key(step: Dict) -> Optional[str]:
    step_type = step.get('type')
    params = step.get('params', {})

    if step_type == 'stt':
        model = params.get('model') or params.get('audio_model')
        return f"audio:{model}" if model else None
    if step_type == 'text_gen':
        return params.get('model')
    if step_type in {'image_gen', 'image_edit', 'image_upscale', 'image_depth', 'image_segment'}:
        model = params.get('model')
        return f"image:{model}" if model else None
    if step_type in {'embed', 'embedding'}:
        model = params.get('model')
        return f"embedding:{model}" if model else None
    if step_type in {'video_gen', 'video'}:
        model = params.get('model')
        return f"video:{model}" if model else None
    return None


async def _run_scheduled_step(step: Dict, context: Dict, http_request) -> Dict:
    from codai.queue.manager import queue_manager

    model_key = _infer_step_model_key(step)
    if not model_key:
        return await _run_step(step, context, http_request)

    request_id = f"pipeline-step-{uuid.uuid4().hex[:8]}"
    lease = await queue_manager.acquire(request_id, model_key)
    try:
        return await _run_step(step, context, http_request)
    finally:
        await queue_manager.release(lease)


async def _execute_pipeline(pipeline_def: Dict, pipeline_input: str, http_request) -> Dict:
    """Execute all steps of a pipeline definition."""
    context = {'input': pipeline_input}
    steps_output = []

    for i, step in enumerate(pipeline_def.get('steps', [])):
        try:
            out = await _run_scheduled_step(step, context, http_request)
            context[f'step{i}'] = out
            steps_output.append({'step': i, 'type': step['type'],
                                  'label': step.get('label', step['type']), **out})
        except Exception as e:
            steps_output.append({'step': i, 'type': step['type'],
                                  'label': step.get('label', step['type']),
                                  'error': str(e)})
            if not step.get('continue_on_error', False):
                break

    return {
        'created': int(time.time()),
        'pipeline': pipeline_def.get('name', pipeline_def.get('id', 'custom')),
        'steps': steps_output,
        'data': [context.get(f'step{len(steps_output)-1}', {})] if steps_output else [],
    }


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------

class PipelineStep(BaseModel):
    type: str
    label: Optional[str] = None
    params: Dict[str, Any] = {}
    continue_on_error: Optional[bool] = False
    model_config = ConfigDict(extra='allow')


class PipelineDefinition(BaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str] = ''
    steps: List[PipelineStep]
    model_config = ConfigDict(extra='allow')


class PipelineRunRequest(BaseModel):
    input: Optional[str] = ''
    model_config = ConfigDict(extra='allow')


class AudioUnderstandRequest(BaseModel):
    audio: str
    audio_model: str
    text_model: Optional[str] = None
    input: Optional[str] = ''
    language: Optional[str] = None
    prompt: Optional[str] = None
    model_config = ConfigDict(extra='allow')


class AudioMusicDubRequest(BaseModel):
    """Dub a song into another language over its original backing track.

    Only ``audio`` and ``audio_model`` are required; everything else tunes how the
    lyrics are adapted and how the replacement vocal is produced.
    """
    audio: str
    audio_model: str                          # STT model used on the isolated vocal
    target_lang: Optional[str] = None
    source_lang: Optional[str] = None
    notes: Optional[str] = ''                 # adaptation notes / STT prompt
    # Lyric adaptation: with a text model the lyrics are *adapted* to be singable in
    # the target language; without one we fall back to a literal translation.
    text_model: Optional[str] = None
    # Replacement vocal: by default the isolated original vocal is the cloning
    # reference, so the dub keeps the original singer's timbre.
    voice_name: Optional[str] = None
    ref_text: Optional[str] = None
    speed: Optional[float] = 1.0
    seed: Optional[int] = None
    # Seed-VC singing pass (f0-conditioned) over the synthesised vocal.
    sing_convert: Optional[bool] = True
    diffusion_steps: Optional[int] = 15
    pitch_shift: Optional[int] = 0
    # Stem separation: demucs by default, ffmpeg best-effort when true.
    fallback_mode: Optional[bool] = False
    model_config = ConfigDict(extra='allow')


@router.get('/v1/pipelines/custom', summary="List saved custom pipelines")
async def list_custom_pipelines():
    """List all saved custom pipeline definitions."""
    from codai.admin.routes import config_manager
    if config_manager is None:
        return {'pipelines': []}
    return {'pipelines': config_manager.pipelines_data}


@router.get('/v1/pipelines/step-types', summary="List available pipeline step types")
async def list_step_types():
    """List available step types with their parameter schemas."""
    return {
        'step_types': [
            {'type': t, 'label': STEP_TYPE_LABELS[t], 'params': STEP_PARAMS.get(t, [])}
            for t in STEP_TYPES
        ]
    }


@router.post('/v1/pipelines/custom', summary="Create a custom pipeline")
async def create_custom_pipeline(pipeline: PipelineDefinition):
    """Save a new custom pipeline definition."""
    from codai.admin.routes import config_manager
    if config_manager is None:
        raise HTTPException(status_code=503, detail='Config manager not available')
    data = pipeline.model_dump()
    if not data.get('id'):
        data['id'] = uuid.uuid4().hex[:8]
    # Ensure no duplicate id
    config_manager.pipelines_data = [p for p in config_manager.pipelines_data if p.get('id') != data['id']]
    config_manager.pipelines_data.append(data)
    config_manager.save_pipelines()
    return {'created': True, 'pipeline': data}


@router.put('/v1/pipelines/custom/{pipeline_id}', summary="Update a custom pipeline")
async def update_custom_pipeline(pipeline_id: str, pipeline: PipelineDefinition):
    """Update an existing custom pipeline."""
    from codai.admin.routes import config_manager
    if config_manager is None:
        raise HTTPException(status_code=503, detail='Config manager not available')
    data = pipeline.model_dump()
    data['id'] = pipeline_id
    existing = [p for p in config_manager.pipelines_data if p.get('id') != pipeline_id]
    if len(existing) == len(config_manager.pipelines_data):
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    existing.append(data)
    config_manager.pipelines_data = existing
    config_manager.save_pipelines()
    return {'updated': True, 'pipeline': data}


@router.delete('/v1/pipelines/custom/{pipeline_id}', summary="Delete a custom pipeline")
async def delete_custom_pipeline(pipeline_id: str):
    """Delete a custom pipeline."""
    from codai.admin.routes import config_manager
    if config_manager is None:
        raise HTTPException(status_code=503, detail='Config manager not available')
    before = len(config_manager.pipelines_data)
    config_manager.pipelines_data = [p for p in config_manager.pipelines_data if p.get('id') != pipeline_id]
    if len(config_manager.pipelines_data) == before:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    config_manager.save_pipelines()
    return {'deleted': True, 'id': pipeline_id}


@router.post('/v1/pipelines/custom/{pipeline_id}/run', summary="Run a saved custom pipeline")
async def run_custom_pipeline(pipeline_id: str, body: PipelineRunRequest, http_request: Request = None):
    """Execute a saved custom pipeline."""
    from codai.admin.routes import config_manager
    if config_manager is None:
        raise HTTPException(status_code=503, detail='Config manager not available')
    pipeline_def = next((p for p in config_manager.pipelines_data if p.get('id') == pipeline_id), None)
    if not pipeline_def:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    return await _execute_pipeline(pipeline_def, body.input or '', http_request)


@router.post('/v1/pipelines/run', summary="Run an inline pipeline definition")
async def run_inline_pipeline(pipeline: PipelineDefinition, http_request: Request = None):
    """Execute an inline pipeline definition without saving it."""
    return await _execute_pipeline(pipeline.model_dump(), '', http_request)


@router.post('/v1/pipelines/audio-understand', summary="Transcribe and analyze audio")
async def run_audio_understanding(request: AudioUnderstandRequest, http_request: Request = None):
    """Transcribe and analyze an audio clip in one pass.

    Convenience pipeline that transcribes the input audio and then reasons over the
    transcript (summary/understanding) using the configured text model. Returns the
    transcript together with the model's analysis.
    """
    if not request.audio:
        raise HTTPException(status_code=400, detail='Provide audio input')

    steps = []
    stt_step = {
        'type': 'stt',
        'label': 'Transcribe audio',
        'params': {
            'model': request.audio_model,
            'audio': request.audio,
            'language': request.language,
            'prompt': request.prompt,
            'response_format': 'json',
        },
    }
    stt_out = await _run_scheduled_step(stt_step, {'input': request.input or ''}, http_request)
    transcript = stt_out.get('text') or stt_out.get('output') or ''
    steps.append({'step': 0, 'type': 'stt', 'label': 'Transcribe audio', **stt_out})

    summary = None
    if request.text_model:
        text_step = {
            'type': 'text_gen',
            'label': 'Reason over transcript',
            'params': {
                'model': request.text_model,
                'prompt': f"{request.input or 'Summarize this audio transcript clearly.'}\n\nTranscript:\n{{{{step0.output}}}}",
            },
        }
        text_out = await _run_scheduled_step(text_step, {'input': request.input or '', 'step0': {'output': transcript, 'text': transcript}}, http_request)
        summary = text_out.get('output')
        steps.append({'step': 1, 'type': 'text_gen', 'label': 'Reason over transcript', **text_out})

    return {
        'created': int(time.time()),
        'pipeline': 'audio-understand',
        'transcript': transcript,
        'summary': summary,
        'steps': steps,
        'data': [{'transcript': transcript, 'summary': summary}],
    }


def _argos_translate(text: str, source_lang: Optional[str], target_lang: str) -> Optional[str]:
    """Literal line-by-line translation via argostranslate, or None if unavailable."""
    try:
        import argostranslate.translate
    except ImportError:
        return None
    src = (source_lang or 'en').split('-')[0]
    dst = target_lang.split('-')[0]
    try:
        out = []
        for line in text.split('\n'):
            out.append(argostranslate.translate.translate(line, src, dst) if line.strip() else line)
        return '\n'.join(out)
    except Exception:
        return None


async def _adapt_lyrics(request: 'AudioMusicDubRequest', transcript: str,
                        http_request) -> tuple:
    """Turn the source lyrics into target-language lyrics.

    Returns ``(lyrics, step_dict)``. A text model *adapts* them — matching syllable
    count and keeping the rhyme where it can, which is what makes a dub singable.
    Without one we fall back to a literal argostranslate pass, and if that is missing
    too we return the original and say so rather than pretending.
    """
    if not request.target_lang:
        return transcript, {'type': 'translate', 'label': 'Translate/adapt lyrics',
                            'status': 'skipped', 'reason': 'no target_lang given',
                            'output': transcript}

    if request.text_model:
        prompt = (
            f"Adapt these song lyrics into {request.target_lang}.\n"
            "Rules: keep the meaning, keep roughly the same syllable count per line so "
            "they remain singable to the original melody, preserve the rhyme scheme "
            "where possible, and keep the line breaks exactly as given.\n"
            "Reply with the adapted lyrics only — no commentary.\n"
        )
        if request.notes:
            prompt += f"Additional direction: {request.notes}\n"
        prompt += f"\nLyrics:\n{transcript}"
        step = {'type': 'text_gen', 'label': 'Adapt lyrics',
                'params': {'model': request.text_model, 'prompt': prompt}}
        out = await _run_scheduled_step(step, {'input': ''}, http_request)
        lyrics = (out.get('output') or '').strip()
        if lyrics:
            return lyrics, {'type': 'translate', 'label': 'Adapt lyrics',
                            'status': 'ok', 'engine': 'text_model',
                            'model': request.text_model, 'output': lyrics}

    literal = _argos_translate(transcript, request.source_lang, request.target_lang)
    if literal:
        return literal, {'type': 'translate', 'label': 'Translate lyrics',
                         'status': 'ok', 'engine': 'argostranslate',
                         'output': literal}
    return transcript, {
        'type': 'translate', 'label': 'Translate lyrics', 'status': 'skipped',
        'reason': 'no text_model given and argostranslate is not installed — '
                  'lyrics left in the source language',
        'output': transcript}


async def run_full_music_dub(request: AudioMusicDubRequest, http_request: Request = None):
    """Separate → transcribe → adapt → re-sing → remix.

    Every stage runs for real. Stages that need an optional dependency degrade to a
    documented ``skipped`` state with a reason instead of silently producing a
    placeholder, and the caller can tell from ``complete`` whether the whole chain ran.
    """
    import base64
    import os
    import subprocess
    import tempfile

    from codai.api.audio_stems import (_persist_file, _split_audio,
                                       separate_with_provider)

    steps = []
    warnings = []

    with tempfile.TemporaryDirectory(prefix='codai-musicdub-') as workdir:
        # -- 0. isolate the vocal from the backing track ----------------------
        try:
            raw = base64.b64decode(request.audio.split(',', 1)[1]
                                   if request.audio.startswith('data:') else request.audio)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f'Invalid audio payload: {exc}')

        if request.fallback_mode:
            sep = _split_audio(raw, 'vocals-instrumental', workdir)
        else:
            sep = separate_with_provider(raw, 'vocals-instrumental', workdir)
        paths = {a['name']: a['path'] for a in sep['artifacts']}
        vocals_path, instrumental_path = paths.get('vocals'), paths.get('instrumental')
        if not vocals_path or not instrumental_path:
            raise HTTPException(status_code=500,
                                detail='Stem separation did not return vocals + instrumental')
        steps.append({'step': 0, 'type': 'stems', 'label': 'Isolate vocals and instrumental',
                      'status': 'ok', 'engine': sep['engine'],
                      'limitations': sep.get('limitations', [])})
        if request.fallback_mode:
            warnings.append('stem separation used the best-effort ffmpeg split; '
                            'expect heavy bleed between vocal and backing')

        # -- 1. transcribe the ISOLATED vocal (far better than the full mix) --
        with open(vocals_path, 'rb') as fh:
            vocals_b64 = base64.b64encode(fh.read()).decode()
        stt_step = {
            'type': 'stt',
            'label': 'Transcribe lyrics from the isolated vocal',
            'params': {
                'model': request.audio_model,
                'audio': f'data:audio/wav;base64,{vocals_b64}',
                'language': request.source_lang,
                'prompt': request.notes or None,
                'response_format': 'json',
            },
        }
        stt_out = await _run_scheduled_step(stt_step, {'input': request.notes or ''},
                                            http_request)
        transcript = (stt_out.get('text') or stt_out.get('output') or '').strip()
        steps.append({'step': 1, 'type': 'stt',
                      'label': 'Transcribe lyrics from the isolated vocal',
                      'status': 'ok', **stt_out})
        if not transcript:
            raise HTTPException(status_code=500,
                                detail='No lyrics could be transcribed from the vocal stem')

        # -- 2. adapt the lyrics ----------------------------------------------
        lyrics, translate_step = await _adapt_lyrics(request, transcript, http_request)
        steps.append({'step': 2, **translate_step})
        if translate_step['status'] == 'skipped':
            warnings.append(translate_step['reason'])

        # -- 3. sing the new lyrics in the original singer's voice ------------
        # The isolated vocal doubles as the cloning reference, so the dub keeps the
        # original timbre without the user having to enrol a voice profile.
        from codai.api.voice_clone import _f5tts_clone, _load_voice

        ref_audio_path, ref_text = vocals_path, request.ref_text or transcript
        if request.voice_name:
            meta = _load_voice(request.voice_name)
            if not meta:
                raise HTTPException(status_code=404,
                                    detail=f"Voice '{request.voice_name}' not found")
            ref_audio_path = meta['audio_file']
            ref_text = request.ref_text or meta.get('transcript', '') or transcript

        try:
            sung_bytes = await asyncio.get_event_loop().run_in_executor(
                None, _f5tts_clone, ref_audio_path, ref_text, lyrics,
                request.speed or 1.0, request.seed)
        except ImportError:
            raise HTTPException(
                status_code=501,
                detail='Re-singing the lyrics needs F5-TTS. Run: pip install f5-tts')
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f'Vocal synthesis failed: {exc}')

        new_vocal = os.path.join(workdir, 'new_vocals.wav')
        with open(new_vocal, 'wb') as fh:
            fh.write(sung_bytes)
        steps.append({'step': 3, 'type': 'voice_clone',
                      'label': 'Sing the adapted lyrics in the original voice',
                      'status': 'ok', 'engine': 'f5-tts'})

        # -- 4. optional Seed-VC singing pass ---------------------------------
        converted_path = new_vocal
        if request.sing_convert:
            try:
                from codai.api.voice_convert import _get_wrapper
                wrapper = _get_wrapper()

                def _convert():
                    return wrapper.convert_voice(
                        source=new_vocal, target=vocals_path,
                        diffusion_steps=request.diffusion_steps or 15,
                        length_adjust=1.0, inference_cfg_rate=0.7,
                        f0_condition=True,           # singing mode: keep the melody
                        pitch_shift=request.pitch_shift or 0,
                        stream_output=False)

                audio_out = await asyncio.get_event_loop().run_in_executor(None, _convert)
                if isinstance(audio_out, tuple):
                    audio_out = audio_out[0]
                import numpy as _np
                import soundfile as _sf
                samples = _np.array(audio_out).flatten()
                # Seed-VC can return a near-empty buffer when it finds nothing voiced
                # to track. Only adopt the result once we know it is real audio —
                # writing straight to converted_path would remix silence.
                if samples.size < 1000:
                    raise RuntimeError(
                        f'conversion returned {samples.size} samples — no voiced '
                        'content detected in the vocal stem')
                candidate = os.path.join(workdir, 'converted_vocals.wav')
                _sf.write(candidate, samples, 44100)
                converted_path = candidate
                steps.append({'step': 4, 'type': 'voice_convert',
                              'label': 'Match the original singing voice',
                              'status': 'ok', 'engine': 'seed-vc',
                              'f0_condition': True})
            except ImportError:
                steps.append({'step': 4, 'type': 'voice_convert',
                              'label': 'Match the original singing voice',
                              'status': 'skipped',
                              'reason': 'seed-vc is not installed — using the '
                                        'F5-TTS vocal as-is. Run: pip install seed-vc'})
                warnings.append('seed-vc not installed: the replacement vocal is spoken-'
                                'style TTS rather than pitch-matched singing')
            except Exception as exc:
                steps.append({'step': 4, 'type': 'voice_convert',
                              'label': 'Match the original singing voice',
                              'status': 'failed', 'reason': str(exc)})
                warnings.append(f'singing conversion failed, using the raw vocal: {exc}')
        else:
            steps.append({'step': 4, 'type': 'voice_convert',
                          'label': 'Match the original singing voice',
                          'status': 'skipped', 'reason': 'sing_convert=false'})

        # -- 5. remix over the original instrumental --------------------------
        final_mix = os.path.join(workdir, 'final_mix.wav')
        proc = subprocess.run(
            ['ffmpeg', '-y', '-i', converted_path, '-i', instrumental_path,
             '-filter_complex', 'amix=inputs=2:duration=longest:normalize=0,alimiter',
             final_mix],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise HTTPException(status_code=500,
                                detail=f'Remix failed: {(proc.stderr or "")[-500:]}')
        steps.append({'step': 5, 'type': 'remix',
                      'label': 'Remix the new vocal over the instrumental',
                      'status': 'ok', 'engine': 'ffmpeg'})

        complete = all(s.get('status') == 'ok' for s in steps)
        return {
            'vocals': _persist_file(vocals_path, '.wav', http_request),
            'instrumental': _persist_file(instrumental_path, '.wav', http_request),
            'transcript': transcript,
            'translated_lyrics': lyrics,
            'converted_vocals': _persist_file(converted_path, '.wav', http_request),
            'final_mix': _persist_file(final_mix, '.wav', http_request),
            'steps': steps,
            'warnings': warnings,
            'complete': complete,
        }


@router.post('/v1/pipelines/audio-music-dub', summary="Dub a song into another language")
async def run_audio_music_dub(request: AudioMusicDubRequest, http_request: Request = None):
    """Dub a song into another language while preserving the backing music.

    Splits the track into vocals and instrumental, transcribes and translates the
    lyrics, re-sings/voice-converts the translated vocals, then remixes them over the
    original instrumental. Returns every intermediate stem plus the final mixed result.
    """
    if not request.audio:
        raise HTTPException(status_code=400, detail='Provide audio input')

    result = await run_full_music_dub(request, http_request)
    return {
        'created': int(time.time()),
        'pipeline': 'audio-music-dub',
        # 'complete' = every stage ran; 'degraded' = the mix was produced but at
        # least one stage was skipped or fell back (see `warnings`).
        'status': 'complete' if result['complete'] else 'degraded',
        'warnings': result['warnings'],
        'vocals': result['vocals'],
        'instrumental': result['instrumental'],
        'transcript': result['transcript'],
        'translated_lyrics': result['translated_lyrics'],
        'converted_vocals': result['converted_vocals'],
        'final_mix': result['final_mix'],
        'steps': result['steps'],
        'data': [
            {
                'transcript': result['transcript'],
                'translated_lyrics': result['translated_lyrics'],
                'final_mix': result['final_mix'],
            }
        ],
    }
