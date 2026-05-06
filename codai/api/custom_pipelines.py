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
    "audio_gen":      ("codai.api.audio_gen",     "audio_generate",         None),
    "voice_clone":    ("codai.api.voice_clone",   "clone_voice",            None),
    "voice_convert":  ("codai.api.voice_convert", "convert_voice",          None),
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
    "audio_gen":      "Audio/Music Generation",
    "voice_clone":    "Voice Clone (TTS)",
    "voice_convert":  "Voice Convert (SVC)",
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
    "video_interp":   [("model","text","Model ID"),("video","ref","Source video"),("fps_multiplier","number","FPS multiplier","2")],
    "video_dub":      [("model","text","Model ID"),("video","ref","Source video"),("target_lang","text","Target language"),("source_lang","text","Source language"),("burn_subtitles","checkbox","Burn subtitles")],
    "tts":            [("model","text","Model ID"),("input","textarea","Text ({{stepN.output}})"),("voice","text","Voice","af_sarah"),("speed","number","Speed","1.0")],
    "audio_gen":      [("model","text","Model ID"),("prompt","textarea","Prompt"),("duration","number","Duration (s)","10"),("temperature","number","Temperature","1.0")],
    "voice_clone":    [("text","textarea","Text to synthesize"),("voice_name","text","Voice profile name"),("ref_text","text","Reference transcript"),("speed","number","Speed","1.0")],
    "voice_convert":  [("source_audio","ref","Source audio"),("voice_name","text","Voice profile name"),("f0_condition","checkbox","Singing mode"),("pitch_shift","number","Pitch shift","0"),("diffusion_steps","number","Steps","10")],
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
    # text_gen
    if 'choices' in r:
        out['output'] = r['choices'][0].get('message', {}).get('content', '') if r['choices'] else ''
    # image/video/audio with data array
    if 'data' in r and r['data']:
        item = r['data'][0]
        if isinstance(item, dict):
            out['url'] = item.get('url', '')
            for k, v in item.items():
                out[k] = v
    # tts audio field
    if 'audio' in r:
        out['audio'] = r['audio']
        out['output'] = r['audio']
    return out


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


async def _execute_pipeline(pipeline_def: Dict, pipeline_input: str, http_request) -> Dict:
    """Execute all steps of a pipeline definition."""
    context = {'input': pipeline_input}
    steps_output = []

    for i, step in enumerate(pipeline_def.get('steps', [])):
        try:
            out = await _run_step(step, context, http_request)
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


@router.get('/v1/pipelines/custom')
async def list_custom_pipelines():
    """List all saved custom pipeline definitions."""
    from codai.admin.routes import config_manager
    if config_manager is None:
        return {'pipelines': []}
    return {'pipelines': config_manager.pipelines_data}


@router.get('/v1/pipelines/step-types')
async def list_step_types():
    """List available step types with their parameter schemas."""
    return {
        'step_types': [
            {'type': t, 'label': STEP_TYPE_LABELS[t], 'params': STEP_PARAMS.get(t, [])}
            for t in STEP_TYPES
        ]
    }


@router.post('/v1/pipelines/custom')
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


@router.put('/v1/pipelines/custom/{pipeline_id}')
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


@router.delete('/v1/pipelines/custom/{pipeline_id}')
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


@router.post('/v1/pipelines/custom/{pipeline_id}/run')
async def run_custom_pipeline(pipeline_id: str, body: PipelineRunRequest, http_request: Request = None):
    """Execute a saved custom pipeline."""
    from codai.admin.routes import config_manager
    if config_manager is None:
        raise HTTPException(status_code=503, detail='Config manager not available')
    pipeline_def = next((p for p in config_manager.pipelines_data if p.get('id') == pipeline_id), None)
    if not pipeline_def:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    return await _execute_pipeline(pipeline_def, body.input or '', http_request)


@router.post('/v1/pipelines/run')
async def run_inline_pipeline(pipeline: PipelineDefinition, http_request: Request = None):
    """Execute an inline pipeline definition without saving it."""
    return await _execute_pipeline(pipeline.model_dump(), '', http_request)
