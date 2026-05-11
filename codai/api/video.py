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
Video generation and manipulation endpoints for the codai API.

Endpoints:
  POST /v1/video/generations   – t2v | i2v | v2v | ti2v | interp
  POST /v1/video/upscale       – video super-resolution
  POST /v1/video/subtitle      – subtitle generation / burn-in
  POST /v1/video/interpolate   – frame interpolation (increase FPS)
  POST /v1/video/dub           – translation + TTS dubbing
"""

import asyncio
import base64
import io
import os
import subprocess
import tempfile
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request

from codai.models.manager import multi_model_manager
from codai.pydantic.videorequest import (
    VideoGenerationRequest, VideoGenerationResponse,
    VideoUpscaleRequest, VideoSubtitleRequest,
    VideoInterpolateRequest, VideoDubRequest,
    CharacterDialogLine,
)
from codai.api.images import _disable_safety_checker

router = APIRouter()

global_args = None
global_file_path = None

# =============================================================================
# Video generation progress tracking
# =============================================================================
_vid_progress: dict = {
    "current": 0, "total": 0, "active": False,
    "started_at": 0.0, "it_per_s": 0.0,
}

def _vid_progress_reset(total: int):
    _vid_progress["current"] = 0
    _vid_progress["total"] = total
    _vid_progress["active"] = True
    _vid_progress["started_at"] = time.monotonic()
    _vid_progress["it_per_s"] = 0.0

def _vid_progress_done():
    _vid_progress["current"] = _vid_progress["total"]
    _vid_progress["active"] = False

def _vid_progress_step(step: int):
    _vid_progress["current"] = step
    elapsed = time.monotonic() - _vid_progress["started_at"]
    if elapsed > 0 and step > 0:
        _vid_progress["it_per_s"] = round(step / elapsed, 2)


def set_global_args(args):
    global global_args
    global_args = args


def set_global_file_path(path):
    global global_file_path
    global_file_path = path


# =============================================================================
# Shared helpers
# =============================================================================

def _derive_device() -> str:
    if global_args:
        for attr in ('image_vulkan_device', 'vulkan_device'):
            d = getattr(global_args, attr, None)
            if d is not None:
                return f"cuda:{d}"
    return "cuda:0"


def _decode_b64_or_url(data: str) -> bytes:
    if not data:
        return b''
    if data.startswith("data:"):
        _, enc = data.split(",", 1)
        return base64.b64decode(enc)
    if data.startswith("http://") or data.startswith("https://"):
        import urllib.request
        with urllib.request.urlopen(data, timeout=60) as r:
            return r.read()
    return base64.b64decode(data)


def _pil_from_b64(data: str):
    from PIL import Image as PILImage
    return PILImage.open(io.BytesIO(_decode_b64_or_url(data))).convert("RGB")


def _build_url(filename: str, http_request) -> str:
    from codai.api.urlutils import build_file_url
    return build_file_url(filename, http_request)


def _save_file(data: bytes, ext: str, http_request) -> dict:
    filename = f"{uuid.uuid4().hex}.{ext}"
    if global_file_path:
        os.makedirs(global_file_path, exist_ok=True)
        out_path = os.path.join(global_file_path, filename)
        with open(out_path, 'wb') as f:
            f.write(data)
        return {"url": _build_url(filename, http_request)}
    else:
        return {f"b64_{ext}": base64.b64encode(data).decode()}


def _frames_to_mp4(frames, fps: int) -> bytes:
    import imageio, numpy as np
    frames = [np.array(f) for f in frames]
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp_path = tmp.name
    imageio.mimsave(tmp_path, frames, fps=fps, codec='libx264', quality=8)
    with open(tmp_path, 'rb') as f:
        data = f.read()
    os.unlink(tmp_path)
    return data


def _video_bytes_to_path(video_b64: str) -> str:
    """Decode a base64/URL video to a temp file path."""
    raw = _decode_b64_or_url(video_b64)
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp.write(raw)
    tmp.close()
    return tmp.name


# =============================================================================
# Pipeline loading
# =============================================================================

def _detect_pipeline_class(model_name: str, mode: str):
    """Return the appropriate diffusers pipeline class."""
    n = model_name.lower()
    try:
        from diffusers import (
            CogVideoXPipeline, CogVideoXImageToVideoPipeline,
            LTXPipeline, LTXImageToVideoPipeline,
            StableVideoDiffusionPipeline,
            I2VGenXLPipeline,
            AnimateDiffPipeline,
        )
        if 'cogvideox' in n or 'cogvideo' in n:
            return CogVideoXImageToVideoPipeline if (mode in ('i2v', 'ti2v')) else CogVideoXPipeline
        if 'ltx' in n:
            return LTXImageToVideoPipeline if (mode in ('i2v', 'ti2v')) else LTXPipeline
        if 'svd' in n or 'stable-video-diffusion' in n:
            return StableVideoDiffusionPipeline
        if 'i2vgen' in n:
            return I2VGenXLPipeline
        if 'animatediff' in n or 'animateddiff' in n:
            return AnimateDiffPipeline
    except ImportError:
        pass
    try:
        from diffusers import DiffusionPipeline
        return DiffusionPipeline
    except ImportError:
        return None


def _load_video_pipeline(model_name: str, device: str, mode: str):
    import torch, gc
    PClass = _detect_pipeline_class(model_name, mode)
    if PClass is None:
        raise RuntimeError("diffusers not installed: pip install diffusers")
    precision = getattr(global_args, 'image_precision', 'bf16') if global_args else 'bf16'
    dtype_map = {'bf16': torch.bfloat16, 'f16': torch.float16, 'f32': torch.float32}
    torch_dtype = dtype_map.get(precision, torch.bfloat16)
    offload = getattr(global_args, 'offload_strategy', None) if global_args else None

    for attempt in range(3):
        try:
            pipe = PClass.from_pretrained(model_name, torch_dtype=torch_dtype)
            if offload == 'sequential' or attempt >= 2:
                pipe.enable_sequential_cpu_offload()
            elif offload == 'model' or attempt >= 1:
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(device)
            return pipe
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and attempt < 2:
                gc.collect()
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise


# =============================================================================
# Frame interpolation model loading
# =============================================================================

def _load_rife(model_name: str, device: str):
    """Load RIFE frame interpolation model."""
    try:
        # Try rife-ncnn-vulkan first (subprocess)
        import shutil
        if shutil.which('rife-ncnn-vulkan'):
            return ('rife_ncnn', None)
    except Exception:
        pass
    # Fallback: use IFNet from a HF repo
    try:
        from diffusers import IFPipeline  # noqa – just checking if diffusers has it
    except ImportError:
        pass
    return ('rife_hf', model_name)


# =============================================================================
# Generation logic
# =============================================================================

def _build_call_kwargs(request: VideoGenerationRequest) -> dict:
    kw = {}
    if request.prompt:
        kw['prompt'] = request.prompt
    if request.negative_prompt:
        kw['negative_prompt'] = request.negative_prompt
    if request.num_inference_steps:
        kw['num_inference_steps'] = request.num_inference_steps
    if request.guidance_scale:
        kw['guidance_scale'] = request.guidance_scale
    if request.num_frames:
        kw['num_frames'] = request.num_frames
    if request.width and request.height:
        kw['width'] = request.width
        kw['height'] = request.height
    if request.seed is not None:
        import torch
        kw['generator'] = torch.Generator().manual_seed(request.seed)
    return kw


def _apply_camera_motion(kw: dict, camera_motion: str):
    """Inject camera motion hint into pipeline kwargs (model-dependent)."""
    # CogVideoX supports camera_motion natively
    if camera_motion:
        kw['camera_motion'] = camera_motion


def _resolve_character_inputs(request) -> tuple[List[str], List[str]]:
    """Return (flat_image_list, name_list) from any combination of request fields."""
    images: List[str] = []
    names: List[str] = []

    # 1. Expand named saved profiles
    if request.character_profiles:
        try:
            from codai.api.characters import resolve_character_profiles
            images += resolve_character_profiles(request.character_profiles)
            names += list(request.character_profiles)
        except Exception:
            pass

    # 2. Named character slots [{name, images:[...]}, ...]
    if request.characters:
        for slot in request.characters:
            slot_imgs = slot.get('images') or []
            images += slot_imgs
            if slot.get('name'):
                names.append(slot['name'])

    # 3. Legacy flat list
    if request.character_references:
        images += list(request.character_references)
        if request.character_names:
            names += list(request.character_names)

    return images, names


def _apply_character_refs(kw: dict, character_references: List[str], strength: float,
                           names: Optional[List[str]] = None):
    """Apply character reference images to pipeline kwargs."""
    if not character_references:
        return
    imgs = [_pil_from_b64(r) for r in character_references]
    kw['ip_adapter_image'] = imgs[0] if len(imgs) == 1 else imgs
    kw['ip_adapter_scale'] = strength


def _run_pipeline(pipe, kw: dict):
    result = pipe(**kw)
    frames_raw = getattr(result, 'frames', None) or result[0]
    if isinstance(frames_raw, list) and isinstance(frames_raw[0], list):
        return frames_raw[0]
    return list(frames_raw)


def _generate_video(pipe, request: VideoGenerationRequest):
    mode = request.mode or ('i2v' if (request.image or request.init_image)
                             else 'v2v' if request.video else 't2v')
    fps = request.fps or 8
    kw = _build_call_kwargs(request)
    kw.setdefault('num_inference_steps', 25)
    kw.setdefault('guidance_scale', 7.5)
    kw.setdefault('num_frames', 16)

    _vid_progress_reset(kw['num_inference_steps'])

    def _vid_step_cb(pipe, step_index, timestep, callback_kwargs):
        _vid_progress_step(step_index + 1)
        return callback_kwargs

    try:
        kw['callback_on_step_end'] = _vid_step_cb
    except Exception:
        pass

    _apply_camera_motion(kw, request.camera_motion)

    char_images, char_names = _resolve_character_inputs(request)
    if char_images:
        _apply_character_refs(kw, char_images, request.character_strength or 0.8, char_names)
        # Prepend character names to prompt for better conditioning
        if char_names and kw.get('prompt'):
            names_hint = ', '.join(char_names)
            kw['prompt'] = f"{names_hint}. {kw['prompt']}"

    init_src = request.init_image or request.image

    if mode == 'i2v' and init_src:
        kw['image'] = _pil_from_b64(init_src)
        kw.pop('prompt', None)  # SVD doesn't take text

    elif mode == 'ti2v' and init_src:
        kw['image'] = _pil_from_b64(init_src)
        # prompt stays — model uses both

    elif mode == 'interp':
        if not init_src or not request.end_image:
            raise ValueError("interp mode requires both init_image and end_image")
        kw['image'] = _pil_from_b64(init_src)
        kw['image_end'] = _pil_from_b64(request.end_image)
        kw.pop('prompt', None)

    elif mode == 'v2v' and request.video:
        kw['video'] = _decode_b64_or_url(request.video)
        if request.strength is not None:
            kw['strength'] = request.strength

    frames = _run_pipeline(pipe, kw)
    _vid_progress_done()
    return frames, fps


# =============================================================================
# Post-processing helpers
# =============================================================================

def _postprocess_video(mp4_bytes: bytes, request: VideoGenerationRequest,
                       http_request, temp_paths: list) -> bytes:
    """Apply upscale / interpolation / audio / dialog steps to a raw mp4 blob."""
    path = _tmp_write(mp4_bytes, '.mp4')
    temp_paths.append(path)

    if request.upscale_output:
        path = _ffmpeg_upscale(path, request.upscale_factor or 2, temp_paths)

    if request.interpolate_output and request.fps_multiplier:
        path = _rife_interpolate(path, request.fps_multiplier, temp_paths)

    if request.add_audio:
        path = _add_audio_to_video(path, request, temp_paths)

    if request.dialogs:
        path = _process_dialogs(path, request.dialogs,
                                request.lip_sync_method or 'wav2lip', temp_paths)

    if request.generate_subtitles or request.burn_subtitles:
        path = _add_subtitles(path, request, temp_paths)

    with open(path, 'rb') as f:
        return f.read()


def _tmp_write(data: bytes, ext: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _ffmpeg_upscale(path: str, factor: int, temps: list) -> str:
    out = tempfile.mktemp(suffix='_up.mp4')
    temps.append(out)
    scale = f"scale=iw*{factor}:ih*{factor}:flags=lanczos"
    cmd = ['ffmpeg', '-y', '-i', path, '-vf', scale, '-c:a', 'copy', out]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        import logging
        logging.getLogger(__name__).warning(
            "ffmpeg upscale failed (rc=%d): %s", r.returncode, r.stderr.decode(errors='replace')
        )
        return path  # fallback to original if ffmpeg fails
    return out


def _rife_interpolate(path: str, multiplier: int, temps: list) -> str:
    out = tempfile.mktemp(suffix='_rife.mp4')
    temps.append(out)
    import logging, shutil
    _log = logging.getLogger(__name__)
    if shutil.which('rife-ncnn-vulkan'):
        frames_dir = tempfile.mkdtemp()
        out_dir = tempfile.mkdtemp()
        temps += [frames_dir, out_dir]
        r = subprocess.run(['ffmpeg', '-y', '-i', path, f'{frames_dir}/%08d.png'],
                           capture_output=True)
        if r.returncode != 0:
            _log.warning("ffmpeg frame extraction failed: %s", r.stderr.decode(errors='replace'))
        else:
            r = subprocess.run(['rife-ncnn-vulkan', '-i', frames_dir, '-o', out_dir,
                                '-m', 'rife-v4'], capture_output=True)
            if r.returncode != 0:
                _log.warning("rife-ncnn-vulkan failed: %s", r.stderr.decode(errors='replace'))
            else:
                r = subprocess.run(['ffmpeg', '-y', '-r', str(multiplier * 8), '-i',
                                    f'{out_dir}/%08d.png', '-c:v', 'libx264', out],
                                   capture_output=True)
                if r.returncode != 0:
                    _log.warning("ffmpeg reassembly failed: %s", r.stderr.decode(errors='replace'))
                elif os.path.exists(out):
                    return out
    # Simple ffmpeg minterpolate fallback
    cmd = ['ffmpeg', '-y', '-i', path, '-filter:v',
           f'minterpolate=fps={multiplier * 8}', '-c:a', 'copy', out]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        _log.warning("ffmpeg minterpolate failed: %s", r.stderr.decode(errors='replace'))
        return path
    return out


def _add_audio_to_video(path: str, request: VideoGenerationRequest,
                         temps: list) -> str:
    out = tempfile.mktemp(suffix='_audio.mp4')
    temps.append(out)

    if request.audio_file:
        audio_path = _tmp_write(_decode_b64_or_url(request.audio_file), '.wav')
        temps.append(audio_path)
    elif request.tts_text:
        audio_path = _generate_tts(request.tts_text, request.tts_voice,
                                     request.tts_speed or 1.0, temps)
    else:
        return path  # nothing to add

    if not audio_path or not os.path.exists(audio_path):
        return path

    cmd = ['ffmpeg', '-y', '-i', path, '-i', audio_path,
           '-c:v', 'copy', '-c:a', 'aac', '-shortest', out]
    r = subprocess.run(cmd, capture_output=True)
    return out if r.returncode == 0 else path


def _generate_tts(text: str, voice: Optional[str], speed: float,
                   temps: list) -> Optional[str]:
    """Quick TTS using kokoro or edge-tts — returns wav file path."""
    try:
        import edge_tts, asyncio as _aio
        voice_id = voice or 'en-US-JennyNeural'
        out = tempfile.mktemp(suffix='.mp3')
        temps.append(out)
        tts = edge_tts.Communicate(text, voice_id, rate=f"+{int((speed - 1) * 100)}%")
        _aio.get_event_loop().run_until_complete(tts.save(out))
        return out
    except ImportError:
        pass
    try:
        from kokoro import KPipeline
        import soundfile as sf, numpy as np
        pipe = KPipeline(lang_code='a')
        audio, sr = pipe(text, voice=voice or 'af_sky', speed=speed)
        out = tempfile.mktemp(suffix='.wav')
        temps.append(out)
        sf.write(out, np.concatenate(audio), sr)
        return out
    except ImportError:
        pass
    return None


def _get_audio_duration(path: str) -> float:
    """Return audio/video duration in seconds via ffprobe."""
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _generate_tts_for_line(line: CharacterDialogLine, temps: list) -> Optional[str]:
    """Generate TTS for a single dialog line, using the voice profile's reference audio if available."""
    voice = line.voice
    text = line.text
    speed = line.speed or 1.0
    lang = line.lang

    # Try to load voice profile reference audio first (for kokoro/RVC cloning)
    ref_audio = None
    if voice:
        try:
            from codai.api.voice_clone import _load_voice, _voice_path
            meta = _load_voice(voice)
            audio_file = meta.get('audio_file') or meta.get('audio_path')
            if audio_file and os.path.isfile(audio_file):
                ref_audio = audio_file
        except Exception:
            pass

    # edge_tts with voice id
    try:
        import edge_tts, asyncio as _aio
        voice_id = voice if (voice and not ref_audio) else (
            f"{lang.split('-')[0]}-" if lang else 'en-'
        ) + 'US-JennyNeural'
        if not voice or ref_audio:
            voice_id = 'en-US-JennyNeural'
        out = tempfile.mktemp(suffix='.mp3')
        temps.append(out)
        tts = edge_tts.Communicate(text, voice_id, rate=f"+{int((speed - 1) * 100)}%")
        _aio.get_event_loop().run_until_complete(tts.save(out))
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return out
    except Exception:
        pass

    # kokoro with optional reference voice
    try:
        from kokoro import KPipeline
        import soundfile as sf, numpy as np
        lang_code = 'a'
        if lang:
            lang_code = lang.split('-')[0][:1].lower()
        pipe = KPipeline(lang_code=lang_code)
        kokoro_voice = voice if (voice and not ref_audio) else 'af_sky'
        audio, sr = pipe(text, voice=kokoro_voice, speed=speed)
        out = tempfile.mktemp(suffix='.wav')
        temps.append(out)
        sf.write(out, np.concatenate(audio), sr)
        return out
    except Exception:
        pass

    return None


def _mix_dialog_audio(clips: list, temps: list) -> Optional[str]:
    """
    Mix a list of (start_time_sec, audio_path) clips into one audio file.
    Uses ffmpeg adelay + amix. Returns path to mixed audio or None.
    """
    if not clips:
        return None
    if len(clips) == 1:
        return clips[0][1]

    # Build complex filter: delay each stream, then amix
    filter_parts = []
    inputs = []
    for i, (start_sec, apath) in enumerate(clips):
        inputs += ['-i', apath]
        delay_ms = int(start_sec * 1000)
        filter_parts.append(f'[{i}]adelay={delay_ms}|{delay_ms}[a{i}]')

    mix_inputs = ''.join(f'[a{i}]' for i in range(len(clips)))
    filter_parts.append(f'{mix_inputs}amix=inputs={len(clips)}:duration=longest:dropout_transition=0[out]')
    filter_str = ';'.join(filter_parts)

    out = tempfile.mktemp(suffix='_mixed.wav')
    temps.append(out)
    cmd = ['ffmpeg', '-y'] + inputs + ['-filter_complex', filter_str, '-map', '[out]', out]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0 and os.path.exists(out):
        return out

    # Fallback: simple concatenation (ignore timing)
    import logging
    logging.getLogger(__name__).warning(
        "ffmpeg amix failed (rc=%d), falling back to concat: %s",
        r.returncode, r.stderr.decode(errors='replace'))
    list_path = tempfile.mktemp(suffix='.txt')
    temps.append(list_path)
    with open(list_path, 'w') as f:
        for _, apath in clips:
            f.write(f"file '{apath}'\n")
    out2 = tempfile.mktemp(suffix='_cat.wav')
    temps.append(out2)
    r2 = subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                         '-i', list_path, out2], capture_output=True)
    return out2 if r2.returncode == 0 else None


def _apply_lipsync(video_path: str, audio_path: str, method: str, temps: list) -> str:
    """Apply lip sync to video using wav2lip or sadtalker. Returns new video path."""
    import logging, shutil
    _log = logging.getLogger(__name__)
    out = tempfile.mktemp(suffix='_lipsync.mp4')
    temps.append(out)

    if method == 'wav2lip':
        wav2lip_bin = shutil.which('wav2lip') or shutil.which('Wav2Lip')
        if wav2lip_bin:
            cmd = [wav2lip_bin, '--face', video_path, '--audio', audio_path, '--outfile', out]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode == 0 and os.path.exists(out):
                return out
            _log.warning("wav2lip failed (rc=%d): %s", r.returncode, r.stderr.decode(errors='replace'))
        else:
            # Try wav2lip Python API
            try:
                import inference as wav2lip_inference  # noqa
                wav2lip_inference.main(face=video_path, audio=audio_path, outfile=out)
                if os.path.exists(out):
                    return out
            except Exception as e:
                _log.warning("wav2lip Python API failed: %s", e)

    elif method == 'sadtalker':
        sadtalker_bin = shutil.which('sadtalker')
        if sadtalker_bin:
            cmd = [sadtalker_bin, '--driven_audio', audio_path, '--source_video', video_path,
                   '--result_dir', os.path.dirname(out)]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode == 0:
                # Find output file
                out_dir = os.path.dirname(out)
                for f in sorted(os.listdir(out_dir)):
                    if f.endswith('.mp4'):
                        return os.path.join(out_dir, f)
            _log.warning("sadtalker failed (rc=%d): %s", r.returncode, r.stderr.decode(errors='replace'))

    # Fallback: just mux audio onto video without lip sync
    _log.warning("Lip sync unavailable (%s not found/working), merging audio only", method)
    out_fallback = tempfile.mktemp(suffix='_nosync.mp4')
    temps.append(out_fallback)
    cmd = ['ffmpeg', '-y', '-i', video_path, '-i', audio_path,
           '-c:v', 'copy', '-c:a', 'aac', '-shortest', out_fallback]
    r = subprocess.run(cmd, capture_output=True)
    return out_fallback if r.returncode == 0 else video_path


def _process_dialogs(path: str, dialogs: list, lip_sync_method: str, temps: list) -> str:
    """
    Generate TTS for each dialog line, mix with correct timing, apply lip sync to full video.
    Returns new video path.
    """
    import logging
    _log = logging.getLogger(__name__)

    # First pass: generate TTS audio for each line and calculate timing
    clips = []  # (start_sec, audio_path)
    cursor = 0.0
    for line in dialogs:
        if not line.text.strip():
            continue
        audio_path = _generate_tts_for_line(line, temps)
        if not audio_path or not os.path.exists(audio_path):
            _log.warning("TTS generation failed for dialog line: %r", line.text[:40])
            continue

        if line.start_time is not None:
            start = float(line.start_time)
        else:
            start = cursor

        duration = _get_audio_duration(audio_path)
        clips.append((start, audio_path))
        cursor = start + duration + 0.1  # small gap between sequential lines

    if not clips:
        return path

    # Mix all clips into one audio track
    mixed_audio = _mix_dialog_audio(clips, temps)
    if not mixed_audio or not os.path.exists(mixed_audio):
        return path

    # Determine if any line wants lip sync
    wants_lip_sync = any(getattr(line, 'lip_sync', True) for line in dialogs if line.text.strip())

    if wants_lip_sync and lip_sync_method:
        return _apply_lipsync(path, mixed_audio, lip_sync_method, temps)

    # No lip sync — just mux audio
    out = tempfile.mktemp(suffix='_dialog.mp4')
    temps.append(out)
    cmd = ['ffmpeg', '-y', '-i', path, '-i', mixed_audio,
           '-c:v', 'copy', '-c:a', 'aac', '-shortest', out]
    r = subprocess.run(cmd, capture_output=True)
    return out if r.returncode == 0 else path


def _add_subtitles(path: str, request: VideoGenerationRequest, temps: list) -> str:
    """Transcribe video audio → subtitles, optionally burn them in."""
    try:
        import whisper
    except ImportError:
        return path  # skip if whisper not available

    srt_path = _whisper_transcribe(path, request.subtitle_language,
                                    request.whisper_model, temps)
    if not srt_path:
        return path

    if request.translate_subtitles and request.subtitle_target_lang:
        srt_path = _translate_srt(srt_path, request.subtitle_target_lang, temps)

    if request.burn_subtitles:
        out = tempfile.mktemp(suffix='_sub.mp4')
        temps.append(out)
        # Use ASS-style subtitle filter for better styling
        style = request.subtitle_style or 'default'
        vf = f"subtitles={srt_path}"
        if style == 'karaoke':
            vf = f"ass={srt_path}"
        cmd = ['ffmpeg', '-y', '-i', path, '-vf', vf, '-c:a', 'copy', out]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            return out

    return path


def _whisper_transcribe(video_path: str, language: Optional[str],
                         model_name: Optional[str], temps: list) -> Optional[str]:
    try:
        import whisper as _whisper
        model = _whisper.load_model(model_name or 'base')
        result = model.transcribe(video_path, language=language)
        srt_path = tempfile.mktemp(suffix='.srt')
        temps.append(srt_path)
        with open(srt_path, 'w') as f:
            for i, seg in enumerate(result['segments'], 1):
                def _fmt(t):
                    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
                    return f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')
                f.write(f"{i}\n{_fmt(seg['start'])} --> {_fmt(seg['end'])}\n{seg['text'].strip()}\n\n")
        return srt_path
    except Exception:
        return None


def _translate_srt(srt_path: str, target_lang: str, temps: list) -> str:
    """Translate SRT using argostranslate or fall back to original."""
    try:
        import argostranslate.package, argostranslate.translate
        with open(srt_path) as f:
            content = f.read()
        lines = content.split('\n')
        translated = []
        for line in lines:
            if line and not line[0].isdigit() and '-->' not in line:
                line = argostranslate.translate.translate(line, 'en', target_lang)
            translated.append(line)
        out = tempfile.mktemp(suffix='.srt')
        temps.append(out)
        with open(out, 'w') as f:
            f.write('\n'.join(translated))
        return out
    except Exception:
        return srt_path


# =============================================================================
# Progress endpoint
# =============================================================================

@router.get("/v1/video/progress")
async def get_video_progress():
    """Return current video generation step progress including speed."""
    elapsed = time.monotonic() - _vid_progress["started_at"] if _vid_progress["active"] else 0.0
    return {
        "current":  _vid_progress["current"],
        "total":    _vid_progress["total"],
        "active":   _vid_progress["active"],
        "pct":      int(_vid_progress["current"] / _vid_progress["total"] * 100)
                    if _vid_progress["total"] > 0 else 0,
        "it_per_s": _vid_progress["it_per_s"],
        "elapsed":  round(elapsed, 1),
    }


# =============================================================================
# Main generation endpoint
# =============================================================================

@router.post("/v1/video/generations", response_model=VideoGenerationResponse)
async def video_generations(request: VideoGenerationRequest,
                             http_request: Request = None):
    """
    Generate video.

    Modes (request.mode):
      t2v   – text-to-video
      i2v   – image-to-video (init_image required)
      v2v   – video-to-video (video required)
      ti2v  – text + image → video (prompt is primary driver)
      interp – frame interpolation (init_image + end_image required)
    """
    if not request.model:
        raise HTTPException(status_code=400, detail="model is required")

    # Infer mode from inputs if not set
    if not request.mode or request.mode == 't2v':
        if request.init_image or request.image:
            request.mode = 'ti2v' if request.prompt else 'i2v'
        elif request.end_image:
            request.mode = 'interp'
        elif request.video:
            request.mode = 'v2v'

    model_info = multi_model_manager.request_model(request.model, model_type="video")
    model_name = model_info.get('model_name')
    if not model_name:
        err = model_info.get('error', f"Model '{request.model}' not found")
        raise HTTPException(status_code=404, detail=err)

    model_key = model_info['model_key']
    pipe = model_info.get('model_object')

    if pipe is None:
        device = _derive_device()
        try:
            pipe = await asyncio.get_event_loop().run_in_executor(
                None, _load_video_pipeline, model_name, device, request.mode)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load video model: {e}")
        multi_model_manager.models[model_key] = pipe
        multi_model_manager.current_model_key = model_key

    if getattr(request, 'disable_safety_checker', False):
        _disable_safety_checker(pipe)

    try:
        frames, fps = await asyncio.get_event_loop().run_in_executor(
            None, _generate_video, pipe, request)
    except Exception as e:
        _vid_progress_done()
        raise HTTPException(status_code=500, detail=f"Video generation failed: {e}")

    # Encode raw frames to MP4
    try:
        import imageio, numpy as np
        frame_np = [np.array(f) for f in frames]
        mp4_bytes = _frames_to_mp4(frame_np, fps)
    except ImportError:
        raise HTTPException(status_code=500,
                            detail="imageio[ffmpeg] required: pip install imageio[ffmpeg]")

    # Post-processing pipeline (upscale, audio, subtitles, …)
    temps = []
    try:
        needs_post = any([
            request.upscale_output,
            request.interpolate_output,
            request.add_audio,
            bool(request.dialogs),
            request.generate_subtitles,
            request.burn_subtitles,
        ])
        if needs_post:
            mp4_bytes = await asyncio.get_event_loop().run_in_executor(
                None, _postprocess_video, mp4_bytes, request, http_request, temps)
    finally:
        for t in temps:
            try:
                if os.path.isfile(t):
                    os.unlink(t)
                elif os.path.isdir(t):
                    import shutil
                    shutil.rmtree(t, ignore_errors=True)
            except Exception:
                pass

    result = _save_file(mp4_bytes, 'mp4', http_request)

    try:
        from codai.api.archive import archive_manager
        asyncio.get_event_loop().create_task(asyncio.to_thread(
            archive_manager.save_generation,
            "video", "/v1/video/generations",
            request.model,
            request.prompt or "",
            {
                "mode": request.mode,
                "num_frames": request.num_frames,
                "fps": request.fps,
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
            },
            [(mp4_bytes, "mp4")],
        ))
    except Exception:
        pass

    return VideoGenerationResponse(created=int(time.time()), data=[result])


# =============================================================================
# Video upscale endpoint
# =============================================================================

@router.post("/v1/video/upscale")
async def video_upscale(request: VideoUpscaleRequest, http_request: Request = None):
    """
    Upscale a video using ffmpeg lanczos or Real-ESRGAN.
    The model field can be 'realesrgan' or any registered video_upscaling model.
    """
    raw = _decode_b64_or_url(request.video)
    temps = []
    try:
        in_path = _tmp_write(raw, '.mp4')
        temps.append(in_path)
        out_path = await asyncio.get_event_loop().run_in_executor(
            None, _ffmpeg_upscale, in_path, request.upscale_factor or 2, temps)
        with open(out_path, 'rb') as f:
            out_bytes = f.read()
    finally:
        for t in temps:
            try:
                os.unlink(t)
            except Exception:
                pass

    result = _save_file(out_bytes, 'mp4', http_request)
    return {"created": int(time.time()), "data": [result]}


# =============================================================================
# Subtitle generation endpoint
# =============================================================================

@router.post("/v1/video/subtitle")
async def video_subtitle(request: VideoSubtitleRequest, http_request: Request = None):
    """
    Generate subtitles for a video.
    Returns SRT/VTT text or a URL to the video with burned-in subtitles.
    """
    raw = _decode_b64_or_url(request.video)
    temps = []
    try:
        in_path = _tmp_write(raw, '.mp4')
        temps.append(in_path)

        srt_path = await asyncio.get_event_loop().run_in_executor(
            None, _whisper_transcribe, in_path, request.language, None, temps)
        if not srt_path:
            raise HTTPException(status_code=500,
                                detail="Whisper not installed: pip install openai-whisper")

        if request.translate and request.target_lang:
            srt_path = await asyncio.get_event_loop().run_in_executor(
                None, _translate_srt, srt_path, request.target_lang, temps)

        if request.burn:
            out_path = tempfile.mktemp(suffix='_sub.mp4')
            temps.append(out_path)
            cmd = ['ffmpeg', '-y', '-i', in_path,
                   '-vf', f'subtitles={srt_path}',
                   '-c:a', 'copy', out_path]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                raise HTTPException(status_code=500,
                                    detail=f"ffmpeg subtitle burn failed: {r.stderr.decode()}")
            with open(out_path, 'rb') as f:
                out_bytes = f.read()
            result = _save_file(out_bytes, 'mp4', http_request)
            return {"created": int(time.time()), "data": [result]}

        # Return raw subtitle text
        with open(srt_path) as f:
            srt_text = f.read()
        return {"created": int(time.time()), "data": [{"text": srt_text, "format": "srt"}]}

    finally:
        for t in temps:
            try:
                os.unlink(t)
            except Exception:
                pass


# =============================================================================
# Frame interpolation endpoint
# =============================================================================

@router.post("/v1/video/interpolate")
async def video_interpolate(request: VideoInterpolateRequest, http_request: Request = None):
    """
    Increase video FPS via frame interpolation.
    Supports rife-ncnn-vulkan (if installed) or ffmpeg minterpolate fallback.
    """
    temps = []
    try:
        if request.video:
            raw = _decode_b64_or_url(request.video)
            in_path = _tmp_write(raw, '.mp4')
            temps.append(in_path)
        elif request.init_image and request.end_image:
            # Build a 2-frame video from the two images, then interpolate
            from PIL import Image as PILImage
            import numpy as np, imageio
            img1 = _pil_from_b64(request.init_image)
            img2 = _pil_from_b64(request.end_image)
            in_path = tempfile.mktemp(suffix='.mp4')
            temps.append(in_path)
            imageio.mimsave(in_path, [np.array(img1), np.array(img2)],
                            fps=2, codec='libx264')
        else:
            raise HTTPException(status_code=400,
                                detail="Provide either video or init_image + end_image")

        mult = request.fps_multiplier or 2
        out_path = await asyncio.get_event_loop().run_in_executor(
            None, _rife_interpolate, in_path, mult, temps)
        with open(out_path, 'rb') as f:
            out_bytes = f.read()

    finally:
        for t in temps:
            try:
                os.unlink(t)
            except Exception:
                pass

    result = _save_file(out_bytes, 'mp4', http_request)
    return {"created": int(time.time()), "data": [result]}


# =============================================================================
# Video dubbing endpoint
# =============================================================================

@router.post("/v1/video/dub")
async def video_dub(request: VideoDubRequest, http_request: Request = None):
    """
    Translate and re-dub a video.
    Pipeline: Whisper → translate → TTS → merge audio → (optional) lip sync.
    """
    raw = _decode_b64_or_url(request.video)
    temps = []
    try:
        in_path = _tmp_write(raw, '.mp4')
        temps.append(in_path)

        # 1. Transcribe
        srt_path = await asyncio.get_event_loop().run_in_executor(
            None, _whisper_transcribe, in_path, request.source_lang, None, temps)
        if not srt_path:
            raise HTTPException(status_code=500, detail="Whisper not available")

        # 2. Translate subtitles
        if request.target_lang:
            srt_path = await asyncio.get_event_loop().run_in_executor(
                None, _translate_srt, srt_path, request.target_lang, temps)

        # 3. Generate dubbed audio from translated text
        with open(srt_path) as f:
            srt_content = f.read()
        plain_text = '\n'.join(
            line for line in srt_content.split('\n')
            if line and not line[0].isdigit() and '-->' not in line
        )
        audio_path = await asyncio.get_event_loop().run_in_executor(
            None, _generate_tts, plain_text, None, 1.0, temps)

        if not audio_path:
            raise HTTPException(status_code=500, detail="TTS generation failed (install edge-tts or kokoro)")

        # 4. Merge dubbed audio with video
        out_path = tempfile.mktemp(suffix='_dubbed.mp4')
        temps.append(out_path)
        cmd = ['ffmpeg', '-y', '-i', in_path, '-i', audio_path,
               '-map', '0:v', '-map', '1:a',
               '-c:v', 'copy', '-c:a', 'aac', '-shortest', out_path]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            raise HTTPException(status_code=500,
                                detail=f"Audio merge failed: {r.stderr.decode()}")

        # 5. Burn subtitles if requested
        if request.burn_subtitles:
            sub_out = tempfile.mktemp(suffix='_sub.mp4')
            temps.append(sub_out)
            cmd2 = ['ffmpeg', '-y', '-i', out_path,
                    '-vf', f'subtitles={srt_path}',
                    '-c:a', 'copy', sub_out]
            r2 = subprocess.run(cmd2, capture_output=True)
            if r2.returncode == 0:
                out_path = sub_out

        with open(out_path, 'rb') as f:
            out_bytes = f.read()

    finally:
        for t in temps:
            try:
                os.unlink(t)
            except Exception:
                pass

    result = _save_file(out_bytes, 'mp4', http_request)
    return {"created": int(time.time()), "data": [result]}