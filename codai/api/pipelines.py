"""
Server-side pipeline endpoints — multi-step generation chains.

POST /v1/pipelines/image-to-video   — generate image then animate it
POST /v1/pipelines/story            — LLM script → images → video → TTS narration
POST /v1/pipelines/video-dub        — transcribe → translate → TTS dub → burn subtitles
POST /v1/pipelines/audio-dub        — transcribe audio/video → translate → clone voice → replace audio
"""

import asyncio
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

router = APIRouter()

# ---------------------------------------------------------------------------
# Helpers — thin wrappers that call the existing endpoint logic directly
# ---------------------------------------------------------------------------

async def _post_json(path: str, body: dict, http_request: Request):
    """Issue a pipeline sub-request to another coderai API endpoint, returning the
    parsed JSON result.

    A pipeline chains several modalities (image, video, text, TTS, transcription),
    and each model may live on a DIFFERENT engine. So we must NOT call the handlers
    in-process (that would force every step onto whichever engine received the
    pipeline request). Route each sub-step through the front (the single API), which
    dispatches it to the engine that owns that model; falls back to in-process in
    single-process mode. Mirrors codai.api.characters / environments."""
    import json as _json
    from fastapi import HTTPException
    from codai.broker.asgi_bridge import execute_api_request
    resp = await execute_api_request(
        http_request, method="POST", path=path,
        headers={"Content-Type": "application/json"},
        body=_json.dumps(body).encode())
    status = resp.get("status_code", 500)
    raw = resp.get("body") or b""
    if status >= 400:
        try:
            detail = _json.loads(raw).get("detail", raw.decode("utf-8", "replace"))
        except Exception:
            detail = raw.decode("utf-8", "replace")
        raise HTTPException(status_code=status, detail=f"{path} failed: {detail}")
    try:
        return _json.loads(raw) if raw else {}
    except Exception as e:
        raise HTTPException(status_code=502,
                            detail=f"{path} returned a non-JSON response: {e}")


def _img_url(result) -> str:
    """Extract URL from an image generation result dict."""
    data = result.get('data', [{}])
    item = data[0] if data else {}
    return item.get('url') or ('data:image/png;base64,' + item['b64_json'] if item.get('b64_json') else None)


def _vid_url(result) -> str:
    data = result.get('data', [{}])
    item = data[0] if data else {}
    return item.get('url') or ('data:video/mp4;base64,' + item['b64_mp4'] if item.get('b64_mp4') else None)


def _aud_url(result) -> str:
    if isinstance(result, dict):
        if result.get('audio'):
            return 'data:audio/mp3;base64,' + result['audio']
        data = result.get('data', [{}])
        item = data[0] if data else {}
        if item.get('url'):
            return item['url']
        for k, v in item.items():
            if k.startswith('b64_'):
                return f'data:audio/{k[4:]};base64,{v}'
    return None


# ---------------------------------------------------------------------------
# Pipeline 1: Image → Video
# ---------------------------------------------------------------------------

class ImageToVideoPipelineRequest(BaseModel):
    prompt: str
    image_model: str
    video_model: str
    # image params
    image_size: Optional[str] = "1024x1024"
    image_steps: Optional[int] = None
    image_cfg: Optional[float] = None
    image_seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    # video params
    num_frames: Optional[int] = 16
    fps: Optional[int] = 8
    num_inference_steps: Optional[int] = 25
    guidance_scale: Optional[float] = 7.5
    video_seed: Optional[int] = None
    camera_motion: Optional[str] = None
    # audio
    add_audio: Optional[bool] = False
    audio_type: Optional[str] = None
    audio_prompt: Optional[str] = None
    # post
    upscale_output: Optional[bool] = False
    upscale_factor: Optional[int] = 2
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


@router.post("/v1/pipelines/image-to-video", summary="Image-to-video pipeline")
async def pipeline_image_to_video(request: ImageToVideoPipelineRequest, http_request: Request = None):
    """Generate an image then animate it into a video."""
    steps = []

    # Step 1: generate image
    img_body = {
        "model": request.image_model,
        "prompt": request.prompt,
        "size": request.image_size,
        "response_format": "url",
    }
    if request.image_steps:   img_body["steps"] = request.image_steps
    if request.image_cfg:     img_body["guidance_scale"] = request.image_cfg
    if request.image_seed:    img_body["seed"] = request.image_seed
    if request.negative_prompt: img_body["negative_prompt"] = request.negative_prompt

    try:
        img_result = await _post_json('/v1/images/generations', img_body, http_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

    img_url = _img_url(img_result if isinstance(img_result, dict) else img_result.__dict__)
    if not img_url:
        raise HTTPException(status_code=500, detail="Image generation returned no image")
    steps.append({"step": "image", "url": img_url})

    # Step 2: animate image → video
    vid_body = {
        "model": request.video_model,
        "mode": "i2v",
        "prompt": request.prompt,
        "init_image": img_url,
        "num_frames": request.num_frames,
        "fps": request.fps,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "response_format": "url",
    }
    if request.video_seed:    vid_body["seed"] = request.video_seed
    if request.camera_motion: vid_body["camera_motion"] = request.camera_motion
    if request.add_audio:
        vid_body["add_audio"] = True
        vid_body["audio_type"] = request.audio_type
        vid_body["audio_prompt"] = request.audio_prompt
    if request.upscale_output:
        vid_body["upscale_output"] = True
        vid_body["upscale_factor"] = request.upscale_factor

    try:
        vid_result = await _post_json('/v1/video/generations', vid_body, http_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video generation failed: {e}")

    vid_url = _vid_url(vid_result if isinstance(vid_result, dict) else vid_result.__dict__)
    steps.append({"step": "video", "url": vid_url})

    return {
        "created": int(time.time()),
        "pipeline": "image-to-video",
        "steps": steps,
        "data": [{"url": vid_url, "image_url": img_url}],
    }


# ---------------------------------------------------------------------------
# Pipeline 2: Video Dub
# ---------------------------------------------------------------------------

class VideoDubPipelineRequest(BaseModel):
    model: str
    video: str                          # base64 or URL
    target_lang: str
    source_lang: Optional[str] = None
    voice_clone: Optional[bool] = False
    burn_subtitles: Optional[bool] = True
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


@router.post("/v1/pipelines/video-dub", summary="Video dubbing pipeline")
async def pipeline_video_dub(request: VideoDubPipelineRequest, http_request: Request = None):
    """Transcribe → translate → TTS dub → burn subtitles."""
    body = {
        "model": request.model,
        "video": request.video,
        "target_lang": request.target_lang,
        "source_lang": request.source_lang,
        "voice_clone": request.voice_clone,
        "burn_subtitles": request.burn_subtitles,
        "response_format": request.response_format,
    }
    try:
        result = await _post_json('/v1/video/dub', body, http_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video dub failed: {e}")

    vid_url = _vid_url(result if isinstance(result, dict) else result.__dict__)
    return {
        "created": int(time.time()),
        "pipeline": "video-dub",
        "data": [{"url": vid_url}],
    }


# ---------------------------------------------------------------------------
# Pipeline 3: Full Story (LLM → images → video → TTS narration)
# ---------------------------------------------------------------------------

class StoryPipelineRequest(BaseModel):
    story: str
    text_model: str
    image_model: str
    video_model: str
    tts_model: Optional[str] = None
    tts_voice: Optional[str] = "af_sarah"
    num_scenes: Optional[int] = 3
    num_frames: Optional[int] = 16
    fps: Optional[int] = 8
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


@router.post("/v1/pipelines/story", summary="Story pipeline (multi-scene)")
async def pipeline_story(request: StoryPipelineRequest, http_request: Request = None):
    """LLM generates script → image per scene → animate first scene → optional TTS narration."""
    n = min(request.num_scenes or 3, 6)

    # Step 1: LLM script
    try:
        script_result = await _post_json('/v1/chat/completions', {
            "model": request.text_model,
            "messages": [{"role": "user", "content":
                f"Write a {n}-scene visual script for this story. "
                f"For each scene write exactly: SCENE X: [brief visual description, one sentence]. "
                f"Story: {request.story}"}],
            "stream": False,
        }, http_request)
        if hasattr(script_result, 'body'):
            import json
            script_result = json.loads(script_result.body)
        script_text = script_result['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Script generation failed: {e}")

    import re
    scenes = re.findall(r'SCENE \d+:\s*(.+)', script_text) or [request.story]
    scenes = scenes[:n]

    steps = [{"step": "script", "text": script_text, "scenes": scenes}]

    # Step 2: image per scene (parallel)
    async def _gen_image(desc):
        try:
            r = await _post_json('/v1/images/generations', {
                "model": request.image_model,
                "prompt": desc,
                "response_format": "url",
            }, http_request)
            return _img_url(r if isinstance(r, dict) else r.__dict__)
        except Exception:
            return None

    img_urls = await asyncio.gather(*[_gen_image(s) for s in scenes])
    img_urls = [u for u in img_urls if u]
    steps.append({"step": "images", "urls": img_urls})

    if not img_urls:
        raise HTTPException(status_code=500, detail="All image generations failed")

    # Step 3: animate first scene
    try:
        vid_result = await _post_json('/v1/video/generations', {
            "model": request.video_model,
            "mode": "i2v",
            "prompt": scenes[0],
            "init_image": img_urls[0],
            "num_frames": request.num_frames,
            "fps": request.fps,
            "response_format": "url",
        }, http_request)
        vid_url = _vid_url(vid_result if isinstance(vid_result, dict) else vid_result.__dict__)
    except Exception as e:
        vid_url = None
        steps.append({"step": "video", "error": str(e)})
    else:
        steps.append({"step": "video", "url": vid_url})

    # Step 4: TTS narration (optional)
    aud_url = None
    if request.tts_model:
        narration = " ".join(scenes)
        try:
            aud_result = await _post_json('/v1/audio/speech', {
                "model": request.tts_model,
                "input": narration,
                "voice": request.tts_voice or "af_sarah",
                "response_format": "mp3",
            }, http_request)
            aud_url = _aud_url(aud_result if isinstance(aud_result, dict) else aud_result.__dict__)
        except Exception as e:
            steps.append({"step": "tts", "error": str(e)})
        else:
            steps.append({"step": "tts", "url": aud_url})

    return {
        "created": int(time.time()),
        "pipeline": "story",
        "steps": steps,
        "data": [{
            "video_url": vid_url,
            "image_urls": img_urls,
            "audio_url": aud_url,
        }],
    }


# ---------------------------------------------------------------------------
# Pipeline 4: Audio Dub (transcribe → translate → clone voice → replace audio)
# ---------------------------------------------------------------------------

class AudioDubPipelineRequest(BaseModel):
    """
    Dub an audio or video file using a cloned voice.

    Steps:
    1. Transcribe source audio/video with Whisper
    2. Optionally translate the transcript
    3. Synthesize dubbed audio with F5-TTS voice cloning
    4. If input is video: replace the audio track (ffmpeg)
       If input is audio: return the dubbed audio directly
    """
    # Input — provide one of:
    video: Optional[str] = None         # base64/URL video
    audio: Optional[str] = None         # base64/URL audio-only file

    # Voice cloning — provide one of:
    voice_name: Optional[str] = None    # saved voice profile name
    ref_audio: Optional[str] = None     # base64 reference audio
    ref_text: Optional[str] = None      # transcript of ref_audio

    # Transcription
    source_lang: Optional[str] = None   # source language hint (auto-detect if None)
    whisper_model: Optional[str] = None # whisper model size (base, small, medium, large)

    # Translation
    target_lang: Optional[str] = None   # translate to this language before dubbing
                                        # if None, dub in original language

    # TTS
    speed: Optional[float] = 1.0
    seed: Optional[int] = None

    # Video output options
    burn_subtitles: Optional[bool] = False

    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


@router.post("/v1/pipelines/audio-dub", summary="Audio dubbing pipeline")
async def pipeline_audio_dub(request: AudioDubPipelineRequest, http_request: Request = None):
    """Transcribe → (translate) → clone voice → replace audio track."""
    import os, tempfile, subprocess, base64

    if not request.video and not request.audio:
        raise HTTPException(status_code=400, detail="Provide video or audio")
    if not request.voice_name and not request.ref_audio:
        raise HTTPException(status_code=400, detail="Provide voice_name or ref_audio for cloning")

    from codai.api.video import _decode_b64_or_url, _tmp_write, _whisper_transcribe, _translate_srt
    from codai.api.voice_clone import _load_voice, _decode_audio, _f5tts_clone

    temps = []
    steps = []

    try:
        # Decode input
        is_video = bool(request.video)
        raw = _decode_b64_or_url(request.video or request.audio)
        ext = '.mp4' if is_video else '.wav'
        in_path = _tmp_write(raw, ext)
        temps.append(in_path)

        # Step 1: Transcribe
        srt_path = await asyncio.get_event_loop().run_in_executor(
            None, _whisper_transcribe, in_path, request.source_lang,
            request.whisper_model, temps)
        if not srt_path:
            raise HTTPException(status_code=500, detail="Transcription failed — Whisper not available")

        with open(srt_path) as f:
            srt_content = f.read()
        steps.append({"step": "transcribe", "srt": srt_content})

        # Step 2: Translate (optional)
        if request.target_lang:
            srt_path = await asyncio.get_event_loop().run_in_executor(
                None, _translate_srt, srt_path, request.target_lang, temps)
            with open(srt_path) as f:
                srt_content = f.read()
            steps.append({"step": "translate", "lang": request.target_lang, "srt": srt_content})

        # Extract plain text from SRT
        plain_text = ' '.join(
            line.strip() for line in srt_content.split('\n')
            if line.strip() and not line.strip()[0].isdigit() and '-->' not in line
        )

        # Step 3: Resolve reference audio for voice cloning
        ref_audio_path = None
        ref_text = request.ref_text or ''
        if request.voice_name:
            meta = _load_voice(request.voice_name)
            if not meta:
                raise HTTPException(status_code=404, detail=f"Voice '{request.voice_name}' not found")
            ref_audio_path = meta['audio_file']
            ref_text = ref_text or meta.get('transcript', '')
        else:
            audio_bytes, aext = _decode_audio(request.ref_audio)
            tmp = tempfile.NamedTemporaryFile(suffix=aext, delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            ref_audio_path = tmp.name
            temps.append(ref_audio_path)

        if not ref_text:
            raise HTTPException(status_code=400, detail="ref_text required for voice cloning")

        # Step 4: Clone voice
        try:
            dubbed_bytes = await asyncio.get_event_loop().run_in_executor(
                None, _f5tts_clone,
                ref_audio_path, ref_text, plain_text,
                request.speed or 1.0, request.seed,
            )
        except ImportError:
            raise HTTPException(status_code=501, detail="f5-tts not installed. Run: pip install f5-tts")

        dubbed_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        dubbed_path.write(dubbed_bytes)
        dubbed_path.close()
        dubbed_path = dubbed_path.name
        temps.append(dubbed_path)
        steps.append({"step": "clone_voice"})

        # Step 5: Replace audio / return
        if is_video:
            out_path = tempfile.mktemp(suffix='_dubbed.mp4')
            temps.append(out_path)
            cmd = ['ffmpeg', '-y', '-i', in_path, '-i', dubbed_path,
                   '-map', '0:v', '-map', '1:a',
                   '-c:v', 'copy', '-c:a', 'aac', '-shortest', out_path]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Audio merge failed: {r.stderr.decode()}")

            if request.burn_subtitles:
                sub_out = tempfile.mktemp(suffix='_sub.mp4')
                temps.append(sub_out)
                r2 = subprocess.run(
                    ['ffmpeg', '-y', '-i', out_path, '-vf', f'subtitles={srt_path}',
                     '-c:a', 'copy', sub_out], capture_output=True)
                if r2.returncode == 0:
                    out_path = sub_out

            with open(out_path, 'rb') as f:
                out_bytes = f.read()
            out_b64 = base64.b64encode(out_bytes).decode()
            steps.append({"step": "merge_video"})
            result_data = [{"b64_mp4": out_b64}]
        else:
            out_b64 = base64.b64encode(dubbed_bytes).decode()
            result_data = [{"b64_wav": out_b64}]

        # Save to file path if configured
        if http_request:
            from codai.api.voice_clone import _save_audio_response
            # reuse save logic for the output
            pass

        return {
            "created": int(time.time()),
            "pipeline": "audio-dub",
            "steps": steps,
            "data": result_data,
        }

    finally:
        for t in temps:
            try:
                os.unlink(t)
            except Exception:
                pass
