"""Pydantic models for video generation API."""

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class VideoGenerationRequest(BaseModel):
    model: str
    prompt: str = ""
    negative_prompt: Optional[str] = None

    # Dimensions
    width: Optional[int] = 512
    height: Optional[int] = 512

    # Temporal
    num_frames: Optional[int] = None        # model default if None
    fps: Optional[int] = None               # output FPS

    # Diffusion
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None

    # Mode
    # t2v  – text-to-video
    # i2v  – image-to-video (init_image required)
    # v2v  – video-to-video (video required)
    # ti2v – text + init image → video (like i2v but prompt is primary driver)
    # interp – frame interpolation (init_image + end_image)
    mode: Optional[str] = "t2v"

    # Input media (base64 or URL)
    image: Optional[str] = None             # alias for init_image
    init_image: Optional[str] = None        # first/reference frame
    end_image: Optional[str] = None         # last frame (for interp mode)
    video: Optional[str] = None             # input video (v2v / audio manipulation)
    strength: Optional[float] = None        # denoising strength for v2v

    # Camera motion hint
    camera_motion: Optional[str] = None     # zoom-in | zoom-out | pan-left | pan-right | tilt-up | tilt-down | rotate

    # ── Character consistency ─────────────────────────────────────────────
    character_references: Optional[List[str]] = None  # list of base64/URL reference images
    character_strength: Optional[float] = 0.8
    character_names: Optional[List[str]] = None       # optional names per reference

    # ── Audio generation / manipulation ──────────────────────────────────
    add_audio: Optional[bool] = False
    audio_type: Optional[str] = None        # music | speech | sfx | ambient
    audio_prompt: Optional[str] = None      # prompt for music/sfx generation
    audio_file: Optional[str] = None        # existing audio to add (base64/URL)
    tts_text: Optional[str] = None          # text for speech synthesis
    tts_voice: Optional[str] = None         # TTS voice id
    tts_speed: Optional[float] = 1.0
    sync_audio: Optional[bool] = False      # sync audio timing to video
    lip_sync: Optional[bool] = False        # warp mouth to match audio
    lip_sync_method: Optional[str] = "wav2lip"  # wav2lip | sadtalker

    # ── Subtitles ────────────────────────────────────────────────────────
    generate_subtitles: Optional[bool] = False
    burn_subtitles: Optional[bool] = False
    subtitle_language: Optional[str] = None   # source language hint
    translate_subtitles: Optional[bool] = False
    subtitle_target_lang: Optional[str] = None
    subtitle_style: Optional[str] = "default"  # default | karaoke | minimal
    whisper_model: Optional[str] = None       # which whisper variant to use

    # ── Video dubbing ─────────────────────────────────────────────────────
    dub_video: Optional[bool] = False
    dub_target_lang: Optional[str] = None
    dub_source_lang: Optional[str] = None
    voice_clone: Optional[bool] = False       # clone original speaker voice

    # ── Post-processing ───────────────────────────────────────────────────
    upscale_output: Optional[bool] = False
    upscale_factor: Optional[int] = 2
    interpolate_output: Optional[bool] = False  # increase FPS after generation
    fps_multiplier: Optional[int] = 2           # e.g. 2 → 2× FPS via frame interp
    convert_to_3d: Optional[bool] = False
    depth_method: Optional[str] = "midas"       # midas | zoe | depth-anything

    # ── Memory / offload ─────────────────────────────────────────────────
    offload_strategy: Optional[str] = None   # sequential | model | none

    # Nulls pipeline safety_checker / safety_concept so uncensored fine-tunes
    # are not blocked. Has no effect on models without a safety checker.
    disable_safety_checker: Optional[bool] = False

    # ── Output ───────────────────────────────────────────────────────────
    response_format: Optional[str] = "url"   # url | b64_mp4
    n: int = 1
    user: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class VideoGenerationResponse(BaseModel):
    created: int
    data: List[Dict]
    model_config = ConfigDict(extra="allow")


# ── Standalone operation requests ─────────────────────────────────────────────

class VideoUpscaleRequest(BaseModel):
    model: str
    video: str                          # base64/URL input video
    upscale_factor: Optional[int] = 2
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


class VideoSubtitleRequest(BaseModel):
    model: str
    video: str                          # base64/URL input video
    language: Optional[str] = None
    translate: Optional[bool] = False
    target_lang: Optional[str] = None
    burn: Optional[bool] = False
    style: Optional[str] = "default"
    response_format: Optional[str] = "srt"  # srt | vtt | json | burned_video
    model_config = ConfigDict(extra="allow")


class VideoInterpolateRequest(BaseModel):
    model: str
    video: Optional[str] = None          # base64/URL input video  (mutually exclusive with init/end)
    init_image: Optional[str] = None     # first frame
    end_image: Optional[str] = None      # last frame
    fps_multiplier: Optional[int] = 2
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


class VideoDubRequest(BaseModel):
    model: str
    video: str
    target_lang: str
    source_lang: Optional[str] = None
    voice_clone: Optional[bool] = False
    burn_subtitles: Optional[bool] = False
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")
