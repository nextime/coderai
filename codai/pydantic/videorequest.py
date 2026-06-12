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

"""Pydantic models for video generation API."""

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class VideoLoraConfig(BaseModel):
    """A LoRA adapter to apply to the video diffusion pipeline for one request.

    The weights may be supplied in several ways (resolved server-side, in this
    priority): `id` ("name:<registered>" trained LoRA, or "sha256:<hex>" uploaded
    blob), inline `file`/`data` base64, a `url` to download, or the legacy
    `model`/`path` local path / HF id. This lets a client on a different machine
    use a LoRA without sharing a filesystem with the server."""
    model: Optional[str] = Field(None, description="Legacy: local path or HF id of the weights (shared-filesystem only).")
    path: Optional[str] = Field(None, description="Alias of `model` — local path to the .safetensors weights.")
    id: Optional[str] = Field(None, description='Registry/blob reference: "name:<registered-lora>" or "sha256:<hex>" (from /v1/loras/upload).')
    url: Optional[str] = Field(None, description="HTTP(S) URL the server downloads and caches.")
    file: Optional[str] = Field(None, description="Base64 of the .safetensors file (or a data: URI). Sent inline so no shared filesystem is needed.")
    data: Optional[str] = Field(None, description="Alias of `file` — inline base64 weights.")
    weight: float = Field(1.0, description="Adapter strength / scale applied when fusing the LoRA.")
    name: Optional[str] = Field(None, description="Optional adapter name (defaults derived from the reference).")
    model_config = ConfigDict(extra="allow")


class CharacterDialogLine(BaseModel):
    """One spoken line in a multi-character dialog sequence."""
    character: Optional[str] = None    # character profile name (used for lip-sync face)
    voice: Optional[str] = None        # voice profile name OR TTS voice id
    text: str                          # dialog text to synthesise
    start_time: Optional[float] = None # explicit offset in seconds (None = sequential)
    lip_sync: Optional[bool] = True    # apply lip sync for this line
    lang: Optional[str] = None         # language hint for TTS
    speed: Optional[float] = 1.0
    model_config = ConfigDict(extra="allow")


class VideoGenerationRequest(BaseModel):
    model: str = Field(..., description="Model id to generate with (must be a configured video model).")
    prompt: str = Field("", description="Text prompt describing the video. Optional for pure i2v.")
    negative_prompt: Optional[str] = Field(None, description="What to avoid in the output.")

    # Dimensions
    width: Optional[int] = Field(512, description="Output width in pixels.")
    height: Optional[int] = Field(512, description="Output height in pixels.")

    # Temporal
    num_frames: Optional[int] = Field(None, description="Number of frames to generate (model default if omitted).")
    fps: Optional[int] = Field(None, description="Output frame rate (model default if omitted).")

    # Diffusion
    num_inference_steps: Optional[int] = Field(None, description="Denoising steps (model/acceleration default if omitted).")
    guidance_scale: Optional[float] = Field(None, description="Classifier-free guidance scale (model/acceleration default if omitted).")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility.")

    mode: Optional[str] = Field("t2v", description=(
        "Generation mode: 't2v' text-to-video; 'i2v' image-to-video (init_image required, "
        "prompt dropped); 'ti2v' text+init image (prompt is primary driver); 'v2v' "
        "video-to-video (video required); 'interp' frame interpolation (init_image+end_image). "
        "The server gracefully falls back between Wan t2v/i2v pipelines when a model only "
        "supports one."))

    # Input media (base64 or URL)
    image: Optional[str] = Field(None, description="Alias for init_image (base64 or URL).")
    init_image: Optional[str] = Field(None, description="First/reference frame for i2v/ti2v (base64 or URL).")
    end_image: Optional[str] = Field(None, description="Last frame, for 'interp' mode (base64 or URL).")
    video: Optional[str] = Field(None, description="Input video for v2v / audio manipulation (base64 or URL).")
    strength: Optional[float] = Field(None, description="Denoising strength for v2v (0–1).")

    # Camera motion hint
    camera_motion: Optional[str] = None     # zoom-in | zoom-out | pan-left | pan-right | tilt-up | tilt-down | rotate

    # ── Character consistency ─────────────────────────────────────────────
    characters: Optional[List[dict]] = Field(None, description='Per-character references, each {"name": "Alice", "images": ["b64...", ...]}.')
    character_references: Optional[List[str]] = Field(None, description="Legacy flat list of base64/URL reference images.")
    character_strength: Optional[float] = Field(0.8, description="IP-Adapter scale for character references.")
    character_names: Optional[List[str]] = Field(None, description="Optional names aligned with character_references.")
    character_profiles: Optional[List[str]] = Field(None, description="Saved character profile names to load (resolved server-side).")

    loras: Optional[List[VideoLoraConfig]] = Field(None, description="Per-request LoRA adapters (e.g. trained per-character identity LoRAs) fused into the pipeline.")

    # ── Audio generation / manipulation ──────────────────────────────────
    add_audio: Optional[bool] = Field(False, description="Add an audio track to the generated video.")
    audio_type: Optional[str] = Field(None, description="Audio to generate/add: 'music', 'speech', 'sfx' or 'ambient'.")
    audio_prompt: Optional[str] = Field(None, description="Prompt for generated music/sfx.")
    audio_file: Optional[str] = Field(None, description="Existing audio to add (base64/URL).")
    tts_text: Optional[str] = Field(None, description="Text to synthesize as speech.")
    tts_voice: Optional[str] = Field(None, description="TTS voice id for synthesized speech.")
    tts_speed: Optional[float] = Field(1.0, description="TTS speaking speed multiplier.")
    sync_audio: Optional[bool] = Field(False, description="Sync audio timing to the video.")
    lip_sync: Optional[bool] = Field(False, description="Warp the mouth to match the audio.")
    lip_sync_method: Optional[str] = Field("wav2lip", description="Lip-sync engine: 'wav2lip' or 'sadtalker'.")

    # ── Multi-character dialog ────────────────────────────────────────────
    dialogs: Optional[List[CharacterDialogLine]] = Field(None, description="Ordered spoken lines for multi-character dialog with per-line voice/lip-sync.")

    # ── Subtitles ────────────────────────────────────────────────────────
    generate_subtitles: Optional[bool] = Field(False, description="Generate subtitles (SRT) for the video.")
    burn_subtitles: Optional[bool] = Field(False, description="Burn the subtitles into the video frames.")
    subtitle_language: Optional[str] = Field(None, description="Source language hint for subtitle transcription.")
    translate_subtitles: Optional[bool] = Field(False, description="Translate subtitles to subtitle_target_lang.")
    subtitle_target_lang: Optional[str] = Field(None, description="Target language for translated subtitles.")
    subtitle_style: Optional[str] = Field("default", description="Subtitle style: 'default', 'karaoke' or 'minimal'.")
    whisper_model: Optional[str] = Field(None, description="Whisper variant used for transcription.")

    # ── Video dubbing ─────────────────────────────────────────────────────
    dub_video: Optional[bool] = Field(False, description="Dub the video's speech into another language.")
    dub_target_lang: Optional[str] = Field(None, description="Target language for dubbing.")
    dub_source_lang: Optional[str] = Field(None, description="Source language of the original speech.")
    voice_clone: Optional[bool] = Field(False, description="Clone the original speaker's voice when dubbing.")

    # ── Post-processing ───────────────────────────────────────────────────
    upscale_output: Optional[bool] = Field(False, description="Upscale the generated video.")
    upscale_factor: Optional[int] = Field(2, description="Upscaling factor.")
    interpolate_output: Optional[bool] = Field(False, description="Increase FPS via frame interpolation after generation.")
    fps_multiplier: Optional[int] = Field(2, description="FPS multiplier for interpolation (e.g. 2 → 2× FPS).")
    convert_to_3d: Optional[bool] = Field(False, description="Produce a depth-based 3D/stereo version.")
    depth_method: Optional[str] = Field("midas", description="Depth estimator: 'midas', 'zoe' or 'depth-anything'.")

    # ── Memory / offload ─────────────────────────────────────────────────
    offload_strategy: Optional[str] = Field(None, description="VRAM offload strategy: 'sequential', 'model' or 'none'.")

    disable_safety_checker: Optional[bool] = Field(False, description=(
        "Null out the diffusers safety_checker/safety_concept so uncensored fine-tunes "
        "are not blocked. No effect on models that ship no safety checker (SDXL/Flux/Wan)."))

    # ── Output ───────────────────────────────────────────────────────────
    response_format: Optional[str] = Field("url", description="How to return the result: 'url' or 'b64_mp4'.")
    n: int = Field(1, description="Number of videos to generate.")
    user: Optional[str] = Field(None, description="Opaque end-user identifier (passthrough).")

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