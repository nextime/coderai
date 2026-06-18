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

"""Model capabilities module."""

from dataclasses import dataclass
from threading import Lock
from typing import List, Optional
import json
import os
import re
import time


@dataclass
class ModelCapabilities:
    """Represents what a model can do."""
    # Language / multimodal
    text_generation: bool = False       # LLM chat/completion
    image_to_text: bool = False         # VQA, captioning, vision LLMs
    embeddings: bool = False            # Text/image embeddings

    # Image generation & editing
    image_generation: bool = False      # Text-to-image (SD, Flux, …)
    image_to_image: bool = False        # img2img denoising
    inpainting: bool = False            # Inpaint with mask
    controlnet: bool = False            # ControlNet-guided generation

    # Image analysis & processing
    depth_estimation: bool = False      # Monocular depth (MiDaS, DPT, ZoeDepth)
    image_segmentation: bool = False    # SAM, Mask R-CNN
    image_upscaling: bool = False       # ESRGAN, SwinIR, Real-ESRGAN
    face_restoration: bool = False      # CodeFormer, GFPGAN
    object_detection: bool = False      # YOLO, DETR
    style_transfer: bool = False        # Neural style transfer

    # Video generation & editing
    video_generation: bool = False      # Text-to-video (CogVideoX, LTX, …)
    image_to_video: bool = False        # Image-to-video (SVD, I2VGen, …)
    video_to_video: bool = False        # Video style transfer / enhancement
    video_interpolation: bool = False   # Frame interpolation (FILM, RIFE)
    video_upscaling: bool = False       # Video super-resolution

    # Audio: speech
    speech_to_text: bool = False        # Whisper transcription
    text_to_speech: bool = False        # Kokoro, Bark, XTTS
    subtitle_generation: bool = False   # WhisperX / forced alignment subtitles

    # Audio: generation & manipulation
    audio_generation: bool = False      # MusicGen, AudioLDM2, StableAudio
    audio_to_audio: bool = False        # Denoising, source separation, …

    # Video + audio pipelines
    lip_sync: bool = False              # Wav2Lip, SadTalker
    video_dubbing: bool = False         # Translation + TTS + lip sync

    # 3D generation and conversion
    image_to_3d: bool = False           # 2D image → stereo / anaglyph / depth / mesh
    video_to_3d: bool = False           # 2D video → stereo / anaglyph / depth video
    model_3d_generation: bool = False   # text / image → 3D model (GLB)
    model_3d_to_image: bool = False     # 3D model → rendered 2D image / video

    def to_list(self) -> List[str]:
        out = []
        for name, val in self.__dataclass_fields__.items():
            if getattr(self, name):
                out.append(name)
        return out

    def __str__(self):
        return ", ".join(self.to_list()) or "none"


def detect_model_capabilities(model_name: str) -> ModelCapabilities:
    """
    Detect model capabilities from the model name/ID.
    Heuristic only — actual capabilities depend on the checkpoint.
    Returns all detected capabilities (multimodal models may have multiple).
    """
    caps = ModelCapabilities()
    if not model_name:
        return caps

    n = model_name.lower()

    # ── 3D generation ────────────────────────────────────────────────────────
    if any(x in n for x in ['triposr', 'tsr', 'shap-e', 'shape-e', 'point-e', 'pointe',
                              'zero123', 'wonder3d', 'instant3d', 'one-2-3-45',
                              'mvdiffusion', 'syncdreamer', 'dreamgaussian']):
        caps.model_3d_generation = True
        caps.image_to_3d = True
        caps.model_3d_to_image = True
        return caps

    # ── Video generation ─────────────────────────────────────────────────────
    if any(x in n for x in ['cogvideox', 'cogvideo', 'ltx-video', 'ltxvideo',
                              'hunyuan-video', 'mochi-1', 'dynamicrafter',
                              'animatediff', 'text2video', 'modelscope-t2v',
                              'zeroscope', 'lavie']):
        caps.video_generation = True
        return caps

    if any(x in n for x in ['wan2.1-t2v', 'wan-t2v']):
        caps.video_generation = True
        return caps

    # Image-to-video
    if any(x in n for x in ['stable-video-diffusion', 'svd',
                              'i2vgen-xl', 'i2vgen', 'cogvideox-i2v',
                              'wan2.1-i2v', 'wan-i2v', 'img2vid',
                              'image2video', 'motionctrl']):
        caps.image_to_video = True
        return caps

    # Wan generic (detect sub-variant)
    if 'wan' in n and ('video' in n or 'diffuser' in n):
        if 'i2v' in n:
            caps.image_to_video = True
        else:
            caps.video_generation = True
        return caps

    # Video interpolation
    if any(x in n for x in ['film-net', 'film_net', 'film', 'rife', 'ifnet',
                            'flavr', 'dain', 'frame-interp', 'interpolat']):
        caps.video_interpolation = True
        return caps

    # Video upscaling / super-resolution
    if any(x in n for x in ['real-basicvsr', 'basicvsr', 'edvr',
                              'video-enhance', 'videoswin-sr']):
        caps.video_upscaling = True
        return caps

    # Video-to-video
    if any(x in n for x in ['tokenflow', 'text2video-zero', 'vid2vid',
                              'rerender-a-video', 'controlvideo']):
        caps.video_to_video = True
        return caps

    # ── Audio ────────────────────────────────────────────────────────────────
    if any(x in n for x in ['musicgen', 'audiogen', 'audioldm', 'stable-audio',
                              'mustango', 'noise2music', 'jukebox', 'audiocraft']):
        caps.audio_generation = True
        return caps

    if any(x in n for x in ['demucs', 'spleeter', 'asteroid', 'open-unmix']):
        caps.audio_to_audio = True
        return caps

    if any(x in n for x in ['whisper', 'faster-whisper', 'distil-whisper',
                              'wav2vec', 'hubert', 'seamless']):
        caps.speech_to_text = True
        caps.subtitle_generation = True
        return caps

    if any(x in n for x in ['kokoro', 'xtts', 'bark', 'tortoise',
                              'speecht5', 'matcha-tts', 'voicebox']):
        caps.text_to_speech = True
        return caps

    # Lip sync / dubbing
    if any(x in n for x in ['wav2lip', 'sadtalker', 'dinet', 'videoretalking']):
        caps.lip_sync = True
        caps.audio_generation = True
        caps.video_generation = True
        return caps

    # ── Image: upscaling (checked before general SD rule to catch SD-family upscalers) ──
    # 'hat-'/'dat-' are short, ambiguous tokens (e.g. they appear inside
    # "chat-", "update-"); require a word boundary before them so a text "chat"
    # model isn't mistaken for the HAT/DAT super-resolution checkpoints.
    if (any(x in n for x in ['real-esrgan', 'esrgan', 'swinir', 'edsr',
                              'bsrgan',
                              'x2-upscaler', 'x4-upscaler', 'x2_upscaler', 'x4_upscaler',
                              'latent-upscaler', 'latent_upscaler',
                              'ldm-super-resolution', 'rcan-', 'sr3-'])
            or re.search(r'\b[hd]at-', n)):
        caps.image_upscaling = True
        caps.image_to_image = True
        return caps

    # ── Image: generation ────────────────────────────────────────────────────
    if any(x in n for x in ['inpaint', 'instruct-pix2pix', 'paint-by-example']):
        caps.inpainting = True
        caps.image_generation = True
        caps.image_to_image = True
        return caps

    if 'controlnet' in n:
        caps.controlnet = True
        caps.image_generation = True
        return caps

    if any(x in n for x in [
        # Stable Diffusion family
        'stable-diffusion', 'sd15', 'sdxl', 'sd-xl', 'sd-turbo', 'sdxl-turbo',
        'sd3', 'stable-cascade', 'stable-zero123',
        # Other major T2I families
        'flux', 'flux-dev', 'flux-schnell',
        'kandinsky', 'deepfloyd', 'pixart', 'hunyuan-dit',
        'dalle', 'dall-e',
        'playground', 'playgroundai',
        'imagen', 'parti-',
        # Community / fine-tuned SD models (common naming conventions)
        'dreamshaper', 'dreamlike', 'juggernaut', 'revanimated',
        'epicrealism', 'absolutereality', 'counterfeit', 'deliberate',
        'anything-v', 'openjourney', 'realvis', 'photon-',
        'waifu', 'pony', 'realistic-vision', 'realistic_vision',
        'aingdiffusion', 'majicmix', 'chillout', 'ghostmix',
        # Speed / distilled variants (turbo, lightning, hyper)
        'image-turbo', 'image-lightning', 'image-hyper', 'image-flash',
        'diffusion-turbo', 'diffusion-lightning',
        # Explicit T2I naming
        'text-to-image', 'txt2img', 'text2img', 't2i-adapter',
        # Generic diffusion bases
        'latent-diffusion', 'ldm-',
    ]):
        caps.image_generation = True
        caps.image_to_image = True
        caps.inpainting = True    # most SD/SDXL/Flux checkpoints support inpainting via mask
        return caps

    # ── Image: analysis / processing ─────────────────────────────────────────
    if any(x in n for x in ['midas', 'dpt-depth', 'dpt-large', 'zoe-depth',
                              'depth-anything', 'marigold']):
        caps.depth_estimation = True
        caps.image_to_text = True  # Image analysis models process images
        return caps

    if any(x in n for x in ['sam2', 'sam-', '-sam', 'segment-anything',
                              'mask-rcnn', 'fastsam']):
        caps.image_segmentation = True
        caps.image_to_text = True
        return caps

    if any(x in n for x in ['codeformer', 'gfpgan', 'restoreformer']):
        caps.face_restoration = True
        caps.image_upscaling = True
        caps.image_to_image = True
        return caps

    if any(x in n for x in ['yolo', 'detr', 'owlvit', 'rtdetr', 'dino']):
        caps.object_detection = True
        caps.image_to_text = True
        return caps

    # ── Vision / multimodal LLMs ─────────────────────────────────────────────
    if any(x in n for x in [
        'vision', 'vl-', '-vl', 'llava', 'llava-next', 'llavanext',
        'qwen2-vl', 'qwen-vl', 'qwen2.5-vl',
        'phi-4-mini', 'phi-3-vision', 'phi3-vision',
        'pixtral', 'mistral-pixtral',
        'clip', 'clip-vit',
        'blip', 'blip-2', 'blip2',
        'internvl', 'internlm-xcomposer',
        'moondream', 'idefics', 'idefics2', 'idefics3',
        'cogvlm', 'cogvlm2',
        'minigpt', 'minigpt4',
        'flamingo', 'openflamingo',
        'paligemma', 'gemma-2-vl',
        'deepseek-vl', 'deepseek-vl2',
        'minicpm-v', 'minicpmv',
        'mllm', 'multimodal',
    ]):
        caps.image_to_text = True
        caps.text_generation = True
        return caps

    # ── Embeddings ───────────────────────────────────────────────────────────
    if any(x in n for x in [
        'embed', 'bge-', 'e5-', 'minilm',
        'sentence-transformer', 'sentence-bert',
        'nomic-embed', 'nomic-text',
        'instructor-', 'gte-', 'jina-embed', 'jina-clip',
        'all-mpnet', 'all-minilm', 'paraphrase-',
        'multilingual-e5', 'multilingual-mpnet',
        'text-embedding', 'voyage-',
    ]):
        caps.embeddings = True
        return caps

    # Default: text generation
    caps.text_generation = True
    return caps


# ── HuggingFace pipeline_tag → capability fields ─────────────────────────────
_PIPELINE_TAG_CAPS: dict = {
    'text-generation':                ['text_generation'],
    'text2text-generation':           ['text_generation'],
    'image-to-text':                  ['image_to_text', 'text_generation'],
    'visual-question-answering':      ['image_to_text', 'text_generation'],
    'image-text-to-text':             ['image_to_text', 'text_generation'],
    'text-to-image':                  ['image_generation', 'image_to_image'],
    'unconditional-image-generation': ['image_generation'],
    'image-to-image':                 ['image_to_image'],   # sub-typed below
    'automatic-speech-recognition':   ['speech_to_text'],
    'audio-to-audio':                 ['audio_to_audio'],
    'text-to-speech':                 ['text_to_speech'],
    'text-to-audio':                  ['audio_generation'],
    'text-to-video':                  ['video_generation'],
    'image-to-video':                 ['image_to_video'],
    'feature-extraction':             ['embeddings'],
    'sentence-similarity':            ['embeddings'],
    'depth-estimation':               ['depth_estimation', 'image_to_text'],
    'image-segmentation':             ['image_segmentation', 'image_to_text'],
    'object-detection':               ['object_detection', 'image_to_text'],
    'image-classification':           ['image_to_text'],
    'video-classification':           ['image_to_text'],
    'zero-shot-image-classification': ['image_to_text'],
    'mask-generation':                ['image_segmentation', 'image_to_text'],
}


def detect_capabilities_from_pipeline_tag(
    pipeline_tag: str, model_name: str = ""
) -> ModelCapabilities:
    """
    Detect capabilities using the HuggingFace pipeline_tag as the primary signal,
    supplemented by name heuristics for sub-types (e.g. upscaling vs I2I).
    Falls back to detect_model_capabilities when tag is absent or unrecognised.
    """
    tag = (pipeline_tag or "").lower().strip()
    fields = _PIPELINE_TAG_CAPS.get(tag)

    if not fields:
        return detect_model_capabilities(model_name)

    caps = ModelCapabilities()
    for f in fields:
        if hasattr(caps, f):
            setattr(caps, f, True)

    # For generic image-to-image, use name heuristics to identify sub-type
    if tag == 'image-to-image' and model_name:
        name_caps = detect_model_capabilities(model_name)
        if name_caps.image_upscaling:
            caps.image_upscaling = True
        if name_caps.inpainting:
            caps.inpainting = True
            caps.image_generation = True
        if name_caps.face_restoration:
            caps.face_restoration = True
            caps.image_upscaling = True
        if name_caps.style_transfer:
            caps.style_transfer = True

    return caps


# ── Persistent capability cache ───────────────────────────────────────────────
# Populated from HF search results (pipeline_tag-based, authoritative).
# Used as first-pass during local model scans so pipeline_tag info survives offline.

_CAP_CACHE_TTL = 90 * 86400  # 90 days
_cap_cache: dict = {}
_cap_cache_path: Optional[str] = None
_cap_cache_lock = Lock()


def init_capability_cache(config_dir: str) -> None:
    """Load the on-disk capability cache. Call once at server startup."""
    global _cap_cache, _cap_cache_path
    _cap_cache_path = os.path.join(config_dir, "capability_cache.json")
    try:
        with open(_cap_cache_path) as f:
            loaded = json.load(f)
        now = time.time()
        # Prune stale entries on load
        _cap_cache = {k: v for k, v in loaded.items()
                      if now - v.get("ts", 0) < _CAP_CACHE_TTL}
    except Exception:
        _cap_cache = {}


def _flush_capability_cache() -> None:
    if _cap_cache_path is None:
        return
    try:
        with open(_cap_cache_path, "w") as f:
            json.dump(_cap_cache, f, indent=2)
    except Exception:
        pass


def update_capability_cache(model_id: str, caps: ModelCapabilities) -> None:
    """Store authoritative (pipeline_tag-derived) capabilities and persist to disk."""
    if not model_id:
        return
    with _cap_cache_lock:
        _cap_cache[model_id] = {"caps": caps.to_list(), "ts": int(time.time())}
        _flush_capability_cache()


def lookup_capability_cache(model_id: str) -> Optional[ModelCapabilities]:
    """Return cached ModelCapabilities, or None if absent or expired."""
    if not model_id:
        return None
    entry = _cap_cache.get(model_id)
    if not entry:
        return None
    if time.time() - entry.get("ts", 0) > _CAP_CACHE_TTL:
        with _cap_cache_lock:
            _cap_cache.pop(model_id, None)
        return None
    caps = ModelCapabilities()
    for field in entry.get("caps", []):
        if hasattr(caps, field):
            setattr(caps, field, True)
    return caps