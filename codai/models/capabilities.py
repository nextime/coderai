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
from typing import List


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

    # ── Video generation ─────────────────────────────────────────────────────
    if any(x in n for x in ['cogvideox', 'cogvideo', 'ltx-video', 'ltxvideo',
                              'hunyuan-video', 'mochi-1', 'dynamicrafter',
                              'animatediff', 'text2video', 'modelscope-t2v',
                              'zeroscope', 'lavie']):
        caps.video_generation = True
        caps.text_generation = True  # T2V models also do text
        return caps

    if any(x in n for x in ['wan2.1-t2v', 'wan-t2v']):
        caps.video_generation = True
        caps.text_generation = True
        return caps

    # Image-to-video
    if any(x in n for x in ['stable-video-diffusion', 'svd',
                              'i2vgen-xl', 'i2vgen', 'cogvideox-i2v',
                              'wan2.1-i2v', 'wan-i2v', 'img2vid',
                              'image2video', 'motionctrl']):
        caps.image_to_video = True
        caps.image_to_text = True  # I2V models process images
        return caps

    # Wan generic (detect sub-variant)
    if 'wan' in n and ('video' in n or 'diffuser' in n):
        if 'i2v' in n:
            caps.image_to_video = True
            caps.image_to_text = True
        else:
            caps.video_generation = True
            caps.text_generation = True
        return caps

    # Video interpolation
    if any(x in n for x in ['film-net', 'rife', 'flavr', 'dain', 'frame-interp']):
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
        caps.text_generation = True  # T2A models process text
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
        caps.text_generation = True  # TTS models process text
        return caps

    # Lip sync / dubbing
    if any(x in n for x in ['wav2lip', 'sadtalker', 'dinet', 'videoretalking']):
        caps.lip_sync = True
        caps.audio_generation = True
        caps.video_generation = True
        return caps

    # ── Image: generation ────────────────────────────────────────────────────
    if any(x in n for x in ['inpaint', 'instruct-pix2pix', 'paint-by-example']):
        caps.inpainting = True
        caps.image_generation = True
        caps.image_to_image = True
        caps.text_generation = True  # T2I models process text
        return caps

    if 'controlnet' in n:
        caps.controlnet = True
        caps.image_generation = True
        caps.text_generation = True
        return caps

    if any(x in n for x in ['stable-diffusion', 'sd15', 'sdxl', 'sd-xl',
                              'playground', 'flux', 'kandinsky', 'deepfloyd',
                              'pixart', 'dalle', 'waifu', 'pony',
                              'realistic-vision', 'realistic_vision']):
        caps.image_generation = True
        caps.image_to_image = True
        caps.inpainting = True    # most SD/SDXL/Flux support inpainting variant
        caps.text_generation = True  # T2I models process text
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

    if any(x in n for x in ['real-esrgan', 'esrgan', 'swinir', 'edsr',
                              'bsrgan', 'hat-', 'dat-']):
        caps.image_upscaling = True
        caps.image_to_image = True
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
    if any(x in n for x in ['vision', 'vl-', '-vl', 'llava', 'qwen2-vl',
                              'qwen-vl', 'phi-4-mini', 'pixtral', 'clip',
                              'blip', 'internvl', 'moondream', 'idefics',
                              'cogvlm', 'minigpt', 'flamingo']):
        caps.image_to_text = True
        caps.text_generation = True
        return caps

    # ── Embeddings ───────────────────────────────────────────────────────────
    if any(x in n for x in ['embed', 'bge-', 'e5-', 'minilm',
                              'sentence-transformer', 'nomic-embed',
                              'instructor-', 'gte-', 'jina-embed']):
        caps.embeddings = True
        caps.text_generation = True  # Embedding models process text
        return caps

    # ── GGUF quantised text models ───────────────────────────────────────────
    if '.gguf' in n or 'gguf' in n:
        caps.text_generation = True
        return caps

    # Default: text generation
    caps.text_generation = True
    return caps