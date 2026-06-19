"""Pluggable text-to-speech backends for the /v1/audio/speech endpoint.

Dispatches a TTS request to the right engine based on the model family:

* **kokoro**  → ``kokoro-onnx`` (ONNX runtime, no torch/spaCy needed). Requires a
  ``kokoro-*.onnx`` model file and a ``voices-*.bin`` file; both are auto-resolved
  from a local dir / HF repo, or downloaded from the kokoro-onnx release.
* **coqui / XTTS** → ``coqui-TTS`` (``pip install coqui-tts``) when installed.
* **parler** → ``parler-tts`` (expressive; voice/emotion/delivery/speed are steered
  through a natural-language description prompt) when installed.
* **anything else** → transformers ``pipeline("text-to-speech")`` (SpeechT5, Bark,
  VITS / MMS-TTS, …).

Every backend returns ``(samples: np.float32 [-1, 1], sample_rate: int)`` which is
then encoded to the requested container by :func:`encode_audio`.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Official kokoro-onnx model + voices release (used when files aren't local).
_KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)
_KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)


class MissingEngineError(RuntimeError):
    """Raised when the optional engine for a TTS family isn't installed."""


def _family(model_name: str) -> str:
    """Classify a TTS model name into a backend family."""
    n = (model_name or "").lower()
    if "kokoro" in n:
        return "kokoro"
    if "xtts" in n or "coqui" in n:
        return "coqui"
    if "parler" in n:
        return "parler"
    if "bark" in n:
        return "bark"
    return "transformers"


# Discrete emotion presets a family can steer at synthesis time. Empty unless an
# engine actually supports it — clients surface an emotion picker only when this
# is non-empty, so the control stays hidden for engines that can't honour it.
_FAMILY_EMOTIONS: dict[str, list[str]] = {
    # Parler steers these through its natural-language description prompt.
    "parler": ["neutral", "happy", "sad", "angry", "excited", "calm", "fearful"],
    # Bark has no true emotion knob; it inserts matching non-verbal cues in text.
    "bark": ["neutral", "laughter", "sigh", "gasp"],
}

# Delivery / vocal styles a family can steer (whisper, shout/scream, tone, …).
# Empty unless an engine actually honours it — kept separate from emotions so a
# client can offer "how it's said" independently of "what's felt".
_FAMILY_STYLES: dict[str, list[str]] = {
    "parler": ["normal", "whispering", "shouting", "monotone", "expressive"],
    "bark": ["normal", "whispering", "singing", "emphasis"],
}


def family_emotions(model_name: str) -> list[str]:
    """Emotions the given model can steer, or [] when none are available."""
    return list(_FAMILY_EMOTIONS.get(_family(model_name), []))


def family_styles(model_name: str) -> list[str]:
    """Delivery styles (whisper/shout/tone/…) the model can steer, or []."""
    return list(_FAMILY_STYLES.get(_family(model_name), []))


# --------------------------------------------------------------------------- #
# kokoro-onnx
# --------------------------------------------------------------------------- #

def _cache_dir() -> Path:
    base = os.environ.get("CODERAI_TTS_CACHE") or os.path.expanduser("~/.coderai/tts_cache")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    import urllib.request
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  Downloading TTS asset: {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def _resolve_kokoro_files(model_path: str, config: dict) -> Tuple[str, str]:
    """Return (onnx_model_path, voices_path) for kokoro-onnx.

    Order of resolution: explicit config fields → files alongside a local dir /
    .onnx path → download the official release into the TTS cache.
    """
    voices_path = (config or {}).get("voices_path")
    onnx_path = (config or {}).get("model_path") or model_path

    cand = Path(onnx_path) if onnx_path else None
    if cand and cand.is_dir():
        onnx = next(iter(sorted(cand.glob("*.onnx"))), None)
        vb = next(iter(sorted(cand.glob("voices*.bin"))), None)
        if onnx:
            onnx_path = str(onnx)
        if vb and not voices_path:
            voices_path = str(vb)
    elif cand and cand.suffix == ".onnx" and cand.exists():
        onnx_path = str(cand)
        if not voices_path:
            sib = next(iter(sorted(cand.parent.glob("voices*.bin"))), None)
            if sib:
                voices_path = str(sib)

    # Fall back to the official release files in the cache.
    if not (onnx_path and Path(onnx_path).exists()):
        onnx_path = str(_download(_KOKORO_MODEL_URL, _cache_dir() / "kokoro-v1.0.onnx"))
    if not (voices_path and Path(voices_path).exists()):
        voices_path = str(_download(_KOKORO_VOICES_URL, _cache_dir() / "voices-v1.0.bin"))
    return onnx_path, voices_path


class _KokoroBackend:
    family = "kokoro"
    default_voice = "af_sarah"

    def __init__(self, model_path: str, config: dict):
        from kokoro_onnx import Kokoro
        onnx_path, voices_path = _resolve_kokoro_files(model_path, config)
        print(f"  kokoro-onnx model={onnx_path} voices={voices_path}")
        self._kokoro = Kokoro(onnx_path, voices_path)

    def synthesize(self, text: str, voice: str, speed: float, lang: str,
                   emotion: str = "", style: str = "") -> Tuple[np.ndarray, int]:
        samples, sr = self._kokoro.create(
            text, voice=voice or self.default_voice, speed=speed or 1.0,
            lang=lang or "en-us",
        )
        return np.asarray(samples, dtype=np.float32), int(sr)

    def voices(self):
        try:
            return sorted(self._kokoro.get_voices())
        except Exception:
            return []


# --------------------------------------------------------------------------- #
# transformers pipeline("text-to-speech")
# --------------------------------------------------------------------------- #

class _TransformersBackend:
    family = "transformers"
    default_voice = ""

    def __init__(self, model_name: str, config: dict):
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        print(f"  transformers TTS pipeline model={model_name} device={device}")
        self._pipe = pipeline("text-to-speech", model=model_name, device=device)

    def synthesize(self, text: str, voice: str, speed: float, lang: str,
                   emotion: str = "", style: str = "") -> Tuple[np.ndarray, int]:
        out = self._pipe(text)
        audio = np.asarray(out["audio"], dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
        return audio, int(out["sampling_rate"])

    def voices(self):
        return []


# --------------------------------------------------------------------------- #
# coqui / XTTS (optional)
# --------------------------------------------------------------------------- #

class _CoquiBackend:
    family = "coqui"
    default_voice = ""

    def __init__(self, model_name: str, config: dict):
        try:
            from TTS.api import TTS  # coqui-tts
        except ImportError as e:
            raise MissingEngineError(
                "Coqui/XTTS models need the coqui-tts package: "
                "pip install coqui-tts"
            ) from e
        import torch
        self._cfg = config or {}
        self._tts = TTS(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    def synthesize(self, text: str, voice: str, speed: float, lang: str,
                   emotion: str = "", style: str = "") -> Tuple[np.ndarray, int]:
        # XTTS can clone from a reference wav, use one of its built-in speakers,
        # or fall back to a default. `voice` may be a wav path, a built-in
        # speaker name, or (e.g. a Kokoro id like "af_sarah") neither — so only
        # treat it as a clone source when it's an actual file.
        kwargs = {"text": text, "language": (lang or "en")[:2]}
        speakers = list(getattr(self._tts, "speakers", None) or [])
        cfg_wav = self._cfg.get("speaker_wav")
        if voice and os.path.isfile(voice):
            kwargs["speaker_wav"] = voice
        elif cfg_wav and os.path.isfile(cfg_wav):
            kwargs["speaker_wav"] = cfg_wav
        elif voice and voice in speakers:
            kwargs["speaker"] = voice
        elif speakers:
            # Multi-speaker model (e.g. XTTS-v2) needs *a* speaker; pick a default.
            kwargs["speaker"] = self._cfg.get("speaker") or speakers[0]
        try:
            kwargs["speed"] = float(speed) if speed else 1.0
            wav = np.asarray(self._tts.tts(**kwargs), dtype=np.float32)
        except TypeError:
            kwargs.pop("speed", None)   # some coqui models don't accept speed
            wav = np.asarray(self._tts.tts(**kwargs), dtype=np.float32)
        sr = int(getattr(self._tts.synthesizer, "output_sample_rate", 24000))
        return wav, sr

    def voices(self):
        return list(getattr(self._tts, "speakers", None) or [])


# --------------------------------------------------------------------------- #
# Parler-TTS (optional) — expressive, description-prompt driven
# --------------------------------------------------------------------------- #

class _ParlerBackend:
    family = "parler"
    default_voice = ""

    def __init__(self, model_name: str, config: dict):
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
        except ImportError as e:
            raise MissingEngineError(
                "Parler-TTS isn't installed. NOTE: parler-tts pins an old "
                "transformers/tokenizers/huggingface-hub that conflict with this "
                "server — do NOT pip install it into this environment. Run it in a "
                "separate venv as its own service, or use an expressive engine that "
                "works with this stack (e.g. Bark via the transformers pipeline)."
            ) from e
        import torch
        self._cfg = config or {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(self._device)
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._sr = int(self._model.config.sampling_rate)

    def _describe(self, voice: str, speed: float, emotion: str, style: str) -> str:
        # Parler is steered by a free-text description of the delivery; map the
        # UI controls (voice name, emotion, delivery style, speed) into one.
        speaker = (voice or "").strip()
        if speaker and (os.sep in speaker or speaker.lower().startswith(("af_", "am_", "bf_", "bm_"))):
            speaker = ""   # a file path or a Kokoro id is not a Parler speaker name
        who = speaker or self._cfg.get("speaker") or "A speaker"
        bits = [f"{who} speaks"]
        if emotion and emotion != "neutral":
            bits.append(f"in a {emotion} tone")
        smap = {"whispering": "whispering softly", "shouting": "shouting loudly",
                "monotone": "in a flat monotone", "expressive": "in a very expressive, animated way"}
        if style and style not in ("", "normal"):
            bits.append(smap.get(style, style))
        try:
            sp = float(speed or 1.0)
        except (TypeError, ValueError):
            sp = 1.0
        bits.append(f"at a {'slow' if sp < 0.9 else 'fast' if sp > 1.15 else 'moderate'} pace")
        return (" ".join(bits) +
                ". The recording is very high quality, the voice clear and close up "
                "with no background noise.")

    def synthesize(self, text: str, voice: str, speed: float, lang: str,
                   emotion: str = "", style: str = "") -> Tuple[np.ndarray, int]:
        description = self._cfg.get("description") or self._describe(voice, speed, emotion, style)
        input_ids = self._tok(description, return_tensors="pt").input_ids.to(self._device)
        prompt_ids = self._tok(text, return_tensors="pt").input_ids.to(self._device)
        gen = self._model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
        audio = np.asarray(gen.cpu().numpy().squeeze(), dtype=np.float32)
        return audio, self._sr

    def voices(self):
        return []


# --------------------------------------------------------------------------- #
# Bark (suno/bark) — expressive via text markup; works with current transformers
# --------------------------------------------------------------------------- #

class _BarkBackend:
    family = "bark"
    default_voice = "v2/en_speaker_6"
    # Curated English Bark presets by gender (speaker_6 is a clear male, speaker_9
    # is the commonly-used female). Override via config: "bark_voice_male" /
    # "bark_voice_female", or "bark_voices": {"male": ..., "female": ...}.
    _BARK_MALE = "v2/en_speaker_6"
    _BARK_FEMALE = "v2/en_speaker_9"

    def __init__(self, model_name: str, config: dict):
        # Uses the stable AutoProcessor + BarkModel API (not the pipeline) so
        # voice presets and generation params are passed reliably.
        from transformers import AutoProcessor, BarkModel
        import torch
        self._cfg = config or {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._proc = AutoProcessor.from_pretrained(model_name)
        self._model = BarkModel.from_pretrained(model_name).to(self._device)
        self._sr = int(getattr(self._model.generation_config, "sample_rate", 24000))

    def _markup(self, text: str, emotion: str, style: str) -> str:
        # Bark is steered by in-text cues rather than parameters.
        if style == "emphasis":
            text = text.upper()
        elif style == "singing":
            text = f"♪ {text} ♪"
        elif style == "whispering":
            text = f"[whispers] {text}"
        cue = {"laughter": "[laughs] ", "sigh": "[sighs] ", "gasp": "[gasps] "}.get(emotion, "")
        return cue + text

    def _resolve_preset(self, voice: str) -> str:
        v = (voice or "").strip()
        # An explicit Bark preset passes straight through.
        if v and ("speaker" in v or v.startswith("v2/")):
            return v
        # The editor sends Kokoro-style ids whose 2nd char is the gender
        # (af_/bf_ = female, am_/bm_ = male). Map that to a gendered preset.
        lv = v.lower()
        gender = "male" if (len(lv) >= 2 and lv[1] == "m") else \
                 ("female" if (len(lv) >= 2 and lv[1] == "f") else "")
        vmap = self._cfg.get("bark_voices") or {}
        if gender == "male":
            return self._cfg.get("bark_voice_male") or vmap.get("male") or self._BARK_MALE
        if gender == "female":
            return self._cfg.get("bark_voice_female") or vmap.get("female") or self._BARK_FEMALE
        return self._cfg.get("voice_preset") or self.default_voice

    def synthesize(self, text: str, voice: str, speed: float, lang: str,
                   emotion: str = "", style: str = "") -> Tuple[np.ndarray, int]:
        import torch
        # Speed isn't controllable in Bark; the voice maps to a gendered preset.
        preset = self._resolve_preset(voice)
        prompt = self._markup(text, emotion, style)
        inputs = self._proc(prompt, voice_preset=preset)
        inputs = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            audio = self._model.generate(**inputs)
        arr = np.asarray(audio.cpu().numpy().squeeze(), dtype=np.float32)
        return arr, self._sr

    def voices(self):
        return [f"v2/en_speaker_{i}" for i in range(10)]


# --------------------------------------------------------------------------- #
# Parler over HTTP — the real engine runs in an isolated venv as a microservice
# (parler-tts pins an old transformers that conflicts with this server's stack).
# --------------------------------------------------------------------------- #

class _RemoteParlerBackend:
    family = "parler"
    default_voice = ""

    def __init__(self, config: dict, managed_model: Optional[str] = None):
        self._cfg = config or {}
        self._url = str(self._cfg["service_url"]).rstrip("/")
        # When coderai launched the worker itself, remember the model so the
        # manager's eviction (which calls cleanup()) can shut it down.
        self._managed_model = managed_model

    def synthesize(self, text: str, voice: str, speed: float, lang: str,
                   emotion: str = "", style: str = "") -> Tuple[np.ndarray, int]:
        import io
        import requests
        import soundfile as sf
        payload = {"text": text, "voice": voice, "speed": speed,
                   "emotion": emotion, "style": style, "language": lang}
        if self._cfg.get("description"):
            payload["description"] = self._cfg["description"]
        resp = requests.post(self._url + "/speak", json=payload, timeout=600)
        resp.raise_for_status()
        data, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32), int(sr)

    def voices(self):
        return []

    def cleanup(self):
        # Called by the model manager on eviction; stop the worker we launched.
        if self._managed_model:
            try:
                from codai.api import parler_worker
                parler_worker.stop_service(self._managed_model)
            except Exception:
                pass


def load_backend(model_name: str, model_path: Optional[str], config: Optional[dict]):
    """Instantiate the TTS backend for ``model_name`` (cached by the caller)."""
    fam = _family(model_name)
    config = config or {}
    if fam == "kokoro":
        return _KokoroBackend(model_path or model_name, config)
    if fam == "coqui":
        return _CoquiBackend(model_name, config)
    if fam == "bark":
        return _BarkBackend(model_name, config)
    if fam == "parler":
        # An explicit service_url points at an externally-run service. Otherwise
        # coderai fully manages the worker: bootstrap its venv, spawn it, and
        # route to it — no manual setup needed.
        if config.get("service_url"):
            return _RemoteParlerBackend(config)
        from codai.api import parler_worker
        url = parler_worker.ensure_service(model_name)
        return _RemoteParlerBackend({**config, "service_url": url},
                                    managed_model=model_name)
    return _TransformersBackend(model_name, config)


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #

def encode_audio(samples: np.ndarray, sample_rate: int, fmt: str) -> Tuple[bytes, str]:
    """Encode float samples to the requested container, returning (bytes, fmt).

    WAV/FLAC/OGG go straight through soundfile; mp3 (and anything else) is muxed
    via ffmpeg when available, else falls back to WAV.
    """
    import soundfile as sf
    fmt = (fmt or "wav").lower()
    samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)

    sf_formats = {"wav": "WAV", "flac": "FLAC", "ogg": "OGG"}
    if fmt in sf_formats:
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format=sf_formats[fmt])
        return buf.getvalue(), fmt

    # mp3/other: write WAV then transcode with ffmpeg if present.
    wav = io.BytesIO()
    sf.write(wav, samples, sample_rate, format="WAV")
    wav_bytes = wav.getvalue()
    import shutil
    import subprocess
    if shutil.which("ffmpeg"):
        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-f", "wav", "-i", "pipe:0", "-f", fmt, "pipe:1"],
                input=wav_bytes, stdout=subprocess.PIPE, check=True,
            )
            return proc.stdout, fmt
        except Exception as exc:
            print(f"  ffmpeg transcode to {fmt} failed ({exc}); returning WAV")
    return wav_bytes, "wav"
