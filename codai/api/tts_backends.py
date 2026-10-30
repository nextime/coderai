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
    if "melo" in n:
        return "melo"
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


#: What each family can SPEAK. Every entry is the engine's own declared set,
#: read from the model rather than assumed:
#:
#:   coqui   XTTS-v2's own config.json `languages` (17)
#:   kokoro  the accents encoded in its voice ids -- see _KOKORO_ACCENTS below,
#:           which this is derived from at import so the two cannot drift
#:   bark    suno/bark's speaker_embeddings bank: 13 languages x speakers 0-9
#:   parler  parler-tts-mini-multilingual's model card (8 European languages)
#:
#: This exists because the languages were never advertised. All four TTS models
#: published `languages: null` on /v1/models while the STT models published
#: theirs correctly, so a client was told nothing and could only discover a
#: language by trying it -- which is how a 17-language engine gets taken for an
#: English-and-Italian one. A card's own `languages` still wins: this is the
#: default for a card that says nothing, not an override of one that does.
#:
#: `melo` and the generic `transformers` fallback are absent on purpose: what
#: they can say depends on the checkpoint, and a wrong list is worse than none.
_FAMILY_LANGUAGES: dict[str, list[str]] = {
    "coqui": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
              "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"],
    "bark": ["de", "en", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt",
             "ru", "tr", "zh"],
    "parler": ["en", "fr", "de", "es", "it", "pt", "pl", "nl"],
}


def family_languages(model_name: str, config: Optional[dict] = None) -> list[str]:
    """Languages this model can speak, or [] when nothing is known.

    A ``languages`` list on the model's own card wins -- an operator who has
    restricted or extended a model knows something this table does not.
    """
    if config:
        declared = config.get("languages")
        if declared:
            return [str(x) for x in declared]
    fam = _family(model_name)
    if fam == "kokoro":
        # Derived from the voice-id accents, so adding a voice family adds its
        # language without a second list to remember.
        return sorted({lang for lang, _ in _KOKORO_ACCENTS.values()})
    return list(_FAMILY_LANGUAGES.get(fam, []))


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


#: Kokoro's voice names, by (path, mtime) of the file they came from. Reading
#: them is a zip-index read of a 28 MB archive, and /v1/models is polled.
_KOKORO_VOICE_MEMO: dict = {}


def _kokoro_voices_file_if_present(config: Optional[dict] = None):
    """The voices file, ONLY if it is already on disk.

    Deliberately not :func:`_resolve_kokoro_files`: that downloads the official
    release when nothing local matches, and a model LISTING must never start a
    28 MB download. An absent file just means no voice names yet.
    """
    cand = (config or {}).get("voices_path")
    paths = [Path(cand)] if cand else []
    paths.append(_cache_dir() / "voices-v1.0.bin")
    for p in paths:
        try:
            if p.is_file() and p.stat().st_size > 0:
                return p
        except Exception:
            continue
    return None


def _kokoro_static_voices(config: Optional[dict] = None) -> list:
    """Kokoro's selectable voices, named without loading the model.

    The names come from the very file the backend selects out of, so the list
    cannot promise a voice synthesis would then reject. Empty when the file is
    not on disk — a listing does not fetch it.
    """
    f = _kokoro_voices_file_if_present(config)
    if f is None:
        return []
    try:
        st = f.stat()
        stamp = (str(f), st.st_mtime_ns, st.st_size)
        if _KOKORO_VOICE_MEMO.get("stamp") != stamp:
            import numpy as _np
            with _np.load(str(f)) as z:        # npz: reads the zip index only
                names = sorted(str(k) for k in z.files)
            _KOKORO_VOICE_MEMO.clear()
            _KOKORO_VOICE_MEMO["stamp"] = stamp
            _KOKORO_VOICE_MEMO["names"] = names
        return list(_KOKORO_VOICE_MEMO.get("names") or [])
    except Exception:
        return []


#: espeak-ng names English by REGION -- ``en-us``, ``en-gb`` -- and rejects the
#: bare code outright: `language "en" is not supported by the espeak backend`.
#: Every OpenAI-compatible client in existence sends "en", so Kokoro's English
#: voices were unreachable while its Italian ones worked, which reads as the
#: model being broken rather than as a two-letter code needing a region.
#: Only codes espeak actually refuses belong here. Measured against the running
#: engine 2026-10-30, one language per request: "de", "it", "nl", "pl", "tr",
#: "ru", "ko", "ar", "cs", "hu" and "sv" are all accepted bare and must pass
#: through untouched -- but "fr" is NOT, which this comment previously asserted
#: it was. French is regioned in espeak exactly like English (fr-fr, fr-be,
#: fr-ch), so Kokoro's French voice answered every request with a 500 while its
#: Italian one worked. Do not add a code here on the strength of the pattern:
#: probe it, because the pattern is what got "fr" wrong.
_ESPEAK_REGIONLESS = {
    "en": "en-us",
    "fr": "fr-fr",
    "pt": "pt-br",
    "zh": "cmn",
}


def espeak_language(lang: str) -> str:
    """``lang`` as espeak-ng names it."""
    want = str(lang or "").strip().lower().replace("_", "-")
    if not want:
        return "en-us"
    return _ESPEAK_REGIONLESS.get(want, want)


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
        _lang = espeak_language(lang)
        try:
            samples, sr = self._kokoro.create(
                text, voice=voice or self.default_voice, speed=speed or 1.0,
                lang=_lang,
            )
        except Exception as exc:
            # A language espeak does not know is a request problem, not a
            # server one, and the message has to say which language and what
            # to send instead -- "RuntimeError: language ... is not supported
            # by the espeak backend" reached the client as a 500.
            if "espeak" in str(exc).lower():
                raise ValueError(
                    f"this voice cannot speak {lang!r}: espeak-ng has no such "
                    f"language (tried {_lang!r}). espeak names some languages "
                    f"by region — send 'en-us' or 'en-gb' rather than 'en'."
                ) from exc
            raise
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
        self._cfg = config or {}
        try:
            from TTS.api import TTS  # coqui-tts, in this venv
        except ImportError as e:
            # Not here — but coqui-tts cannot share a transformers-5 venv, so
            # "not here" is the normal case. Look for it in a venv of its own,
            # the way parler and the OCR engines are found. A pod image with a
            # verified TTS venv used to answer "pip install coqui-tts" at this
            # exact line: the venv was there and nothing knew to use it.
            from codai.api import xtts_worker
            # Build the venv on first use, the way parler does. Opt out with
            # CODERAI_XTTS_AUTOINSTALL=0 on a host that manages its own venvs
            # or has no network.
            _ok, _why = xtts_worker.health()
            if not _ok and os.environ.get("CODERAI_XTTS_AUTOINSTALL", "1") != "0":
                xtts_worker.bootstrap()
                _ok, _why = xtts_worker.health()
            if _ok:
                print(f"[tts] coqui-tts from {xtts_worker.venv_dir()}", flush=True)
                self._tts = xtts_worker.XttsSubprocess(model_name)
                return
            # Say what is actually wrong with the venv. This used to report
            # that no venv had been found, over a venv that was there and
            # populated, which sent the operator looking in the wrong place.
            raise MissingEngineError(
                f"Coqui/XTTS models need their own venv ({xtts_worker.venv_dir()}), "
                f"and this one is not usable: {_why}. It cannot be installed "
                f"beside the server's transformers; coderai builds it on first "
                f"use unless CODERAI_XTTS_AUTOINSTALL=0."
            ) from e
        import torch
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

class _RemoteSpeechBackend:
    """A TTS engine that runs in its own venv and answers over HTTP.

    Used for the engines whose dependencies can't share this venv: parler-tts and
    MeloTTS. Both expose the same tiny contract — POST /speak -> WAV bytes."""
    default_voice = ""

    def __init__(self, config: dict, family: str = "parler",
                 managed_model: Optional[str] = None):
        self.family = family
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


# --------------------------------------------------------------------------- #
# Built-in voices
# --------------------------------------------------------------------------- #
#
# Clients need to offer a voice picker BEFORE the model is loaded, and a listing
# endpoint must never load one to answer -- a cheap availability check that
# loads is how a catalogue route turns into a several-hundred-MB loader.
#
# So each backend's own `voices` property stays authoritative, and what it
# returns is remembered the first time that model is loaded. A later listing is
# served from that memory, and says which it was.

#: How a family picks a voice at all. Not every engine HAS a voice list: parler
#: is steered by a natural-language description, which is why its `voices` is
#: empty and a client that only looks at the list concludes, wrongly, that it
#: cannot be directed.
_FAMILY_VOICE_SELECTION: dict[str, str] = {
    "kokoro": "builtin",
    "coqui": "builtin+clone",   # a named speaker, or speaker_wav to clone
    "bark": "builtin",
    "parler": "described",      # free-text description, no speaker names
    "melo": "builtin",
}

#: Kokoro encodes the accent and the gender in the id itself (``af_sarah`` =
#: American female). Derived, never guessed -- a family whose ids carry nothing
#: reports null rather than an invented gender.
_KOKORO_ACCENTS = {
    "a": ("en-us", "American English"), "b": ("en-gb", "British English"),
    "e": ("es", "Spanish"), "f": ("fr", "French"), "h": ("hi", "Hindi"),
    "i": ("it", "Italian"), "j": ("ja", "Japanese"), "p": ("pt-br", "Portuguese"),
    "z": ("zh", "Chinese"),
}


def _voice_meta(family: str, voice_id: str) -> dict:
    """What is genuinely known about one voice id.

    Unknown fields are ``None``, not a plausible-looking default: a gender shown
    against the wrong voice is worse than no gender at all.
    """
    vid = str(voice_id or "")
    out = {"id": vid, "name": vid, "gender": None, "languages": []}
    if family == "kokoro" and len(vid) >= 2 and vid[1] in ("f", "m"):
        acc = _KOKORO_ACCENTS.get(vid[0])
        out["gender"] = "female" if vid[1] == "f" else "male"
        if acc:
            out["languages"] = [acc[0]]
        if "_" in vid:
            out["name"] = vid.split("_", 1)[1].replace("-", " ").title()
    elif family == "bark":
        # v2/en_speaker_3 -> the language is in the id, the speaker is a number.
        tail = vid.split("/")[-1]
        if "_speaker_" in tail:
            out["languages"] = [tail.split("_speaker_")[0]]
            out["name"] = tail.replace("_", " ")
    return out


def _voices_cache_file() -> Path:
    return _cache_dir() / "tts_voices.json"


def remember_voices(model_name: str, backend) -> None:
    """Record a loaded backend's voice list so it can be listed without loading.

    Best effort in every direction: a backend that cannot enumerate, an
    unwritable cache or a malformed existing file must not affect synthesis,
    which is what the caller is actually doing.
    """
    try:
        voices = [str(v) for v in (getattr(backend, "voices", None) or [])]
    except Exception:
        return
    if not voices:
        return
    try:
        import json
        f = _voices_cache_file()
        try:
            cur = json.loads(f.read_text()) if f.is_file() else {}
            if not isinstance(cur, dict):
                cur = {}
        except Exception:
            cur = {}
        if cur.get(model_name) == voices:
            return
        cur[model_name] = voices
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cur, indent=2))
        tmp.replace(f)
    except Exception:
        pass


#: The parsed voices cache, keyed by (path, mtime, size). /v1/models is a
#: POLLED endpoint and now asks for voices on every TTS entry, so re-reading and
#: re-parsing the file per model per poll is a cost with no new information in
#: it. Re-read as soon as the file changes, which is what a load records.
_VOICES_MEMO: dict = {}


def _remembered_voices(model_name: str) -> list:
    try:
        import json
        f = _voices_cache_file()
        if not f.is_file():
            return []
        st = f.stat()
        stamp = (str(f), st.st_mtime_ns, st.st_size)
        cached = _VOICES_MEMO.get("stamp")
        if cached != stamp:
            parsed = json.loads(f.read_text())
            _VOICES_MEMO.clear()
            _VOICES_MEMO["stamp"] = stamp
            _VOICES_MEMO["data"] = parsed if isinstance(parsed, dict) else {}
        return [str(v) for v in (_VOICES_MEMO.get("data", {}).get(model_name) or [])]
    except Exception:
        return []


def family_voices(model_name: str, backend=None,
                  config: Optional[dict] = None) -> dict:
    """The model's built-in voices, and where the list came from.

    ``source`` is part of the answer on purpose: "none" means nothing is known
    yet, which a caller should show differently from an engine that genuinely
    has no voice list (``selection: "described"``).
    """
    fam = _family(model_name)
    ids, source = [], "none"
    if backend is not None:
        try:
            ids = [str(v) for v in (getattr(backend, "voices", None) or [])]
            if ids:
                source = "loaded"
        except Exception:
            ids = []
    if not ids:
        ids = _remembered_voices(model_name)
        if ids:
            source = "remembered"
    if not ids and fam == "kokoro":
        ids = _kokoro_static_voices(config)
        if ids:
            source = "static"
    if not ids and fam == "bark":
        # Deterministic ids, so they can be named without loading anything.
        # ALL of bark's languages, not just English: suno/bark ships a speaker
        # bank of 13 languages x speakers 0-9, and listing only the English ten
        # is what made a 13-language engine look like an English one. Ordered
        # English-first so a client taking voices[0] as a default still gets an
        # English speaker.
        ids = [f"v2/en_speaker_{i}" for i in range(10)] + [
            f"v2/{lang}_speaker_{i}"
            for lang in _FAMILY_LANGUAGES["bark"] if lang != "en"
            for i in range(10)
        ]
        source = "static"
    return {
        "voices": [_voice_meta(fam, v) for v in ids],
        "voices_source": source,
        "voice_selection": _FAMILY_VOICE_SELECTION.get(fam, "builtin"),
    }


def load_backend(model_name: str, model_path: Optional[str], config: Optional[dict]):
    """Instantiate the TTS backend for ``model_name`` (cached by the caller)."""
    backend = _load_backend(model_name, model_path, config)
    # Remember what it can say in, so a voice picker can be offered later
    # without loading the model just to ask.
    remember_voices(model_name, backend)
    return backend


def _load_backend(model_name: str, model_path: Optional[str], config: Optional[dict]):
    fam = _family(model_name)
    config = config or {}
    # Every backend that has to RESOLVE a checkpoint gets the card's path, with
    # the request's name only as a fallback. `model_name` is whatever the caller
    # asked for, which for a card registered under an alias is the alias -- and
    # an alias is not a repo id. Kokoro already did this; the others took
    # `model_name`, and both isolated workers failed on it, each in its own way:
    #
    #   XTTS:   "XTTS venv could not load 'XTTS-v2': ValueError: not enough
    #            values to unpack (expected 4, got 1)" -- coqui splits a model
    #            name into type/lang/dataset/model, and the bare alias has no
    #            separator at all (the HF id 'coqui/XTTS-v2' has one, which is
    #            the case the worker's translation table was written for).
    #   parler: "parler-tts-mini-multilingual is not the path to a directory
    #            containing a file named config.json" -- the org is missing, so
    #            the real repo 'parler-tts/parler-tts-mini-multilingual' is
    #            never looked up, and the worker reports it as "not in the cache,
    #            download it from the model interface first" for a model that IS
    #            in the cache.
    _ckpt = model_path or model_name
    if fam == "kokoro":
        return _KokoroBackend(_ckpt, config)
    if fam == "coqui":
        return _CoquiBackend(_ckpt, config)
    if fam == "bark":
        return _BarkBackend(_ckpt, config)
    if fam == "melo":
        # MeloTTS is OpenVoice V2's base speaker model. Its pins conflict with this
        # stack the way parler's do, so it runs in an isolated venv worker and is
        # reached over HTTP — same contract, same managed lifecycle.
        if config.get("service_url"):
            return _RemoteSpeechBackend(config, family="melo")
        from codai.api import melotts_worker
        url = melotts_worker.ensure_service(_ckpt)
        return _RemoteSpeechBackend({**config, "service_url": url}, family="melo",
                                    managed_model=_ckpt)
    if fam == "parler":
        # An explicit service_url points at an externally-run service. Otherwise
        # coderai fully manages the worker: bootstrap its venv, spawn it, and
        # route to it — no manual setup needed.
        if config.get("service_url"):
            return _RemoteSpeechBackend(config, family="parler")
        from codai.api import parler_worker
        url = parler_worker.ensure_service(_ckpt)
        return _RemoteSpeechBackend({**config, "service_url": url},
                                    family="parler", managed_model=_ckpt)
    return _TransformersBackend(_ckpt, config)


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


# --------------------------------------------------------------------------- #
# Whether an engine can run at all, before anything is loaded
# --------------------------------------------------------------------------- #
#
# coqui-tts and parler-tts each need a transformers the server cannot have, so
# each runs in a venv of its own, built on first use. That build downloads
# several GB, which is a surprising thing for a client to discover inside its
# first synthesis request -- reported from production as "speech engines listed
# but not installed", with a 501 and a 500 as the only explanation.
#
# This answers the question before a user picks a voice, and it answers it
# WITHOUT importing anything from the venv: an import probe into a foreign
# interpreter costs seconds, and a catalogue call that costs seconds is how a
# listing route becomes a loader.

#: family -> the packages whose presence makes the venv usable.
#:
#: `torch` is in every list on purpose. A `pip install coqui-tts` brought the
#: engine and its ninety dependencies and NO torch, so the venv looked
#: complete, this route reported `env_ready: true`, and the first synthesis
#: failed on `import TTS` — measured on zeiss, 2026-10-29. The engine package
#: alone is not evidence that the venv can run.
_VENV_FAMILIES = {
    "coqui": ("TTS", "torch"),
    "parler": ("parler_tts", "torch"),
    "melo": ("melo", "torch"),
}


def _family_venv_dir(fam: str):
    """Where this family's isolated venv lives, or None if it needs none."""
    try:
        if fam == "coqui":
            from codai.api.xtts_worker import venv_dir
            return venv_dir()
        if fam == "parler":
            from codai.api.parler_worker import _VENV_DIR
            return _VENV_DIR
        if fam == "melo":
            from codai.api.melotts_worker import _VENV_DIR
            return _VENV_DIR
    except Exception:
        return None
    return None


def environment_report(model_name: str) -> dict:
    """``{needs_venv, env_ready, env_detail}`` for this model's engine.

    ``env_ready`` false does not mean the engine is unusable: the venv is built
    on first use. It means the first request will spend minutes building it,
    which is exactly what a client wants to know before offering the voice.
    """
    from pathlib import Path

    fam = _family(model_name)
    modules = _VENV_FAMILIES.get(fam)
    if not modules:
        return {"needs_venv": False, "env_ready": True, "env_detail": ""}
    d = _family_venv_dir(fam)
    if d is None:
        return {"needs_venv": True, "env_ready": False,
                "env_detail": "this engine's venv location is not resolvable here"}
    try:
        from codai.api import venv_env
        # probe=False: filesystem only. This route is a catalogue call.
        ok, why = venv_env.venv_usable(d, probe=False)
    except Exception as exc:
        return {"needs_venv": True, "env_ready": False,
                "env_detail": f"the venv could not be inspected ({exc})"}
    if not ok:
        return {"needs_venv": True, "env_ready": False,
                "env_detail": f"{d}: {why or 'it is not there yet'}"}
    # The package directories, not an import: a populated venv is the
    # evidence, and importing into it would cost seconds on a catalogue call.
    missing = [m for m in modules
               if not any(Path(d).glob(f"lib/python*/site-packages/{m}"))]
    if missing:
        return {"needs_venv": True, "env_ready": False,
                "env_detail": (f"{d} exists but {', '.join(missing)} "
                               f"{'is' if len(missing) == 1 else 'are'} not "
                               f"installed in it")}
    return {"needs_venv": True, "env_ready": True, "env_detail": str(d)}
