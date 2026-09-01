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

"""Pluggable speech-to-text backends (non whisper-server).

Mirrors the duck-typed family pattern of :mod:`codai.api.tts_backends`: each
backend is a plain class exposing a uniform ``transcribe`` method, and
:func:`load_stt_backend` is the factory that dispatches on a model's configured
``backend`` (falling back to a name heuristic).

Backends
--------
* ``wav2vec2`` — HF ``transformers`` CTC ASR (also HuBERT / MMS / Seamless). Runs
  in-process on the GPU engine (torch is already in this venv).
* ``vosk``     — Kaldi-based offline STT. CPU. One model directory per language.
* ``nemo`` / ``canary`` — NVIDIA NeMo (Canary / Parakeet). NeMo pins a
  conflicting stack, so it runs in an isolated venv worker
  (:mod:`codai.api.canary_worker`); this module talks to it over HTTP.

Every backend's ``transcribe`` returns a dict::

    {"text": str, "segments": [{"start": float, "end": float, "text": str}, ...],
     "language": str | None}

``segments`` may be empty when a backend can't produce word/segment timings.
"""

import os
import subprocess
import tempfile
import wave
from typing import Optional


# --------------------------------------------------------------------------- #
# family detection
# --------------------------------------------------------------------------- #

def _family(model_name: str, config: Optional[dict]) -> str:
    """Classify a model into an STT backend family.

    An explicit ``backend`` in config is authoritative (that's what the model
    page sets); otherwise fall back to substring heuristics on the name."""
    cfg = config or {}
    b = (cfg.get("backend") or "").strip().lower()
    if b in ("wav2vec2", "wav2vec", "transformers-asr", "hf-asr"):
        return "wav2vec2"
    if b == "crisperwhisper":
        return "crisperwhisper"
    if b in ("whisper-hf", "whisper-transformers"):
        return "whisper_hf"
    if b == "vosk":
        return "vosk"
    if b in ("nemo", "canary", "parakeet"):
        return "nemo"
    n = (model_name or "").lower()
    if "vosk" in n:
        return "vosk"
    if any(x in n for x in ("canary", "parakeet", "nemo", ".nemo")):
        return "nemo"
    # CrisperWhisper runs in its own isolated venv worker (verbatim + word
    # timestamps need a transformers version the shared 5.x venv can't provide).
    if "crisperwhisper" in n:
        return "crisperwhisper"
    # Other HF Whisper checkpoints = seq2seq Whisper via the shared-venv pipeline.
    if "whisper" in n:
        return "whisper_hf"
    if any(x in n for x in ("wav2vec", "hubert", "mms-", "seamless", "wavlm")):
        return "wav2vec2"
    # Default for an unknown non-whisper audio model: treat as an HF ASR model.
    return "wav2vec2"


# --------------------------------------------------------------------------- #
# audio helpers
# --------------------------------------------------------------------------- #

def _to_wav_16k_mono(src_path: str) -> str:
    """Convert any audio file to a 16 kHz mono PCM-s16le WAV via ffmpeg.

    Returns the path to a new temp WAV (caller deletes it). Raises if ffmpeg
    fails. Vosk needs exactly this format; other backends can decode themselves."""
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", "-f", "wav", out],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        try:
            os.unlink(out)
        except OSError:
            pass
        raise RuntimeError(
            "ffmpeg conversion failed: "
            + (proc.stderr.decode("utf-8", "replace").strip().splitlines() or [""])[-1])
    return out


# --------------------------------------------------------------------------- #
# Wav2Vec2 / HF transformers ASR
# --------------------------------------------------------------------------- #

class _Wav2Vec2Backend:
    """HF ``transformers`` automatic-speech-recognition pipeline.

    Works for Wav2Vec2 / HuBERT / WavLM / MMS / Seamless CTC checkpoints. Most of
    these are language-specific (or, for MMS/XLSR, need an adapter per language);
    ``language`` is passed through when the underlying model supports it and is
    otherwise informational only."""

    family = "wav2vec2"

    def __init__(self, model_name: str, model_path: Optional[str], config: dict):
        self.model_name = model_name
        self.config = config or {}
        self._pipe = None
        # Whisper HF checkpoints (e.g. CrisperWhisper) are seq2seq and take
        # generate kwargs (language/task); CTC models (wav2vec2) don't.
        self._is_whisper = bool(self.config.get("_is_whisper")) or \
            "whisper" in (model_name or "").lower()

    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline
        use_cuda = torch.cuda.is_available()
        device = 0 if use_cuda else -1
        target = self.config.get("model_path") or self.model_name
        # Load in FP16 on GPU to roughly halve VRAM (~3GB→~1.5GB) — CTC inference is
        # fine in half precision. CPU stays FP32. Override with config dtype
        # (float16 | bfloat16 | float32).
        pipe_kwargs = dict(
            model=target,
            device=device,
            chunk_length_s=int(self.config.get("chunk_length_s", 30) or 30),
            stride_length_s=int(self.config.get("stride_length_s", 5) or 5),
        )
        if use_cuda:
            dt = str(self.config.get("dtype") or "float16").lower()
            dtype = {"float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
                     "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}.get(dt)
            if dtype is not None:
                pipe_kwargs["torch_dtype"] = dtype
        try:
            self._pipe = pipeline("automatic-speech-recognition", **pipe_kwargs)
        except Exception:
            # Some checkpoints/ops reject half precision — fall back to FP32.
            pipe_kwargs.pop("torch_dtype", None)
            self._pipe = pipeline("automatic-speech-recognition", **pipe_kwargs)

    def transcribe(self, audio_path: str, language: Optional[str] = None,
                   prompt: Optional[str] = None, temperature: float = 0.0,
                   target_language: Optional[str] = None,
                   word_timestamps: bool = False) -> dict:
        self._ensure()
        gen_kwargs = {}
        if self._is_whisper:
            # Whisper seq2seq: choose transcribe vs translate, pass source language.
            gen_kwargs["task"] = "translate" if target_language else "transcribe"
            if language:
                gen_kwargs["language"] = language
        elif language and self.config.get("language_aware"):
            # MMS / Seamless accept a target language; plain wav2vec2 ignores it.
            gen_kwargs["language"] = language
        # return_timestamps: CTC uses "chunk" / "word". Whisper uses "word" or
        # True (segment) — but some checkpoints (e.g. CrisperWhisper) ship a custom
        # generation config whose timestamp postprocessing breaks on transformers
        # 5.x. So default Whisper to TEXT-ONLY (None) and only attempt timestamps
        # when explicitly requested, falling back to text on any failure.
        if word_timestamps:
            mode = "word"
        elif self._is_whisper:
            mode = None
        else:
            mode = "chunk"
        # Decode to 16 kHz mono WAV first so any container (mp4/m4a/webm/…) is
        # handled uniformly — feeding a raw mp4 path can yield empty audio.
        wav = None
        try:
            wav = _to_wav_16k_mono(audio_path)
            src = wav
        except Exception:
            src = audio_path

        def _run(ts):
            if ts is None:
                return self._pipe(src, generate_kwargs=gen_kwargs or None)
            return self._pipe(src, return_timestamps=ts, generate_kwargs=gen_kwargs or None)

        try:
            out = _run(mode)
        except Exception as e:
            if mode is None:
                raise
            # Timestamp path failed (checkpoint/transformers incompatibility) —
            # retry text-only so transcription still succeeds.
            print(f"[stt] {self.model_name}: return_timestamps={mode!r} failed "
                  f"({type(e).__name__}); retrying text-only")
            try:
                out = _run(None)
            except Exception:
                out = self._pipe(src)
        finally:
            if wav:
                try:
                    os.unlink(wav)
                except OSError:
                    pass
        text = (out.get("text") if isinstance(out, dict) else str(out)) or ""
        segments, words = [], []
        for ch in (out.get("chunks") or []) if isinstance(out, dict) else []:
            ts = ch.get("timestamp") or (None, None)
            item = {"start": ts[0] or 0.0, "end": ts[1] or 0.0,
                    "text": (ch.get("text") or "").strip()}
            if word_timestamps:
                words.append({"word": item["text"], "start": item["start"],
                              "end": item["end"]})
            else:
                segments.append(item)
        return {"text": text.strip(), "segments": segments, "words": words,
                "language": language}

    def cleanup(self):
        self._pipe = None


# --------------------------------------------------------------------------- #
# Vosk
# --------------------------------------------------------------------------- #

class _VoskBackend:
    """Kaldi-based offline recognizer. CPU-only, one model dir per language.

    The ``model_path`` (or ``model_name``) must point at an unpacked Vosk model
    directory. Produces word-level timings, coalesced into ~one segment per Vosk
    result block."""

    family = "vosk"

    def __init__(self, model_name: str, model_path: Optional[str], config: dict):
        self.model_name = model_name
        self.config = config or {}
        self._model = None
        self._model_dir = (self.config.get("model_path") or model_path
                           or model_name)

    def _ensure(self):
        if self._model is not None:
            return
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        if not os.path.isdir(self._model_dir):
            raise RuntimeError(
                f"Vosk model directory not found: {self._model_dir}. Download/unpack "
                "the language model and set its path in the model config.")
        self._model = Model(self._model_dir)

    def transcribe(self, audio_path: str, language: Optional[str] = None,
                   prompt: Optional[str] = None, temperature: float = 0.0,
                   target_language: Optional[str] = None,
                   word_timestamps: bool = False) -> dict:
        import json
        from vosk import KaldiRecognizer
        self._ensure()
        wav_path = _to_wav_16k_mono(audio_path)
        try:
            wf = wave.open(wav_path, "rb")
            rec = KaldiRecognizer(self._model, wf.getframerate())
            rec.SetWords(True)
            segments, texts, words = [], [], []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    _collect_vosk_result(json.loads(rec.Result()), segments, texts, words)
            _collect_vosk_result(json.loads(rec.FinalResult()), segments, texts, words)
            wf.close()
            return {"text": " ".join(t for t in texts if t).strip(),
                    "segments": segments, "words": words,
                    "language": language or self.config.get("language")}
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def cleanup(self):
        self._model = None


def _collect_vosk_result(r: dict, segments: list, texts: list, words: list = None):
    txt = (r.get("text") or "").strip()
    if not txt:
        return
    texts.append(txt)
    wlist = r.get("result") or []
    if wlist:
        segments.append({"start": wlist[0].get("start", 0.0),
                         "end": wlist[-1].get("end", 0.0), "text": txt})
        if words is not None:
            for w in wlist:
                words.append({"word": w.get("word", ""), "start": w.get("start", 0.0),
                              "end": w.get("end", 0.0)})


# --------------------------------------------------------------------------- #
# NVIDIA NeMo (Canary / Parakeet) — isolated-venv worker over HTTP
# --------------------------------------------------------------------------- #

class _RemoteNemoBackend:
    """Talks to the managed NeMo worker (isolated venv). Canary supports source
    ``language`` selection and X↔EN translation via ``target_language``."""

    family = "nemo"

    def __init__(self, model_name: str, config: dict, service_url: Optional[str] = None):
        self.model_name = model_name
        self.config = config or {}
        self.service_url = (service_url or self.config.get("service_url") or "").rstrip("/")

    def transcribe(self, audio_path: str, language: Optional[str] = None,
                   prompt: Optional[str] = None, temperature: float = 0.0,
                   target_language: Optional[str] = None,
                   word_timestamps: bool = False) -> dict:
        import requests
        params = {}
        if language:
            params["language"] = language
        if target_language:
            params["target_language"] = target_language
        if word_timestamps:
            params["word_timestamps"] = "1"
        with open(audio_path, "rb") as f:
            resp = requests.post(f"{self.service_url}/transcribe",
                                 params=params, data=f.read(), timeout=1800,
                                 headers={"Content-Type": "application/octet-stream"})
        if resp.status_code != 200:
            raise RuntimeError(f"NeMo worker error {resp.status_code}: {resp.text[:300]}")
        j = resp.json()
        if j.get("error"):
            raise RuntimeError(j["error"])
        return {"text": (j.get("text") or "").strip(),
                "segments": j.get("segments") or [],
                "words": j.get("words") or [],
                "language": j.get("language") or language}

    def cleanup(self):
        # Tear the isolated worker down so eviction actually frees its VRAM.
        try:
            from codai.api import canary_worker
            canary_worker.stop_service(self.model_name)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# CrisperWhisper — isolated-venv worker (verbatim + word timestamps)
# --------------------------------------------------------------------------- #

class _RemoteCrisperWhisperBackend:
    """Talks to the managed CrisperWhisper worker (isolated venv, pinned
    transformers) — verbatim transcription with precise word timestamps."""

    family = "crisperwhisper"

    def __init__(self, model_name: str, config: dict, service_url: Optional[str] = None):
        self.model_name = model_name
        self.config = config or {}
        self.service_url = (service_url or self.config.get("service_url") or "").rstrip("/")

    def transcribe(self, audio_path: str, language: Optional[str] = None,
                   prompt: Optional[str] = None, temperature: float = 0.0,
                   target_language: Optional[str] = None,
                   word_timestamps: bool = False) -> dict:
        import requests
        params = {"task": "translate" if target_language else "transcribe"}
        if language:
            params["language"] = language
        with open(audio_path, "rb") as f:
            resp = requests.post(f"{self.service_url}/transcribe", params=params,
                                 data=f.read(), timeout=1800,
                                 headers={"Content-Type": "application/octet-stream"})
        if resp.status_code != 200:
            raise RuntimeError(f"CrisperWhisper worker error {resp.status_code}: {resp.text[:300]}")
        j = resp.json()
        if j.get("error"):
            raise RuntimeError(j["error"])
        return {"text": (j.get("text") or "").strip(),
                "segments": j.get("segments") or [],
                "words": j.get("words") or [],
                "language": j.get("language") or language}

    def cleanup(self):
        try:
            from codai.api import crisperwhisper_worker
            crisperwhisper_worker.stop_service(self.model_name)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #

def load_stt_backend(model_name: str, model_path: Optional[str], config: Optional[dict]):
    """Instantiate the STT backend for ``model_name`` (cached by the caller).

    Dispatches on the model's configured ``backend`` (set from the model page),
    falling back to a name heuristic. NeMo models are served by an isolated venv
    worker, which this call bootstraps/starts on demand."""
    config = config or {}
    fam = _family(model_name, config)
    # The requested `model` is usually an alias (e.g. "vosk-en"); the real model
    # location — a Vosk dir, an HF id, or a NeMo checkpoint id — is the config's
    # `path`. Resolve it so backends never get the alias as their model reference.
    real = (model_path or config.get("model_path") or config.get("path") or model_name)
    if fam == "vosk":
        return _VoskBackend(real, real, config)
    if fam == "nemo":
        if config.get("service_url"):
            return _RemoteNemoBackend(real, config)
        from codai.api import canary_worker
        url = canary_worker.ensure_service(real, config)
        return _RemoteNemoBackend(real, {**config, "service_url": url}, service_url=url)
    if fam == "crisperwhisper":
        if config.get("service_url"):
            return _RemoteCrisperWhisperBackend(real, config)
        from codai.api import crisperwhisper_worker
        url = crisperwhisper_worker.ensure_service(real, config)
        return _RemoteCrisperWhisperBackend(real, {**config, "service_url": url},
                                            service_url=url)
    if fam == "whisper_hf":
        return _Wav2Vec2Backend(real, real, {**config, "_is_whisper": True})
    return _Wav2Vec2Backend(real, real, config)


# Family names that this module (not whisper-server / faster-whisper) handles.
STT_FAMILIES = {"wav2vec2", "vosk", "nemo", "whisper_hf", "crisperwhisper"}


def resolve_family(model_name: str, config: Optional[dict]) -> str:
    """Public helper for the dispatcher: which family serves this model."""
    return _family(model_name, config)
