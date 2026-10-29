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

from codai.api.lang_codes import normalize as _norm_lang


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

# --------------------------------------------------------------------------- #
# What a speech model actually needs on the card
# --------------------------------------------------------------------------- #
#
# A transcription pass peaks well above the weights it loaded: the encoder runs
# over a 30 s window, and for a verbatim model (CrisperWhisper) the timestamp
# head runs over it again. The number that used to be reserved was 3 GB for
# every whisper-family model, which the isolated worker's own comment had
# already measured as about a third of the real peak -- so eviction freed a
# third of the room needed, the load "succeeded", and the pass OOMed with the
# card full of models nobody had asked to keep.
#
# Three numbers per model, all of them measured after the first pass:
#
#   weights    what the load put on the card
#   peak       the high-water mark of a pass
#   activation peak - weights: what a RESIDENT model still needs per pass
#
# Until a model has run once, `_FIRST_PASS_ACTIVATION_GB` stands in for the
# third one. It is deliberately modest: the failure it prevents was an 86 MiB
# allocation, and asking for a gigabyte on a card this model is already
# resident on is cheap, where asking for eight would evict a working set for
# nothing.

#: Family defaults for the whole footprint, used only until a real measurement
#: exists. whisper_hf matches crisperwhisper_worker.DEFAULT_VRAM_GB, which was
#: measured rather than guessed.
_FAMILY_RESERVE_GB = {"whisper_hf": 8.0, "crisperwhisper": 8.0, "wav2vec2": 3.0}
_DEFAULT_RESERVE_GB = 2.0
_FIRST_PASS_ACTIVATION_GB = 1.0

#: model key -> measured activation (peak - weights), high-water mark.
_activation_gb: dict = {}


def model_key(model_name: str) -> str:
    """The key the model manager tracks a speech model's residency under."""
    return f"audio:{model_name}"


def _manager():
    from codai.models.manager import multi_model_manager
    return multi_model_manager


def reserve_gb(model_name: str, config: Optional[dict] = None,
               family: Optional[str] = None) -> float:
    """What to free before LOADING ``model_name``.

    Order of trust: the operator's ``used_vram_gb`` from the model page, then
    the peak this very model measured on an earlier run, then the family
    default. Same order as the isolated worker uses, because they are the same
    model loaded two ways and they must not disagree about its size.
    """
    cfg = config or {}
    explicit = 0.0
    try:
        explicit = float(cfg.get("used_vram_gb") or 0.0)
    except (TypeError, ValueError):
        explicit = 0.0
    if explicit > 0:
        return explicit
    try:
        measured = float(_manager()._measured_vram_gb.get(model_key(model_name)) or 0.0)
    except Exception:
        measured = 0.0
    if measured > 0:
        return measured
    fam = (family or "").strip().lower()
    return _FAMILY_RESERVE_GB.get(fam, _DEFAULT_RESERVE_GB)


def activation_gb(model_name: str) -> float:
    """What a RESIDENT model still needs for one pass (measured, else the floor)."""
    got = _activation_gb.get(model_key(model_name)) or 0.0
    return got if got > 0 else _FIRST_PASS_ACTIVATION_GB


def record_pass(model_name: str, weights_gb: float, activation: float) -> None:
    """Remember what a pass cost: its weights and what it needed on top.

    ``activation`` is the DELTA over what was already allocated when the pass
    started, not the process's absolute peak. The absolute peak would charge
    this model for every other model resident in the same process -- a 16 GB
    image model sitting on the card while a transcription runs would be
    recorded as part of the speech model's footprint, and every later load of
    it would evict the world.

    High-water marks, never averages: a one-word clip touches a fraction of
    what a full 30 s window does, and an average would under-reserve for the
    longer audio that follows.
    """
    key = model_key(model_name)
    try:
        act = max(0.0, float(activation or 0.0))
        if act > (_activation_gb.get(key) or 0.0):
            _activation_gb[key] = act
        reserve = max(0.0, float(weights_gb or 0.0)) + act
        if reserve > 0:
            mgr = _manager()
            prev = float(mgr._measured_vram_gb.get(key) or 0.0)
            if reserve > prev + 0.05:
                mgr._measured_vram_gb[key] = reserve
                print(f"[stt] measured {reserve:.2f} GB for {model_name} "
                      f"({weights_gb:.2f} weights + {act:.2f} per pass)"
                      + (f", was {prev:.2f}" if prev else ""), flush=True)
    except Exception as exc:
        print(f"[stt] could not record the measurement: {exc}", flush=True)


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
        self._weights_gb = 0.0
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
        _before = torch.cuda.memory_allocated() / 1e9 if use_cuda else 0.0
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
        # What the weights cost, so a pass can report what IT cost on top (see
        # record_pass). Measured rather than assumed: the reserve that used to
        # be hard-coded here was about a third of the real peak.
        if use_cuda:
            self._weights_gb = max(0.0, torch.cuda.memory_allocated() / 1e9 - _before)

    def transcribe(self, audio_path: str, language: Optional[str] = None,
                   prompt: Optional[str] = None, temperature: float = 0.0,
                   target_language: Optional[str] = None,
                   word_timestamps: bool = False) -> dict:
        self._ensure()
        gen_kwargs = {}
        if self._is_whisper:
            # Whisper seq2seq: choose transcribe vs translate, pass source language.
            # Whisper's `language` is an English NAME, so a regional tag from a
            # TTS voice ("en-us") has to be mapped first -- handed through as is
            # it raises "Unsupported language: en-us" and the whole request 500s.
            # An unmappable tag becomes None, i.e. auto-detect.
            gen_kwargs["task"] = "translate" if target_language else "transcribe"
            lang = _norm_lang(language, "whisper_hf")
            if lang:
                gen_kwargs["language"] = lang
        elif language and self.config.get("language_aware"):
            # MMS / Seamless accept a target language; plain wav2vec2 ignores it.
            lang = _norm_lang(language, "wav2vec2")
            if lang:
                gen_kwargs["language"] = lang
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
            # `generate_kwargs` is OMITTED when empty rather than passed as
            # None: a CTC pipeline (wav2vec2, MMS) iterates it unconditionally
            # and raises `TypeError: 'NoneType' object is not iterable`, which
            # the retry below then swallowed into a text-only answer. So every
            # wav2vec2 request for word timings returned none of them, and the
            # only trace was one line in the log.
            kw = {}
            if gen_kwargs:
                kw["generate_kwargs"] = gen_kwargs
            if ts is not None:
                kw["return_timestamps"] = ts
            return self._pipe(src, **kw)

        # Room for THIS pass, on a card the model is already resident on. A
        # resident model is invisible to the load-time reservation: the TTS and
        # visual models of the same job load around it, fill the card to the
        # edge, and the next 30 s window dies on an 86 MiB allocation -- once
        # per scene, with every caption losing its word timings (measured on a
        # production render, 2026-10-29).
        _need = activation_gb(self.model_name)
        _cuda = False
        _held = 0.0
        try:
            import torch as _t
            _cuda = _t.cuda.is_available()
            if _cuda:
                _manager().ensure_free_vram(
                    _need, reason=f"a transcription pass on {self.model_name}")
                # What is allocated BEFORE the pass, so what the pass itself
                # needs can be measured as a delta rather than as the whole
                # process's peak -- which would include every other model
                # resident beside this one.
                _t.cuda.reset_peak_memory_stats()
                _held = _t.cuda.memory_allocated() / 1e9
        except Exception as _exc:
            print(f"[stt] could not reserve room for the pass ({_exc})", flush=True)

        def _run_or_make_room(ts):
            """One pass; on an OOM, free the card and try once more.

            A transcription is seconds of work on audio that already exists, so
            the second attempt is the cheap outcome -- the expensive one is a
            render whose captions have no timings."""
            try:
                return _run(ts)
            except Exception as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                print(f"[stt] {self.model_name}: out of VRAM mid-pass — "
                      f"freeing {_need:.2f} GB and retrying once", flush=True)
                try:
                    import torch as _t2
                    _manager()._evict_models_for_vram(
                        max(_need, reserve_gb(self.model_name, self.config,
                                              getattr(self, "family", None))))
                    _t2.cuda.empty_cache()
                except Exception:
                    pass
                return _run(ts)

        try:
            out = _run_or_make_room(mode)
        except Exception as e:
            if mode is None:
                raise
            # Timestamp path failed (checkpoint/transformers incompatibility) —
            # retry text-only so transcription still succeeds.
            print(f"[stt] {self.model_name}: return_timestamps={mode!r} failed "
                  f"({type(e).__name__}); retrying text-only")
            try:
                out = _run_or_make_room(None)
            except Exception:
                out = self._pipe(src)
        finally:
            if _cuda:
                try:
                    import torch as _t3
                    record_pass(self.model_name, self._weights_gb,
                                _t3.cuda.max_memory_allocated() / 1e9 - _held)
                except Exception:
                    pass
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
                    # Vosk chooses the language by model directory and ignores
                    # the argument; report the normalised code so the response
                    # doesn't echo a dialect tag Vosk never honoured.
                    "language": (_norm_lang(language, "vosk")
                                 or self.config.get("language"))}
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
        # Canary names its languages by ISO-639-1 code, so a regional tag from a
        # TTS voice ("pt-br") is reduced to "pt" first. Unmappable -> omitted,
        # which the worker reads as auto-detect.
        lang = _norm_lang(language, "nemo")
        if lang:
            params["language"] = lang
        if target_language:
            tgt = _norm_lang(target_language, "nemo")
            if tgt:
                params["target_language"] = tgt
        if word_timestamps:
            params["word_timestamps"] = "1"
        with open(audio_path, "rb") as f:
            audio = f.read()

        def _post():
            return requests.post(f"{self.service_url}/transcribe", params=params,
                                 data=audio, timeout=1800,
                                 headers={"Content-Type": "application/octet-stream"})
        try:
            resp = _post()
        except requests.exceptions.ConnectionError:
            # Self-heal: worker died/restarted → re-ensure and retry once.
            from codai.api import canary_worker
            self.service_url = canary_worker.ensure_service(
                self.model_name, self.config).rstrip("/")
            resp = _post()
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
        # The worker hands `language` to transformers' Whisper generate(), which
        # accepts only the English NAME: "en-us" raised "Unsupported language"
        # and surfaced as a 500 on every captioned scene (2026-10-27).
        lang = _norm_lang(language, "crisperwhisper")
        if lang:
            params["language"] = lang
        with open(audio_path, "rb") as f:
            audio = f.read()

        from codai.api import crisperwhisper_worker

        def _post():
            return requests.post(f"{self.service_url}/transcribe", params=params,
                                 data=audio, timeout=1800,
                                 headers={"Content-Type": "application/octet-stream"})
        # Room for the pass before asking for it. The worker is resident and
        # idle between scenes, which is invisible to the load-time reservation:
        # the TTS and visual models of the same job load around it and fill the
        # card, and then every scene came back
        #
        #   CrisperWhisper worker error 500: CUDA out of memory. Tried to
        #   allocate 86.00 MiB
        #
        # (measured on a production render, 2026-10-29). 86 MiB is what a
        # resident model still needs, and nothing was keeping it.
        _need = activation_gb(self.model_name)
        try:
            _manager().ensure_free_vram(
                _need, reason=f"a transcription pass on {self.model_name}")
        except Exception as _exc:
            print(f"[stt] could not reserve room for the pass ({_exc})", flush=True)

        # Busy for the length of the pass: the worker is a separate process, so the
        # manager cannot tell a loaded-and-idle one from a loaded-and-transcribing
        # one, and the VRAM releaser must not tear this down between the two lines
        # below.
        with crisperwhisper_worker.inflight(self.model_name):
            try:
                resp = _post()
            except requests.exceptions.ConnectionError:
                # Worker died/restarted since this handle was cached — re-ensure it
                # and retry once (self-heal, so a stale service_url doesn't stick).
                self.service_url = crisperwhisper_worker.ensure_service(
                    self.model_name, self.config).rstrip("/")
                resp = _post()
            if resp.status_code == 500 and "out of memory" in (resp.text or "").lower():
                # The worker's own OOM, in another process: free the card from
                # this side and ask again. A pass is seconds of work on audio
                # that already exists; the expensive outcome is a render whose
                # captions have no timings at all.
                _want = max(_need, crisperwhisper_worker.reserved_vram_gb(
                    self.model_name, self.config))
                print(f"[stt] {self.model_name}: the worker ran out of VRAM — "
                      f"freeing {_want:.2f} GB and retrying once", flush=True)
                try:
                    _manager()._evict_models_for_vram(_want)
                except Exception:
                    pass
                resp = _post()
        if resp.status_code != 200:
            raise RuntimeError(f"CrisperWhisper worker error {resp.status_code}: {resp.text[:300]}")
        j = resp.json()
        if j.get("error"):
            raise RuntimeError(j["error"])
        # The worker reports its own peak, which is the only honest number for a
        # footprint that lives in another process. Feed it back so the next
        # reservation is measured instead of guessed.
        crisperwhisper_worker.record_measurement(
            self.model_name, float(j.get("vram_gb") or 0.0))
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

#: Backends whose ``transcribe()`` fills ``words`` with per-WORD start/end
#: times. Not every one does, and the difference is invisible from the outside:
#: a backend that ignores ``word_timestamps`` answers 200 with timings missing,
#: so a caller that needs them (burned captions, which must sit on the syllable)
#: silently falls back to estimating from character counts.
#:
#:   * the HF pipeline families return them (``return_timestamps="word"``)
#:   * vosk returns them (``SetWords(True)``)
#:   * faster-whisper, the default when no backend is named, returns them
#:   * whisper-server returns TEXT ONLY -- the whisper.cpp runner's JSON has no
#:     word array -- and the NeMo worker ignores the flag
_WORD_TIMING_BACKENDS = {"", "wav2vec2", "whisper_hf", "crisperwhisper", "vosk",
                         "faster-whisper", "faster_whisper"}


def reports_word_timings(backend) -> bool:
    """Whether a model served by ``backend`` can return per-word timings.

    Conservative about names it does not know: used to PREFER a model, never to
    refuse one, and the caller checks the actual result as well.
    """
    return str(backend or "").strip().lower() in _WORD_TIMING_BACKENDS


def resolve_family(model_name: str, config: Optional[dict]) -> str:
    """Public helper for the dispatcher: which family serves this model."""
    return _family(model_name, config)
