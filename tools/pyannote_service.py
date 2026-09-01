#!/usr/bin/env python3
"""Standalone pyannote.audio speaker-diarization HTTP microservice — own venv.

pyannote.audio pins torch/lightning/speechbrain that conflict with coderai's
transformers 5.x, so it runs here behind a stdlib HTTP shim, managed by
``codai.api.pyannote_worker``.

Endpoints:
    GET  /health    -> {"ok": true, "model": ...}
    POST /diarize   -> {"segments":[{start,end,speaker}], "num_speakers": N}
                       body: raw audio bytes;
                       query: ?num_speakers=<int>&min_speakers=<int>&max_speakers=<int>

Audio is transcoded to 16 kHz mono WAV via ffmpeg before inference. The gated
model needs HF_TOKEN in the environment; set CODERAI_DIARIZATION_MODEL to an
ungated mirror to avoid the token.
"""

import argparse
import json
import os
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs


def _to_wav_16k_mono(raw: bytes) -> str:
    fd, src = tempfile.mkstemp(suffix=".input")
    os.write(fd, raw)
    os.close(fd)
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        p = subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1",
             "-c:a", "pcm_s16le", "-f", "wav", out],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            raise RuntimeError("ffmpeg conversion failed: " +
                               p.stderr.decode("utf-8", "replace")[-300:])
        return out
    finally:
        try:
            os.unlink(src)
        except OSError:
            pass


def _crop_wav(src: str, start: float, end: float) -> str:
    """Write a temp WAV of src[start:end] seconds (for per-speaker embedding)."""
    import soundfile as sf
    data, sr = sf.read(src)
    a = max(0, int(start * sr))
    b = min(len(data), int(end * sr))
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(out, data[a:b] if b > a else data, sr)
    return out


class _Engine:
    def __init__(self, model_name: str):
        import torch
        # torch 2.6 flipped torch.load's `weights_only` default to True, which
        # rejects pyannote's own checkpoints (they pickle a TorchVersion global) →
        # UnpicklingError. We load our own trusted pyannote weights in an isolated
        # venv, so force weights_only=False for all loads before importing pyannote.
        _orig_load = torch.load
        def _load(*a, **k):
            k["weights_only"] = False   # force (pyannote passes True explicitly)
            return _orig_load(*a, **k)
        torch.load = _load
        from pyannote.audio import Pipeline
        self.model_name = model_name
        token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
                 or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        # pyannote.audio 3.1+ renamed the auth kwarg to `token`; older releases use
        # `use_auth_token`. Try the new name, fall back to the old.
        try:
            self._pipe = Pipeline.from_pretrained(model_name, token=token)
        except TypeError:
            self._pipe = Pipeline.from_pretrained(model_name, use_auth_token=token)
        if self._pipe is None:
            raise RuntimeError(
                f"Pipeline.from_pretrained('{model_name}') returned None — the model is "
                "gated/unavailable. Accept its conditions on HF and set HF_TOKEN, or set "
                "CODERAI_DIARIZATION_MODEL to an ungated mirror.")
        # Device: pyannote.audio's Pipeline.to() is unreliable with very new torch
        # (2.13/cu130) — it leaves the segmentation input on CPU while weights move
        # to CUDA ("Expected all tensors to be on the same device"). So default to
        # CPU (always correct) and only attempt CUDA when explicitly opted in via
        # CODERAI_DIARIZATION_DEVICE=cuda (use once torch is pinned to a compatible
        # version). The GPU→CPU retry in diarize() is the safety net either way.
        # Run on GPU by default (the venv pins a pyannote-compatible torch — see
        # requirements-pyannote.txt). CODERAI_DIARIZATION_DEVICE=cpu forces CPU;
        # the GPU→CPU retry in diarize() is the safety net if a device mismatch
        # slips through on some torch build.
        self._device = "cpu"
        want = (os.environ.get("CODERAI_DIARIZATION_DEVICE") or "cuda").strip().lower()
        try:
            if want == "cuda" and torch.cuda.is_available():
                self._pipe.to(torch.device("cuda"))
                self._device = "cuda"
        except Exception:
            pass

    def diarize(self, wav_path: str, num_speakers=None,
                min_speakers=None, max_speakers=None, embed_backend=None) -> dict:
        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = int(num_speakers)
        else:
            if min_speakers:
                kwargs["min_speakers"] = int(min_speakers)
            if max_speakers:
                kwargs["max_speakers"] = int(max_speakers)
        # Run on GPU; if this pyannote.audio + torch combo doesn't move the
        # segmentation input to CUDA ("Expected all tensors to be on the same
        # device"), fall back to CPU and retry so diarization still works.
        import torch as _torch
        try:
            ann = self._pipe(wav_path, **kwargs)
        except RuntimeError as e:
            if "same device" not in str(e) or self._device == "cpu":
                raise
            print("[pyannote] GPU device mismatch — falling back to CPU (pin torch "
                  "to a pyannote-compatible version to keep it on the 3090).", flush=True)
            self._pipe.to(_torch.device("cpu"))
            self._device = "cpu"
            ann = self._pipe(wav_path, **kwargs)
        segments = []
        speakers = set()
        for turn, _, speaker in ann.itertracks(yield_label=True):
            speakers.add(speaker)
            segments.append({"start": float(turn.start), "end": float(turn.end),
                             "speaker": str(speaker)})
        segments.sort(key=lambda s: s["start"])
        result = {"segments": segments, "num_speakers": len(speakers)}
        # For speaker identification: embed one representative slice (the longest
        # turn) per diarized speaker, so the caller can match each against enrolled
        # voiceprints. Returned as {SPEAKER_XX: [vector]}.
        if embed_backend:
            spk_emb = {}
            longest = {}
            for s in segments:
                dur = s["end"] - s["start"]
                if dur > longest.get(s["speaker"], (0.0, None))[0]:
                    longest[s["speaker"]] = (dur, s)
            for spk, (_, s) in longest.items():
                crop = None
                try:
                    crop = _crop_wav(wav_path, s["start"], s["end"])
                    spk_emb[spk] = _speaker_embed(crop, backend=embed_backend)["embedding"]
                except Exception as e:
                    print(f"[pyannote] per-speaker embed failed for {spk}: {e}", flush=True)
                finally:
                    if crop:
                        try:
                            os.unlink(crop)
                        except OSError:
                            pass
            result["speaker_embeddings"] = spk_emb
        return result


ENGINE = None

# Speaker-embedding backends, loaded lazily and cached by "backend:model".
_EMB_CACHE = {}


def _hf_token():
    return (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def _speaker_embed(wav_path: str, backend: str, model: str = None,
                   window: float = None, step: float = None) -> dict:
    """Return a fixed-dim speaker embedding for the audio.

    Backends: ``ecapa`` (speechbrain ECAPA-TDNN, ungated), ``pyannote`` /
    ``wespeaker`` (pyannote.audio embedding models — may be gated, use HF_TOKEN).

    If ``window`` (seconds) is given, slide a window of that length with hop
    ``step`` (default = window) over the audio and return one embedding per
    window as ``windows``: [{start, end, embedding}], instead of a single
    whole-file ``embedding``."""
    import torch
    backend = (backend or "ecapa").lower()
    key = f"{backend}:{model or 'default'}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fn = _EMB_CACHE.get(key)
    if fn is None:
        if backend in ("ecapa", "speechbrain"):
            from speechbrain.inference.speaker import EncoderClassifier
            src = model or "speechbrain/spkrec-ecapa-voxceleb"
            clf = EncoderClassifier.from_hparams(source=src, run_opts={"device": dev})

            def fn(path, _clf=clf):
                import torchaudio
                sig, _sr = torchaudio.load(path)
                out = _clf.encode_batch(sig.to(dev))
                return out.squeeze().detach().cpu().float().tolist()
        elif backend in ("pyannote", "wespeaker", "resnet"):
            from pyannote.audio import Model, Inference
            src = model or ("pyannote/wespeaker-voxceleb-resnet34-LM"
                            if backend != "pyannote" else "pyannote/embedding")
            m = Model.from_pretrained(src, token=_hf_token())
            if m is None:
                raise RuntimeError(
                    f"embedding model '{src}' is gated/unavailable — accept its "
                    f"conditions on HF (https://huggingface.co/{src}) with the token's "
                    "account, or use backend=wespeaker / backend=ecapa (ungated).")
            inf = Inference(m, window="whole")
            try:
                inf.to(torch.device(dev))
            except Exception:
                pass

            def fn(path, _inf=inf):
                import numpy as np
                v = _inf(path)
                return np.asarray(v).squeeze().astype(float).tolist()
        else:
            raise ValueError(f"unknown speaker-embedding backend: {backend}")
        _EMB_CACHE[key] = fn
    _model_out = model or key.split(":", 1)[1]
    if window and float(window) > 0:
        import soundfile as sf
        win = float(window)
        hop = float(step) if step and float(step) > 0 else win
        dur = float(sf.info(wav_path).duration)
        windows = []
        t = 0.0
        while t < dur:
            end = min(t + win, dur)
            crop = _crop_wav(wav_path, t, end)
            try:
                emb = fn(crop)
                windows.append({"start": round(t, 3), "end": round(end, 3),
                                "embedding": emb})
            finally:
                try:
                    os.unlink(crop)
                except OSError:
                    pass
            if end >= dur:
                break
            t += hop
        dim = len(windows[0]["embedding"]) if windows else 0
        return {"windows": windows, "count": len(windows), "dim": dim,
                "window": win, "step": hop, "backend": backend, "model": _model_out}
    emb = fn(wav_path)
    return {"embedding": emb, "dim": len(emb), "backend": backend,
            "model": _model_out}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._send(200, json.dumps({"ok": True, "model": ENGINE.model_name}).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/diarize", "/embed"):
            self._send(404, b'{"error":"not found"}')
            return
        wav = None
        try:
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b""
            if not raw:
                self._send(400, b'{"error":"empty audio"}')
                return
            q = parse_qs(parsed.query)
            def _q(k):
                v = q.get(k)
                return v[0] if v else None
            wav = _to_wav_16k_mono(raw)
            if parsed.path == "/embed":
                result = _speaker_embed(wav, backend=_q("backend"), model=_q("model"),
                                        window=_q("window"), step=_q("step"))
                self._send(200, json.dumps(result).encode())
                return
            result = ENGINE.diarize(wav, num_speakers=_q("num_speakers"),
                                    min_speakers=_q("min_speakers"),
                                    max_speakers=_q("max_speakers"),
                                    embed_backend=_q("embed"))
            self._send(200, json.dumps(result).encode())
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, json.dumps({"error": str(e)}).encode())
        finally:
            if wav:
                try:
                    os.unlink(wav)
                except OSError:
                    pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Standalone pyannote diarization HTTP service")
    ap.add_argument("--model", default=os.environ.get("CODERAI_DIARIZATION_MODEL")
                    or "pyannote/speaker-diarization-3.1")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8125)
    args = ap.parse_args(argv)

    global ENGINE
    print(f"Loading pyannote pipeline {args.model} …")
    ENGINE = _Engine(args.model)
    print(f"Ready: {args.model} — serving on http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
