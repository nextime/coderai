#!/usr/bin/env python3
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

"""LongCat-Video generation service — runs INSIDE the isolated Python 3.10 venv.

Started and owned by :mod:`codai.api.longcat_worker`; speaks plain JSON over HTTP on
loopback. It must never import ``codai.*``: this interpreter has torch 2.6, transformers
4.41 and numpy 1.26, and pulling the server package in would drag FastAPI and the main
venv's expectations into it. The contracts both sides share live in
``tools/longcat_common.py``, loaded by path.

``LongCatVideoPipeline`` is in the upstream repo's own ``longcat_video`` package (not on
PyPI), so ``--source`` is prepended to sys.path.

Endpoints:
    GET  /health    → {"ok", "loaded", "model", "stages"}
    GET  /progress  → {"active", "stage", "segment", "segments", "step", "steps"}
    POST /generate  → {"mp4_b64", "num_frames", "fps", "stages", "segments"}
    POST /unload    → {"ok"}  (drops the pipeline, keeps the process)
"""

import argparse
import base64
import importlib.util
import io
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _load_common():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "longcat_common.py")
    spec = importlib.util.spec_from_file_location("longcat_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LC = _load_common()

_lock = threading.RLock()
_state = {
    "pipe": None, "model": "", "source": "", "dtype": "bfloat16",
    "loaded_lora": None,          # which stage's LoRA is currently fused
}
_progress = {"active": False, "stage": "", "segment": 0, "segments": 0,
             "step": 0, "steps": 0}


def log(msg):
    print(f"[longcat] {msg}", flush=True)


# ── loading ───────────────────────────────────────────────────────────────────

def load_pipeline(checkpoint_dir: str, source_dir: str, dtype: str = "bfloat16",
                  offload: str = ""):
    """Build the pipeline from a checkpoint directory. Idempotent."""
    with _lock:
        if _state["pipe"] is not None:
            return _state["pipe"]

        problems = LC.source_problems(source_dir) + LC.checkpoint_problems(checkpoint_dir)
        if problems:
            raise ValueError("; ".join(problems))

        src = os.path.expanduser(source_dir)
        if src not in sys.path:
            sys.path.insert(0, src)

        import torch
        from transformers import AutoTokenizer, UMT5EncoderModel
        from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
        from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
        from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
        from longcat_video.modules.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler)

        td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}.get(str(dtype).lower(), torch.bfloat16)
        root = os.path.expanduser(checkpoint_dir)
        log(f"loading from {root} (dtype={dtype}, offload={offload or 'none'}) …")

        tokenizer = AutoTokenizer.from_pretrained(root, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            root, subfolder="text_encoder", torch_dtype=td)
        vae = AutoencoderKLWan.from_pretrained(root, subfolder="vae", torch_dtype=td)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            root, subfolder="scheduler")
        dit = LongCatVideoTransformer3DModel.from_pretrained(
            root, subfolder="dit", torch_dtype=td)

        pipe = LongCatVideoPipeline(tokenizer=tokenizer, text_encoder=text_encoder,
                                    vae=vae, scheduler=scheduler, dit=dit)
        # Offload is opt-in: the 13.6B DiT wants ~27 GB at bf16 and the reported peak for
        # a full profile is ~41.6 GB, so a smaller card needs it. Not every build of the
        # upstream pipeline exposes the diffusers hooks, hence the guarded calls.
        placed = False
        if offload in ("model", "sequential"):
            fn = getattr(pipe, f"enable_{offload}_cpu_offload", None)
            if callable(fn):
                fn()
                placed = True
                log(f"{offload} CPU offload enabled")
            else:
                log(f"WARNING: this pipeline has no enable_{offload}_cpu_offload(); "
                    f"loading fully on device instead")
        if not placed:
            if torch.cuda.is_available():
                pipe.to("cuda")
            else:
                log("WARNING: no CUDA device visible — this will be extremely slow")
        _state.update(pipe=pipe, model=root, source=src, dtype=dtype, loaded_lora=None)
        log("ready")
        return pipe


def unload(reason: str = ""):
    with _lock:
        if _state["pipe"] is None:
            return
        _state["pipe"] = None
        _state["loaded_lora"] = None
        try:
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        log(f"unloaded{f' ({reason})' if reason else ''}")


def _apply_stage_lora(pipe, stage: str, checkpoint_dir: str):
    """Fuse the LoRA a stage needs, or drop it for the base stage.

    Stages 2 and 3 ARE their LoRAs — the distilled stage is 16 steps because of
    cfg_step_lora, and refinement is 720p because of refinement_lora. Switching stage
    without switching the adapter silently produces the wrong thing."""
    want = {"distill": "distill", "refinement": "refinement"}.get(stage)
    if _state["loaded_lora"] == want:
        return
    unload_fn = getattr(pipe, "unload_lora_weights", None)
    if _state["loaded_lora"] is not None and callable(unload_fn):
        unload_fn()
    if want:
        rel = LC.LORA_FILES[want]
        path = os.path.join(os.path.expanduser(checkpoint_dir), rel)
        load_fn = getattr(pipe, "load_lora_weights", None)
        if not callable(load_fn):
            raise RuntimeError(
                f"stage '{stage}' needs {rel} but this pipeline exposes no "
                f"load_lora_weights()")
        load_fn(path)
        log(f"stage '{stage}': fused {rel}")
    _state["loaded_lora"] = want


# ── generation ────────────────────────────────────────────────────────────────

def _mux(frames, fps: int) -> bytes:
    """Frames (PIL images) → mp4 bytes, via imageio-ffmpeg in the venv."""
    import imageio
    import numpy as np
    buf = io.BytesIO()
    writer = imageio.get_writer(buf, format="mp4", fps=int(fps), codec="libx264",
                               quality=None, ffmpeg_params=["-crf", "18",
                                                            "-pix_fmt", "yuv420p"])
    try:
        for f in frames:
            writer.append_data(np.asarray(f.convert("RGB")))
    finally:
        writer.close()
    return buf.getvalue()


def generate(body: dict) -> dict:
    """One generation. Step 2 scope: text-to-video, one segment, single GPU."""
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")

    stages = LC.resolve_stages(body.get("quality") or "fast", body.get("stage") or "")
    checkpoint = body.get("model") or _state["model"]
    problems = LC.checkpoint_problems(checkpoint, stages)
    if problems:
        raise ValueError("; ".join(problems))

    pipe = load_pipeline(checkpoint, body.get("source") or _state["source"],
                         body.get("dtype") or _state["dtype"],
                         body.get("offload") or "")

    height = int(body.get("height") or LC.DEFAULT_BASE_SIZE[0])
    width = int(body.get("width") or LC.DEFAULT_BASE_SIZE[1])
    num_frames = int(body.get("num_frames") or LC.DEFAULT_NUM_FRAMES)
    fps = int(body.get("fps") or LC.DEFAULT_FPS)
    seed = body.get("seed")

    frames = None
    with _lock:
        _progress.update(active=True, stage="", segment=0, segments=1, step=0, steps=0)
    try:
        for stage in stages:
            params = LC.stage_params(stage, body.get("num_inference_steps"),
                                     body.get("guidance_scale"))
            _apply_stage_lora(pipe, stage, checkpoint)
            with _lock:
                _progress.update(stage=stage, step=0,
                                 steps=params["num_inference_steps"])

            def _cb(_pipe, step_index, _t, kw):
                with _lock:
                    _progress["step"] = int(step_index) + 1
                return kw

            call = {"prompt": prompt, "height": height, "width": width,
                    "num_frames": num_frames, **params}
            if seed is not None:
                import torch
                call["generator"] = torch.Generator(
                    device="cuda" if torch.cuda.is_available() else "cpu"
                ).manual_seed(int(seed))
            # The refinement stage takes the PREVIOUS stage's frames as its input — that
            # is what makes it a refinement and not a second generation.
            if stage == "refinement":
                if frames is None:
                    raise ValueError(
                        "the refinement stage needs frames from an earlier stage; run "
                        "quality='best' or pass stage='base' first")
                call["video"] = frames
            try:
                out = pipe(callback_on_step_end=_cb, **call)
            except TypeError:
                out = pipe(**call)          # a build without the callback hook
            frames = getattr(out, "frames", out)
            if frames and isinstance(frames[0], (list, tuple)):
                frames = frames[0]
            log(f"stage '{stage}': {len(frames)} frames")
    finally:
        with _lock:
            _progress.update(active=False, stage="", step=0)

    if stages[-1] == "refinement":
        fps = int(body.get("fps") or 30)
    mp4 = _mux(frames, fps)
    return {"mp4_b64": base64.b64encode(mp4).decode(), "num_frames": len(frames),
            "fps": fps, "stages": list(stages), "segments": 1,
            "vram_gb": _peak_vram_gb()}


def _peak_vram_gb() -> float:
    """The peak VRAM this process reached, in GB, or 0.0.

    Reported back so the engine can size future reservations from a MEASUREMENT instead
    of an estimate. It has to come from in here: the pipeline loads lazily on the first
    request, and a host-side free-VRAM delta would also be counting whatever else was
    loading or evicting on the card at the same time. torch's own peak counter is
    exactly this process's allocation."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        # reserved, not allocated: the caching allocator's arenas are what the card is
        # actually unable to give to anything else.
        return round(torch.cuda.max_memory_reserved() / (1024 ** 3), 2)
    except Exception:
        return 0.0


# ── HTTP ──────────────────────────────────────────────────────────────────────

def make_handler():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            return

        def _json(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def _read_json(self):
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}") if length else {}

        def do_GET(self):
            if self.path.startswith("/health"):
                self._json({"ok": True, "loaded": _state["pipe"] is not None,
                            "model": _state["model"], "stages": list(LC.STAGES)})
            elif self.path.startswith("/progress"):
                with _lock:
                    self._json(dict(_progress))
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            try:
                if self.path.startswith("/generate"):
                    self._json(generate(self._read_json()))
                elif self.path.startswith("/unload"):
                    unload("requested")
                    self._json({"ok": True})
                else:
                    self._json({"error": "not found"}, 404)
            except ValueError as exc:
                # A bad request (no prompt, missing LoRA, wrong stage order) is not a
                # fault: say so with a 400 so the caller does not retry it.
                self._json({"error": str(exc)}, 400)
            except Exception as exc:
                traceback.print_exc()
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description="LongCat-Video generation service")
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument("--source", default="", help="LongCat-Video repo checkout")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--offload", default="", help="'' | model | sequential")
    ap.add_argument("--attention", default="xformers", help="xformers | flash | sdpa")
    ap.add_argument("--preload", action="store_true",
                    help="load the checkpoint at startup instead of on first request")
    args = ap.parse_args(argv)

    _state.update(model=os.path.expanduser(args.model),
                  source=os.path.expanduser(args.source),
                  dtype=args.dtype)
    # Upstream reads the attention backend from the model config; the env var is what its
    # modules honour, and leaving it unset means FlashAttention-2, which is not installed
    # by default (requirements-longcat.txt installs xformers instead).
    os.environ.setdefault("LONGCAT_ATTENTION", args.attention)

    if args.preload:
        load_pipeline(_state["model"], _state["source"], args.dtype, args.offload)

    server = ThreadingHTTPServer((args.host, args.port), make_handler())
    log(f"serving on http://{args.host}:{args.port} (model={_state['model']})")
    server.serve_forever()


if __name__ == "__main__":
    main()
