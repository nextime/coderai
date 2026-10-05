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
# Set by POST /yield: stop at the next SEGMENT boundary and return what has been
# generated, so another model can have the card without the request being lost.
_yield_flag = threading.Event()


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


def torch_gc():
    """What upstream calls between segments. The allocator holds onto a segment's
    activations otherwise, and the next one starts that much closer to the ceiling."""
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _frames_from_output(output):
    """The pipeline returns ``method(...)[0]`` — an array of frames normalised to [0, 1],
    NOT an object with a ``.frames`` attribute. Upstream converts with
    ``(tensor * 255).clamp(0, 255).to(uint8)``; we go on to PIL because the refinement
    stage takes PIL frames as its input and because that is what the muxer wants."""
    import numpy as np
    from PIL import Image
    arr = np.asarray(output)
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return [Image.fromarray(f).convert("RGB") for f in arr]


def _resolution_name(height: int, width: int) -> str:
    """generate_i2v and generate_vc take a resolution NAME, where generate_t2v takes
    height/width. Mapping it here keeps that asymmetry out of the request payload."""
    return "720p" if max(int(height or 0), int(width or 0)) > 1000 else "480p"


def _one_pass(pipe, task: str, stage: str, ctx: dict, cond=None, log=print):
    """One pipeline call: the first segment of a task, or a continuation of it.

    ``cond`` is the previous segment's frames (PIL). When present the call is always
    generate_vc, whatever the task started as — that is how a continuation continues."""
    import torch

    params = LC.stage_params(stage, ctx.get("num_inference_steps"),
                             ctx.get("guidance_scale"))
    common = {
        "prompt": ctx["prompt"],
        "num_frames": ctx["num_frames"],
        "num_inference_steps": params["num_inference_steps"],
        "guidance_scale": params["guidance_scale"],
    }
    if ctx.get("generator") is not None:
        common["generator"] = ctx["generator"]
    # The distilled stage is this FLAG, not a LoRA we fuse ourselves — upstream passes
    # use_distill=True to the same method and the pipeline applies cfg_step_lora.
    if stage == "distill":
        common["use_distill"] = True
    elif ctx.get("negative_prompt"):
        # The distilled demos pass no negative prompt (guidance is 1.0, so it does
        # nothing); the 50-step passes do.
        common["negative_prompt"] = ctx["negative_prompt"]

    if cond is not None:
        kw = dict(common)
        kw.update(video=cond, num_cond_frames=ctx["num_cond_frames"],
                  resolution=ctx["resolution"], use_kv_cache=True,
                  offload_kv_cache=bool(ctx.get("offload_kv_cache")))
        if stage == "distill":
            kw["enhance_hf"] = False
        return pipe.generate_vc(**kw)[0]

    if task == "i2v":
        return pipe.generate_i2v(image=ctx["image"], resolution=ctx["resolution"],
                                 **common)[0]
    if task == "vc":
        kw = dict(common)
        kw.update(video=ctx["cond_video"], num_cond_frames=ctx["num_cond_frames"],
                  resolution=ctx["resolution"], use_kv_cache=True,
                  offload_kv_cache=bool(ctx.get("offload_kv_cache")))
        if stage == "distill":
            kw["enhance_hf"] = False
        return pipe.generate_vc(**kw)[0]
    return pipe.generate_t2v(height=ctx["height"], width=ctx["width"], **common)[0]


def _run_segments(pipe, task: str, stage: str, ctx: dict, log=print):
    """Generate ``num_segments`` segments, feeding each one's tail into the next.

    This is upstream's loop: a call emits ``num_frames`` but its first
    ``num_cond_frames`` re-render the previous segment's tail, so only the remainder is
    new — ``all_generated_frames.extend(new_video[num_cond_frames:])`` — and the fresh
    frames become the next call's conditioning. Native pretrained continuation is why
    this does not drift in colour or quality the way chained frame-tail conditioning on
    other models does.

    Between segments is also the one safe point to hand the GPU over, so the yield flag
    is checked here: a release that arrives mid-request costs at most one segment instead
    of the whole generation."""
    segments = max(1, int(ctx.get("num_segments") or 1))
    k = int(ctx["num_cond_frames"])
    acc: list = []
    cur = ctx.get("cond_video")
    first_is_continuation = cur is not None
    yielded = 0

    for index in range(segments):
        with _lock:
            _progress.update(segment=index + 1, segments=segments, step=0,
                             steps=LC.stage_params(
                                 stage, ctx.get("num_inference_steps"),
                                 ctx.get("guidance_scale"))["num_inference_steps"])
        out = _one_pass(pipe, task, stage, ctx,
                        cond=cur if (index or first_is_continuation) else None, log=log)
        new = _frames_from_output(out)
        if index == 0 and not first_is_continuation:
            acc.extend(new)
        else:
            acc.extend(new[k:])
        cur = new
        log(f"stage '{stage}' segment {index + 1}/{segments}: "
            f"{len(acc)} frames so far")
        torch_gc()
        if index + 1 < segments and _yield_flag.is_set():
            # Another model is waiting for the card. Stop at this boundary and return
            # what exists: a shorter video plus a warning beats losing the request, and
            # beats blocking the swap for the remaining segments.
            yielded = segments - (index + 1)
            log(f"yielding the GPU after segment {index + 1}/{segments} "
                f"({yielded} not generated)")
            break
    return acc, yielded


def _mux(frames, fps: int) -> bytes:
    """Frames (PIL) → mp4 bytes. Upstream writes with libx264 at crf 18 for the 480p
    stages and crf 10 for refinement; this keeps 18 and lets the engine re-encode if a
    model entry sets output_crf."""
    import imageio
    import numpy as np
    buf = io.BytesIO()
    writer = imageio.get_writer(buf, format="mp4", fps=int(fps), codec="libx264",
                                ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"])
    try:
        for f in frames:
            writer.append_data(np.asarray(f.convert("RGB")))
    finally:
        writer.close()
    return buf.getvalue()


def _decode_image(data: str):
    from PIL import Image
    raw = data.split(",", 1)[1] if str(data).startswith("data:") else data
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


def _encode_image(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _peak_vram_gb() -> float:
    """The peak VRAM this process reached, in GB, or 0.0.

    Reported back so the engine sizes future reservations from a MEASUREMENT instead of
    an estimate. It has to come from in here: the pipeline loads lazily on the first
    request, so a host-side free-VRAM delta would also be counting whatever else was
    loading or evicting on the card. torch's own counter is this process's allocation —
    reserved rather than allocated, because the caching allocator's arenas are what the
    card is actually unable to give to anything else."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return round(torch.cuda.max_memory_reserved() / (1024 ** 3), 2)
    except Exception:
        return 0.0


def generate(body: dict) -> dict:
    """One request: text-to-video, image-to-video or continuation, over N segments.

    Tasks map to upstream's entry points — generate_t2v (height/width), generate_i2v
    (image + resolution name) and generate_vc (conditioning video) — plus generate_refine
    for the 720p stage, which takes the earlier stage's frames as ``stage1_video``.
    """
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")

    stages = LC.resolve_stages(body.get("quality") or "fast", body.get("stage") or "")
    checkpoint = body.get("model") or _state["model"]
    problems = LC.checkpoint_problems(checkpoint, stages)
    if problems:
        raise ValueError("; ".join(problems))

    task = (body.get("task") or "t2v").strip().lower()
    if task not in ("t2v", "i2v", "vc"):
        raise ValueError(f"unknown task {task!r}; expected t2v, i2v or vc")

    pipe = load_pipeline(checkpoint, body.get("source") or _state["source"],
                         body.get("dtype") or _state["dtype"],
                         body.get("offload") or "")

    height = int(body.get("height") or LC.DEFAULT_BASE_SIZE[0])
    width = int(body.get("width") or LC.DEFAULT_BASE_SIZE[1])
    num_frames = int(body.get("num_frames") or LC.DEFAULT_NUM_FRAMES)
    cond_frames = int(body.get("num_cond_frames") or LC.DEFAULT_COND_FRAMES)
    if cond_frames >= num_frames:
        raise ValueError(f"num_cond_frames ({cond_frames}) must be smaller than "
                         f"num_frames ({num_frames}) — a segment would add nothing")
    fps = int(body.get("fps") or LC.DEFAULT_FPS)

    segments = body.get("num_segments")
    if not segments and body.get("total_frames"):
        segments = LC.segments_for(int(body["total_frames"]), num_frames, cond_frames)
    segments = max(1, int(segments or 1))

    image = None
    if task == "i2v":
        if not body.get("image"):
            raise ValueError("task 'i2v' needs an image")
        image = _decode_image(body["image"])
    cond_video = None
    if task == "vc":
        if not body.get("cond_frames_b64"):
            raise ValueError("task 'vc' needs cond_frames_b64 (the previous tail)")
        cond_video = [_decode_image(x) for x in body["cond_frames_b64"]]
        if len(cond_video) < cond_frames:
            raise ValueError(f"task 'vc' needs at least {cond_frames} conditioning "
                             f"frames, got {len(cond_video)}")

    generator = None
    if body.get("seed") is not None:
        import torch
        generator = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        ).manual_seed(int(body["seed"]))

    ctx = {
        "prompt": prompt, "negative_prompt": body.get("negative_prompt") or "",
        "height": height, "width": width, "num_frames": num_frames,
        "num_cond_frames": cond_frames, "num_segments": segments,
        "resolution": _resolution_name(height, width),
        "num_inference_steps": body.get("num_inference_steps"),
        "guidance_scale": body.get("guidance_scale"),
        "offload_kv_cache": body.get("offload_kv_cache"),
        "image": image, "cond_video": cond_video, "generator": generator,
    }

    _yield_flag.clear()
    frames = None
    yielded = 0
    with _lock:
        _progress.update(active=True, stage="", segment=0, segments=segments,
                         step=0, steps=0)
    try:
        for stage in stages:
            with _lock:
                _progress["stage"] = stage
            if stage == "refinement":
                if frames is None:
                    raise ValueError(
                        "the refinement stage needs frames from an earlier stage — run "
                        "quality='best', or pass stage='base' first and feed its frames "
                        "back with stage='refinement'")
                params = LC.stage_params(stage, body.get("num_inference_steps"),
                                         body.get("guidance_scale"))
                kw = {"prompt": prompt, "stage1_video": frames,
                      "num_inference_steps": params["num_inference_steps"]}
                if generator is not None:
                    kw["generator"] = generator
                if body.get("spatial_refine_only") is not None:
                    kw["spatial_refine_only"] = bool(body["spatial_refine_only"])
                if image is not None:
                    kw["image"] = image
                    kw["num_cond_frames"] = 1
                log(f"stage 'refinement': refining {len(frames)} frames")
                frames = _frames_from_output(pipe.generate_refine(**kw)[0])
            else:
                frames, yielded = _run_segments(pipe, task, stage, ctx, log=log)
            log(f"stage '{stage}': {len(frames)} frames")
    finally:
        _yield_flag.clear()
        with _lock:
            _progress.update(active=False, stage="", step=0)

    if stages[-1] == "refinement":
        fps = int(body.get("fps") or 30)
    mp4 = _mux(frames, fps)
    out = {"mp4_b64": base64.b64encode(mp4).decode(), "num_frames": len(frames),
           "fps": fps, "stages": list(stages), "task": task,
           "segments": max(1, segments - yielded), "vram_gb": _peak_vram_gb()}
    if yielded:
        out["warning"] = (f"stopped after {segments - yielded} of {segments} segments to "
                          f"hand the GPU to another model; the video is shorter than "
                          f"requested")
    # The tail a caller needs to continue from here, so chaining requests does not mean
    # re-deriving which frames overlap.
    out["tail_b64"] = [_encode_image(f) for f in frames[-cond_frames:]]
    return out


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
                elif self.path.startswith("/yield"):
                    # Stop at the next segment boundary. The in-flight request still
                    # returns — shorter, with a warning — so a swap costs one segment
                    # rather than the whole generation.
                    _yield_flag.set()
                    self._json({"ok": True})
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
