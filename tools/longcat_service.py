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
    "family": "", "variant": "bf16", "use_int8": False, "use_distill": False,
    "cp_split_hw": None, "audio_encoder": None,
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
        family = LC.family_of(root)
        use_int8 = bool(_state.get("use_int8"))
        use_distill_lora = bool(_state.get("use_distill"))

        if use_int8:
            # avatar-1.5 only: a pre-quantised DiT under base_model_int8/, loaded by the
            # repo's own helper rather than from_pretrained.
            from longcat_video.modules.longcat_video_dit import load_quantized_dit
            dit = load_quantized_dit(root, subfolder=LC.INT8_SUBDIR,
                                     cp_split_hw=_state.get("cp_split_hw"))
            log(f"loaded the INT8 DiT from {LC.INT8_SUBDIR}/")
        else:
            kw = {"torch_dtype": td}
            if str(_state.get("variant") or "").lower() == "fp8":
                # Community FP8 weights. The layouts differ between publishers (Kijai's
                # scaled format is not the same as a plain e4m3 dump), so this loads it
                # as the dtype it is and says so rather than pretending to normalise
                # them: if a repo needs its own loader, the load fails here with the
                # real reason instead of producing noise.
                try:
                    kw["torch_dtype"] = torch.float8_e4m3fn
                    log("loading the DiT as float8_e4m3fn (community FP8 weights)")
                except AttributeError:
                    log("WARNING: this torch has no float8_e4m3fn — loading in "
                        f"{dtype} instead")
            dit = LongCatVideoTransformer3DModel.from_pretrained(
                root, subfolder="dit", **kw)

        if family:
            # The avatar families are a DIFFERENT pipeline class with its own methods
            # (generate_at2v / generate_ai2v / generate_avc) and two guidance scales.
            from longcat_video.pipeline_longcat_video_avatar import (
                LongCatVideoAvatarPipeline)
            pipe = LongCatVideoAvatarPipeline(
                tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
                scheduler=scheduler, dit=dit)
            _state["audio_encoder"] = _load_audio_encoder(root, family)
            log(f"avatar family '{family}' "
                f"(audio encoder: {LC.AVATAR_ENCODERS.get(family)})")
        else:
            pipe = LongCatVideoPipeline(tokenizer=tokenizer, text_encoder=text_encoder,
                                        vae=vae, scheduler=scheduler, dit=dit)
        _state["family"] = family

        if use_distill_lora and family:
            # The avatar distilled pass is dmd_lora at 8 steps — NOT the base model's
            # cfg_step_lora at 16, which use_distill=True on the base pipeline applies
            # internally.
            try:
                dit.load_lora(os.path.join(root, LC.DMD_LORA), name="dmd",
                              lora_network_dim=LC.DMD_NETWORK_DIM,
                              lora_network_alpha=LC.DMD_NETWORK_ALPHA)
            except AttributeError:
                # Older/newer builds name it differently; the enable step is what counts.
                pass
            if hasattr(dit, "enable_loras"):
                dit.enable_loras(["dmd"])
                log(f"enabled the DMD LoRA (dim={LC.DMD_NETWORK_DIM}, "
                    f"alpha={LC.DMD_NETWORK_ALPHA})")
            else:
                raise RuntimeError(
                    "use_distill needs the DiT's enable_loras() — this build of "
                    "longcat_video does not expose it")
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
        _state.update(pipe=pipe, model=root, source=src, dtype=dtype)
        log("ready")
        return pipe


def _load_audio_encoder(root: str, family: str):
    """The family's audio encoder: chinese-wav2vec2-base for Avatar, whisper-large-v3 for
    Avatar-1.5. Upstream ships it inside the checkpoint; fall back to the hub id."""
    name = LC.AVATAR_ENCODERS.get(family)
    if not name:
        return None
    local = os.path.join(os.path.expanduser(root), name)
    path = local if os.path.isdir(local) else name
    try:
        from longcat_video.pipeline_longcat_video_avatar import get_audio_encoder
        return get_audio_encoder(path, family)
    except ImportError:
        # The helper moved; the pipeline can still embed audio itself in that case.
        log(f"WARNING: get_audio_encoder is not importable — relying on the "
            f"pipeline's own audio embedding")
        return None


def unload(reason: str = ""):
    with _lock:
        if _state["pipe"] is None:
            return
        _state["pipe"] = None
        _state["audio_encoder"] = None
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

    if task in LC.AVATAR_TASKS:
        return _avatar_pass(pipe, task, dict(ctx, stage=stage), cond=cond)

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


def _audio_embedding(pipe, audio_b64: str, fps: int, family: str):
    """Decode the audio track and embed it the way the avatar pipeline expects.

    Upstream: librosa.load(..., sr=16000) then pipe.get_audio_embedding(speech, fps=…,
    sample_rate=…, model_type=…). The 16 kHz is the encoder's rate, not a preference."""
    import tempfile
    import librosa
    raw = audio_b64.split(",", 1)[1] if str(audio_b64).startswith("data:") else audio_b64
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        fh.write(base64.b64decode(raw))
        path = fh.name
    try:
        speech, sr = librosa.load(path, sr=16000)
        device = 0
        try:
            import torch
            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        except Exception:
            pass
        return pipe.get_audio_embedding(speech, fps=fps, device=device,
                                        sample_rate=sr, model_type=family)
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def _avatar_pass(pipe, task: str, ctx: dict, cond=None):
    """One avatar call. at2v / ai2v to start, avc to continue.

    The avatar pipeline takes TWO guidance scales (text and audio) where the base one
    takes a single guidance_scale, and output_type='both' is what upstream asks for."""
    params = LC.avatar_stage_params(ctx.get("stage") or "base",
                                    ctx.get("num_inference_steps"),
                                    ctx.get("text_guidance_scale"),
                                    ctx.get("audio_guidance_scale"))
    kw = {
        "prompt": ctx["prompt"], "height": ctx["height"], "width": ctx["width"],
        "num_frames": ctx["num_frames"], "audio_emb": ctx["audio_emb"],
        "output_type": "both", **params,
    }
    if ctx.get("negative_prompt"):
        kw["negative_prompt"] = ctx["negative_prompt"]
    if ctx.get("generator") is not None:
        kw["generator"] = ctx["generator"]
    if ctx.get("use_distill"):
        kw["use_distill"] = True

    if cond is not None:
        kw.update(video=cond, num_cond_frames=ctx["num_cond_frames"],
                  use_kv_cache=True,
                  offload_kv_cache=bool(ctx.get("offload_kv_cache")),
                  enhance_hf=False)
        for opt in ("ref_img_index", "mask_frame_range"):
            if ctx.get(opt) is not None:
                kw[opt] = ctx[opt]
        return pipe.generate_avc(**kw)[0]
    if task == "ai2v":
        return pipe.generate_ai2v(image=ctx["image"], **kw)[0]
    return pipe.generate_at2v(**kw)[0]


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
    if task not in ("t2v", "i2v", "vc") + LC.AVATAR_TASKS:
        raise ValueError(f"unknown task {task!r}; expected t2v, i2v, vc, "
                         f"{' or '.join(LC.AVATAR_TASKS)}")

    # Variant and family gating BEFORE loading: INT8 and the DMD distillation exist only
    # for avatar-1.5, and asking for them elsewhere would quietly load something else.
    family = LC.family_of(checkpoint)
    variant = str(body.get("variant") or "bf16").strip().lower()
    use_int8 = bool(body.get("use_int8")) or variant == "int8"
    use_distill_lora = bool(body.get("use_distill"))
    problems = LC.variant_problems(variant, family, use_int8, use_distill_lora)
    if task in LC.AVATAR_TASKS:
        problems += LC.avatar_problems(checkpoint, family, use_int8, use_distill_lora)
    elif family:
        problems.append(
            f"this is the {family} checkpoint, which serves the audio-driven tasks "
            f"({'/'.join(LC.AVATAR_TASKS)}); use the base LongCat-Video weights for "
            f"'{task}'")
    if problems:
        raise ValueError("; ".join(problems))

    _state.update(variant=variant, use_int8=use_int8, use_distill=use_distill_lora)
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
    audio_emb = None
    if task in LC.AVATAR_TASKS:
        if not body.get("audio"):
            raise ValueError(f"task {task!r} needs an audio track")
        audio_emb = _audio_embedding(pipe, body["audio"], fps, family)
    if task == "ai2v":
        if not body.get("image"):
            raise ValueError("task 'ai2v' needs a reference image")
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
        "audio_emb": audio_emb, "use_distill": use_distill_lora,
        "text_guidance_scale": body.get("text_guidance_scale"),
        "audio_guidance_scale": body.get("audio_guidance_scale"),
        "ref_img_index": body.get("ref_img_index"),
        "mask_frame_range": body.get("mask_frame_range"),
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
                            "model": _state["model"], "stages": list(LC.STAGES),
                            "family": _state.get("family") or "",
                            "variant": _state.get("variant") or "bf16"})
            elif self.path.startswith("/progress"):
                with _lock:
                    self._json(dict(_progress))
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            try:
                if self.path.startswith("/generate"):
                    job = self._read_json()
                    if int(_state.get("cp_size") or 1) > 1:
                        # Every rank must enter the same pipeline call, or the
                        # collectives deadlock.
                        cp_broadcast(job)
                    self._json(generate(job))
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


# ── context parallelism ───────────────────────────────────────────────────────
#
# Upstream scales by splitting the DiT's spatial dimensions across GPUs: NCCL is
# initialised BEFORE the model loads, `init_context_parallel(cp_size)` is called, the DiT
# gets `cp_split_hw`, and every rank runs `pipe.to(local_rank)`. Only rank 0 writes the
# output.
#
# That is awkward behind an HTTP service, and the awkwardness is the point: all ranks must
# enter the same pipeline call together or the collectives deadlock. So rank 0 owns the
# socket and BROADCASTS each job to the others, which sit in a loop waiting for one. A
# rank-0-only server that just called generate() would hang on the first collective.

_CP_SHUTDOWN = {"__shutdown__": True}


def cp_init(cp_size: int):
    """Initialise NCCL + context parallelism. Returns (rank, local_rank)."""
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if cp_size <= 1:
        return rank, local_rank
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    # Must happen before the DiT is built, which is why load_pipeline is not called
    # until after this returns.
    from longcat_video.context_parallel.context_parallel_util import (
        init_context_parallel)
    init_context_parallel(cp_size)
    log(f"context parallel: rank {rank}/{dist.get_world_size()} "
        f"(local_rank {local_rank}, cp_size {cp_size})")
    return rank, local_rank


def cp_broadcast(job):
    """Send a job from rank 0 to every other rank, or receive one. Returns the job."""
    import torch.distributed as dist
    box = [job]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def cp_worker_loop():
    """Ranks other than 0: wait for a job, run it, discard the result, repeat.

    The result is discarded deliberately — rank 0's copy is the one that gets muxed and
    returned. What matters here is entering the same collective."""
    while True:
        job = cp_broadcast(None)
        if not job or job.get("__shutdown__"):
            log("context-parallel worker shutting down")
            return
        try:
            generate(job)
        except Exception as exc:
            # A failure on rank 0 raises there too; log and keep the loop alive so one
            # bad request does not strand the group.
            log(f"context-parallel worker error: {type(exc).__name__}: {exc}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="LongCat-Video generation service")
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument("--source", default="", help="LongCat-Video repo checkout")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--offload", default="", help="'' | model | sequential")
    ap.add_argument("--attention", default="xformers", help="xformers | flash | sdpa")
    ap.add_argument("--context-parallel-size", type=int, default=1,
                    help="GPUs to split the DiT across (launch under torchrun)")
    ap.add_argument("--preload", action="store_true",
                    help="load the checkpoint at startup instead of on first request")
    args = ap.parse_args(argv)

    cp_size = max(1, int(args.context_parallel_size))
    _state.update(model=os.path.expanduser(args.model),
                  source=os.path.expanduser(args.source),
                  dtype=args.dtype, cp_size=cp_size)
    # Upstream reads the attention backend from the model config; the env var is what its
    # modules honour, and leaving it unset means FlashAttention-2, which is not installed
    # by default (requirements-longcat.txt installs xformers instead).
    os.environ.setdefault("LONGCAT_ATTENTION", args.attention)

    rank, _local = (0, 0)
    if cp_size > 1:
        # The repo checkout has to be importable before init_context_parallel.
        src = _state["source"]
        if src and src not in sys.path:
            sys.path.insert(0, src)
        rank, _local = cp_init(cp_size)
        # Splitting height and width across the group is what cp_size buys; the DiT
        # takes it at construction.
        _state["cp_split_hw"] = cp_size

    if args.preload:
        load_pipeline(_state["model"], _state["source"], args.dtype, args.offload)

    if rank != 0:
        # No socket on these ranks: one job, one broadcast, every rank in the same call.
        cp_worker_loop()
        return

    server = ThreadingHTTPServer((args.host, args.port), make_handler())
    log(f"serving on http://{args.host}:{args.port} (model={_state['model']}"
        + (f", cp_size={cp_size}" if cp_size > 1 else "") + ")")
    try:
        server.serve_forever()
    finally:
        if cp_size > 1:
            try:
                cp_broadcast(dict(_CP_SHUTDOWN))
            except Exception:
                pass


if __name__ == "__main__":
    main()
