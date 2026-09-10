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

"""MiniMax-H3 video+audio generation service (runs in the isolated H3 venv).

H3 needs diffusers >= 0.40, where it exists ONLY as Modular Diffusers blocks —
`MiniMaxH3Blocks` / `MiniMaxH3ModularPipeline`, reached through
`ModularPipeline.from_pretrained(...)`. There is no `DiffusionPipeline` half, so it
cannot go through coderai's `_detect_pipeline_class()` loader; and diffusers 0.40
can't be dropped into the main venv without moving every other pipeline's floor.
Hence the same shape as the pyannote/NeMo workers: an isolated venv, this script as
a local HTTP service, and codai/api/h3_worker.py owning its lifecycle.

Model facts this service encodes (they are checkpoint contracts, not preferences):

  * Fixed 24 fps. `num_frames` is snapped UP to the next `17n + 5` the video VAE can
    decode, and the resulting duration must land between 5 and 15 seconds.
  * Canvas axes are multiples of 32; the released checkpoint's short edge is 768
    with an area cap of 768*1344.
  * The checkpoint is guidance-distilled: there is NO negative prompt and NO
    guidance scale, and every step is a single forward pass.
  * Video and audio are denoised jointly in one packed sequence, so a generation
    always returns a soundtrack — muxed into the returned mp4.
  * Three workflows, each loading its own transformer partition (~61.7 GB each):
      t2va  – prompt only
      fl2va – prompt + first and/or last keyframe   (`transformer/`)
      ref2va– prompt + up to 12 omni references     (`transformer_ref/`)
    Loading WITHOUT a workflow pulls both partitions, so this service always picks
    one from the request and reloads when the workflow changes.

Endpoints:
  GET  /health    → {"ok":true,"loaded":bool,"workflow":str|null,"model":str}
  POST /generate  → {"mp4_b64":..., "fps":24, "num_frames":N, "width":W, "height":H,
                     "sampling_rate":SR, "workflow":...}
  POST /unload    → drop the pipeline and free VRAM (the process stays up)
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The checkpoint contracts (frame/canvas arithmetic, workflow choice, media
# handling) are shared with the in-engine path — see tools/h3_common.py.
try:
    from h3_common import (H3_CANVAS_MULTIPLE, H3_FPS, H3_FRAMES_CHUNK,   # noqa: F401
                           H3_FRAMES_REMAINDER, H3_MAX_SECONDS, H3_MIN_SECONDS,
                           build_references, log, mux, pick_workflow,
                           snap_axis, snap_frames, _decode_to_file, _load_image)
except ImportError:  # started from elsewhere: load the sibling module by path
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "h3_common", os.path.join(os.path.dirname(os.path.abspath(__file__)), "h3_common.py"))
    h3_common = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(h3_common)
    from h3_common import *  # noqa: F401,F403  (module registered above)
    globals().update({k: getattr(h3_common, k) for k in dir(h3_common) if not k.startswith("__")})

_state = {
    "pipe": None,
    "workflow": None,
    "loras": [],          # [(path, weight)] currently fused
    "model": None,
    "dtype": "bfloat16",
    "device_map": "",
    "offload": "",
    "lock": threading.RLock(),
}


# ── model lifecycle ───────────────────────────────────────────────────────────

def _torch_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get((name or "").lower(), torch.bfloat16)


def unload(reason: str = "") -> None:
    with _state["lock"]:
        if _state["pipe"] is None:
            return
        log(f"unloading pipeline{(' (' + reason + ')') if reason else ''}")
        _state["pipe"] = None
        _state["workflow"] = None
        _state["loras"] = []
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def load_pipeline(workflow: str):
    """Load (or reuse) the modular pipeline for one workflow."""
    import torch
    from diffusers import ModularPipeline

    with _state["lock"]:
        if _state["pipe"] is not None and _state["workflow"] == workflow:
            return _state["pipe"]
        if _state["pipe"] is not None:
            # A different workflow means the OTHER transformer partition — there is
            # no way to keep both on a consumer card.
            unload(f"switching workflow {_state['workflow']} -> {workflow}")

        model = _state["model"]
        dtype = _torch_dtype(_state["dtype"])
        log(f"loading {model} (workflow={workflow}, dtype={_state['dtype']}) …")
        t0 = time.time()
        pipe = ModularPipeline.from_pretrained(model, workflow=workflow)

        load_kw = {"dtype": dtype}
        if _state["device_map"]:
            # Per-component device_map, forwarded to each from_pretrained.
            load_kw["device_map"] = _state["device_map"]
        pipe.load_components(workflow=workflow, **load_kw)

        offload = (_state["offload"] or "").lower()
        if offload in ("group", "leaf"):
            _apply_group_offload(pipe, dtype, leaf=(offload == "leaf"))
        elif not _state["device_map"]:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log(f"moving components to {device}")
            pipe.to(device)

        _state["pipe"] = pipe
        _state["workflow"] = workflow
        log(f"ready in {time.time() - t0:.0f}s")
        return pipe


def _apply_group_offload(pipe, dtype, leaf: bool = False) -> None:
    """Stream the big modules between CPU and GPU a group at a time.

    33B of weights don't fit on a consumer card; group offload keeps only the
    executing group resident. Applied per component because a ModularPipeline has
    no pipeline-level offload helper.
    """
    import torch
    if not torch.cuda.is_available():
        log("group offload requested but no CUDA device — staying on CPU")
        return
    kind = "leaf_level" if leaf else "block_level"
    for name in ("transformer", "transformer_ref", "text_encoder", "vae", "audio_vae"):
        comp = getattr(pipe, name, None)
        if comp is None or not hasattr(comp, "enable_group_offload"):
            continue
        try:
            kw = dict(onload_device=torch.device("cuda"),
                      offload_device=torch.device("cpu"),
                      offload_type=kind, use_stream=True)
            if not leaf:
                kw["num_blocks_per_group"] = 1
            comp.enable_group_offload(**kw)
            log(f"group offload ({kind}) enabled on {name}")
        except Exception as exc:
            log(f"group offload failed on {name} ({exc}); moving it to cuda instead")
            try:
                comp.to("cuda")
            except Exception:
                pass


def apply_loras(pipe, loras: list) -> None:
    """Fuse the requested LoRAs (MiniMaxH3LoraLoaderMixin), replacing any others."""
    want = [(str(l.get("path") or ""), float(l.get("weight") or 1.0))
            for l in (loras or []) if l.get("path")]
    if want == _state["loras"]:
        return
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass
    _state["loras"] = []
    if not want:
        return
    names, weights = [], []
    for i, (path, weight) in enumerate(want):
        adapter = f"lora_{i}"
        log(f"loading LoRA {path} @ {weight}")
        pipe.load_lora_weights(path, adapter_name=adapter)
        names.append(adapter)
        weights.append(weight)
    try:
        pipe.set_adapters(names, weights)
    except Exception as exc:
        log(f"set_adapters failed ({exc}) — adapters load at weight 1.0")
    _state["loras"] = want


# ── generation ────────────────────────────────────────────────────────────────

def generate(body: dict) -> dict:
    import torch

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required (every H3 workflow needs one)")

    workflow = body.get("workflow") or pick_workflow(body)
    pipe = load_pipeline(workflow)
    apply_loras(pipe, body.get("loras"))

    num_frames = snap_frames(body.get("num_frames") or H3_MIN_SECONDS * H3_FPS)
    kwargs = {
        "prompt": prompt,
        "num_frames": num_frames,
        "num_inference_steps": int(body.get("num_inference_steps") or 50),
        "output_type": "pil",
    }
    if body.get("width") and body.get("height"):
        kwargs["width"] = snap_axis(body["width"], 1344)
        kwargs["height"] = snap_axis(body["height"], 768)

    seed = body.get("seed")
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs["generator"] = torch.Generator(device=device).manual_seed(int(seed))

    temps: list = []
    try:
        if body.get("image"):
            kwargs["image"] = _load_image(body["image"])
        if body.get("last_image"):
            kwargs["last_image"] = _load_image(body["last_image"])
        if body.get("references"):
            kwargs["references"] = build_references(body["references"], temps)

        log(f"generating: workflow={workflow} frames={num_frames} "
            f"steps={kwargs['num_inference_steps']} "
            f"size={kwargs.get('width', 'auto')}x{kwargs.get('height', 'auto')}")
        t0 = time.time()
        out = pipe(output=["videos", "audio", "sampling_rate"], **kwargs)
        took = time.time() - t0

        videos = out["videos"] if isinstance(out, dict) else out.videos
        audio = (out.get("audio") if isinstance(out, dict) else getattr(out, "audio", None))
        rate = (out.get("sampling_rate") if isinstance(out, dict)
                else getattr(out, "sampling_rate", None))
        frames = videos[0] if (videos and isinstance(videos, (list, tuple))) else videos
        mp4 = mux(frames, audio, rate)
        log(f"done in {took:.0f}s ({len(mp4) / 1e6:.1f} MB)")
        return {
            "mp4_b64": base64.b64encode(mp4).decode(),
            "fps": H3_FPS,
            "num_frames": num_frames,
            "width": kwargs.get("width"),
            "height": kwargs.get("height"),
            "sampling_rate": rate,
            "workflow": workflow,
            "seconds": round(num_frames / H3_FPS, 2),
        }
    finally:
        for path in temps:
            try:
                os.unlink(path)
            except OSError:
                pass


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
                self._json({"ok": True,
                            "loaded": _state["pipe"] is not None,
                            "workflow": _state["workflow"],
                            "model": _state["model"]})
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
                # Bad request (missing prompt, unknown reference type) — not a fault.
                self._json({"error": str(exc)}, 400)
            except Exception as exc:
                traceback.print_exc()
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MiniMax-H3 generation service")
    ap.add_argument("--model", default="MiniMaxAI/MiniMax-H3",
                    help="HF id or local path of the H3 checkpoint")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="",
                    help="Per-component device_map passed to from_pretrained (e.g. 'auto', 'balanced')")
    ap.add_argument("--offload", default="group", choices=["", "group", "leaf"],
                    help="Group-offload the big modules (default: group — 33B does not fit a consumer card)")
    ap.add_argument("--preload", action="store_true",
                    help="Load the t2va workflow at startup instead of on first request")
    args = ap.parse_args(argv)

    _state["model"] = args.model
    _state["dtype"] = args.dtype
    _state["device_map"] = args.device_map
    _state["offload"] = args.offload

    if args.preload:
        try:
            load_pipeline("t2va")
        except Exception as exc:
            log(f"preload failed: {exc}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler())
    log(f"service listening on http://{args.host}:{args.port} (model {args.model})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
