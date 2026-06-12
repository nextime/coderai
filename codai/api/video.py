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

"""
Video generation and manipulation endpoints for the codai API.

Endpoints:
  POST /v1/video/generations   – t2v | i2v | v2v | ti2v | interp
  POST /v1/video/upscale       – video super-resolution
  POST /v1/video/subtitle      – subtitle generation / burn-in
  POST /v1/video/interpolate   – frame interpolation (increase FPS)
  POST /v1/video/dub           – translation + TTS dubbing
"""

import asyncio
import base64
import io
import os
import subprocess
import tempfile
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request

from codai.models.manager import multi_model_manager
from codai.pydantic.videorequest import (
    VideoGenerationRequest, VideoGenerationResponse,
    VideoUpscaleRequest, VideoSubtitleRequest,
    VideoInterpolateRequest, VideoDubRequest,
    CharacterDialogLine,
)
from codai.api.images import _disable_safety_checker
from codai.tasks import task_registry, TaskCancelled

router = APIRouter()

global_args = None
global_file_path = None

# =============================================================================
# Video generation progress tracking
# =============================================================================
_vid_progress: dict = {
    "current": 0, "total": 0, "active": False,
    "started_at": 0.0, "it_per_s": 0.0,
    "phase": "idle", "model": "",
}

def _vid_progress_loading(model_name: str = ""):
    _vid_progress["phase"] = "loading"
    _vid_progress["active"] = True
    _vid_progress["current"] = 0
    _vid_progress["total"] = 0
    _vid_progress["it_per_s"] = 0.0
    _vid_progress["started_at"] = time.monotonic()
    _vid_progress["model"] = model_name or ""

def _vid_progress_reset(total: int):
    _vid_progress["current"] = 0
    _vid_progress["total"] = total
    _vid_progress["active"] = True
    _vid_progress["phase"] = "generating"
    _vid_progress["started_at"] = time.monotonic()
    _vid_progress["it_per_s"] = 0.0

def _vid_progress_done():
    _vid_progress["current"] = _vid_progress["total"]
    _vid_progress["active"] = False
    _vid_progress["phase"] = "idle"

def _vid_progress_step(step: int):
    _vid_progress["current"] = step
    elapsed = time.monotonic() - _vid_progress["started_at"]
    if elapsed > 0 and step > 0:
        _vid_progress["it_per_s"] = round(step / elapsed, 2)


def set_global_args(args):
    global global_args
    global_args = args


def set_global_file_path(path):
    global global_file_path
    global_file_path = path


# =============================================================================
# Shared helpers
# =============================================================================

def _derive_device() -> str:
    if global_args:
        for attr in ('image_vulkan_device', 'vulkan_device'):
            d = getattr(global_args, attr, None)
            if d is not None:
                return f"cuda:{d}"
    return "cuda:0"


def _decode_b64_or_url(data: str) -> bytes:
    if not data:
        return b''
    if data.startswith("data:"):
        _, enc = data.split(",", 1)
        return base64.b64decode(enc)
    if data.startswith("http://") or data.startswith("https://"):
        import urllib.request
        with urllib.request.urlopen(data, timeout=60) as r:
            return r.read()
    return base64.b64decode(data)


def _pil_from_b64(data: str):
    from PIL import Image as PILImage
    return PILImage.open(io.BytesIO(_decode_b64_or_url(data))).convert("RGB")


def _build_url(filename: str, http_request) -> str:
    from codai.api.urlutils import build_file_url
    return build_file_url(filename, http_request)


def _save_file(data: bytes, ext: str, http_request) -> dict:
    filename = f"{uuid.uuid4().hex}.{ext}"
    if global_file_path:
        os.makedirs(global_file_path, exist_ok=True)
        out_path = os.path.join(global_file_path, filename)
        with open(out_path, 'wb') as f:
            f.write(data)
        return {"url": _build_url(filename, http_request)}
    else:
        return {f"b64_{ext}": base64.b64encode(data).decode()}


def _encode_mp4_pyav(frames, fps: int, crf: int) -> bytes:
    """Encode RGB frames to H.264 MP4 via PyAV with an explicit CRF (quality).

    CRF is libx264's quality knob: lower = higher quality / bigger file
    (0=lossless, ~18 visually lossless, 23 default, 28 small). PyAV gives us
    direct codec-option control that imageio's mimsave does not expose.
    """
    import av, numpy as np
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        path = tmp.name
    container = av.open(path, mode='w')
    try:
        h, w = frames[0].shape[:2]
        stream = container.add_stream('libx264', rate=int(fps) or 1)
        stream.width = int(w)
        stream.height = int(h)
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': str(int(crf))}
        for f in frames:
            arr = f if f.dtype == np.uint8 else np.clip(f, 0, 255).astype(np.uint8)
            if arr.ndim == 2:  # grayscale → rgb
                arr = np.stack([arr] * 3, axis=-1)
            vframe = av.VideoFrame.from_ndarray(arr[..., :3], format='rgb24')
            for pkt in stream.encode(vframe):
                container.mux(pkt)
        for pkt in stream.encode():  # flush
            container.mux(pkt)
    finally:
        container.close()
    with open(path, 'rb') as fh:
        data = fh.read()
    os.unlink(path)
    return data


def _enc_dbg(msg: str) -> None:
    """Verbose encode-path logging (only when --debug is active)."""
    try:
        from codai.api.state import get_global_debug
        if get_global_debug():
            print(f"  [mp4-encode] {msg}")
    except Exception:
        pass


def _frames_to_mp4(frames, fps: int, crf: Optional[int] = None) -> bytes:
    import imageio, numpy as np
    frames = [np.asarray(f) for f in frames]

    # Diffusion pipelines emit float32 frames in [0, 1]. Both PyAV and imageio
    # want uint8 [0, 255]; if we hand them floats, imageio re-converts and logs
    # "Lossy conversion from float32 to uint8" ONCE PER FRAME, flooding the log.
    # Convert ourselves (correct + quiet). Some pipelines already give uint8 or
    # float in [0, 255]; detect the range from the global max.
    if frames:
        f0 = frames[0]
        if np.issubdtype(f0.dtype, np.floating):
            try:
                gmax = max((float(f.max()) for f in frames if f.size), default=1.0)
            except ValueError:
                gmax = 1.0
            scale = 255.0 if gmax <= 1.0 + 1e-3 else 1.0
            frames = [np.clip(f * scale, 0, 255).round().astype(np.uint8) for f in frames]
        elif f0.dtype != np.uint8:
            frames = [np.clip(f, 0, 255).astype(np.uint8) for f in frames]

    # Report whether the optional imageio-ffmpeg plugin is available — it lets the
    # imageio path honour quality kwargs (the default PyAV plugin does not).
    try:
        import imageio_ffmpeg as _iioff
        _enc_dbg(f"imageio-ffmpeg available (v{getattr(_iioff, '__version__', '?')})")
        _have_iioff = True
    except Exception as e:
        _enc_dbg(f"imageio-ffmpeg NOT installed ({type(e).__name__}); "
                 f"PyAV is the only imageio mp4 backend")
        _have_iioff = False

    # When a per-model CRF (quality) is configured, encode via PyAV directly so
    # the quality knob is actually applied (mimsave can't pass CRF either way).
    if crf is not None:
        try:
            _enc_dbg(f"encoding via PyAV with crf={crf}")
            data = _encode_mp4_pyav(frames, fps, crf)
            _enc_dbg(f"PyAV crf={crf} encode OK ({len(data)} bytes)")
            return data
        except Exception as e:
            # ALWAYS surface this (not just in debug).  Distinguish "av not
            # installed" from other errors.  We still honour CRF below via the
            # imageio FFMPEG plugin's output_params.
            import traceback
            _kind = ("PyAV (av) not installed" if isinstance(e, ImportError)
                     else type(e).__name__)
            print(f"  [mp4-encode] PyAV crf encode FAILED — {_kind}: {e} "
                  f"— falling back to imageio-ffmpeg")
            _enc_dbg("PyAV traceback:\n" + traceback.format_exc())

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp_path = tmp.name
    # imageio's MP4 backend differs by version/install.  When a CRF is configured
    # we use the FFMPEG plugin, which honours it via output_params; otherwise we
    # fall back through codec/quality kwargs (the PyAV plugin only takes codec).
    if crf is not None:
        _crf_params = ['-crf', str(int(crf))]
        _attempts = [
            # Force the FFMPEG plugin so output_params (the CRF) is applied.
            dict(format='FFMPEG', fps=fps, codec='libx264', output_params=_crf_params),
            dict(fps=fps, codec='libx264', output_params=_crf_params),
            # Last resort — produce a video even if CRF couldn't be applied.
            dict(fps=fps, codec='libx264', quality=8),
            dict(fps=fps, codec='libx264'),
            dict(fps=fps),
        ]
    else:
        _attempts = [
            dict(fps=fps, codec='libx264', quality=8),  # imageio-ffmpeg
            dict(fps=fps, codec='libx264'),             # PyAV (codec only)
            dict(fps=fps),                              # minimal
        ]
    last_err = None
    used = None
    for kw in _attempts:
        try:
            imageio.mimsave(tmp_path, frames, **kw)
            used = kw
            last_err = None
            break
        except Exception as e:  # TypeError (kwarg rejected) or plugin/runtime error
            _enc_dbg(f"imageio.mimsave rejected/failed kwargs={kw}: {type(e).__name__}: {e}")
            last_err = e
            continue
    if last_err is not None:
        print(f"  [mp4-encode] imageio.mimsave failed on all kwarg sets: {last_err}")
        raise last_err
    if crf is not None and 'output_params' not in used:
        print(f"  [mp4-encode] WARNING: CRF {crf} could NOT be applied via imageio "
              f"(used kwargs={used}); output uses default quality")
    else:
        _enc_dbg(f"imageio.mimsave OK with kwargs={used}")
    with open(tmp_path, 'rb') as f:
        data = f.read()
    os.unlink(tmp_path)
    return data


def _video_bytes_to_path(video_b64: str) -> str:
    """Decode a base64/URL video to a temp file path."""
    raw = _decode_b64_or_url(video_b64)
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp.write(raw)
    tmp.close()
    return tmp.name


# =============================================================================
# Pipeline loading
# =============================================================================

def _detect_pipeline_class(model_name: str, mode: str):
    """Return the appropriate diffusers pipeline class."""
    n = model_name.lower()
    try:
        from diffusers import (
            CogVideoXPipeline, CogVideoXImageToVideoPipeline,
            LTXPipeline, LTXImageToVideoPipeline,
            StableVideoDiffusionPipeline,
            I2VGenXLPipeline,
            AnimateDiffPipeline,
        )
        if 'cogvideox' in n or 'cogvideo' in n:
            return CogVideoXImageToVideoPipeline if (mode in ('i2v', 'ti2v')) else CogVideoXPipeline
        if 'ltx' in n:
            return LTXImageToVideoPipeline if (mode in ('i2v', 'ti2v')) else LTXPipeline
        if 'svd' in n or 'stable-video-diffusion' in n:
            return StableVideoDiffusionPipeline
        if 'i2vgen' in n:
            return I2VGenXLPipeline
        if 'animatediff' in n or 'animateddiff' in n:
            return AnimateDiffPipeline
        if 'wan' in n:
            # Wan ships separate t2v and i2v transformers; pick the i2v pipeline
            # for the keyframe bridge when an init image is supplied.
            if mode in ('i2v', 'ti2v'):
                try:
                    from diffusers import WanImageToVideoPipeline
                    return WanImageToVideoPipeline
                except ImportError:
                    pass
            try:
                from diffusers import WanPipeline
                return WanPipeline
            except ImportError:
                pass
    except ImportError:
        pass
    try:
        from diffusers import DiffusionPipeline
        return DiffusionPipeline
    except ImportError:
        return None


def _gguf_needs_wan_prefix(path: str) -> bool:
    """Return True if this GGUF has bare WAN tensor names (no model.diffusion_model. prefix)."""
    import struct
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                return False
            f.read(4)  # version
            f.read(8)  # tensor_count
            kv_count = struct.unpack('<Q', f.read(8))[0]

            def _read_str(fh):
                n = struct.unpack('<Q', fh.read(8))[0]
                return fh.read(n)

            arch = b''
            for _ in range(kv_count):
                key = _read_str(f)
                vtype = struct.unpack('<I', f.read(4))[0]
                if vtype == 8:   # string
                    val = _read_str(f)
                    if key == b'general.architecture':
                        arch = val
                elif vtype in (0, 1, 7):  # u8/i8/bool
                    f.read(1)
                elif vtype in (2, 3):    # u16/i16
                    f.read(2)
                elif vtype in (4, 5, 6): # u32/i32/f32
                    f.read(4)
                elif vtype in (10, 11, 12): # u64/i64/f64
                    f.read(8)
                elif vtype == 9:         # array — skip
                    atype = struct.unpack('<I', f.read(4))[0]
                    alen  = struct.unpack('<Q', f.read(8))[0]
                    # skip array payload: only handle simple element types
                    sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
                    if atype in sizes:
                        f.read(alen * sizes[atype])
                    else:
                        return False  # complex array — bail out safely
                else:
                    return False  # unknown type — bail out

            if arch.lower() != b'wan':
                return False

            # Read first tensor name; if it lacks the expected prefix, rewrite is needed
            name = _read_str(f)
            return not name.startswith(b'model.diffusion_model.')
    except Exception:
        return False


def _ensure_wan_prefixed_gguf(src_path: str) -> str:
    """Return a path to a GGUF with model.diffusion_model. prefixed tensor names.

    If src_path already has the prefix, returns it unchanged.
    Otherwise creates a sibling file <name>.sdcpp.gguf by rewriting the header
    and streaming the original data section — no full file load into memory.
    """
    import os as _os, struct, shutil

    dst_path = src_path + '.sdcpp.gguf'
    if _os.path.exists(dst_path) and _os.path.getmtime(dst_path) >= _os.path.getmtime(src_path):
        print(f"  [gguf] using cached prefixed GGUF: {dst_path}")
        return dst_path

    prefix = b'model.diffusion_model.'
    print(f"  [gguf] rewriting tensor names with '{prefix.decode()}' prefix …")

    with open(src_path, 'rb') as src:
        magic   = src.read(4)
        version = src.read(4)
        tensor_count_bytes = src.read(8)
        kv_count_bytes     = src.read(8)
        tensor_count = struct.unpack('<Q', tensor_count_bytes)[0]
        kv_count     = struct.unpack('<Q', kv_count_bytes)[0]

        # ── collect raw KV bytes (pass through unchanged) ──────────────────
        kv_bytes = bytearray()

        def _read_str_raw(fh):
            n_bytes = fh.read(8)
            n = struct.unpack('<Q', n_bytes)[0]
            data = fh.read(n)
            return n_bytes + data, data

        for _ in range(kv_count):
            raw_key, _ = _read_str_raw(src)
            kv_bytes += raw_key
            vtype_bytes = src.read(4)
            vtype = struct.unpack('<I', vtype_bytes)[0]
            kv_bytes += vtype_bytes
            if vtype == 8:
                raw_val, _ = _read_str_raw(src)
                kv_bytes += raw_val
            elif vtype in (0, 1, 7):
                kv_bytes += src.read(1)
            elif vtype in (2, 3):
                kv_bytes += src.read(2)
            elif vtype in (4, 5, 6):
                kv_bytes += src.read(4)
            elif vtype in (10, 11, 12):
                kv_bytes += src.read(8)
            elif vtype == 9:
                atype_bytes = src.read(4)
                alen_bytes  = src.read(8)
                atype = struct.unpack('<I', atype_bytes)[0]
                alen  = struct.unpack('<Q', alen_bytes)[0]
                sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
                elem_size = sizes.get(atype, 0)
                arr_data = src.read(alen * elem_size) if elem_size else b''
                kv_bytes += atype_bytes + alen_bytes + arr_data

        # ── collect tensor info, prefixing each name ────────────────────────
        ti_bytes = bytearray()
        for _ in range(tensor_count):
            raw_nlen, name = _read_str_raw(src)
            new_name = prefix + name
            ti_bytes += struct.pack('<Q', len(new_name)) + new_name
            n_dims_bytes = src.read(4)
            n_dims = struct.unpack('<I', n_dims_bytes)[0]
            ti_bytes += n_dims_bytes
            ti_bytes += src.read(n_dims * 8)  # shape (u64 each)
            ti_bytes += src.read(4)            # dtype
            ti_bytes += src.read(8)            # offset within data section

        # Alignment: GGUF data section starts at next 32-byte boundary
        ALIGN = 32
        header_size = 4 + 4 + 8 + 8 + len(kv_bytes) + len(ti_bytes)
        pad = (ALIGN - header_size % ALIGN) % ALIGN
        data_offset = src.tell() + ((ALIGN - src.tell() % ALIGN) % ALIGN)
        src.seek(data_offset)

        with open(dst_path, 'wb') as dst:
            dst.write(magic + version + tensor_count_bytes + kv_count_bytes)
            dst.write(kv_bytes)
            dst.write(ti_bytes)
            dst.write(b'\x00' * pad)
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)

    print(f"  [gguf] prefixed GGUF written: {dst_path}")
    return dst_path


def _load_sdcpp_video_model(model_path: str, offload: str = None, model_cfg: dict = None):
    """Load a GGUF video model via stable-diffusion.cpp."""
    try:
        from stable_diffusion_cpp import StableDiffusion
        import stable_diffusion_cpp.stable_diffusion_cpp as _sd_cpp
    except ImportError:
        raise RuntimeError("stable-diffusion-cpp-python required: pip install stable-diffusion-cpp-python")

    import os as _os
    model_cfg = model_cfg or {}

    # Resolve bare filename to absolute path from the GGUF cache
    if not _os.path.isabs(model_path) and not _os.path.exists(model_path):
        try:
            from codai.models.cache import get_model_cache_dir
            candidate = _os.path.join(get_model_cache_dir(), model_path)
            if _os.path.exists(candidate):
                model_path = candidate
        except Exception:
            pass
    if not _os.path.exists(model_path):
        raise FileNotFoundError(f"GGUF video model not found: {model_path}")

    # WAN DiT-only GGUFs (e.g. QuantStack) contain only the denoiser — no VAE or
    # text encoders.  They must be loaded via diffusion_model_path, not model_path.
    # sd.cpp internally prepends "model.diffusion_model." when reading tensors from
    # diffusion_model_path, so the original bare-named file is passed directly.
    # VAE / text encoders must be supplied separately via model_cfg component paths.
    _is_wan_dit = _gguf_needs_wan_prefix(model_path)
    if _is_wan_dit:
        print(f"Loading sd.cpp video model (WAN DiT, diffusion_model_path): {model_path}")
        kwargs = {'diffusion_model_path': model_path, 'verbose': True}
    else:
        print(f"Loading sd.cpp video model: {model_path}")
        kwargs = {'model_path': model_path, 'verbose': True}
    if offload in ('model', 'cpu', 'sequential'):
        kwargs['offload_params_to_cpu'] = True
        kwargs['keep_clip_on_cpu'] = True
        kwargs['keep_vae_on_cpu'] = True

    # sd.cpp VRAM budget for graph-cut layer execution (0 = disabled)
    max_vram = float(model_cfg.get('max_vram') or 0)
    if max_vram > 0:
        kwargs['max_vram'] = max_vram

    # Flash attention variants (sdcpp_flash_attn = full, sdcpp_diffusion_flash_attn = DiT only)
    if model_cfg.get('sdcpp_flash_attn'):
        kwargs['flash_attn'] = True
    if model_cfg.get('sdcpp_diffusion_flash_attn'):
        kwargs['diffusion_flash_attn'] = True

    # Inject component paths from per-model configuration
    for key in ('vae_path', 't5xxl_path', 'clip_l_path', 'clip_g_path',
                'clip_vision_path', 'lora_model_dir'):
        val = (model_cfg.get(key) or '').strip()
        if val:
            kwargs[key] = val
            print(f"  [sd.cpp] {key}: {val}")

    # A single LoRA file associated with this model: derive lora_model_dir from its parent
    # and append <lora:basename:1.0> to the default prompt via lora_model_dir.
    # sd.cpp needs a directory, so point it at the file's parent directory.
    lora_path = (model_cfg.get('lora_path') or '').strip()
    if lora_path and 'lora_model_dir' not in kwargs:
        import os as _os2
        kwargs['lora_model_dir'] = _os2.path.dirname(lora_path) or '.'
        print(f"  [sd.cpp] lora_path: {lora_path} → lora_model_dir: {kwargs['lora_model_dir']}")

    _load_errors = []

    @_sd_cpp.sd_log_callback
    def _log_cb(level, text, data):
        if text:
            line = text.decode('utf-8', errors='replace').rstrip()
            if line:
                print(f"  [sd.cpp] {line}", flush=True)
                # Capture fatal load errors so we can raise instead of returning
                # a broken object with a NULL internal pointer.
                if any(k in line for k in (
                    'load tensors from model loader failed',
                    'failed to read tensor info',
                    'failed to load model',
                )):
                    _load_errors.append(line)

    _sd_cpp.sd_set_log_callback(_log_cb, None)
    try:
        model = StableDiffusion(**kwargs)
    finally:
        _sd_cpp.sd_set_log_callback(None, None)

    if _load_errors:
        hint = ""
        if 'No text encoders' in ''.join(_load_errors) or _is_wan_dit:
            hint = (
                " This GGUF contains only the diffusion model weights. "
                "Configure t5xxl_path (and optionally vae_path) in the model's "
                "CoderAI settings to point at the separate text-encoder and VAE files."
            )
        raise RuntimeError(f"sd.cpp failed to load model: {_load_errors[-1]}.{hint}")

    return model


def _generate_sdcpp_video(sd_model, request, model_cfg=None):
    """Generate frames via stable-diffusion.cpp and return (frames, fps)."""
    mode = request.mode or 't2v'
    fps = request.fps or 8
    num_frames = request.num_frames or 25

    # Acceleration/distillation defaults (sd.cpp can't fuse a diffusers LoRA, but
    # we can still honour the preset's low step-count / guidance, and inject the
    # distill LoRA via sd.cpp's "<lora:name:weight>" prompt syntax when a
    # lora_model_dir is configured).
    from codai.models.acceleration import resolve_acceleration
    _accel = resolve_acceleration(model_cfg)
    _accel_steps = _accel.get('steps') if _accel else None
    _accel_cfg = _accel.get('guidance_scale') if _accel else None
    steps = request.num_inference_steps or _accel_steps or 20
    cfg_scale = request.guidance_scale or _accel_cfg or 7.0

    prompt = request.prompt or ''
    if _accel and _accel.get('lora') and (model_cfg or {}).get('lora_model_dir'):
        from codai.models.acceleration import _split_lora_ref
        _repo, _wn = _split_lora_ref(_accel['lora'])
        _lname = (_wn or _repo).rsplit('/', 1)[-1]
        for _suf in ('.safetensors', '.ckpt', '.pt', '.bin'):
            if _lname.endswith(_suf):
                _lname = _lname[: -len(_suf)]
        prompt = f"{prompt} <lora:{_lname}:{_accel.get('lora_weight') or 1.0}>"

    _vid_progress_reset(steps)

    # sd.cpp runs the whole diffusion in one C call → not interruptible mid-step;
    # register for visibility + step progress (cancel applies once back in Python).
    _tid = task_registry.register(
        "video", title=(prompt or mode or "")[:80],
        model=getattr(request, 'model', '') or '', total=steps)
    task_registry.start(_tid)

    def _progress_cb(step: int, total: int, elapsed: float):
        task_registry.step(_tid, step)
        _vid_progress_step(step)

    kw = {
        'prompt':          prompt,
        'negative_prompt': request.negative_prompt or '',
        'width':           request.width or 512,
        'height':          request.height or 512,
        'video_frames':    num_frames,
        'sample_steps':    steps,
        'cfg_scale':       cfg_scale,
        'seed':            request.seed if request.seed is not None else -1,
        'progress_callback': _progress_cb,
    }

    if (model_cfg or {}).get('vae_tiling'):
        kw['vae_tiling'] = True

    init_src = request.init_image or request.image
    if mode in ('i2v', 'ti2v') and init_src:
        kw['init_image'] = _pil_from_b64(init_src)
    elif mode == 'interp':
        if not init_src or not request.end_image:
            raise ValueError("interp mode requires both init_image and end_image")
        kw['init_image'] = _pil_from_b64(init_src)
        kw['end_image']  = _pil_from_b64(request.end_image)

    try:
        frames = sd_model.generate_video(**kw)
    except Exception as e:
        task_registry.finish(_tid, "error", str(e)[:200])
        _vid_progress_done()
        raise
    _vid_progress_done()
    task_registry.finish(_tid, "done")
    return list(frames), fps


def _free_pipeline_vram(pipe) -> None:
    """Thoroughly release a diffusers pipeline's GPU memory.

    Quantized (bitsandbytes) and device_map pipelines REJECT `.to('cpu')`, so a
    naive move doesn't free VRAM and leaves the weights resident — the cause of
    the OOM death spiral where every subsequent request OOMs on load. Remove
    accelerate offload hooks, break component references, then collect + empty
    the CUDA cache so the next load starts from clean VRAM.
    """
    import gc as _gc
    try:
        import torch as _t
    except Exception:
        _t = None
    try:
        if pipe is not None:
            _comps = getattr(pipe, 'components', {}) or {}
            try:
                from accelerate.hooks import remove_hook_from_submodules
                for _c in _comps.values():
                    if hasattr(_c, 'modules'):
                        try:
                            remove_hook_from_submodules(_c)
                        except Exception:
                            pass
            except Exception:
                pass
            for _cn in list(_comps):
                try:
                    setattr(pipe, _cn, None)
                except Exception:
                    pass
    except Exception:
        pass
    for _ in range(3):
        _gc.collect()
    if _t is not None:
        try:
            if _t.cuda.is_available():
                _t.cuda.synchronize()
                _t.cuda.empty_cache()
        except Exception:
            pass
    try:
        from codai.models.manager import _trim_cpu_ram
        _trim_cpu_ram()
    except Exception:
        pass


def _lora_file_size_gb(ref) -> float:
    """Size (GB) of a distill LoRA ref ('repo:weight' / local path) from cache."""
    if not ref:
        return 0.0
    try:
        import os
        from codai.models.acceleration import _split_lora_ref
        if os.path.isfile(ref):
            return os.path.getsize(ref) / 1e9
        repo, weight = _split_lora_ref(ref)
        if repo and os.path.isfile(repo):
            return os.path.getsize(repo) / 1e9
        if repo and weight:
            from huggingface_hub import try_to_load_from_cache
            p = try_to_load_from_cache(repo, weight)
            if isinstance(p, str) and os.path.isfile(p):
                return os.path.getsize(p) / 1e9
    except Exception:
        pass
    return 0.0


def _accel_vram_gb(model_cfg: dict) -> float:
    """VRAM the fused acceleration/distill LoRA(s) add. Sums the actual cached
    file sizes when known (both Wan2.2 experts), else a conservative reserve when
    acceleration is enabled but the size can't be resolved."""
    try:
        from codai.models.acceleration import resolve_acceleration
        a = resolve_acceleration(model_cfg)
        if not a:
            return 0.0
        refs = [r for r in (a.get('lora_high'), a.get('lora_low'), a.get('lora')) if r]
        refs = list(dict.fromkeys(refs))  # a single LoRA on both experts counts once
        total, known = 0.0, False
        for r in refs:
            sz = _lora_file_size_gb(r)
            if sz > 0:
                total += sz
                known = True
        if known:
            return total
        return 2.5 if refs else 0.0  # accel on but size unknown → reserve
    except Exception:
        return 0.0


def _video_runtime_reserve_gb(request) -> float:
    """Rough VRAM headroom for the denoise activations + VAE-decode spike, which
    scales with frame count × resolution. Keeps the auto-offload decision from
    being too optimistic for long/high-res clips."""
    try:
        nf = int(getattr(request, 'num_frames', None) or 16)
        w = int(getattr(request, 'width', None) or 512)
        h = int(getattr(request, 'height', None) or 512)
        base = 3.0 * (nf / 16.0) * ((w * h) / (512.0 * 512.0))
        return max(2.0, min(12.0, base))
    except Exception:
        return 3.0


def _load_video_pipeline(model_name: str, device: str, mode: str, offload: str = None, model_cfg: dict = None):
    # GGUF models go through stable-diffusion.cpp, not diffusers
    from codai.api.images import _is_gguf_model
    if _is_gguf_model(model_name):
        return _load_sdcpp_video_model(model_name, offload, model_cfg)

    import sys, time, torch, gc
    PClass = _detect_pipeline_class(model_name, mode)
    if PClass is None:
        raise RuntimeError("diffusers not installed: pip install diffusers")
    # Per-model precision wins; fall back to bfloat16 for video (NOT the global
    # image_precision, which may be f32 for image models and is wrong for video).
    _model_precision = (model_cfg or {}).get('precision') or 'bf16'
    dtype_map = {'bf16': torch.bfloat16, 'f16': torch.float16, 'f32': torch.float32}
    torch_dtype = dtype_map.get(_model_precision, torch.bfloat16)

    # ── Pipeline disk cache (--pipeline-cache) ───────────────────────────────
    # When a previously-built, quantized pipeline is cached on disk, load the
    # pre-quantized weights from there (no re-download / re-quantization). The
    # cache holds the BASE pipeline only — the acceleration LoRA is re-fused by
    # the caller per load — so the source just swaps from the HF id to the cache
    # dir and the quantization config is skipped (the saved components carry it).
    _orig_model_name = model_name
    _pc_save_path = None
    _loading_from_cache = False
    try:
        from codai.models import pipeline_cache as _pcache
        if _pcache.enabled():
            _pc_path = _pcache.path(model_name, model_cfg)
            if _pcache.valid(_pc_path):
                print(f"  [pipeline-cache] HIT — loading pre-quantized pipeline "
                      f"from {_pc_path}")
                model_name = _pc_path
                _loading_from_cache = True
            else:
                _pc_save_path = _pc_path
                print(f"  [pipeline-cache] MISS — will build, then cache for next start")
    except Exception as _pce:
        print(f"  [pipeline-cache] unavailable ({_pce})")

    # ── Quantization (from per-model config) ─────────────────────────────────
    # Per-component overrides ('component_quantization' map) win; otherwise the
    # global load_in_4bit/8bit flag is applied to every heavy component (the
    # diffusion backbone transformer*/unet AND text encoder(s) — e.g. Wan2.2's
    # UMT5 text_encoder is ~11 GB in bf16 and must be quantized to fit on-GPU).
    from codai.models.hf_loading import (
        build_pipeline_quant_config, build_gguf_pipeline_components)
    if _loading_from_cache:
        # Cached components already carry their quantization config; don't rebuild
        # it (and don't re-inject GGUF components — they were baked into the cache).
        _quant_config, _quant_desc = None, ''
        _gguf_components, _gguf_desc = {}, ''
    else:
        _quant_config, _quant_desc = build_pipeline_quant_config(
            model_name, model_cfg, torch_dtype)
        if _quant_config is not None:
            print(f"  Video quantization: {_quant_desc}")
        # GGUF-quantized components (Q5_K/Q6_K etc.) are loaded from their .gguf
        # files and injected into the pipeline as pre-built components.
        _gguf_components, _gguf_desc = build_gguf_pipeline_components(
            model_name, model_cfg, torch_dtype)
        if _gguf_components:
            print(f"  Video GGUF components: {_gguf_desc}")

    def _with_quant(kw: dict) -> dict:
        """Inject quantization_config + GGUF components into from_pretrained kwargs."""
        if _quant_config is not None:
            kw = {**kw, 'quantization_config': _quant_config}
        if _gguf_components:
            kw = {**kw, **_gguf_components}
        return kw
    # Explicit parameter wins; fall back to global CLI arg
    if offload is None:
        offload = getattr(global_args, 'offload_strategy', None) if global_args else None
    # Normalise UI values to diffusers vocabulary
    if offload == 'cpu':
        offload = 'model'
    elif offload in ('disk', 'offload'):
        offload = 'disk'

    # Resolve offload directory for disk-offload fallback.
    _offload_dir = (
        (model_cfg or {}).get('offload_dir')
        or (getattr(global_args, 'offload_dir', None) if global_args else None)
        or os.path.join(os.path.expanduser('~'), '.cache', 'coderai', 'offload')
    )

    import psutil as _psutil

    def _is_oom(exc: Exception) -> bool:
        s = str(exc).lower()
        return 'out of memory' in s or 'cannot allocate' in s or 'killed' in s

    pipe = None  # bound here so _clear_mem can release a failed attempt's pipe

    def _clear_mem():
        # CRITICAL: drop the partial pipeline from the FAILED attempt before the
        # next strategy allocates a new one. Without this, a from_pretrained that
        # loaded components and then OOM'd (e.g. on .to(device)) leaves the whole
        # pipeline referenced by `pipe`, so the next attempt holds TWO copies and
        # OOMs again — the VRAM death spiral. Remove accelerate offload hooks and
        # break component references so the GPU tensors are actually collectable.
        nonlocal pipe
        try:
            if pipe is not None:
                _comps = getattr(pipe, 'components', {}) or {}
                try:
                    from accelerate.hooks import remove_hook_from_submodules
                    for _comp in _comps.values():
                        if hasattr(_comp, 'modules'):  # an nn.Module component
                            try:
                                remove_hook_from_submodules(_comp)
                            except Exception:
                                pass
                except Exception:
                    pass
                for _cn in list(_comps):
                    try:
                        setattr(pipe, _cn, None)
                    except Exception:
                        pass
        finally:
            pipe = None
        gc.collect()
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _mem_snapshot(label: str = "") -> dict:
        """Collect and print GPU VRAM / CPU RAM / disk state."""
        snap = {}
        try:
            vm = _psutil.virtual_memory()
            snap['ram_total_gb'] = vm.total / 1e9
            snap['ram_used_gb']  = vm.used  / 1e9
            snap['ram_free_gb']  = vm.available / 1e9
        except Exception:
            snap['ram_free_gb'] = 0
        try:
            if torch.cuda.is_available():
                free_v, total_v = torch.cuda.mem_get_info()
                snap['vram_total_gb'] = total_v / 1e9
                snap['vram_used_gb']  = (total_v - free_v) / 1e9
                snap['vram_free_gb']  = free_v / 1e9
        except Exception:
            pass
        try:
            du = _psutil.disk_usage(_offload_dir if os.path.exists(_offload_dir) else '/')
            snap['disk_free_gb'] = du.free / 1e9
            snap['disk_used_gb'] = du.used / 1e9
        except Exception:
            pass

        parts = []
        if 'vram_free_gb' in snap:
            parts.append(f"VRAM {snap['vram_used_gb']:.1f}/{snap['vram_total_gb']:.1f} GB"
                         f" ({snap['vram_free_gb']:.1f} GB free)")
        if 'ram_free_gb' in snap:
            parts.append(f"RAM {snap['ram_used_gb']:.1f}/{snap['ram_total_gb']:.1f} GB"
                         f" ({snap['ram_free_gb']:.1f} GB free)")
        if 'disk_free_gb' in snap:
            parts.append(f"disk {snap['disk_free_gb']:.1f} GB free")
        tag = f"[{label}] " if label else ""
        print(f"  {tag}Memory: {' | '.join(parts)}")
        return snap

    def _report_device_map(pipe) -> None:
        """Print a summary of which layers landed on which device."""
        device_counts: dict = {}
        # Diffusers pipelines expose components dict
        components = getattr(pipe, 'components', {})
        for comp_name, comp in components.items():
            if comp is None or not hasattr(comp, 'hf_device_map'):
                continue
            dm = comp.hf_device_map  # dict: layer_name → device
            for layer, dev in dm.items():
                dev_str = str(dev)
                device_counts[dev_str] = device_counts.get(dev_str, 0) + 1
        if device_counts:
            summary = ', '.join(f"{d}: {n} layers" for d, n in sorted(device_counts.items()))
            print(f"  Device map: {summary}")
        else:
            # Try the pipeline-level hf_device_map
            dm = getattr(pipe, 'hf_device_map', None)
            if dm:
                by_dev: dict = {}
                for layer, dev in dm.items():
                    dev_str = str(dev)
                    by_dev[dev_str] = by_dev.get(dev_str, 0) + 1
                summary = ', '.join(f"{d}: {n} layers" for d, n in sorted(by_dev.items()))
                print(f"  Device map: {summary}")

    def _report_offload_dir_size() -> None:
        """Print how much disk space the offload directory is using."""
        if not os.path.isdir(_offload_dir):
            return
        try:
            total = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(_offload_dir)
                for f in files
            )
            print(f"  Offload dir: {_offload_dir} — {total / 1e9:.2f} GB on disk")
        except Exception:
            pass

    def _enable_vae_memory_opts(pipe) -> None:
        """Enable VAE tiling + slicing on a diffusers video pipeline.

        The VAE decode (turning denoised latents into RGB frames) is the single
        biggest VRAM spike in video generation — it happens AFTER the whole
        denoise loop, so an OOM there throws away a completed generation and
        forces an expensive full-pipeline reload + retry.  Tiling/slicing splits
        that decode into chunks, cutting the peak VRAM with negligible quality
        impact.  Model-agnostic: only calls the methods that exist.  Controlled
        by the per-model `vae_tiling` config (default ON for video); set it to
        false to disable for max-quality decode when VRAM is plentiful.
        """
        if not (model_cfg or {}).get('vae_tiling', True):
            _enc_dbg("VAE tiling disabled by config (vae_tiling=false)")
            return
        enabled = []
        for meth in ('enable_vae_tiling', 'enable_vae_slicing'):
            fn = getattr(pipe, meth, None)
            if callable(fn):
                try:
                    fn()
                    enabled.append(meth.replace('enable_vae_', ''))
                except Exception as _e:
                    _enc_dbg(f"{meth} failed: {_e}")
        # Some pipelines only expose these on the .vae submodule.
        if not enabled:
            vae = getattr(pipe, 'vae', None)
            for meth in ('enable_tiling', 'enable_slicing'):
                fn = getattr(vae, meth, None) if vae is not None else None
                if callable(fn):
                    try:
                        fn()
                        enabled.append(meth.replace('enable_', ''))
                    except Exception as _e:
                        _enc_dbg(f"vae.{meth} failed: {_e}")
        if enabled:
            print(f"  VAE memory opt: {' + '.join(enabled)} enabled "
                  f"(reduces decode-time VRAM spike)")
        else:
            _enc_dbg("no VAE tiling/slicing methods available on this pipeline")

    def _report_loaded(pipe, strategy: str) -> None:
        """Print a post-load summary: strategy, device placement, memory state."""
        _enable_vae_memory_opts(pipe)
        print(f"  ✓ Video pipeline loaded — strategy: {strategy}")
        _report_device_map(pipe)
        _report_offload_dir_size()
        _mem_snapshot("after load")
        # Persist the freshly-built quantized pipeline to the disk cache so the
        # next start can skip the rebuild. Only on a cache MISS (we didn't load
        # from it) and when --pipeline-cache is on. Best-effort; never fatal.
        if _pc_save_path and not _loading_from_cache:
            try:
                from codai.models import pipeline_cache as _pcache
                _pcache.save(pipe, _pc_save_path,
                             model_name=_orig_model_name, model_cfg=model_cfg)
            except Exception as _se:
                print(f"  [pipeline-cache] save skipped ({_se})")

    # NOTE: we deliberately do NOT lower sys.setswitchinterval here.  A previous
    # version set it to 0.001s to keep the asyncio loop responsive during the
    # GIL-heavy diffusers from_pretrained, but forcing a GIL switch every 1 ms for
    # the whole (multi-minute) load caused severe scheduler thrashing — CPU load
    # average > 10 and a sluggish machine.  torch releases the GIL during the
    # actual tensor work, and the load already runs in an executor thread, so the
    # default switch interval is the right choice.
    _old_interval = sys.getswitchinterval()
    try:
        # ── Balanced GPU+CPU strategy (explicit or auto-selected) ────────────
        # Uses device_map='balanced' with a configurable GPU-VRAM cap so all
        # available GPU is filled first and only the remainder spills to CPU
        # RAM (and disk if needed). This is the preferred strategy when the
        # model won't fit entirely in VRAM but should maximise GPU utilisation.
        # `gpu_percent` (0–100) controls what fraction of FREE VRAM to occupy.
        # Configured GPU cap for balanced (per-model balanced_gpu_percent, else 80).
        # This is the STARTING cap for the balanced chain; on OOM it steps down.
        _gpu_pct = float((model_cfg or {}).get('balanced_gpu_percent') or 80)

        def _load_balanced(gpu_pct: float):
            """device_map='balanced' capping GPU at gpu_pct% of free VRAM, spilling
            to CPU then disk. Assigns the outer `pipe` so a failed attempt's VRAM
            is reclaimable via _clear_mem. Raises (RuntimeError/MemoryError) on OOM."""
            nonlocal pipe
            try:
                _free_v2, _ = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
                _avail_gpu_gb = (_free_v2 / 1e9) * (gpu_pct / 100.0)
            except Exception:
                _avail_gpu_gb = 0.0
            _cpu_avail_gb = min(48, max(4, int(
                _psutil.virtual_memory().available * 0.60 / 1e9)))
            _mem_snapshot(f"before balanced {gpu_pct:.0f}% GPU+CPU load")
            print(f"  Video load strategy: balanced GPU+CPU "
                  f"({gpu_pct:.0f}% GPU → {_avail_gpu_gb:.1f} GiB / "
                  f"CPU {_cpu_avail_gb} GiB / overflow → {_offload_dir})")
            os.makedirs(_offload_dir, exist_ok=True)
            try:
                pipe = PClass.from_pretrained(**_with_quant(dict(
                    pretrained_model_name_or_path=model_name,
                    torch_dtype=torch_dtype,
                    device_map='balanced',
                    max_memory={0: f'{_avail_gpu_gb:.2f}GiB',
                                'cpu': f'{_cpu_avail_gb}GiB'},
                    offload_folder=_offload_dir,
                    offload_buffers=True,
                    low_cpu_mem_usage=True,
                )))
            except (TypeError, ValueError):
                # Pipeline doesn't accept device_map — fall to model CPU offload.
                pipe = PClass.from_pretrained(**_with_quant(dict(
                    pretrained_model_name_or_path=model_name,
                    torch_dtype=torch_dtype, low_cpu_mem_usage=True)))
                pipe.enable_model_cpu_offload()
            _report_loaded(pipe, f"balanced {gpu_pct:.0f}%GPU+CPU")
            return pipe

        def _try_balanced_chain():
            """Try balanced starting at the configured GPU %, stepping down through
            60% then 40% on OOM. Returns the pipe, or None if every step OOM'd
            (caller then falls through to model/sequential/disk offload)."""
            # Start at the configured cap, then step down through the standard
            # checkpoints (80/60/40) that sit below it. So 90% → 90,80,60,40;
            # 80% → 80,60,40; 70% → 70,60,40; 50% → 50,40.
            _pcts = sorted({_gpu_pct} | {p for p in (80.0, 60.0, 40.0)
                                         if p < _gpu_pct}, reverse=True)
            for _i, _pct in enumerate(_pcts):
                try:
                    return _load_balanced(_pct)
                except (RuntimeError, MemoryError) as _e:
                    if not _is_oom(_e):
                        raise
                    _nxt = _pcts[_i + 1] if _i + 1 < len(_pcts) else None
                    if _nxt is not None:
                        print(f"  Video: balanced {_pct:.0f}% GPU OOM — "
                              f"retrying at {_nxt:.0f}% GPU…")
                    else:
                        print(f"  Video: balanced {_pct:.0f}% GPU OOM — "
                              f"falling back to sequential CPU offload…")
                    _clear_mem()
            return None

        def _load_sequential():
            """Most aggressive fit: stream each submodule GPU↔CPU during the
            forward pass (slowest, lowest VRAM). Assigns the outer `pipe`. Raises
            on OOM."""
            nonlocal pipe
            _mem_snapshot("before sequential CPU offload load")
            print("  Video load strategy: sequential CPU offload "
                  "(each submodule GPU↔CPU during forward; slowest, lowest VRAM)")
            pipe = PClass.from_pretrained(**_with_quant(dict(
                pretrained_model_name_or_path=model_name,
                torch_dtype=torch_dtype, low_cpu_mem_usage=True)))
            pipe.enable_sequential_cpu_offload()
            _report_loaded(pipe, "sequential CPU offload")
            return pipe

        def _try_balanced_then_sequential():
            """Balanced chain (configured% → 60 → 40), then sequential CPU offload
            if all balanced steps OOM. Returns the pipe, or None if even sequential
            OOMs (caller falls through to the disk-offload attempts)."""
            p = _try_balanced_chain()
            if p is not None:
                return p
            _clear_mem()
            try:
                return _load_sequential()
            except (RuntimeError, MemoryError) as _e:
                if not _is_oom(_e):
                    raise
                print(f"  Video: sequential CPU offload OOM ({_e}) — "
                      f"trying disk offload…")
                _clear_mem()
            return None

        if offload == 'balanced':
            pipe = _try_balanced_then_sequential()
            if pipe is not None:
                return pipe
            # Even sequential OOM'd → continue to the disk-offload attempts.

        # ── Attempt 0: full GPU ──────────────────────────────────────────────
        if offload not in ('model', 'sequential', 'disk', 'balanced'):
            _mem_snapshot("before full-GPU load")
            _q = " + quantized" if _quant_config is not None else ""
            print(f"  Video load strategy: full GPU ({torch_dtype}{_q})")
            try:
                time.sleep(0)
                if _quant_config is not None:
                    # Quantized weights load directly to GPU via device_map;
                    # bitsandbytes models cannot be moved with .to() afterwards.
                    pipe = PClass.from_pretrained(**_with_quant(dict(
                        pretrained_model_name_or_path=model_name,
                        torch_dtype=torch_dtype,
                        device_map='cuda',
                        low_cpu_mem_usage=True)))
                else:
                    pipe = PClass.from_pretrained(
                        model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
                    time.sleep(0)
                    pipe = pipe.to(device)
                _report_loaded(pipe, "full GPU" + _q)
                return pipe
            except (RuntimeError, MemoryError) as e:
                if not _is_oom(e):
                    raise
                print(f"  Video: full-GPU OOM ({e}) — falling back to balanced "
                      f"GPU+CPU (starting at {_gpu_pct:.0f}% GPU)…")
                _clear_mem()
                # Graceful degrade: balanced at the configured %, then 60%, then
                # 40%, then sequential CPU offload, before the slower disk paths.
                pipe = _try_balanced_then_sequential()
                if pipe is not None:
                    return pipe
                print("  Video: balanced + sequential all OOM — trying disk offload…")
                _clear_mem()

        # ── Attempt 1: model CPU offload ─────────────────────────────────────
        if offload not in ('sequential', 'disk'):
            _mem_snapshot("before model-CPU-offload load")
            print(f"  Video load strategy: model CPU offload"
                  f" (each module GPU↔CPU during forward pass)")
            try:
                time.sleep(0)
                pipe = PClass.from_pretrained(**_with_quant(dict(
                    pretrained_model_name_or_path=model_name,
                    torch_dtype=torch_dtype, low_cpu_mem_usage=True)))
                pipe.enable_model_cpu_offload()
                _report_loaded(pipe, "model CPU offload")
                return pipe
            except (RuntimeError, MemoryError) as e:
                if not _is_oom(e):
                    raise
                print(f"  Video: model CPU offload OOM ({e})"
                      f" — trying GPU+CPU+disk offload…")
                _clear_mem()

        # ── Attempt 2: GPU + CPU + disk offload via device_map='auto' ──────────
        os.makedirs(_offload_dir, exist_ok=True)
        _cpu_gb = min(32, max(2, int(_psutil.virtual_memory().available * 0.50 / 1e9)))
        if offload != 'disk':
            _mem_snapshot("before GPU+CPU+disk offload")
            print(f"  Video load strategy: GPU+CPU+disk offload"
                  f" — GPU 2 GiB / CPU {_cpu_gb} GiB / overflow → {_offload_dir}")
            try:
                time.sleep(0)
                try:
                    # diffusers PIPELINES only accept device_map='balanced'
                    # (not 'auto', which is a transformers-model value).
                    pipe = PClass.from_pretrained(**_with_quant(dict(
                        pretrained_model_name_or_path=model_name,
                        torch_dtype=torch_dtype,
                        device_map='balanced',
                        max_memory={0: '2GiB', 'cpu': f'{_cpu_gb}GiB'},
                        offload_folder=_offload_dir,
                        offload_buffers=True,
                        low_cpu_mem_usage=True,
                    )))
                except (TypeError, ValueError):
                    # device_map unsupported/invalid for this pipeline → load on
                    # CPU then stream each module GPU↔CPU during the forward pass.
                    pipe = PClass.from_pretrained(**_with_quant(dict(
                        pretrained_model_name_or_path=model_name,
                        torch_dtype=torch_dtype, low_cpu_mem_usage=True)))
                    pipe.enable_sequential_cpu_offload()
                _report_loaded(pipe, f"GPU 2 GiB + CPU {_cpu_gb} GiB + disk")
                return pipe
            except (RuntimeError, MemoryError) as e:
                if not _is_oom(e):
                    raise
                print(f"  Video: GPU+CPU+disk OOM ({e}) — trying minimal-RAM disk offload…")
                _clear_mem()
                _cpu_gb = min(8, max(2, int(_psutil.virtual_memory().available * 0.40 / 1e9)))

        # ── Attempt 3: minimal RAM, maximum disk offload ──────────────────────
        _mem_snapshot("before pure-disk offload")
        print(f"  Video load strategy: pure disk offload"
              f" — GPU 2 GiB / CPU {_cpu_gb} GiB / everything else on disk"
              f" (slow — offload dir: {_offload_dir})")
        try:
            pipe = PClass.from_pretrained(**_with_quant(dict(
                pretrained_model_name_or_path=model_name,
                torch_dtype=torch_dtype,
                device_map='balanced',
                max_memory={0: '2GiB', 'cpu': f'{_cpu_gb}GiB'},
                offload_folder=_offload_dir,
                offload_buffers=True,
                low_cpu_mem_usage=True,
            )))
        except (TypeError, ValueError):
            pipe = PClass.from_pretrained(**_with_quant(dict(
                pretrained_model_name_or_path=model_name,
                torch_dtype=torch_dtype, low_cpu_mem_usage=True)))
            pipe.enable_sequential_cpu_offload()
            try:
                from accelerate.hooks import AlignDevicesHook
                for comp in (pipe.components.values() if hasattr(pipe, 'components') else []):
                    for m in (comp.modules() if hasattr(comp, 'modules') else []):
                        hook = getattr(m, '_hf_hook', None)
                        if isinstance(hook, AlignDevicesHook) and hasattr(hook, 'offload_buffers'):
                            hook.offload_buffers = True
            except Exception:
                pass
        except (RuntimeError, MemoryError):
            # Even the last-resort strategy OOM'd — release any partial pipeline
            # so the failure doesn't leave leaked VRAM for the next request.
            _clear_mem()
            raise
        _report_loaded(pipe, f"pure disk (CPU {_cpu_gb} GiB cap)")
        return pipe
    finally:
        # Restore (no-op unless something else changed it mid-load).
        sys.setswitchinterval(_old_interval)


# =============================================================================
# Frame interpolation model loading
# =============================================================================

def _load_rife(model_name: str, device: str):
    """Load RIFE frame interpolation model."""
    try:
        # Try rife-ncnn-vulkan first (subprocess)
        import shutil
        if shutil.which('rife-ncnn-vulkan'):
            return ('rife_ncnn', None)
    except Exception:
        pass
    # Fallback: use IFNet from a HF repo
    try:
        from diffusers import IFPipeline  # noqa – just checking if diffusers has it
    except ImportError:
        pass
    return ('rife_hf', model_name)


# =============================================================================
# Generation logic
# =============================================================================

def _build_call_kwargs(request: VideoGenerationRequest) -> dict:
    kw = {}
    if request.prompt:
        kw['prompt'] = request.prompt
    if request.negative_prompt:
        kw['negative_prompt'] = request.negative_prompt
    if request.num_inference_steps:
        kw['num_inference_steps'] = request.num_inference_steps
    if request.guidance_scale:
        kw['guidance_scale'] = request.guidance_scale
    if request.num_frames:
        kw['num_frames'] = request.num_frames
    if request.width and request.height:
        kw['width'] = request.width
        kw['height'] = request.height
    if request.seed is not None:
        import torch
        kw['generator'] = torch.Generator().manual_seed(request.seed)
    return kw


def _apply_camera_motion(kw: dict, camera_motion: str):
    """Inject camera motion hint into pipeline kwargs (model-dependent)."""
    # CogVideoX supports camera_motion natively
    if camera_motion:
        kw['camera_motion'] = camera_motion


def _resolve_character_inputs(request) -> tuple[List[str], List[str]]:
    """Return (flat_image_list, name_list) from any combination of request fields."""
    images: List[str] = []
    names: List[str] = []

    # 1. Expand named saved profiles
    if request.character_profiles:
        try:
            from codai.api.characters import resolve_character_profiles
            images += resolve_character_profiles(request.character_profiles)
            names += list(request.character_profiles)
        except Exception:
            pass

    # 2. Named character slots [{name, images:[...]}, ...]
    if request.characters:
        for slot in request.characters:
            slot_imgs = slot.get('images') or []
            images += slot_imgs
            if slot.get('name'):
                names.append(slot['name'])

    # 3. Legacy flat list
    if request.character_references:
        images += list(request.character_references)
        if request.character_names:
            names += list(request.character_names)

    return images, names


def _pipeline_supports_ip_adapter(pipe) -> bool:
    """Return True if the pipeline's __call__ accepts ip_adapter_image."""
    import inspect
    try:
        sig = inspect.signature(pipe.__call__)
        return 'ip_adapter_image' in sig.parameters
    except Exception:
        return False


def _apply_character_refs(kw: dict, character_references: List[str], strength: float,
                           names: Optional[List[str]] = None, pipe=None):
    """Apply character reference images to pipeline kwargs when supported."""
    if not character_references:
        return
    # Only inject ip_adapter_image if the pipeline actually accepts it.
    # Models like WanPipeline don't support IP-Adapter; for those we rely
    # solely on the character-name text prompt hint added by the caller.
    if pipe is not None and not _pipeline_supports_ip_adapter(pipe):
        return
    imgs = [_pil_from_b64(r) for r in character_references]
    kw['ip_adapter_image'] = imgs[0] if len(imgs) == 1 else imgs
    kw['ip_adapter_scale'] = strength


def _unload_video_loras(pipe):
    """Remove any LoRA adapters so a cached pipeline is clean for the next request."""
    try:
        if hasattr(pipe, 'unload_lora_weights'):
            pipe.unload_lora_weights()
    except Exception as e:
        print(f"  [video][lora] unload failed: {e}")
    try:
        pipe._coderai_active_loras = ()
    except Exception:
        pass


def _present_adapters(pipe) -> set:
    """Adapter names actually registered on the pipe's PEFT-capable components.

    After load_lora_weights, an adapter only appears here if the LoRA state dict
    had keys matching that component. An SD/SDXL UNet LoRA loaded onto a video DiT
    (e.g. WanTransformer3DModel) matches nothing, so its name never shows up.
    """
    present = set()
    for attr in ('transformer', 'transformer_2', 'unet', 'prior',
                 'text_encoder', 'text_encoder_2'):
        comp = getattr(pipe, attr, None)
        pc = getattr(comp, 'peft_config', None)
        if pc:
            try:
                present |= set(pc.keys())
            except Exception:
                pass
    return present


def _lora_signature(loras) -> tuple:
    """Normalized identity of a requested LoRA set: ((model, name, weight), ...).

    Two requests with the same signature need the same adapters at the same
    weights, so the already-loaded adapters can be reused as-is.
    """
    if not loras:
        return ()
    sig = []
    for i, lora in enumerate(loras):
        model = getattr(lora, 'model', None) or (lora.get('model') if isinstance(lora, dict) else None)
        if not model:
            continue
        name = (getattr(lora, 'name', None) if not isinstance(lora, dict) else lora.get('name')) or f"lora_{i}"
        w = getattr(lora, 'weight', None) if not isinstance(lora, dict) else lora.get('weight')
        sig.append((str(model), str(name), float(w if w is not None else 1.0)))
    return tuple(sig)


def _sync_video_loras(pipe, loras) -> None:
    """Make the pipeline's active LoRA adapters match `loras`, reusing what is
    already loaded when the request asks for the exact same set.

    The video model stays cached across clips, and consecutive clips of one match
    typically request the identical fighter+environment LoRAs. Reloading them from
    disk every clip is wasted I/O + fusion latency, so we cache the active set on
    the pipe object (`_coderai_active_loras`) and only swap when it actually
    changes. A different set (or no LoRAs) triggers a clean unload + reload.
    """
    desired = _lora_signature(loras)
    active = getattr(pipe, '_coderai_active_loras', ())
    if desired == active:
        if desired:
            print(f"  [video][lora] reusing active adapters {[s[1] for s in desired]} (no reload)")
        return
    # Set changed → drop whatever is loaded, then load the new set (if any).
    if active:
        _unload_video_loras(pipe)
    if not desired:
        return
    if not hasattr(pipe, 'load_lora_weights'):
        print("  [video][lora] pipeline does not support LoRA — skipping")
        return
    # Remember this request as the active set even if some adapters turn out
    # incompatible, so identical follow-up clips don't re-attempt the same load.
    pipe._coderai_active_loras = desired
    model_cls = type(getattr(pipe, 'transformer', None)
                      or getattr(pipe, 'unet', None) or pipe).__name__
    loaded = []  # (name, weight) that actually registered on the model
    before = _present_adapters(pipe)
    # Wan2.2 A14B is a two-expert MoE — a LoRA loaded only into `transformer`
    # leaves the low-noise expert (transformer_2) un-adapted, so the concept
    # fades out as denoising hands over to it. Load into both when present.
    _has_t2 = getattr(pipe, 'transformer_2', None) is not None
    for model, name, w in desired:
        try:
            pipe.load_lora_weights(model, adapter_name=name)
        except Exception as e:
            print(f"  [video][lora] failed to load '{name}': {e}")
            continue
        if _has_t2:
            try:
                pipe.load_lora_weights(model, adapter_name=name,
                                       load_into_transformer_2=True)
            except Exception as e:
                print(f"  [video][lora] '{name}' not loaded into transformer_2 "
                      f"(low-noise expert): {e}")
        now = _present_adapters(pipe)
        if name in now and name not in before:
            loaded.append((name, w))
            before = now
        else:
            # Keys matched nothing on this architecture (e.g. an SD/SDXL image
            # LoRA on a Wan video transformer). Skip it; it would not affect output.
            print(f"  [video][lora] '{name}' has no weights matching {model_cls} "
                  f"— skipping (incompatible LoRA for this video model)")
    if not loaded:
        print("  [video][lora] no compatible adapters — generating without LoRA")
        _unload_video_loras(pipe)
        pipe._coderai_active_loras = desired  # _unload reset it; restore for dedup
        return
    try:
        pipe.set_adapters([n for n, _ in loaded], [w for _, w in loaded])
        print(f"  [video][lora] applied: {[n for n, _ in loaded]} "
              f"weights={[w for _, w in loaded]}")
    except Exception as e:
        print(f"  [video][lora] could not activate LoRA weights: {e}")
        _unload_video_loras(pipe)
        pipe._coderai_active_loras = desired


def _run_pipeline(pipe, kw: dict):
    result = pipe(**kw)
    # NB: `getattr(result, 'frames', None) or result[0]` is WRONG — when .frames
    # is a numpy array, `bool(array)` raises "truth value of an array ... is
    # ambiguous". Check for None explicitly instead of using `or`.
    frames_raw = getattr(result, 'frames', None)
    if frames_raw is None:
        frames_raw = result[0]
    # Unwrap a batch dimension: frames can be [[frame, frame, ...]] (list of
    # one video) or a numpy array shaped (batch, frames, H, W, C).
    if isinstance(frames_raw, list):
        if frames_raw and isinstance(frames_raw[0], list):
            return frames_raw[0]
        return frames_raw
    try:
        import numpy as _np
        if isinstance(frames_raw, _np.ndarray) and frames_raw.ndim == 5:
            frames_raw = frames_raw[0]  # drop batch dim
    except Exception:
        pass
    return list(frames_raw)


def _wan_in_channels(pipe):
    """Input-channel count of a Wan pipeline's transformer patch-embed.

    16 → text-to-video (t2v) transformer; 36 → image-to-video (i2v, which packs
    16 noise + 16 image + 4 mask latent channels). Returns None if undetermined.
    """
    t = getattr(pipe, 'transformer', None)
    if t is None:
        return None
    cfg = getattr(t, 'config', None)
    ic = getattr(cfg, 'in_channels', None) if cfg is not None else None
    if ic:
        try:
            return int(ic)
        except Exception:
            pass
    # Authoritative fallback: the conv weight shape [out, in, ...].
    w = getattr(getattr(t, 'patch_embedding', None), 'weight', None)
    try:
        if w is not None and w.ndim >= 2:
            return int(w.shape[1])
    except Exception:
        pass
    return None


def _maybe_t2v_fallback(pipe, kw, mode):
    """If an image-to-video Wan pipeline is backed by a text-to-video transformer
    (16 in-channels), rebuild it as a plain WanPipeline that REUSES the same
    components and run as t2v with the keyframe dropped — instead of crashing on a
    16-vs-36 channel mismatch. Returns (pipe_to_use, mode).

    The rebuilt t2v view shares the transformer/VAE/text-encoder objects, so any
    fused acceleration and per-request LoRAs applied to that transformer carry
    over unchanged. The view is cached on the i2v pipe so repeated clips reuse it
    (keeping _sync_video_loras' adapter dedup intact across a match).
    """
    if type(pipe).__name__ != 'WanImageToVideoPipeline' or 'image' not in kw:
        return pipe, mode
    if _wan_in_channels(pipe) != 16:
        return pipe, mode  # genuine i2v model — leave it alone
    view = getattr(pipe, '_coderai_t2v_view', None)
    if view is None:
        try:
            import inspect
            from diffusers import WanPipeline
            allowed = set(inspect.signature(WanPipeline.__init__).parameters)
            comps = {k: v for k, v in pipe.components.items()
                     if k in allowed and v is not None}
            view = WanPipeline(**comps)
            if getattr(pipe, '_coderai_accel', None) is not None:
                view._coderai_accel = pipe._coderai_accel
            pipe._coderai_t2v_view = view
            print("  [video] model is text-to-video (transformer in_channels=16) but "
                  "i2v/ti2v was requested — running t2v, keyframe ignored")
        except Exception as e:
            print(f"  [video] t2v fallback failed ({e}); attempting i2v as requested")
            return pipe, mode
    kw.pop('image', None)
    return view, 't2v'


def _maybe_i2v_fallback(pipe, kw, mode):
    """Reverse of _maybe_t2v_fallback: if a t2v request (no init image) lands on a
    WanPipeline whose transformer is actually image-to-video (36 in-channels), the
    t2v forward would mismatch (it builds 16-channel input). Rebuild as a
    WanImageToVideoPipeline reusing the same components (image_encoder/processor
    are optional and simply absent here) and seed it with a neutral gray frame so
    the prompt still drives the clip. Returns (pipe_to_use, mode).

    This is a graceful degrade — an i2v model without a real keyframe can't lock a
    first frame, so the neutral seed yields essentially prompt-driven output.
    """
    if type(pipe).__name__ != 'WanPipeline' or 'image' in kw:
        return pipe, mode
    if _wan_in_channels(pipe) != 36:
        return pipe, mode  # genuine t2v model — fine as-is
    view = getattr(pipe, '_coderai_i2v_view', None)
    if view is None:
        try:
            import inspect
            from diffusers import WanImageToVideoPipeline
            allowed = set(inspect.signature(WanImageToVideoPipeline.__init__).parameters)
            comps = {k: v for k, v in pipe.components.items()
                     if k in allowed and v is not None}
            view = WanImageToVideoPipeline(**comps)
            if getattr(pipe, '_coderai_accel', None) is not None:
                view._coderai_accel = pipe._coderai_accel
            pipe._coderai_i2v_view = view
            print("  [video] model is image-to-video (transformer in_channels=36) but "
                  "t2v was requested — seeding a neutral frame (prompt-driven)")
        except Exception as e:
            print(f"  [video] i2v fallback failed ({e}); attempting t2v as requested")
            return pipe, mode
    from PIL import Image as _Image
    w = int(kw.get('width') or 512)
    h = int(kw.get('height') or 512)
    kw['image'] = _Image.new('RGB', (w, h), (128, 128, 128))
    return view, 'ti2v'


def _generate_video(pipe, request: VideoGenerationRequest):
    mode = request.mode or ('i2v' if (request.image or request.init_image)
                             else 'v2v' if request.video else 't2v')
    fps = request.fps or 8
    kw = _build_call_kwargs(request)
    # Acceleration/distillation defaults (Lightning / Lightx2v): when the model has
    # a fused distill LoRA, default to its low step-count / guidance instead of the
    # standard 25 steps / 7.5 CFG. The request always wins if it set these — note
    # _build_call_kwargs only populates them when the request specified them, so
    # setdefault below correctly leaves an explicit request value untouched.
    _accel = getattr(pipe, '_coderai_accel', None)
    # Only trust the preset's low step-count / guidance when the distill LoRA
    # actually fused. If it didn't (e.g. the Wan2.2 low-noise expert never got its
    # LoRA, or the ref failed to load), running 4 steps un-distilled collapses the
    # video to a solid colour — so fall back to a safe step count instead.
    _accel_fused = getattr(pipe, '_coderai_accel_fused', None)
    if _accel and _accel_fused is not False:
        from codai.models.acceleration import accel_call_defaults
        for _k, _v in accel_call_defaults(_accel).items():
            kw.setdefault(_k, _v)
    elif _accel and _accel_fused is False:
        print("  [video][accel] distill LoRA not fused — ignoring the preset's "
              "low step count and using safe defaults (25 steps) to avoid a "
              "collapsed/solid-colour result")
    kw.setdefault('num_inference_steps', 25)
    kw.setdefault('guidance_scale', 7.5)
    kw.setdefault('num_frames', 16)

    _vid_progress_reset(kw['num_inference_steps'])

    _tid = task_registry.register(
        "video", title=(request.prompt or mode or "")[:80],
        model=getattr(request, 'model', '') or '', total=kw['num_inference_steps'])
    task_registry.start(_tid)

    def _vid_step_cb(pipe, step_index, timestep, callback_kwargs):
        # Cooperative cancellation: abort at the next step boundary if cancelled.
        task_registry.raise_if_cancelled(_tid)
        # Cooperative pause: block here while the user has paused this task.
        task_registry.wait_if_paused(_tid)
        task_registry.step(_tid, step_index + 1)
        _vid_progress_step(step_index + 1)
        # Mid-generation thermal checkpoint: pause between denoise steps if the
        # CPU/GPU went over the limit during this (multi-minute) generation.
        try:
            from codai.models.thermal import checkpoint as _thermal_checkpoint
            _thermal_checkpoint(context="video-gen")
        except Exception:
            pass
        return callback_kwargs

    try:
        kw['callback_on_step_end'] = _vid_step_cb
    except Exception:
        pass

    _apply_camera_motion(kw, request.camera_motion)

    char_images, char_names = _resolve_character_inputs(request)
    if char_images:
        _apply_character_refs(kw, char_images, request.character_strength or 0.8, char_names, pipe=pipe)
        # Always prepend character names to prompt for text conditioning
        # (sole mechanism for pipelines that don't support IP-Adapter)
        if char_names and kw.get('prompt'):
            names_hint = ', '.join(char_names)
            kw['prompt'] = f"{names_hint}. {kw['prompt']}"

    init_src = request.init_image or request.image

    if mode == 'i2v' and init_src:
        kw['image'] = _pil_from_b64(init_src)
        kw.pop('prompt', None)  # SVD doesn't take text

    elif mode == 'ti2v' and init_src:
        kw['image'] = _pil_from_b64(init_src)
        # prompt stays — model uses both

    elif mode == 'interp':
        if not init_src or not request.end_image:
            raise ValueError("interp mode requires both init_image and end_image")
        kw['image'] = _pil_from_b64(init_src)
        kw['image_end'] = _pil_from_b64(request.end_image)
        kw.pop('prompt', None)

    elif mode == 'v2v' and request.video:
        kw['video'] = _decode_b64_or_url(request.video)
        if request.strength is not None:
            kw['strength'] = request.strength

    # Graceful pipeline/model fallbacks for Wan, in case the requested mode and the
    # model's actual capability disagree (selecting the pipeline class by request
    # mode can mismatch the transformer's input channels):
    #   * ti2v/i2v request on a t2v model (16-ch) → run t2v, drop the keyframe.
    #   * t2v request on an i2v model (36-ch)     → run i2v with a neutral seed frame.
    # Both rebuild a sibling pipeline that REUSES the same components, so fused
    # acceleration and per-request LoRAs on the shared transformer carry over.
    pipe, mode = _maybe_t2v_fallback(pipe, kw, mode)
    pipe, mode = _maybe_i2v_fallback(pipe, kw, mode)

    # Per-request LoRA adapters (e.g. per-character identity LoRAs). Sync the
    # pipeline's adapters to this request's set, REUSING them if identical to the
    # previous clip's (common within a match) and only swapping when they differ.
    # Left loaded after the run so the next clip with the same set pays nothing.
    _sync_video_loras(pipe, getattr(request, 'loras', None))
    try:
        frames = _run_pipeline(pipe, kw)
    except TaskCancelled:
        _vid_progress_done()
        raise  # global handler finishes the task (cancelled) + returns HTTP 499
    except Exception as e:
        task_registry.finish(_tid, "error", str(e)[:200])
        _vid_progress_done()
        raise
    _vid_progress_done()
    task_registry.finish(_tid, "done")
    return frames, fps


# =============================================================================
# Post-processing helpers
# =============================================================================

def _postprocess_video(mp4_bytes: bytes, request: VideoGenerationRequest,
                       http_request, temp_paths: list) -> bytes:
    """Apply upscale / interpolation / audio / dialog steps to a raw mp4 blob."""
    path = _tmp_write(mp4_bytes, '.mp4')
    temp_paths.append(path)

    if request.upscale_output:
        path = _ffmpeg_upscale(path, request.upscale_factor or 2, temp_paths)

    if request.interpolate_output and request.fps_multiplier:
        path = _rife_interpolate(path, request.fps_multiplier, temp_paths)

    if request.add_audio:
        path = _add_audio_to_video(path, request, temp_paths)

    if request.dialogs:
        path = _process_dialogs(path, request.dialogs,
                                request.lip_sync_method or 'wav2lip', temp_paths)

    if request.generate_subtitles or request.burn_subtitles:
        path = _add_subtitles(path, request, temp_paths)

    with open(path, 'rb') as f:
        return f.read()


def _tmp_write(data: bytes, ext: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _ffmpeg_upscale(path: str, factor: int, temps: list) -> str:
    out = tempfile.mktemp(suffix='_up.mp4')
    temps.append(out)
    scale = f"scale=iw*{factor}:ih*{factor}:flags=lanczos"
    cmd = ['ffmpeg', '-y', '-i', path, '-vf', scale, '-c:a', 'copy', out]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        import logging
        logging.getLogger(__name__).warning(
            "ffmpeg upscale failed (rc=%d): %s", r.returncode, r.stderr.decode(errors='replace')
        )
        return path  # fallback to original if ffmpeg fails
    return out


def _rife_interpolate(path: str, multiplier: int, temps: list) -> str:
    out = tempfile.mktemp(suffix='_rife.mp4')
    temps.append(out)
    import logging, shutil
    _log = logging.getLogger(__name__)
    if shutil.which('rife-ncnn-vulkan'):
        frames_dir = tempfile.mkdtemp()
        out_dir = tempfile.mkdtemp()
        temps += [frames_dir, out_dir]
        r = subprocess.run(['ffmpeg', '-y', '-i', path, f'{frames_dir}/%08d.png'],
                           capture_output=True)
        if r.returncode != 0:
            _log.warning("ffmpeg frame extraction failed: %s", r.stderr.decode(errors='replace'))
        else:
            r = subprocess.run(['rife-ncnn-vulkan', '-i', frames_dir, '-o', out_dir,
                                '-m', 'rife-v4'], capture_output=True)
            if r.returncode != 0:
                _log.warning("rife-ncnn-vulkan failed: %s", r.stderr.decode(errors='replace'))
            else:
                r = subprocess.run(['ffmpeg', '-y', '-r', str(multiplier * 8), '-i',
                                    f'{out_dir}/%08d.png', '-c:v', 'libx264', out],
                                   capture_output=True)
                if r.returncode != 0:
                    _log.warning("ffmpeg reassembly failed: %s", r.stderr.decode(errors='replace'))
                elif os.path.exists(out):
                    return out
    # Simple ffmpeg minterpolate fallback
    cmd = ['ffmpeg', '-y', '-i', path, '-filter:v',
           f'minterpolate=fps={multiplier * 8}', '-c:a', 'copy', out]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        _log.warning("ffmpeg minterpolate failed: %s", r.stderr.decode(errors='replace'))
        return path
    return out


def _add_audio_to_video(path: str, request: VideoGenerationRequest,
                         temps: list) -> str:
    out = tempfile.mktemp(suffix='_audio.mp4')
    temps.append(out)

    if request.audio_file:
        audio_path = _tmp_write(_decode_b64_or_url(request.audio_file), '.wav')
        temps.append(audio_path)
    elif request.tts_text:
        audio_path = _generate_tts(request.tts_text, request.tts_voice,
                                     request.tts_speed or 1.0, temps)
    else:
        return path  # nothing to add

    if not audio_path or not os.path.exists(audio_path):
        return path

    cmd = ['ffmpeg', '-y', '-i', path, '-i', audio_path,
           '-c:v', 'copy', '-c:a', 'aac', '-shortest', out]
    r = subprocess.run(cmd, capture_output=True)
    return out if r.returncode == 0 else path


def _generate_tts(text: str, voice: Optional[str], speed: float,
                   temps: list) -> Optional[str]:
    """Quick TTS using kokoro or edge-tts — returns wav file path."""
    try:
        import edge_tts, asyncio as _aio
        voice_id = voice or 'en-US-JennyNeural'
        out = tempfile.mktemp(suffix='.mp3')
        temps.append(out)
        tts = edge_tts.Communicate(text, voice_id, rate=f"+{int((speed - 1) * 100)}%")
        _aio.get_event_loop().run_until_complete(tts.save(out))
        return out
    except ImportError:
        pass
    try:
        from kokoro import KPipeline
        import soundfile as sf, numpy as np
        pipe = KPipeline(lang_code='a')
        audio, sr = pipe(text, voice=voice or 'af_sky', speed=speed)
        out = tempfile.mktemp(suffix='.wav')
        temps.append(out)
        sf.write(out, np.concatenate(audio), sr)
        return out
    except ImportError:
        pass
    return None


def _get_audio_duration(path: str) -> float:
    """Return audio/video duration in seconds via ffprobe."""
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _generate_tts_for_line(line: CharacterDialogLine, temps: list) -> Optional[str]:
    """Generate TTS for a single dialog line, using the voice profile's reference audio if available."""
    voice = line.voice
    text = line.text
    speed = line.speed or 1.0
    lang = line.lang

    # Try to load voice profile reference audio first (for kokoro/RVC cloning)
    ref_audio = None
    if voice:
        try:
            from codai.api.voice_clone import _load_voice, _voice_path
            meta = _load_voice(voice)
            audio_file = meta.get('audio_file') or meta.get('audio_path')
            if audio_file and os.path.isfile(audio_file):
                ref_audio = audio_file
        except Exception:
            pass

    # edge_tts with voice id
    try:
        import edge_tts, asyncio as _aio
        voice_id = voice if (voice and not ref_audio) else (
            f"{lang.split('-')[0]}-" if lang else 'en-'
        ) + 'US-JennyNeural'
        if not voice or ref_audio:
            voice_id = 'en-US-JennyNeural'
        out = tempfile.mktemp(suffix='.mp3')
        temps.append(out)
        tts = edge_tts.Communicate(text, voice_id, rate=f"+{int((speed - 1) * 100)}%")
        _aio.get_event_loop().run_until_complete(tts.save(out))
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return out
    except Exception:
        pass

    # kokoro with optional reference voice
    try:
        from kokoro import KPipeline
        import soundfile as sf, numpy as np
        lang_code = 'a'
        if lang:
            lang_code = lang.split('-')[0][:1].lower()
        pipe = KPipeline(lang_code=lang_code)
        kokoro_voice = voice if (voice and not ref_audio) else 'af_sky'
        audio, sr = pipe(text, voice=kokoro_voice, speed=speed)
        out = tempfile.mktemp(suffix='.wav')
        temps.append(out)
        sf.write(out, np.concatenate(audio), sr)
        return out
    except Exception:
        pass

    return None


def _mix_dialog_audio(clips: list, temps: list) -> Optional[str]:
    """
    Mix a list of (start_time_sec, audio_path) clips into one audio file.
    Uses ffmpeg adelay + amix. Returns path to mixed audio or None.
    """
    if not clips:
        return None
    if len(clips) == 1:
        return clips[0][1]

    # Build complex filter: delay each stream, then amix
    filter_parts = []
    inputs = []
    for i, (start_sec, apath) in enumerate(clips):
        inputs += ['-i', apath]
        delay_ms = int(start_sec * 1000)
        filter_parts.append(f'[{i}]adelay={delay_ms}|{delay_ms}[a{i}]')

    mix_inputs = ''.join(f'[a{i}]' for i in range(len(clips)))
    filter_parts.append(f'{mix_inputs}amix=inputs={len(clips)}:duration=longest:dropout_transition=0[out]')
    filter_str = ';'.join(filter_parts)

    out = tempfile.mktemp(suffix='_mixed.wav')
    temps.append(out)
    cmd = ['ffmpeg', '-y'] + inputs + ['-filter_complex', filter_str, '-map', '[out]', out]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0 and os.path.exists(out):
        return out

    # Fallback: simple concatenation (ignore timing)
    import logging
    logging.getLogger(__name__).warning(
        "ffmpeg amix failed (rc=%d), falling back to concat: %s",
        r.returncode, r.stderr.decode(errors='replace'))
    list_path = tempfile.mktemp(suffix='.txt')
    temps.append(list_path)
    with open(list_path, 'w') as f:
        for _, apath in clips:
            f.write(f"file '{apath}'\n")
    out2 = tempfile.mktemp(suffix='_cat.wav')
    temps.append(out2)
    r2 = subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                         '-i', list_path, out2], capture_output=True)
    return out2 if r2.returncode == 0 else None


def _apply_lipsync(video_path: str, audio_path: str, method: str, temps: list) -> str:
    """Apply lip sync to video using wav2lip or sadtalker. Returns new video path."""
    import logging, shutil
    _log = logging.getLogger(__name__)
    out = tempfile.mktemp(suffix='_lipsync.mp4')
    temps.append(out)

    if method == 'wav2lip':
        wav2lip_bin = shutil.which('wav2lip') or shutil.which('Wav2Lip')
        if wav2lip_bin:
            cmd = [wav2lip_bin, '--face', video_path, '--audio', audio_path, '--outfile', out]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode == 0 and os.path.exists(out):
                return out
            _log.warning("wav2lip failed (rc=%d): %s", r.returncode, r.stderr.decode(errors='replace'))
        else:
            # Try wav2lip Python API
            try:
                import inference as wav2lip_inference  # noqa
                wav2lip_inference.main(face=video_path, audio=audio_path, outfile=out)
                if os.path.exists(out):
                    return out
            except Exception as e:
                _log.warning("wav2lip Python API failed: %s", e)

    elif method == 'sadtalker':
        sadtalker_bin = shutil.which('sadtalker')
        if sadtalker_bin:
            cmd = [sadtalker_bin, '--driven_audio', audio_path, '--source_video', video_path,
                   '--result_dir', os.path.dirname(out)]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode == 0:
                # Find output file
                out_dir = os.path.dirname(out)
                for f in sorted(os.listdir(out_dir)):
                    if f.endswith('.mp4'):
                        return os.path.join(out_dir, f)
            _log.warning("sadtalker failed (rc=%d): %s", r.returncode, r.stderr.decode(errors='replace'))

    # Fallback: just mux audio onto video without lip sync
    _log.warning("Lip sync unavailable (%s not found/working), merging audio only", method)
    out_fallback = tempfile.mktemp(suffix='_nosync.mp4')
    temps.append(out_fallback)
    cmd = ['ffmpeg', '-y', '-i', video_path, '-i', audio_path,
           '-c:v', 'copy', '-c:a', 'aac', '-shortest', out_fallback]
    r = subprocess.run(cmd, capture_output=True)
    return out_fallback if r.returncode == 0 else video_path


def _process_dialogs(path: str, dialogs: list, lip_sync_method: str, temps: list) -> str:
    """
    Generate TTS for each dialog line, mix with correct timing, apply lip sync to full video.
    Returns new video path.
    """
    import logging
    _log = logging.getLogger(__name__)

    # First pass: generate TTS audio for each line and calculate timing
    clips = []  # (start_sec, audio_path)
    cursor = 0.0
    for line in dialogs:
        if not line.text.strip():
            continue
        audio_path = _generate_tts_for_line(line, temps)
        if not audio_path or not os.path.exists(audio_path):
            _log.warning("TTS generation failed for dialog line: %r", line.text[:40])
            continue

        if line.start_time is not None:
            start = float(line.start_time)
        else:
            start = cursor

        duration = _get_audio_duration(audio_path)
        clips.append((start, audio_path))
        cursor = start + duration + 0.1  # small gap between sequential lines

    if not clips:
        return path

    # Mix all clips into one audio track
    mixed_audio = _mix_dialog_audio(clips, temps)
    if not mixed_audio or not os.path.exists(mixed_audio):
        return path

    # Determine if any line wants lip sync
    wants_lip_sync = any(getattr(line, 'lip_sync', True) for line in dialogs if line.text.strip())

    if wants_lip_sync and lip_sync_method:
        return _apply_lipsync(path, mixed_audio, lip_sync_method, temps)

    # No lip sync — just mux audio
    out = tempfile.mktemp(suffix='_dialog.mp4')
    temps.append(out)
    cmd = ['ffmpeg', '-y', '-i', path, '-i', mixed_audio,
           '-c:v', 'copy', '-c:a', 'aac', '-shortest', out]
    r = subprocess.run(cmd, capture_output=True)
    return out if r.returncode == 0 else path


def _add_subtitles(path: str, request: VideoGenerationRequest, temps: list) -> str:
    """Transcribe video audio → subtitles, optionally burn them in."""
    try:
        import whisper
    except ImportError:
        return path  # skip if whisper not available

    srt_path = _whisper_transcribe(path, request.subtitle_language,
                                    request.whisper_model, temps)
    if not srt_path:
        return path

    if request.translate_subtitles and request.subtitle_target_lang:
        srt_path = _translate_srt(srt_path, request.subtitle_target_lang, temps)

    if request.burn_subtitles:
        out = tempfile.mktemp(suffix='_sub.mp4')
        temps.append(out)
        # Use ASS-style subtitle filter for better styling
        style = request.subtitle_style or 'default'
        vf = f"subtitles={srt_path}"
        if style == 'karaoke':
            vf = f"ass={srt_path}"
        cmd = ['ffmpeg', '-y', '-i', path, '-vf', vf, '-c:a', 'copy', out]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            return out

    return path


def _whisper_transcribe(video_path: str, language: Optional[str],
                         model_name: Optional[str], temps: list) -> Optional[str]:
    try:
        import whisper as _whisper
        model = _whisper.load_model(model_name or 'base')
        result = model.transcribe(video_path, language=language)
        srt_path = tempfile.mktemp(suffix='.srt')
        temps.append(srt_path)
        with open(srt_path, 'w') as f:
            for i, seg in enumerate(result['segments'], 1):
                def _fmt(t):
                    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
                    return f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')
                f.write(f"{i}\n{_fmt(seg['start'])} --> {_fmt(seg['end'])}\n{seg['text'].strip()}\n\n")
        return srt_path
    except Exception:
        return None


def _translate_srt(srt_path: str, target_lang: str, temps: list) -> str:
    """Translate SRT using argostranslate or fall back to original."""
    try:
        import argostranslate.package, argostranslate.translate
        with open(srt_path) as f:
            content = f.read()
        lines = content.split('\n')
        translated = []
        for line in lines:
            if line and not line[0].isdigit() and '-->' not in line:
                line = argostranslate.translate.translate(line, 'en', target_lang)
            translated.append(line)
        out = tempfile.mktemp(suffix='.srt')
        temps.append(out)
        with open(out, 'w') as f:
            f.write('\n'.join(translated))
        return out
    except Exception:
        return srt_path


# =============================================================================
# Progress endpoint
# =============================================================================

@router.get("/v1/video/progress", summary="Video generation progress")
async def get_video_progress():
    """Return current video generation step progress including speed."""
    elapsed = time.monotonic() - _vid_progress["started_at"] if _vid_progress["active"] else 0.0
    return {
        "current":  _vid_progress["current"],
        "total":    _vid_progress["total"],
        "active":   _vid_progress["active"],
        "phase":    _vid_progress.get("phase", "idle"),
        "model":    _vid_progress.get("model", ""),
        "pct":      int(_vid_progress["current"] / _vid_progress["total"] * 100)
                    if _vid_progress["total"] > 0 else 0,
        "it_per_s": _vid_progress["it_per_s"],
        "elapsed":  round(elapsed, 1),
    }


# =============================================================================
# Main generation endpoint
# =============================================================================

@router.post("/v1/video/generations", response_model=VideoGenerationResponse, summary="Generate video")
async def video_generations(request: VideoGenerationRequest,
                             http_request: Request = None):
    """
    Generate video.

    Modes (request.mode):
      t2v   – text-to-video
      i2v   – image-to-video (init_image required)
      v2v   – video-to-video (video required)
      ti2v  – text + image → video (prompt is primary driver)
      interp – frame interpolation (init_image + end_image required)
    """
    if not request.model:
        raise HTTPException(status_code=400, detail="model is required")
    _vid_progress_loading(request.model)

    # Infer mode from inputs if not set
    if not request.mode or request.mode == 't2v':
        if request.init_image or request.image:
            request.mode = 'ti2v' if request.prompt else 'i2v'
        elif request.end_image:
            request.mode = 'interp'
        elif request.video:
            request.mode = 'v2v'

    # Run in a thread: request_model may block while waiting for a busy text
    # model to finish its in-flight request before evicting it.  Blocking here
    # on the event loop would deadlock that very request.
    # Reserve VRAM for any per-request LoRA adapters so eviction frees enough
    # headroom for base weights + adapters before the pipeline loads.
    _lora_extra_gb = 0.0
    if getattr(request, 'loras', None):
        # Resolve any id/url/inline-file LoRA refs to concrete local paths now, in
        # the async handler, so a missing blob / unknown name returns a clean 400
        # before we touch the model. Downstream code then reads lora.model as usual.
        from codai.api.loras import resolve_request_loras
        resolve_request_loras(request.loras)
        try:
            _lora_extra_gb = multi_model_manager._lora_vram_gb(request.loras)
        except Exception:
            _lora_extra_gb = 0.0

    model_info = await asyncio.to_thread(
        multi_model_manager.request_model, request.model, "video",
        extra_vram_gb=_lora_extra_gb)
    model_name = model_info.get('model_name')
    if not model_name:
        err = model_info.get('error', f"Model '{request.model}' not found")
        raise HTTPException(status_code=404, detail=err)

    model_key = model_info['model_key']
    pipe = model_info.get('model_object')
    # Always define _model_cfg — it's used later regardless of whether we load now.
    _model_cfg = model_info.get('config') or {}

    # Refuse to load onto a poisoned CUDA context — it would just re-assert.
    if getattr(multi_model_manager, 'cuda_context_poisoned', False):
        raise HTTPException(status_code=503, detail=(
            "CUDA context corrupted by an earlier device-side assert "
            f"({multi_model_manager.cuda_poison_reason}). Restart coderai to recover."))

    # Always resolve the device up front — the OOM-retry path below also needs
    # it, and when the pipeline is already cached the `if pipe is None` block
    # (where it used to be set) is skipped, which caused an UnboundLocalError
    # ("cannot access local variable 'device'") in the retry.
    device = _derive_device()

    if pipe is None:
        _offload = _model_cfg.get('offload_strategy') or None
        # 'auto' (the default) means "let coderai pick from available VRAM" — it is
        # NOT a diffusers strategy, and passing it through lands on the full-GPU
        # path that then disk-thrashes. Normalise it to None so the VRAM check
        # below decides between full-GPU and balanced GPU+CPU.
        if _offload == 'auto':
            _offload = None
        # Auto-select "balanced" strategy when the model (including runtime
        # reserve: KV/activation spike, VAE decode) exceeds available VRAM even
        # after eviction. Going straight to "balanced" (GPU-first + CPU spill)
        # avoids the expensive OOM → free → reload cycle that wastes ~1 hr of
        # shard reloading only to end up at the same place. The GPU cap is 80%
        # of free VRAM (or the per-model balanced_gpu_percent if configured) so
        # we leave breathing room for activations and the decode spike.
        if _offload is None:
            try:
                import torch as _t
                if _t.cuda.is_available():
                    _free_v, _ = _t.cuda.mem_get_info()
                    _free_gb = _free_v / 1e9
                    # `_get_model_used_vram_gb` is the *measured total* footprint —
                    # it already includes the runtime/activation reserve AND the
                    # fused acceleration LoRA (it's measured after fusion). So do
                    # NOT re-add those (that over-counts and wrongly forces the
                    # slow balanced+disk path). Per-request LoRAs are extra.
                    _base_gb = multi_model_manager._get_model_used_vram_gb(
                        model_key, model_name)
                    # full-GPU only needs the WEIGHTS to fit at load time (the
                    # bundled ~runtime reserve is a gen-time allowance, and the
                    # full-GPU path has its own OOM→offload fallback). Keep a
                    # headroom margin so a model that *marginally* fits uses the
                    # much faster full-GPU strategy rather than balanced+disk.
                    _need_gb = _base_gb + _lora_extra_gb
                    _margin = 2.5  # ≈ the bundled runtime reserve
                    if _base_gb > 0 and _free_gb < (_need_gb - _margin):
                        _gpu_pct = float(_model_cfg.get('balanced_gpu_percent') or 80)
                        print(f"  VRAM well short for full-GPU load "
                              f"({_need_gb:.1f} GB measured need + LoRA; "
                              f"{_free_gb:.1f} GB free) — auto-selecting balanced "
                              f"strategy ({_gpu_pct:.0f}% GPU + CPU spill)")
                        _offload = 'balanced'
                    else:
                        print(f"  Full-GPU load looks viable "
                              f"({_need_gb:.1f} GB measured need, {_free_gb:.1f} GB "
                              f"free) — using full GPU (it falls back to offload on OOM)")
            except Exception:
                pass
        # Snapshot free VRAM so we can record the real footprint after load.
        _vram_before = multi_model_manager.vram_before_load()
        from codai.tasks import loading_task
        try:
            with loading_task(model_name, model_type="video"):
                pipe = await asyncio.get_event_loop().run_in_executor(
                    None, _load_video_pipeline, model_name, device, request.mode, _offload, _model_cfg)
        except Exception as e:
            multi_model_manager._mark_cuda_poisoned_if_fatal(e)
            if getattr(multi_model_manager, 'cuda_context_poisoned', False):
                raise HTTPException(status_code=503, detail=(
                    "CUDA context corrupted (device-side assert) while loading the "
                    "video model. Restart coderai to recover. "
                    f"Original error: {str(e).splitlines()[0]}"))
            # Self-heal: a failed load from a (possibly stale/corrupt) pipeline
            # cache should drop the cache and rebuild once rather than wedging.
            _retried_fresh = False
            try:
                from codai.models import pipeline_cache as _pcache
                if _pcache.enabled() and _pcache.valid(_pcache.path(model_name, _model_cfg)):
                    print(f"  [pipeline-cache] load failed ({str(e).splitlines()[0]}) "
                          f"— invalidating cache and rebuilding")
                    _pcache.invalidate(model_name, _model_cfg)
                    with loading_task(model_name, model_type="video"):
                        pipe = await asyncio.get_event_loop().run_in_executor(
                            None, _load_video_pipeline, model_name, device,
                            request.mode, _offload, _model_cfg)
                    _retried_fresh = True
            except Exception:
                _retried_fresh = False
            if not _retried_fresh:
                raise HTTPException(status_code=500, detail=f"Failed to load video model: {e}")
        # Fuse any configured acceleration/distillation LoRA (Lightning / Lightx2v /
        # LCM) into the freshly loaded pipeline. Done once at load; cached pipes keep
        # it. No-op for sd.cpp pipes and when no acceleration is configured.
        try:
            from codai.models.acceleration import resolve_acceleration, apply_accel_to_pipeline
            _accel = resolve_acceleration(_model_cfg)
            _accel_is_sdcpp = False
            try:
                from stable_diffusion_cpp import StableDiffusion as _SDc
                _accel_is_sdcpp = isinstance(pipe, _SDc)
            except ImportError:
                pass
            if _accel and not _accel_is_sdcpp:
                print(f"  [video][accel] applying {_accel.get('preset')} "
                      f"(steps={_accel.get('steps')}, guidance={_accel.get('guidance_scale')})")
                apply_accel_to_pipeline(pipe, _accel)
        except Exception as _e:
            print(f"  [video][accel] skipped: {_e}")
        multi_model_manager.models[model_key] = pipe
        multi_model_manager.current_model_key = model_key
        # Record the real VRAM used. record_vram_delta only persists when no
        # used_vram_gb is configured (it writes the separate measured_vram_gb).
        try:
            multi_model_manager.record_vram_delta(model_key, _vram_before)
        except Exception:
            pass

    if getattr(request, 'disable_safety_checker', False):
        _disable_safety_checker(pipe)

    _is_sdcpp_video = False
    try:
        from stable_diffusion_cpp import StableDiffusion as _SD
        _is_sdcpp_video = isinstance(pipe, _SD)
    except ImportError:
        pass

    try:
        if _is_sdcpp_video:
            frames, fps = await asyncio.get_event_loop().run_in_executor(
                None, _generate_sdcpp_video, pipe, request, _model_cfg)
        else:
            frames, fps = await asyncio.get_event_loop().run_in_executor(
                None, _generate_video, pipe, request)
    except Exception as e:
        _vid_progress_done()
        _err_str = str(e).lower()
        _is_oom = "out of memory" in _err_str or ("cuda" in _err_str and "memory" in _err_str)
        # On OOM: free the GPU pipeline, reload with CPU offload, and retry once.
        if _is_oom and not _is_sdcpp_video:
            import gc, torch as _torch, psutil as _ps
            # Choose retry offload strategy based on current free RAM.
            # enable_model_cpu_offload() loads ALL weights into RAM first — if RAM
            # is already tight that triggers another OOM kill.  Use sequential+disk
            # offload instead when free RAM < 2× the pipeline's last known RAM usage.
            _free_ram_gb = _ps.virtual_memory().available / 1e9
            _retry_strategy = 'model' if _free_ram_gb > 20 else 'sequential'
            print(f"Video generation OOM — freeing pipeline, retrying with "
                  f"'{_retry_strategy}' offload "
                  f"({_free_ram_gb:.1f} GB RAM free)…")
            multi_model_manager.models.pop(model_key, None)
            multi_model_manager.model_pools.pop(model_key, None)
            # Fully release the OOM'd pipeline's VRAM before reloading — a naive
            # .to('cpu') silently fails on quantized/device_map pipelines, which
            # is what leaks VRAM and snowballs into the OOM death spiral.
            _free_pipeline_vram(pipe)
            pipe = None
            try:
                pipe = await asyncio.get_event_loop().run_in_executor(
                    None, _load_video_pipeline, model_name, device, request.mode,
                    _retry_strategy, _model_cfg)
                multi_model_manager.models[model_key] = pipe
                multi_model_manager.current_model_key = model_key
                frames, fps = await asyncio.get_event_loop().run_in_executor(
                    None, _generate_video, pipe, request)
            except Exception as e2:
                multi_model_manager.models.pop(model_key, None)
                multi_model_manager.model_pools.pop(model_key, None)
                # Release the failed retry pipeline too, so the NEXT request
                # starts from clean VRAM instead of inheriting the leak.
                _free_pipeline_vram(pipe)
                pipe = None
                raise HTTPException(status_code=500, detail=f"Video generation failed (OOM retry): {e2}")
        else:
            # Non-OOM failure: evict the cached pipeline so the next request
            # attempts a clean reload rather than reusing a broken object.
            multi_model_manager.models.pop(model_key, None)
            multi_model_manager.model_pools.pop(model_key, None)
            _free_pipeline_vram(pipe)
            pipe = None
            raise HTTPException(status_code=500, detail=f"Video generation failed: {e}")

    # Encode raw frames to MP4 (per-model output quality via CRF when configured).
    try:
        import imageio, numpy as np
        frame_np = [np.array(f) for f in frames]
        _crf = (_model_cfg or {}).get('output_crf')
        try:
            _crf = int(_crf) if _crf is not None else None
        except (TypeError, ValueError):
            _crf = None
        mp4_bytes = _frames_to_mp4(frame_np, fps, crf=_crf)
    except ImportError:
        raise HTTPException(status_code=500,
                            detail="imageio[ffmpeg] required: pip install imageio[ffmpeg]")

    # Post-processing pipeline (upscale, audio, subtitles, …)
    temps = []
    try:
        needs_post = any([
            request.upscale_output,
            request.interpolate_output,
            request.add_audio,
            bool(request.dialogs),
            request.generate_subtitles,
            request.burn_subtitles,
        ])
        if needs_post:
            mp4_bytes = await asyncio.get_event_loop().run_in_executor(
                None, _postprocess_video, mp4_bytes, request, http_request, temps)
    finally:
        for t in temps:
            try:
                if os.path.isfile(t):
                    os.unlink(t)
                elif os.path.isdir(t):
                    import shutil
                    shutil.rmtree(t, ignore_errors=True)
            except Exception:
                pass

    result = _save_file(mp4_bytes, 'mp4', http_request)

    try:
        from codai.api.archive import archive_manager
        asyncio.get_event_loop().create_task(asyncio.to_thread(
            archive_manager.save_generation,
            "video", "/v1/video/generations",
            request.model,
            request.prompt or "",
            {
                "mode": request.mode,
                "num_frames": request.num_frames,
                "fps": request.fps,
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
            },
            [(mp4_bytes, "mp4")],
        ))
    except Exception:
        pass

    return VideoGenerationResponse(created=int(time.time()), data=[result])


# =============================================================================
# Video upscale endpoint
# =============================================================================

@router.post("/v1/video/upscale", summary="Upscale a video")
async def video_upscale(request: VideoUpscaleRequest, http_request: Request = None):
    """
    Upscale a video using ffmpeg lanczos or Real-ESRGAN.
    The model field can be 'realesrgan' or any registered video_upscaling model.
    """
    raw = _decode_b64_or_url(request.video)
    temps = []
    try:
        in_path = _tmp_write(raw, '.mp4')
        temps.append(in_path)
        out_path = await asyncio.get_event_loop().run_in_executor(
            None, _ffmpeg_upscale, in_path, request.upscale_factor or 2, temps)
        with open(out_path, 'rb') as f:
            out_bytes = f.read()
    finally:
        for t in temps:
            try:
                os.unlink(t)
            except Exception:
                pass

    result = _save_file(out_bytes, 'mp4', http_request)
    return {"created": int(time.time()), "data": [result]}


# =============================================================================
# Subtitle generation endpoint
# =============================================================================

@router.post("/v1/video/subtitle", summary="Subtitle / caption a video")
async def video_subtitle(request: VideoSubtitleRequest, http_request: Request = None):
    """
    Generate subtitles for a video.
    Returns SRT/VTT text or a URL to the video with burned-in subtitles.
    """
    raw = _decode_b64_or_url(request.video)
    temps = []
    try:
        in_path = _tmp_write(raw, '.mp4')
        temps.append(in_path)

        srt_path = await asyncio.get_event_loop().run_in_executor(
            None, _whisper_transcribe, in_path, request.language, None, temps)
        if not srt_path:
            raise HTTPException(status_code=500,
                                detail="Whisper not installed: pip install openai-whisper")

        if request.translate and request.target_lang:
            srt_path = await asyncio.get_event_loop().run_in_executor(
                None, _translate_srt, srt_path, request.target_lang, temps)

        if request.burn:
            out_path = tempfile.mktemp(suffix='_sub.mp4')
            temps.append(out_path)
            cmd = ['ffmpeg', '-y', '-i', in_path,
                   '-vf', f'subtitles={srt_path}',
                   '-c:a', 'copy', out_path]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                raise HTTPException(status_code=500,
                                    detail=f"ffmpeg subtitle burn failed: {r.stderr.decode()}")
            with open(out_path, 'rb') as f:
                out_bytes = f.read()
            result = _save_file(out_bytes, 'mp4', http_request)
            return {"created": int(time.time()), "data": [result]}

        # Return raw subtitle text
        with open(srt_path) as f:
            srt_text = f.read()
        return {"created": int(time.time()), "data": [{"text": srt_text, "format": "srt"}]}

    finally:
        for t in temps:
            try:
                os.unlink(t)
            except Exception:
                pass


# =============================================================================
# Frame interpolation endpoint
# =============================================================================

@router.post("/v1/video/interpolate", summary="Interpolate video frames")
async def video_interpolate(request: VideoInterpolateRequest, http_request: Request = None):
    """
    Increase video FPS via frame interpolation.
    Supports rife-ncnn-vulkan (if installed) or ffmpeg minterpolate fallback.
    """
    temps = []
    try:
        if request.video:
            raw = _decode_b64_or_url(request.video)
            in_path = _tmp_write(raw, '.mp4')
            temps.append(in_path)
        elif request.init_image and request.end_image:
            # Build a 2-frame video from the two images, then interpolate
            from PIL import Image as PILImage
            import numpy as np, imageio
            img1 = _pil_from_b64(request.init_image)
            img2 = _pil_from_b64(request.end_image)
            in_path = tempfile.mktemp(suffix='.mp4')
            temps.append(in_path)
            imageio.mimsave(in_path, [np.array(img1), np.array(img2)],
                            fps=2, codec='libx264')
        else:
            raise HTTPException(status_code=400,
                                detail="Provide either video or init_image + end_image")

        mult = request.fps_multiplier or 2
        out_path = await asyncio.get_event_loop().run_in_executor(
            None, _rife_interpolate, in_path, mult, temps)
        with open(out_path, 'rb') as f:
            out_bytes = f.read()

    finally:
        for t in temps:
            try:
                os.unlink(t)
            except Exception:
                pass

    result = _save_file(out_bytes, 'mp4', http_request)
    return {"created": int(time.time()), "data": [result]}


# =============================================================================
# Video dubbing endpoint
# =============================================================================

@router.post("/v1/video/dub", summary="Dub a video")
async def video_dub(request: VideoDubRequest, http_request: Request = None):
    """
    Translate and re-dub a video.
    Pipeline: Whisper → translate → TTS → merge audio → (optional) lip sync.
    """
    raw = _decode_b64_or_url(request.video)
    temps = []
    try:
        in_path = _tmp_write(raw, '.mp4')
        temps.append(in_path)

        # 1. Transcribe
        srt_path = await asyncio.get_event_loop().run_in_executor(
            None, _whisper_transcribe, in_path, request.source_lang, None, temps)
        if not srt_path:
            raise HTTPException(status_code=500, detail="Whisper not available")

        # 2. Translate subtitles
        if request.target_lang:
            srt_path = await asyncio.get_event_loop().run_in_executor(
                None, _translate_srt, srt_path, request.target_lang, temps)

        # 3. Generate dubbed audio from translated text
        with open(srt_path) as f:
            srt_content = f.read()
        plain_text = '\n'.join(
            line for line in srt_content.split('\n')
            if line and not line[0].isdigit() and '-->' not in line
        )
        audio_path = await asyncio.get_event_loop().run_in_executor(
            None, _generate_tts, plain_text, None, 1.0, temps)

        if not audio_path:
            raise HTTPException(status_code=500, detail="TTS generation failed (install edge-tts or kokoro)")

        # 4. Merge dubbed audio with video
        out_path = tempfile.mktemp(suffix='_dubbed.mp4')
        temps.append(out_path)
        cmd = ['ffmpeg', '-y', '-i', in_path, '-i', audio_path,
               '-map', '0:v', '-map', '1:a',
               '-c:v', 'copy', '-c:a', 'aac', '-shortest', out_path]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            raise HTTPException(status_code=500,
                                detail=f"Audio merge failed: {r.stderr.decode()}")

        # 5. Burn subtitles if requested
        if request.burn_subtitles:
            sub_out = tempfile.mktemp(suffix='_sub.mp4')
            temps.append(sub_out)
            cmd2 = ['ffmpeg', '-y', '-i', out_path,
                    '-vf', f'subtitles={srt_path}',
                    '-c:a', 'copy', sub_out]
            r2 = subprocess.run(cmd2, capture_output=True)
            if r2.returncode == 0:
                out_path = sub_out

        with open(out_path, 'rb') as f:
            out_bytes = f.read()

    finally:
        for t in temps:
            try:
                os.unlink(t)
            except Exception:
                pass

    result = _save_file(out_bytes, 'mp4', http_request)
    return {"created": int(time.time()), "data": [result]}