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
2D ↔ 3D conversion and 3D model generation endpoints.

Endpoints:
  POST /v1/images/to3d      – 2D image → stereo/anaglyph/depth-map/mesh (GLB)
  POST /v1/images/from3d    – 3D model (GLB/OBJ) → rendered 2D image
  POST /v1/video/to3d       – 2D video → 3D video (frame-by-frame)
  POST /v1/video/from3d     – 3D model → turntable video
  POST /v1/3d/generate      – text / image → 3D model (GLB)
"""

import asyncio
import base64
import io
import os
import subprocess
import tempfile
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

router = APIRouter()

global_args = None
global_file_path = None


def set_global_args(args):
    global global_args
    global_args = args


def set_global_file_path(path):
    global global_file_path
    global_file_path = path


# =============================================================================
# Shared helpers
# =============================================================================

def _decode_b64(data: str) -> bytes:
    if data.startswith("data:"):
        _, enc = data.split(",", 1)
        return base64.b64decode(enc)
    return base64.b64decode(data)


def _build_url(filename: str, http_request) -> str:
    from codai.api.urlutils import build_file_url
    return build_file_url(filename, http_request)


def _save_output(data: bytes, ext: str, http_request) -> dict:
    filename = f"{uuid.uuid4().hex}.{ext}"
    if global_file_path:
        os.makedirs(global_file_path, exist_ok=True)
        out_path = os.path.join(global_file_path, filename)
        with open(out_path, 'wb') as f:
            f.write(data)
        return {"url": _build_url(filename, http_request)}
    return {f"b64_{ext}": base64.b64encode(data).decode()}


def _derive_device() -> str:
    if global_args:
        for attr in ('image_vulkan_device', 'vulkan_device'):
            d = getattr(global_args, attr, None)
            if d is not None:
                return f"cuda:{d}"
    return "cuda:0"


def _img_to_bytes(img_pil, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img_pil.save(buf, format=fmt)
    return buf.getvalue()


# =============================================================================
# Depth estimation
# =============================================================================

def _configured_depth_model_id() -> str:
    """Return the configured spatial/depth model ID, or empty string if none."""
    try:
        from codai.models.manager import multi_model_manager
        if multi_model_manager.spatial_models:
            return multi_model_manager.spatial_models[0]
    except Exception:
        pass
    return ""


def _estimate_depth(img_pil) -> np.ndarray:
    """Return depth array normalised to 0–1 (float32). Higher = closer."""
    device = _derive_device()

    # Prefer transformers depth-estimation pipeline (use configured model if available)
    try:
        from transformers import pipeline as hf_pipeline
        model_id = _configured_depth_model_id() or None
        pipe = hf_pipeline("depth-estimation", model=model_id, device=device)
        result = pipe(img_pil)
        depth = np.array(result['depth'], dtype=np.float32)
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            return (depth - d_min) / (d_max - d_min)
        return np.zeros_like(depth)
    except Exception:
        pass

    # MiDaS small fallback
    try:
        import torch
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        model.eval().to(device)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        inp = transforms(np.array(img_pil)).to(device)
        with torch.no_grad():
            depth = model(inp).squeeze().cpu().numpy()
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            return (depth - d_min) / (d_max - d_min)
        return np.zeros_like(depth)
    except Exception as e:
        raise RuntimeError(f"Depth estimation failed (no backend available): {e}")


# =============================================================================
# Stereo helpers
# =============================================================================

def _depth_to_stereo(img_pil, depth: np.ndarray, max_shift: int = 20):
    """Return (left_pil, right_pil) with horizontal parallax shift."""
    import cv2
    img_arr = np.array(img_pil)
    h, w = img_arr.shape[:2]
    depth_r = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    shift_map = (depth_r * max_shift).astype(np.float32)
    x_coords = np.arange(w, dtype=np.float32)
    y_coords = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords)
    map_x_l = (xx + shift_map).clip(0, w - 1)
    map_x_r = (xx - shift_map).clip(0, w - 1)
    left_arr = cv2.remap(img_arr, map_x_l, yy, cv2.INTER_LINEAR)
    right_arr = cv2.remap(img_arr, map_x_r, yy, cv2.INTER_LINEAR)
    from PIL import Image as PILImage
    return PILImage.fromarray(left_arr), PILImage.fromarray(right_arr)


def _make_anaglyph(left_pil, right_pil):
    """Red-cyan anaglyph: red channel from left eye, green+blue from right eye."""
    from PIL import Image as PILImage
    left = np.array(left_pil.convert("RGB"), dtype=np.uint8)
    right = np.array(right_pil.convert("RGB"), dtype=np.uint8)
    ana = np.zeros_like(left)
    ana[:, :, 0] = left[:, :, 0]
    ana[:, :, 1] = right[:, :, 1]
    ana[:, :, 2] = right[:, :, 2]
    return PILImage.fromarray(ana)


# =============================================================================
# Depth → GLB mesh
# =============================================================================

def _depth_to_mesh_glb(img_pil, depth: np.ndarray, depth_scale: float = 0.3) -> bytes:
    """Build a textured grid mesh from depth + colour, export as GLB."""
    import trimesh
    import cv2
    from PIL import Image as PILImage

    img_arr = np.array(img_pil.convert("RGB"))
    h, w = img_arr.shape[:2]
    max_dim = 256
    scale = min(max_dim / w, max_dim / h, 1.0)
    nw, nh = max(int(w * scale), 2), max(int(h * scale), 2)
    img_s = cv2.resize(img_arr, (nw, nh), interpolation=cv2.INTER_AREA)
    depth_s = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_AREA)

    ys, xs = np.mgrid[0:nh, 0:nw]
    x_norm = (xs / (nw - 1)) * 2 - 1
    y_norm = (ys / (nh - 1)) * 2 - 1
    z = depth_s * depth_scale

    vertices = np.column_stack([x_norm.ravel(), -y_norm.ravel(), z.ravel()])

    faces = []
    for row in range(nh - 1):
        for col in range(nw - 1):
            tl = row * nw + col
            tr = tl + 1
            bl = tl + nw
            br = bl + 1
            faces.append([tl, bl, tr])
            faces.append([tr, bl, br])
    faces = np.array(faces)

    uvs = np.column_stack([xs.ravel() / (nw - 1), ys.ravel() / (nh - 1)])
    texture_pil = PILImage.fromarray(img_s)
    material = trimesh.visual.texture.SimpleMaterial(image=texture_pil)
    visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)

    glb_buf = io.BytesIO()
    mesh.export(glb_buf, file_type='glb')
    return glb_buf.getvalue()


# =============================================================================
# 3D → 2D rendering
# =============================================================================

def _build_camera_pose(eye: np.ndarray) -> np.ndarray:
    target = np.zeros(3)
    up = np.array([0.0, 1.0, 0.0])
    z_axis = (eye - target)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis_norm = np.linalg.norm(x_axis)
    if x_axis_norm < 1e-6:
        # Eye is directly above/below — use a different up
        up = np.array([0.0, 0.0, 1.0])
        x_axis = np.cross(up, z_axis)
        x_axis_norm = np.linalg.norm(x_axis)
    x_axis /= x_axis_norm
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


def _render_mesh_to_png(mesh_bytes: bytes, mesh_fmt: str,
                         distance: float, elevation: float, azimuth: float,
                         width: int, height: int) -> bytes:
    import trimesh
    import pyrender
    import math

    mesh = trimesh.load(io.BytesIO(mesh_bytes), file_type=mesh_fmt)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.apply_translation(-mesh.centroid)
    max_ext = max(mesh.extents) if max(mesh.extents) > 0 else 1.0
    mesh.apply_scale(1.0 / max_ext)

    scene = pyrender.Scene(bg_color=[0.12, 0.12, 0.14, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    el = math.radians(elevation)
    az = math.radians(azimuth)
    eye = np.array([
        distance * math.cos(el) * math.sin(az),
        distance * math.sin(el),
        distance * math.cos(el) * math.cos(az),
    ], dtype=np.float32)

    pose = _build_camera_pose(eye)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=pose)

    renderer = pyrender.OffscreenRenderer(width, height)
    color, _ = renderer.render(scene)
    renderer.delete()

    from PIL import Image as PILImage
    return _img_to_bytes(PILImage.fromarray(color))


def _render_turntable_mp4(mesh_bytes: bytes, mesh_fmt: str,
                           n_frames: int, fps: int,
                           elevation: float, distance: float,
                           width: int, height: int) -> bytes:
    import trimesh
    import pyrender
    import math
    import imageio

    mesh = trimesh.load(io.BytesIO(mesh_bytes), file_type=mesh_fmt)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.apply_translation(-mesh.centroid)
    max_ext = max(mesh.extents) if max(mesh.extents) > 0 else 1.0
    mesh.apply_scale(1.0 / max_ext)

    renderer = pyrender.OffscreenRenderer(width, height)
    scene = pyrender.Scene(bg_color=[0.12, 0.12, 0.14, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
    el = math.radians(elevation)

    frames_list = []
    for i in range(n_frames):
        az = 2 * math.pi * i / n_frames
        eye = np.array([
            distance * math.cos(el) * math.sin(az),
            distance * math.sin(el),
            distance * math.cos(el) * math.cos(az),
        ], dtype=np.float32)
        pose = _build_camera_pose(eye)
        cam_node = scene.add(camera, pose=pose)
        light_node = scene.add(light, pose=pose)
        color, _ = renderer.render(scene)
        scene.remove_node(cam_node)
        scene.remove_node(light_node)
        frames_list.append(color)

    renderer.delete()

    out_path = tempfile.mktemp(suffix='_3d_turntable.mp4')
    imageio.mimsave(out_path, frames_list, fps=fps, codec='libx264', quality=8)
    with open(out_path, 'rb') as f:
        data = f.read()
    os.unlink(out_path)
    return data


# =============================================================================
# 3D model generation backends
# =============================================================================

def _generate_3d_triposr(image_bytes: bytes) -> bytes:
    import torch
    from PIL import Image as PILImage
    from tsr.system import TSR

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model = model.to(device)
    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    with torch.no_grad():
        scene_codes = model([img], device=device)
    meshes = model.extract_mesh(scene_codes, resolution=256)
    glb_buf = io.BytesIO()
    meshes[0].export(glb_buf, file_type='glb')
    return glb_buf.getvalue()


def _generate_3d_shape_e(prompt: Optional[str], image_bytes: Optional[bytes],
                          steps: int) -> bytes:
    import torch
    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_model, load_config
    from shap_e.util.notebooks import decode_latent_mesh

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm = load_model('transmitter', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    if image_bytes:
        from PIL import Image as PILImage
        cond_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        model = load_model('image300M', device=device)
        latents = sample_latents(
            batch_size=1, model=model, diffusion=diffusion,
            guidance_scale=3.0,
            model_kwargs=dict(images=[cond_img]),
            progress=True, clip_denoised=True, use_fp16=True,
            use_karras=True, karras_steps=steps,
            sigma_min=1e-3, sigma_max=160, s_churn=0,
        )
    else:
        model = load_model('text300M', device=device)
        latents = sample_latents(
            batch_size=1, model=model, diffusion=diffusion,
            guidance_scale=15.0,
            model_kwargs=dict(texts=[prompt or "a 3D object"]),
            progress=True, clip_denoised=True, use_fp16=True,
            use_karras=True, karras_steps=steps,
            sigma_min=1e-3, sigma_max=160, s_churn=0,
        )

    mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
    glb_buf = io.BytesIO()
    mesh.write_glb(glb_buf)
    return glb_buf.getvalue()


def _generate_3d_from_depth(image_bytes: bytes) -> bytes:
    """Fallback: depth-based mesh from an image when no dedicated 3D model is available."""
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    depth = _estimate_depth(img)
    return _depth_to_mesh_glb(img, depth)


# =============================================================================
# Video 2D→3D frame processing
# =============================================================================

def _process_video_to_3d(video_bytes: bytes, method: str, max_shift: int) -> bytes:
    import shutil
    from PIL import Image as PILImage

    temps = []
    try:
        in_path = tempfile.mktemp(suffix='.mp4')
        temps.append(in_path)
        with open(in_path, 'wb') as f:
            f.write(video_bytes)

        frames_dir = tempfile.mkdtemp()
        temps.append(frames_dir)
        subprocess.run(['ffmpeg', '-y', '-i', in_path, f'{frames_dir}/%08d.png'],
                       capture_output=True, check=True)

        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate', '-of', 'default=nw=1:nk=1', in_path],
            capture_output=True, text=True)
        fps_str = probe.stdout.strip() or '25/1'
        num, den = fps_str.split('/')
        fps = float(num) / float(den)

        out_dir = tempfile.mkdtemp()
        temps.append(out_dir)

        for fname in sorted(os.listdir(frames_dir)):
            fpath = os.path.join(frames_dir, fname)
            img = PILImage.open(fpath).convert("RGB")
            depth = _estimate_depth(img)

            if method == "depth":
                result_img = PILImage.fromarray((depth * 255).astype(np.uint8))
            elif method == "anaglyph":
                left, right = _depth_to_stereo(img, depth, max_shift)
                result_img = _make_anaglyph(left, right)
            else:  # stereo
                left, right = _depth_to_stereo(img, depth, max_shift)
                w, h = left.size
                result_img = PILImage.new("RGB", (w * 2, h))
                result_img.paste(left, (0, 0))
                result_img.paste(right, (w, 0))

            result_img.save(os.path.join(out_dir, fname))

        out_path = tempfile.mktemp(suffix='_3d.mp4')
        temps.append(out_path)
        subprocess.run(
            ['ffmpeg', '-y', '-framerate', str(fps), '-i', f'{out_dir}/%08d.png',
             '-i', in_path, '-map', '0:v', '-map', '1:a?',
             '-c:v', 'libx264', '-c:a', 'copy', '-shortest', out_path],
            capture_output=True, check=True)

        with open(out_path, 'rb') as f:
            return f.read()

    finally:
        for t in temps:
            try:
                if os.path.isdir(t):
                    shutil.rmtree(t)
                else:
                    os.unlink(t)
            except Exception:
                pass


# =============================================================================
# Endpoint: Image 2D → 3D
# =============================================================================

class ImageTo3DRequest(BaseModel):
    image: str
    method: Optional[str] = "stereo"       # stereo | anaglyph | depth | mesh
    max_shift: Optional[int] = 20
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


@router.post("/v1/images/to3d", summary="Image to 3D model")
async def image_to_3d(request: ImageTo3DRequest, http_request: Request = None):
    """Convert a 2D image to a 3D representation.

    Methods:
      stereo   – side-by-side left/right pair (PNG)
      anaglyph – red-cyan 3D image (PNG)
      depth    – normalised depth map (PNG)
      mesh     – textured 3D mesh (GLB) built from depth
    """
    raw = _decode_b64(request.image)
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(raw)).convert("RGB")
    method = request.method or "stereo"

    try:
        depth = await asyncio.get_event_loop().run_in_executor(None, _estimate_depth, img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Depth estimation failed: {e}")

    try:
        if method == "depth":
            out_img = PILImage.fromarray((depth * 255).astype(np.uint8))
            result = _save_output(_img_to_bytes(out_img), "png", http_request)

        elif method == "anaglyph":
            left, right = await asyncio.get_event_loop().run_in_executor(
                None, _depth_to_stereo, img, depth, request.max_shift or 20)
            ana = await asyncio.get_event_loop().run_in_executor(
                None, _make_anaglyph, left, right)
            result = _save_output(_img_to_bytes(ana), "png", http_request)

        elif method == "mesh":
            glb = await asyncio.get_event_loop().run_in_executor(
                None, _depth_to_mesh_glb, img, depth)
            result = _save_output(glb, "glb", http_request)

        else:  # stereo (default)
            left, right = await asyncio.get_event_loop().run_in_executor(
                None, _depth_to_stereo, img, depth, request.max_shift or 20)
            w, h = left.size
            composite = PILImage.new("RGB", (w * 2, h))
            composite.paste(left, (0, 0))
            composite.paste(right, (w, 0))
            result = _save_output(_img_to_bytes(composite), "png", http_request)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"2D→3D conversion failed: {e}")

    return {"created": int(time.time()), "data": [result], "method": method}


# =============================================================================
# Endpoint: 3D model → 2D image
# =============================================================================

class ImageFrom3DRequest(BaseModel):
    model_data: str                          # base64 GLB / OBJ
    format: Optional[str] = "glb"
    camera_distance: Optional[float] = 2.0
    camera_elevation: Optional[float] = 30.0
    camera_azimuth: Optional[float] = 45.0
    width: Optional[int] = 512
    height: Optional[int] = 512
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


@router.post("/v1/images/from3d", summary="Render a 3D model to an image")
async def image_from_3d(request: ImageFrom3DRequest, http_request: Request = None):
    """Render a 3D model (GLB/OBJ) to a 2D PNG image from a specified camera angle."""
    raw = _decode_b64(request.model_data)
    try:
        png_bytes = await asyncio.get_event_loop().run_in_executor(
            None, _render_mesh_to_png, raw,
            request.format or "glb",
            request.camera_distance or 2.0,
            request.camera_elevation or 30.0,
            request.camera_azimuth or 45.0,
            request.width or 512,
            request.height or 512,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"3D render failed: {e}")

    result = _save_output(png_bytes, "png", http_request)
    return {"created": int(time.time()), "data": [result]}


# =============================================================================
# Endpoint: Video 2D → 3D
# =============================================================================

class VideoTo3DRequest(BaseModel):
    video: str
    method: Optional[str] = "anaglyph"     # stereo | anaglyph | depth
    max_shift: Optional[int] = 15
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


@router.post("/v1/video/to3d", summary="Video to 3D model")
async def video_to_3d(request: VideoTo3DRequest, http_request: Request = None):
    """Convert a 2D video to a 3D video frame-by-frame.

    Methods:
      stereo   – side-by-side left/right video
      anaglyph – red-cyan 3D video
      depth    – greyscale depth video
    """
    raw = _decode_b64(request.video)
    method = request.method or "anaglyph"

    try:
        out_bytes = await asyncio.get_event_loop().run_in_executor(
            None, _process_video_to_3d, raw, method, request.max_shift or 15)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"ffmpeg error: {e.stderr.decode()[:200]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video 2D→3D failed: {e}")

    result = _save_output(out_bytes, "mp4", http_request)
    return {"created": int(time.time()), "data": [result], "method": method}


# =============================================================================
# Endpoint: 3D model → turntable video
# =============================================================================

class VideoFrom3DRequest(BaseModel):
    model_data: str                          # base64 GLB
    format: Optional[str] = "glb"
    frames: Optional[int] = 36
    fps: Optional[int] = 12
    camera_elevation: Optional[float] = 20.0
    camera_distance: Optional[float] = 2.5
    width: Optional[int] = 512
    height: Optional[int] = 512
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


@router.post("/v1/video/from3d", summary="Render a 3D model to a video")
async def video_from_3d(request: VideoFrom3DRequest, http_request: Request = None):
    """Render a 3D model as a 360° turntable video."""
    raw = _decode_b64(request.model_data)
    try:
        mp4_bytes = await asyncio.get_event_loop().run_in_executor(
            None, _render_turntable_mp4,
            raw, request.format or "glb",
            request.frames or 36, request.fps or 12,
            request.camera_elevation or 20.0, request.camera_distance or 2.5,
            request.width or 512, request.height or 512,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"3D→video render failed: {e}")

    result = _save_output(mp4_bytes, "mp4", http_request)
    return {"created": int(time.time()), "data": [result]}


# =============================================================================
# Endpoint: Generate 3D model
# =============================================================================

class Generate3DRequest(BaseModel):
    prompt: Optional[str] = None
    image: Optional[str] = None
    model: Optional[str] = None
    steps: Optional[int] = 64
    seed: Optional[int] = None
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


@router.post("/v1/3d/generate", summary="Generate a 3D model from a prompt")
async def generate_3d(request: Generate3DRequest, http_request: Request = None):
    """Generate a 3D model (GLB) from a text prompt and/or an image.

    Backend priority:
      1. TripoSR   – image-to-3D mesh (best quality, requires image)
      2. Shap-E    – text or image to 3D latent mesh
      3. Depth mesh – depth-map fallback (requires image, no ML 3D backend needed)
    """
    if not request.prompt and not request.image:
        raise HTTPException(status_code=400, detail="Either prompt or image is required")

    image_bytes = _decode_b64(request.image) if request.image else None

    # 1. TripoSR (image only)
    if image_bytes:
        try:
            glb = await asyncio.get_event_loop().run_in_executor(
                None, _generate_3d_triposr, image_bytes)
            result = _save_output(glb, "glb", http_request)
            return {"created": int(time.time()), "data": [result], "backend": "triposr"}
        except (ImportError, ModuleNotFoundError):
            pass
        except Exception:
            pass  # fall through to Shap-E

    # 2. Shap-E
    try:
        glb = await asyncio.get_event_loop().run_in_executor(
            None, _generate_3d_shape_e, request.prompt, image_bytes, request.steps or 64)
        result = _save_output(glb, "glb", http_request)
        return {"created": int(time.time()), "data": [result], "backend": "shap-e"}
    except (ImportError, ModuleNotFoundError):
        pass
    except Exception:
        pass

    # 3. Depth-based mesh fallback (image required)
    if image_bytes:
        try:
            glb = await asyncio.get_event_loop().run_in_executor(
                None, _generate_3d_from_depth, image_bytes)
            result = _save_output(glb, "glb", http_request)
            return {"created": int(time.time()), "data": [result], "backend": "depth-mesh"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"3D generation failed: {e}")

    raise HTTPException(
        status_code=500,
        detail=(
            "No 3D generation backend available. "
            "Install one of: pip install tsr  OR  pip install shap-e. "
            "For image-only mesh (no ML), provide an image — depth-based fallback will be used."
        )
    )
