"""
Face swap endpoint.

POST /v1/images/faceswap  — swap face in image or video frames
"""

import asyncio
import base64
import io
import os
import subprocess
import tempfile
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from PIL import Image
from pydantic import BaseModel, ConfigDict

from codai.api.images import save_image_response
from codai.platform_paths import default_insightface_model_path

router = APIRouter()

global_args = None
global_file_path = None

_INSWAPPER_MODEL_PATH = str(default_insightface_model_path())
_INSWAPPER_HF_REPO = 'deepinsight/inswapper'
_INSWAPPER_HF_FILE = 'inswapper_128.onnx'

_face_app = None      # FaceAnalysis singleton
_swapper = None       # INSwapper singleton


def set_global_args(args):
    global global_args
    global_args = args


def set_global_file_path(path):
    global global_file_path
    global_file_path = path


def _ensure_model():
    """Download inswapper_128.onnx if not present."""
    if os.path.exists(_INSWAPPER_MODEL_PATH):
        return
    os.makedirs(os.path.dirname(_INSWAPPER_MODEL_PATH), exist_ok=True)
    print(f'Downloading inswapper_128.onnx from HuggingFace…')
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=_INSWAPPER_HF_REPO,
            filename=_INSWAPPER_HF_FILE,
            local_dir=os.path.dirname(_INSWAPPER_MODEL_PATH),
        )
        if path != _INSWAPPER_MODEL_PATH:
            import shutil
            shutil.move(path, _INSWAPPER_MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f'Failed to download inswapper model: {e}')


def _get_face_app():
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def _get_swapper():
    global _swapper
    if _swapper is None:
        _ensure_model()
        from insightface.model_zoo import get_model
        _swapper = get_model(_INSWAPPER_MODEL_PATH, download=False)
        _swapper.prepare(ctx_id=0)
    return _swapper


def _decode_image(data: str) -> np.ndarray:
    """Decode base64 or data-URI image to BGR numpy array."""
    if data.startswith('data:'):
        _, b64 = data.split(',', 1)
        data = b64
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not decode image')
    return img


def _swap_faces(source_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
    """Swap all faces in target_img with the face from source_img."""
    app = _get_face_app()
    swapper = _get_swapper()

    src_faces = app.get(source_img)
    if not src_faces:
        raise ValueError('No face detected in source image')
    src_face = src_faces[0]

    tgt_faces = app.get(target_img)
    if not tgt_faces:
        return target_img  # no face to swap in target, return as-is

    result = target_img.copy()
    for tgt_face in tgt_faces:
        result = swapper.get(result, tgt_face, src_face, paste_back=True)
    return result


def _decode_b64_or_url(data: str) -> bytes:
    if data.startswith('data:'):
        _, b64 = data.split(',', 1)
        return base64.b64decode(b64)
    if data.startswith('http'):
        import urllib.request
        with urllib.request.urlopen(data, timeout=30) as r:
            return r.read()
    return base64.b64decode(data)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class FaceSwapRequest(BaseModel):
    source_face: str            # base64/data-URI image containing the source face
    target: str                 # base64/data-URI image OR video to swap into
    target_type: Optional[str] = 'image'   # 'image' or 'video'
    response_format: Optional[str] = 'url'
    model_config = ConfigDict(extra='allow')


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post('/v1/images/faceswap', summary="Swap faces between images")
async def faceswap(request: FaceSwapRequest, http_request: Request = None):
    """
    Swap the face from source_face into every face found in target.
    target_type: 'image' (default) or 'video'.
    """
    try:
        _ensure_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        src_img = _decode_image(request.source_face)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid source_face: {e}')

    if request.target_type == 'video':
        return await _faceswap_video(src_img, request, http_request)
    else:
        return await _faceswap_image(src_img, request, http_request)


async def _faceswap_image(src_img, request, http_request):
    try:
        tgt_img = _decode_image(request.target)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid target: {e}')

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _swap_faces, src_img, tgt_img)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Face swap failed: {e}')

    pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    img_data = save_image_response(pil_img, request.response_format, http_request)
    return {'created': int(time.time()), 'data': [img_data]}


async def _faceswap_video(src_img, request, http_request):
    raw = _decode_b64_or_url(request.target)
    temps = []
    try:
        # Write input video
        in_tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        in_tmp.write(raw); in_tmp.close()
        in_path = in_tmp.name
        temps.append(in_path)

        # Extract frames
        frames_dir = tempfile.mkdtemp()
        temps.append(frames_dir)
        subprocess.run(
            ['ffmpeg', '-y', '-i', in_path, f'{frames_dir}/%08d.png'],
            capture_output=True, check=True)

        # Get FPS for reassembly
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate', '-of', 'default=nw=1:nk=1', in_path],
            capture_output=True, text=True)
        fps_str = probe.stdout.strip() or '25/1'
        num, den = fps_str.split('/')
        fps = float(num) / float(den)

        # Swap faces in each frame
        frame_files = sorted(os.listdir(frames_dir))

        def _process_frames():
            app = _get_face_app()
            swapper = _get_swapper()
            src_faces = app.get(src_img)
            if not src_faces:
                raise ValueError('No face detected in source image')
            src_face = src_faces[0]
            for fname in frame_files:
                fpath = os.path.join(frames_dir, fname)
                frame = cv2.imread(fpath)
                if frame is None:
                    continue
                tgt_faces = app.get(frame)
                for tgt_face in tgt_faces:
                    frame = swapper.get(frame, tgt_face, src_face, paste_back=True)
                cv2.imwrite(fpath, frame)

        await asyncio.get_event_loop().run_in_executor(None, _process_frames)

        # Reassemble video (copy original audio)
        out_path = tempfile.mktemp(suffix='_swapped.mp4')
        temps.append(out_path)
        subprocess.run(
            ['ffmpeg', '-y', '-framerate', str(fps), '-i', f'{frames_dir}/%08d.png',
             '-i', in_path, '-map', '0:v', '-map', '1:a?',
             '-c:v', 'libx264', '-c:a', 'copy', '-shortest', out_path],
            capture_output=True, check=True)

        with open(out_path, 'rb') as f:
            out_bytes = f.read()

        if global_file_path:
            import uuid
            fname = f'{uuid.uuid4().hex}_swapped.mp4'
            fpath = os.path.join(global_file_path, fname)
            os.makedirs(global_file_path, exist_ok=True)
            with open(fpath, 'wb') as f:
                f.write(out_bytes)
            from codai.api.urlutils import build_file_url
            data = [{'url': build_file_url(fname, http_request)}]
        else:
            data = [{'b64_mp4': base64.b64encode(out_bytes).decode()}]

        return {'created': int(time.time()), 'data': data}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f'ffmpeg error: {e.stderr.decode()[:200]}')
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Video face swap failed: {e}')
    finally:
        import shutil
        for t in temps:
            try:
                if os.path.isdir(t):
                    shutil.rmtree(t)
                else:
                    os.unlink(t)
            except Exception:
                pass
