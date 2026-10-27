# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Cut the presenter out of their background.

`presenter.remove_background` asks for the person, not the frame they were
filmed in: a bubble over the B-roll looks composited when it is a circle of
someone's room, and cut out when it is the person alone.

What this produces is a **greyscale mask video** — white where the person is —
with the same frame count and rate as the presenter clip. That is exactly what
`render.composite_pip` already feeds to ffmpeg's `alphamerge`, which is why
nothing downstream needs to know a model was involved: with a shape mask the
bubble is a circle, with this mask it is a person.

Two engines, in order:

* a **TorchScript module** named by ``CODERAI_MATTING_MODEL`` — a Robust Video
  Matting export is the obvious one, and being recurrent it is the most stable
  over time. Loaded with ``torch.jit``, so no repo code is fetched or trusted.
* **torchvision's DeepLabV3**, whose COCO/VOC person class is enough for a
  bubble and which is already in the stack. No configuration, no new
  dependency; the weights come from the torch hub cache.

Both run on whatever device the caller's GPU lease covers and are dropped
immediately afterwards: this is a ~100 MB segmenter borrowing the GPU in the
middle of a compose job, not a model the manager should have to evict.
"""

import os
import subprocess
import threading
from typing import Callable, List, Optional

#: Inference runs on the short side of this box, not the canvas. A mask is a
#: silhouette — the extra pixels cost time and buy nothing, and the mask is
#: scaled back up by the writer anyway.
INFER_SIDE = 512

#: COCO-with-VOC-labels index of "person" in torchvision's segmentation heads.
_PERSON_CLASS = 15

_LOCK = threading.Lock()


class MattingError(RuntimeError):
    """Matting could not run. Never fatal: the caller warns and composites the
    presenter as filmed, which is the documented fallback."""


def configured_model() -> str:
    """A TorchScript matting module to prefer over the built-in segmenter."""
    return (os.environ.get("CODERAI_MATTING_MODEL") or "").strip()


def engine_name() -> Optional[str]:
    """Which engine would run, without loading anything."""
    path = configured_model()
    if path and os.path.isfile(path):
        return f"torchscript:{os.path.basename(path)}"
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception:
        return None
    return "deeplabv3"


def available() -> bool:
    return engine_name() is not None


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #

class _Segmenter:
    """``__call__(batch) -> alpha`` for a float batch in NCHW, 0..1."""

    def __init__(self, module, device, kind: str):
        self.m = module
        self.device = device
        self.kind = kind
        self._rec = [None] * 4          # recurrent state, for RVM-style modules
        self._call = None

    def _candidates(self, x):
        """TorchScript matting exports disagree on their signature the way the
        FILM ports do, so try the conventions rather than pin one."""
        return [
            lambda: self.m(x, *self._rec, 0.25),
            lambda: self.m(x, *self._rec),
            lambda: self.m(x, 0.25),
            lambda: self.m(x),
        ]

    def _as_alpha(self, out):
        import torch
        if isinstance(out, dict):
            for key in ("pha", "alpha", "out"):
                if key in out:
                    out = out[key]
                    break
            else:
                out = next(iter(out.values()))
        if isinstance(out, (list, tuple)):
            # RVM returns (fgr, pha, r1..r4): keep the state, take the alpha.
            if len(out) >= 2 and torch.is_tensor(out[1]):
                if len(out) >= 6:
                    self._rec = list(out[2:6])
                out = out[1]
            else:
                out = out[0]
        if not torch.is_tensor(out):
            raise MattingError(f"matting module returned {type(out).__name__}")
        if out.dim() == 4 and out.shape[1] > 1:
            # A segmentation logit map: the person channel against the rest.
            out = out.softmax(dim=1)[:, _PERSON_CLASS:_PERSON_CLASS + 1]
        elif out.dim() == 3:
            out = out.unsqueeze(1)
        return out.clamp(0, 1)

    def __call__(self, batch):
        import torch
        with torch.no_grad():
            if self.kind == "deeplabv3":
                return self._as_alpha(self.m(batch)["out"])
            if self._call is not None:
                return self._as_alpha(self._call())
            last = None
            for cand in self._candidates(batch):
                try:
                    out = cand()
                except Exception as exc:        # try the next convention
                    last = exc
                    continue
                self._call = lambda c=cand: c()
                return self._as_alpha(out)
            raise MattingError(f"matting module call failed: {last}")


def _load(device) -> _Segmenter:
    import torch

    path = configured_model()
    if path:
        if not os.path.isfile(path):
            raise MattingError(f"CODERAI_MATTING_MODEL is not a file: {path}")
        try:
            module = torch.jit.load(path, map_location=device)
        except Exception as exc:
            raise MattingError(
                f"{os.path.basename(path)} is not a TorchScript module "
                f"(torch.jit.load: {exc})")
        module.eval()
        return _Segmenter(module, device, "torchscript")

    # torch downloads its weights into TORCH_HOME, which defaults to the user's
    # home -- inside a container that is a fresh 42 MB every restart. coderai's
    # cache volume is where everything else this install downloads already
    # lives, so put them there unless the operator said otherwise.
    if not os.environ.get("TORCH_HOME"):
        cache = os.environ.get("CODERAI_CACHE_DIR")
        if cache and os.path.isdir(cache):
            try:
                os.makedirs(os.path.join(cache, "torch"), exist_ok=True)
                os.environ["TORCH_HOME"] = os.path.join(cache, "torch")
            except OSError:
                pass

    try:
        from torchvision.models.segmentation import (
            DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large)
    except Exception as exc:
        raise MattingError(f"torchvision has no segmentation models here ({exc})")
    try:
        net = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    except Exception as exc:
        # First use downloads the weights; an install with no network says so
        # here rather than halfway through a job.
        raise MattingError(f"the segmentation weights could not be loaded ({exc})")
    net.eval().to(device)
    return _Segmenter(net, device, "deeplabv3")


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# the mask video
# --------------------------------------------------------------------------- #

def _probe_size(src: str) -> tuple:
    from codai.compose import media
    info = media.probe(src)
    w, h = int(info.get("width") or 0), int(info.get("height") or 0)
    if w <= 0 or h <= 0:
        raise MattingError(f"{os.path.basename(src)} has no video size")
    return w, h


def matte(src: str, out_path: str, *, fps: int, cancel: Optional[threading.Event] = None,
          log: Optional[Callable[[str], None]] = None,
          gpu_lease: Optional[Callable] = None) -> str:
    """Write a greyscale mask video for ``src`` and return its path.

    One frame in, one frame out, at the source's own size and rate, so the mask
    can be used as an alpha stream against the clip it came from.
    """
    import numpy as np
    import torch

    from codai.compose.media import ffmpeg_bin

    if not available():
        raise MattingError("no matting engine: install torchvision or set "
                           "CODERAI_MATTING_MODEL to a TorchScript matting model")
    sw, sh = _probe_size(src)
    scale = INFER_SIDE / float(min(sw, sh))
    iw = max(32, int(round(sw * scale / 32)) * 32)
    ih = max(32, int(round(sh * scale / 32)) * 32)

    reader = subprocess.Popen(
        [ffmpeg_bin(), "-v", "error", "-i", src, "-vf", f"scale={iw}:{ih}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    # tmix smooths the flicker a per-frame segmenter has and gblur feathers the
    # edge, which is what keeps a cut-out from looking stencilled. Both are on
    # the MASK, so they cost nothing in the colour image.
    writer = subprocess.Popen(
        [ffmpeg_bin(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{iw}x{ih}", "-r", str(int(fps) or 30), "-i", "-",
         "-vf", f"tmix=frames=3,gblur=sigma=1.5,scale={sw}:{sh},format=gray",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", out_path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    lease = gpu_lease() if callable(gpu_lease) else None
    frame_bytes = iw * ih * 3
    done = 0
    try:
        with _LOCK:
            device = _device()
            model = _load(device)
            try:
                while True:
                    if cancel is not None and cancel.is_set():
                        # The job's own cancellation type, so a cancelled render
                        # aborts here instead of being warned about and resumed.
                        from codai.compose.render import Cancelled
                        raise Cancelled()
                    batch: List[bytes] = []
                    while len(batch) < 8:
                        raw = reader.stdout.read(frame_bytes)
                        if not raw or len(raw) < frame_bytes:
                            break
                        batch.append(raw)
                    if not batch:
                        break
                    arr = np.frombuffer(b"".join(batch), dtype=np.uint8).reshape(
                        len(batch), ih, iw, 3)
                    x = torch.from_numpy(arr.copy()).to(device).permute(0, 3, 1, 2)
                    x = x.float().div_(255.0)
                    alpha = model(x)
                    if alpha.shape[-2:] != (ih, iw):
                        alpha = torch.nn.functional.interpolate(
                            alpha, size=(ih, iw), mode="bilinear", align_corners=False)
                    out = (alpha[:, 0] * 255).round().clamp(0, 255).to(torch.uint8)
                    writer.stdin.write(out.cpu().numpy().tobytes())
                    done += len(batch)
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        try:
            reader.stdout.close()
        except Exception:
            pass
        reader.wait(timeout=30)
        try:
            writer.stdin.close()
        except Exception:
            pass
        writer.wait(timeout=120)
        if lease is not None:
            try:
                lease.close()
            except Exception:
                pass

    if done == 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise MattingError("the mask video came out empty")
    if log:
        log(f"matting: {done} frames with {engine_name()}")
    return out_path
