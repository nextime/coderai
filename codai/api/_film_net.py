# CoderAI - in-process FILM frame interpolation engine.
#
# FILM (Frame Interpolation for Large Motion) is distributed in several forms.
# The most portable torch artifact is a TorchScript module, which we can load and
# run in-process with no extra architecture code. This wrapper loads such a model
# and exposes the same `.interpolate(a, b)` (t=0.5) interface the RIFE engine uses,
# so the dispatcher is engine-agnostic. Higher fps multipliers come from the
# caller's recursive midpoint bisection.
import torch


class _FilmWrapper:
    """Adapts a loaded FILM module to `.interpolate(img0, img1) -> mid` at t=0.5.

    FILM ports expose a few call conventions; we try the common ones and cache
    whichever works for this module."""

    def __init__(self, module, device):
        self.m = module
        self.device = device
        self._call = None

    @torch.no_grad()
    def interpolate(self, img0, img1):
        t = torch.full((img0.shape[0], 1), 0.5, device=img0.device, dtype=img0.dtype)
        candidates = [
            lambda: self.m({"x0": img0, "x1": img1, "time": t})["image"],
            lambda: self.m({"x0": img0, "x1": img1, "time": t[:, :1, None, None]})["image"],
            lambda: self.m(img0, img1, 0.5),
            lambda: self.m(img0, img1, t),
            lambda: self.m(img0, img1),
        ]
        if self._call is not None:
            return self._call()
        last = None
        for c in candidates:
            try:
                out = c()
                if isinstance(out, dict):
                    out = out.get("image") or next(iter(out.values()))
                self._call = c
                return out
            except Exception as e:  # try the next calling convention
                last = e
        raise RuntimeError(f"FILM module call failed (incompatible signature): {last}")


def load_film(weights_path: str, device):
    """Load a FILM model. Supports a TorchScript module (.pt/.pth saved via
    torch.jit) — the portable in-process form. Raises with guidance otherwise."""
    try:
        module = torch.jit.load(weights_path, map_location=device)
        module.eval()
        return _FilmWrapper(module, device)
    except Exception as e:
        raise RuntimeError(
            "FILM weights must be a TorchScript module for in-process use "
            f"(torch.jit.load failed: {e}). Provide a scripted FILM model, or use "
            "a RIFE interpolation model instead.")
