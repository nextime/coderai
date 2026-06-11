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

"""TurboQuant-style data-free vector quantization.

A faithful, dependency-light implementation of the core idea behind TurboQuant
(Zandieh et al., *TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate*, arXiv:2504.19874, ICLR 2026): randomly rotate each vector so
its coordinates become near-Gaussian and well concentrated, then apply a simple
per-coordinate uniform scalar quantizer. The rotation makes the cheap uniform
quantizer near rate-distortion optimal, and — crucially for retrieval — the
reconstruction is **unbiased**, so inner products / cosine similarities between
quantized vectors are preserved in expectation.

What this gives coderai: an optional compact representation for ``/v1/embeddings``
output (4–8× smaller than float32) whose dot products match the full-precision
embeddings, suitable for storing in a vector DB.

Scope / honesty: this is the rotation + scalar-quantization core (a *data-free,
calibration-free* quantizer). It does **not** implement the paper's extra 1-bit
QJL residual stage, which buys a little more accuracy at the same bit budget;
the structure here is deliberately simple, deterministic and fast (an O(d log d)
randomized Hadamard transform, no stored rotation matrix, no torch dependency).

The rotation is keyed by ``(dim, seed)`` only — never by the data — so every
vector from the same model lands in the *same* rotated space and quantized
vectors remain mutually comparable. Use a fixed ``seed`` per deployment.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Default global seed for the keyed rotation. Stable across processes so the
# same embedding model always quantizes into the same space.
DEFAULT_SEED = 0x7B_C0_DE


# --- Optional upstream library backend ------------------------------------
# When the `turboquant-py` package (pip install "turboquant-py[torch]") is
# installed, the float-reconstruction path can delegate to its QJL/inner-product
# quantizer, which adds the paper's 1-bit residual stage this built-in core
# omits. It is used opportunistically and every call is guarded — any import or
# API mismatch transparently falls back to the built-in NumPy implementation, so
# there is never a hard dependency. Control via CODERAI_TURBOQUANT_LIB:
#   auto (default) = use the library if importable; off/0 = always built-in.
_LIB_MODE = os.environ.get("CODERAI_TURBOQUANT_LIB", "auto").strip().lower()


def _lib():
    if _LIB_MODE in ("off", "0", "false", "no", "none"):
        return None
    try:
        import turboquant as _tq  # turboquant-py
        return _tq
    except Exception:
        return None


def have_library() -> bool:
    """True if the optional turboquant-py backend is importable and enabled."""
    return _lib() is not None


def backend_name() -> str:
    """Name of the active reconstruction backend ('turboquant-py' or 'builtin')."""
    return "turboquant-py" if have_library() else "builtin"


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _signs(dim_padded: int, seed: int) -> np.ndarray:
    """Deterministic ±1 sign vector keyed by (dim_padded, seed)."""
    rng = np.random.default_rng(np.uint64(seed) ^ np.uint64(dim_padded))
    return rng.integers(0, 2, size=dim_padded, dtype=np.int8).astype(np.float32) * 2.0 - 1.0


def _fwht(a: np.ndarray) -> np.ndarray:
    """In-place fast Walsh-Hadamard transform along the last axis.

    ``a`` must have a power-of-two last dimension. Returns the *unnormalized*
    transform (H with H@H == n*I); callers scale by 1/sqrt(n) for orthonormality.
    """
    a = a.astype(np.float32, copy=True)
    n = a.shape[-1]
    h = 1
    while h < n:
        # vectorized butterfly over the last axis
        a = a.reshape(*a.shape[:-1], n // (2 * h), 2, h)
        x = a[..., 0, :]
        y = a[..., 1, :]
        a = np.concatenate([x + y, x - y], axis=-1)
        a = a.reshape(*a.shape[:-2], n)
        h *= 2
    return a


def _rotate(x: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Orthonormal randomized Hadamard rotation R(x) = (1/sqrt(P)) H (s ⊙ x)."""
    p = signs.shape[0]
    return _fwht(x * signs) / np.sqrt(p, dtype=np.float32)


def _irotate(y: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Inverse rotation R^-1(y) = s ⊙ ((1/sqrt(P)) H y) (R is orthonormal)."""
    p = signs.shape[0]
    return (_fwht(y) / np.sqrt(p, dtype=np.float32)) * signs


def _clip_radius(dim_padded: int) -> float:
    """Clip range for the rotated *unit* vector's coordinates.

    After rotating a unit vector, each coordinate is ~N(0, 1/P); ~4 sigma covers
    the distribution with negligible clipping while keeping the quantizer step
    small. Returns r so coordinates are quantized over [-r, r].
    """
    return 4.0 / np.sqrt(float(dim_padded))


@dataclass
class TurboQuantMeta:
    method: str
    bits: int
    seed: int
    dim: int            # original embedding dimension
    dim_padded: int     # power-of-two rotation size
    radius: float
    bytes_per_vector: int


def _parse_quant_spec(spec: Optional[str]) -> Optional[int]:
    """Map a request quantization string to a bit width, or None.

    Accepts ``turbo`` (=8 bit), ``turbo8``/``turbo6``/``turbo4``/``turbo2``,
    or a bare integer string. Returns None for falsy / unrecognized values so
    callers can treat it as "no quantization".
    """
    if not spec:
        return None
    s = str(spec).strip().lower().replace("-", "").replace("_", "")
    if s in ("turbo", "turboquant"):
        return 8
    if s.startswith("turbo"):
        s = s[5:]
    if s.isdigit():
        b = int(s)
        return b if b in (2, 4, 6, 8) else None
    return None


def _as_2d(vectors) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(vectors, dtype=np.float32)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    return arr, single


def _prepare(vectors, bits: int, seed: int):
    """Rotate unit-normalized vectors and return (codes, norms, signs, meta)."""
    arr, single = _as_2d(vectors)
    n, dim = arr.shape
    p = _next_pow2(dim)
    signs = _signs(p, seed)

    norms = np.linalg.norm(arr, axis=1, keepdims=True).astype(np.float32)
    safe = np.where(norms == 0.0, 1.0, norms)
    unit = arr / safe

    padded = np.zeros((n, p), dtype=np.float32)
    padded[:, :dim] = unit
    rot = _rotate(padded, signs)                      # ~N(0, 1/P) coordinates

    r = _clip_radius(p)
    levels = (1 << bits) - 1
    q = np.clip((rot + r) / (2.0 * r), 0.0, 1.0)
    codes = np.rint(q * levels).astype(np.uint16)     # in [0, levels]

    meta = TurboQuantMeta(
        method="turboquant", bits=bits, seed=seed, dim=dim, dim_padded=p,
        radius=float(r), bytes_per_vector=(p * bits + 7) // 8 + 2,
    )
    return codes, norms.reshape(-1), signs, meta, single


def _decode_codes(codes: np.ndarray, norms: np.ndarray, signs: np.ndarray,
                  meta: TurboQuantMeta) -> np.ndarray:
    """Inverse of :func:`_prepare` — unbiased reconstruction back to dim ``meta.dim``."""
    r = meta.radius
    levels = (1 << meta.bits) - 1
    rot = codes.astype(np.float32) / levels * (2.0 * r) - r
    padded = _irotate(rot, signs)
    out = padded[:, :meta.dim] * norms.reshape(-1, 1)
    return out


# ---------------------------------------------------------------------------
# Bit packing (generic 2/4/6/8-bit, vectorized via numpy bit-planes)
# ---------------------------------------------------------------------------

def _pack_bits(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack per-row uint codes (each ``bits`` wide) into bytes. Returns (n, ceil(P*bits/8))."""
    n, p = codes.shape
    planes = ((codes[:, :, None] >> np.arange(bits, dtype=np.uint16)) & 1).astype(np.uint8)
    flat = planes.reshape(n, p * bits)
    return np.packbits(flat, axis=1)


def _unpack_bits(packed: np.ndarray, p: int, bits: int) -> np.ndarray:
    """Inverse of :func:`_pack_bits`."""
    n = packed.shape[0]
    flat = np.unpackbits(packed, axis=1)[:, : p * bits]
    planes = flat.reshape(n, p, bits).astype(np.uint16)
    weights = (np.uint16(1) << np.arange(bits, dtype=np.uint16))
    return (planes * weights).sum(axis=2).astype(np.uint16)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _reconstruct_library(arr2d: np.ndarray, bits: int, seed: int) -> Optional[np.ndarray]:
    """Reconstruct via turboquant-py; None if the library is unavailable / errors."""
    tq = _lib()
    if tq is None:
        return None
    try:
        TQ = tq.TurboQuant(dim=int(arr2d.shape[1]), bit_width=int(bits),
                           mode="mse", seed=int(seed) & 0x7FFFFFFF)
        rec = TQ.dequantize(TQ.quantize(arr2d))
        rec = np.asarray(rec, dtype=np.float32)
        if rec.shape != arr2d.shape:
            return None
        return rec
    except Exception:
        return None


def _reconstruct_builtin(arr2d: np.ndarray, bits: int, seed: int) -> np.ndarray:
    codes, norms, signs, meta, _ = _prepare(arr2d, bits, seed)
    return _decode_codes(codes, norms, signs, meta)


def reconstruct(vectors, bits: int, seed: int = DEFAULT_SEED,
                backend: str = "builtin") -> List[List[float]]:
    """Quantize then dequantize — the lossy float reconstruction.

    The returned vectors are the same shape as the input and behave (in inner
    product / cosine) like ``bits``-bit TurboQuant-stored embeddings.

    ``backend`` selects the implementation explicitly:
      * ``builtin``  — the built-in NumPy quantizer (always available).
      * ``library``  — the upstream ``turboquant-py`` (QJL inner-product mode);
        raises if it is not installed/enabled, rather than silently degrading.
      * ``auto``     — library if available, else built-in.
    """
    arr, single = _as_2d(vectors)
    b = (backend or "builtin").strip().lower()
    out = None
    if b in ("library", "external", "turboquant-py", "turboquantpy", "auto"):
        out = _reconstruct_library(arr, bits, seed)
        if out is None and b != "auto":
            raise RuntimeError(
                "TurboQuant 'library' backend selected but turboquant-py is "
                "unavailable or failed — install \"turboquant-py[torch]\" or "
                "switch the model's TurboQuant backend to 'builtin'.")
    if out is None:
        out = _reconstruct_builtin(arr, bits, seed)
    lst = out.tolist()
    return lst[0] if single else lst


def quantize_packed(vectors, bits: int, seed: int = DEFAULT_SEED
                    ) -> Tuple[List[bytes], TurboQuantMeta]:
    """Quantize to the compact wire form: one ``bytes`` blob per vector.

    Each blob is ``[float16 norm][packed b-bit rotated codes]``. Decode with
    :func:`unpack_blob` using the returned :class:`TurboQuantMeta`.
    """
    codes, norms, _signs, meta, _single = _prepare(vectors, bits, seed)
    packed = _pack_bits(codes, bits)
    norm16 = norms.astype(np.float16)
    blobs = [norm16[i].tobytes() + packed[i].tobytes() for i in range(packed.shape[0])]
    return blobs, meta


def quantize_base64(vectors, bits: int, seed: int = DEFAULT_SEED
                    ) -> Tuple[List[str], TurboQuantMeta]:
    """Like :func:`quantize_packed` but each blob base64-encoded (JSON-friendly)."""
    blobs, meta = quantize_packed(vectors, bits, seed)
    return [base64.b64encode(b).decode("ascii") for b in blobs], meta


def unpack_blob(blob: bytes, meta: TurboQuantMeta) -> List[float]:
    """Decode a single packed blob (or base64 str) back to a float vector."""
    if isinstance(blob, str):
        blob = base64.b64decode(blob)
    norm = np.frombuffer(blob[:2], dtype=np.float16).astype(np.float32)
    packed = np.frombuffer(blob[2:], dtype=np.uint8)[None, :]
    codes = _unpack_bits(packed, meta.dim_padded, meta.bits)
    signs = _signs(meta.dim_padded, meta.seed)
    out = _decode_codes(codes, norm, signs, meta)
    return out[0].tolist()


if __name__ == "__main__":
    # Self-test: rotation round-trips, and quantization preserves inner products.
    rng = np.random.default_rng(1)
    d = 384
    X = rng.standard_normal((64, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    p = _next_pow2(d)
    s = _signs(p, DEFAULT_SEED)
    pad = np.zeros((X.shape[0], p), dtype=np.float32); pad[:, :d] = X
    assert np.allclose(_irotate(_rotate(pad, s), s), pad, atol=1e-4), "rotation not invertible"

    for bits in (8, 4, 2):
        R = np.asarray(reconstruct(X, bits))
        # cosine between original and reconstruction
        cos = (X * R).sum(1) / (np.linalg.norm(R, axis=1) + 1e-9)
        # preservation of pairwise inner products
        G0 = X @ X.T
        G1 = R @ R.T
        err = np.abs(G0 - G1).mean()
        blobs, meta = quantize_packed(X, bits)
        rt = np.asarray(unpack_blob(blobs[0], meta))
        assert np.allclose(rt, R[0], atol=1e-5), "packed blob != reconstruct"
        print(f"bits={bits}: mean|Δip|={err:.4f}  meanCos={cos.mean():.4f}  "
              f"bytes/vec={meta.bytes_per_vector} (float32={d*4})")
    print("turboquant self-test OK")
