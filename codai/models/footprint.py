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

"""What a loaded model actually occupies, measured from the model itself.

The previous measurement was a process-RSS delta taken across the load:

    measured_ram = self._get_own_ram_gb() - ram_before

RSS is the wrong instrument for this. It counts file-backed pages that the
kernel maps on a read -- the mmap'd safetensors/GGUF the loader just walked --
and ``_get_own_ram_gb`` sums RSS over the engine *and every child*, so pages
shared between those processes (torch, the CUDA runtime, libc, the interpreter)
are added once per process. It is also a delta over a window, so anything else
loading, evicting or faulting pages in that window lands in the figure.

Measured against ground truth on a production host, 2026-10-29:

    gemma-2-9b-it.Q4_K_M    5.37 GB on disk  ->  reported 30.81 GB   (5.7x)
    LTX-2.5, six loads, one config:  22.11, 22.11, 23.28, 4.50, 19.02, 29.80 GB

One of those readings (4.50 GB) came from a load that placed a single layer and
then raised; another (29.80 GB) was taken after an abandoned resident attempt
had moved a transformer and a text encoder to host RAM, so it counted that
attempt as well. Both were persisted as authoritative under
``force_vram_update`` and planned against, which is how a model that fits ends
up on the CPU.

This module answers the question directly instead: walk the loaded object and
add up the bytes its tensors really occupy, then add the parts that are not
weights but are still required -- the KV cache for the configured context, and
a runtime reserve for activations and the decode spike.

It is deliberately placement-independent, matching the contract
``measured_vram_gb`` already documents: offload moves weights between GPU and
host, it does not change how much the model needs. The split is reported
separately (:attr:`Footprint.weights_gpu_gb` / :attr:`weights_cpu_gb`) for
callers that care where the bytes currently sit.

Quantization needs no correction factor here, which is the point of measuring
the object: a bitsandbytes ``Params4bit`` *is* a uint8 tensor, so
``numel() * element_size()`` already yields the packed size, and GPTQ/AWQ keep
their ``qweight``/``qzeros``/``scales`` as ordinary tensors that the walk sees.
What those formats hide is accounted for explicitly: bnb's ``quant_state``
(absmax, the code book, nested second-level state) lives outside
``parameters()`` and is real memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Bytes per element of a llama.cpp KV cache type, relative to the raw dtype.
# Mirrors codai/backends/vulkan.py's table; kept here so this module has no
# import-time dependency on a backend.
_KV_TYPE_BITS = {
    "f32": 32.0, "f16": 16.0, "bf16": 16.0,
    "q8_0": 8.5, "q8_1": 9.0,
    "q5_0": 5.5, "q5_1": 6.0,
    "q4_0": 4.5, "q4_1": 5.0,
}

# Fraction of the weight footprint to reserve for activations and the decode
# spike, by model type. Diffusion caps its spike with VAE tiling/slicing, so it
# does not scale with context the way an autoregressive model does.
_ACTIVATION_RESERVE = {
    "video": 0.08,
    "image": 0.08,
    "audio_gen": 0.08,
    "tts": 0.05,
    "embedding": 0.05,
}
_ACTIVATION_RESERVE_DEFAULT = 0.08
# Autoregressive models get a flat activation share; their context cost is the
# KV cache, which is computed exactly rather than guessed as a fraction.
_ACTIVATION_RESERVE_AUTOREGRESSIVE = 0.10


@dataclass
class Footprint:
    """A model's memory requirement, broken into the parts it is made of."""

    weights_gpu_gb: float = 0.0
    weights_cpu_gb: float = 0.0
    kv_gb: float = 0.0
    activations_gb: float = 0.0
    #: False when something could not be accounted for -- the caller must then
    #: NOT persist this as authoritative. Every gap is named in :attr:`notes`.
    complete: bool = True
    #: True when :attr:`kv_gb` came from a measurement rather than the estimate.
    kv_measured: bool = False
    #: True when the object was walkable and held NO weights -- a failed or
    #: partial load. Distinct from an object we simply cannot introspect, and
    #: the difference decides whether a fallback estimate is legitimate.
    empty_load: bool = False
    #: False when the weights total is known but its GPU/host SPLIT is not.
    #: True of every GGUF: there is no torch object to walk, so the total comes
    #: from the file size, which says nothing about where llama.cpp put the
    #: layers. The size was being banked in ``weights_cpu_gb`` and so reported
    #: as host-resident -- measured on zeiss 2026-10-30, a Gemma-4-14B IQ4_XS
    #: loaded with `n_gpu_layers: all` and holding 10.1 GB of VRAM described
    #: itself as "weights 7.71 GB (0.00 on GPU, 7.71 on host)", and the Tasks
    #: page called it "CPU offload 7.71 GB". llama.cpp's own layer placement is
    #: the authority on the split; this helper must not invent one.
    weights_split_known: bool = True
    notes: list = field(default_factory=list)

    @property
    def weights_gb(self) -> float:
        return self.weights_gpu_gb + self.weights_cpu_gb

    @property
    def total_gb(self) -> float:
        """Everything the model needs, wherever its weights currently sit."""
        return round(self.weights_gb + self.kv_gb + self.activations_gb, 3)

    def describe(self) -> str:
        bits = [f"weights {self.weights_gb:.2f} GB"]
        if not self.weights_split_known:
            # Saying nothing is right here. The alternative that shipped said
            # "0.00 on GPU" about models that were entirely on the GPU.
            bits[-1] += " (placement not measured here)"
        elif self.weights_cpu_gb > 0.005:
            bits[-1] += (f" ({self.weights_gpu_gb:.2f} on GPU, "
                         f"{self.weights_cpu_gb:.2f} on host)")
        if self.kv_gb > 0.005:
            bits.append(f"KV {self.kv_gb:.2f} GB"
                        + ("" if self.kv_measured else " (est)"))
        if self.activations_gb > 0.005:
            bits.append(f"activations {self.activations_gb:.2f} GB")
        out = f"{self.total_gb:.2f} GB total ({', '.join(bits)})"
        if not self.complete:
            out += " [INCOMPLETE: " + "; ".join(self.notes) + "]"
        return out


# --------------------------------------------------------------------------
# tensor-level accounting
# --------------------------------------------------------------------------

def _dtype_itemsize(dtype) -> int:
    """Bytes per element of a torch dtype, without needing a real tensor.

    A meta tensor has no storage to ask, so size it from the dtype it declares.
    """
    try:
        return int(dtype.itemsize)
    except (AttributeError, TypeError):
        pass
    try:                                        # older torch
        import torch
        return int(torch.empty((), dtype=dtype).element_size())
    except Exception:
        return 2                                # fp16/bf16 is the common case


def _tensor_bytes(t) -> tuple:
    """``(bytes, device_kind, exact)`` for one tensor.

    ``device_kind`` is 'gpu' or 'cpu'. ``exact`` is False for a meta tensor,
    whose bytes are DECLARED (numel x dtype) rather than measured -- accelerate
    leaves a meta placeholder where it has moved the real weights to host RAM
    or to disk, and the model still needs them.
    """
    try:
        is_meta = bool(getattr(t, "is_meta", False)) or str(t.device) == "meta"
    except Exception:
        is_meta = False
    if is_meta:
        return _dtype_itemsize(t.dtype) * t.numel(), "cpu", False
    try:
        nbytes = t.numel() * t.element_size()
    except Exception:
        return 0, "cpu", False
    try:
        kind = "gpu" if t.device.type in ("cuda", "xpu", "hpu", "mps") else "cpu"
    except Exception:
        kind = "cpu"
    return nbytes, kind, True


def _storage_key(t):
    """Identity of the tensor's backing storage, or None when it has none.

    Tied weights (an embedding shared with the LM head) and shared buffers must
    be counted once. Two tensors are the same memory when their storage pointer
    matches, which is not the same thing as the tensors being equal.
    """
    try:
        return t.untyped_storage().data_ptr()
    except Exception:
        try:
            return t.storage().data_ptr()
        except Exception:
            return None


def _quant_state_tensors(param):
    """Tensors a bitsandbytes ``Params4bit`` keeps outside ``parameters()``.

    A 4-bit parameter is a uint8 blob plus a ``quant_state`` holding the
    per-block absmax, the code book and -- when double quantization is on -- a
    nested second-level state. That is real memory and is invisible to a plain
    parameter walk.
    """
    qs = getattr(param, "quant_state", None)
    if qs is None:
        return
    seen = set()
    stack = [qs]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        for attr in ("absmax", "code", "offset", "shape_offset"):
            t = getattr(node, attr, None)
            if t is not None and hasattr(t, "numel"):
                yield t
        nested = getattr(node, "state2", None)
        if nested is not None:
            stack.append(nested)


def walk_module_bytes(module) -> dict:
    """Bytes held by one module's parameters and buffers, split GPU/host.

    Returns ``{'gpu': int, 'cpu': int, 'declared': int, 'tensors': int}``,
    where ``declared`` is the subset sized from a meta placeholder rather than
    from real storage. Buffers are walked as well as parameters because the
    pre-quantized formats keep their payload there: GPTQ and AWQ register
    ``qweight``/``qzeros``/``scales`` as buffers, so a parameters-only walk
    reports a 4-bit model as nearly empty.
    """
    out = {"gpu": 0, "cpu": 0, "declared": 0, "tensors": 0}
    if module is None or not hasattr(module, "parameters"):
        return out
    seen = set()

    def _add(t):
        key = _storage_key(t)
        if key is not None:
            if key in seen:
                return
            seen.add(key)
        nbytes, kind, exact = _tensor_bytes(t)
        if nbytes <= 0:
            return
        out[kind] += nbytes
        out["tensors"] += 1
        if not exact:
            out["declared"] += nbytes

    try:
        params = list(module.parameters(recurse=True))
    except Exception:
        params = []
    for p in params:
        _add(p)
        for extra in _quant_state_tensors(p):
            _add(extra)
    try:
        for b in module.buffers(recurse=True):
            _add(b)
    except Exception:
        pass
    return out


def _submodules(obj):
    """``(name, module)`` pairs to account for, for a module or a pipeline.

    A diffusers pipeline is a container: its weights live in components
    (transformer, text_encoder, vae, ...) alongside things that hold none
    (schedulers, tokenizers, feature extractors). Walking the pipeline itself
    would miss the components entirely, since it is not an ``nn.Module``.
    """
    comps = getattr(obj, "components", None)
    if isinstance(comps, dict) and comps:
        for name, comp in comps.items():
            if comp is not None and hasattr(comp, "parameters"):
                yield name, comp
        return
    if hasattr(obj, "parameters"):
        yield "", obj
        return
    # A text/vision model is registered as a ModelManager wrapper, so the real
    # module hangs off .backend.model. Without this an HF LLM falls through to
    # the process-delta fallback even though it is perfectly walkable.
    inner = getattr(getattr(obj, "backend", None), "model", None)
    if inner is not None and hasattr(inner, "parameters"):
        yield "backend.model", inner
        # A vision projector is a separate module beside the LM.
        for attr in ("mmproj", "vision_model", "vision_tower", "multi_modal_projector"):
            extra = getattr(inner, attr, None) or getattr(obj, attr, None)
            if extra is not None and hasattr(extra, "parameters") and extra is not inner:
                yield f"backend.{attr}", extra


def measure_weights(obj) -> tuple:
    """``(gpu_gb, cpu_gb, notes, ok)`` for a loaded torch model or pipeline.

    A note of ``EMPTY_PIPELINE`` means the object IS a pipeline and we walked
    it and it holds no weights at all. That is a failed or partial load, not an
    object we cannot introspect, and the caller must not substitute a guess for
    it.
    """
    notes, gpu, cpu, declared, tensors = [], 0, 0, 0, 0
    parts = list(_submodules(obj))
    if not parts:
        if isinstance(getattr(obj, "components", None), dict):
            return 0.0, 0.0, ["EMPTY_PIPELINE: walked every component and none "
                              "holds any weights — the load did not complete"], False
        return 0.0, 0.0, ["object exposes no torch modules"], False
    for name, mod in parts:
        r = walk_module_bytes(mod)
        gpu += r["gpu"]
        cpu += r["cpu"]
        declared += r["declared"]
        tensors += r["tensors"]
    if tensors == 0:
        return 0.0, 0.0, ["EMPTY_PIPELINE: walked the model and found no "
                          "tensors — the load did not complete"], False
    if declared:
        notes.append(f"{declared / 1e9:.2f} GB sized from meta placeholders "
                     f"(accelerate moved it off the GPU)")
    return gpu / 1e9, cpu / 1e9, notes, True


# --------------------------------------------------------------------------
# KV cache
# --------------------------------------------------------------------------

def _kv_bits(name) -> float:
    return _KV_TYPE_BITS.get(str(name or "f16").strip().lower().replace("-", "_"), 16.0)


def kv_cache_gb(geom: dict, n_ctx: int, type_k=None, type_v=None,
                batch: int = 1, swa_full: bool = True) -> float:
    """KV cache for ``n_ctx`` tokens: layers x kv_heads x (k+v dims) x bits.

    Uses the number of KEY/VALUE heads, not attention heads: a GQA model such
    as Qwen3 shares one KV head across several query heads, so sizing by
    attention heads over-states the cache several-fold.

    Sliding-window layers are sized to their window rather than the full
    context when ``swa_full`` is False, which is what llama.cpp's ``swa_full``
    controls: Gemma alternates a windowed layer with a global one, so a
    full-context figure over-states those layers by ``n_ctx / window``.

    This is an ESTIMATE and callers must treat it as one. Per-layer KV-head
    arrays, shared KV and architecture-specific cache layouts are not
    modelled, and on this codebase's own history such a prediction has been
    wrong by 5x in both directions. Prefer a measured figure
    (``measured_kv_gb`` for the context in question) whenever one exists.
    Returns 0.0 when the geometry is unknown, so the caller can say so rather
    than guess.
    """
    layers = int(geom.get("block_count") or geom.get("num_hidden_layers") or 0)
    kv_heads = int(geom.get("head_count_kv") or geom.get("num_key_value_heads")
                   or geom.get("head_count") or geom.get("num_attention_heads") or 0)
    if not layers or not kv_heads or n_ctx <= 0:
        return 0.0
    k_dim = int(geom.get("key_length") or geom.get("head_dim") or 0)
    v_dim = int(geom.get("value_length") or k_dim or 0)
    if not k_dim:
        heads = int(geom.get("head_count") or geom.get("num_attention_heads") or 0)
        emb = int(geom.get("embedding_length") or geom.get("hidden_size") or 0)
        if heads and emb:
            k_dim = v_dim = emb // heads
    if not k_dim:
        return 0.0
    per_layer = kv_heads * ((k_dim * _kv_bits(type_k)
                             + v_dim * _kv_bits(type_v)) / 8.0)
    window = int(geom.get("sliding_window") or geom.get("attention.sliding_window") or 0)
    swa_layers = int(geom.get("swa_layer_count") or 0)
    if window and not swa_full and 0 < window < n_ctx:
        # Alternating pattern unless the geometry states otherwise: llama.cpp
        # reports is_swa per layer, so a caller that knows the real count
        # should pass swa_layer_count.
        if not swa_layers:
            swa_layers = layers // 2
        swa_layers = min(swa_layers, layers)
        tokens = swa_layers * window + (layers - swa_layers) * n_ctx
    else:
        tokens = layers * n_ctx
    return max(0, batch) * per_layer * tokens / 1e9


def hf_geometry(obj) -> dict:
    """Geometry for KV sizing, read from a transformers config when present."""
    cfg = getattr(obj, "config", None)
    if cfg is None:
        cfg = getattr(getattr(obj, "model", None), "config", None)
    if cfg is None:
        # A ModelManager wrapper keeps the real module at .backend.model, which
        # is where its config lives too. Missing this silently drops the KV term
        # for every HF text model -- the one kind that definitely has one.
        _inner = getattr(getattr(obj, "backend", None), "model", None)
        cfg = getattr(_inner, "config", None)
    if cfg is None:
        return {}
    def _g(*names):
        for n in names:
            v = getattr(cfg, n, None)
            if isinstance(v, int) and v > 0:
                return v
        return 0
    geom = {
        "num_hidden_layers": _g("num_hidden_layers", "n_layer", "num_layers"),
        "num_attention_heads": _g("num_attention_heads", "n_head"),
        "num_key_value_heads": _g("num_key_value_heads", "num_kv_heads"),
        "hidden_size": _g("hidden_size", "n_embd", "d_model"),
        "head_dim": _g("head_dim"),
    }
    return {k: v for k, v in geom.items() if v}


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------

def _is_autoregressive(model_type: str) -> bool:
    return str(model_type or "").lower() in ("text", "vision", "")


def _configured_ctx(cfg: dict) -> int:
    for key in ("n_ctx", "context_size", "max_model_len", "max_position_embeddings"):
        try:
            v = int((cfg or {}).get(key) or 0)
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v
    return 0



def _measured_kv_for(measured, n_ctx: int) -> float:
    """A previously measured KV size for this context, or 0.0.

    Recorded per n_ctx because the cache scales with it, so a figure taken at
    4k says nothing about 128k. Accepts a bare number (treated as applying to
    the context asked for) or a ``{n_ctx: gb}`` mapping, which is how
    ``measured_kv_gb`` is persisted.
    """
    if measured is None:
        return 0.0
    if isinstance(measured, dict):
        for key in (str(n_ctx), n_ctx):
            if key in measured:
                try:
                    return float(measured[key] or 0.0)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0
    try:
        return float(measured or 0.0)
    except (TypeError, ValueError):
        return 0.0

def measure(obj, cfg: dict = None, model_type: str = "", *,
            geom: dict = None, file_size_gb: float = 0.0,
            batch: int = 1, measured_kv_gb=None, swa_full: bool = True) -> Footprint:
    """Measure what ``obj`` occupies, plus the context and runtime it needs.

    ``obj`` is a torch module, a diffusers pipeline, or None when the model is
    not a torch object at all (a GGUF served by llama.cpp, or an isolated
    subprocess worker). In that case pass ``file_size_gb``: the weights are
    exactly the file, since a GGUF is loaded as-is with no dtype conversion.

    The KV term prefers ``measured_kv_gb`` -- llama.cpp reports its own KV
    buffer size exactly, and predicting it from metadata has been wrong by 5x
    in both directions on this codebase's own models. It is computed from
    ``geom``/``cfg`` only when no measurement is available, and the footprint
    then says so in :attr:`Footprint.notes`.

    A footprint whose weights could not be accounted for comes back with
    ``complete=False`` and must not be persisted.
    """
    cfg = cfg or {}
    fp = Footprint()

    if obj is not None:
        gpu, cpu, notes, ok = measure_weights(obj)
        fp.weights_gpu_gb, fp.weights_cpu_gb = gpu, cpu
        fp.notes.extend(notes)
        fp.empty_load = any(n.startswith("EMPTY_PIPELINE") for n in notes)
        if not ok:
            if fp.empty_load:
                # Nothing to fall back TO: we looked, and the model has no
                # weights. A size invented from elsewhere is how a load that
                # placed one layer came to be recorded as the model's footprint.
                fp.complete = False
            elif file_size_gb > 0:
                fp.weights_cpu_gb = float(file_size_gb)
                fp.weights_split_known = False
                fp.notes.append("fell back to the on-disk size")
            else:
                fp.complete = False
    elif file_size_gb > 0:
        # A GGUF: the weights ARE the file, loaded as-is. That fixes the total
        # and says nothing about placement, so the split stays unknown rather
        # than defaulting to "all on host".
        fp.weights_cpu_gb = float(file_size_gb)
        fp.weights_split_known = False
    else:
        fp.complete = False
        fp.notes.append("nothing to measure: no model object and no file size")

    # KV cache -- exact where the geometry is known, and only for the model
    # kinds that keep one. A diffusion pipeline has no KV cache.
    if _is_autoregressive(model_type):
        n_ctx = _configured_ctx(cfg)
        meas = _measured_kv_for(measured_kv_gb, n_ctx)
        if meas > 0:
            fp.kv_gb = round(meas, 3)
            fp.kv_measured = True
        else:
            g = dict(geom or {})
            if not g and obj is not None:
                g = hf_geometry(obj)
            if n_ctx and g:
                fp.kv_gb = round(kv_cache_gb(g, n_ctx, cfg.get("cache_type_k"),
                                             cfg.get("cache_type_v"), batch=batch,
                                             swa_full=swa_full), 3)
                if fp.kv_gb:
                    fp.notes.append(f"KV is an ESTIMATE ({fp.kv_gb:.2f} GB at "
                                    f"n_ctx={n_ctx}); no measured_kv_gb recorded yet")
                else:
                    fp.notes.append(f"KV not sized: geometry incomplete ({sorted(g)})")
            elif n_ctx:
                fp.notes.append("KV not sized: model geometry unknown")

    base = fp.weights_gb + fp.kv_gb
    share = (_ACTIVATION_RESERVE_AUTOREGRESSIVE if _is_autoregressive(model_type)
             else _ACTIVATION_RESERVE.get(str(model_type or "").lower(),
                                          _ACTIVATION_RESERVE_DEFAULT))
    fp.activations_gb = round(base * share, 3)

    fp.weights_gpu_gb = round(fp.weights_gpu_gb, 3)
    fp.weights_cpu_gb = round(fp.weights_cpu_gb, 3)
    return fp


def extra_file_gb(*paths) -> float:
    """Total size of companion weight files (an mmproj projector, a LoRA).

    These load on top of the model and are not reachable by walking it when
    they are held by a separate handler.
    """
    total = 0
    for p in paths:
        if not p:
            continue
        try:
            total += os.path.getsize(os.path.expanduser(str(p)))
        except OSError:
            continue
    return total / 1e9
