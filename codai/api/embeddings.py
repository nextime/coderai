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
Embeddings endpoint — OpenAI-compatible.
POST /v1/embeddings
Supports sentence-transformers, BGE, E5, nomic-embed, etc.
"""

import asyncio
import base64
import time
from typing import List

from fastapi import APIRouter, HTTPException, Request

from codai.models.manager import multi_model_manager
from codai.pydantic.embedrequest import EmbeddingsRequest, EmbeddingsResponse, EmbeddingObject

router = APIRouter()

global_args = None


def set_global_args(args):
    global global_args
    global_args = args


def _derive_device() -> str:
    if global_args:
        d = getattr(global_args, 'vulkan_device', None)
        if d is not None:
            return f"cuda:{d}"
    return "cuda:0"


class _EmbeddingModel:
    """A loaded embedding model, tagged with its backend.

    Iterates as ``(backend, model)`` so ``backend, model = obj`` unpacking keeps
    working everywhere, but also exposes ``cleanup()`` — which is what the model
    manager's eviction path (``_evict_one`` / ``ModelInstancePool.cleanup_all``)
    calls to move weights off the GPU. Without it a bare tuple falls through
    those branches and its VRAM is only reclaimed implicitly by gc.
    """

    __slots__ = ("backend", "model")

    def __init__(self, backend, model):
        self.backend = backend
        self.model = model

    def __iter__(self):
        yield self.backend
        yield self.model

    def cleanup(self):
        try:
            if self.backend == 'sentence_transformers':
                if hasattr(self.model, 'to'):
                    self.model.to('cpu')
            elif self.backend in ('clip', 'transformers'):
                # model is (processor_or_tokenizer, hf_model, device)
                hf_model = self.model[1]
                if hf_model is not None and hasattr(hf_model, 'to'):
                    hf_model.to('cpu')
        except Exception:
            pass
        self.model = None


def _trust_remote_code(model_config: dict = None) -> bool:
    cfg = model_config or {}
    raw = cfg.get('_raw_cfg') if isinstance(cfg.get('_raw_cfg'), dict) else {}
    return bool(cfg.get('trust_remote_code') or raw.get('trust_remote_code'))


# Vision+text dual encoders (CLIP/SigLIP family). These expose get_text_features()
# and get_image_features(), whose projection heads put both modalities in ONE
# shared space — the whole point of a multimodal embedding model.
_DUAL_ENCODER_TYPES = {
    'clip', 'clip_vision_model', 'siglip', 'siglip2', 'chinese_clip',
    'altclip', 'blip', 'blip-2', 'blip_2', 'x_clip', 'metaclip_2',
}


def _is_dual_encoder(model_name: str, trust: bool) -> bool:
    """True if the HF config describes a vision+text dual encoder."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust)
    except Exception:
        return False
    return (hasattr(cfg, 'vision_config')
            or str(getattr(cfg, 'model_type', '')) in _DUAL_ENCODER_TYPES)


def _has_st_modules(model_name: str) -> bool:
    """True if the repo ships a sentence-transformers modules.json.

    Matters for dual encoders: when ST has a native recipe (e.g.
    sentence-transformers/clip-ViT-B-32, jina-clip-v2) it handles both
    modalities itself. When it doesn't, wrapping a raw CLIPModel in ST gives
    unprojected text hidden states — a *different* space from the image
    features — so we must drive the model through transformers instead.
    """
    import os
    try:
        if os.path.isdir(model_name):
            return os.path.isfile(os.path.join(model_name, 'modules.json'))
        from huggingface_hub import file_exists
        return bool(file_exists(model_name, 'modules.json'))
    except Exception:
        return False


def _load_embedding_model(model_name: str, device: str, model_config: dict = None):
    from codai.models.hf_loading import build_from_pretrained_kwargs
    trust = _trust_remote_code(model_config)

    # A dual encoder without an ST recipe must go down the transformers path so
    # text and images share one space; everything else prefers ST.
    prefer_clip = _is_dual_encoder(model_name, trust) and not _has_st_modules(model_name)

    if not prefer_clip:
        try:
            from sentence_transformers import SentenceTransformer
            # sentence-transformers honours quantization via model_kwargs.
            fp = build_from_pretrained_kwargs(model_config)
            st_kwargs = {}
            if 'quantization_config' in fp:
                st_kwargs['model_kwargs'] = {'quantization_config': fp['quantization_config']}
            if trust:
                st_kwargs['trust_remote_code'] = True
            model = SentenceTransformer(model_name, device=device, **st_kwargs)
            return _EmbeddingModel('sentence_transformers', model)
        except ImportError:
            pass

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        fp = build_from_pretrained_kwargs(model_config)
        if trust:
            fp['trust_remote_code'] = True
        model = AutoModel.from_pretrained(model_name, **fp)
        if 'quantization_config' not in fp and 'device_map' not in fp:
            model = model.to(device)
        if hasattr(model, 'get_text_features') and hasattr(model, 'get_image_features'):
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust)
            return _EmbeddingModel('clip', (processor, model, device))
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
        return _EmbeddingModel('transformers', (tokenizer, model, device))
    except Exception as e:
        raise RuntimeError(f"Cannot load embedding model '{model_name}': {e}")


def _supports_images(model_obj) -> bool:
    """True if this loaded model can embed images into the same space as text."""
    backend, model = model_obj
    if backend == 'clip':
        return True
    if backend != 'sentence_transformers':
        return False
    try:
        first = model._first_module()
    except Exception:
        return False
    # ST's CLIPModel module, or a custom (trust_remote_code) module that carries
    # an image processor — both accept PIL images in encode().
    return (type(first).__name__.lower().startswith('clip')
            or hasattr(first, 'processor')
            or hasattr(first, 'image_processor'))


def _decode_image(src: str):
    """Accept a data URI, http(s) URL, local file path or bare base64 blob."""
    import io
    import os
    import re
    from PIL import Image

    if not isinstance(src, str) or not src.strip():
        raise ValueError("empty image reference")
    s = src.strip()

    if s.startswith(('http://', 'https://')):
        import requests
        resp = requests.get(s, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert('RGB')
    if s.startswith('data:'):
        s = s.split(',', 1)[1] if ',' in s else ''
    elif os.path.isfile(s):
        return Image.open(s).convert('RGB')

    try:
        raw = base64.b64decode(re.sub(r'\s+', '', s), validate=True)
    except Exception as e:
        raise ValueError(
            f"image is not a URL, data URI, file path or base64 blob: {e}")
    return Image.open(io.BytesIO(raw)).convert('RGB')


def _clip_feats(raw):
    """Extract the projected shared-space vector from a get_*_features() result.

    transformers <5 returns the projected tensor directly; transformers 5.x
    returns a BaseModelOutputWithPooling whose `pooler_output` holds the
    projected (shared-space) embedding.
    """
    import torch
    if isinstance(raw, torch.Tensor):
        return raw
    for attr in ('text_embeds', 'image_embeds', 'pooler_output'):
        val = getattr(raw, attr, None)
        if val is not None:
            return val
    raise RuntimeError("CLIP model returned no usable feature tensor")


def _embed_texts(model_obj, texts: List[str], dimensions=None) -> List[List[float]]:
    backend, model = model_obj
    if backend == 'sentence_transformers':
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        results = [v.tolist() for v in vecs]
    elif backend == 'clip':
        import torch
        import torch.nn.functional as F
        processor, hf_model, device = model
        inputs = processor(text=texts, padding=True, truncation=True,
                           return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = _clip_feats(hf_model.get_text_features(**inputs))
        feats = F.normalize(feats, dim=-1)
        results = [row.cpu().tolist() for row in feats]
    else:
        import torch
        tokenizer, hf_model, device = model
        encoded = tokenizer(texts, padding=True, truncation=True,
                            return_tensors='pt', max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = hf_model(**encoded)
        # mean-pool last hidden state
        token_embs = out.last_hidden_state
        attention = encoded['attention_mask'].unsqueeze(-1).float()
        mean_emb = (token_embs * attention).sum(1) / attention.sum(1)
        import torch.nn.functional as F
        mean_emb = F.normalize(mean_emb, dim=-1)
        results = [row.cpu().tolist() for row in mean_emb]

    if dimensions:
        results = [v[:dimensions] for v in results]
    return results


def _embed_images(model_obj, images: List[str], dimensions=None) -> List[List[float]]:
    """Embed images into the same vector space as _embed_texts()."""
    backend, model = model_obj
    pil_images = [_decode_image(src) for src in images]

    if backend == 'sentence_transformers':
        vecs = model.encode(pil_images, convert_to_numpy=True,
                            normalize_embeddings=True)
        results = [v.tolist() for v in vecs]
    elif backend == 'clip':
        import torch
        import torch.nn.functional as F
        processor, hf_model, device = model
        inputs = processor(images=pil_images, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = _clip_feats(hf_model.get_image_features(**inputs))
        feats = F.normalize(feats, dim=-1)
        results = [row.cpu().tolist() for row in feats]
    else:
        raise ValueError("model is text-only")

    if dimensions:
        results = [v[:dimensions] for v in results]
    return results


@router.post("/v1/embeddings", response_model=EmbeddingsResponse, summary="Create embeddings")
async def create_embeddings(request: EmbeddingsRequest, http_request: Request = None):
    """
    OpenAI-compatible embeddings endpoint.
    """
    # Register a task so embeddings appear in the unified task list, like every
    # other model type. Finished on success or error below.
    from codai.tasks import task_registry
    _title = (request.input if isinstance(request.input, str)
              else ("image embeddings" if request.image is not None else "embeddings"))
    _tid = task_registry.register(
        "embedding", title=str(_title)[:80], model=(request.model or "embedding"))
    task_registry.start(_tid)
    try:
        _resp = await _run_embeddings(request, http_request)
        task_registry.finish(_tid, "done")
        return _resp
    except HTTPException:
        task_registry.finish(_tid, "error")
        raise
    except Exception as e:
        task_registry.finish(_tid, "error", str(e)[:200])
        raise


async def _run_embeddings(request: EmbeddingsRequest, http_request: Request = None):
    """Core embeddings logic; registered as a task by create_embeddings()."""
    model_info = await asyncio.to_thread(
        multi_model_manager.request_model, request.model, model_type="embedding")
    model_name = model_info.get('model_name')
    if not model_name:
        err = model_info.get('error', f"Model '{request.model}' not found")
        raise HTTPException(status_code=404, detail=err)

    model_key = model_info['model_key']
    model_obj = model_info.get('model_object')

    _emb_cfg = (multi_model_manager.config.get(f"embedding:{model_name}")
                or multi_model_manager.config.get(model_name) or {})

    if model_obj is None:
        device = _derive_device()
        from codai.tasks import loading_task
        # Snapshot VRAM around the load so the model's real footprint is measured
        # and recorded — this is what lets a later request for another model size
        # its eviction correctly (and lets this model be evicted to reclaim VRAM).
        _snap = multi_model_manager.vram_before_load()
        try:
            with loading_task(model_name, model_type="embedding"):
                model_obj = await asyncio.get_event_loop().run_in_executor(
                    None, _load_embedding_model, model_name, device, _emb_cfg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load embedding model: {e}")
        # Register through add_model (pool + models_in_vram bookkeeping) rather than
        # a bare dict assignment, so eviction/unload treat it like every other model.
        multi_model_manager.add_model(model_key, model_obj)
        multi_model_manager.current_model_key = model_key
        multi_model_manager.record_vram_delta(model_key, _snap)

    texts: List[str] = []
    if request.input is not None:
        texts = [request.input] if isinstance(request.input, str) else list(request.input)
    images: List[str] = []
    if request.image is not None:
        images = [request.image] if isinstance(request.image, str) else list(request.image)
    if not texts and not images:
        raise HTTPException(
            status_code=400, detail="Provide 'input' (text) and/or 'image'.")

    if images and not _supports_images(model_obj):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is text-only; image embedding needs a "
                   "multimodal model (CLIP/SigLIP family, e.g. "
                   "sentence-transformers/clip-ViT-B-32 or jinaai/jina-clip-v2).")

    # Text vectors first, then image vectors — indices follow that order.
    vectors: List[List[float]] = []
    try:
        if texts:
            vectors += await asyncio.get_event_loop().run_in_executor(
                None, _embed_texts, model_obj, texts, request.dimensions)
        if images:
            vectors += await asyncio.get_event_loop().run_in_executor(
                None, _embed_images, model_obj, images, request.dimensions)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # Optional TurboQuant vector quantization (data-free, inner-product preserving).
    # The per-model config block (turboquant: {enabled, backend, bits}) is the
    # source of truth for enable/disable + which implementation to use; the
    # per-request `quantization` field triggers it and can override the bit width.
    from codai.models import turboquant as _tq
    _raw = _emb_cfg.get('_raw_cfg') if isinstance(_emb_cfg.get('_raw_cfg'), dict) else {}
    tq_cfg = _emb_cfg.get('turboquant') or _raw.get('turboquant') or {}
    tq_enabled = tq_cfg.get('enabled', None)         # None = no explicit model setting
    tq_backend = (tq_cfg.get('backend') or 'builtin')

    quant_meta = None
    quant_bits = None
    req_spec = getattr(request, 'quantization', None)
    if not req_spec and tq_enabled and tq_cfg.get('bits'):
        req_spec = f"turbo{tq_cfg.get('bits')}"   # model-configured default
    if req_spec:
        if tq_enabled is False:
            raise HTTPException(
                status_code=400,
                detail="TurboQuant is disabled for this model (enable it in the "
                       "model configuration).")
        quant_bits = _tq._parse_quant_spec(req_spec)
        if quant_bits is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported quantization '{req_spec}' "
                       "(use 'turbo'/'turbo8'/'turbo6'/'turbo4'/'turbo2')")

    if quant_bits is not None and request.encoding_format == 'base64':
        # Compact wire form: each embedding is base64 of [f16 norm][packed codes].
        # The compact packing is the built-in wire format regardless of backend
        # (the upstream library exposes its own opaque store, not per-vector blobs).
        blobs, meta = await asyncio.get_event_loop().run_in_executor(
            None, _tq.quantize_base64, vectors, quant_bits)
        data = [EmbeddingObject(index=i, embedding=b) for i, b in enumerate(blobs)]
        quant_meta = {
            "method": meta.method, "bits": meta.bits, "seed": meta.seed,
            "dim": meta.dim, "dim_padded": meta.dim_padded, "radius": meta.radius,
            "bytes_per_vector": meta.bytes_per_vector, "backend": "builtin",
            "layout": "base64([float16 norm][packbits(rotated b-bit codes, MSB-first per numpy.packbits)])",
        }
    elif quant_bits is not None:
        # Lossy reconstruction returned as plain floats (quantized-store fidelity).
        try:
            vectors = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _tq.reconstruct(vectors, quant_bits, backend=tq_backend))
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        data = [EmbeddingObject(index=i, embedding=v) for i, v in enumerate(vectors)]
        eff_backend = tq_backend if tq_backend != 'auto' else _tq.backend_name()
        quant_meta = {"method": "turboquant", "bits": quant_bits,
                      "encoding": "float-reconstruction", "backend": eff_backend}
    elif request.encoding_format == 'base64':
        import struct
        data = [EmbeddingObject(
            index=i,
            embedding=base64.b64encode(struct.pack(f'{len(v)}f', *v)).decode()
        ) for i, v in enumerate(vectors)]
    else:
        data = [EmbeddingObject(index=i, embedding=v) for i, v in enumerate(vectors)]

    total_tokens = sum(len(t.split()) for t in texts)
    resp = EmbeddingsResponse(
        data=data,
        model=request.model,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    )
    if quant_meta is not None:
        resp.quantization = quant_meta
    return resp