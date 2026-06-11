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


def _load_embedding_model(model_name: str, device: str, model_config: dict = None):
    from codai.models.hf_loading import build_from_pretrained_kwargs
    try:
        from sentence_transformers import SentenceTransformer
        # sentence-transformers honours quantization via model_kwargs.
        fp = build_from_pretrained_kwargs(model_config)
        st_kwargs = {}
        if 'quantization_config' in fp:
            st_kwargs['model_kwargs'] = {'quantization_config': fp['quantization_config']}
        model = SentenceTransformer(model_name, device=device, **st_kwargs)
        return ('sentence_transformers', model)
    except ImportError:
        pass

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        fp = build_from_pretrained_kwargs(model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, **fp)
        if 'quantization_config' not in fp and 'device_map' not in fp:
            model = model.to(device)
        return ('transformers', (tokenizer, model, device))
    except Exception as e:
        raise RuntimeError(f"Cannot load embedding model '{model_name}': {e}")


def _embed_texts(model_obj, texts: List[str], dimensions=None) -> List[List[float]]:
    backend, model = model_obj
    if backend == 'sentence_transformers':
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        results = [v.tolist() for v in vecs]
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


@router.post("/v1/embeddings", response_model=EmbeddingsResponse, summary="Create embeddings")
async def create_embeddings(request: EmbeddingsRequest, http_request: Request = None):
    """
    OpenAI-compatible embeddings endpoint.
    """
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
        try:
            model_obj = await asyncio.get_event_loop().run_in_executor(
                None, _load_embedding_model, model_name, device, _emb_cfg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load embedding model: {e}")
        multi_model_manager.models[model_key] = model_obj
        multi_model_manager.current_model_key = model_key

    texts = [request.input] if isinstance(request.input, str) else request.input

    try:
        vectors = await asyncio.get_event_loop().run_in_executor(
            None, _embed_texts, model_obj, texts, request.dimensions)
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