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


def _load_embedding_model(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        return ('sentence_transformers', model)
    except ImportError:
        pass

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
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


@router.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(request: EmbeddingsRequest, http_request: Request = None):
    """
    OpenAI-compatible embeddings endpoint.
    """
    model_info = multi_model_manager.request_model(request.model, model_type="embedding")
    model_name = model_info.get('model_name')
    if not model_name:
        err = model_info.get('error', f"Model '{request.model}' not found")
        raise HTTPException(status_code=404, detail=err)

    model_key = model_info['model_key']
    model_obj = model_info.get('model_object')

    if model_obj is None:
        device = _derive_device()
        try:
            model_obj = await asyncio.get_event_loop().run_in_executor(
                None, _load_embedding_model, model_name, device)
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

    if request.encoding_format == 'base64':
        import struct
        data = [EmbeddingObject(
            index=i,
            embedding=base64.b64encode(struct.pack(f'{len(v)}f', *v)).decode()
        ) for i, v in enumerate(vectors)]
    else:
        data = [EmbeddingObject(index=i, embedding=v) for i, v in enumerate(vectors)]

    total_tokens = sum(len(t.split()) for t in texts)
    return EmbeddingsResponse(
        data=data,
        model=request.model,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    )