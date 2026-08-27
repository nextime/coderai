# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.

"""Cross-encoder reranking endpoint (``/v1/rerank``).

Scores query↔document relevance with a sequence-classification cross-encoder
(e.g. ``BAAI/bge-reranker-v2-m3``). The model is loaded NATIVELY via transformers
(``AutoModelForSequenceClassification`` — no extra dependency) and cached +
VRAM-managed through :data:`multi_model_manager` like every other model, so it
participates in eviction and thermal gating (``request_model`` waits on the
thermal governor before serving).

Response shape mirrors the common rerank APIs (Cohere/Jina): a list of
``{index, relevance_score}`` sorted by score, optionally with the document text.
``relevance_score`` is the sigmoid of the cross-encoder logit, in [0, 1]
(monotonic, so ranking order is preserved).
"""

import asyncio
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from codai.models.manager import multi_model_manager

router = APIRouter()

global_args = None


def set_global_args(args):
    global global_args
    global_args = args


# Per-model-key asyncio locks so a burst of first-requests loads the model ONCE.
_load_locks: dict = {}


class RerankRequest(BaseModel):
    model: str = Field(..., description="Reranker model id (e.g. 'bge-reranker-v2-m3').")
    query: str = Field(..., description="The search query.")
    documents: List[str] = Field(..., description="Documents to score against the query.")
    top_n: Optional[int] = Field(None, description="Return only the top N results.")
    return_documents: Optional[bool] = Field(
        False, description="Include the document text in each result.")
    max_length: Optional[int] = Field(
        None, description="Max tokens per query+document pair (default: model n_ctx or 8192).")
    model_config = ConfigDict(extra="allow")


def _derive_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _is_reranker(obj) -> bool:
    return isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "reranker"


def _load_reranker(model_name: str, device: str, model_config: dict):
    """Load a cross-encoder as ('reranker', (tokenizer, model, device))."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from codai.models.hf_loading import build_from_pretrained_kwargs

    fp = build_from_pretrained_kwargs(model_config or {})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **fp)
    if 'quantization_config' not in fp and 'device_map' not in fp:
        model = model.to(device)
    model.eval()
    return ("reranker", (tokenizer, model, device))


def _score(model_obj, query: str, docs: List[str], max_length: int) -> List[float]:
    """Score each (query, doc) pair → sigmoid(logit) in [0, 1]."""
    import torch

    _tag, (tokenizer, model, device) = model_obj
    pairs = [[query, d] for d in docs]
    enc = tokenizer(pairs, padding=True, truncation=True,
                    max_length=int(max_length), return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits.view(-1).float()
        scores = torch.sigmoid(logits)
    return scores.cpu().tolist()


@router.post("/v1/rerank", summary="Rerank documents against a query (cross-encoder)")
async def create_rerank(request: RerankRequest, http_request: Request = None):
    if not request.documents:
        raise HTTPException(status_code=400, detail="'documents' must be a non-empty list")
    if not request.query:
        raise HTTPException(status_code=400, detail="'query' is required")

    # Resolve + reserve the model through the manager (thermal wait, caching, routing).
    # Rerankers are registered in the embedding_models category (same transformers/nvidia
    # infra), so resolve with model_type="embedding" to pass the manager's type
    # validation; rerank.py then loads it as a cross-encoder (not an embedder).
    model_info = await asyncio.to_thread(
        multi_model_manager.request_model, request.model, "embedding")
    model_name = model_info.get('model_name')
    if not model_name:
        raise HTTPException(status_code=404,
                            detail=model_info.get('error', f"Model '{request.model}' not found"))
    model_key = model_info['model_key']
    model_obj = model_info.get('model_object')

    if not _is_reranker(model_obj):
        lock = _load_locks.setdefault(model_key, asyncio.Lock())
        async with lock:
            model_obj = multi_model_manager.models.get(model_key)
            if not _is_reranker(model_obj):
                device = _derive_device()
                _cfg = multi_model_manager.config.get(model_name) or {}
                _snap = multi_model_manager.vram_before_load()
                try:
                    model_obj = await asyncio.get_event_loop().run_in_executor(
                        None, _load_reranker, model_name, device, _cfg)
                except Exception as e:
                    raise HTTPException(status_code=500,
                                        detail=f"Failed to load reranker '{model_name}': {e}")
                multi_model_manager.add_model(model_key, model_obj)
                multi_model_manager.current_model_key = model_key
                try:
                    multi_model_manager.record_vram_delta(model_key, _snap)
                except Exception:
                    pass

    _cfg = multi_model_manager.config.get(model_name) or {}
    max_len = int(request.max_length or _cfg.get('n_ctx') or 8192)
    try:
        scores = await asyncio.get_event_loop().run_in_executor(
            None, _score, model_obj, request.query, request.documents, max_len)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rerank failed: {e}")

    results = [{"index": i, "relevance_score": float(s)} for i, s in enumerate(scores)]
    results.sort(key=lambda r: r["relevance_score"], reverse=True)
    if request.top_n:
        results = results[:max(0, int(request.top_n))]
    if request.return_documents:
        for r in results:
            r["document"] = request.documents[r["index"]]

    total_tokens = len(request.query.split()) + sum(len(d.split()) for d in request.documents)
    return {
        "object": "rerank.result",
        "model": request.model,
        "results": results,
        "usage": {"total_tokens": total_tokens},
    }
