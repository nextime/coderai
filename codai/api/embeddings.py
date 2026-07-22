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

    __slots__ = ("backend", "model", "lock")

    def __init__(self, backend, model):
        self.backend = backend
        self.model = model
        # llama.cpp contexts are NOT thread-safe: two executor threads calling
        # embed() on one ctx segfault the engine. Serialize per model instance.
        if backend == 'llama':
            import threading
            self.lock = threading.Lock()
        else:
            self.lock = None

    def __iter__(self):
        yield self.backend
        yield self.model

    def cleanup(self):
        try:
            if self.backend == 'llama':
                # llama.cpp model: close() frees the ctx + weights (VRAM incl.)
                if hasattr(self.model, 'close'):
                    self.model.close()
            elif self.backend == 'sentence_transformers':
                if hasattr(self.model, 'to'):
                    self.model.to('cpu')
            elif self.backend in ('clip', 'transformers', 'vision', 'qwenvl'):
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

# IMAGE-ONLY encoders (DINOv2, plain ViT, …). No text tower, no tokenizer: they
# embed images into their own space (image↔image similarity / retrieval). Text
# requests against these return a clear 400.
_VISION_ONLY_TYPES = {
    'dinov2', 'dinov2_with_registers', 'vit', 'vit_mae', 'vit_msn',
    'swin', 'swinv2', 'beit', 'deit', 'convnext', 'convnextv2',
    'ijepa', 'videomae',
}


def _hf_model_type(model_name: str, trust: bool) -> str:
    """The HF config's model_type for a repo, or '' when unreadable."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust)
        return str(getattr(cfg, 'model_type', ''))
    except Exception:
        return ''


def _is_vision_only(model_name: str, trust: bool) -> bool:
    """True if the HF config describes an image-only encoder (no text tower)."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust)
    except Exception:
        return False
    return (str(getattr(cfg, 'model_type', '')) in _VISION_ONLY_TYPES
            and not hasattr(cfg, 'vision_config'))


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

    # GGUF file → llama.cpp in embedding mode (works on whatever backend this
    # build targets: Vulkan on the radeon engine, CUDA on nvidia). Text-only —
    # llama.cpp's embedding path has no image tower wired here.
    if str(model_name).lower().endswith('.gguf'):
        try:
            from llama_cpp import Llama
            cfg = model_config or {}
            raw = cfg.get('_raw_cfg') if isinstance(cfg.get('_raw_cfg'), dict) else {}
            n_ctx = int(cfg.get('n_ctx') or raw.get('n_ctx') or 2048)
            n_gpu_layers = cfg.get('n_gpu_layers', raw.get('n_gpu_layers', -1))
            model = Llama(
                model_path=model_name,
                embedding=True,
                n_ctx=n_ctx,
                # CRITICAL: embedding mode asserts (SIGABRT, killing the whole
                # engine) when an input has more tokens than the batch —
                # GGML_ASSERT(out_ids.size() == n_outputs). Size both batches
                # to the full context so any input we accept can be processed;
                # _embed_texts truncates inputs to n_ctx to close the loop.
                n_batch=n_ctx,
                n_ubatch=n_ctx,
                n_gpu_layers=int(n_gpu_layers if n_gpu_layers is not None else -1),
                verbose=False,
            )
            return _EmbeddingModel('llama', model)
        except Exception as e:
            raise RuntimeError(
                f"Cannot load GGUF embedding model '{model_name}': {e}")

    # Image-only encoder (DINOv2/ViT…): no tokenizer, no text tower — neither ST
    # nor the text paths below can load it. Dedicated processor+model backend.
    if _is_vision_only(model_name, trust):
        try:
            from transformers import AutoImageProcessor, AutoModel
            fp = build_from_pretrained_kwargs(model_config)
            if trust:
                fp['trust_remote_code'] = True
            processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=trust)
            model = AutoModel.from_pretrained(model_name, **fp)
            if 'quantization_config' not in fp and 'device_map' not in fp:
                model = model.to(device)
            return _EmbeddingModel('vision', (processor, model, device))
        except Exception as e:
            raise RuntimeError(f"Cannot load image embedding model '{model_name}': {e}")

    # Qwen2-VL-family embedders (GME-Qwen2-VL…): loaded NATIVELY, not through the
    # repo's sentence-transformers custom code — that code pins transformers<4.52
    # (it reaches into Qwen2-VL internals that were since refactored) and would
    # refuse to load on this stack. The model is plain Qwen2-VL weights; the GME
    # embedding is the last-token hidden state under their documented chat prompt,
    # which _qwenvl_embed reproduces. No trust_remote_code needed.
    _qwen_mt = _hf_model_type(model_name, trust)
    if _qwen_mt in ('qwen2_vl', 'qwen2_5_vl'):
        try:
            import torch
            # CONCRETE native classes, not Auto*: GME's config.json auto_map routes
            # the Auto entry points into its remote module, whose import-time guard
            # rejects transformers>=4.52. The concrete classes never consult
            # auto_map, so the pinned remote code is bypassed entirely.
            if _qwen_mt == 'qwen2_vl':
                from transformers import Qwen2VLForConditionalGeneration as _VLM
                from transformers import Qwen2VLProcessor as _PROC
            else:
                from transformers import Qwen2_5_VLForConditionalGeneration as _VLM
                try:
                    from transformers import Qwen2_5_VLProcessor as _PROC
                except ImportError:
                    from transformers import AutoProcessor as _PROC
            fp = build_from_pretrained_kwargs(model_config)
            if 'quantization_config' not in fp:
                fp.setdefault('dtype', torch.float16)   # GME reference runs fp16
            # GME's image-token budget (min/max pixels) from its reference code.
            processor = _PROC.from_pretrained(
                model_name, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
            processor.tokenizer.padding_side = 'right'
            model = _VLM.from_pretrained(model_name, **fp)
            if 'quantization_config' not in fp and 'device_map' not in fp:
                model = model.to(device)
            model.eval()
            return _EmbeddingModel('qwenvl', (processor, model, device))
        except Exception as e:
            raise RuntimeError(f"Cannot load Qwen-VL embedding model '{model_name}': {e}")

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
    """True if this loaded model can embed images (shared space for clip/ST
    multimodal; image-only space for the 'vision' backend)."""
    backend, model = model_obj
    if backend in ('clip', 'vision', 'qwenvl'):
        return True
    if backend != 'sentence_transformers':
        return False
    try:
        first = model._first_module()
    except Exception:
        return False
    # ST's CLIPModel module accepts PIL images in encode() directly.
    if type(first).__name__.lower().startswith('clip'):
        return True
    # A custom (trust_remote_code) multimodal module (e.g. GME) carries a
    # PROCESSOR that can handle images. A bare hasattr check is not enough —
    # ST's plain Transformer module also exposes a (text) processor attribute,
    # which made text-only models look image-capable and turned image requests
    # into a 500 instead of a clean 400.
    proc = getattr(first, 'processor', None) or getattr(first, 'image_processor', None)
    if proc is None:
        return False
    return (hasattr(proc, 'image_processor')
            or 'image' in type(proc).__name__.lower())


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


def _truncate_dims(results, dimensions):
    """Matryoshka-style truncation: keep the first N dims, then RE-normalize —
    a truncated slice of a unit vector is no longer unit-norm, and downstream
    cosine/dot-product math assumes normalized embeddings (this matches how
    OpenAI applies `dimensions`). Meaningful for MRL-trained models
    (Qwen3-Embedding: 32-2560); others degrade gracefully but aren't trained
    for truncation."""
    if not dimensions:
        return results
    out = []
    for v in results:
        t = v[:dimensions]
        n = sum(x * x for x in t) ** 0.5
        out.append([x / n for x in t] if n > 0 else t)
    return out


def _qwenvl_embed(model_tuple, items, dimensions=None):
    """GME-style embedding on a native Qwen2-VL: last-token hidden state under the
    GME chat prompt (mirrors the repo's custom_st tokenize/forward, which we can't
    run — it pins transformers<4.52). `items` are {'text': …} / {'image': PIL} dicts;
    each batch is modality-homogeneous (all-text or all-image), matching GME's own
    'consistent batch' requirement."""
    import torch
    import torch.nn.functional as F
    processor, hf_model, device = model_tuple

    instruction = 'You are a helpful assistant.'
    prompts, images = [], []
    for it in items:
        body = ''
        img = it.get('image')
        if img is not None:
            body += '<|vision_start|><|image_pad|><|vision_end|>'
            images.append(img)
        if it.get('text'):
            body += it['text']
        prompts.append(
            f'<|im_start|>system\n{instruction}<|im_end|>\n'
            f'<|im_start|>user\n{body}<|im_end|>\n'
            f'<|im_start|>assistant\n<|endoftext|>')

    inputs = processor(text=prompts, images=images or None, padding='longest',
                       truncation=True, max_length=1800, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(hf_model.dtype)
    with torch.no_grad():
        out = hf_model(**inputs, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states[-1]
    # Right padding → the last REAL token (the <|endoftext|>) is at mask.sum()-1.
    idx = inputs['attention_mask'].sum(dim=1) - 1
    emb = hs[torch.arange(hs.shape[0], device=hs.device), idx]
    emb = F.normalize(emb.float(), dim=-1)
    results = [row.cpu().tolist() for row in emb]
    return _truncate_dims(results, dimensions)


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
    elif backend == 'qwenvl':
        return _qwenvl_embed(model, [{'text': t} for t in texts], dimensions)
    elif backend == 'llama':
        # llama.cpp embedding mode. With a pooling type baked into the GGUF the
        # result is one vector per input; without, per-token vectors — mean-pool
        # those. Normalize either way (llama.cpp does not normalize).
        import math
        results = []
        # Generous safety margin: embed() re-tokenizes the chunk text itself
        # (with BOS/specials), and a detokenize→retokenize roundtrip is NOT
        # token-count-stable — a window cut at exactly n_ctx can re-inflate
        # past it and trip llama.cpp's GGML_ASSERT (SIGABRT).
        try:
            _tok_limit = max(64, model.n_ctx() - 64)
        except Exception:
            _tok_limit = 1984

        def _retok_len(txt):
            return len(model.tokenize(txt.encode('utf-8', 'ignore'),
                                      add_bos=True, special=True))

        def _fit_chunk(toks):
            """Detokenize a token window, shrinking until its RE-tokenized
            length verifiably fits the context (roundtrip can inflate)."""
            end = len(toks)
            while end > 0:
                txt = model.detokenize(toks[:end]).decode('utf-8', 'ignore')
                n = _retok_len(txt)
                if n <= _tok_limit:
                    return txt, end
                end -= max(8, n - _tok_limit)
            return '', 0

        def _embed_one(chunk_text):
            emb = model.embed(chunk_text)
            if emb and isinstance(emb[0], (list, tuple)):
                n = len(emb)
                emb = [sum(col) / n for col in zip(*emb)]
            return emb

        # Serialize on the model's lock: llama.cpp ctxs are not thread-safe and
        # concurrent embed() calls from parallel requests segfault the engine.
        import contextlib
        _mlock = getattr(model_obj, 'lock', None)
        with (_mlock if _mlock is not None else contextlib.nullcontext()):
            for t in texts:
                # An input longer than the context window would abort llama.cpp
                # (GGML_ASSERT). Instead of cutting the text, CHUNK it into
                # verified context-sized windows, embed each, and combine with a
                # token-count-weighted mean — no content is dropped;
                # single-chunk inputs take the direct path.
                _chunks = None
                try:
                    if _retok_len(t) <= _tok_limit:
                        _chunks = [(t, 1)]
                    else:
                        _toks = model.tokenize(t.encode('utf-8', 'ignore'),
                                               add_bos=True, special=False)
                        _chunks = []
                        _i = 0
                        while _i < len(_toks):
                            _txt, _used = _fit_chunk(_toks[_i:_i + _tok_limit])
                            if _used <= 0:
                                break  # pathological token — skip it
                            _chunks.append((_txt, _used))
                            _i += _used
                except Exception:
                    _chunks = [(t, 1)]
                if not _chunks:
                    _chunks = [(t[:2000], 1)]
                if len(_chunks) == 1:
                    emb = _embed_one(_chunks[0][0])
                else:
                    _acc = None
                    _tot = 0
                    for _ct, _cw in _chunks:
                        _e = _embed_one(_ct)
                        if _acc is None:
                            _acc = [x * _cw for x in _e]
                        else:
                            for _j, _x in enumerate(_e):
                                _acc[_j] += _x * _cw
                        _tot += _cw
                    emb = [x / _tot for x in _acc]
                norm = math.sqrt(sum(x * x for x in emb)) or 1.0
                results.append([x / norm for x in emb])
    elif backend == 'vision':
        raise ValueError(
            "this embedding model is image-only (no text tower) — send 'image' "
            "instead of 'input', or use a text/multimodal embedding model")
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

    return _truncate_dims(results, dimensions)


def _embed_images(model_obj, images: List[str], dimensions=None) -> List[List[float]]:
    """Embed images into the same vector space as _embed_texts()."""
    backend, model = model_obj
    pil_images = [_decode_image(src) for src in images]

    if backend == 'sentence_transformers':
        try:
            # ST's native CLIP wrapper takes PIL images directly.
            vecs = model.encode(pil_images, convert_to_numpy=True,
                                normalize_embeddings=True)
        except Exception:
            # Custom multimodal ST modules (e.g. GME-Qwen2-VL's custom_st) take
            # {"image": …} dicts instead of bare PIL objects.
            vecs = model.encode([{'image': im} for im in pil_images],
                                convert_to_numpy=True, normalize_embeddings=True)
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
    elif backend == 'qwenvl':
        return _qwenvl_embed(model, [{'image': im} for im in pil_images], dimensions)
    elif backend == 'vision':
        # Image-only encoder (DINOv2/ViT…): CLS/pooled token of the vision
        # transformer is the image representation.
        import torch
        import torch.nn.functional as F
        processor, hf_model, device = model
        inputs = processor(images=pil_images, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = hf_model(**inputs)
        feats = getattr(out, 'pooler_output', None)
        if feats is None:
            feats = out.last_hidden_state[:, 0]   # CLS token
        feats = F.normalize(feats, dim=-1)
        results = [row.cpu().tolist() for row in feats]
    else:
        raise ValueError("model is text-only")

    return _truncate_dims(results, dimensions)


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


# Per-model-key asyncio locks serializing embedding model loads (see the
# comment at the acquire site in _run_embeddings).
_load_locks: dict = {}


async def _load_embedding_locked(request, model_key: str, model_name: str,
                                 _emb_cfg: dict):
    """Load an embedding model — called with the per-model load lock HELD.

    Re-checks the registry first (another request may have finished the load
    while we waited on the lock), then loads with the standard evict-and-retry
    contract every other model type follows."""
    # Re-check under the lock: the request that held the lock before us has
    # usually just loaded the model.
    model_obj = multi_model_manager.models.get(model_key)
    if model_obj is not None:
        return model_obj

    device = _derive_device()
    from codai.tasks import loading_task
    # Snapshot VRAM around the load so the model's real footprint is measured
    # and recorded — this is what lets a later request for another model size
    # its eviction correctly (and lets this model be evicted to reclaim VRAM).
    _snap = multi_model_manager.vram_before_load()

    def _cuda_oom(exc: Exception) -> bool:
        s = str(exc)
        return ('CUDA' in s or 'cuda' in s or 'out of memory' in s.lower()
                or 'CUBLAS' in s)

    try:
        with loading_task(model_name, model_type="embedding"):
            try:
                model_obj = await asyncio.get_event_loop().run_in_executor(
                    None, _load_embedding_model, model_name, device, _emb_cfg)
            except Exception as e:
                # Same contract as every other model type: if the load hit
                # OOM (e.g. a video/image model owns the card, or a
                # concurrent load raced request_model's free-VRAM check),
                # make room — evicting idle models and WAITING on busy ones
                # (_evict_models_for_vram waits for the active model to go
                # idle) — then retry the load once instead of failing.
                if not _cuda_oom(e):
                    raise
                print(f"[embeddings] load OOM ({str(e)[:100]}) — "
                      f"evicting to make room and retrying")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                # If another model (e.g. a video pipeline) is mid-load, its
                # VRAM reservation is active — wait for that load to finish
                # (bounded) before evicting/retrying, exactly like queued
                # requests for any other model wait out a load.
                try:
                    import time as _time
                    _deadline = _time.time() + 300
                    while (getattr(multi_model_manager,
                                   '_loading_reservations', None)
                           and _time.time() < _deadline):
                        await asyncio.sleep(2)
                except Exception:
                    pass
                # extra_vram_gb keeps the eviction meaningful even when the
                # model has no VRAM estimate yet (first-ever load).
                await asyncio.get_event_loop().run_in_executor(
                    None, multi_model_manager.ensure_vram_for,
                    model_key, model_name, 2.0)
                model_obj = await asyncio.get_event_loop().run_in_executor(
                    None, _load_embedding_model, model_name, device, _emb_cfg)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Failed to load embedding model: {e}")
    # Register through add_model (pool + models_in_vram bookkeeping) rather than
    # a bare dict assignment, so eviction/unload treat it like every other model.
    multi_model_manager.add_model(model_key, model_obj)
    multi_model_manager.current_model_key = model_key
    multi_model_manager.record_vram_delta(model_key, _snap)
    return model_obj


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
        # Serialize loads per model: a burst of first-requests (a bulk indexer)
        # otherwise ALL see model_obj None and EACH loads its own copy — several
        # 8 GB instances of the same embedder stacked on the card (observed as
        # pool_instances=3/6), which is what actually filled the GPU. One
        # request loads; the rest wait on the lock and reuse the loaded model.
        _lock = _load_locks.setdefault(model_key, asyncio.Lock())
        async with _lock:
            model_obj = await _load_embedding_locked(
                request, model_key, model_name, _emb_cfg)

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
        # Encode-time CUDA OOM: another workload grabbed the card's remaining
        # VRAM mid-flight (no room left for the forward pass's scratch buffers).
        # Behave like any other model — evict idle models / wait on busy ones —
        # then retry the encode once.
        s = str(e)
        if not ('CUDA' in s or 'CUBLAS' in s or 'out of memory' in s.lower()):
            raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
        print(f"[embeddings] encode OOM ({s[:100]}) — evicting to make room and retrying")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            # Only scratch room is needed (the model is already resident) — and
            # marking it active first keeps the eviction pass off the embedding
            # model itself while it frees other idle models.
            multi_model_manager.active_in_vram = model_key
            await asyncio.get_event_loop().run_in_executor(
                None, multi_model_manager._evict_models_for_vram, 2.0)
            vectors = []
            if texts:
                vectors += await asyncio.get_event_loop().run_in_executor(
                    None, _embed_texts, model_obj, texts, request.dimensions)
            if images:
                vectors += await asyncio.get_event_loop().run_in_executor(
                    None, _embed_images, model_obj, images, request.dimensions)
        except ValueError as e2:
            raise HTTPException(status_code=400, detail=str(e2))
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Embedding failed: {e2}")

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