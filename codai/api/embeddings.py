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
        if backend in ('llama', 'llama-vl', 'dinov2cpp'):
            import threading
            self.lock = threading.Lock()
        else:
            self.lock = None

    def __iter__(self):
        yield self.backend
        yield self.model

    def cleanup(self):
        # Eviction can fire while a request is mid-encode (embeddings hold no
        # pool ref, so _is_key_busy sees idle). Closing a llama ctx / killing
        # the subprocess under a running embed segfaults the engine — take the
        # same per-model lock the embed paths hold so cleanup WAITS for the
        # in-flight call to finish.
        _l = getattr(self, 'lock', None)
        if _l is not None:
            _l.acquire()
        try:
            self._cleanup_locked()
        finally:
            if _l is not None:
                _l.release()

    def _cleanup_locked(self):
        try:
            if self.backend == 'dinov2cpp':
                # persistent embed server subprocess — terminating it frees
                # its (V)RAM; eviction restarts it on the next request.
                try:
                    self.model.terminate()
                    self.model.wait(timeout=10)
                except Exception:
                    try:
                        self.model.kill()
                    except Exception:
                        pass
            elif self.backend == 'llama-vl':
                llm, mctx = self.model
                try:
                    from llama_cpp import mtmd_cpp as _M
                    if mctx:
                        _M.mtmd_free(mctx)
                except Exception:
                    pass
                if hasattr(llm, 'close'):
                    llm.close()
            elif self.backend == 'llama':
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


def _os_basename(p) -> str:
    import os
    return os.path.basename(str(p))


def _load_embedding_model(model_name: str, device: str, model_config: dict = None):
    from codai.models.hf_loading import build_from_pretrained_kwargs
    trust = _trust_remote_code(model_config)

    # GGUF file → llama.cpp in embedding mode (works on whatever backend this
    # build targets: Vulkan on the radeon engine, CUDA on nvidia). Text-only —
    # llama.cpp's embedding path has no image tower wired here.
    # DINOv2 GGUF → the dinov2.cpp `dinov2-embed` server (ggml Vulkan/CPU;
    # llama.cpp cannot load a ViT-only arch). One persistent subprocess holds
    # the model; requests stream image paths in and JSON embeddings out. Its
    # CLS output matches the HF 'vision' backend at ~0.997 cosine.
    if (str(model_name).lower().endswith('.gguf')
            and 'dinov2' in _os_basename(model_name).lower()):
        import os as _os
        import subprocess
        _bin = _os.environ.get('DINOV2_EMBED_BIN', '/opt/coderai/bin/dinov2-embed')
        if not _os.path.isfile(_bin):
            raise RuntimeError(
                f"dinov2-embed binary not found at {_bin} — build it with "
                "packaging/dinov2cpp/build.sh")
        cfg = model_config or {}
        raw = cfg.get('_raw_cfg') if isinstance(cfg.get('_raw_cfg'), dict) else {}
        env = dict(_os.environ)
        _ngl = cfg.get('n_gpu_layers', raw.get('n_gpu_layers', -1))
        if _ngl == 0:
            env['DINOV2_FORCE_CPU'] = '1'
        # Reap any ORPHANED embed server for this model first: if the engine
        # process crashed, its child survived re-parented (still holding VRAM)
        # and the respawned engine would stack a second copy next to it.
        try:
            subprocess.run(['pkill', '-f', f'dinov2-embed -m {model_name}'],
                           timeout=10)
        except Exception:
            pass

        def _die_with_parent():
            # PR_SET_PDEATHSIG: the kernel kills the child if the engine dies,
            # so a crashed engine can never leak a VRAM-holding orphan again.
            try:
                import ctypes
                ctypes.CDLL('libc.so.6').prctl(1, 9)  # (PR_SET_PDEATHSIG, SIGKILL)
            except Exception:
                pass

        proc = subprocess.Popen(
            [_bin, '-m', model_name, '-t', '8'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1,
            preexec_fn=_die_with_parent)
        # wait for the ready line (model load), skipping loader chatter
        import json as _json
        import time as _time
        _deadline = _time.time() + 300
        while _time.time() < _deadline:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("dinov2-embed exited during load")
            line = line.strip()
            if line.startswith('{'):
                try:
                    if _json.loads(line).get('ready'):
                        break
                except Exception:
                    pass
        else:
            proc.kill()
            raise RuntimeError("dinov2-embed load timed out")
        return _EmbeddingModel('dinov2cpp', proc)

    if str(model_name).lower().endswith('.gguf'):
        try:
            import os as _os
            import llama_cpp as _L
            from llama_cpp import Llama
            cfg = model_config or {}
            raw = cfg.get('_raw_cfg') if isinstance(cfg.get('_raw_cfg'), dict) else {}
            n_ctx = int(cfg.get('n_ctx') or raw.get('n_ctx') or 2048)
            n_gpu_layers = cfg.get('n_gpu_layers', raw.get('n_gpu_layers', -1))
            n_gpu_layers = int(n_gpu_layers if n_gpu_layers is not None else -1)
            # A configured multimodal projector (mmproj) upgrades this to a
            # VISION-capable embedder: the llama.cpp mtmd image tower feeds an
            # embeddings context (pooling LAST), reproducing the GME scheme —
            # last-token hidden state under the GME chat prompt — natively on
            # whatever backend this llama.cpp build targets (Vulkan/CUDA/CPU).
            _mmproj = cfg.get('mmproj') or raw.get('mmproj')
            if _mmproj and _os.path.isfile(str(_mmproj)):
                from llama_cpp import mtmd_cpp as _M
                llm = Llama(
                    model_path=model_name,
                    embedding=True,
                    pooling_type=_L.LLAMA_POOLING_TYPE_LAST,
                    n_ctx=n_ctx, n_batch=n_ctx, n_ubatch=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )
                mp = _M.mtmd_context_params_default()
                mp.use_gpu = (n_gpu_layers != 0)
                mp.print_timings = False
                mp.n_threads = 8
                # Bound the vision token budget: qwen2-vl dynamic resolution
                # turns a large photo into thousands of image tokens, and the
                # vision tower's transient compute buffer then spikes VRAM —
                # which evicts co-resident models mid-encode on a small card.
                try:
                    mp.image_max_tokens = int(
                        cfg.get('image_max_tokens')
                        or raw.get('image_max_tokens') or 1024)
                except Exception:
                    pass
                mctx = _M.mtmd_init_from_file(
                    str(_mmproj).encode(), llm._model.model, mp)
                if not mctx:
                    llm.close()
                    raise RuntimeError(f"mtmd projector load failed: {_mmproj}")
                return _EmbeddingModel('llama-vl', (llm, mctx))
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
                n_gpu_layers=n_gpu_layers,
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
    if backend in ('clip', 'vision', 'qwenvl', 'llama-vl', 'dinov2cpp'):
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


def _resize_image_to_token_budget(img, max_tokens, patch=28):
    """Downscale a PIL image so the qwen2-vl vision tower can't emit more than
    ``max_tokens`` image tokens. Token count ≈ (W//patch) * (H//patch) (28px
    patches; observed 896×896 → 1024 tokens). mtmd's own ``image_max_tokens``
    budget is NOT honoured by this llama.cpp build (2048-token batches slipped
    through and overflowed the KV context → 'failed to find a memory slot' →
    HTTP 500), so we clamp here BEFORE handing the bitmap to mtmd. Aspect ratio
    is preserved; only oversize images are touched."""
    try:
        w, h = int(img.width), int(img.height)
    except Exception:
        return img
    tw, th = max(1, w // patch), max(1, h // patch)
    if tw * th <= max_tokens or max_tokens < 1:
        return img
    import math
    scale = math.sqrt(max_tokens / float(tw * th))
    nw = max(patch, (int(w * scale) // patch) * patch)
    nh = max(patch, (int(h * scale) // patch) * patch)
    # Rounding up to a whole patch can nudge back over budget — trim the longer
    # side one patch at a time until it fits.
    while (nw // patch) * (nh // patch) > max_tokens and (nw > patch or nh > patch):
        if nw >= nh:
            nw = max(patch, nw - patch)
        else:
            nh = max(patch, nh - patch)
    try:
        from PIL import Image as _PILImage
        resample = getattr(_PILImage, "Resampling", _PILImage).LANCZOS
        out = img.resize((nw, nh), resample)
        print(f"[embeddings] resized oversize image {w}x{h} "
              f"(~{tw*th} tok) -> {nw}x{nh} (~{(nw//patch)*(nh//patch)} tok) "
              f"to fit vision token budget {max_tokens}", flush=True)
        return out
    except Exception:
        return img


def _llama_vl_embed(model_obj, items, dimensions=None):
    """GME-style embedding on a GGUF Qwen2-VL via llama.cpp mtmd: last-token
    hidden state (ctx pooling LAST) under the GME chat prompt — the same scheme
    as the HF `_qwenvl_embed`, so both backends produce the SAME vector space
    (verified ~0.89 cosine agreement at Q4). `items` are {'text': str} /
    {'image': PIL.Image} dicts. Caller need not hold the model lock."""
    import ctypes
    import math
    import llama_cpp as _L
    from llama_cpp import mtmd_cpp as _M
    llm, mctx = model_obj.model
    ctx = llm._ctx.ctx
    n_embd = llm.n_embd()
    try:
        n_ctx = llm.n_ctx()
    except Exception:
        n_ctx = 2048
    marker = _M.mtmd_default_marker()
    if isinstance(marker, bytes):
        marker = marker.decode()

    def _prompt(body):
        return (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{body}<|im_end|>\n"
                f"<|im_start|>assistant\n<|endoftext|>")

    def _seq_embedding():
        e = _L.llama_get_embeddings_seq(ctx, 0)
        if not e:
            e = _L.llama_get_embeddings_ith(ctx, -1)
        if not e:
            raise RuntimeError("llama.cpp returned no embedding")
        v = [e[i] for i in range(n_embd)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def _clear_kv():
        try:
            _L.llama_memory_clear(_L.llama_get_memory(ctx), True)
        except Exception:
            try:
                llm.reset()
            except Exception:
                pass

    out = []
    _mlock = getattr(model_obj, 'lock', None)
    import contextlib
    with (_mlock if _mlock is not None else contextlib.nullcontext()):
        for it in items:
            _clear_kv()
            img = it.get('image')
            if img is not None:
                # PIL image → raw RGB bitmap straight into mtmd (its own stb
                # decoder can't read AVIF/WebP variants PIL already handled).
                rgb = img.convert('RGB')
                # Guard the KV context: cap image tokens at n_ctx minus a reserve
                # for the chat-prompt frame (system/user/vision markers ~30 tok).
                # Without this a large photo emits >n_ctx image tokens and mtmd's
                # decode fails ('no memory slot') → 500. n_ctx is sized to fit a
                # full-budget image (see the model's n_ctx in models.json).
                rgb = _resize_image_to_token_budget(rgb, max(256, n_ctx - 128))
                raw = rgb.tobytes()
                buf = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
                bmp = _M.mtmd_bitmap_init(rgb.width, rgb.height, buf)
                if not bmp:
                    raise ValueError("mtmd bitmap init failed")
                chunks = _M.mtmd_input_chunks_init()
                try:
                    itext = _M.mtmd_input_text(
                        text=_prompt(marker).encode(),
                        add_special=True, parse_special=True)
                    arr = (_M.mtmd_bitmap_p_ctypes * 1)(bmp)
                    rc = _M.mtmd_tokenize(
                        mctx, chunks, ctypes.byref(itext),
                        ctypes.cast(arr, ctypes.POINTER(ctypes.c_void_p)), 1)
                    if rc != 0:
                        raise RuntimeError(f"mtmd_tokenize failed ({rc})")
                    _np = ctypes.c_int(0)
                    rc = _M.mtmd_helper_eval_chunks(
                        mctx, ctx, chunks, 0, 0, n_ctx, True,
                        ctypes.byref(_np))
                    if rc != 0:
                        raise RuntimeError(f"mtmd eval failed ({rc})")
                finally:
                    _M.mtmd_input_chunks_free(chunks)
                    _M.mtmd_bitmap_free(bmp)
            else:
                body = it.get('text') or ''
                toks = llm.tokenize(_prompt(body).encode('utf-8', 'ignore'),
                                    add_bos=True, special=True)
                if len(toks) > n_ctx - 8:
                    # over-long text: shrink the BODY, keep the prompt frame
                    _bt = llm.tokenize(body.encode('utf-8', 'ignore'),
                                       add_bos=False, special=False)
                    _keep = max(16, (n_ctx - 8) - (len(toks) - len(_bt)))
                    body = llm.detokenize(_bt[:_keep]).decode('utf-8', 'ignore')
                    toks = llm.tokenize(_prompt(body).encode('utf-8', 'ignore'),
                                        add_bos=True, special=True)[:n_ctx - 4]
                batch = _L.llama_batch_init(len(toks), 0, 1)
                try:
                    batch.n_tokens = len(toks)
                    for i, tk in enumerate(toks):
                        batch.token[i] = tk
                        batch.pos[i] = i
                        batch.n_seq_id[i] = 1
                        batch.seq_id[i][0] = 0
                        batch.logits[i] = 1 if i == len(toks) - 1 else 0
                    if _L.llama_decode(ctx, batch) != 0:
                        raise RuntimeError("llama_decode failed")
                finally:
                    _L.llama_batch_free(batch)
            out.append(_seq_embedding())
    return _truncate_dims(out, dimensions)


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
    elif backend == 'llama-vl':
        return _llama_vl_embed(model_obj, [{'text': t} for t in texts], dimensions)
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
    elif backend in ('vision', 'dinov2cpp'):
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
    elif backend == 'llama-vl':
        return _llama_vl_embed(model_obj, [{'image': im} for im in pil_images],
                               dimensions)
    elif backend == 'dinov2cpp':
        # dinov2-embed subprocess: temp-file the PIL images, send paths, read
        # JSON lines back; normalize (the binary emits raw CLS values).
        import contextlib
        import json as _json
        import math
        import os
        import tempfile
        proc = model
        results = []
        _mlock = getattr(model_obj, 'lock', None)
        with (_mlock if _mlock is not None else contextlib.nullcontext()):
            if proc.poll() is not None:
                raise RuntimeError("dinov2-embed process died — retry (it will reload)")
            for im in pil_images:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    tmp = f.name
                try:
                    im.convert('RGB').save(tmp, 'PNG')
                    proc.stdin.write(tmp + "\n")
                    proc.stdin.flush()
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            raise RuntimeError("dinov2-embed died mid-request")
                        line = line.strip()
                        if not line.startswith('{'):
                            continue
                        d = _json.loads(line)
                        if 'error' in d:
                            raise ValueError(f"dinov2-embed: {d['error']}")
                        v = d['embedding']
                        n = math.sqrt(sum(x * x for x in v)) or 1.0
                        results.append([x / n for x in v])
                        break
                finally:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
        return _truncate_dims(results, dimensions)
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