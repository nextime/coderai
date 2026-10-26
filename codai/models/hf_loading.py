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

"""Shared HuggingFace/transformers loading helper.

Translates a coderai per-model configuration (the uniform models.json schema)
into ``from_pretrained`` kwargs so that EVERY transformers-based loader
(spatial, embedding, audio-gen, vision, …) honours the same quantization,
offload, flash-attention and memory settings as the text/image/video loaders.

The configuration is the single source of truth — nothing here reads CLI args.
"""

import os
from typing import Any, Dict, Optional


def resolve_offload_dir(offload_dir) -> str:
    """Return an ABSOLUTE, writable disk-offload directory.

    A RELATIVE value (e.g. the legacy './offload' — whether the built-in default or
    a stale per-model override saved into models.json) resolves against the CWD,
    which in the OCI image is the READ-ONLY /opt/coderai/app tree → makedirs raises
    EACCES. So:
      * an absolute configured path is respected as-is;
      * a relative/empty one INHERITS the configured GLOBAL offload dir
        (global_args.offload_dir / config.offload.directory) when that's absolute,
        then CODERAI_OFFLOAD_DIR (the container's writable default), then the user
        cache — never the CWD.
    This keeps the configuration authoritative while making a relative value mean
    'use the configured offload location', not 'write next to the (read-only) app'.
    """
    if offload_dir:
        p = os.path.expanduser(str(offload_dir))
        if os.path.isabs(p):
            return p
    try:
        from codai.api.state import get_global_args
        g = getattr(get_global_args(), "offload_dir", None)
        if g:
            gp = os.path.expanduser(str(g))
            if os.path.isabs(gp):
                return gp
    except Exception:
        pass
    env = os.environ.get("CODERAI_OFFLOAD_DIR")
    if env:
        return os.path.expanduser(env)
    return os.path.join(os.path.expanduser("~"), ".cache", "coderai", "offload")


def _norm(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the per-model config dict, unwrapping a forwarded `_raw_cfg`."""
    if not cfg:
        return {}
    raw = cfg.get('_raw_cfg') if isinstance(cfg, dict) else None
    merged = dict(cfg)
    if isinstance(raw, dict):
        # Raw entry fills any key the translated kwargs didn't set.
        for k, v in raw.items():
            merged.setdefault(k, v)
    return merged


def resolve_dtype(cfg: Optional[Dict[str, Any]], default: str = 'bf16'):
    """Resolve torch dtype from the model's `precision` setting."""
    import torch
    precision = (_norm(cfg).get('precision') or default)
    return {
        'bf16': torch.bfloat16,
        'f16':  torch.float16,
        'fp16': torch.float16,
        'f32':  torch.float32,
        'fp32': torch.float32,
    }.get(precision, torch.bfloat16 if default == 'bf16' else torch.float32)


def build_quantization_config(cfg: Optional[Dict[str, Any]]):
    """Build a transformers BitsAndBytesConfig from the model config, or None."""
    c = _norm(cfg)
    load_in_4bit = bool(c.get('load_in_4bit', False))
    load_in_8bit = bool(c.get('load_in_8bit', False))
    if not (load_in_4bit or load_in_8bit):
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
        if load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=resolve_dtype(cfg),
                bnb_4bit_use_double_quant=True,
            )
        return BitsAndBytesConfig(load_in_8bit=True)
    except Exception as e:
        print(f"  Quantization requested but unavailable: {e}")
        return None


def _is_gguf_value(v) -> bool:
    """True if a component_quantization value points to a GGUF file (path/URL)."""
    return isinstance(v, str) and v.strip().lower().endswith('.gguf')


def _normalize_quant_mode(mode) -> Optional[str]:
    """Normalize a quant mode string to '2bit' / '4bit' / '8bit' / None.

    2-bit uses the quanto backend (optimum-quanto); 4/8-bit use bitsandbytes.
    GGUF file values are handled separately (see build_gguf_pipeline_components).
    """
    if mode in (None, '', 'none', 'off', False):
        return None
    if _is_gguf_value(mode):
        return None  # GGUF handled elsewhere
    m = str(mode).lower().replace('-', '').replace('_', '').replace(' ', '')
    if m in ('2bit', '2', 'int2', 'quanto2'):
        return '2bit'
    if m in ('4bit', '4', 'int4', 'nf4', 'bnb4'):
        return '4bit'
    if m in ('8bit', '8', 'int8', 'bnb8'):
        return '8bit'
    return None


def _discover_components(model_name: str) -> Dict[str, Any]:
    """Return {component_name: [library, class_name]} from the pipeline config."""
    out: Dict[str, Any] = {}
    try:
        from diffusers import DiffusionPipeline
        for name, spec in DiffusionPipeline.load_config(model_name).items():
            if name.startswith('_'):
                continue
            if isinstance(spec, (list, tuple)) and spec:
                out[name] = list(spec)
    except Exception:
        pass
    return out


def build_pipeline_quant_config(model_name: str, cfg: Optional[Dict[str, Any]], dtype):
    """Build a diffusers PipelineQuantizationConfig from a per-model config.

    Honours an optional per-component override map ``component_quantization``
    (e.g. {"transformer": "4bit", "text_encoder": "8bit", "vae": "none"}).
    Supported per-component modes:
      - "4bit" / "8bit": bitsandbytes (default backend)
      - "2bit": optimum-quanto (int2) — requires `pip install optimum-quanto`
      - a "*.gguf" path/URL: handled by build_gguf_pipeline_components, NOT here
    When the map is absent it falls back to the global ``load_in_4bit`` /
    ``load_in_8bit`` flag applied to all heavy components.

    Returns ``(quant_config, description)`` or ``(None, '')``.
    """
    c = _norm(cfg)
    comp_q = c.get('component_quantization') or {}
    global_4 = bool(c.get('load_in_4bit', False))
    global_8 = bool(c.get('load_in_8bit', False))
    if not comp_q and not (global_4 or global_8):
        return None, ''

    try:
        from diffusers.quantizers import PipelineQuantizationConfig
        from diffusers import BitsAndBytesConfig as DiffBnb
        from transformers import BitsAndBytesConfig as TfBnb
    except Exception as e:
        print(f"  Pipeline quantization unavailable: {e}")
        return None, ''

    comp_lib = {n: (s[0] if isinstance(s, list) and s else 'diffusers')
                for n, s in _discover_components(model_name).items()}

    def _is_heavy(name: str) -> bool:
        return (name.startswith('transformer') or name == 'unet'
                or name.startswith('text_encoder'))

    def _quanto_cfg(lib: str):
        # optimum-quanto int2 via the diffusers / transformers QuantoConfig.
        import importlib.util
        _have_quanto = False
        try:
            _have_quanto = importlib.util.find_spec('optimum.quanto') is not None
        except Exception:
            _have_quanto = False
        if not _have_quanto:
            print("  2-bit requested but optimum-quanto is not installed — "
                  "run `pip install optimum-quanto`. Skipping (component stays "
                  "full precision).")
            return None
        try:
            if lib == 'transformers':
                from transformers import QuantoConfig as QC
            else:
                from diffusers import QuantoConfig as QC
            return QC(weights='int2')
        except Exception as e:
            print(f"  2-bit (quanto) unavailable: {e}")
            return None

    def _mk(lib: str, mode: str):
        if mode == '2bit':
            return _quanto_cfg(lib)
        BnB = TfBnb if lib == 'transformers' else DiffBnb
        if mode == '4bit':
            return BnB(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                       bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        return BnB(load_in_8bit=True)

    def _bnb_incompatible(name: str) -> bool:
        # bitsandbytes (4/8-bit) and optimum-quanto (2-bit) only quantize
        # nn.Linear. A fully-convolutional component (the VAE) has no Linear
        # layers, so applying them triggers a hard "no linear modules were found"
        # error. Such components must stay full precision (a smaller VAE comes
        # from a GGUF VAE instead, handled separately).
        n = (name or '').lower()
        return n == 'vae' or n.endswith('_vae') or n.startswith('vae')

    quant_mapping: Dict[str, Any] = {}
    descs = []
    if comp_q:
        for name, raw_mode in comp_q.items():
            mode = _normalize_quant_mode(raw_mode)  # GGUF/none → None here
            if mode is None:
                continue
            if _bnb_incompatible(name):
                print(f"  Skipping {mode} for '{name}': it has no Linear layers "
                      f"(conv-only) — bitsandbytes/quanto cannot quantize it; "
                      f"leaving full precision (use a GGUF VAE to shrink the VAE).")
                continue
            cfg_obj = _mk(comp_lib.get(name, 'diffusers'), mode)
            if cfg_obj is not None:
                quant_mapping[name] = cfg_obj
                descs.append(f"{name}:{mode}")
    else:
        mode = '4bit' if global_4 else '8bit'
        targets = [n for n in comp_lib if _is_heavy(n)] or \
            ['transformer', 'transformer_2', 'text_encoder', 'unet']
        for name in targets:
            if _bnb_incompatible(name):
                continue
            cfg_obj = _mk(comp_lib.get(name, 'diffusers'), mode)
            if cfg_obj is not None:
                quant_mapping[name] = cfg_obj
                descs.append(f"{name}:{mode}")

    if not quant_mapping:
        return None, ''
    try:
        return PipelineQuantizationConfig(quant_mapping=quant_mapping), ', '.join(descs)
    except Exception as e:
        print(f"  Pipeline quantization build failed: {e}")
        return None, ''


_SINGLE_FILE_SUFFIXES = ('.safetensors', '.ckpt')

# Tried in order when the config does not name a pipeline class. Each one's
# from_single_file infers the rest of the architecture from the checkpoint's own
# tensors, so the first class whose inference succeeds is the right one; SDXL
# comes first because that is what the community ships most of.
_SINGLE_FILE_CLASSES = ('StableDiffusionXLPipeline', 'StableDiffusionPipeline')


def is_single_file_ref(ref) -> bool:
    """True when a model path names ONE checkpoint file, not a repo.

    Two shapes are accepted, because community checkpoints arrive as both:

      ``/AI/models/someMix_v18.safetensors``   a file on this machine
      ``org/repo:someMix_v18.safetensors``     one file out of a repo that
                                               holds several versions
    """
    s = str(ref or '').strip()
    if not s:
        return False
    head = s.rpartition(':')[2] if ':' in s else s
    return head.lower().endswith(_SINGLE_FILE_SUFFIXES)


def resolve_single_file(ref) -> str:
    """The local path of the checkpoint `ref` names, fetching it if needed."""
    s = str(ref or '').strip()
    if os.path.isfile(s):
        return s
    if ':' in s:
        repo, _, filename = s.rpartition(':')
        repo = repo.strip()
        if repo and filename:
            from huggingface_hub import hf_hub_download
            # Already-downloaded files resolve from the cache without network.
            return hf_hub_download(repo, filename)
    raise FileNotFoundError(f"single-file checkpoint not found: {ref}")


def build_single_file_pipeline(model_name: str, cfg: Optional[Dict[str, Any]],
                               dtype, extra: Optional[Dict[str, Any]] = None):
    """Build a pipeline from a single ComfyUI/A1111-style checkpoint file.

    Returns None when `model_name` is not a single-file reference, so callers
    can use it as a branch on their ordinary repo load.

    Civitai — which is where most community fine-tunes live — ships one
    ``.safetensors`` holding the whole pipeline, not a diffusers directory
    tree. ``from_pretrained`` cannot open one, and until this existed such a
    checkpoint could only be served if somebody had published a converted copy
    on the Hub. Conversions lag: the fine-tune asked for here (v18) had
    conversions only up to v14.1.
    """
    if not is_single_file_ref(model_name):
        return None

    import diffusers
    path = resolve_single_file(model_name)
    c = _norm(cfg)

    named = (c.get('pipeline_class') or '').strip()
    candidates = [named] if named else list(_SINGLE_FILE_CLASSES)

    kwargs: Dict[str, Any] = {'torch_dtype': dtype}
    # Quantization is honoured when the class accepts it (a 12B single-file
    # transformer needs it on a 24 GB card); retried without it otherwise,
    # rather than failing a load that would have worked unquantized.
    if extra and extra.get('quantization_config') is not None:
        kwargs['quantization_config'] = extra['quantization_config']

    errors = []
    for cls_name in candidates:
        cls = getattr(diffusers, cls_name, None)
        if cls is None or not hasattr(cls, 'from_single_file'):
            errors.append(f"{cls_name}: no such pipeline class with from_single_file")
            continue
        for attempt_kwargs in ([kwargs, {'torch_dtype': dtype}]
                               if 'quantization_config' in kwargs else [kwargs]):
            try:
                print(f"  Loading single-file checkpoint as {cls_name}: {path}")
                return cls.from_single_file(path, **attempt_kwargs)
            except TypeError as e:
                errors.append(f"{cls_name}: {e}")
                continue        # unsupported kwarg — retry without quantization
            except Exception as e:
                errors.append(f"{cls_name}: {e}")
                break           # this class is wrong for the file; try the next
    raise RuntimeError(
        "could not load single-file checkpoint " + str(path) + ". Tried: "
        + "; ".join(errors)
        + ". Set the model's pipeline_class if the architecture is not one "
          "from_single_file can infer.")


def build_gguf_pipeline_components(model_name: str, cfg: Optional[Dict[str, Any]], dtype):
    """Load pipeline components from GGUF files (Q2_K..Q8_0 — incl. 5/6-bit).

    For each ``component_quantization`` entry whose value is a ``*.gguf`` path or
    URL, load that component via ``<Class>.from_single_file(..., GGUFQuantization
    Config)`` so it can be passed to the pipeline's ``from_pretrained`` as a
    pre-built component (e.g. ``transformer=<model>``).  The bit-width (Q5_K,
    Q6_K, …) is embedded in the GGUF file itself.

    Returns ``(components_dict, description)``; empty dict when none configured.
    Only diffusers components (transformer*/unet/vae) are supported here;
    GGUF text encoders are uncommon and skipped with a note.
    """
    c = _norm(cfg)
    comp_q = c.get('component_quantization') or {}
    gguf_entries = {n: v for n, v in comp_q.items() if _is_gguf_value(v)}
    if not gguf_entries:
        return {}, ''

    try:
        import diffusers
        from diffusers import GGUFQuantizationConfig
    except Exception as e:
        print(f"  GGUF components unavailable: {e}")
        return {}, ''

    specs = _discover_components(model_name)  # name -> [library, class_name]
    components: Dict[str, Any] = {}
    descs = []
    for name, path in gguf_entries.items():
        spec = specs.get(name)
        if not spec or spec[0] != 'diffusers':
            print(f"  GGUF skip '{name}': only diffusers components supported "
                  f"(got {spec}).")
            continue
        cls_name = spec[1] if len(spec) > 1 else None
        cls = getattr(diffusers, cls_name, None) if cls_name else None
        if cls is None or not hasattr(cls, 'from_single_file'):
            print(f"  GGUF skip '{name}': no loadable class {cls_name}.")
            continue
        try:
            print(f"  Loading GGUF component '{name}' from {path}")
            # `config` names the architecture the file's tensors belong to.
            # Without it, from_single_file INFERS one -- and its fallback is
            # Stable Diffusion 1.5, so a Qwen-Image / Flux / Wan transformer
            # fails with "stable-diffusion-v1-5 does not appear to have a file
            # named config.json". The base pipeline's own component subfolder
            # is the right answer and is already at hand.
            model = cls.from_single_file(
                path.strip(),
                config=model_name,
                subfolder=name,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
            components[name] = model
            descs.append(f"{name}:gguf")
        except Exception as e:
            print(f"  GGUF load failed for '{name}' ({path}): {e}")
    return components, ', '.join(descs)


def build_component_models(model_name: str, cfg: Optional[Dict[str, Any]], dtype):
    """Components taken from a DIFFERENT repo than the pipeline's own.

    ``component_models`` maps a component name to a repo id or local path, e.g.
    ``{"transformer": "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19"}``. The
    named component is loaded from there and handed to the pipeline's
    ``from_pretrained`` as a prebuilt object, exactly as the GGUF components
    are.

    This is how a fine-tuned transformer is served without republishing the
    whole pipeline around it. Several Qwen-Image-Edit derivatives ship as a
    transformer and nothing else -- the base supplies the VAE, the text encoder
    and the scheduler -- and the upstream recipe for them is literally
    ``QwenImageEditPlusPipeline.from_pretrained(base, transformer=
    QwenImageTransformer2DModel.from_pretrained(derivative))``.

    The component's own entry in ``component_quantization`` still applies, so a
    bf16 or fp8 transformer too large for the card can be quantized as it
    loads: ``{"transformer": "4bit"}`` alongside the override above.

    Returns ``(components_dict, description)``; empty when none configured.
    """
    c = _norm(cfg)
    overrides = c.get('component_models') or {}
    if not overrides:
        return {}, ''

    try:
        import diffusers
    except Exception as e:
        print(f"  component_models unavailable: {e}")
        return {}, ''

    comp_q = c.get('component_quantization') or {}
    specs = _discover_components(model_name)   # name -> [library, class_name]
    components: Dict[str, Any] = {}
    descs = []
    for name, source in overrides.items():
        source = str(source or '').strip()
        if not source:
            continue
        spec = specs.get(name)
        cls_name = spec[1] if spec and len(spec) > 1 else None
        cls = getattr(diffusers, cls_name, None) if cls_name else None
        if cls is None:
            print(f"  component_models skip '{name}': the base pipeline has no "
                  f"loadable class for it (got {spec}).")
            continue
        kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        mode = str(comp_q.get(name) or '').strip().lower()
        if mode in ("4bit", "8bit"):
            try:
                from diffusers import BitsAndBytesConfig as DiffBnb
                kwargs["quantization_config"] = DiffBnb(
                    **({"load_in_4bit": True, "bnb_4bit_compute_dtype": dtype}
                       if mode == "4bit" else {"load_in_8bit": True}))
            except Exception as e:
                print(f"  component_models '{name}': {mode} unavailable ({e}), "
                      f"loading unquantized")
        try:
            print(f"  Loading component '{name}' from {source}"
                  f"{' (' + mode + ')' if mode else ''}")
            components[name] = cls.from_pretrained(source, **kwargs)
            descs.append(f"{name}:{source.split('/')[-1]}{'/' + mode if mode else ''}")
        except Exception as e:
            # RAISE. A component named here IS the model: `component_models`
            # is how one published id serves a fine-tuned transformer on
            # another pipeline's VAE and text encoder. Skipping it loads the
            # BASE component instead, so the id quietly serves a different
            # model than its name promises -- and the load goes on to succeed,
            # which is the failure nobody can see.
            #
            # Measured in production 2026-10-25: this skipped the V19
            # transformer ("Blockwise 4bit quantization only supports 16/32-bit
            # floats, but got torch.float8_e4m3fn"), the pipeline fell back to
            # plain Qwen-Image-Edit-2511, then OOMed trying to hold it whole --
            # and the client was told the model id did not exist.
            hint = ""
            if "float8" in str(e) or "fp8" in str(e).lower():
                hint = (f" The source is stored in fp8, which bitsandbytes "
                        f"cannot quantize from (it needs 16- or 32-bit "
                        f"floats). Serve this component from a GGUF of it "
                        f"instead -- component_quantization accepts a .gguf "
                        f"path -- or drop '{name}' from component_quantization "
                        f"and give the model enough room to load it as it is.")
            raise RuntimeError(
                f"component '{name}' could not be loaded from {source!r}"
                f"{' as ' + mode if mode else ''}: {e}.{hint}") from e
    return components, ', '.join(descs)


def build_cached_components(model_name: str, cfg: Optional[Dict[str, Any]], dtype):
    """Incremental per-component cache for big bnb-quantized video pipelines.

    A Wan2.2 A14B pipeline often won't fit on the GPU and can only load with
    offload — and an offloaded pipeline can't be save_pretrained wholesale. But
    each component IS small enough to pass through the GPU alone, where bnb
    quantizes it to a uniform 4-bit. So for every bnb-quantizable heavy component
    we: load it from the per-component cache if present, else build it fresh from
    the model (quantizing on the GPU) and cache it — then move it to CPU to free
    the GPU slot for the next one. Peak VRAM is ONE component even when the whole
    model doesn't fit; the result is injected into the pipeline's from_pretrained
    exactly like GGUF components, and the offload ladder places them at runtime.

    Caching is incremental and resumable: a load evicted before all components are
    cached leaves the finished ones valid, and the next load mixes cached with
    freshly-built (now cached) ones until the set is complete.

    Returns ``(components_dict, description)``. Empty when disabled (kill-switch
    CODERAI_INCREMENTAL_CACHE=0, --pipeline-cache off, or no bnb quant). Fully
    fallback-safe: any per-component failure skips that component (the normal
    pipeline load handles it) and is logged."""
    import os
    if os.environ.get("CODERAI_INCREMENTAL_CACHE", "1") == "0":
        return {}, ''
    c = _norm(cfg)
    comp_q = c.get('component_quantization') or {}
    global_4 = bool(c.get('load_in_4bit', False))
    global_8 = bool(c.get('load_in_8bit', False))
    if not comp_q and not (global_4 or global_8):
        return {}, ''
    try:
        from codai.models import pipeline_cache as _pcache
        if not _pcache.enabled():
            return {}, ''
    except Exception:
        return {}, ''
    try:
        import diffusers
        import transformers
        from diffusers import BitsAndBytesConfig as DiffBnb
        from transformers import BitsAndBytesConfig as TfBnb
    except Exception as e:
        print(f"  [pipeline-cache] incremental build unavailable: {e}")
        return {}, ''

    specs = _discover_components(model_name)  # name -> [library, class_name]

    def _is_heavy(name: str) -> bool:
        return (name.startswith('transformer') or name == 'unet'
                or name.startswith('text_encoder'))

    def _bnb_incompatible(name: str) -> bool:
        n = (name or '').lower()
        return n == 'vae' or n.endswith('_vae') or n.startswith('vae')

    # Which components to handle, and at what bnb width.
    targets: Dict[str, str] = {}
    if comp_q:
        for name, raw in comp_q.items():
            mode = _normalize_quant_mode(raw)  # gguf/none/2bit -> not bnb here
            if mode in ('4bit', '8bit') and not _bnb_incompatible(name) and name in specs:
                targets[name] = mode
    else:
        mode = '4bit' if global_4 else '8bit'
        for name in specs:
            if _is_heavy(name) and not _bnb_incompatible(name):
                targets[name] = mode
    if not targets:
        return {}, ''

    def _mk(lib: str, mode: str):
        BnB = TfBnb if lib == 'transformers' else DiffBnb
        if mode == '4bit':
            return BnB(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                       bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        return BnB(load_in_8bit=True)

    def _cls(name):
        spec = specs.get(name) or []
        lib = spec[0] if spec else 'diffusers'
        cls_name = spec[1] if len(spec) > 1 else None
        mod = transformers if lib == 'transformers' else diffusers
        return (getattr(mod, cls_name, None) if cls_name else None), lib

    components: Dict[str, Any] = {}
    descs = []
    for name, mode in targets.items():
        try:
            cls, lib = _cls(name)
            if cls is None or not hasattr(cls, 'from_pretrained'):
                continue
            cached = _pcache.component_valid(model_name, cfg, name)
            if cached:
                cdir = _pcache.component_dir(model_name, cfg, name)
                print(f"  [pipeline-cache] component '{name}' HIT — loading 4-bit from cache")
                comp = cls.from_pretrained(
                    cdir, quantization_config=_mk(lib, mode),
                    torch_dtype=dtype, device_map={'': 0})
            else:
                print(f"  [pipeline-cache] component '{name}' MISS — building 4-bit "
                      f"(one component on GPU, then cached)")
                comp = cls.from_pretrained(
                    model_name, subfolder=name, quantization_config=_mk(lib, mode),
                    torch_dtype=dtype, device_map={'': 0})
                _pcache.save_component(comp, model_name, cfg, name)
            # Free the single GPU slot for the next component; the offload ladder
            # re-places it at runtime. bnb 4-bit survives the move to CPU (verified).
            try:
                comp = comp.to('cpu')
                import torch as _t
                if _t.cuda.is_available():
                    _t.cuda.synchronize()
                    _t.cuda.empty_cache()
            except Exception:
                pass
            components[name] = comp
            descs.append(f"{name}:{mode}{'(cached)' if cached else '(built)'}")
        except Exception as e:
            print(f"  [pipeline-cache] component '{name}' incremental build failed "
                  f"({str(e).splitlines()[0] if str(e) else type(e).__name__}) — "
                  f"leaving it to the normal pipeline load")

    # If every heavy component is now cached, complete the cache DIR into a
    # normally-loadable pipeline (light components + model_index.json + marker), so
    # subsequent loads use the monolithic HIT path — from_pretrained(dir,
    # device_map=…) with NO injection. That lets big-clip generation fit via
    # device_map (the injected-component path is stuck on hook-based offload, where
    # 'model' OOMs big forwards and 'sequential' hits a bnb Params4bit bug).
    if components and len(components) == len(targets):
        try:
            finalize_monolithic_cache(model_name, cfg, dtype, set(components.keys()))
        except Exception as _fe:
            print(f"  [pipeline-cache] monolithic finalize skipped ({_fe})")

    return components, ', '.join(descs)


def finalize_monolithic_cache(model_name: str, cfg: Optional[Dict[str, Any]],
                              dtype, built_names) -> bool:
    """Complete a per-component cache DIR into a pipeline from_pretrained() can load
    directly: save the LIGHT components the heavy build didn't cover (vae /
    scheduler / tokenizer / …), copy model_index.json, then mark it complete.

    Best-effort and conservative: it marks the cache complete ONLY if every
    component named in model_index.json is present, so a partial dir is left
    UNMARKED (valid() stays False) and the injection path keeps working. Returns
    True only when the monolithic cache is fully written + marked."""
    import os
    import json
    try:
        from codai.models import pipeline_cache as _pcache
        import diffusers
        import transformers
    except Exception as e:
        print(f"  [pipeline-cache] finalize unavailable: {e}")
        return False
    cdir = _pcache.path(model_name, cfg)
    try:
        idx = diffusers.DiffusionPipeline.load_config(model_name)
    except Exception as e:
        print(f"  [pipeline-cache] finalize: can't read model_index ({e})")
        return False
    comps = {k: v for k, v in idx.items()
             if not k.startswith('_') and isinstance(v, (list, tuple))
             and len(v) >= 2 and v[1]}
    for name, spec in comps.items():
        sub = os.path.join(cdir, name)
        # Already cached? (heavy component subdir, or a light one from a prior run)
        if (os.path.isfile(os.path.join(sub, 'config.json'))
                or os.path.isfile(os.path.join(sub, 'tokenizer_config.json'))
                or os.path.isfile(os.path.join(sub, 'scheduler_config.json'))):
            continue
        lib, cls_name = spec[0], spec[1]
        mod = transformers if lib == 'transformers' else diffusers
        cls = getattr(mod, cls_name, None)
        if cls is None or not hasattr(cls, 'from_pretrained'):
            print(f"  [pipeline-cache] finalize: no loadable class for '{name}' "
                  f"({spec}) — leaving monolithic cache incomplete")
            return False
        try:
            print(f"  [pipeline-cache] finalize: caching light component '{name}'")
            try:
                obj = cls.from_pretrained(model_name, subfolder=name, torch_dtype=dtype)
            except TypeError:
                # tokenizer / scheduler don't accept torch_dtype
                obj = cls.from_pretrained(model_name, subfolder=name)
            obj.save_pretrained(sub)
            del obj
        except Exception as e:
            print(f"  [pipeline-cache] finalize: couldn't cache '{name}' ({e}) — "
                  f"leaving monolithic cache incomplete")
            return False
    try:
        with open(os.path.join(cdir, 'model_index.json'), 'w') as f:
            json.dump(idx, f)
    except Exception as e:
        print(f"  [pipeline-cache] finalize: couldn't write model_index.json ({e})")
        return False
    ok = _pcache.mark_monolithic_complete(model_name, cfg)
    if ok:
        print(f"  [pipeline-cache] monolithic cache complete → next load uses "
              f"device_map from cache (no injection)")
    return ok


def build_from_pretrained_kwargs(
    cfg: Optional[Dict[str, Any]],
    *,
    default_precision: str = 'bf16',
    enable_flash: bool = True,
) -> Dict[str, Any]:
    """Build common ``from_pretrained`` kwargs from a coderai model config.

    Honours: load_in_4bit/8bit, flash_attention, offload_strategy, offload_dir,
    max_gpu_percent, manual_ram_gb, no_ram, precision.
    """
    import torch
    c = _norm(cfg)
    kwargs: Dict[str, Any] = {
        'trust_remote_code': True,
        'low_cpu_mem_usage': True,
        'torch_dtype': resolve_dtype(cfg, default_precision),
    }

    # Quantization (transformers BitsAndBytesConfig)
    quant = build_quantization_config(cfg)
    if quant is not None:
        kwargs['quantization_config'] = quant
        bits = 4 if c.get('load_in_4bit') else 8
        print(f"  HF quantization: {bits}-bit (bitsandbytes)")

    # Flash attention — honour any of the three flash flags (the sdcpp ones are
    # no-ops for transformers models, so an enabled sdcpp flag still signals the
    # user's intent to use Flash-Attention-2 here).
    _flash = (c.get('flash_attention', c.get('flash_attn', False))
              or c.get('sdcpp_flash_attn', False)
              or c.get('sdcpp_diffusion_flash_attn', False))
    if enable_flash and _flash:
        try:
            import flash_attn  # noqa: F401
            kwargs['attn_implementation'] = 'flash_attention_2'
            print("  Flash Attention 2 enabled")
        except Exception:
            print("  Flash Attention 2 requested but not installed — ignoring")

    # Offload / device placement
    no_ram = bool(c.get('no_ram', False))
    offload_strategy = (c.get('offload_strategy') or 'auto')
    max_gpu_percent = c.get('max_gpu_percent')

    if not torch.cuda.is_available():
        return kwargs  # CPU-only host; let transformers place on CPU

    if no_ram or offload_strategy == 'none':
        # Everything on GPU, no CPU spill.
        kwargs['device_map'] = {'': 0}
        return kwargs

    # Build a max_memory map so large models split GPU → CPU RAM → disk.
    try:
        import psutil
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        headroom = 512 * 1024 * 1024
        if max_gpu_percent is not None:
            gpu_budget = int(total_vram * max(0.0, min(1.0, float(max_gpu_percent) / 100.0)))
            gpu_budget = min(gpu_budget, max(0, free_vram - headroom))
        else:
            gpu_budget = max(0, free_vram - headroom)

        manual_ram_gb = c.get('manual_ram_gb')
        if manual_ram_gb:
            cpu_budget = int(float(manual_ram_gb) * 1e9)
        else:
            cpu_budget = max(0, psutil.virtual_memory().available - int(4e9))

        # Global RAM cap: clamp the CPU-offload budget to the headroom remaining under
        # the server-wide ceiling, so the overflow spills to the disk offload folder
        # (set below) instead of pushing process RSS past the cap. Read live from
        # global_args (same pattern as pipeline_cache). None = no cap (legacy behaviour).
        try:
            from codai.api.state import get_global_args as _gga
            _ga = _gga()
            _cap = getattr(_ga, 'max_ram_gb', None) if _ga else None
            if _cap:
                _used = psutil.Process().memory_info().rss
                _headroom = int(float(_cap) * 1e9) - _used
                # Keep a small floor so a single component can still land in RAM; the
                # rest goes to disk. Never raise the budget above the cap headroom.
                cpu_budget = max(0, min(cpu_budget, _headroom))
        except Exception:
            pass

        kwargs['device_map'] = 'auto'
        kwargs['max_memory'] = {0: gpu_budget, 'cpu': cpu_budget}

        # Disk overflow when offloading is allowed. Resolve to an absolute writable
        # dir (a relative per-model './offload' inherits the configured global dir).
        offload_dir = resolve_offload_dir(c.get('offload_dir'))
        os.makedirs(offload_dir, exist_ok=True)
        kwargs['offload_folder'] = offload_dir
        kwargs['offload_buffers'] = True
    except Exception as e:
        print(f"  Could not build offload map ({e}); loading with device_map=auto")
        kwargs['device_map'] = 'auto'

    return kwargs


def pipeline_device_kwargs(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return kwargs for HF ``pipeline(...)`` honouring quantization/offload.

    HF pipelines accept ``model_kwargs`` (passed to from_pretrained) and a
    ``device_map``/``torch_dtype``.  We funnel the same config through.
    """
    base = build_from_pretrained_kwargs(cfg)
    pk: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {}
    for k in ('quantization_config', 'attn_implementation', 'max_memory',
              'offload_folder', 'offload_buffers', 'low_cpu_mem_usage',
              'trust_remote_code'):
        if k in base:
            model_kwargs[k] = base[k]
    if 'torch_dtype' in base:
        pk['torch_dtype'] = base['torch_dtype']
    if 'device_map' in base:
        pk['device_map'] = base['device_map']
    if model_kwargs:
        pk['model_kwargs'] = model_kwargs
    return pk
