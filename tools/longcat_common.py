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

"""LongCat-Video checkpoint + segment contracts, shared by the engine and the worker.

Loaded BY PATH from both sides (``importlib.util.spec_from_file_location``): the isolated
Python 3.10 venv cannot import ``codai.*`` — that would pull FastAPI and the whole server
into it — and the engine must not import the worker's torch. So anything both sides have
to agree on lives here, in plain stdlib Python that runs on 3.10 and 3.13 alike.

Nothing here imports torch, PIL or the ``longcat_video`` package.
"""

import os

# The checkpoint layout the upstream pipeline expects under one directory: every
# from_pretrained call takes `checkpoint_dir` plus one of these subfolders.
CHECKPOINT_SUBDIRS = ("tokenizer", "text_encoder", "vae", "scheduler", "dit")

# The avatar checkpoints are a DiT and an audio encoder, nothing else: no tokenizer,
# no text_encoder, no vae. Their README says so in its frontmatter
# (`base_model: meituan-longcat/LongCat-Video`) and upstream's own instructions
# download BOTH repos. So an avatar family borrows the shared components from the base
# checkpoint and brings only its own transformer, scheduler and LoRA.
SHARED_SUBDIRS = ("tokenizer", "text_encoder", "vae")
AVATAR_OWN_SUBDIRS = ("scheduler",)
# Their bf16 transformer is under base_model/, where the base checkpoint uses dit/.
AVATAR_DIT_SUBDIR = "base_model"
BASE_DIT_SUBDIR = "dit"
# Where the shared components come from when none is configured.
BASE_REPO = "meituan-longcat/LongCat-Video"

# The two LoRAs that make stages 2 and 3 what they are, under `lora/`.
LORA_FILES = {
    "distill": "lora/cfg_step_lora.safetensors",
    "refinement": "lora/refinement_lora.safetensors",
}

# Upstream's own numbers for one generation unit.
DEFAULT_NUM_FRAMES = 93          # frames produced per pipeline call
DEFAULT_COND_FRAMES = 13         # of those, how many are the previous segment's tail
DEFAULT_BASE_SIZE = (480, 832)   # (height, width) for the base stage
DEFAULT_FPS = 15                 # base/distill write 15; refinement may write 30

# The three coarse-to-fine stages, and which LoRA each needs. Stage 3 takes stage 1's
# frames as input, which is what makes the stages separable and resumable.
STAGES = ("base", "distill", "refinement")

# What each quality preset runs. "fast" is the default: the distilled stage is 16 steps
# against the base stage's 50 for output most callers cannot tell apart.
PRESETS = {
    "draft": ("base",),
    "fast": ("distill",),
    "best": ("base", "refinement"),
}

STAGE_DEFAULTS = {
    #            steps, guidance
    "base":        (50, 4.0),
    "distill":     (16, 1.0),
    "refinement":  (50, 1.0),
}


# ── variants ─────────────────────────────────────────────────────────────────
# Which checkpoint family a model entry points at, and what each supports. Getting this
# wrong is not a crash but a wrong answer: --use_int8 and the DMD distillation exist only
# for avatar-v1.5, and asking for them elsewhere would silently load something else.
VARIANTS = ("bf16", "fp8", "int8", "gguf")

# The avatar families are separate HuggingFace repos with their own audio encoders.
AVATAR_ENCODERS = {
    "avatar": "chinese-wav2vec2-base",      # v1.0
    "avatar-1.5": "whisper-large-v3",
}

# Only avatar-1.5 ships an INT8 DiT (under base_model_int8/) and the DMD LoRA, and its
# distilled path is REQUIRED rather than optional.
INT8_FAMILIES = ("avatar-1.5",)
DMD_FAMILIES = ("avatar-1.5",)
INT8_SUBDIR = "base_model_int8"
DMD_LORA = "lora/dmd_lora.safetensors"
DMD_NETWORK_DIM = 128
DMD_NETWORK_ALPHA = 64

# The avatar distilled pass is NOT the base model's: 8 steps against dmd_lora, where the
# base model's is 16 against cfg_step_lora.
AVATAR_STAGE_DEFAULTS = {
    #            steps, text guidance, audio guidance
    "base":        (50, 4.0, 4.0),
    "distill":     (8, 1.0, 1.0),
}

AVATAR_TASKS = ("at2v", "ai2v")


def family_of(model_path: str = "", variant: str = "") -> str:
    """Which checkpoint family a path names: '', 'avatar' or 'avatar-1.5'.

    Read from the path because the family IS the repo — meituan-longcat/LongCat-Video,
    -Avatar and -Avatar-1.5 are three downloads with different audio encoders."""
    h = str(model_path or "").lower().replace("_", "-")
    if "avatar-1.5" in h or "avatar1.5" in h:
        return "avatar-1.5"
    if "avatar" in h:
        return "avatar"
    return ""


def variant_problems(variant: str, family: str = "", use_int8: bool = False,
                     use_distill: bool = False, checkpoint_dir: str = "") -> list:
    """Reasons this combination cannot be served, or [].

    Checked before loading so an unsupported flag is an error the operator can act on,
    not a silently different model."""
    problems = []
    v = (variant or "bf16").strip().lower()
    if v not in VARIANTS:
        problems.append(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    if v == "gguf":
        problems.append(
            "the GGUF variants are packaged for ComfyUI's gguf loader, which coderai "
            "does not implement — use bf16, fp8, or the official int8 (avatar-1.5)")
    if v == "int8" or use_int8:
        # Upstream SHIPS int8 only for avatar-1.5 — but the model page can now build
        # it for any checkpoint (tools/longcat_quantize.py), so what decides this is
        # whether the weights are actually there, not which family wrote them.
        built = bool(checkpoint_dir) and os.path.isdir(
            os.path.join(os.path.expanduser(checkpoint_dir), INT8_SUBDIR))
        if not built and family not in INT8_FAMILIES:
            problems.append(
                f"no {INT8_SUBDIR}/ in this checkpoint, and upstream ships INT8 only "
                f"for {'/'.join(INT8_FAMILIES)} (this reads as "
                f"{family or 'the base model'}) — use “Build INT8 weights” on the "
                f"model page first")
    if use_distill and family and family not in DMD_FAMILIES:
        problems.append(
            f"the DMD distillation exists only for {'/'.join(DMD_FAMILIES)}; "
            f"this checkpoint reads as {family}")
    if family == "avatar-1.5" and not (use_distill or v == "int8"):
        # Upstream states v1.5 requires distilled sampling.
        problems.append(
            "avatar-1.5 requires distilled sampling — set use_distill on the model "
            "entry (its 8-step DMD pass is the supported path)")
    return problems


def avatar_stage_params(stage: str, steps=None, text_guidance=None,
                        audio_guidance=None) -> dict:
    """Steps and the TWO guidance scales the avatar pipeline takes."""
    key = stage if stage in AVATAR_STAGE_DEFAULTS else "base"
    base_steps, base_text, base_audio = AVATAR_STAGE_DEFAULTS[key]
    return {
        "num_inference_steps": int(steps) if steps else base_steps,
        "text_guidance_scale": float(text_guidance) if text_guidance is not None
                               else base_text,
        "audio_guidance_scale": float(audio_guidance) if audio_guidance is not None
                                else base_audio,
    }


def avatar_problems(checkpoint_dir: str, family: str, use_int8: bool = False,
                    use_distill: bool = False) -> list:
    """What an avatar checkpoint is missing for this configuration, or []."""
    if not family:
        return ["this checkpoint is not an avatar model (its path names no avatar "
                "family) — the audio-driven tasks need LongCat-Video-Avatar or "
                "-Avatar-1.5"]
    root = os.path.expanduser(checkpoint_dir or "")
    problems = []
    if use_int8 and not os.path.isdir(os.path.join(root, INT8_SUBDIR)):
        problems.append(f"use_int8 needs {INT8_SUBDIR}/ in {root}")
    if use_distill and not os.path.isfile(os.path.join(root, DMD_LORA)):
        problems.append(f"use_distill needs {DMD_LORA} in {root}")
    return problems


def resolve_stages(preset: str = "fast", stage: str = "") -> tuple:
    """Which stages one request runs.

    An explicit ``stage`` wins — that is the resumable per-stage path, where the caller
    drives each stage itself. Otherwise the preset decides."""
    s = (stage or "").strip().lower()
    if s:
        if s not in STAGES:
            raise ValueError(f"unknown stage {s!r}; expected one of {STAGES}")
        return (s,)
    p = (preset or "fast").strip().lower()
    if p not in PRESETS:
        raise ValueError(f"unknown quality preset {p!r}; expected one of "
                         f"{tuple(PRESETS)}")
    return PRESETS[p]


def stage_params(stage: str, steps=None, guidance=None) -> dict:
    """Steps/guidance for a stage, with the caller's overrides applied."""
    base_steps, base_guidance = STAGE_DEFAULTS[stage]
    return {
        "num_inference_steps": int(steps) if steps else base_steps,
        "guidance_scale": float(guidance) if guidance is not None else base_guidance,
    }


def segments_for(total_frames: int, num_frames: int = DEFAULT_NUM_FRAMES,
                 cond_frames: int = DEFAULT_COND_FRAMES) -> int:
    """How many pipeline calls produce at least ``total_frames``.

    Each call emits ``num_frames`` but the first ``cond_frames`` of them re-render the
    previous segment's tail, so a segment only ADDS ``num_frames - cond_frames``. Getting
    this wrong silently truncates long output, so both sides compute it here."""
    new_per_segment = max(1, int(num_frames) - int(cond_frames))
    if total_frames <= int(num_frames):
        return 1
    extra = int(total_frames) - int(num_frames)
    return 1 + (extra + new_per_segment - 1) // new_per_segment


def frames_for(segments: int, num_frames: int = DEFAULT_NUM_FRAMES,
               cond_frames: int = DEFAULT_COND_FRAMES) -> int:
    """The inverse: how many frames ``segments`` calls yield."""
    segments = max(1, int(segments))
    return int(num_frames) + (segments - 1) * (int(num_frames) - int(cond_frames))


def duration_seconds(segments: int, fps: int = DEFAULT_FPS,
                     num_frames: int = DEFAULT_NUM_FRAMES,
                     cond_frames: int = DEFAULT_COND_FRAMES) -> float:
    return frames_for(segments, num_frames, cond_frames) / float(max(1, fps))


def resolve_checkpoint(ref: str) -> str:
    """A checkpoint reference resolved to a directory on disk.

    models.json stores what the model page registered — "meituan-longcat/LongCat-Video",
    a repo id, not a path. Everything downstream opens files BY PATH (config.json, the
    VAE, the INT8 shards), so handing the id straight on made every load fail with
    "checkpoint directory does not exist" before a byte was read.

    Resolved against the HF hub cache by hand rather than through huggingface_hub, so
    the server's 3.13 venv and the service's 3.10 venv cannot disagree about where the
    weights are, and so a checkpoint that was never downloaded stays an error instead of
    becoming a 74 GB download nobody asked for. An unresolvable reference is returned
    unchanged, which leaves the existing "does not exist" message to name it.
    """
    import glob

    if not ref:
        return ref
    path = os.path.expanduser(ref)
    if os.path.isdir(path) or os.path.isabs(path) or "/" not in ref:
        return path

    roots = [os.environ[v] for v in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
             if os.environ.get(v)]
    if os.environ.get("HF_HOME"):
        roots.append(os.path.join(os.environ["HF_HOME"], "hub"))
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))

    folder = "models--" + ref.replace("/", "--")
    for root in roots:
        snaps = [d for d in glob.glob(os.path.join(root, folder, "snapshots", "*"))
                 if os.path.isdir(d)]
        if snaps:
            # Newest wins, so a re-pulled revision beats a stale one left behind.
            return max(snaps, key=os.path.getmtime)
    return path


# The VAE's spatial stride (8) times the patch size (2): a latent cell per 16 pixels.
VAE_SIDE_STRIDE = 16
VAE_TEMPORAL_STRIDE = 4
# What the checkpoint asks for when block-sparse attention is on.
DEFAULT_BSA_CHUNK = (4, 4, 4)


def bsa_problems(num_frames: int, height: int, width: int, chunk=None) -> list:
    """Why block-sparse attention cannot run THIS geometry, or [].

    flash_attn_bsa_3d tiles the latent into 3D chunks and asserts the latent divides
    evenly by the chunk on every axis:

        assert Tq % tq == 0 and Hq % hq == 0 and Wq % wq == 0

    With the shipped chunk of (4, 4, 4) that means sides divisible by 64 — the VAE's
    16-pixel latent cell times 4 — and a latent frame count divisible by 4, which puts
    num_frames at 13, 29, 45, 61, 77, 93 … Upstream's own default 480x832 is NOT one of
    them (480/16 = 30), which is presumably why the flag ships off: an AssertionError
    out of the first denoising step says nothing about geometry.
    """
    tq, hq, wq = tuple(chunk or DEFAULT_BSA_CHUNK)
    problems = []
    try:
        frames, h, w = int(num_frames), int(height), int(width)
    except (TypeError, ValueError):
        return ["block-sparse attention needs numeric num_frames/height/width"]

    for name, value, need in (("height", h, hq), ("width", w, wq)):
        cell = VAE_SIDE_STRIDE * need
        if value % cell:
            nearest = max(cell, round(value / cell) * cell)
            problems.append(
                f"{name} must be divisible by {cell} for block-sparse attention "
                f"({VAE_SIDE_STRIDE}px latent cell x chunk {need}); got {value}, "
                f"try {nearest}")

    latent_t = (frames - 1) // VAE_TEMPORAL_STRIDE + 1
    if (frames - 1) % VAE_TEMPORAL_STRIDE or latent_t % tq:
        # latent depth is (F-1)/4 + 1, and it must divide by tq — which puts the valid
        # frame counts at step*k + base, e.g. 13, 29, 45 … for a chunk of 4, NOT the
        # step*k + 1 that the 4n+1 generation rule would suggest.
        step = VAE_TEMPORAL_STRIDE * tq
        base = step - VAE_TEMPORAL_STRIDE + 1
        lower = ((frames - base) // step) * step + base if frames >= base else base
        options = [n for n in (lower, lower + step) if n >= base]
        problems.append(
            f"num_frames must leave a latent depth divisible by {tq} for block-sparse "
            f"attention ({step}n+1: 13, 29, 45, 61, 77, 93 …); got {frames}"
            + (f", try {' or '.join(str(n) for n in options)}" if options else ""))
    return problems


# Weight quantisation for the pipeline COMPONENTS, as bytes per element once loaded.
# The DiT has upstream's own INT8 path (base_model_int8/); these cover the components
# that do not, the UMT5-XXL text encoder above all: at bf16 it is ~11 GB, which is 45%
# of why a 13.6 GB INT8 DiT still does not fit a 24 GB card.
COMPONENT_QUANTS = {
    "none": None,       # load at the pipeline dtype
    "int8": 1.0,
    "nf4": 0.5,         # 4 bits plus absmax/quant-map overhead, which rounds to this
    "fp4": 0.5,
}

# What a quantised component ACTUALLY costs, as a fraction of its bf16 size — MEASURED
# on UMT5-XXL on a 3090, not derived from the bits per weight. bitsandbytes replaces
# Linear layers only: 144 of them here, while the 256k-token embedding table and one
# linear per layer stay at the load dtype. That leaves 6.13 GB unquantisable, so the
# bits-per-weight arithmetic was ~3x too optimistic (it predicted 2.8 GB for NF4; the
# card sees 8.0).
COMPONENT_QUANT_RESIDENT_FRACTION = {
    "int8": 0.89,       # 9.76 GB of an 11.0 GB bf16 encoder
    "nf4": 0.73,        # 8.00 GB
    "fp4": 0.73,
}


def component_quant_fraction(value: str):
    """Fraction of the bf16 size a quantised component keeps, or None for no change."""
    return COMPONENT_QUANT_RESIDENT_FRACTION.get(str(value or "none").strip().lower())


def component_quant_problems(name: str, value: str) -> list:
    """Reasons this component cannot be quantised that way, or []."""
    v = str(value or "none").strip().lower()
    if v not in COMPONENT_QUANTS:
        return [f"{name} must be one of {sorted(COMPONENT_QUANTS)}; got {value!r}"]
    return []


def component_quant_bpe(value: str):
    """Bytes per element for a quantised component, or None to keep the load dtype."""
    return COMPONENT_QUANTS.get(str(value or "none").strip().lower())


# ── prompt-embedding cache ────────────────────────────────────────────────────
#
# The UMT5-XXL text encoder is ~11 GB and runs ONCE per request, to turn a prompt into
# a few MB of embeddings. v0.2.76 stopped re-running it per segment; caching the result
# on DISK stops re-running it per request, and — because the entry is keyed on
# everything that can change the answer — lets the encoder be loaded LAZILY: a request
# whose prompt is already cached never builds it at all, which is ~11 GB of host RAM
# and a minute of load time that nothing has to spend.
#
# Off by default. Embeddings are derived data, but they are also several MB each and
# live under the user's cache directory, so growing that is something to opt into.
PROMPT_CACHE_MODES = ("off", "disk")

# Each entry is prompt_embeds + mask + the negative pair, 512x4096 at bf16 ≈ 4 MB per
# side. A cap rather than a count, because a long prompt list is the case that runs away.
DEFAULT_PROMPT_CACHE_MAX_GB = 2.0

# Bumped when the stored tuple's shape or the key's inputs change, so a cache written by
# an older build is ignored rather than misread.
PROMPT_CACHE_FORMAT = 1


def prompt_cache_problems(mode: str, max_gb=None) -> list:
    """Reasons this prompt-cache configuration cannot be used, or []."""
    problems = []
    m = str(mode or "off").strip().lower()
    if m not in PROMPT_CACHE_MODES:
        problems.append(f"prompt_cache must be one of {sorted(PROMPT_CACHE_MODES)}; "
                        f"got {mode!r}")
    if max_gb is not None and str(max_gb).strip() != "":
        try:
            if float(max_gb) <= 0:
                problems.append(f"prompt_cache_max_gb must be positive; got {max_gb!r}")
        except (TypeError, ValueError):
            problems.append(f"prompt_cache_max_gb must be a number; got {max_gb!r}")
    return problems


def prompt_cache_key(*, checkpoint: str, dtype: str, text_encoder_quant: str,
                     args: tuple = (), kwargs: dict = None) -> str:
    """A stable filename-safe key for one encode_prompt call.

    Keyed on everything that changes the EMBEDDINGS and nothing that does not. The
    checkpoint and the encoder's quantisation are in here because an NF4 encoder does
    not produce the bf16 encoder's numbers -- reusing one for the other would be a
    silent quality change, which is worse than a cache miss. ``device`` is deliberately
    excluded: it moves the result, it does not alter it.
    """
    import hashlib
    import json as _json

    def _plain(v):
        """Reduce an argument to something JSON can hold, deterministically."""
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [_plain(x) for x in v]
        if isinstance(v, set):
            return sorted(_plain(x) for x in v)
        if isinstance(v, dict):
            return {str(k): _plain(v[k]) for k in sorted(v, key=str)}
        # A torch.device, a torch.dtype, anything else: its repr is stable within a
        # build, and `device` is dropped before we get here.
        return repr(v)

    kwargs = dict(kwargs or {})
    kwargs.pop("device", None)
    payload = {
        "format": PROMPT_CACHE_FORMAT,
        # basename, not the full path: the same checkpoint reached through a symlink or
        # a differently-mounted hfcache is the same encoder.
        "checkpoint": os.path.basename(os.path.normpath(str(checkpoint or ""))),
        "dtype": str(dtype or ""),
        "text_encoder_quant": str(text_encoder_quant or "none").strip().lower(),
        "args": _plain(list(args)),
        "kwargs": _plain(kwargs),
    }
    blob = _json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def prompt_cache_dir(configured: str = "") -> str:
    """Where the prompt cache lives. Under the coderai cache root unless overridden."""
    if str(configured or "").strip():
        return os.path.expanduser(str(configured).strip())
    root = os.environ.get("CODERAI_HOME") or os.path.join("~", ".coderai")
    return os.path.expanduser(os.path.join(root, "prompt_cache", "longcat"))


def dit_subdir(family: str = "") -> str:
    """Which subfolder holds the bf16 transformer for this family."""
    return AVATAR_DIT_SUBDIR if family else BASE_DIT_SUBDIR


def shared_checkpoint(checkpoint_dir: str = "", configured: str = "") -> str:
    """Where tokenizer/text_encoder/vae come from.

    For the base checkpoint that is itself. An avatar checkpoint does not ship them,
    so it borrows them from the base weights — configured explicitly, or the upstream
    repo id, which resolve_checkpoint() turns into a local snapshot.
    """
    if configured:
        return resolve_checkpoint(configured)
    if family_of(checkpoint_dir):
        return resolve_checkpoint(BASE_REPO)
    return checkpoint_dir


def checkpoint_problems(checkpoint_dir: str, stages=(), family: str = "",
                        shared_dir: str = "") -> list:
    """Human-readable reasons this directory cannot serve ``stages``, or [].

    Checked BEFORE loading: a missing subfolder otherwise surfaces minutes in, as a
    from_pretrained traceback from inside the venv, and a missing LoRA surfaces only
    when the stage that needs it starts."""
    problems = []
    if not checkpoint_dir:
        return ["no checkpoint directory configured"]
    root = os.path.expanduser(checkpoint_dir)
    if not os.path.isdir(root):
        return [f"checkpoint directory does not exist: {root}"]
    if family:
        # Its own parts…
        for sub in AVATAR_OWN_SUBDIRS:
            if not os.path.isdir(os.path.join(root, sub)):
                problems.append(f"missing {sub}/ in {root}")
        if not any(os.path.isdir(os.path.join(root, d))
                   for d in (AVATAR_DIT_SUBDIR, INT8_SUBDIR)):
            problems.append(
                f"missing {AVATAR_DIT_SUBDIR}/ and {INT8_SUBDIR}/ in {root} — an "
                f"avatar checkpoint needs at least one transformer")
        # …and the shared ones, from wherever they were borrowed.
        shared = os.path.expanduser(shared_dir or "")
        if not shared or not os.path.isdir(shared):
            problems.append(
                f"the {family} checkpoint has no tokenizer/text_encoder/vae of its "
                f"own and no base checkpoint was found to borrow them from "
                f"(looked for {BASE_REPO})")
        else:
            for sub in SHARED_SUBDIRS:
                if not os.path.isdir(os.path.join(shared, sub)):
                    problems.append(f"missing {sub}/ in the base checkpoint {shared}")
    else:
        for sub in CHECKPOINT_SUBDIRS:
            if not os.path.isdir(os.path.join(root, sub)):
                problems.append(f"missing {sub}/ in {root}")
    for stage in stages:
        if stage == "distill":
            rel = LORA_FILES["distill"]
        elif stage == "refinement":
            rel = LORA_FILES["refinement"]
        else:
            continue
        if not os.path.isfile(os.path.join(root, rel)):
            problems.append(f"stage '{stage}' needs {rel}, which is not in {root}")
    return problems


def source_problems(src_dir: str) -> list:
    """Reasons this is not a usable LongCat-Video source checkout, or [].

    ``LongCatVideoPipeline`` is in the repo's own ``longcat_video`` package, not on PyPI,
    so the source tree is as much a prerequisite as the venv."""
    if not src_dir:
        return ["no LongCat-Video source directory configured"]
    root = os.path.expanduser(src_dir)
    if not os.path.isdir(root):
        return [f"LongCat-Video source directory does not exist: {root}"]
    pkg = os.path.join(root, "longcat_video")
    if not os.path.isdir(pkg):
        return [f"{root} has no longcat_video/ package — is it the LongCat-Video repo?"]
    needed = ("pipeline_longcat_video.py", os.path.join("modules", "longcat_video_dit.py"))
    return [f"missing longcat_video/{n}" for n in needed
            if not os.path.isfile(os.path.join(pkg, n))]
