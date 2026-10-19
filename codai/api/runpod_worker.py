# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""RunPod worker — per-model config parsing + endpoint resolution.

This module owns the RunPod-specific model config (parsed from the model's
models.json ``runpod`` block) and turns it into a live OpenAI base URL that
:mod:`codai.backends.runpod` proxies to.

Two modes (per-model ``mode``):
  * ``serverless`` — RunPod autoscales an endpoint; coderai just proxies to
    ``<serverless_base>/<endpoint_id>/openai/v1`` with the account API key.
    RunPod owns the worker lifecycle; there is nothing for coderai to provision.
  * ``pods`` — coderai provisions, load-balances and scales a POOL of GPU pods
    itself (lifecycle in this module; see RunpodPodPool). [built in a later phase]
  * ``auto`` — pick pods-vs-serverless by the selection criteria. [later phase]

The account settings (API key, endpoints, global caps) live in
:class:`codai.config.RunpodConfig`; this module reads the per-model block.
"""

from codai.api import pod_http
import os
from collections import OrderedDict as _OrderedDict
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Per-model config
# --------------------------------------------------------------------------- #
@dataclass
class RunpodModelConfig:
    """Parsed per-model ``runpod`` block. All fields optional with sane defaults."""
    mode: str = "pods"                       # pods | serverless | auto
    # --- selection (pods) ---
    cloud_types: list = field(default_factory=lambda: ["SECURE"])   # allowed pools
    selection_criteria: str = "cheaper"      # cheaper | faster
    gpu_type: str = ""                       # explicit RunPod gpuTypeId (optional)
    # Ordered PREFERENCE instead of a single card: tried best-first, and with
    # allow_other_gpus the search falls through to anything else that fits.
    # `gpu_type` above is folded in as a single-entry preference.
    gpu_types: list = field(default_factory=list)
    # When every preferred card is unavailable, may we rent a different one that
    # still satisfies min_vram_gb and max_hourly_usd? OFF by default: a pinned
    # card is sometimes pinned for a reason (an architecture a kernel needs, or
    # a price), and silently renting something else would be a surprise. ON is
    # what you want for "serve the request, A40 or not" — and then the VRAM
    # floor and the $/hr ceiling are the real constraints, so check that
    # something actually fits under them.
    allow_other_gpus: bool = False
    min_vram_gb: float = 0.0
    max_hourly_usd: float = 0.0              # per-model $/hr GPU ceiling (0 = none)
    # GPUs per pod. RunPod rents multi-GPU pods, and a 154 GB model on one 80 GB
    # card is exactly where you want two or four. Three things have to agree:
    # the price ceiling is for the WHOLE pod (N cards), min_vram_gb is the total
    # across them, and the engine has to shard — vLLM by tensor parallelism,
    # llama.cpp by split mode. All three are handled from this one number.
    gpu_count: int = 1
    allow_spot: bool = False
    # --- serving (pods) ---
    image: str = ""                          # blank = image picked from `engine`
    # Which server the pod runs. "auto" reads the model: an HF repo goes to vLLM,
    # a GGUF goes to llama.cpp (vLLM's GGUF support is experimental — single-file
    # only, a limited architecture list, and it still wants the original repo's
    # tokenizer — so it is not a drop-in for the llama.cpp catalogue).
    engine: str = "auto"                     # auto | vllm | llamacpp | coderai | custom
    # Readiness probe path. Blank = the engine's default (/v1/models for the
    # OpenAI servers, /healthz for a coderai pod).
    health_path: str = ""
    # Verbatim docker args, replacing whatever the engine would generate. The
    # escape hatch for `engine: custom` and for images with their own CLI.
    docker_args: str = ""
    # RunPod container-registry credential id for a private image; blank falls
    # back to the account-wide one.
    registry_auth_id: str = ""
    # Attach a RunPod network volume to this pool's pods. Blank falls back to the
    # account-wide one. Weights, uploads and adapters on it outlive the pod.
    network_volume_id: str = ""
    # One volume per region: {"EU-SE-1": "vol-a", "EU-RO-1": "vol-b"}. Lets this
    # pool keep its `data_centers` fallback list AND persistent storage, which a
    # single network_volume_id cannot (it pins every pod to its own region).
    # Supersedes network_volume_id for this pool when set; falls back to the
    # account-wide map. A one-entry map behaves exactly like the single id.
    network_volumes: dict = field(default_factory=dict)
    # Per-model override for the account's GLOBAL volume (see RunpodConfig).
    # Region-independent, so it never pins the pod's data centre.
    global_volume_id: str = ""
    volume_mount_path: str = ""              # blank = the account default
    volume_path: str = ""                    # weights already on the volume (file or dir)
    # Reach the pod at its public ip:port instead of through RunPod's HTTP proxy.
    # The proxy is fronted by Cloudflare: TLS, but a 100 s idle limit per request
    # and a request-body cap. Direct TCP has neither, and a coderai image brings
    # its own TLS on it — so it is the DEFAULT for our images (None = decide by
    # image: direct for coderai, proxy for vLLM/llama.cpp/custom, which cannot
    # take a certificate). True/False force it either way.
    direct_tcp: Optional[bool] = None
    # Keep this pod's Python dependencies on the VOLUME instead of in the image,
    # and boot a small image that uses them. The first pod builds the venv (a few
    # minutes); every pod after skips both the 7 GB image pull and the install.
    # Needs a network volume. Opt-in per pool — never a default, because it
    # trades a fast pull for slower imports off network storage.
    venv_on_volume: bool = False
    venv_name: str = ""                      # blank = the profile/capability name
    slim_image: str = ""                     # blank = the published slim image
    # Share one set of pods with every other model naming the same pool, instead
    # of renting a card each. Only possible when the pod's server can serve more
    # than one model: a coderai pod is a whole coderai and picks the model from
    # the request, while a vLLM pod is launched `--model X` and a llama.cpp pod
    # `-hf one.gguf` — those serve exactly one, and sharing is refused.
    pool: str = ""
    # vLLM's own quantization flag (awq, gptq, bitsandbytes, fp8 …). A local
    # entry's load_in_4bit/8bit does NOT carry over: that is how the transformers
    # backend loads weights here, and vLLM quantizes its own way.
    quantization: str = ""
    # Where the POD gets the weights. A local path is meaningless there, so one
    # of these must resolve:
    #   auto   — work it out: an explicit hf_repo, a path that IS a repo id, or a
    #            repo id recovered from the HuggingFace cache path; then model_url.
    #   hf     — download `hf_repo` from HuggingFace.
    #   url    — download `model_url` directly (how most one-off GGUFs are had).
    #   upload — push the local weights to the pod. Only a coderai pod can take
    #            this: vLLM and llama.cpp are launched with the model and need it
    #            to exist before the container starts.
    source: str = "auto"                     # auto | hf | url | upload
    hf_repo: str = ""                        # explicit HuggingFace repo id
    model_url: str = ""                      # explicit direct download URL
    # The URL points at a tar of a multi-file model, to be extracted on the pod.
    model_url_is_tar: bool = False
    # Bearer token the pod requires on every request. A RunPod proxy URL is
    # reachable by anyone who learns it, so a pod without one is an open GPU on
    # the public internet. Blank does NOT mean "no auth": the pool generates a
    # random token per pool and launches the pod with it. Set it explicitly only
    # when something other than coderai must also call the pod.
    api_key: str = ""
    # Escape hatch for a pod whose server genuinely cannot take a token (an image
    # with no auth support). Explicit, so it can never happen by accident.
    allow_open_pod: bool = False
    served_model: str = ""                   # HF id the pod/endpoint serves
    # GGUF to pull on a llama.cpp pod, in llama.cpp's own `-hf` form:
    # "user/repo:Q4_K_M" or "user/repo:file.gguf". Needed because a local
    # /AI/…/foo.gguf path means nothing on a rented machine.
    hf_gguf: str = ""
    container_disk_gb: int = 40
    volume_gb: int = 0
    port: int = 8000
    ctx: int = 0
    env: dict = field(default_factory=dict)  # extra pod env (HF_TOKEN, etc.)
    # --- scaling (pods) ---
    min_pods: int = 0
    # Keep one pod running for this model at all times, so a request never waits
    # for a cold boot (image pull + weight download — minutes). OFF by default and
    # deliberately so: a warm pod bills every hour of every day, including the
    # ones where nobody asks it anything.
    keep_warm: bool = False
    max_pods: int = 1
    # Grow the pool when the least-loaded pod already has this many requests in
    # flight. This is what makes a pool track real concurrency: without it the
    # pool only ever grows when it has NO healthy pod, so fifty concurrent
    # requests would all pile onto pod #1 while max_pods sat unused.
    scale_up_inflight_per_pod: int = 4
    # How many pods may be BOOTING at once when the load justifies them. 1 was
    # the old behaviour, and it was not a choice: the grower held a single
    # boolean for the whole ~12-minute boot, so a burst needing four pods took
    # forty minutes to be served by four. It is still a real limit and not just
    # a formality, because the pods the demand justifies are also the GPUs that
    # start billing together — so this is the knob that says how fast an
    # operator is willing to spend to absorb a burst. Growth never exceeds
    # max_pods, and the cost caps still refuse each pod individually.
    #
    # Also bounds filling a WARM FLOOR (min_pods, keep_warm, or a client
    # lease). That used to rent one pod per scaler pass and block the
    # thread until it was ready, so a declared floor of 8 took ~40 minutes
    # to exist. A floor is a declaration rather than a prediction, so the
    # deficit is filled at once -- but still no more of them booting
    # together than this, because it is the same question about money.
    scale_up_max_parallel_boots: int = 4
    # Scaling DOWN again when the pressure that justified the pods is gone.
    # Idle teardown alone is not enough: it only fires on a pod with NO work and
    # no request for idle_timeout_s, so a fleet that grew to absorb a burst and
    # is now trickling keeps every pod alive — each one touched just often enough
    # to look used, all of them a fraction busy, and the bill is the peak for as
    # long as the trickle lasts. So the pool also asks how many pods the CURRENT
    # concurrency needs and retires the excess, after holding the observation
    # this long (a burst that dips for a moment must not cost a pod that is
    # needed again seconds later, and a cold pod takes minutes to replace).
    # Load the weights BEFORE calling the pod ready, by sending it one tiny
    # inference. /v1/models answers as soon as the server is up, which is
    # minutes before it can actually serve: a vLLM pod still has to download
    # the checkpoint and build its KV cache on first use. So "ready" meant
    # "reachable", every client read it as "usable", and the whole load landed
    # inside somebody's first request — 807s for one OCR page, against clients
    # timing out at 180-300s, and the conclusion that serving was broken.
    #
    # The pod bills through the load either way. This only decides whether the
    # wait is visible to the operator (a pod that is not ready yet) or hidden
    # inside a request that looks hung.
    warmup_on_boot: bool = True
    #: Budget for it. Separate from load_timeout_s: answering /v1/models and
    #: having the weights resident are different milestones, and a checkpoint
    #: re-downloaded on every pod (no network volume) is the slow one.
    warmup_timeout_s: int = 900
    scale_down_after_s: int = 180
    # Which surplus pod to give back first. Off, the pool retires the least
    # loaded one (and among equals the one used least recently, whose prefix
    # cache is coldest). On, PRICE leads: the dearest pod goes first.
    #
    # Only a mixed fleet has anything to choose between — pods rented at
    # different times take whatever the market offered, so the same pool can hold
    # a $0.35 card and a $0.79 one for the same work, and shedding the cheap one
    # saves less than half as much. Note what "dearest" tends to mean: an
    # on-demand pod costs more than a spot pod, so this prefers to keep the
    # interruptible ones.
    scale_down_costliest_first: bool = False
    # Hard ceiling on concurrent requests per pod (0 = no ceiling, let the remote
    # server's own batching absorb them). Above it, a request waits for a slot
    # instead of being piled on — use it for engines that degrade badly under
    # concurrency rather than queueing internally.
    max_inflight_per_pod: int = 0
    # Send a conversation back to the pod that already served it, so the remote's
    # prefix/KV cache still holds its context. Off = always pick the least-loaded.
    sticky_sessions: bool = True
    idle_timeout_s: int = 300                # destroy a pod this long after its last request
    # --- placement ---
    # RunPod data centre (region) for this model's pods, e.g. "EU-RO-1". Blank
    # falls back to the account-wide setting, and a network volume still wins
    # over both: a volume lives in one region and a pod elsewhere cannot attach
    # it. Pin a region for latency or for where the data is allowed to be.
    data_center: str = ""
    # Several acceptable regions, tried in order on a capacity miss. This is how
    # "anywhere in the EU" is said: a single data_center pins one region and a
    # blank one lets RunPod pick ANY region on earth, which is the wrong answer
    # for data that may not leave the EU. Ignored when a NETWORK volume is
    # attached — that volume exists in one region and nothing else can reach it.
    data_centers: list = field(default_factory=list)
    # --- how hard the pod's own server may be driven ---
    # A rented pod carries no config file, so these ride as env. The defaults
    # (2 in flight, 6 queued) are sized for a shared workstation engine and
    # throttle a dedicated A40 to a fraction of what vLLM's max_num_seqs would
    # serve: the app's admission gate saturates long before the GPU does, and
    # everything over the limit gets 429. Set these to roughly the remote
    # engine's own batch width. 0 = leave the pod's defaults alone.
    pod_max_parallel_requests: int = 0
    pod_queue_max_size: int = 0
    #: What the two numbers above, and `scale_up_inflight_per_pod`, are stated
    #: FOR: the first card in `gpu_types`. Every other card in a mixed fleet is
    #: scaled against it, by VRAM, because VRAM is what bounds how many
    #: requests fit at once (KV cache) and it is knowable without measuring
    #: anything.
    #:
    #: VRAM is NOT a throughput ratio, and the difference is worth stating: an
    #: A100 SXM 80GB holds 1.67x an A40's memory but has 2.93x its bandwidth,
    #: while an L40S holds exactly as much as an A40 and is ~1.24x faster. So
    #: VRAM under-credits a faster card and gives a same-size one nothing. When
    #: that matters, name the card here and the number wins:
    #:     {"NVIDIA A100-SXM4-80GB": 2.9, "NVIDIA L40S": 1.25}
    #: A card not named falls back to the VRAM ratio.
    gpu_weights: dict = field(default_factory=dict)
    #: PER-CARD limits, which say outright what `gpu_weights` can only imply:
    #:
    #:   "gpu_limits": {
    #:     "NVIDIA A100-SXM4-80GB": {"parallel": 186, "queue": 744, "scale_up": 93},
    #:     "NVIDIA A40":            {"parallel": 64,  "queue": 256, "scale_up": 32}
    #:   }
    #:
    #: `parallel`/`queue` are what the POD's own admission gate is launched
    #: with (it is env, so it is per pod and therefore per card); `scale_up` is
    #: the in-flight depth at which that card justifies another pod;
    #: `inflight` caps what this process will send one.
    #:
    #: This exists because a scaled single number cannot be right for long. The
    #: model-level numbers describe ONE card, and which card that is was the
    #: first entry of `gpu_types` -- so changing the rental preference silently
    #: re-scaled the concurrency, and a figure measured on an A40 came to
    #: describe an A100 by a factor nobody had measured. A table states what was
    #: measured, per card, and nothing is multiplied.
    #:
    #: A card absent from the table falls back to `gpu_weights`, then to the
    #: VRAM ratio, then to the model-level numbers unscaled.
    gpu_limits: dict = field(default_factory=dict)
    # --- warm-pod schedule ---
    # When set, the warm floor (min_pods/keep_warm) only applies inside the
    # window; outside it the floor is 0 and warm pods are terminated, so a model
    # that is only used in office hours stops billing overnight. It does NOT
    # block requests: one arriving outside the window still cold-starts a pod,
    # exactly as min_pods=0 always has.
    schedule_enabled: bool = False
    # Let a CLIENT ask for a warm floor of its own, up to max_pods, through
    # POST /v1/runpod/scale. Off by default: a client that can raise the floor
    # can raise the bill, so it is opt-in per model. The client's floor is a
    # LEASE — it expires after client_scale_ttl_s unless renewed — because the
    # failure that matters is not a wrong number, it is a client that asks for
    # four pods and then dies, leaving them billing until someone notices.
    allow_client_scale: bool = False
    client_scale_ttl_s: int = 1800
    schedule_days: list = field(default_factory=list)   # 0=Mon … 6=Sun; empty = every day
    schedule_start: str = ""                 # "HH:MM" local to schedule_tz
    schedule_end: str = ""                   # "HH:MM"; end <= start means overnight
    schedule_tz: str = ""                    # IANA name; blank = the server's local zone
    # Boot budgets — bigger models need longer (image pull + weight download).
    # Until the pod exposes its port. The port appears only AFTER the image is
    # pulled, so this covers the download: 15 GB at a cold machine's ~25 MB/s
    # is ten minutes. 900 was set on the pool class and this default was left at
    # 300 — two numbers for one thing, and the smaller one silently won.
    boot_timeout_s: int = 900
    load_timeout_s: int = 600                # until vLLM answers /v1/models
    # --- serverless ---
    endpoint_id: str = ""                    # reference an existing serverless endpoint
    min_workers: int = 0
    max_workers: int = 1
    # --- cost budget (cumulative spend, distinct from the $/hr rate cap) ---
    cost_limit_usd: float = 0.0              # 0 = unlimited
    cost_period: str = "unlimited"           # hour | day | week | month | unlimited

    @property
    def is_serverless(self) -> bool:
        return (self.mode or "").lower() == "serverless"


# Default OpenAI-compatible pod image (vLLM's official OpenAI server). The pod
# pulls the model named by ``served_model`` (needs HF_TOKEN in env for gated repos).
DEFAULT_POD_IMAGE = "vllm/vllm-openai:latest"

# llama.cpp's own server image. Its OpenAI surface (/v1/models, /v1/chat/completions)
# matches vLLM's closely enough that the readiness probe and the proxy are unchanged.
# This is what serves the GGUF catalogue remotely: those models have no safetensors
# repo to hand vLLM, and vLLM's GGUF loader is experimental (single-file only, a
# limited architecture list, and it needs the original repo for the tokenizer).
LLAMACPP_POD_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"

#: Published trimmed coderai images, one per capability — a whole coderai
#: carrying only that capability's dependencies (packaging/runpod/). A capability
#: pod with no `image` uses the one for its own capability, so turning a
#: capability remote needs no image name typed in. Override per capability with
#: `image`, or point the whole set elsewhere with CODERAI_CAPABILITY_IMAGE_REPO
#: (e.g. your own fork, or a private registry).
CAPABILITY_IMAGE_REPO = os.environ.get(
    "CODERAI_CAPABILITY_IMAGE_REPO", "ghcr.io/nextime/coderai")
CAPABILITY_IMAGE_TAG = os.environ.get("CODERAI_CAPABILITY_IMAGE_TAG", "latest")

#: Capabilities with a published image. One without a row here has no image of
#: its own and needs `image` set explicitly — better than defaulting to a tag
#: that does not exist and failing at pull time, minutes into a pod boot.
PUBLISHED_CAPABILITY_IMAGES = (
    "images", "video", "embeddings", "ocr", "tts", "stt", "voice", "audio",
    "faceswap",
    # Stacks that cannot share the main venv get an image of their own, rather
    # than 6 GB added to one everybody pulls. Each carries a baked venv — see
    # packaging/runpod/profiles/<name>.venv-*.txt — and each exists because a
    # real conflict was measured, not assumed:
    "speaker",       # pyannote: torchaudio.AudioMetaData removed in 2.11
    "tts-xtts",      # coqui-tts: imports a transformers 4.x symbol
    "stt-nemo",      # NeMo Canary: large, and its own torch
    "stt-crisper",   # CrisperWhisper: transformers 4.46 AND torch 2.5
    "ocr-paddle",    # paddlepaddle-gpu: brings its own CUDA runtime
    "video-longcat", # LongCat-Video: Python 3.10 + torch 2.6+cu124 + transformers 4.41,
                     #   so TWO interpreters in one image (see its own Dockerfile). The
                     #   plain 'video' image cannot serve it at all.
    # Text: a coderai pod that serves LLMs. Unlike a vLLM/llama.cpp pod it can be
    # SENT a local LoRA adapter, which is the only way a local text adapter runs
    # on RunPod. Build and push it with packaging/runpod before using it.
    "text",
    # The native MoE engines — ds4, colibri, kimi-k3-in-c — compiled for every
    # pod GPU generation rather than copied from this machine (whose ds4 is an
    # sm_86 build against CUDA 13, and whose colibri has no GPU backend). Picked
    # by a model's `backend`, not its section: it is still a text model.
    "engines",
    # ktransformers: SGLang + kt-kernel in their own venv on the light core.
    # kt-kernel dispatches at runtime (AMX → AVX-512 → AVX2 llamafile → AMD
    # BLIS), so one image serves whatever CPU the pod host has.
    "engines-kt",
    # The two LLM servers as coderai pods rather than upstream images: the
    # same kernels, plus what a coderai pod gives — TLS on the direct path,
    # /boot, seeding, local LoRAs, the Tasks page. `llama` is llama-cpp-python
    # built with CUDA for every pod GPU on the full core (the GGUF backend
    # asks torch about the card); `vllm` is vLLM in its own venv on the light
    # core, driven by coderai's vLLM proxy backend.
    "llama",
    "vllm",
    # The same vLLM pod pinned to 0.20.1 — the version `surya-ocr` spawns and
    # validates against — because serving surya-ocr-2 on 0.27.1 loses every
    # accented character while the model's own bytes decode cleanly. Rent it for
    # the OCR model only; everything else on it is pinned back seven minor
    # releases. See packaging/runpod/profiles/vllm-ocr.txt for the measurement.
    "vllm-ocr",
)

#: Model backends served by an engine image, and which image. ds4, colibri
#: and k3 are C binaries and share one image. ktransformers is a Python stack
#: (kt-kernel + SGLang, pinning their own torch) that takes a venv of its own,
#: so it has an image of its own rather than doubling the first.
_ENGINE_POD_BACKENDS = ("ds4", "colibri", "k3", "kt", "vllm")
_ENGINE_POD_IMAGE = {"kt": "engines-kt", "vllm": "vllm"}
#: runpod-block `engine` values that mean "a coderai pod running THIS server",
#: as opposed to `vllm`/`llamacpp`, which mean the upstream images.
#:
#: `coderai-vllm-ocr` is the same server pinned to vLLM **0.20.1**, the version
#: `surya-ocr` itself spawns and validates against (surya/settings.py:96). It
#: exists because serving surya-ocr-2 on 0.27.1 loses every accented character
#: while the model's own bytes decode cleanly — see
#: packaging/runpod/profiles/vllm-ocr.txt for the measurement. Rent it for the
#: OCR model only: everything else on that image would be pinned back with it.
_CODERAI_SERVER_ENGINES = {"coderai-vllm": "vllm", "coderai-llama": "llama",
                           "coderai-vllm-ocr": "vllm-ocr"}

#: Capabilities served by a published image other than their own name.
_CAPABILITY_IMAGE_ALIASES = {
    "rerank": "embeddings",      # same sentence-transformers stack
    # `speaker` used to ride the STT image on the assumption it was "the STT
    # deps plus a bit". It was not: that image shipped nothing for diarization
    # or voiceprints at all, so a speaker pod booted with no way to embed a
    # voice. pyannote needs a torch the main venv cannot have, so it has its own
    # image now.
    "stems": "audio",
    "audio_clean": "audio",
    "audio_gen": "audio",
}


#: models.json section -> the capability whose pod image serves it. This is what
#: lets a NON-TEXT model be sent to RunPod individually: the pod image follows
#: from what the model is, not from whether its path ends in .gguf.
MODEL_TYPE_CAPABILITY = {
    "text_models": "text",
    "gguf_models": "text",
    "vision_models": "text",
    "image_models": "images",
    "video_models": "video",
    "audio_models": "stt",
    "tts_models": "tts",
    "embedding_models": "embeddings",
    "spatial_models": "spatial",
    # OCR engines are picked by name (paddle|doctr|surya) rather than registered
    # like a model, so nothing could place one individually: a test had to move
    # the whole capability, which is a production routing switch. An entry in
    # this section gives an engine the same per-model placement everything else
    # already has.
    "ocr_models": "ocr",
    "audio_gen_models": "audio_gen",
}

#: Sections served by an LLM pod (vLLM / llama.cpp) rather than a capability pod.
_LLM_MODEL_TYPES = ("text_models", "gguf_models", "vision_models")


def entry_model_type(entry: dict) -> str:
    """The models.json SECTION this entry lives in ("embedding_models", ...), or "".

    In models.json a model's kind is the dict KEY it sits under, not a field on
    the entry -- and the entry is all that travels to a pod. Matched by IDENTITY
    first (the same dict object the loader holds), falling back to id/path so a
    copied or rebuilt entry still resolves. Returns "" on any failure: this
    feeds a seeding path, and a guess is never worth an exception there.
    """
    try:
        from codai.admin.routes import config_manager as _cm
        md = getattr(_cm, "models_data", None) if _cm else None
        if not isinstance(md, dict):
            return ""
        for section, lst in md.items():
            if not isinstance(lst, list):
                continue
            for m in lst:
                if m is entry:
                    return str(section)
        want = str(entry.get("id") or entry.get("path") or "").strip().lower()
        if not want:
            return ""
        for section, lst in md.items():
            if not isinstance(lst, list):
                continue
            for m in lst:
                if isinstance(m, dict) and str(
                        m.get("id") or m.get("path") or "").strip().lower() == want:
                    return str(section)
    except Exception:
        pass
    return ""


def _in_embedding_section(entry: dict) -> bool:
    """Whether this entry lives in models.json's `embedding_models`.

    By IDENTITY first (the same dict object the loader holds), falling back to
    the id -- so an entry copied or rebuilt still resolves. Returns False on any
    failure: a capability guess is never worth an exception in a placement path.
    """
    try:
        from codai.admin.routes import config_manager as _cm
        md = getattr(_cm, "models_data", None) if _cm else None
        lst = (md or {}).get("embedding_models")
        if not isinstance(lst, list):
            return False
        for m in lst:
            if m is entry:
                return True
        want = str(entry.get("id") or entry.get("path") or "").strip().lower()
        if not want:
            return False
        for m in lst:
            if isinstance(m, dict) and str(
                    m.get("id") or m.get("path") or "").strip().lower() == want:
                return True
    except Exception:
        pass
    return False


def model_capability(entry: dict, include_text: bool = False) -> str:
    """The capability a model entry belongs to.

    Text models return '' by default: they normally go to vLLM or llama.cpp, not
    to a coderai pod. ``include_text`` asks for the capability anyway, which is
    what picks the image once something HAS decided a text model needs a coderai
    pod (a local LoRA adapter, today).
    """
    if not isinstance(entry, dict):
        return ""
    # An explicit override, for a model that serves something its section does
    # not imply. Voice cloning is the case that forced this: it runs on a TTS
    # model (XTTS, F5) but is a different endpoint and a different pod image, so
    # section alone maps it to 'tts' and it could never be placed as 'voice'.
    explicit = str(entry.get("capability") or "").strip().lower()
    if explicit:
        return explicit
    # An entry in `embedding_models` IS an embeddings model. Nothing told
    # model_capability that: it receives a bare dict, and _model_entry_for()
    # iterates md.values() and discards the section key it matched in.
    #
    # Before this, the capability came from `runpod.engine` being one of the
    # coderai-* server engines -- so an embedding model with `engine: coderai`
    # (the correct setting for a capability pod) resolved to NO capability, the
    # front could not classify it, and it was loaded LOCALLY on a GPU-less
    # orchestrator. It failed as "tries to load on the wrong machine" rather
    # than as "this needs a capability", which is a long way from the cause.
    #
    # Deliberately only this one section. The others work today, and remapping
    # them all would change placement for every model kind to fix one -- a
    # blast radius this does not need.
    if _in_embedding_section(entry):
        return "embeddings"
    # A LongCat-Video model is a video model, but the 'video' image cannot serve it: its
    # pipeline is not in diffusers and it needs its own Python 3.10 venv. Without this
    # it would be placed on an image that boots, passes its health check, accepts the
    # request and then has nothing to run it with — the same trap the OCR profile
    # documents for Surya.
    if str(entry.get("backend") or "").strip().lower() in (
            "longcat", "longcat-video", "longcat_video"):
        return "video-longcat"
    types = entry.get("model_types") or [entry.get("model_type") or ""]
    for mt in types:
        if mt in _LLM_MODEL_TYPES and not include_text:
            return ""
        cap = MODEL_TYPE_CAPABILITY.get(mt)
        if cap:
            return cap
    return ""


def default_capability_image(capability: str) -> str:
    """The published image for a capability, or '' when there is none."""
    name = _CAPABILITY_IMAGE_ALIASES.get(capability, capability)
    if name not in PUBLISHED_CAPABILITY_IMAGES:
        return ""
    return f"{CAPABILITY_IMAGE_REPO}-{name}:{CAPABILITY_IMAGE_TAG}"


def _looks_like_repo_id(name: str) -> bool:
    """'Org/Model' — something HuggingFace can resolve, not a filesystem path."""
    n = (name or "").strip()
    return bool(n) and "/" in n and not n.startswith(("/", "~", ".")) \
        and not n.startswith(("http://", "https://")) and n.count("/") == 1


def resolve_model_source(entry: dict, mcfg: "RunpodModelConfig") -> tuple:
    """Where the POD gets this model's weights: ("hf"|"url"|"upload"|"", value).

    A local path exists only on this machine. The pod has to fetch the model
    itself — or be given it — so this works out which, preferring what costs
    nothing: a HuggingFace repo id the pod pulls at datacenter speed, then an
    explicit URL, and only then an upload from here.

    The repo id is often recoverable even when the configured path is local:
    weights downloaded through coderai live in the HuggingFace cache, whose
    directory names encode the repo (``models--Org--Name``).
    """
    entry = entry or {}
    want = (getattr(mcfg, "source", "auto") or "auto").lower()
    url = (getattr(mcfg, "model_url", "") or "").strip()
    path = str(entry.get("path") or "").strip()

    if want == "upload":
        return ("upload", path)
    if want == "url":
        return ("url", url) if url else ("", "")
    if want == "hf":
        repo = (getattr(mcfg, "hf_repo", "") or "").strip()
        return ("hf", repo) if repo else ("", "")

    # auto, most-preferred first
    for cand in (getattr(mcfg, "hf_repo", ""), getattr(mcfg, "served_model", ""), path):
        if _looks_like_repo_id(cand):
            return ("hf", cand.strip())
    if path.startswith(("http://", "https://")):
        return ("url", path)
    from_cache = _hf_repo_id_from_cache_path(path)
    if from_cache:
        return ("hf", from_cache)
    if url:
        return ("url", url)
    return ("", "")


def _hf_repo_id_from_cache_path(path: str) -> str:
    """Recover 'Owner/Repo' from a HuggingFace hub cache path, or ''.

    Cache layout: .../hub/models--OWNER--REPO/snapshots/<hash>/<file>. The first
    '--' after 'models--' separates owner from repo.
    """
    for part in str(path or "").replace("\\", "/").split("/"):
        if part.startswith("models--"):
            rest = part[len("models--"):]
            sep = rest.find("--")
            if sep != -1:
                return rest[:sep] + "/" + rest[sep + 2:]
    return ""


def _looks_like_gguf(name: str) -> bool:
    return (name or "").strip().lower().endswith(".gguf")


def resolve_pod_engine(mcfg: "RunpodModelConfig", model_key: str = "",
                       model_path: str = "", entry: dict = None) -> str:
    """Decide which server a pod runs for this model.

    An explicit ``engine`` wins. Otherwise the model's OWN KIND decides: an
    image/video/TTS/STT/embedding model goes to a coderai capability pod (vLLM
    cannot serve a diffusion pipeline), a GGUF to llama.cpp, and an HF repo id to
    vLLM.
    """
    want = (mcfg.engine or "auto").strip().lower()
    if want in _CODERAI_SERVER_ENGINES:
        return "coderai"
    if want in ("vllm", "llamacpp", "coderai", "custom"):
        return want
    # A non-text model is served by a whole coderai carrying that capability.
    if entry is None:
        entry = _model_entry(model_key)
    if model_capability(entry or {}):
        return "coderai"
    # A model pinned to a native engine is served by a coderai carrying that
    # engine: vLLM cannot load a DeepSeek-V4 GGUF laid out for ds4, and there
    # is no colibri outside of coderai.
    if engine_pod_backend(entry or {}, mcfg):
        return "coderai"
    # A text model whose LoRA lives only on this disk has one option: a coderai
    # pod, which can be sent the adapter. vLLM and llama.cpp resolve adapters
    # themselves at launch and have no endpoint to receive one.
    try:
        from codai.models.text_loras import configured_specs, portable_specs
        _portable, _local = portable_specs(configured_specs(entry or {}))
        if _local:
            print(f"[lora] {model_key!r} has adapters that exist only here "
                  f"({', '.join(str(s.get('source')) for s in _local)}) — using a "
                  "coderai pod, which can be sent them", flush=True)
            return "coderai"
    except Exception:
        pass
    if mcfg.hf_gguf:
        return "llamacpp"
    if _looks_like_gguf(mcfg.served_model) or _looks_like_gguf(model_path) \
            or _looks_like_gguf(model_key):
        return "llamacpp"
    return "vllm"


def direct_tcp_for(mcfg: "RunpodModelConfig", engine: str) -> bool:
    """Whether this pod is reached at its public ip:port rather than the proxy.

    An explicit setting wins. Otherwise a coderai image goes direct — it
    serves TLS on that path, and the proxy's 100 s cutoff has already cost a
    voice clone — while vLLM's, llama.cpp's and custom images stay behind the
    proxy, which is the only TLS they have.
    """
    explicit = getattr(mcfg, "direct_tcp", None)
    if explicit is not None:
        return bool(explicit)
    return (engine or "") == "coderai"


def engine_pod_backend(entry: dict, mcfg: "RunpodModelConfig" = None) -> str:
    """The native engine a pod must run for this model, or ''.

    Two ways to say it. A RunPod-only model has `backend: runpod` — that slot
    is taken — so it names the engine on its runpod block (`engine: ds4`),
    beside vllm and llamacpp. A LOCAL model pinned to `backend: ds4` that
    bursts to RunPod carries the pin itself. Either way the answer is the
    same image and the same seed.
    """
    want = str(getattr(mcfg, "engine", "") or "").strip().lower() if mcfg else ""
    if want in _ENGINE_POD_BACKENDS:
        return want
    if want in ("coderai-vllm", "coderai-vllm-ocr"):
        # Both are coderai's vLLM backend; they differ only in which vLLM the
        # image carries, which is the image's business and not the engine's.
        return "vllm"
    b = str((entry or {}).get("backend") or "").strip().lower()
    return b if b in _ENGINE_POD_BACKENDS else ""


def pod_image_for_entry(entry: dict, mcfg: "RunpodModelConfig" = None,
                        model_key: str = "", model_path: str = "") -> str:
    """The image this model's pods WOULD run, resolved the way a launch does.

    `runpod.image` is OPTIONAL and usually absent. Naming `engine:
    coderai-vllm-ocr` already says which published image to pull, and
    `pod_plan` resolves that at launch time. So anything that needs to know
    which image a pool runs BEFORE a pod exists has to do the same resolution;
    reading the raw `image` field instead answers "" for the ordinary
    configuration, and "" is indistinguishable from "a plain image".

    That is not hypothetical. The OCR pipeline's "do these pods rasterise for
    themselves" gate read the raw field, saw "", and ran the whole pipeline on
    the orchestrator for every page — while the pods it was renting carried the
    rasteriser and served only the model. Nothing failed and nothing was
    logged; a GPU-less box simply did the work, at 47 GB resident.

    Returns "" only when nothing determines an image, which is the one case
    `pod_plan` refuses to launch. Pure: no network, no side effects, safe to
    call before anything is rented.
    """
    entry = entry or {}
    if mcfg is None:
        mcfg = parse_model_runpod(entry.get("runpod") or {})
    if mcfg.image:
        return mcfg.image
    engine = resolve_pod_engine(mcfg, model_key, model_path, entry)
    if engine == "llamacpp":
        return LLAMACPP_POD_IMAGE
    if engine not in ("coderai", "custom"):
        return DEFAULT_POD_IMAGE
    want = str(mcfg.engine or "").strip().lower()
    if want in _CODERAI_SERVER_ENGINES:
        return default_capability_image(_CODERAI_SERVER_ENGINES[want])
    backend = engine_pod_backend(entry, mcfg)
    if backend:
        return default_capability_image(_ENGINE_POD_IMAGE.get(backend, "engines"))
    cap = model_capability(entry, include_text=True)
    return default_capability_image(cap) if cap else ""


#: Readiness probe per engine. A coderai pod answers /healthz long before any
#: model is loaded, which is exactly what we want: it is ready to be asked.
_HEALTH_PATHS = {"vllm": "/v1/models", "llamacpp": "/v1/models",
                 "coderai": "/healthz", "custom": "/healthz"}


def pod_plan(mcfg: "RunpodModelConfig", served: str, model_key: str = "",
             model_path: str = "", api_key: str = "", entry: dict = None,
             seed_entries: list = None) -> dict:
    """Everything needed to launch one pod: image, docker args, health path.

    This is the seam that lets a pool serve something other than an LLM — a whole
    coderai (so images/video/TTS/… can run on a rented GPU with the same budgets,
    autoscaling and idle reaping), or any image at all with `engine: custom`.
    """
    if entry is None:
        entry = _model_entry(model_key)
    engine = resolve_pod_engine(mcfg, model_key, model_path, entry)
    if engine == "llamacpp":
        image = mcfg.image or LLAMACPP_POD_IMAGE
        args = mcfg.docker_args or _llamacpp_docker_args(mcfg, served, entry)
    elif engine in ("coderai", "custom"):
        # One path, whether the image is configured or defaulted: _plan() carries
        # the auth token AND the catalogue the pod must know about. Splitting
        # them meant a pod with an explicit image got no models and refused
        # everything with "not available".
        capability_hint = model_capability(entry or {}, include_text=True)
        image = pod_image_for_entry(entry or {}, mcfg, model_key, model_path)
        if not image:
            raise RuntimeError(
                f"RunPod {engine} pod: set `image` on the runpod block to the image "
                "to run. Capability pods default to the published "
                f"{CAPABILITY_IMAGE_REPO}-<capability> image; this one has none, so "
                "name it explicitly.")
        return _plan(engine, image, mcfg.docker_args, mcfg, api_key, entry, served,
                     seed_entries, capability=capability_hint)
    else:
        image = mcfg.image or DEFAULT_POD_IMAGE
        args = mcfg.docker_args or _vllm_docker_args(mcfg, served, entry)
    # Lock the pod to a bearer token. Each server takes it differently: the two
    # OpenAI servers as a flag, a coderai pod through the environment its auth
    # middleware reads.
    env = {}
    if api_key:
        if engine in ("vllm", "llamacpp"):
            if "--api-key" not in args:
                args = (args + " " if args else "") + f"--api-key {api_key}"
        elif engine == "coderai":
            env["CODERAI_API_TOKEN"] = api_key
        else:
            # A custom image: we cannot know its flag, so pass it in the
            # environment under both the coderai and the common vLLM name.
            env["CODERAI_API_TOKEN"] = api_key
            env["VLLM_API_KEY"] = api_key
    plan = {"engine": engine, "image": image, "args": args, "env": env,
            "health_path": mcfg.health_path or _HEALTH_PATHS.get(engine, "/v1/models")}
    _add_staging(plan, mcfg, entry or {}, engine)
    return plan


#: How to start each server when we override the image's ENTRYPOINT to stage
#: downloads first. `command -v` keeps it working if the binary moves.
_SERVER_CMD = {
    "vllm": "python3 -m vllm.entrypoints.openai.api_server",
    "llamacpp": '"$(command -v llama-server || echo /app/llama-server)"',
}


def _add_staging(plan: dict, mcfg: "RunpodModelConfig", entry: dict, engine: str) -> None:
    """Make the pod fetch what it needs before its server starts.

    The third way to get weights onto a vLLM/llama.cpp pod, alongside "it is on
    HuggingFace" and "it is a coderai pod we can upload to": put the file on any
    HTTP host the pod can reach and have the pod download it at boot. Those
    images run their server AS the entrypoint, so this overrides the entrypoint
    and execs the server after fetching.
    """
    if engine not in _SERVER_CMD:
        return
    downloads = staged_downloads(mcfg, entry, engine)
    if not downloads:
        return

    args = plan["args"]
    # The server args already name the staged path (staged_model_dest), so only
    # add --model when nothing set it — never a second one.
    for item in downloads:
        if item.get("lora_name") or "--model " in args:
            continue
        args += f" --model {item['dest']}"
    staged_loras = [i for i in downloads if i.get("lora_name")]
    if staged_loras:
        if engine == "vllm" and "--enable-lora" not in args:
            mods = " ".join(f"{i['lora_name']}={i['dest']}" for i in staged_loras)
            args += f" --enable-lora --lora-modules {mods} --max-lora-rank 64"
        elif engine == "llamacpp" and "--lora" not in args:
            # llama-server takes one adapter file, with an optional scale.
            first = staged_loras[0]
            weight = float(first.get("weight", 1.0) or 1.0)
            args += (f" --lora-scaled {first['dest']} {weight:g}" if weight != 1.0
                     else f" --lora {first['dest']}")
            if len(staged_loras) > 1:
                print(f"[lora] llama.cpp applies one adapter — using "
                      f"{first['lora_name']!r}", flush=True)

    plan["args"] = args
    plan["entrypoint"] = ["/bin/sh", "-c"]
    plan["start_cmd"] = [stage_script(downloads, f"{_SERVER_CMD[engine]} {args}")]
    names = ", ".join(i["dest"].rsplit("/", 1)[-1] for i in downloads)
    print(f"[runpod] pod will download before serving: {names}", flush=True)


#: Engines that serve frontier-size mixture-of-experts models. Their weights are
#: not "large" in the way a diffusion model is large — DeepSeek-V4 is ~154 GB and
#: they go well past that — so a pod that downloads them pays an hour or two of
#: rental before it answers anything, every cold start.
_HUGE_WEIGHT_ENGINES = ("colibri", "ds4", "k3", "ktransformers")


def weight_transfer_warning(entry: dict, mcfg: "RunpodModelConfig",
                            account=None) -> str:
    """A warning to show prominently, or '' when there is nothing to warn about.

    The cost of these models is the DOWNLOAD, not the GPU: the machinery around
    it (price ranking, idle reaping, spend caps) is tuned for images of ~10 GB
    and models of a few. Renting a card and then spending ninety minutes filling
    its disk is a different shape of expensive, and it should never be a
    surprise — so it is said at configuration time, at provision time, and in
    the test run.
    """
    engine = str(getattr(mcfg, "engine", "") or "").strip().lower()
    backend = str((entry or {}).get("backend") or "").strip().lower()
    if engine not in _HUGE_WEIGHT_ENGINES and backend not in _HUGE_WEIGHT_ENGINES:
        return ""
    vol, _mount = volume_for(mcfg, account)
    if vol:
        return ""
    name = str((entry or {}).get("path") or "this model")
    return (f"{name} runs on a {engine or backend} engine, whose models are "
            f"100 GB and up — DeepSeek-V4 alone is ~154 GB. With no network "
            f"volume configured, EVERY cold pod downloads all of it before it "
            f"can answer, which is typically one to two hours of rental per "
            f"boot. Attach a network volume (network_volume_id) so the weights "
            f"are fetched once and every later pod attaches to them instead.")


#: Room the pod needs beyond the weights: the image unpacks onto the same disk
#: (a 9 GB image is ~15 GB unpacked), plus HF's download cache keeps the .incomplete
#: file beside the final one until it is done.
_DISK_HEADROOM_GB = 25
#: A model this large is not something to size a disk for by guesswork — it is
#: the huge-weight case, and belongs on a volume. The warning covers it.
_DISK_SIZING_CAP_GB = 400


def estimate_weights_gb(entry: dict, mcfg: "RunpodModelConfig" = None) -> float:
    """How many GB the model's weights will take on the pod, or 0.0 if unknown.

    A stated size wins: `weights_gb` on the model entry (or the runpod block)
    is taken as-is. It is the only honest answer for a URL download, an upload,
    or a local file — none of which the estimator can see — and for an HF repo
    it beats a heuristic that deliberately errs high. Set it when you know.

    Otherwise asked of HuggingFace before renting, because the alternative was
    observed: a pod boots, pulls the image, starts the download, fills its disk,
    and dies at load_timeout_s having paid for every minute of it. The size of a
    repo is one metadata call; the disk it needs is that plus headroom.
    """
    for src in (entry or {}, (entry or {}).get("runpod") or {}):
        stated = src.get("weights_gb") if isinstance(src, dict) else None
        if stated is not None:
            try:
                val = float(stated)
                if val > 0:
                    return val
            except (TypeError, ValueError):
                pass
    try:
        from codai.api.runpod_worker import resolve_model_source
        kind, value = resolve_model_source(entry or {}, mcfg) if mcfg else ("hf", "")
    except Exception:
        kind, value = "hf", ""
    repo = value if kind == "hf" and value else str((entry or {}).get("path") or "")
    if not repo or repo.startswith(("/", "~", ".")) or "/" not in repo:
        return 0.0                       # a local path or a URL: nothing to ask
    try:
        from huggingface_hub import model_info as _hf_model_info
        files = _hf_model_info(repo, files_metadata=True).siblings or []
    except Exception:
        return 0.0
    pattern = str((entry or {}).get("file_pattern") or "")
    if pattern:
        import fnmatch
        pats = [pattern] if "/" in pattern else [f"*{pattern}"]
        files = [f for f in files if any(fnmatch.fnmatch(f.rfilename, q) for q in pats)]
        return sum((f.size or 0) for f in files) / 1e9
    # No pattern: count the weights a load will actually fetch, not the whole
    # repo. SDXL's repo is 77 GB, of which a diffusers load pulls 28: the rest is
    # the same weights again as Flax .msgpack, legacy .bin and ONNX exports.
    # Sizing a disk to the repo would rent 100 GB for a 28 GB model.
    def _loaded(name: str) -> bool:
        low = name.lower()
        if low.endswith((".onnx", ".onnx_data", ".msgpack", ".h5", ".tflite", ".ot")):
            return False
        if low.endswith(".bin") and any(f.rfilename.endswith(".safetensors") for f in files):
            return False                 # .bin only matters when no safetensors exist
        if "onnx" in low or "/flax" in low or "openvino" in low:
            return False
        return low.endswith((".safetensors", ".bin", ".gguf", ".pt", ".pth", ".ckpt",
                             ".json", ".txt", ".model", ".tiktoken", ".spm"))
    return sum((f.size or 0) for f in files if _loaded(f.rfilename)) / 1e9


def disk_for(entry: dict, mcfg: "RunpodModelConfig", account=None) -> tuple:
    """(container_disk_gb, note) — the configured disk, grown to fit the weights.

    Grows, never shrinks: an explicit container_disk_gb larger than the estimate
    is respected, and one too small for the model is raised with a note saying
    by how much and why. Says nothing when the estimate is unavailable.

    With a network volume attached the weights land THERE, not on the container
    disk — that is the point of the volume — so the disk only has to hold the
    unpacked image and is left at its configured size.
    """
    configured = int(getattr(mcfg, "container_disk_gb", 40) or 40)
    vol, _ = volume_for(mcfg, account)
    if vol:
        return configured, ""
    weights = estimate_weights_gb(entry, mcfg)
    if weights <= 0:
        return configured, ""
    needed = int(weights + _DISK_HEADROOM_GB + 0.999)
    if needed > _DISK_SIZING_CAP_GB:
        return configured, (f"weights are ~{weights:.0f} GB — beyond what a container "
                            f"disk should hold; this model belongs on a network volume")
    if needed <= configured:
        return configured, ""
    return needed, (f"container disk raised {configured} -> {needed} GB: the weights "
                    f"are ~{weights:.1f} GB and the image and download cache need "
                    f"~{_DISK_HEADROOM_GB} GB beside them")


def volume_map_for(mcfg: "RunpodModelConfig", account) -> dict:
    """{REGION: volume_id} for this pool, or {}.

    The pool's own map wins whole — not merged with the account's. A model that
    lists its regions' volumes is making a complete statement about where its
    data may live, and silently inheriting another region from the account map
    would place weights somewhere it did not name.
    """
    own = _parse_volume_map(getattr(mcfg, "network_volumes", None))
    if own:
        return own
    return _parse_volume_map(getattr(account, "network_volumes", None))


def volume_for(mcfg: "RunpodModelConfig", account, dc: str = "") -> tuple:
    """(volume_id, mount_path) for this pool in region ``dc``, or ('', '').

    With a per-region map configured, ``dc`` selects the volume — so the caller
    that creates a pod must pass the region its candidate was ranked for, and
    does. Callers with no region in hand (sizing the container disk, warning
    about huge weights, validating venv_on_volume, building a path under the
    mount) ask only whether storage exists and where it is mounted, both of
    which are the same for every region; they get the first entry so those
    checks keep working, and the mount path is shared by construction.
    """
    vmap = volume_map_for(mcfg, account)
    if vmap:
        key = str(dc or "").strip().upper()
        vol = vmap.get(key) or ("" if key else next(iter(vmap.values())))
    else:
        vol = (getattr(mcfg, "network_volume_id", "") or "").strip() \
            or (getattr(account, "network_volume_id", "") or "").strip()
    if not vol:
        return "", ""
    mount = (getattr(mcfg, "volume_mount_path", "") or "").strip() \
        or (getattr(account, "volume_mount_path", "") or "").strip() or "/workspace"
    return vol, mount.rstrip("/") or "/workspace"


def global_volume_for(mcfg: "RunpodModelConfig", account) -> str:
    """The GLOBAL volume id for this pool, or ''.

    Deliberately separate from :func:`volume_for`: a global volume is
    region-independent, so unlike a network volume it must NOT pin the pod's
    data centre. Keeping them apart is what lets the scaler rent whichever
    region has a card while the weights stay in one place.
    """
    return ((getattr(mcfg, "global_volume_id", "") or "").strip()
            or (getattr(account, "global_volume_id", "") or "").strip())


#: The dependency-free image that runs coderai from a venv on a volume.
SLIM_POD_IMAGE = os.environ.get("CODERAI_SLIM_POD_IMAGE",
                                f"{CAPABILITY_IMAGE_REPO}-slim:{CAPABILITY_IMAGE_TAG}")


def venv_bootstrap_script(mount: str, profile: str, venv_name: str = "",
                          port: int = 8000) -> str:
    """Start command for a slim pod: build the venv on the volume if needed, run.

    The first pod to use a given venv pays for the install; every pod after finds
    it and starts in seconds. A completion marker is written LAST and checked
    first, so a pod that dies mid-install cannot leave a half-built venv that
    later pods would import from and fail on in confusing ways.

    Concurrent builders are the other hazard — two pods starting together would
    install into the same directory. A lock directory (mkdir is atomic) makes the
    second one wait for the first rather than interleave with it.
    """
    name = (venv_name or profile or "default").strip()
    venv = f"{mount}/venvs/{name}"
    reqs = "/opt/coderai/app/packaging/runpod/profiles"
    return "\n".join([
        "set -eu",
        f'VENV={_sh_quote(venv)}',
        f'LOCK={_sh_quote(venv + ".lock")}',
        f'MARK={_sh_quote(venv + "/.ready")}',
        # Wait for another pod that is already building this venv.
        'for i in $(seq 1 180); do',
        '  [ -f "$MARK" ] && break',
        '  if mkdir "$LOCK" 2>/dev/null; then',
        '    echo "[venv] building $VENV (first pod pays for this)"',
        '    python -m venv "$VENV"',
        f'    "$VENV/bin/python" -m pip install --upgrade pip',
        f'    "$VENV/bin/python" -m pip install -r {reqs}/core.txt',
        f'    "$VENV/bin/python" -m pip install -r {reqs}/{profile}.txt',
        '    touch "$MARK"',        # last: a crash leaves no usable marker
        '    rmdir "$LOCK" || true',
        '    break',
        '  fi',
        '  echo "[venv] another pod is building $VENV — waiting"',
        '  sleep 10',
        'done',
        '[ -f "$MARK" ] || { echo "[venv] $VENV never became ready"; exit 1; }',
        'echo "[venv] using $VENV"',
        # TLS from env, as boot.sh does it (this script replaces boot.sh).
        'TLS=""; if [ -n "$CODERAI_TLS_CERT" ] && [ -n "$CODERAI_TLS_KEY" ]; then '
        'mkdir -p /run/coderai-tls && chmod 700 /run/coderai-tls; '
        'printf "%s\\n" "$CODERAI_TLS_CERT" > /run/coderai-tls/cert.pem; '
        'printf "%s\\n" "$CODERAI_TLS_KEY" > /run/coderai-tls/key.pem; '
        'chmod 600 /run/coderai-tls/key.pem; unset CODERAI_TLS_KEY; '
        'TLS="--ssl-certfile /run/coderai-tls/cert.pem --ssl-keyfile /run/coderai-tls/key.pem"; fi',
        f'exec "$VENV/bin/python" -m uvicorn codai.api.app:app --host 0.0.0.0 '
        f'--port {int(port)} $TLS',
    ])


#: Engine settings that describe the MODEL and how to run it, and so travel to
#: a pod. Not in the list, deliberately: enabled (the pod gets its own switch),
#: install_dir / repo_url / auto_build (the image has the binary), host / port /
#: service_url (this machine's wiring), model_path (a local file).
_ENGINE_FORWARDED_FIELDS = {
    "ds4": ("ctx", "ssd_streaming", "extra_args", "extra_env", "model_variant",
            "auto_download", "expert_cache_reserve_gb", "model_id"),
    "colibri": ("ctx", "kv_slots", "cap", "cuda_expert_gb", "extra_args",
                "extra_env", "model_id"),
    "k3": ("ctx", "preset", "trunk_gb", "cache_gb", "extra_args", "extra_env",
           "model_id"),
    "kt": ("ctx", "extra_args", "extra_env", "model_id"),
    "vllm": ("ctx", "gpu_memory_utilization", "tensor_parallel_size", "max_num_seqs",
             "dtype", "quantization", "extra_args", "extra_env",
             # Only ever forwarded when TRUE: the loop below skips False, so
             # adding it here cannot change an install that has not asked for it.
             # A pod needs it to recover a byte-level vocabulary's characters from
             # the token ids (see VllmConfig.return_token_ids).
             "return_token_ids"),
}
#: The config attribute behind each engine name (ktransformers is the odd one).
_ENGINE_CONFIG_ATTR = {"kt": "ktransformers"}


def engine_config_for_pod(engine: str) -> dict:
    """This install's settings for a native engine, minus what is local to it."""
    try:
        from codai.admin.routes import config_manager
        cfg = getattr(getattr(config_manager, "config", None),
                      _ENGINE_CONFIG_ATTR.get(engine, engine), None)
    except Exception:
        cfg = None
    if cfg is None:
        return {}
    out = {}
    for k in _ENGINE_FORWARDED_FIELDS.get(engine, ()):
        if not hasattr(cfg, k):
            continue
        v = getattr(cfg, k)
        if v in (None, "", 0, False):
            continue
        out[k] = v
    if engine == "ds4" and "auto_download" not in out:
        # A pod that cannot fetch its own weights is a pod that answers
        # nothing. Locally this is opt-in because the download is 154 GB onto
        # someone's disk; on a pod the disk was rented for exactly that.
        out["auto_download"] = True
    return out


def pod_volume_path(mcfg: "RunpodModelConfig", account=None) -> str:
    """Where the weights already are ON THE POD, or ''.

    `volume_path` on the runpod block names a file or directory on the network
    volume — relative to its mount, or absolute. It is how a 150 GB engine
    model is used without any pod ever downloading it: put it there once.
    """
    rel = str(getattr(mcfg, "volume_path", "") or "").strip()
    if not rel:
        return ""
    if rel.startswith("/"):
        return rel
    _vol, mount = volume_for(mcfg, account)
    return f"{mount.rstrip('/')}/{rel}"


def volume_env(mount: str) -> dict:
    """Point everything that writes big files at the volume.

    This is what makes a volume worth attaching. Without it a pod downloads its
    weights to container disk and throws them away on teardown, so every cold pod
    pays the download again; with it, the second pod finds them already there.
    The same path is where uploaded models and LoRA adapters land, so a file sent
    once is visible to every later pod — the "bounce host" role, without a third
    party in the middle.
    """
    if not mount:
        return {}
    return {
        "HF_HOME": f"{mount}/huggingface",
        "HUGGINGFACE_HUB_CACHE": f"{mount}/huggingface/hub",
        "TRANSFORMERS_CACHE": f"{mount}/huggingface/transformers",
        "DIFFUSERS_CACHE": f"{mount}/diffusers",
        "CODERAI_MODELS_DIR": f"{mount}/models",
        "CODERAI_CACHE_DIR": f"{mount}/cache",
        # ds4's download_model.sh writes here; on the volume it survives the
        # pod, and the move into the GGUF cache beside it is a rename rather
        # than a 154 GB copy.
        "DS4_GGUF_DIR": f"{mount}/cache/ds4",
    }


def pod_ocr_engine(requested: str) -> str:
    """The OCR engine a pod will actually serve for a request naming ``requested``.

    Mirrors the env a pod is launched with: the VLM engines (surya, olmOCR,
    chandra) need a model server the image does not carry, so a pod serves docTR
    for them. The gateway uses this to rewrite the request's own `engine` field,
    because the pod honours what the request asks for over its configured default.
    """
    r = (requested or "").strip().lower()
    if r in ("surya", "olmocr", "chandra"):
        return "doctr"
    return r if r in ("paddle", "doctr") else "doctr"


def _surya_accepted() -> bool:
    """Whether this deployment has accepted Surya's licence."""
    try:
        from codai.admin.routes import config_manager
        return bool(getattr(getattr(config_manager, "config", None),
                            "ocr", None).surya_accept_license)
    except Exception:
        return False


def _chandra_accepted() -> bool:
    """Whether this deployment has accepted Chandra's weights licence."""
    try:
        from codai.admin.routes import config_manager
        return bool(getattr(getattr(getattr(config_manager, "config", None),
                                    "ocr", None), "chandra_accept_license", False))
    except Exception:
        return False


def _ocr_cfg():
    """This install's OcrConfig, or None. The pod has no config of its own — every
    OCR setting it runs with is seeded from here."""
    try:
        from codai.admin.routes import config_manager
        return getattr(getattr(config_manager, "config", None), "ocr", None)
    except Exception:
        return None


#: The image that carries the OCR pipeline: surya's client venv plus pypdfium2.
#: Selecting it IS selecting the mode -- see pod_runs_ocr_pipeline().
OCR_PIPELINE_IMAGE_HINT = "coderai-vllm-ocr"


def image_runs_ocr_pipeline(image: str) -> bool:
    """Whether `image` carries the OCR pipeline (the surya venv + rasteriser).

    Matched on the image NAME rather than probed, because the decision is made
    before any pod exists. A private mirror or a pinned tag still matches, since
    the hint is a substring of the repository.
    """
    return OCR_PIPELINE_IMAGE_HINT in str(image or "").strip().lower()


def pod_runs_ocr_pipeline(mcfg, image: str = "") -> bool:
    """Whether THIS model's pods run the OCR pipeline themselves.

    THE IMAGE IS THE SWITCH, and deliberately the only one:

      * `coderai-vllm`      — the pod serves the model; the ORCHESTRATOR
                              rasterises and runs surya, as it always has.
      * `coderai-vllm-ocr`  — the pod carries the surya client venv and
                              pypdfium2, so it is sent a one-page PDF and does
                              the whole page itself.

    A separate "run it on the pod" flag was built first and then removed: two
    switches that can disagree is precisely the invisible mismatch that cost
    three rounds of blaming a pod image this morning. Choosing the image IS
    choosing the mode, it is already in models.json and on the model page, and
    an image that cannot run a pipeline can no longer be asked to.
    """
    return image_runs_ocr_pipeline(str(image or getattr(mcfg, "image", "") or ""))


def _ocr_external_server(engine: str):
    """The off-box OpenAI-compatible endpoint configured for a VLM OCR engine, or "".

    This is what decides whether a pod can actually serve surya/olmOCR/chandra. All
    three are VLMs
    and need a model server: locally they ride coderai's own vLLM, which a pod has
    none of, and baking one in would mean a SECOND torch in the image (the vLLM venv
    alone is ~10.5 GB against a ~12 GB ceiling). But in the served modes the engine
    only speaks HTTP — PIL and requests, both already in the image — so a pod CAN run
    them when the operator has pointed them at a server, including at a vLLM pod."""
    o = _ocr_cfg()
    if o is None:
        return ""
    if engine == "olmocr":
        mode = str(getattr(o, "olmocr_serve", "") or "").strip().lower()
        url = str(getattr(o, "olmocr_server_url", "") or "").strip()
        return url if (mode == "server" and url) else ""
    if engine == "chandra":
        mode = str(getattr(o, "chandra_serve", "") or "").strip().lower()
        url = str(getattr(o, "chandra_server_url", "") or "").strip()
        return url if (mode == "server" and url) else ""
    if engine == "surya":
        mode = str(getattr(o, "surya_serve", "") or "").strip().lower()
        url = str(getattr(o, "surya_server_url", "") or "").strip()
        return url if (mode == "llamacpp" and url) else ""
    return ""


#: Seed lists already announced, so a plan recomputed every scaler pass does
#: not repeat itself. Bounded: this must never become a place memory grows.
_PLAN_SAID: dict = {}


def _plan(engine, image, args, mcfg, api_key, entry, served,
          seed_entries: list = None, capability: str = "") -> dict:  # noqa: D401
    """Finish a plan for a coderai pod: auth plus the models it must serve."""
    import json as _json
    env = {}
    if api_key:
        env["CODERAI_API_TOKEN"] = api_key
    # How hard the pod's own server may be driven. Without this the pod runs the
    # shared-workstation defaults and its admission gate, not its GPU, is the
    # ceiling.
    if getattr(mcfg, "pod_max_parallel_requests", 0):
        env["CODERAI_SERVER_MAX_PARALLEL_REQUESTS"] = str(int(mcfg.pod_max_parallel_requests))
    if getattr(mcfg, "pod_queue_max_size", 0):
        env["CODERAI_SERVER_QUEUE_MAX_SIZE"] = str(int(mcfg.pod_queue_max_size))
    # A pod has ONE client — the install that rented it — so per-IP rate limiting
    # throttles its owner and nobody else. The default 60 requests/min on
    # /v1/chat/completions is below what a single A40 serves, so it threw away
    # GPU time that is being paid for by the hour, and made three OCR engines
    # look like they fell over at 32 concurrent pages when they were simply out
    # of window. Admission here is the pod's parallel-request and queue limits.
    env["CODERAI_SERVER_RATE_LIMIT_ENABLED"] = "0"
    # This pod serves a shared POOL: several names over one copy of the weights,
    # which for text models means a base and its adapters. vLLM cannot turn
    # `--enable-lora` on after it has started, so an engine launched without it
    # has to RELOAD the base model the first time an adapter arrives — minutes,
    # on a card billed by the hour. Telling the pod now costs a buffer
    # allocation and lets an adapter be added to the running engine instead.
    if str(getattr(mcfg, "pool", "") or "").strip():
        env["CODERAI_VLLM_ENABLE_LORA"] = "1"
    seeds = []
    if entry:
        kind, value = resolve_model_source(entry, mcfg)
        one = seed_model_env(entry, served, value if kind in ("hf", "url") else "",
                             volume_path=pod_volume_path(mcfg, _account_hint()),
                             engine=engine_pod_backend(entry, mcfg))
        if one:
            seeds.extend(_json.loads(one))
    for extra in (seed_entries or []):
        # …with the ENGINE too. Only the pool's own entry carried one, so a
        # sibling arrived on the pod with no backend and fell through to the
        # pod's default:
        #     503 Model 'diogene-alpha' failed to load: Unknown backend: cpu
        # It did not bite while an adapter collapsed into its base on the pod —
        # it was never a model of its own to load. Making it one exposed this.
        #
        # The POOL's engine, always. A pool is one pod running one image, so
        # siblings cannot each have their own: seeding a sibling's different
        # engine would describe something the pod cannot satisfy, and the
        # failure would land minutes later inside a rented machine.
        one = seed_model_env(extra, seed_served_name(extra), "",
                             engine=engine_pod_backend(entry or {}, mcfg))
        if one:
            seeds.extend(_json.loads(one))
    if seeds:
        # De-duplicate by what the pod will SERVE it as, not by path: a capability
        # pod's list can overlap the one model a per-model pool also seeds, which
        # is what this guards against — but a base model and a LoRA of it share
        # one path by definition, so keying on path silently dropped the adapter
        # and the pod was "told about 1 model(s)" when it needed two.
        seen, unique = set(), []
        for s in seeds:
            key = (s.get("path"), s.get("alias") or "", s.get("id") or "",
                   _json.dumps(s.get("loras") or [], sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            unique.append(s)
        env["CODERAI_SEED_MODELS"] = _json.dumps(unique)
        # Once per distinct seed list. A plan is recomputed on every scaler
        # pass — roughly every fifteen seconds while a pod boots — and saying
        # the same sentence each time buries the lines that matter in a boot
        # log somebody is reading precisely because something went wrong.
        _sig = (str(image), env["CODERAI_SEED_MODELS"])
        if _PLAN_SAID.get(_sig) is None:
            while len(_PLAN_SAID) > 64:
                _PLAN_SAID.pop(next(iter(_PLAN_SAID)))
            _PLAN_SAID[_sig] = True
            print(f"[runpod] pod will be told about {len(unique)} model(s)",
                  flush=True)
    # A pod rented for a subsystem that ships disabled has to arrive with it on.
    # An OCR pod booted, accepted the request and answered "OCR subsystem is
    # disabled (enable it in Settings → OCR)" — a settings screen nobody will
    # open on a machine that exists for the next four minutes.
    cap = capability or model_capability(entry or {}, include_text=True)
    if cap == "audio_gen":
        # audiocraft and transformers are not equivalent — melody conditioning
        # and AudioGen exist only in the first — so the choice travels to the pod
        # rather than being decided by whatever happens to be installed there.
        choice = str((entry or {}).get("audio_backend") or "").strip().lower()
        if choice in ("audiocraft", "transformers"):
            env["CODERAI_AUDIO_BACKEND"] = choice
    # A HuggingFace token, for gated models. The local install has one in its
    # environment; a pod has nothing unless it is sent — so pyannote, which is
    # gated, loaded from its venv, reached from_pretrained, and stopped there
    # with "the model is gated/unavailable". Sent as pod ENVIRONMENT only: it is
    # a secret, and never belongs in an image, a log, or the catalogue.
    if "HF_TOKEN" not in env:
        try:
            from codai.api.pyannote_worker import _resolve_hf_token
            _tok = _resolve_hf_token(entry or {})
        except Exception:
            _tok = ""
        if _tok:
            env["HF_TOKEN"] = _tok

    # Subsystems whose image bakes a venv: tell the worker where it is, or it
    # tries to build one on a machine rented by the second. Harmless on an image
    # that has no such venv — the worker falls back to its own search.
    for _var, _name in (("CODERAI_PYANNOTE_VENV", "pyannote"),
                        ("CODERAI_NEMO_VENV", "nemo"),
                        ("CODERAI_CRISPERWHISPER_VENV", "transformers"),
                        ("CODERAI_XTTS_VENV", "TTS")):
        env.setdefault(_var, f"/opt/coderai/venvs/{_name}")

    if cap in ("tts", "voice"):
        # coqui XTTS is under the CPML licence and asks for acceptance with an
        # interactive prompt on stdout — which on a pod both corrupts the worker
        # protocol and blocks forever on input(). The decision is made HERE by
        # configuring the model at all, exactly as surya's licence is; the pod
        # inherits it and coqui skips the prompt.
        env.setdefault("COQUI_TOS_AGREED", "1")

    # A pod on an image that CARRIES the pipeline runs it, whatever capability
    # the model is filed under. The image is the switch (pod_runs_ocr_pipeline),
    # so this does not sit inside the `cap == "ocr"` branch: the model that
    # serves surya-ocr-2 is a VLM like any other, and filing it as an OCR
    # capability pod is not what makes its pod able to read a page.
    if pod_runs_ocr_pipeline(mcfg, image):
        _o = _ocr_cfg()
        # WHICH engine this pod's pipeline is for. A pod serves one model, and
        # that model is what decides -- surya-ocr-2, chandra-ocr-2 or olmOCR.
        # Hardcoding surya told a chandra pod to run surya against chandra's
        # model id, which would have produced plausible-looking text from the
        # wrong engine.
        _served_l = str(served or "").lower()
        _eng = "surya"
        for _cand in ("chandra", "olmocr", "surya"):
            if _cand in _served_l:
                _eng = _cand
                break
        env["CODERAI_OCR_ENABLED"] = "1"
        env[f"CODERAI_OCR_{_eng.upper()}_ENABLED"] = "1"
        env["CODERAI_OCR_DEFAULT_ENGINE"] = _eng
        # The baked venv, so the pod does not try to BUILD one on a rented
        # machine (minutes, and it would land on the container disk). Only
        # surya has one: chandra and olmOCR in served mode speak HTTP from the
        # main venv (PIL + requests), which is why the pod side is nearly free
        # for them and needed a 1.6 GB venv for surya.
        if _eng == "surya":
            env["CODERAI_OCR_SURYA_VENV"] = "/opt/coderai/venvs/surya"
        # Surya drives the model through the pod's OWN coderai, in the same
        # "model" mode the orchestrator uses -- which is what makes the gateway
        # mint it an attach token instead of needing a bearer hand-wired here.
        # Over loopback, so a page resumed four times costs four local round
        # trips rather than four across the internet.
        env[f"CODERAI_OCR_{_eng.upper()}_SERVE"] = "model"
        env[f"CODERAI_OCR_{_eng.upper()}_MODEL_ID"] = str(served or "").strip()
        # Licences are inherited, never decided by the pod: surya is GPL and
        # chandra's weights are a modified OpenRAIL-M. Asking for the engine by
        # name HERE is the decision.
        if _eng == "surya":
            env["CODERAI_OCR_SURYA_ACCEPT_LICENSE"] = "1"
        elif _eng == "chandra":
            env["CODERAI_OCR_CHANDRA_ACCEPT_LICENSE"] = "1"
        # This image carries NO docTR -- `auto` would route a failed page to an
        # engine that is not there. This engine or nothing, said out loud.
        env["CODERAI_OCR_FALLBACK_ENGINES"] = "off"
        # The pod launches its OWN vLLM from its OWN config, so a vLLM setting
        # made on the renting install does not reach it. That is not a detail:
        # `--return-tokens-as-token-ids` is what the accent repair rebuilds
        # from, an operator had set it in `vllm.extra_args` here, and the pods
        # silently ran without it -- a fixed bug, back, invisibly. The flag is
        # now forced for byte-level OCR models in vllm_worker, and the
        # operator's own choice travels too, so neither door is the only one.
        env["CODERAI_VLLM_RETURN_TOKEN_IDS"] = "1"
        # Same reasoning for the degraded-page re-read, and one more: only the
        # ORCHESTRATOR can reach another engine's pods. A pod re-reading its own
        # degraded page would re-read it on the one engine that just produced
        # it, which is the retry that already failed.
        env["CODERAI_OCR_RETRY_DEGRADED"] = "off"
        # And the TUNING. Every one of these was set in response to a measured
        # failure, and a pod reading pages at surya's defaults instead would be
        # indistinguishable from the "pre-fix pod image" that got blamed three
        # times for orchestrator bugs.
        # Per-engine tuning, by name, so a chandra pod gets chandra's budgets
        # rather than surya's. Every one of these was set in response to a
        # measured failure, and a pod running the engine's own defaults instead
        # is indistinguishable from the "pre-fix pod image" that got blamed
        # three times for orchestrator bugs.
        _E = _eng.upper()
        _per_engine = [(f"CODERAI_OCR_{_E}_DPI", f"{_eng}_dpi"),
                       (f"CODERAI_OCR_{_E}_MAX_TOKENS", f"{_eng}_max_tokens"),
                       (f"CODERAI_OCR_{_E}_MAX_CONTINUATIONS",
                        f"{_eng}_max_continuations"),
                       (f"CODERAI_OCR_{_E}_CONTEXT_TOKENS",
                        f"{_eng}_context_tokens"),
                       (f"CODERAI_OCR_{_E}_MAX_TOKENS_FLOOR",
                        f"{_eng}_max_tokens_floor"),
                       (f"CODERAI_OCR_{_E}_MAX_RETRIES", f"{_eng}_max_retries"),
                       (f"CODERAI_OCR_{_E}_INSTANCES", f"{_eng}_instances"),
                       (f"CODERAI_OCR_{_E}_LANES", f"{_eng}_lanes"),
                       (f"CODERAI_OCR_{_E}_LANGS", f"{_eng}_langs")]
        if _eng == "surya":
            _per_engine += [
                ("CODERAI_OCR_SURYA_REPEAT_MAX_REPEATS",
                 "surya_repeat_max_repeats"),
                ("CODERAI_OCR_SURYA_REPEAT_WINDOW", "surya_repeat_window"),
                ("CODERAI_OCR_SURYA_REPEAT_SCALING", "surya_repeat_scaling")]
        for _var, _attr in _per_engine + [
                            ("CODERAI_OCR_DPI", "dpi"),
                            ("CODERAI_OCR_MAX_CONCURRENCY", "max_concurrency"),
                            ("CODERAI_OCR_PAGE_WINDOW", "page_window"),
                            ("CODERAI_OCR_RASTER_WORKERS", "raster_workers"),
                            ("CODERAI_OCR_BLANK_PAGE_MAX_CHARS",
                             "blank_page_max_chars"),
                            ("CODERAI_OCR_DETECT_MODE", "detect_mode"),
                            ("CODERAI_OCR_DETECT_MODEL_PATH", "detect_model_path"),
                            ("CODERAI_OCR_DETECT_CONF", "detect_conf")]:
            _val = getattr(_o, _attr, None) if _o is not None else None
            if _val not in (None, "", 0, 0.0):
                env[_var] = str(_val)
        for _var, _attr in (("CODERAI_OCR_SURYA_REPAIR_ACCENTS",
                             "surya_repair_accents"),
                            ("CODERAI_OCR_SURYA_TOKEN_TRACE", "surya_token_trace")):
            if _o is not None and getattr(_o, _attr, None) is not None:
                env[_var] = "1" if getattr(_o, _attr) else "0"
        # The detector's weights are operator-supplied and a released image
        # carries none, so say plainly when a mode is asked for that the pod
        # cannot honour rather than letting it report a page with no stamps.
        if str(getattr(_o, "detect_mode", "off") or "off").lower() in (
                "detector", "both") and not str(
                    getattr(_o, "detect_model_path", "") or "").strip():
            print("[runpod] ocr.detect_mode wants the YOLO detector but "
                  "ocr.detect_model_path is empty — a pod cannot acquire weights "
                  "it was never given, and will report no stamps rather than "
                  "fail. Set the path (staged on the pod) or use detect_mode="
                  "layout.", flush=True)
        print(f"[runpod] the {_eng} OCR PIPELINE runs on this pod ({image}): it "
              f"is sent one page as a PDF and reads it on the card, with its own "
              f"vLLM over loopback", flush=True)

    if cap == "ocr":
        env["CODERAI_OCR_ENABLED"] = "1"
        # Which engine this pod is for. An entry is routable by its ALIAS before its
        # path (the `alias or path or id` convention used throughout the manager), and
        # aliases are how one model carries several configs and how several models sit
        # behind one endpoint — so an aliased OCR entry must not be misread as "no
        # engine named that" and silently downgraded. Take the first candidate that
        # names a real engine; fall back to the path, as before.
        _cands = [str((entry or {}).get("alias") or "").strip().lower(),
                  str(served or "").strip().lower(),
                  str((entry or {}).get("path") or "").strip().lower()]
        wanted = next((c for c in _cands
                        if c in ("paddle", "doctr", "surya", "olmocr", "chandra")),
                      _cands[-1])
        # Remember what was ASKED for: the licence decision below is made by asking
        # for surya by name, and that has to survive a downgrade to docTR.
        _asked = wanted
        if wanted in ("surya", "olmocr", "chandra"):
            # Surya 0.22 is a VLM ("surya-ocr-2") and olmOCR-2 is one outright, not a
            # self-contained OCR library: both need a model server. Locally they ride
            # coderai's own vLLM engine, which a pod has none of, and baking one in
            # would put a SECOND torch in the image (the vLLM venv alone is ~10.5 GB
            # against a ~12 GB ceiling).
            #
            # But in their SERVED modes these engines only speak HTTP, and PIL and
            # requests are already in the image. So when the operator has pointed the
            # engine at a reachable endpoint — including at a vLLM pod, which is the
            # composition this is for — the pod runs the engine it was asked for, and
            # the settings are seeded below. Only with nothing configured does it fall
            # back, because loading a VLM with no server reaches inference and dies.
            _srv = _ocr_external_server(wanted)
            if _srv:
                _o = _ocr_cfg()
                if wanted == "olmocr":
                    env["CODERAI_OCR_OLMOCR_SERVE"] = "server"
                    env["CODERAI_OCR_OLMOCR_SERVER_URL"] = _srv
                    for _v, _a in (("CODERAI_OCR_OLMOCR_MODEL", "olmocr_model"),
                                   ("CODERAI_OCR_OLMOCR_MODEL_ID", "olmocr_model_id")):
                        _val = str(getattr(_o, _a, "") or "").strip()
                        if _val:
                            env[_v] = _val
                elif wanted == "chandra":
                    env["CODERAI_OCR_CHANDRA_SERVE"] = "server"
                    env["CODERAI_OCR_CHANDRA_SERVER_URL"] = _srv
                    for _v, _a in (("CODERAI_OCR_CHANDRA_MODEL", "chandra_model"),
                                   ("CODERAI_OCR_CHANDRA_MODEL_ID", "chandra_model_id")):
                        _val = str(getattr(_o, _a, "") or "").strip()
                        if _val:
                            env[_v] = _val
                else:
                    env["CODERAI_OCR_SURYA_SERVE"] = "llamacpp"
                    env["CODERAI_OCR_SURYA_SERVER_URL"] = _srv
                print(f"[runpod] {wanted} on a pod will use the configured server at "
                      f"{_srv} (the image carries no VLM server of its own)", flush=True)
            else:
                print(f"[runpod] {wanted} on a pod needs a VLM server the image does "
                      f"not carry and none is configured (set "
                      f"ocr.{wanted}_server_url and the served mode) — this OCR "
                      f"request will be served by docTR", flush=True)
                wanted = "doctr"
        if wanted not in ("paddle", "doctr", "surya", "olmocr", "chandra"):
            # The pod images ship docTR, the only engine that runs in-process.
            wanted = "doctr"
        env["CODERAI_OCR_DEFAULT_ENGINE"] = wanted
        # The subsystem flag is not enough: each engine has a gate of its own,
        # and all but paddle default to off.
        env[f"CODERAI_OCR_{wanted.upper()}_ENABLED"] = "1"
        if wanted == "paddle":
            # The paddle image bakes it, because paddlepaddle brings its own
            # CUDA runtime. Point the engine at that interpreter rather than
            # letting it try to build a venv on a rented machine.
            env["CODERAI_OCR_PADDLE_VENV"] = "/opt/coderai/venvs/paddleocr"
        # The fallback chain (ocr.fallback_engines) must not name an engine this image
        # cannot serve: "auto" would send a failed page to surya/olmocr, which on a pod
        # have no server unless one was seeded just above. Pin it to what is present.
        _fb = ["doctr"] if wanted != "paddle" else ["paddle", "doctr"]
        if wanted not in _fb:
            _fb.insert(0, wanted)
        env["CODERAI_OCR_FALLBACK_ENGINES"] = ",".join(_fb)
        if _asked == "surya" or _surya_accepted():
            # Surya is GPL and gated on an explicit acceptance. The pod inherits
            # the decision made HERE — it cannot make it for itself, and asking
            # for surya by name is that decision.
            env["CODERAI_OCR_SURYA_ACCEPT_LICENSE"] = "1"
        if _asked == "chandra" and _chandra_accepted():
            # Chandra's weights are a modified OpenRAIL-M, on NARROWER terms than
            # Surya's GPL (no competitive use against Datalab's API, $2M ceiling).
            # So asking for it by name is NOT the acceptance — unlike surya above,
            # this one is only ever inherited from a decision already recorded in
            # this install's config. A pod cannot accept a licence on an
            # operator's behalf.
            env["CODERAI_OCR_CHANDRA_ACCEPT_LICENSE"] = "1"

    eng = engine_pod_backend(entry or {}, mcfg)
    if eng:
        # The engine ships disabled, like OCR does, and its settings live in
        # THIS install's config.json (ctx, expert cache, extra args, download
        # variant). The pod gets the switch and the settings as environment —
        # never its install_dir or ports, which describe this machine.
        env[f"CODERAI_{eng.upper()}_ENABLED"] = "1"
        if eng == "kt":
            # SGLang + kt-kernel live in the image's own venv; the worker must
            # launch from there, not from coderai's interpreter.
            env.setdefault("CODERAI_KT_VENV", "/opt/coderai/venvs/sglang")
        if eng == "vllm":
            env.setdefault("CODERAI_VLLM_VENV", "/opt/coderai/venvs/vllm")
        forwarded = engine_config_for_pod(eng)
        # The MODEL's own engine settings beat this install's. engine_config_for_pod
        # reads config.json, which describes the local engine, so a pod was
        # launched with the install-wide context window and a per-model `ctx` on
        # the runpod block did nothing: an entry asking for 18432 got a pod
        # serving 32768. Harmless-looking until something publishes the window —
        # an OCR engine's resumption arithmetic is driven by it, and two
        # different numbers for one pod is how a resume either stops early or
        # overruns and 400s.
        _own = int(getattr(mcfg, "ctx", 0) or 0)
        if _own:
            if int(forwarded.get("ctx") or 0) != _own:
                print(f"[runpod] pod context window {_own} (this model's own "
                      f"ctx, over the install's "
                      f"{forwarded.get('ctx') or 'default'})", flush=True)
            forwarded["ctx"] = _own
        if eng == "vllm" and "tensor_parallel_size" in forwarded:
            # The pod was rented with gpu_count cards; the local setting
            # describes this machine's cards. The pod's count wins.
            forwarded.pop("tensor_parallel_size", None)
        if eng == "vllm" and int(getattr(mcfg, "gpu_count", 1) or 1) > 1:
            forwarded["tensor_parallel_size"] = int(mcfg.gpu_count)
        if forwarded:
            env[f"CODERAI_{eng.upper()}_CONFIG"] = _json.dumps(forwarded)
        warn = weight_transfer_warning(entry, mcfg, _account_hint())
        if warn:
            print(f"[runpod] WARNING: {warn}", flush=True)

    plan = {"engine": engine, "image": image, "args": args, "env": env,
            "health_path": mcfg.health_path or _HEALTH_PATHS.get(engine, "/healthz")}

    # Dependencies on the volume instead of in the image: a small image, and a
    # venv the first pod builds and every later pod reuses.
    if getattr(mcfg, "venv_on_volume", False):
        vol_id, mount = volume_for(mcfg, _account_hint())
        if not vol_id:
            raise RuntimeError(
                "RunPod: `venv_on_volume` needs a network volume — the venv has "
                "nowhere to live otherwise. Set network_volume_id (account-wide "
                "or on this pod block), or turn venv_on_volume off.")
        profile = (capability or model_capability(entry or {}, include_text=True)
                   or "text")
        plan["image"] = mcfg.slim_image or SLIM_POD_IMAGE
        plan["entrypoint"] = ["/bin/sh", "-c"]
        plan["start_cmd"] = [venv_bootstrap_script(
            mount, profile, mcfg.venv_name, mcfg.port or 8000)]
        print(f"[runpod] pod will run from a venv on the volume "
              f"({mount}/venvs/{mcfg.venv_name or profile}) with image "
              f"{plan['image']}", flush=True)
    return plan


def _account_hint():
    """The active RunPod account config, for defaults the plan needs."""
    try:
        from codai.models.manager import get_active_runpod_config
        return get_active_runpod_config()
    except Exception:
        return None


def seed_served_name(entry: dict) -> str:
    """The name a pod must answer to for a seed entry that is not the pool's own.

    The pool's OWN model is seeded with the name the pool serves it under; the
    others were seeded with nothing, so `seed_model_env` set no alias and the pod
    registered them under their `path`. For a LoRA that is fatal rather than
    cosmetic: an adapter's path IS its base, so the second name over one set of
    weights collapsed into the first and the pod answered 404 under it for the
    rest of its paid life. Which of the two broke depended on which model
    happened to rent the pod.
    """
    block = entry.get("runpod") if isinstance(entry.get("runpod"), dict) else {}
    return (str(block.get("served_model") or "").strip()
            or str(entry.get("id") or "").strip()
            or str(entry.get("alias") or "").strip())


def seed_model_env(entry: dict, served: str = "", source: str = "",
                   volume_path: str = "", engine: str = "") -> str:
    """The model entry to register on a fresh pod, as JSON for CODERAI_SEED_MODELS.

    A pod starts with an EMPTY catalogue: it would refuse a request for a model
    it has never heard of. So a pod rented for one model is told about that model
    at launch, and pulls the weights from HF on first use. Only the fields that
    describe the model travel — never local paths, which mean nothing there, and
    never this deployment's placement settings, which would make the pod try to
    rent pods of its own.
    """
    import json as _json
    if not isinstance(entry, dict) or not entry:
        return ""
    path = str(entry.get("path") or "").strip()
    if source:
        # Give the pod something it can actually fetch: a repo id it pulls from
        # HuggingFace, or a URL it downloads (coderai's loader takes both).
        path = source
    elif volume_path:
        path = volume_path
    elif (engine or engine_pod_backend(entry)) and path:
        # An engine model named by a local file: the basename travels. ds4
        # resolves it against the pod's GGUF cache — where its own download
        # lands, or where a volume already holds it — and colibri/k3 say
        # plainly that they never download when nothing is there.
        path = path.rsplit("/", 1)[-1]
    elif not path or path.startswith("/"):
        # A local path cannot be resolved on a rented machine, and nothing said
        # where else to get it.
        return ""
    keep = ("path", "model_type", "model_types", "video_subtypes", "capabilities",
            "alias", "config_name", "load_in_4bit", "load_in_8bit", "n_ctx",
            # The native engine a model is pinned to, and its per-model tuning
            # block — what makes the pod run ds4-server for it rather than try
            # to load a 154 GB GGUF with transformers. Other backend pins are
            # stripped below: vulkan/opencl say something about THIS machine.
            "backend", "ds4", "colibri", "k3",
            # …and the vLLM block, for the same reason: tool-call syntax,
            # context and memory shape describe the MODEL, and a pod that does
            # not get them serves it differently from this install.
            "vllm",
            "flash_attention", "model_template", "acceleration", "component_quantization",
            "languages", "supports_translation", "parser", "max_instances",
            # LoRA settings travel too: a coderai pod applies them itself, and
            # load_in_4bit above is what makes a QLoRA adapter load against the
            # base it was trained on.
            "lora_path", "lora_model_dir", "lora_scale", "loras",
            # Which MusicGen implementation to use. Describes the model, not
            # this deployment's placement, so it travels.
            "audio_backend",
            # NOT `host`, `service_url` or `service_token`: those say where a
            # model runs from HERE. On the far side they would make a pod try
            # to forward to itself, or to a machine it cannot reach.
            )
    out = {k: entry[k] for k in keep if k in entry and entry[k] is not None}
    out["path"] = path
    if not out.get("model_type"):
        # WITHOUT THIS the pod registers it as a text model. `model_type` is in
        # `keep` above, but a models.json entry does not carry one -- the kind
        # is the section it sits under, and only the entry travels. The pod's
        # seeder (ConfigManager._seed_models) reads `model_type` and falls back
        # to "text_models", so an embeddings pod answered
        #   "Model 'BAAI/bge-m3' is a text model and cannot be used for
        #    embedding generation. Use the text endpoint instead."
        # after a clean boot and a correct /v1/models -- which reads as the
        # caller's mistake rather than a seeding bug.
        _mt = entry_model_type(entry)
        if _mt:
            out["model_type"] = _mt
    if engine:
        # The pod runs this native engine for the model, whichever way it was
        # said here (the runpod block's engine, or the model's own pin).
        out["backend"] = engine
    elif "backend" in out and not engine_pod_backend(out):
        # runpod, host, vulkan, nvidia…: placement and hardware of THIS install.
        # On the pod they would mean "rent a pod" or "use a card it lacks".
        out.pop("backend", None)
    if volume_path:
        # The weights already sit on the network volume, put there once by
        # hand or by an earlier pod. The pod's engine resolves this path the
        # way it resolves a local one here: a directory for colibri/k3, a
        # .gguf for ds4. No download, no size check, no surprise.
        out["path"] = volume_path
    # Adapters the pod cannot resolve by name are named by CONTENT id instead.
    # The hash is computed here, so the pod is told the id when it is created and
    # sent the bytes before the first request — the two agree without the pod
    # having to exist yet. codai/api/remote_gateway.ensure_text_loras does the
    # sending.
    try:
        from codai.models.text_loras import configured_specs, is_portable, blob_id, local_path
        specs = configured_specs(entry)
        if specs:
            rewritten = []
            for spec in specs:
                src = str(spec.get("source") or "")
                if is_portable(src) or src.startswith("sha256:"):
                    # Resolved against the CONTENT: a catalogue can hold an id
                    # that nothing else computes (see text_loras.canonical_id),
                    # and a pod told that id looks it up, misses, and serves
                    # the base under the adapter's name.
                    from codai.models.text_loras import canonical_id
                    rewritten.append({"path": canonical_id(src),
                                      "weight": spec.get("weight", 1.0),
                                      "name": spec.get("name")})
                    continue
                local = local_path(src)
                # exists(), not isfile(): a PEFT adapter is a DIRECTORY, and
                # isfile() dropped it from the seed without a word. The pod then
                # launched with no --enable-lora and answered under the adapter's
                # name with BASE weights — untuned, and nothing failed.
                bid = blob_id(local) if local and os.path.exists(local) else ""
                if bid:
                    rewritten.append({"path": bid, "weight": spec.get("weight", 1.0),
                                      "name": spec.get("name")})
            for key in ("lora_path", "lora_model_dir", "lora_scale"):
                out.pop(key, None)
            if rewritten:
                out["loras"] = rewritten
            else:
                out.pop("loras", None)
    except Exception as exc:
        print(f"[seed] could not prepare adapters: {exc}", flush=True)
    if served and served != path:
        out["alias"] = served
    return _json.dumps([out])


def _llamacpp_docker_args(mcfg: "RunpodModelConfig", served: str,
                          entry: dict = None) -> str:
    """Args for the llama.cpp server image so it downloads and serves the GGUF.

    llama.cpp pulls weights itself with ``-hf user/repo:QUANT``; a local path is
    meaningless on a rented machine, so ``hf_gguf`` (or an HF-shaped
    ``served_model``) is required.
    """
    args = ["--host", "0.0.0.0", "--port", str(mcfg.port or 8000),
            "--alias", served or "model", "-ngl", "999"]
    if max(1, int(getattr(mcfg, "gpu_count", 1) or 1)) > 1:
        args += ["--split-mode", "layer"]      # spread layers across the pod's cards
    src = (mcfg.hf_gguf or "").strip()
    if src:
        args += ["-hf", src]
    else:
        kind, value = resolve_model_source(entry or {}, mcfg)
        if kind == "hf" and value:
            args += ["-hf", value]
        elif kind == "url" and value:
            # llama-server downloads the file itself; the usual way to serve a
            # one-off GGUF that lives on a plain HTTP host rather than on HF.
            args += ["-mu", value]
        elif kind == "upload":
            raise RuntimeError(
                "RunPod llama.cpp pod: `source: upload` cannot work here — the "
                "server is launched with the model and needs the file before the "
                "container starts. Use `hf_gguf`/`hf_repo`, or `model_url`, or run "
                "this model on a coderai pod (engine: coderai), which can be given "
                "the weights after it boots.")
        else:
            raise RuntimeError(
                "RunPod llama.cpp pod: nothing tells the pod where to get the "
                "weights. Set `hf_gguf` (\"user/repo:Q4_K_M\"), or `model_url` for a "
                "direct download. A local .gguf path does not exist on a rented pod.")
    if mcfg.ctx and mcfg.ctx > 0:
        args += ["-c", str(mcfg.ctx)]
    return _with_extra_args(args, mcfg)


def _with_extra_args(args: list, mcfg: "RunpodModelConfig") -> str:
    """The generated flags, then whatever the operator added on the model entry.

    ``extra_args`` has been accepted on a runpod block (and whitelisted per engine)
    since pods shipped, and until now went nowhere: it was parsed, stored and never
    put on a command line. Setting it looked like tuning and did nothing — which is
    worse than refusing it, because the pod boots, serves, and quietly ignores the
    decision. The two OCR VLMs are the case that noticed: a page-reading model wants
    prefix caching (the prompt repeats on every page) and the Qwen-VL pixel bounds,
    and neither could be expressed.

    Appended verbatim and LAST, so an explicit flag overrides a generated one the way
    it does on any command line. ``docker_args`` still replaces the whole line for
    anyone who wants nothing of ours."""
    extra = (getattr(mcfg, "extra_args", "") or "").strip()
    return " ".join(args) + (" " + extra if extra else "")


def _as_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return default


def _as_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


_DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _parse_name_list(value) -> list:
    """An ordered list of GPU names from a list or a comma string.

    Comma only, not whitespace: a GPU display name has spaces in it ("NVIDIA
    RTX A6000"), so splitting on those would turn one card into three.
    """
    if value in (None, ""):
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    out = []
    for raw in items:
        tok = str(raw).strip()
        if tok and tok not in out:
            out.append(tok)
    return out


def _gpu_matches(gpu: dict, want: str) -> bool:
    """Whether a catalogue entry is the card `want` names.

    Exact on the id or the display name, case-insensitively, and tolerant of a
    missing "NVIDIA " prefix so "A40" finds "NVIDIA A40". NOT a substring match:
    "A40" would otherwise also match "NVIDIA RTX A4000", which is a 16 GB card
    where someone asked for a 48 GB one.
    """
    w = (want or "").strip().lower()
    if not w:
        return False
    for field_ in (gpu.get("id"), gpu.get("display_name")):
        name = str(field_ or "").strip().lower()
        if not name:
            continue
        if name == w or name.removeprefix("nvidia ") == w.removeprefix("nvidia "):
            return True
    return False


def _parse_volume_map(value) -> dict:
    """{REGION: volume_id} from a dict or a "REGION=id,REGION=id" string.

    Region ids are upper-cased (RunPod's own spelling) so a map written in
    lower case still matches the region a candidate was ranked for — the whole
    feature turns on those two strings being equal. Blank ids are dropped
    rather than kept as a region with no storage, which is the one outcome this
    map exists to make impossible.
    """
    if value in (None, ""):
        return {}
    items = []
    if isinstance(value, dict):
        items = list(value.items())
    else:
        for part in str(value).replace(";", ",").split(","):
            if "=" in part:
                k, _, v = part.partition("=")
                items.append((k, v))
    out = {}
    for k, v in items:
        key = str(k or "").strip().upper()
        vol = str(v or "").strip()
        if key and vol:
            out[key] = vol
    return out


def _parse_dc_list(value) -> list:
    """A region allow-list from a list or a comma/space string, order kept.

    Order is the fallback order, so it is preserved rather than sorted: put the
    region you prefer first and the scaler only moves on when it cannot get a
    card there.
    """
    if value in (None, ""):
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).replace(",", " ").split()
    out = []
    for raw in items:
        tok = str(raw).strip().upper()
        if tok and tok not in out:
            out.append(tok)
    return out


def _parse_days(value) -> list:
    """Normalise a weekday selection to a sorted list of 0=Mon … 6=Sun.

    Accepts a list of ints or names, or a comma string of either ("mon,tue" or
    "0,1"). Anything unrecognised is dropped rather than guessed at. An empty
    result means "every day" to every caller.
    """
    if value in (None, ""):
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    out = set()
    for raw in items:
        tok = str(raw).strip().lower()
        if not tok:
            continue
        if tok.isdigit():
            n = int(tok)
            if 0 <= n <= 6:
                out.add(n)
            continue
        for i, name in enumerate(_DAY_NAMES):
            if tok.startswith(name):
                out.add(i)
                break
    return sorted(out)


def _parse_hhmm(value) -> str:
    """Normalise "8:5", "08:05", "0805" to "08:05"; "" when it is not a time."""
    tok = str(value or "").strip()
    if not tok:
        return ""
    if ":" not in tok and tok.isdigit() and len(tok) in (3, 4):
        tok = tok.zfill(4)[:-2] + ":" + tok.zfill(4)[-2:]
    parts = tok.split(":")
    if len(parts) != 2:
        return ""
    try:
        h, m = int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        return ""
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return ""
    return f"{h:02d}:{m:02d}"


def _schedule_zone(name: str):
    """The tzinfo for a schedule, or None for the server's local time."""
    if not name:
        return None
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        # An unknown zone must not silently move the window to UTC: fall back to
        # local time, which is what a blank setting means.
        return None


def schedule_state(cfg: "RunpodModelConfig", now: float = None) -> dict:
    """Where a pool's warm-pod schedule stands right now.

    Returns ``{enabled, in_window, next_change, window, days, tz}``; ``in_window``
    is True whenever the schedule is off, so every caller can treat it as "may I
    keep a pod warm?" without special-casing.
    """
    import datetime as _dt
    state = {"enabled": bool(getattr(cfg, "schedule_enabled", False)),
             "in_window": True, "next_change": None,
             "window": "", "days": list(getattr(cfg, "schedule_days", []) or []),
             "tz": getattr(cfg, "schedule_tz", "") or ""}
    if not state["enabled"]:
        return state
    start, end = cfg.schedule_start, cfg.schedule_end
    state["window"] = f"{start}-{end}"
    zone = _schedule_zone(cfg.schedule_tz)
    ts = time.time() if now is None else now
    local = _dt.datetime.fromtimestamp(ts, zone) if zone else _dt.datetime.fromtimestamp(ts)
    sh, sm = (int(x) for x in start.split(":"))
    eh, em = (int(x) for x in end.split(":"))
    s_min, e_min = sh * 60 + sm, eh * 60 + em
    cur = local.hour * 60 + local.minute
    days = state["days"]
    overnight = e_min <= s_min

    def _day_allowed(d):
        return (not days) or (d in days)

    # The day a window is attributed to is the day it STARTS on, so an overnight
    # Fri 22:00-06:00 belongs to Friday and runs into Saturday morning.
    if overnight:
        inside = ((cur >= s_min and _day_allowed(local.weekday()))
                  or (cur < e_min and _day_allowed((local.weekday() - 1) % 7)))
    else:
        inside = s_min <= cur < e_min and _day_allowed(local.weekday())
    state["in_window"] = inside

    # Next boundary: scan minute boundaries forward a week at most. Cheap, and
    # immune to the DST and overnight edge cases a closed form gets wrong.
    step = _dt.timedelta(minutes=1)
    probe = local.replace(second=0, microsecond=0)
    for _ in range(8 * 24 * 60):
        probe = probe + step
        p_cur = probe.hour * 60 + probe.minute
        if overnight:
            p_in = ((p_cur >= s_min and _day_allowed(probe.weekday()))
                    or (p_cur < e_min and _day_allowed((probe.weekday() - 1) % 7)))
        else:
            p_in = s_min <= p_cur < e_min and _day_allowed(probe.weekday())
        if p_in != inside:
            state["next_change"] = probe.isoformat(timespec="minutes")
            break
    return state


def parse_model_runpod(block: Optional[dict]) -> RunpodModelConfig:
    """Build a RunpodModelConfig from a model entry's ``runpod`` dict (or {})."""
    b = block or {}
    cloud = b.get("cloud_types")
    if isinstance(cloud, str):
        cloud = [c.strip().upper() for c in cloud.split(",") if c.strip()]
    elif isinstance(cloud, list):
        cloud = [str(c).strip().upper() for c in cloud if str(c).strip()]
    else:
        cloud = None
    env = b.get("env")
    if not isinstance(env, dict):
        env = {}
    cfg = RunpodModelConfig()
    cfg.mode = (b.get("mode") or cfg.mode).strip().lower()
    if cloud:
        cfg.cloud_types = cloud
    cfg.selection_criteria = (b.get("selection_criteria") or cfg.selection_criteria).strip().lower()
    cfg.gpu_type = (b.get("gpu_type") or "").strip()
    cfg.min_vram_gb = _as_float(b.get("min_vram_gb"), cfg.min_vram_gb)
    cfg.max_hourly_usd = _as_float(b.get("max_hourly_usd"), cfg.max_hourly_usd)
    cfg.gpu_count = max(1, min(8, _as_int(b.get("gpu_count"), cfg.gpu_count)))
    cfg.allow_spot = _as_bool(b.get("allow_spot"), cfg.allow_spot)
    cfg.image = (b.get("image") or "").strip()
    cfg.engine = (b.get("engine") or cfg.engine).strip().lower() or "auto"
    cfg.health_path = (b.get("health_path") or "").strip()
    cfg.docker_args = (b.get("docker_args") or "").strip()
    cfg.registry_auth_id = (b.get("registry_auth_id") or "").strip()
    cfg.network_volume_id = (b.get("network_volume_id") or "").strip()
    cfg.global_volume_id = (b.get("global_volume_id") or "").strip()
    cfg.data_centers = _parse_dc_list(b.get("data_centers"))
    cfg.gpu_types = _parse_name_list(b.get("gpu_types"))
    cfg.allow_other_gpus = _as_bool(b.get("allow_other_gpus"), cfg.allow_other_gpus)
    cfg.network_volumes = _parse_volume_map(b.get("network_volumes"))
    if cfg.network_volume_id or cfg.network_volumes:
        # RunPod offers network volumes on Secure Cloud only; leaving COMMUNITY
        # in the list just produces candidates the attach would reject.
        cfg.cloud_types = ["SECURE"]
    cfg.volume_mount_path = (b.get("volume_mount_path") or "").strip()
    cfg.volume_path = (b.get("volume_path") or "").strip()
    _dt = b.get("direct_tcp")
    cfg.direct_tcp = None if _dt in (None, "", "auto") else _as_bool(_dt, False)
    cfg.venv_on_volume = _as_bool(b.get("venv_on_volume"), cfg.venv_on_volume)
    cfg.venv_name = (b.get("venv_name") or "").strip()
    cfg.slim_image = (b.get("slim_image") or "").strip()
    cfg.pool = (b.get("pool") or "").strip().lower()
    cfg.quantization = (b.get("quantization") or "").strip()
    cfg.source = (b.get("source") or cfg.source).strip().lower() or "auto"
    cfg.hf_repo = (b.get("hf_repo") or "").strip()
    cfg.model_url = (b.get("model_url") or "").strip()
    cfg.model_url_is_tar = _as_bool(b.get("model_url_is_tar"), cfg.model_url_is_tar)
    cfg.api_key = (b.get("api_key") or "").strip()
    cfg.allow_open_pod = _as_bool(b.get("allow_open_pod"), cfg.allow_open_pod)
    cfg.served_model = (b.get("served_model") or "").strip()
    cfg.hf_gguf = (b.get("hf_gguf") or "").strip()
    cfg.container_disk_gb = _as_int(b.get("container_disk_gb"), cfg.container_disk_gb)
    cfg.volume_gb = _as_int(b.get("volume_gb"), cfg.volume_gb)
    cfg.port = _as_int(b.get("port"), cfg.port) or 8000
    cfg.ctx = _as_int(b.get("ctx"), cfg.ctx)
    cfg.env = {str(k): str(v) for k, v in env.items()}
    cfg.min_pods = max(0, _as_int(b.get("min_pods"), cfg.min_pods))
    cfg.keep_warm = _as_bool(b.get("keep_warm"), cfg.keep_warm)
    if cfg.keep_warm:
        # "Always warm" IS min_pods >= 1; keep one knob authoritative rather than
        # two that can disagree.
        cfg.min_pods = max(1, cfg.min_pods)
    cfg.max_pods = max(1, _as_int(b.get("max_pods"), cfg.max_pods))
    cfg.allow_client_scale = _as_bool(b.get("allow_client_scale"), cfg.allow_client_scale)
    # A zero/absent TTL would mean "never expires", which is the one thing a
    # lease must not mean. Keep the default rather than honour it.
    cfg.client_scale_ttl_s = max(60, _as_int(b.get("client_scale_ttl_s"),
                                             cfg.client_scale_ttl_s) or 1800)
    cfg.scale_up_inflight_per_pod = max(1, _as_int(b.get("scale_up_inflight_per_pod"),
                                                   cfg.scale_up_inflight_per_pod))
    cfg.scale_up_max_parallel_boots = max(1, _as_int(
        b.get("scale_up_max_parallel_boots"), cfg.scale_up_max_parallel_boots))
    # 0 disables shedding entirely (pods then go only when idle_timeout_s says so).
    cfg.scale_down_after_s = max(0, _as_int(b.get("scale_down_after_s"),
                                            cfg.scale_down_after_s))
    cfg.warmup_on_boot = _as_bool(b.get("warmup_on_boot"), cfg.warmup_on_boot)
    cfg.warmup_timeout_s = max(0, _as_int(b.get("warmup_timeout_s"),
                                          cfg.warmup_timeout_s))
    cfg.scale_down_costliest_first = _as_bool(b.get("scale_down_costliest_first"),
                                              cfg.scale_down_costliest_first)
    cfg.max_inflight_per_pod = max(0, _as_int(b.get("max_inflight_per_pod"),
                                              cfg.max_inflight_per_pod))
    cfg.sticky_sessions = _as_bool(b.get("sticky_sessions"), cfg.sticky_sessions)
    cfg.idle_timeout_s = max(0, _as_int(b.get("idle_timeout_s"), cfg.idle_timeout_s))
    cfg.boot_timeout_s = max(30, _as_int(b.get("boot_timeout_s"), cfg.boot_timeout_s))
    cfg.load_timeout_s = max(30, _as_int(b.get("load_timeout_s"), cfg.load_timeout_s))
    cfg.endpoint_id = (b.get("endpoint_id") or "").strip()
    cfg.min_workers = max(0, _as_int(b.get("min_workers"), cfg.min_workers))
    cfg.max_workers = max(1, _as_int(b.get("max_workers"), cfg.max_workers))
    cfg.cost_limit_usd = _as_float(b.get("cost_limit_usd"), cfg.cost_limit_usd)
    cfg.cost_period = (b.get("cost_period") or cfg.cost_period).strip().lower()
    cfg.data_center = (b.get("data_center") or "").strip().upper()
    cfg.pod_max_parallel_requests = max(0, _as_int(b.get("pod_max_parallel_requests"),
                                                   cfg.pod_max_parallel_requests))
    _gl = b.get("gpu_limits")
    if isinstance(_gl, dict):
        out = {}
        for card, lim in _gl.items():
            if not str(card).strip() or not isinstance(lim, dict):
                continue
            row = {}
            for key in ("parallel", "queue", "scale_up", "inflight"):
                try:
                    v = int(lim.get(key))
                except (TypeError, ValueError):
                    continue
                if v >= 0:
                    row[key] = v
            if row:
                out[str(card).strip()] = row
        cfg.gpu_limits = out
    _gw = b.get("gpu_weights")
    if isinstance(_gw, dict):
        out = {}
        for k, v in _gw.items():
            try:
                w = float(v)
            except (TypeError, ValueError):
                continue
            if w > 0 and str(k).strip():
                out[str(k).strip()] = w
        cfg.gpu_weights = out
    cfg.pod_queue_max_size = max(0, _as_int(b.get("pod_queue_max_size"),
                                            cfg.pod_queue_max_size))
    cfg.schedule_enabled = _as_bool(b.get("schedule_enabled"), cfg.schedule_enabled)
    cfg.schedule_days = _parse_days(b.get("schedule_days"))
    cfg.schedule_start = _parse_hhmm(b.get("schedule_start"))
    cfg.schedule_end = _parse_hhmm(b.get("schedule_end"))
    cfg.schedule_tz = (b.get("schedule_tz") or "").strip()
    # A schedule with no usable window would silently pin the floor to 0 forever;
    # treat it as "no schedule" instead, so the model keeps the behaviour it had.
    if cfg.schedule_enabled and not (cfg.schedule_start and cfg.schedule_end):
        cfg.schedule_enabled = False
    return cfg


def _model_entry(model_name: str) -> dict:
    """The model's models.json entry, or {}."""
    try:
        from codai.models.manager import _model_entry_for
        return _model_entry_for(model_name) or {}
    except Exception:
        return {}


def _model_path_for(model_name: str) -> str:
    """The model's local path from models.json (used to spot a GGUF), or ''."""
    try:
        from codai.models.manager import _model_entry_for
        entry = _model_entry_for(model_name) or {}
        return str(entry.get("path") or "")
    except Exception:
        return ""


def model_runpod_block(model_name: str) -> dict:
    """Fetch the ``runpod`` block from the model's models.json entry, or {}."""
    try:
        from codai.models.manager import _model_entry_for
        entry = _model_entry_for(model_name)
        if entry and isinstance(entry.get("runpod"), dict):
            return entry["runpod"]
    except Exception:
        pass
    return {}


# --------------------------------------------------------------------------- #
# Serverless resolution (RunPod owns the lifecycle — we only build the URL)
# --------------------------------------------------------------------------- #
def serverless_base_url(account_cfg, mcfg: RunpodModelConfig) -> str:
    """Return the OpenAI base URL for a serverless endpoint, e.g.
    ``https://api.runpod.ai/v2/<endpoint_id>/openai/v1``."""
    eid = (mcfg.endpoint_id or "").strip()
    if not eid:
        raise RuntimeError(
            "RunPod serverless: no endpoint_id set on the model's runpod config. "
            "Create a serverless endpoint in RunPod (a vLLM worker) and paste its id.")
    base = (getattr(account_cfg, "serverless_base", "") or "https://api.runpod.ai/v2").rstrip("/")
    return f"{base}/{eid}/openai/v1"


def auth_headers(account_cfg) -> dict:
    """Authorization header for RunPod serverless (Bearer API key)."""
    key = (getattr(account_cfg, "api_key", "") or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


# --------------------------------------------------------------------------- #
# Pods mode — coderai-managed pool of remote GPU pods
# --------------------------------------------------------------------------- #
import contextlib
import threading
import time
import uuid
from dataclasses import dataclass as _dc_pod


def _pod_health_ok(url: str, timeout: float = 4.0, path: str = "/v1/models",
                   api_key: str = "") -> bool:
    """True when the server inside the pod answers its readiness path.

    The token matters here: a pod locked with an api-key answers 401 on /v1/models
    without it, which would look exactly like "never became ready".
    """
    import requests
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = pod_http.get(url.rstrip("/") + "/" + path.lstrip("/"), timeout=timeout,
                         headers=headers)
        return r.status_code == 200
    except Exception:
        return False


def narrow_gpu_choice(mcfg: "RunpodModelConfig", gpu_types=None,
                      allow_other_gpus=None) -> dict:
    """Apply a caller's GPU constraints to the operator's, and report the result.

    Returns ``{"gpu_types": [...], "allow_other_gpus": bool, "notes": [...]}``.

    The rule is one-directional and deliberately so: **a caller may narrow the
    operator's choices, never widen them.** Picking the card is normally the
    operator's business -- they set the preference, the cost ceiling and the VRAM
    floor -- but a caller legitimately knows things the operator cannot: that
    this batch must be comparable with yesterday's numbers, that a measurement is
    worthless on a different card, that it would rather wait than pay a
    fallback's price. Letting it say so costs nothing. Letting it widen would
    hand it the operator's budget.

    So:

    * a requested card that the operator did not list is DROPPED, not honoured,
      and said so in `notes`;
    * `allow_other_gpus` may be turned OFF by a caller, never ON;
    * `max_hourly_usd`, `min_vram_gb`, `cloud_types` and `max_pods` are the
      operator's alone and are not reachable from here at all.

    Narrowing to nothing is refused rather than silently ignored: a caller asking
    only for cards the operator forbids has made a mistake worth hearing about,
    and dropping to "anything" would be the opposite of what it asked.
    """
    configured = [str(g).strip() for g in (getattr(mcfg, "gpu_types", None) or [])
                  if str(g).strip()]
    allow_other = bool(getattr(mcfg, "allow_other_gpus", False))
    notes = []

    want = gpu_types
    if isinstance(want, str):
        want = [w.strip() for w in want.split(",") if w.strip()]
    want = [str(w).strip() for w in (want or []) if str(w).strip()]

    out_types = list(configured)
    if want:
        if configured:
            # Case-insensitively, keep the caller's order: it is a preference
            # order, and the caller's ranking is the point of asking.
            lower = {c.lower(): c for c in configured}
            kept, dropped = [], []
            for w in want:
                hit = lower.get(w.lower())
                (kept.append(hit) if hit else dropped.append(w))
            if dropped:
                notes.append(
                    f"ignored {dropped}: not in the operator's gpu_types "
                    f"{configured}")
            if not kept:
                raise ValueError(
                    f"none of the requested GPUs {want} is allowed here; this "
                    f"model's gpu_types is {configured}")
            out_types = kept
        else:
            # No operator preference means "any card within the limits", so a
            # caller naming one is strictly more restrictive.
            out_types = want

    out_allow = allow_other
    if allow_other_gpus is not None:
        if bool(allow_other_gpus):
            if not allow_other:
                notes.append(
                    "ignored allow_other_gpus=true: the operator set it false, "
                    "and a request may only narrow the choice")
        else:
            out_allow = False
    return {"gpu_types": out_types, "allow_other_gpus": out_allow, "notes": notes}


def _rank_gpus(client, mcfg: "RunpodModelConfig", account_cfg, narrow=None) -> list:
    """Ranked list of GPU options across the model's allowed pools per its selection
    criteria (best first). Each item: {gpu_type_id, display_name, memory_gb,
    cloud_type, price, is_spot, bid}. Raises RunpodError if nothing fits the
    constraints. _provision_one tries them in order so a capacity miss on the top
    pick falls through to the next."""
    from codai.api.runpod_client import RunpodError
    catalog = client.list_gpu_types()
    allowed = [c.upper() for c in (mcfg.cloud_types or ["SECURE"])] or ["SECURE"]
    ceiling = mcfg.max_hourly_usd or 0.0
    n = max(1, int(getattr(mcfg, "gpu_count", 1) or 1))
    # The catalogue prices ONE card. A pod with N of them costs N times that and
    # holds N times the VRAM, so both the ceiling and the VRAM floor are checked
    # against the pod, not the card — or a 4-GPU pod would pass a $1/hr ceiling
    # at $0.90 a card and bill $3.60.
    # Preference, not a filter. The old single `gpu_type` was a hard filter, so
    # "A40" meant "A40 or nothing" — and when RunPod had no A40 the request
    # failed instead of taking a card that fit just as well.
    prefer = ([p for p in (getattr(mcfg, "gpu_types", None) or []) if str(p).strip()]
              or ([mcfg.gpu_type] if mcfg.gpu_type.strip() else [])
              or ([getattr(account_cfg, "default_gpu_type", "")]
                  if str(getattr(account_cfg, "default_gpu_type", "") or "").strip()
                  else []))
    allow_other = bool(getattr(mcfg, "allow_other_gpus", False))
    # A caller's narrowed choice, if one is in force (see `narrow_gpu_choice`).
    # Applied here rather than on the config, so it cannot outlive the lease that
    # asked for it or leak into another caller's provisioning.
    if narrow:
        if narrow.get("gpu_types"):
            prefer = list(narrow["gpu_types"])
        if narrow.get("allow_other_gpus") is not None:
            allow_other = bool(narrow["allow_other_gpus"])

    cands = []   # (pref_rank, price, is_spot, cloud_type, gpu, on_demand_price)
    for g in catalog:
        rank = len(prefer)
        for idx_p, want in enumerate(prefer):
            if _gpu_matches(g, want):
                rank = idx_p
                break
        if prefer and rank == len(prefer) and not allow_other:
            continue
        mem = (g.get("memory_gb") or 0) * n
        if mcfg.min_vram_gb and mem < mcfg.min_vram_gb:
            continue
        for ct in allowed:
            on_demand = g.get("secure_price") if ct == "SECURE" else g.get("community_price")
            spot = g.get("spot_price")
            options = [(on_demand, False)]
            if mcfg.allow_spot and spot:
                options.append((spot, True))
            for price, is_spot in options:
                if price is None or price <= 0:
                    continue
                price = price * n
                if ceiling > 0 and price > ceiling:
                    continue
                cands.append((rank, price, is_spot, ct, g, (on_demand or price) * n))
    if not cands:
        raise RunpodError(
            f"RunPod: no GPU in pools {allowed} with ≥{mcfg.min_vram_gb}GB under "
            f"${ceiling}/hr (spot={'on' if mcfg.allow_spot else 'off'}"
            + (f", preferred={prefer}" if prefer else "")
            + (", allow_other_gpus=off — set it to let another card that fits "
               "serve the request" if prefer and not allow_other else "") + ").")

    if (mcfg.selection_criteria or "cheaper").lower() == "faster":
        # Reliability + throughput: SECURE before COMMUNITY, on-demand before spot,
        # more VRAM (throughput proxy) first, then cheapest.
        ct_rank = {"SECURE": 0, "COMMUNITY": 1}
        cands.sort(key=lambda c: (c[0], ct_rank.get(c[3], 9), c[2] is True,
                                  -(c[4].get("memory_gb") or 0), c[1]))
    else:  # cheaper
        cands.sort(key=lambda c: (c[0], c[1], c[2] is False))
    # A preferred card always sorts ahead of a fallback, whatever the criteria:
    # the point of naming one is to get it when it is there.
    return [{"gpu_type_id": g["id"],
             "display_name": (f"{n}x {g['display_name']}" if n > 1 else g["display_name"]),
             "memory_gb": (g.get("memory_gb") or 0) * n, "cloud_type": ct, "price": price,
             "is_spot": is_spot, "bid": on_demand if is_spot else 0.0, "gpu_count": n,
             "preferred": rank < len(prefer)}
            for (rank, price, is_spot, ct, g, on_demand) in cands]


def _select_gpu(client, mcfg: "RunpodModelConfig", account_cfg) -> dict:
    """The single best GPU option (top of _rank_gpus)."""
    return _rank_gpus(client, mcfg, account_cfg)[0]


class BootCancelled(Exception):
    """An anticipatory boot was released before it finished.

    Not a failure: the pod was wanted when the boot started and is not wanted
    now, so it is terminated and the caller carries on. Distinct from a boot
    that FAILED, because the two want opposite handling -- a failure retries on
    another machine, a cancellation must not.
    """


def card_limits(mcfg, card: str) -> dict:
    """``{parallel, queue, scale_up, inflight}`` for this card.

    The per-card table first, then the model-level numbers for anything it does
    not state. Matched the same loose way weights are, because RunPod spells a
    card two ways (catalogue id vs the display name a pod reports).
    """
    base = {"parallel": int(getattr(mcfg, "pod_max_parallel_requests", 0) or 0),
            "queue": int(getattr(mcfg, "pod_queue_max_size", 0) or 0),
            "scale_up": max(1, int(getattr(mcfg, "scale_up_inflight_per_pod", 1) or 1)),
            "inflight": int(getattr(mcfg, "max_inflight_per_pod", 0) or 0)}
    table = getattr(mcfg, "gpu_limits", None) or {}
    name = str(card or "").strip().lower()
    if not name or not table:
        return base
    for key, row in table.items():
        k = str(key).strip().lower()
        if k and (k == name or k in name or name in k) and isinstance(row, dict):
            out = dict(base)
            out.update({kk: int(vv) for kk, vv in row.items()
                        if kk in base and vv is not None})
            out["scale_up"] = max(1, int(out["scale_up"]))
            return out
    return base


def _weight_for(weights: dict, card: str):
    """The configured weight for this card name, or None.

    Matched loosely in both directions because the two sources spell cards
    differently: RunPod's catalogue ids ("NVIDIA A100-SXM4-80GB") and the
    display names a pod reports ("A100 SXM") are not the same string.
    """
    name = str(card or "").strip().lower()
    if not name or not weights:
        return None
    for key, w in weights.items():
        k = str(key).strip().lower()
        if not k:
            continue
        if k == name or k in name or name in k:
            try:
                return float(w)
            except (TypeError, ValueError):
                return None
    return None


def _adopted_card(account_cfg, pod_id: str) -> dict:
    """``{gpu, gpu_count, vram_gb}`` for a pod we are taking over.

    Two sources, cheapest first: the registry, which recorded the card when
    THIS deployment rented the pod, and the account, which knows regardless.
    Blanks when neither answers -- an unidentified card costs a weight of 1.0
    (see `pod_weight`), never a pod.
    """
    out = {}
    try:
        ent = registry_entry(pod_id) or {}
        gpu = str(ent.get("gpu") or "").strip()
        if gpu:
            out = {"gpu": gpu,
                   "gpu_count": int(ent.get("gpu_count") or 1),
                   "vram_gb": float(ent.get("vram_gb") or 0.0)}
    except Exception:
        out = {}
    if out.get("gpu") and out.get("vram_gb"):
        return out
    try:
        from codai.api.runpod_client import RunpodClient
        live = RunpodClient(account_cfg).get_pod(pod_id) or {}
        gpu = str(live.get("gpu") or "").strip()
        if gpu:
            return {"gpu": gpu,
                    "gpu_count": int(live.get("gpu_count") or 1),
                    "vram_gb": float(live.get("vram_gb") or 0.0)}
    except Exception as exc:
        print(f"[runpod] adopting {pod_id}: could not read its card ({exc}); "
              f"it will carry the preferred card's share of the load",
              flush=True)
    return out or {"gpu": ""}


@_dc_pod
class PodHandle:
    pod_id: str
    url: str
    hourly_usd: float
    started_at: float
    gpu: str = ""
    is_spot: bool = False
    healthy: bool = True
    inflight: int = 0
    #: When each in-flight slot was taken, oldest first.
    #:
    #: `inflight` on its own is a ratchet: it is raised in `_pick` and lowered
    #: only in `release`, so any dispatch path that fails to release -- a client
    #: that disconnects, an exception on a branch nobody tested, a leak not yet
    #: found -- raises it permanently. Once it reaches max_inflight_per_pod the
    #: pod is locked out for the rest of its life while its card sits idle, and
    #: the pool reports it as "at capacity", which is the single most misleading
    #: thing this file can say.
    #:
    #: Observed 2026-10-18: pool said inflight=64 of 64 with 63 pages queued;
    #: the pod itself reported `Running: 31, Waiting: 0` at 10.6% KV cache. Half
    #: the slots did not exist. Reconciling against these timestamps lets the
    #: pool recover from leaks it does not know about, which is the only kind
    #: that matters.
    inflight_since: list = field(default_factory=list)
    last_used: float = 0.0
    #: Chosen for retirement and no longer given new work. It stays healthy and
    #: keeps serving what it already has: killing a pod with requests on it to
    #: save money would be paying for the work twice.
    draining: bool = False
    console_url: str = ""
    #: How far its /logs have been drained into our own file.
    log_cursor: int = 0
    #: Where this pod actually is and what it is. Reported to monitoring: the
    #: region matters for latency and for where the data is allowed to live, and
    #: "which model" is the first question anyone asks of a running fleet.
    data_center: str = ""
    cloud_type: str = ""
    gpu_count: int = 1
    #: Total VRAM of this pod (per-card x gpu_count), 0 when unknown. This is
    #: what makes a MIXED fleet work: the operator's concurrency and scale-up
    #: numbers describe the PREFERRED card, and a pod holding more or less
    #: memory than that one gets its share scaled in proportion. 0 means "treat
    #: it as the preferred card" -- never 0 capacity, which would take a
    #: perfectly good pod out of service for want of a number.
    vram_gb: float = 0.0


def _deploy_tag(account_cfg) -> str:
    """Stable, filesystem-safe deployment tag baked into pod names."""
    t = (getattr(account_cfg, "deployment_id", "") or "default").strip() or "default"
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in t)[:24]


# Pod ids currently mid-provision (created on RunPod but not yet in a pool). The
# reaper must NOT kill these. Guarded by _pools_lock.
_provisioning_ids: set = set()
# Live info for booting pods (shown on the stats page as "provisioning").
_provisioning_info: dict = {}


def _vllm_docker_args(mcfg: "RunpodModelConfig", served: str,
                      entry: dict = None) -> str:
    """Args appended to the vLLM OpenAI image entrypoint so it serves ``served``
    on the pod's port, reachable through the RunPod proxy.

    ``served`` must be something the POD can resolve — an HF repo id it will
    download. A local directory is as meaningless there as a local .gguf, and
    surfaces as a pod that boots for minutes and then dies, so it is refused up
    front instead.
    """
    name = (served or "").strip()
    if not _looks_like_repo_id(name):
        kind, value = resolve_model_source(entry or {}, mcfg)
        if kind == "hf" and value:
            name = value
        elif kind == "url" and value:
            # vLLM cannot download a URL itself, but the pod can before vLLM
            # starts: _add_staging fetches it and this points --model at the file.
            name = staged_model_dest(mcfg, value)
        elif kind == "upload":
            raise RuntimeError(
                "RunPod vLLM pod: `source: upload` has no receiver — the vLLM image "
                "has no endpoint to upload to, and its server needs the model at "
                "launch. Put the weights on any HTTP host the pod can reach and set "
                "`model_url` (the pod downloads them before starting), publish them "
                "to HuggingFace, or serve this model on a coderai pod.")
        else:
            raise RuntimeError(
                f"RunPod vLLM pod: {name or '(nothing)'} is not a HuggingFace repo id, "
                "and nothing else says where the pod should get the weights. Set "
                "`hf_repo` (e.g. \"Qwen/Qwen3.5-9B\") on the model's runpod block.")
    args = ["--host", "0.0.0.0", "--port", str(mcfg.port or 8000),
            "--model", name, "--served-model-name", name]
    n = max(1, int(getattr(mcfg, "gpu_count", 1) or 1))
    if n > 1:
        # A multi-GPU pod is only useful if the engine shards across the cards;
        # without this vLLM would load onto GPU 0 alone and the other N-1 cards
        # would be rented and idle.
        args += ["--tensor-parallel-size", str(n)]
    if mcfg.ctx and mcfg.ctx > 0:
        args += ["--max-model-len", str(mcfg.ctx)]
    if mcfg.quantization:
        args += ["--quantization", mcfg.quantization]
    args += _pod_lora_args(entry or {})
    return _with_extra_args(args, mcfg)


#: Where a staged download lands inside the pod. Container disk by default —
#: wiped with the pod, so size `container_disk_gb` for what you stage. With a
#: network volume attached this moves onto it (see staging_dir), and a staged
#: file then survives for every later pod.
STAGE_DIR = "/runpod-volume/staged"


def staging_dir(mcfg: "RunpodModelConfig" = None, account=None) -> str:
    """Where staged downloads land: the volume when there is one, else container disk."""
    try:
        vol_id, mount = volume_for(mcfg, account) if mcfg is not None else ("", "")
    except Exception:
        vol_id, mount = "", ""
    return f"{mount}/staged" if vol_id and mount else STAGE_DIR

#: Fetch with whatever the image actually has. vLLM's image ships python3 but not
#: always curl; llama.cpp's ships curl. Trying in order beats assuming.
_FETCH = ('if command -v curl >/dev/null 2>&1; then curl -fsSL "$U" -o "$F"; '
          'elif command -v wget >/dev/null 2>&1; then wget -qO "$F" "$U"; '
          'else python3 -c "import sys,urllib.request;'
          'urllib.request.urlretrieve(sys.argv[1],sys.argv[2])" "$U" "$F"; fi')


def stage_script(downloads: list, server_cmd: str) -> str:
    """A start command that fetches things, then execs the server.

    This is what lets a vLLM or llama.cpp pod serve weights that live neither on
    HuggingFace nor on the pod: they are downloaded from a URL you control — an
    object store, your own HTTPS host, anything the pod can reach — before the
    server starts. Uploading to the pod cannot work for these images because
    their server needs the file at launch; downloading happens *before* it.

    ``downloads`` is a list of {url, dest, tar} dicts. A tar is extracted into
    ``dest`` and removed, which is how a multi-file model travels.
    """
    lines = ["set -eu", f"mkdir -p {STAGE_DIR}"]
    for item in downloads:
        url = str(item.get("url") or "").strip()
        dest = str(item.get("dest") or "").strip()
        if not url or not dest:
            continue
        lines.append(f'U={_sh_quote(url)}')
        if item.get("tar"):
            lines.append(f'F={_sh_quote(dest + ".tar")}')
            lines.append(_FETCH)
            lines.append(f'mkdir -p {_sh_quote(dest)}')
            lines.append(f'tar -xf {_sh_quote(dest + ".tar")} -C {_sh_quote(dest)}')
            lines.append(f'rm -f {_sh_quote(dest + ".tar")}')
        else:
            lines.append(f'F={_sh_quote(dest)}')
            lines.append(f'mkdir -p "$(dirname {_sh_quote(dest)})"')
            lines.append(_FETCH)
        lines.append(f'echo "[stage] fetched {dest}"')
    lines.append("exec " + server_cmd)
    return "\n".join(lines)


def _sh_quote(s: str) -> str:
    return "'" + str(s).replace("'", "'\\''") + "'"


def staged_model_dest(mcfg: "RunpodModelConfig", url: str) -> str:
    """Where a staged model URL lands in the pod.

    One definition, used by the download step AND by the server args — computing
    it twice is how you get a pod that downloads to one path and serves another.
    A tar becomes a DIRECTORY of that name without the suffix.
    """
    name = str(url or "").rstrip("/").split("/")[-1] or "model"
    if getattr(mcfg, "model_url_is_tar", False):
        for suffix in (".tar.gz", ".tgz", ".tar"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
    return f"{STAGE_DIR}/{name}"


def staged_downloads(mcfg: "RunpodModelConfig", entry: dict, engine: str) -> list:
    """What this pod must fetch before its server can start.

    The model itself when it is a URL (a vLLM/llama.cpp server cannot download
    one), plus any adapter that has a URL rather than a repo id.
    """
    out = []
    kind, value = resolve_model_source(entry or {}, mcfg)
    if kind == "url" and value and engine == "vllm":
        # llama.cpp downloads a model URL itself with -mu; vLLM cannot.
        out.append({"url": value, "dest": staged_model_dest(mcfg, value),
                    "tar": bool(mcfg.model_url_is_tar)})
    for spec in _adapter_urls(entry or {}):
        out.append(spec)
    return out


def _adapter_urls(entry: dict) -> list:
    """Adapters given as a URL — staged so a vLLM pod can load them by path."""
    out = []
    try:
        from codai.models.text_loras import configured_specs
        specs = configured_specs(entry or {})
    except Exception:
        return out
    for spec in specs:
        src = str(spec.get("source") or "")
        if not src.startswith(("http://", "https://")):
            continue
        name = src.rstrip("/").split("/")[-1] or spec.get("name") or "adapter"
        out.append({"url": src, "dest": f"{STAGE_DIR}/loras/{name}",
                    "tar": name.endswith((".tar", ".tar.gz", ".tgz")),
                    "lora_name": spec.get("name"), "weight": spec.get("weight", 1.0)})
    return out


def _pod_lora_args(entry: dict) -> list:
    """vLLM `--lora-modules` for a POD, using only adapters the pod can resolve.

    A pod cannot read this machine's disk, so a local adapter path is no more
    usable there than a local model path. Only HuggingFace repo ids travel; a
    local adapter is reported and skipped rather than baked into a launch command
    that would fail minutes later, inside a pod, with a path nobody can inspect.
    """
    try:
        from codai.models.text_loras import configured_specs
        specs = configured_specs(entry or {})
    except Exception:
        return []
    usable, local = [], []
    for spec in specs:
        src = str(spec.get("source") or "")
        (usable if _looks_like_repo_id(src) else local).append(spec)
    if local:
        names = ", ".join(str(s.get("source")) for s in local)
        print(f"[lora] not sent to the pod (local paths — publish them to "
              f"HuggingFace, or serve this model on a coderai pod, which is sent "
              f"adapters with the request): {names}", flush=True)
    if not usable:
        return []
    args = ["--enable-lora", "--lora-modules"]
    args += [f"{s['name']}={s['source']}" for s in usable]
    args += ["--max-lora-rank", "64"]
    return args


class RunpodPodPool:
    """A pool of coderai-managed RunPod pods for one model.

    * ``acquire()`` returns a ready pod URL (provisioning one on demand, up to
      ``max_pods``, subject to the cost caps) and bumps its in-flight count.
    * ``release()`` drops the in-flight count and stamps last-used.
    * the shared scaler thread (``_scaler_loop``) tears down idle pods to
      ``min_pods`` after ``idle_timeout_s``, reconciles the pool against what
      RunPod actually has, and keeps ``min_pods`` warm.

    Note what the pool does NOT do: probe its own pods on a timer. Nothing here
    re-checks health (this docstring claimed it did for months, and it never
    has). A handle's ``healthy`` flag only goes false when something tried to
    USE the pod and failed, or when the pool itself retired it. So a pod that
    dies with no traffic on it — which is precisely a warm-floor pod — stays
    ``healthy=True`` forever and the floor is satisfied by a machine that no
    longer exists. ``reconcile_pools`` closes that, from the account listing the
    reaper already fetches: authoritative, free, and incapable of mistaking a
    busy pod for a dead one.
    """

    #: Class-level default so a pool built in a test without __init__ -- and
    #: any pickled or partially constructed one -- still has a cancel epoch to
    #: compare against. See `_boot_cancelled`.
    _cancel_epoch = 0

    #: Class-level so an instance built without __init__ (several tests, and
    #: any future path that skips it) reads 0 rather than raising. `_booting` is
    #: pods THIS process is provisioning right now; `_waiting` is acquire()
    #: callers blocked for a slot. Both are counts, because the grower has to
    #: know how MUCH demand there is, not merely whether the fleet is full.
    _booting = 0
    _waiting = 0

    def __init__(self, model_key, account_cfg, mcfg: "RunpodModelConfig", served_name,
                 entry: dict = None):
        #: The model's models.json entry, when this pool serves one specific
        #: model: it decides the pod image and is seeded into the pod.
        self.entry = entry if isinstance(entry, dict) else None
        #: Models a CAPABILITY pod should register at boot (it serves many).
        self.seed_entries: list = []
        #: Looked up once: the data center the network volume lives in.
        self._volume_dc = None
        self.model_key = model_key
        self.account = account_cfg
        self.mcfg = mcfg
        #: Set only for the duration of a volume seed, naming the one region
        #: being filled (see seed_region).
        self._seed_dc = ""
        #: The image this pool launches, learned from the first plan it builds.
        self._image = ""
        self.served = served_name
        self.pods: list = []
        #: affinity key -> pod_id, bounded. A conversation that comes back finds
        #: the pod whose prefix cache still holds its context; without this, a
        #: round-robin across pods re-prefills every turn from scratch.
        self._affinity = _OrderedDict()
        #: Bearer token this pool's pods are launched with. Generated when none is
        #: configured, so a pod is never left open on a public proxy URL.
        self.api_key = (getattr(mcfg, "api_key", "") or "").strip()
        if not self.api_key and not getattr(mcfg, "allow_open_pod", False):
            import secrets
            self.api_key = "cra-" + secrets.token_urlsafe(32)
        #: Set from the pod plan at provision time; the default suits an LLM pod.
        self.health_path = mcfg.health_path or "/v1/models"
        self._cv = threading.Condition(threading.RLock())
        #: Per-CARD arithmetic, memoised. `pod_weight`, `pod_capacity` and
        #: `pod_threshold` depend only on the card's NAME and the model config,
        #: but they were recomputed per pod on every acquire(): `_pick` once per
        #: free pod, `_pods_for_demand` once per serving pod, and
        #: `_preferred_vram_gb` iterating every pod again inside each of those
        #: -- all while holding `_cv`.
        #:
        #: At 70 pods that is ~140 recomputations per acquire, each scanning the
        #: gpu_limits/gpu_weights tables with substring matches. Measured on a
        #: live run: 811 pages waiting, 324 of 352 engine threads parked in
        #: futex_do_wait, /healthz answering in 1.6 ms while every lease
        #: serialised behind the pool lock and the scaler kept renting more pods
        #: -- each new pod making the per-acquire cost higher. The fleet grew
        #: because it could not serve, and could not serve because it had grown.
        #:
        #: Keyed by card name and a generation that bumps when the pod set or
        #: the config changes, because `_preferred_vram_gb` reads the pods.
        self._card_cache: dict = {}
        self._card_gen = 0
        #: When `acquire(grow_inline=False)` last turned pages away. This is the
        #: demand signal now that a waiting page no longer parks a thread in
        #: `acquire` -- `_waiting` only ever counted THREADS, and the
        #: non-blocking path has none to count.
        #:
        #: Counted over a window equal to the caller's RETRY interval, which
        #: makes the count the number of DISTINCT pages waiting rather than the
        #: attempts they have made between them: a page retrying every 2 s
        #: appears exactly once in any 2-second window. A longer window inflates
        #: demand by the retry rate, which is how an earlier build read its own
        #: retries as new work and rented 73 pods for a queue it never served.
        self._denied: list = []
        #: A warm floor a client leased through /v1/runpod/scale, and when the
        #: lease runs out. 0/0 = nobody asked.
        self._client_floor = 0
        self._client_floor_until = 0.0
        self._client_floor_logged = -1
        #: A CEILING the same lease may set, and when it lapses. 0/0 = nobody
        #: asked, and the configured `max_pods` is the only ceiling.
        #:
        #: The floor and the ceiling are not the same request. A client that
        #: says "hold 8 warm" has said nothing about how large the fleet may
        #: become, and the autoscaler is free to grow it to `max_pods` as long
        #: as work keeps arriving -- which is exactly what happened on
        #: 2026-10-18: a staged run asked for 8 pods, got 13, and was heading
        #: for 128, because the only number a request could set was the floor.
        #: Each pod beyond the plan added ~700 s of paid boot producing nothing,
        #: and the realised cost per page came out 16.5x the measured baseline.
        #:
        #: So a lease may ALSO narrow the ceiling, and like `gpu_types` it may
        #: only ever narrow it: the operator's `max_pods` remains the budget,
        #: and there is deliberately no way for a request to raise it.
        self._client_cap = 0
        self._client_cap_until = 0.0
        self._client_cap_logged = -1
        #: GPU constraints a client attached to its lease. They expire with it.
        #: Only ever NARROWER than the operator's own settings -- see
        #: `narrow_gpu_choice`.
        self._client_gpu_types: list = []
        self._client_allow_other = None
        #: Pods THIS process is provisioning right now. A count, not a flag: it
        #: used to be a boolean, and because a pod takes minutes to boot, the
        #: pool could grow by at most one pod per boot however much work was
        #: queued. Measured on 2026-10-13: 64 pages wanting 4 pods got 2, ready
        #: ten minutes apart, and the client reported "no real scale-out".
        self._booting = 0
        #: Bumped every time a client gives the fleet back. A boot that was
        #: started BEFORE the shed and lands after it is a pod nobody asked for
        #: any more -- and until this existed there was no way to say so: a
        #: booting pod is an integer here, not an object, so `shed_now` could
        #: not see it, could not count it and could not cancel it. A client that
        #: asked for one pod, got four on the way, and released them paid all
        #: four boots in full.
        self._cancel_epoch = 0
        #: acquire() callers blocked waiting for a slot. Without this the grower
        #: can see THAT the fleet is full (every pod at its ceiling) but not BY
        #: HOW MUCH, so it could only ever ask for one more pod.
        self._waiting = 0
        self._closed = False
        #: When the fleet first looked larger than the load needs. Scale-down
        #: waits out scale_down_after_s from here, so a momentary dip in a burst
        #: does not cost a pod that is wanted again seconds later.
        self._surplus_since = 0.0
        #: A client said "give them back" and something was still working, so the
        #: pods could not go yet. Without remembering it the intent was simply
        #: lost: the pool fell back to idle_timeout_s (15 minutes on the Digesta
        #: orchestrator) and billed for a fleet nobody had asked to keep.
        self._release_pending = False
        # One log line per window transition, not one every scaler tick.
        self._sched_closed_logged = False

    # -- cost ------------------------------------------------------------- #
    def live_cost_usd(self) -> float:
        """Cost accrued by pods still running (not yet in the ledger)."""
        now = time.time()
        with self._cv:
            return sum((now - p.started_at) / 3600.0 * p.hourly_usd for p in self.pods)

    def hourly_rate(self) -> float:
        with self._cv:
            return sum(p.hourly_usd for p in self.pods)

    def _budget_blocks(self, new_rate: float) -> str:
        """Return a reason string if provisioning a pod at ``new_rate`` $/hr would
        breach a rate or spend cap, else ''."""
        acct = self.account
        # Global instantaneous rate cap.
        gmax = float(getattr(acct, "global_max_hourly_usd", 0) or 0)
        if gmax > 0 and (_all_pools_hourly_rate() + new_rate) > gmax + 1e-9:
            return f"global ${gmax}/hr rate cap"
        # Per-model cumulative spend cap (rolling period).
        from codai.api import runpod_ledger
        m_lim = float(self.mcfg.cost_limit_usd or 0)
        m_period = self.mcfg.cost_period or "unlimited"
        if m_lim > 0 and m_period != "unlimited":
            spent = runpod_ledger.spend_in_period(self.model_key, m_period) + self.live_cost_usd()
            if spent >= m_lim:
                return f"model spend ${spent:.2f} ≥ ${m_lim} / {m_period}"
        # Global cumulative spend cap.
        g_lim = float(getattr(acct, "global_cost_limit_usd", 0) or 0)
        g_period = getattr(acct, "global_cost_period", "unlimited") or "unlimited"
        if g_lim > 0 and g_period != "unlimited":
            g_spent = runpod_ledger.global_spend_in_period(g_period) + _all_pools_live_cost()
            if g_spent >= g_lim:
                return f"global spend ${g_spent:.2f} ≥ ${g_lim} / {g_period}"
        return ""

    # -- provisioning ----------------------------------------------------- #
    def _log_pod_catalogue(self, url: str) -> None:
        """Log what the pod says it can serve, the moment it is up.

        The one check that would have turned a confusing "Model 'x' is not
        available. Use one of: " into an obvious "the pod registered nothing",
        minutes earlier and without reading a single line of our own code.
        """
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = pod_http.get(url.rstrip("/") + "/v1/models", headers=headers, timeout=20)
            if r.status_code != 200:
                print(f"[runpod] pod catalogue unreadable: HTTP {r.status_code} "
                      f"{(r.text or '')[:120]}", flush=True)
                return
            ids = [str(m.get("id")) for m in (r.json().get("data") or [])]
        except Exception as exc:
            print(f"[runpod] pod catalogue unreadable: {exc}", flush=True)
            return
        if ids:
            print(f"[runpod] pod serves {len(ids)} model(s): "
                  f"{', '.join(ids[:8])}{' …' if len(ids) > 8 else ''}", flush=True)
        else:
            print("[runpod] pod serves NO models — it will refuse every request. "
                  "Seeding did not take effect (check CODERAI_SEED_MODELS in the "
                  "pod's env and the pod's own startup log).", flush=True)

    def _data_center_for_volume(self, volume_id: str) -> str:
        """The data center a network volume lives in, or ''.

        A pod can only attach a volume in its own data center, so this pins
        placement instead of letting GPU ranking pick a region where the attach
        will simply fail.
        """
        try:
            from codai.api.runpod_client import RunpodClient
            for vol in RunpodClient(self.account).list_network_volumes():
                if str(vol.get("id") or "") == str(volume_id):
                    dc = str(vol.get("dataCenterId") or "")
                    if dc:
                        print(f"[runpod] volume {volume_id} is in {dc} — pinning pods "
                              "there", flush=True)
                    return dc
        except Exception as exc:
            print(f"[runpod] could not resolve the volume's data center: {exc}",
                  flush=True)
        return ""

    def _create_with_fallback(self, client, ranked, start=0):
        """Try ranked GPU candidates from ``start``; skip capacity misses. Returns
        (pod_id, sel, next_index). Raises if all fail."""
        from codai.api.runpod_client import RunpodError
        port = self.mcfg.port or 8000
        plan = pod_plan(self.mcfg, self.served, str(self.model_key),
                        _model_path_for(self.model_key), api_key=self.api_key,
                        entry=self.entry, seed_entries=self.seed_entries)
        image, args = plan["image"], plan["args"]
        #: What this pool launches. Recorded with every pod, because "the same
        #: model" is not "the same code" — a pinned tag beside :latest is the
        #: difference between a fix being live and only looking live.
        self._image = image
        self.health_path = plan["health_path"]
        # The plan's env, then the model's OWN env on top: an explicit
        # per-model value must beat one the plan resolved from the ambient
        # environment. The other order silently replaced a model's configured
        # HF_TOKEN with whatever this process happened to have.
        env = dict(plan.get("env") or {})
        env.update(self.mcfg.env or {})

        # A network volume, when configured: caches and upload targets move onto
        # it so weights survive the pod. RunPod requires Secure Cloud and the
        # pod's data center to match the volume's, so both are forced here rather
        # than surfacing later as an unexplained capacity error.
        vol_map = volume_map_for(self.mcfg, self.account)
        vol_id, vol_mount = volume_for(self.mcfg, self.account)
        if vol_id:
            env.update(volume_env(vol_mount))
            # With a per-region map the region is chosen FIRST and the volume
            # follows it (resolved per candidate below), so there is nothing to
            # pin — pinning is what a single volume forces, and the map exists
            # to stop paying for it.
            if not vol_map and getattr(self, "_volume_dc", None) is None:
                self._volume_dc = self._data_center_for_volume(vol_id)
        # A GLOBAL volume instead: same caches, no data-centre pinning. RunPod
        # mounts it at /workspace by itself when it is the only volume, which is
        # the mount the pod env already points at; with a network volume also
        # attached it moves to /workspace-global, so attaching both and
        # expecting /workspace to be the global one would be wrong.
        global_vol = global_volume_for(self.mcfg, self.account)
        if global_vol and not vol_id:
            env.update(volume_env(vol_mount or GLOBAL_VOLUME_MOUNT))
        # Direct TCP puts nothing between here and the pod, so the pod brings
        # its own TLS: a certificate from this install's CA, sent like the
        # token. Only a coderai image knows what to do with it — vLLM's and
        # llama.cpp's images stay plain HTTP on this path, and say so.
        self._tls = False
        self._direct = direct_tcp_for(self.mcfg, plan.get("engine"))
        if self._direct:
            # coderai pods only: boot.sh and the venv-on-volume bootstrap both
            # read the env; a custom image is anyone's guess and vLLM's and
            # llama.cpp's images ignore it, which would leave us speaking
            # https to a plain-HTTP server.
            if plan.get("engine") == "coderai":
                from codai.api.pod_tls import pod_env as _tls_env
                env.update(_tls_env(f"{_deploy_tag(self.account)}-{self.model_key}"))
                self._tls = True
            else:
                print(f"[runpod] direct_tcp on a {plan.get('engine')} image: plain "
                      f"HTTP — that image cannot take a certificate; the bearer "
                      f"token travels unencrypted", flush=True)
        last = None
        _skipped_cached = 0
        for i in range(start, len(ranked)):
            sel = ranked[i]
            # THE POD'S OWN GATE, for the card it is about to run on. The plan
            # above was built before a candidate was chosen, so its env carried
            # the model-level numbers whatever card won -- and the gate is env,
            # set once at launch, so a pod rented on a 3x card spent its whole
            # life admitting a 1x card's worth of work.
            _lim = card_limits(self.mcfg, str(sel.get("display_name") or ""))
            if _lim.get("parallel"):
                env["CODERAI_SERVER_MAX_PARALLEL_REQUESTS"] = str(int(_lim["parallel"]))
            if _lim.get("queue"):
                env["CODERAI_SERVER_QUEUE_MAX_SIZE"] = str(int(_lim["queue"]))
            block = self._budget_blocks(sel["price"])
            if block:
                raise RunpodError(f"RunPod budget cap hit — not provisioning ({block}).")
            # `pool:` is dropped from the NAME only — it is legal on RunPod
            # but makes a console listing hard to read, and ownership is
            # decided by the `coderai-<deployment>-` prefix, never by the rest.
            _key = str(self.model_key)
            if _key.startswith("pool:"):
                _key = _key[len("pool:"):]
            name = (f"coderai-{_deploy_tag(self.account)}-"
                    + _key[:20].replace("/", "_").replace(" ", "_")
                    + "-" + uuid.uuid4().hex[:6])
            _dc = sel.get("_dc") or self._data_center()
            # ALREADY REFUSED, recently, for somebody else? Then do not spend a
            # round trip and a GIL slice learning it again. This is the whole
            # saving: 32 boot threads each sweeping ~70 candidates produced 406
            # rejections in 120 s and saturated the core the event loop needed.
            if _is_unavailable(sel.get("gpu_type_id"), _dc):
                _skipped_cached += 1
                continue
            # Say so when this is not the card that was asked for. Renting a
            # different GPU than the configuration names is correct here — it is
            # what allow_other_gpus is for — but it must never be silent.
            _alt = "" if sel.get("preferred", True) else " [FALLBACK — no preferred card available]"
            if vol_map:
                _v = volume_for(self.mcfg, self.account, _dc)[0]
                print(f"[runpod] {_dc or '(no region)'}: attaching volume "
                      f"{_v or '(none — this region has no volume configured)'}",
                      flush=True)
            # ONE pod request walks this whole list until a (card, region) pair
            # has capacity, so this line is printed PER CANDIDATE -- and it used
            # to say "provisioning pod", which reads as a pod being launched.
            # 312 attempts for 10 pods in one afternoon therefore looked like
            # runaway provisioning to an operator reading the log, and was
            # reported as such. Say which attempt this is, out of how many, so
            # the shape of the list is visible instead of inferred.
            print(f"[runpod] {self.model_key!r}: trying candidate "
                  f"{i + 1}/{len(ranked)} for ONE pod — {sel['display_name']} "
                  f"({sel['cloud_type']}{'/spot' if sel['is_spot'] else ''}) ${sel['price']}/hr"
                  + (f" in {_dc}" if _dc else " (region: RunPod's choice)") + _alt,
                  flush=True)
            if _skipped_cached:
                print(f"[runpod] {self.model_key!r}: skipped {_skipped_cached} "
                      f"candidate(s) already known to have no capacity "
                      f"(cached {UNAVAIL_TTL_S:.0f}s)", flush=True)
                _skipped_cached = 0
            try:
                _disk_gb, _disk_note = disk_for(
                    _model_entry(self.served or self.model_key), self.mcfg, self.account)
                if _disk_note:
                    print(f"[runpod] {_disk_note}", flush=True)
                pod_id = client.create_pod(
                    name=name, image=image, gpu_type_id=sel["gpu_type_id"], port=port,
                    gpu_count=int(sel.get("gpu_count") or 1),
                    cloud_type=sel["cloud_type"], container_disk_gb=_disk_gb,
                    volume_gb=self.mcfg.volume_gb, env=env, docker_args=args,
                    is_spot=sel["is_spot"], bid_per_gpu=sel["bid"],
                    data_center_id=_dc,
                    network_volume_id=(volume_for(self.mcfg, self.account, _dc)[0]
                                       if vol_map else vol_id),
                    global_volume_id=global_vol,
                    volume_mount_path=(vol_mount or "/workspace"),
                    registry_auth_id=(self.mcfg.registry_auth_id
                                      or getattr(self.account, "registry_auth_id", "") or ""),
                    entrypoint=plan.get("entrypoint"), start_cmd=plan.get("start_cmd"),
                    direct_tcp=bool(getattr(self, "_direct", False)))
                _forget_unavailable(sel.get("gpu_type_id"), _dc)
                return pod_id, sel, i + 1
            except RunpodError as exc:
                msg = str(exc).lower()
                if "no longer any instances" in msg or "no instances" in msg or "capacity" in msg:
                    # Shared, so the other boot threads skip it instead of each
                    # discovering the same shortage. See `_unavail`.
                    _mark_unavailable(sel.get("gpu_type_id"), _dc)
                    print(f"[runpod] {sel['display_name']} ({sel['cloud_type']}) unavailable — "
                          "trying next candidate", flush=True)
                    last = exc
                    continue
                raise
        raise last or RunpodError("RunPod: no candidate GPU could be deployed (capacity).")

    # The port appears only AFTER the image is pulled, so this budget has to
    # cover the download — the original 300s assumed the container was already
    # there. Measured on the same 7.3 GB capability image: 133s on a machine
    # that had the layers cached, and repeated failures at 300s on cold ones,
    # which needs a sustained ~25 MB/s to beat. Each failure then rented a fresh
    # machine and paid for the whole pull again, three times over.
    #
    # 15 minutes is bounded and cheap to be wrong about (a quarter-hour of a
    # $0.25/hr card is 6 cents) where being too tight costs three pods and still
    # fails. Override per model with `boot_timeout_s`.
    PORT_TIMEOUT_S = 900.0
    HEALTH_TIMEOUT_S = 600.0

    def _pod_log_path(self, pod_id: str) -> str:
        """Where this pod's drained log lives, beside the pod registry.

        Kept after the pod is gone, which is the entire point: two of the three
        pods investigated tonight had already been destroyed by the time there
        was a question to ask.
        """
        base = os.path.dirname(_pod_registry_path())
        d = os.path.join(base, "pod-logs")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{pod_id}.log")

    def _drain_pod_log(self, pod: "PodHandle") -> None:
        """Copy whatever the pod has said since last time into our own file.

        Pull rather than push: the orchestrator can already reach a pod (that is
        how it serves requests), the auth and pinned TLS are already in place,
        and a pod that cannot reach back out still gets its log kept. Bounded at
        both ends — the pod keeps a ring, we keep a tail — because neither disk
        is free.
        """
        url = (pod.url or "").rstrip("/")
        if not url:
            return
        since = getattr(pod, "log_cursor", 0) or 0
        try:
            headers = ({"Authorization": f"Bearer {self.api_key}"}
                       if self.api_key else {})
            reply = pod_http.get(f"{url}/logs?since={since}&limit=2000",
                                 headers=headers, timeout=15)
            if reply.status_code != 200:
                return            # an older image has no /logs; nothing to do
            data = reply.json()
        except Exception:
            return                # unreachable is the health machinery's problem
        lines = data.get("lines") or []
        dropped = int(data.get("dropped") or 0)
        pod.log_cursor = int(data.get("cursor") or since)
        if not lines and not dropped:
            return
        path = self._pod_log_path(pod.pod_id)
        try:
            with open(path, "a") as fh:
                if dropped:
                    # Silence and "nothing happened" must not read the same.
                    fh.write(f"[drain] {dropped} line(s) rolled out of the pod's "
                             f"ring before they were read\n")
                for line in lines:
                    fh.write(line + "\n")
            if os.path.getsize(path) > _POD_LOG_MAX_BYTES:
                with open(path) as fh:
                    keep = fh.readlines()[-(_POD_LOG_KEEP_LINES):]
                with open(path, "w") as fh:
                    fh.writelines(keep)
        except OSError:
            pass

    def _dump_pod_logs(self, client, pod_id, why, url: str = ""):
        """Say what the pod was doing, from whichever source can still answer.

        RunPod publishes no pod-log API — the REST OpenAPI spec lists 23 routes
        and none of them serve logs, which is why every attempt came back 400.
        The console is a human surface. So a coderai pod keeps its own boot
        record and serves it at /boot: if the port ever opened, that is the real
        log, and it covers the container phases too (boot.sh writes them to the
        same file). Tried first, because it is the only one that ever works.
        """
        # Drain whatever it has said first, so the file we keep covers the
        # failure itself and not merely everything up to the last tick.
        try:
            for p in list(self.pods):
                if p.pod_id == pod_id:
                    self._drain_pod_log(p)
                    print(f"[runpod] pod {pod_id} log kept at "
                          f"{self._pod_log_path(pod_id)}", flush=True)
                    break
        except Exception:
            pass
        if url:
            try:
                import requests
                headers = ({"Authorization": f"Bearer {self.api_key}"}
                           if self.api_key else {})
                r = pod_http.get(url.rstrip("/") + "/boot", headers=headers, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    lines = list(data.get("container") or []) + list(data.get("phases") or [])
                    if lines:
                        print(f"[runpod] pod {pod_id} boot record ({why}):\n  "
                              + "\n  ".join(lines[-40:]), flush=True)
                        return
            except Exception:
                pass          # port never opened, or it died before answering
        try:
            info = client.get_pod_logs(pod_id, tail=120)
        except Exception as exc:
            info = {"error": str(exc)}
        logs = (info.get("logs") or "").strip()
        console = info.get("console_url") or ""
        if logs:
            tail = "\n".join(logs.splitlines()[-40:])
            print(f"[runpod] pod {pod_id} vLLM/container log ({why}):\n{tail}", flush=True)
        else:
            print(f"[runpod] pod {pod_id} logs unavailable via API ({info.get('error','')}); "
                  f"check the console: {console}", flush=True)

    # How many different machines to try before giving up. A pod that RunPod accepts
    # but never actually starts (stuck pending — seen in practice) is retried on the
    # NEXT candidate rather than failing the request.
    MAX_PROVISION_ATTEMPTS = 3

    def _pod_answers_us(self, url: str) -> bool:
        """Whether the pod accepts the token we would use for it.

        Probes an AUTHENTICATED path, because the unauthenticated one proves
        nothing: a coderai pod answers /healthz to anybody, which is how a pod
        that 401s every real request passed the probe that adopted it.
        """
        base = (url or "").rstrip("/")
        if not base:
            return False
        try:
            from codai.api import pod_http
            headers = ({"Authorization": f"Bearer {self.api_key}"}
                       if self.api_key else {})
            reply = pod_http.get(base + "/v1/models", headers=headers, timeout=20)
        except Exception as exc:
            # Unreachable is not "unauthorised": leave that to the health
            # machinery rather than destroying a pod over one failed probe.
            print(f"[runpod] token probe against {base} failed: {exc}", flush=True)
            return True
        return reply.status_code not in (401, 403)

    def _adopt_shared_pod(self) -> bool:
        """Take over a pod already running for this pool, instead of renting.

        Two cases, one mechanism: a sibling engine booted one for the same pool,
        or WE booted one before the last restart. The second is the expensive
        one — a warm floor is re-established at every boot, so an hourly
        unattended upgrade rented a fresh A40 each time and left the previous
        one billing until the reaper's 20-minute grace expired.
        """
        # What this pool would launch, and on what. Adoption is only sound when
        # the running pod matches both: the same model on a different image is
        # different code, and a pod rented as a fallback is a card this
        # configuration accepts only when its preferred one is unavailable.
        want_image, want_gpus = "", []
        try:
            plan = pod_plan(self.mcfg, self.served, str(self.model_key),
                            _model_path_for(self.model_key), api_key=self.api_key,
                            entry=self.entry, seed_entries=self.seed_entries)
            want_image = str(plan.get("image") or "")
        except Exception as exc:
            # Without the image this check cannot be made, and making it badly
            # would reject every candidate. Say so and fall back to the model
            # match alone.
            print(f"[runpod] cannot tell which image {self.model_key!r} would "
                  f"launch ({exc}) — adopting on the model match alone",
                  flush=True)
        try:
            narrow = narrow_gpu_choice(self.mcfg, **(self.client_gpu_narrow() or {}))
            want_gpus = list(narrow.get("gpu_types") or [])
        except Exception:
            want_gpus = list(getattr(self.mcfg, "gpu_types", []) or [])
        shared_id, shared_url, shared_key, shared_rate = find_shared_pod(
            str(self.model_key), self.health_path, self.api_key,
            image=want_image, gpus=want_gpus)
        if not shared_url:
            return False
        if shared_key and shared_key != self.api_key:
            # That pod only answers to the token it was launched with. Adopt the
            # token so this pool — and the pods it rents next — speak the same
            # one. Only safe while we hold no pod of our own; otherwise we end
            # up with a pool whose pods want two different tokens.
            with self._cv:
                if self.pods:
                    return False
            self.api_key = shared_key
            print(f"[runpod] adopting pod {shared_id}'s bearer token", flush=True)
        if not self._pod_answers_us(shared_url):
            # A coderai pod's /healthz needs no token, so the probe that found
            # this pod says nothing about whether we can USE it. With no key
            # recorded we fall back to our own — and if that is not the token the
            # pod was launched with, it answers 401 to every request for the rest
            # of its paid life. Seen today: a pod adopted after a restart 401'd
            # every OCR page.
            #
            # One registry per deployment means no recorded key is no key at all:
            # the token is lost and the pod can never serve anyone. Paying for it
            # is the only outcome worse than terminating it.
            print(f"[runpod] pod {shared_id} refuses our bearer token (its own "
                  f"was never recorded) — terminating rather than paying for a "
                  f"pod nobody can use", flush=True)
            try:
                from codai.api.runpod_client import RunpodClient as _RC
                unregister_pod(shared_id)
                _RC(self.account).terminate_pod(shared_id)
            except Exception as exc:
                print(f"[runpod] could not terminate {shared_id}: {exc}", flush=True)
            return False
        print(f"[runpod] reusing the pod already running for {self.model_key!r} "
              f"({shared_id}) instead of renting another", flush=True)
        handle = PodHandle(pod_id=shared_id, url=shared_url,
                           # Its recorded price when we know it: a pod we are
                           # inheriting after our own restart is ours to pay for,
                           # and reporting it free hides it from the $/hr cap.
                           hourly_usd=float(shared_rate or 0.0),
                           # When it was CREATED, not when we took it over. An
                           # adopted pod has been billing since it was rented,
                           # and the hours before the adoption were charged to
                           # nobody: a pod created at 22:27 and adopted at 23:10
                           # was ledgered for 16 minutes when it had run for an
                           # hour. Only cost and uptime reporting read this, so
                           # backdating it corrects the bill and nothing else.
                           started_at=float(
                               registry_entry(shared_id).get("created")
                               or time.time()),
                           # WHICH CARD, not "(adopted)". A pod whose card is
                           # unknown cannot be given a share of the load that
                           # matches it, cannot have its concurrency sized, and
                           # shows up in the fleet report as one anonymous
                           # kind -- which is how a whole day's status output
                           # read `by_gpu: {"(adopted)": 32}`. The registry
                           # knows when we rented it ourselves; the account
                           # knows otherwise.
                           **_adopted_card(self.account, shared_id),
                           healthy=True, last_used=time.time())
        self._add_pod(handle)
        register_pod(shared_id, str(self.model_key), shared_url, self.api_key,
                     hourly_usd=float(shared_rate or 0.0))  # image/gpu kept
        return True

    def _warmup(self, pod_id: str, url: str) -> bool:
        """Make the pod load its weights before anyone calls it ready.

        One token against the served model. Cheap, and it is the only request
        that can tell the difference between "the server is up" and "the model
        is resident" — /v1/models answers the moment the process starts.

        Returns whether it worked. A failure is NOT fatal: the pod may still
        serve (a model whose warm-up shape we guessed wrong, an engine that
        does not take chat completions), and refusing to use a pod we have
        already paid to boot over a diagnostic request would be worse than the
        hidden latency this exists to remove.
        """
        if not getattr(self.mcfg, "warmup_on_boot", True):
            return False
        budget = int(getattr(self.mcfg, "warmup_timeout_s", 900) or 0)
        if budget <= 0:
            return False
        t0 = time.time()
        name = self.served or str(self.model_key)
        # The RIGHT endpoint for what this pod serves. Asking an embeddings pod
        # for a chat completion is a 404 by design —
        #   "Model 'BAAI/bge-m3' is a embedding model and cannot be used for
        #    text generation. Use the embedding endpoint instead."
        # — printed at WARNING volume on every boot of a pod that is perfectly
        # healthy. It reads as a broken pod, and it cost a wrong diagnosis.
        cap = ""
        try:
            cap = model_capability(self.entry or {})
        except Exception:
            cap = ""
        if cap == "embeddings":
            path, payload = "/v1/embeddings", {"model": name, "input": "ok"}
        else:
            path = "/v1/chat/completions"
            payload = {"model": name,
                       "messages": [{"role": "user", "content": "ok"}],
                       "max_tokens": 1, "temperature": 0.0}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            reply = pod_http.post(url.rstrip("/") + path,
                                  json=payload, headers=headers, timeout=budget)
        except Exception as exc:
            print(f"[runpod] pod {pod_id}: warm-up did not complete "
                  f"({type(exc).__name__}: {exc}) — serving anyway, the first "
                  f"real request will pay the model load", flush=True)
            return False
        took = time.time() - t0
        if reply.status_code >= 400:
            print(f"[runpod] pod {pod_id}: warm-up answered HTTP "
                  f"{reply.status_code} after {took:.0f}s — serving anyway "
                  f"({str(reply.text)[:200]})", flush=True)
            return False
        print(f"[runpod] pod {pod_id}: weights resident after {took:.0f}s "
              f"(warm-up inference) — ready means ready", flush=True)
        return True

    def _data_centers(self) -> list:
        """Regions to try, in order. Always at least one entry.

        A network volume overrides everything: it lives in one region and a pod
        elsewhere cannot attach it. Otherwise the model's list, the model's
        single setting, the account's list, the account's single setting, and
        finally [""] — let RunPod choose, which is what blank has always meant.
        """
        # A seed run names the ONE region it is filling. It bypasses the
        # have-a-volume filter below on purpose: the volume is there, empty,
        # and the pod about to boot is how it stops being empty.
        seeding = (getattr(self, "_seed_dc", "") or "").strip().upper()
        if seeding:
            return [seeding]
        vol_map = volume_map_for(self.mcfg, self.account)
        pinned = getattr(self, "_volume_dc", "") or ""
        if pinned and not vol_map:
            return [pinned]
        for src in (getattr(self.mcfg, "data_centers", None),
                    [self.mcfg.data_center] if self.mcfg.data_center else None,
                    getattr(self.account, "data_centers", None),
                    [getattr(self.account, "data_center", "")]
                    if getattr(self.account, "data_center", "") else None):
            vals = [str(v).strip().upper() for v in (src or []) if str(v).strip()]
            if vals:
                return self._with_volumes(vals, vol_map)
        # No region list at all. With a map, its own regions ARE the list: the
        # operator put a volume in each, which is a statement about where this
        # model may run. Without one, blank keeps meaning "RunPod chooses".
        return list(vol_map) if vol_map else [""]

    def _with_volumes(self, regions: list, vol_map: dict) -> list:
        """``regions`` that have a volume in ``vol_map``, order kept.

        A region with no volume is dropped rather than rented without storage:
        that pod would re-download every byte and boot minutes slower, which is
        the cost the map exists to remove, and it would do it invisibly. Saying
        which regions were dropped is the point of the message.
        """
        if not vol_map:
            return regions
        keep = [r for r in regions if r in vol_map]
        drop = [r for r in regions if r not in vol_map]
        if drop:
            print(f"[runpod] {self.model_key!r}: no volume configured for "
                  f"{', '.join(drop)} — not renting there (network_volumes has "
                  f"{', '.join(sorted(vol_map))})", flush=True)
        if not keep:
            from codai.api.runpod_client import RunpodError
            raise RunpodError(
                f"RunPod: none of the allowed regions for {self.model_key!r} "
                f"({', '.join(regions)}) has a volume in network_volumes "
                f"({', '.join(sorted(vol_map))}). Add a volume in one of those "
                f"regions, or widen data_centers to a region that has one.")
        return keep

    def _data_center(self) -> str:
        """Which RunPod region this pool's pods go to.

        The volume's own data centre wins: a network volume lives in one region
        and a pod elsewhere simply cannot attach it. Then the model's setting,
        then the account default.

        A method rather than a local because it was a local: ``dc`` was computed
        inside _create_with_fallback and then read again in _provision_one,
        which has no such name. The pod was created, the PodHandle raised
        NameError, and the pool never recorded the pod it had just rented — so
        provisioning retried and the untracked A40 billed on.
        """
        return self._data_centers()[0]

    def _provision_one(self, cancellable: bool = False):
        """Create + boot one pod; append it healthy. Blocking (minutes).

        Retries on a different machine when a pod is accepted but never boots.

        ``cancellable`` marks an ANTICIPATORY boot -- the warm floor, or growth
        for demand that may be gone by the time the card is ready. Those are
        abandoned, and the pod terminated, if the fleet is released while they
        are on the way: a boot takes minutes, and paying one out to the end for
        a pod that will be shed on arrival is the whole of the complaint. A
        boot a waiting request is blocked on is never cancellable, whatever
        arrives meanwhile -- the request is what the pod is for.
        """
        from codai.api.runpod_client import RunpodClient, RunpodError, pod_console_url
        client = RunpodClient(self.account)
        # Read BEFORE anything is created, so a shed that arrives at any point
        # from here on is seen. Captured even when not cancellable, so the one
        # branch that reads it needs no second condition.
        with self._cv:
            _epoch = self._cancel_epoch

        def _give_back(pod_id, why):
            """Terminate a pod this pool no longer wants and stop tracking it."""
            print(f"[runpod] pod {pod_id} for {self.model_key!r} was released "
                  f"while it was still booting ({why}) — terminating it now "
                  f"rather than paying out the boot for a pod nobody wants",
                  flush=True)
            gone = True
            try:
                client.terminate_pod(pod_id)
            except Exception as exc:
                gone = False
                print(f"[runpod] !! could not terminate {pod_id}: {exc} — it is "
                      f"STILL BILLING; the reaper will take it, or stop it now "
                      f"with POST /v1/runpod/release", flush=True)
            with _pools_lock:
                _provisioning_ids.discard(pod_id)
                _provisioning_info.pop(pod_id, None)
            # Only forget a pod that is actually gone. Dropping the entry after
            # a FAILED terminate throws away the registry's record of a pod
            # that is still running, leaving the name prefix as the only thing
            # that can still identify it as ours.
            if gone:
                try:
                    unregister_pod(pod_id)
                except Exception:
                    pass

        # The caller's narrowing, if a lease is in force. Read per provision so
        # a lapsed lease stops constraining the very next pod.
        ranked = _rank_gpus(client, self.mcfg, self.account,
                            narrow=self.client_gpu_narrow())
        # Each (card, region) pair is its own candidate, so the existing
        # capacity fallback walks regions too: no A40 in the first EU region
        # falls through to the next instead of failing the request.
        dcs = self._data_centers()
        if len(dcs) > 1:
            ranked = [dict(sel, _dc=dc) for sel in ranked for dc in dcs]
            print(f"[runpod] {len(dcs)} regions allowed for {self.model_key!r}: "
                  f"{', '.join(dcs)}", flush=True)
        port = self.mcfg.port or 8000
        idx, attempts, last_exc = 0, 0, None

        while idx < len(ranked) and attempts < self.MAX_PROVISION_ATTEMPTS:
            pod_id, sel, idx = self._create_with_fallback(client, ranked, idx)
            attempts += 1
            console = pod_console_url(pod_id)
            with _pools_lock:
                _provisioning_ids.add(pod_id)   # shield from THIS process's reaper
                _provisioning_info[pod_id] = {
                    "model": self.model_key, "gpu": sel["display_name"],
                    "is_spot": sel["is_spot"], "hourly_usd": sel["price"],
                    "console_url": console, "started_at": time.time()}
            # …and from every SIBLING engine's reaper, which cannot see the set
            # above: each engine has its own pools and its own reaper.
            register_pod(pod_id, str(self.model_key), api_key=self.api_key,
                         # The rate is known HERE, not only once the pod is
                         # ready. A boot interrupted by a restart or a crash
                         # leaves an entry that bills at a real $/hr, and
                         # without this the reaper could only say "the registry
                         # records no rate for it" and drop the money.
                         hourly_usd=float(sel.get("price") or 0.0),
                         image=str(self._image or ""),
                         gpu=str(sel["display_name"]),
                         vram_gb=float(sel.get("memory_gb") or 0.0),
                         gpu_count=int(sel.get("gpu_count") or 1))  # url once ready
            _warn = weight_transfer_warning(_model_entry(self.served or self.model_key),
                                            self.mcfg, self.account)
            if _warn:
                print("[runpod] " + "!" * 68, flush=True)
                for _line in _warn.split(". "):
                    if _line.strip():
                        print(f"[runpod] !! {_line.strip().rstrip('.')}.", flush=True)
                print("[runpod] " + "!" * 68, flush=True)
            print(f"[runpod] pod {pod_id} created for {self.model_key!r} "
                  f"({sel['display_name']} {sel['cloud_type']}) — booting; logs: {console}",
                  flush=True)
            boot_to = float(self.mcfg.boot_timeout_s or self.PORT_TIMEOUT_S)
            load_to = float(self.mcfg.load_timeout_s or self.HEALTH_TIMEOUT_S)
            # Phase timings. A cold pod takes minutes and "it was slow" is not a
            # diagnosis: the image pull, the server start and the model load are
            # different problems with different fixes.
            t_created = time.time()
            url = ""
            try:
                url = client.wait_ready(pod_id, port, ready_timeout=boot_to,
                                        direct_tcp=bool(getattr(self, "_direct", False)),
                                        tls=bool(getattr(self, "_tls", False)))
                t_port = time.time()
                print(f"[runpod] pod {pod_id}: port open after "
                      f"{t_port - t_created:.0f}s (image pull + container start)",
                      flush=True)
                # Wait for the OpenAI server inside the pod (image pull + model load).
                deadline = time.time() + load_to
                while time.time() < deadline:
                    if cancellable and self._boot_cancelled(_epoch):
                        _give_back(pod_id, "during the model load")
                        raise BootCancelled(pod_id)
                    if _pod_health_ok(url, path=self.health_path,
                                      api_key=self.api_key):
                        print(f"[runpod] pod {pod_id}: serving after "
                              f"{time.time() - t_port:.0f}s more "
                              f"({time.time() - t_created:.0f}s total)", flush=True)
                        self._log_pod_catalogue(url)
                        # Pay the model load here, where it is visible as a pod
                        # that is not ready yet, rather than inside the first
                        # request where it looks like a hang.
                        self._warmup(pod_id, url)
                        break
                    time.sleep(5)
                else:
                    raise RunpodError(f"pod {pod_id} OpenAI server did not answer "
                                      f"/v1/models within {int(load_to)}s")
            except BootCancelled:
                # Already terminated and untracked by `_give_back`. NOT a boot
                # failure: retrying on the next machine would rent exactly the
                # card that was just given back.
                raise
            except Exception as exc:
                # Self-diagnose: pull the vLLM/container log before tearing down, so
                # the cause (OOM / bad args / model gate / stuck machine) is in the log.
                last_exc = exc
                print(f"[runpod] pod {pod_id} boot FAILED for {self.model_key!r}: {exc}",
                      flush=True)
                self._dump_pod_logs(client, pod_id, "boot failed", url)
                try:
                    client.terminate_pod(pod_id)
                except Exception as term_exc:
                    # NOT `pass`. This is the moment a pod becomes a leak: the
                    # teardown runs through the same API that just failed, so
                    # the attempt most likely to fail is the one that matters,
                    # and a silent drop left a GPU billing with nothing in any
                    # log naming it. Say it in the words of the thing that went
                    # wrong, and leave the registry entry alone — the reaper
                    # owns a pod it can still see, and that entry is now the
                    # only record that this pod exists.
                    print(f"[runpod] !! pod {pod_id} COULD NOT BE TERMINATED "
                          f"after its boot failed: {term_exc}", flush=True)
                    print(f"[runpod] !! it is still billing at "
                          f"${sel['price']:.2f}/hr. The stale-pod reaper will "
                          f"take it once it is past the grace period; to stop "
                          f"it now: POST /v1/runpod/release {{\"pod_id\": "
                          f"\"{pod_id}\"}} or {console}", flush=True)
                else:
                    # A boot that failed still ran a GPU for the minutes it
                    # took to fail, and no pool ever held this pod, so nothing
                    # else would ever ledger those minutes.
                    try:
                        close_bill_from_registry(pod_id, "boot failed")
                    except Exception:
                        pass
                    unregister_pod(pod_id)
                with _pools_lock:
                    _provisioning_ids.discard(pod_id)
                    _provisioning_info.pop(pod_id, None)
                if attempts < self.MAX_PROVISION_ATTEMPTS and idx < len(ranked):
                    print(f"[runpod] retrying on the next machine "
                          f"(attempt {attempts + 1}/{self.MAX_PROVISION_ATTEMPTS})", flush=True)
                    continue
                raise
            h = PodHandle(pod_id=pod_id, url=url, hourly_usd=sel["price"],
                          started_at=time.time(), gpu=sel["display_name"],
                          is_spot=sel["is_spot"], healthy=True, last_used=time.time(),
                          console_url=console,
                          data_center=(sel.get("_dc") or self._data_center()),
                          cloud_type=str(sel.get("cloud_type") or ""),
                          gpu_count=int(sel.get("gpu_count") or 1),
                          # The candidate's own figure, already multiplied by
                          # gpu_count where the pool offered several cards.
                          vram_gb=float(sel.get("memory_gb") or 0.0))
            # The last chance, and the one that matters most: a shed that
            # lands in the seconds between "serving" and "appended" would
            # otherwise put the pod in the fleet a moment after the fleet was
            # released, where only the idle timer would ever find it.
            if cancellable and self._boot_cancelled(_epoch):
                _give_back(pod_id, "just as it came up")
                raise BootCancelled(pod_id)
            self._add_pod(h)
            with _pools_lock:
                _provisioning_ids.discard(pod_id)
                _provisioning_info.pop(pod_id, None)
            # Publish the URL so a sibling engine needing the same pool reuses
            # this pod instead of renting a second GPU for the same work.
            register_pod(pod_id, str(self.model_key), url, self.api_key,
                         hourly_usd=sel["price"],
                         # So a sibling — or we, after a restart — can tell
                         # whether this pod is even adoptable: same image, and a
                         # card the pool prefers rather than someone's fallback.
                         image=str(self._image or ""),
                         gpu=str(sel["display_name"]),
                         # ...and how big that card is, for the share of the
                         # load it should carry once adopted.
                         vram_gb=float(sel.get("memory_gb") or 0.0),
                         gpu_count=int(sel.get("gpu_count") or 1))
            print(f"[runpod] pod {pod_id} ready for {self.model_key!r} at {url}", flush=True)
            return h
        raise last_exc or RunpodError("RunPod: could not provision a pod.")

    def ensure_ready(self):
        """Provision up to max(min_pods, 1) pods and block until ≥1 is healthy.
        Called at model load so the first request has a pod."""
        want = max(self.mcfg.min_pods, 1)
        with self._cv:
            if any(p.healthy for p in self.pods):
                return
            # A pod takes MINUTES to answer its health probe, and it is not in
            # self.pods until it does. Without this flag the 15 s scaler sees
            # "0 healthy, floor 1" and rents a second one — two A40s where the
            # configuration asked for one.
            if self._booting:
                return
            # ONE PER POD, not one per call. This said `+= 1` and then booted
            # `want` of them, so a model with min_pods 3 had three cards on the
            # way while every other reader of `_booting` -- the 15s scaler's
            # `in_hand`, `_should_grow`, the parallel-boot cap -- believed there
            # was one. Each of those then topped up against a deficit that was
            # already being filled. Asking for N and getting more than N starts
            # here.
            self._booting += want
        try:
            # A pod for this pool may already be up — a sibling engine's, or our
            # own from before the last restart. Renting beside it is how a warm
            # floor of one became two A40s at every boot.
            if self._adopt_shared_pod():
                return          # `finally` gives back all `want`
            # Nothing ready to adopt — but a pod may be mid-boot, registered
            # with no URL yet. That is one pod on the way, not none. It matters
            # most right here: the warm floor is rebuilt at every boot, so an
            # upgrade landing during the ~6 minutes a pod takes to pull its
            # image would rent a second one, every time.
            if sibling_is_provisioning(str(self.model_key)):
                print(f"[runpod] a pod for {self.model_key!r} is already booting "
                      f"— not renting another; the scaler will adopt it",
                      flush=True)
                return
            for _ in range(want):
                try:
                    self._provision_one()
                except Exception as exc:
                    print(f"[runpod] ensure_ready: {exc}", flush=True)
                    raise
        finally:
            with self._cv:
                self._booting = max(0, self._booting - want)
                self._cv.notify_all()

    def _say_no_pod(self, total: int, healthy: int, draining: int,
                    at_cap: int) -> None:
        """One rate-limited line naming WHICH of the four reasons it was."""
        now = time.time()
        if now < getattr(self, "_no_pod_said", 0.0):
            return
        self._no_pod_said = now + 15.0
        if total == 0:
            why = "the pool holds no pods at all"
        elif healthy == 0:
            why = (f"none of its {total} pod(s) is healthy — they are booting, "
                   f"or failing their health probe")
        elif draining >= healthy:
            why = f"all {draining} healthy pod(s) are draining"
        elif at_cap:
            why = (f"all {at_cap} serving pod(s) are at their in-flight cap "
                   f"(max_inflight_per_pod, scaled per card)")
        else:
            why = "no pod was eligible, for a reason this code cannot name"
        print(f"[runpod] {self.model_key!r}: NO POD to hand out — {why} "
              f"[{total} in pool, {healthy} healthy, {draining} draining, "
              f"{at_cap} at cap, {self._booting} booting, "
              f"{self._queued()} waiting]", flush=True)

    def _pick(self, healthy: list, affinity: str):
        """Which pod serves this request: the one that already holds the
        conversation's cache, else the least loaded."""
        free, draining, at_cap = [], 0, 0
        for p in healthy:
            if p.draining:
                draining += 1
                continue
            cap = self.pod_capacity(p)          # scaled to THIS pod's card
            if cap <= 0 or p.inflight < cap:
                free.append(p)
            else:
                at_cap += 1
        if not free:
            # SAY WHY, or the next incident is the last one again. "811 pages
            # waiting with 50 pods up" has four different causes -- every pod
            # unhealthy, every pod draining, every pod at its in-flight cap, or
            # no pod in the list at all -- and they need opposite responses.
            # A live run spent 41 minutes and $43/hr indistinguishable between
            # them, because this returned a bare None and the caller could only
            # report the queue it could not drain.
            with self._cv:
                total = len(self.pods)
            self._say_no_pod(total, len(healthy), draining, at_cap)
            return None
        if affinity and getattr(self.mcfg, "sticky_sessions", True):
            pid = self._affinity.get(affinity)
            for p in free:
                if p.pod_id == pid:
                    return p
        # Weighted least-connections. Plain least-inflight is card-blind: in a
        # mixed fleet it hands an A100 and an A40 the same number of pages, so
        # the fast card drains its share and idles while the slow one straggles
        # -- and a document finishes only when its LAST page lands, so the
        # slowest pod sets the latency the client sees.
        return min(free, key=lambda x: x.inflight / max(0.01, self.pod_weight(x)))

    def _remember_affinity(self, affinity: str, pod: "PodHandle") -> None:
        if not affinity or not getattr(self.mcfg, "sticky_sessions", True):
            return
        self._affinity[affinity] = pod.pod_id
        self._affinity.move_to_end(affinity)
        while len(self._affinity) > 4096:
            self._affinity.popitem(last=False)

    def _preferred_vram_gb(self) -> float:
        """VRAM of the card the operator's numbers were written for.

        The FIRST entry of `gpu_types` -- naming a card is how an operator says
        "this is what I am sizing for". Read from a pod actually running that
        card when one is, because that figure came from RunPod rather than from
        a table in here that would rot. 0 when nothing says, which makes every
        weight 1.0 and the whole mechanism inert: a mixed-fleet optimisation
        must never be able to misroute a single-card fleet.
        """
        prefer = [str(g).strip() for g in (getattr(self.mcfg, "gpu_types", None) or [])
                  if str(g).strip()]
        if not prefer:
            return 0.0
        head = prefer[0].lower()
        with self._cv:
            pods = list(self.pods)
        for p in pods:
            vram = float(getattr(p, "vram_gb", 0.0) or 0.0)
            name = str(getattr(p, "gpu", "") or "").lower()
            if vram > 0 and name and (head in name or name in head):
                return vram
        return 0.0

    #: Must match the caller's retry interval -- see `_denied`.
    DENY_WINDOW_S = 2.0

    def _note_denied(self) -> None:
        """Record one page turned away. Trimmed as it goes."""
        now = time.time()
        with self._cv:
            self._denied.append(now)
            cut = now - (self.DENY_WINDOW_S * 4)
            if self._denied[0] < cut:
                self._denied = [t for t in self._denied if t >= cut]
            if len(self._denied) > _DENY_CAP:
                # A hard cap as well as the age filter. Under a burst every
                # entry is recent, so filtering by age drops nothing and the
                # list grows with the burst -- and keeping more than the cap
                # buys nothing, because only the newest window is ever counted.
                del self._denied[:-_DENY_CAP]

    def _denied_recent(self) -> int:
        """Distinct pages waiting, as far as the pool can tell."""
        cut = time.time() - self.DENY_WINDOW_S
        with self._cv:
            return sum(1 for t in getattr(self, "_denied", ()) if t >= cut)

    def _queued(self) -> int:
        """Work that wants a pod but has none: threads parked, or pages denied.

        Either signal, never their sum -- a thread parked in acquire and a page
        turned away that will retry are the same page counted two ways.
        """
        return max(max(0, self._waiting), self._denied_recent())

    def _bump_cards(self) -> None:
        """Invalidate the per-card memo. Call with `_cv` held.

        Cheap on purpose: a counter, not a clear, so a caller mid-read of the
        old generation finishes against a consistent value rather than a dict
        that emptied underneath it.
        """
        self._card_gen = getattr(self, "_card_gen", 0) + 1
        cache = getattr(self, "_card_cache", None)
        if cache is not None and len(cache) > 256:
            cache.clear()

    def _card_facts(self, name: str) -> dict:
        """`{weight, capacity, threshold}` for this card, computed once.

        The three were separate methods doing overlapping work -- both weight
        and threshold read `gpu_limits`, and capacity called weight -- so one
        memo covers all of them and the tables are scanned once per card per
        generation instead of once per pod per call.
        """
        # getattr-initialised: a pool may be built by a caller that did not run
        # __init__ (a test double, a duck-typed stand-in), and a memo that
        # raises on such a pool would turn an optimisation into an outage.
        cache = getattr(self, "_card_cache", None)
        if cache is None:
            cache = self._card_cache = {}
        key = (name or "", getattr(self, "_card_gen", 0))
        got = cache.get(key)
        if got is not None:
            return got
        facts = {"weight": self._compute_weight(name),
                 "capacity": 0, "threshold": 1.0}
        lim = card_limits(self.mcfg, name)
        base_inflight = int(getattr(self.mcfg, "max_inflight_per_pod", 0) or 0)
        stated_inflight = int(lim.get("inflight") or 0)
        if stated_inflight != base_inflight and stated_inflight > 0:
            facts["capacity"] = stated_inflight
        elif base_inflight > 0:
            facts["capacity"] = max(1, int(round(base_inflight * facts["weight"])))
        base_up = max(1, int(getattr(self.mcfg, "scale_up_inflight_per_pod", 1) or 1))
        stated_up = int(lim.get("scale_up") or 0)
        facts["threshold"] = (float(stated_up) if stated_up and stated_up != base_up
                              else base_up * facts["weight"])
        cache[key] = facts
        return facts

    def pod_weight(self, pod: "PodHandle") -> float:
        """This pod's share of the load — memoised per card. See `_card_facts`."""
        return self._card_facts(
            str(getattr(pod, "gpu", "") or "").strip())["weight"]

    def _compute_weight(self, name: str) -> float:
        """This pod's share of the load, relative to the preferred card.

        An explicit `gpu_weights` entry wins: VRAM bounds how many requests FIT
        but says nothing about how fast they finish, and the two diverge badly
        (A100 SXM: 1.67x an A40's memory, 2.93x its bandwidth; L40S: same
        memory, ~1.24x faster). Absent an entry, the VRAM ratio is the honest
        zero-measurement proxy -- it is right about capacity and conservative
        about speed.

        Clamped to [0.25, 8] and never 0: a pod whose card we cannot identify
        is treated as the preferred one and keeps serving. Taking it out of
        rotation for want of a number would turn a reporting gap into an
        outage.
        """
        prefer_name = ""
        prefer = [str(g).strip() for g
                  in (getattr(self.mcfg, "gpu_types", None) or []) if str(g).strip()]
        if prefer:
            prefer_name = prefer[0]
        # THE TABLE FIRST, and derived rather than configured twice: if the
        # operator stated a scale-up depth for both cards, their ratio IS the
        # share of the load each should carry. One source of truth beats a
        # weights table that can disagree with the limits table.
        table = getattr(self.mcfg, "gpu_limits", None) or {}
        if table and name and prefer_name:
            mine_t = _weight_for({k: 1 for k in table}, name)
            base_t = _weight_for({k: 1 for k in table}, prefer_name)
            if mine_t is not None and base_t is not None:
                a = card_limits(self.mcfg, name)["scale_up"]
                b = card_limits(self.mcfg, prefer_name)["scale_up"]
                if b > 0:
                    return max(0.25, min(8.0, a / float(b)))
        weights = getattr(self.mcfg, "gpu_weights", None) or {}
        mine = _weight_for(weights, name)
        if mine is not None:
            # NORMALISED by the preferred card's own weight, so the configured
            # concurrency and threshold mean what they say: "per preferred
            # card". The operator's table is anchored wherever is convenient
            # (ours is at the A40, because that is what was measured), and
            # multiplying the threshold by an absolute 2.9 for the very card it
            # was written for would size the fleet three times too large.
            base = _weight_for(weights, prefer_name) if prefer_name else None
            return max(0.25, min(8.0, mine / (base or 1.0)))
        base = self._preferred_vram_gb()
        vram = self._vram_of_card(name)
        if base > 0 and vram > 0:
            return max(0.25, min(8.0, vram / base))
        return 1.0

    def _vram_of_card(self, name: str) -> float:
        """VRAM reported by any pod running this card, or 0.

        By card rather than per pod because the weight is a property of the
        card: two A40s do not deserve two different weights, and asking per pod
        is what made this an O(pods) walk inside an O(pods) loop.
        """
        want = str(name or "").strip().lower()
        if not want:
            return 0.0
        with self._cv:
            pods = list(self.pods)
        for p in pods:
            if str(getattr(p, "gpu", "") or "").strip().lower() == want:
                return float(getattr(p, "vram_gb", 0.0) or 0.0)
        return 0.0

    def pod_capacity(self, pod: "PodHandle") -> int:
        """How many requests THIS pod may hold. 0 = unlimited.

        A per-card `inflight` is taken as written; otherwise the model-level
        number is scaled by the card's weight. Memoised — see `_card_facts`.
        """
        return self._card_facts(
            str(getattr(pod, "gpu", "") or "").strip())["capacity"]

    def pod_threshold(self, pod: "PodHandle") -> float:
        """The in-flight depth at which THIS pod justifies another one."""
        return self._card_facts(
            str(getattr(pod, "gpu", "") or "").strip())["threshold"]

    def _pods_for_demand(self, serving: list) -> int:
        """How many pods the work we can SEE justifies, capped at max_pods.

        The work we can see is what the pods are carrying plus what is blocked
        in acquire() waiting for a slot. The waiters are the part that was
        missing: without them the grower knew the fleet was full but not by how
        much, so the only question it could answer was "one more?".
        """
        threshold = max(1, int(self.mcfg.scale_up_inflight_per_pod or 1))
        load = sum(p.inflight for p in serving) + self._queued()
        # Per POD, weighted: the threshold describes the preferred card, so a
        # fleet of bigger cards carries the same load in fewer pods. Dividing
        # by a flat threshold would rent ~3x more A100s than the work needs,
        # which is the same arithmetic that reached 94 pods at 10 % utilisation.
        per_pod = sum(self.pod_threshold(p) for p in serving)
        mean = (per_pod / len(serving)) if serving else float(threshold)
        if mean <= 0:
            mean = float(threshold)
        want = int(-(-load // mean))          # ceil, in weighted pod-equivalents
        return max(1, min(self.effective_max_pods(), want))

    def _boot_cancelled(self, epoch: int) -> bool:
        """Was this boot released while it was still booting?

        Only ANTICIPATORY boots ask -- the warm floor and background growth.
        A boot that a waiting request is blocked on is never cancelled however
        many sheds arrive: the request is the thing the pod is for.
        """
        with self._cv:
            return self._cancel_epoch != epoch

    def _should_grow(self, healthy: list) -> bool:
        """True when real concurrency justifies another pod."""
        return bool(self._grow_reason(healthy))

    def _grow_reason(self, healthy: list) -> str:
        """WHY another pod is justified, in the words of the trigger that fired,
        or '' for no.

        Two independent triggers, and saying which one fired is the whole point
        of this returning a sentence. The caller used to print the DEMAND
        number whichever trigger fired, so a pool growing because its one pod
        was saturated announced "growing to meet 1 pod(s) of demand — 1 up" —
        which reads as a pool renting a second card to meet demand it has
        already met, i.e. as a bug. Observed on a live orchestrator and it cost
        a real investigation to find that the fleet was behaving correctly.

        The pool used to grow only when it had NO healthy pod, so every request
        beyond the first piled onto pod #1 and max_pods never came into play.

        Then it grew one pod at a time — and since `_provisioning` stayed true
        for the whole boot, "at a time" meant one per BOOT. A pod takes ~12
        minutes to become ready (image pull plus warm-up), so a burst needing
        four pods took forty minutes to be served by four, whatever max_pods
        said. Measured on 2026-10-13: 64 pages in flight at 16 per pod, two pods
        rented, ready ten minutes apart.

        So the bound is the DEMAND, not the number one — while never exceeding
        max_pods, and never more than `scale_up_max_parallel_boots` on the way
        at once, because the demand-derived number is also the number of GPUs
        that start billing together.
        """
        in_hand = len(self.pods) + self._booting
        if in_hand >= self.effective_max_pods():
            return ""
        parallel = max(1, int(getattr(self.mcfg, "scale_up_max_parallel_boots", 1) or 1))
        if self._booting >= parallel:
            return ""
        # A draining pod is on its way out and takes no new work, so counting it
        # as capacity would answer "no need to grow" with the one pod that cannot
        # help — and counting its emptying in-flight as the minimum would say the
        # opposite. It is simply not part of the fleet any more.
        serving = [p for p in healthy if not p.draining]
        if not serving:
            # Nothing can serve a request. One on the way is enough to wait for;
            # a second here would be two cards for the same first request, which
            # is how a warm floor of one became two.
            if self._booting:
                return ""
            return "nothing can serve a request yet — renting the first pod"
        # Demand we can measure: what the pods carry plus what is blocked. This
        # is the part that may grow SEVERAL pods at once, because the number is
        # real rather than a guess about the future.
        want = self._pods_for_demand(serving)
        if in_hand < want:
            threshold = max(1, int(self.mcfg.scale_up_inflight_per_pod or 1))
            load = sum(p.inflight for p in serving) + self._queued()
            return (f"{load} page(s) in flight or waiting at {threshold} per pod "
                    f"needs {want} pod(s), and {in_hand} are in hand")
        # And the original, anticipatory trigger: every pod is at its in-flight
        # threshold, so the next request has nowhere good to go. Kept because
        # `max_inflight_per_pod` defaults to unlimited, so requests are often
        # never blocked and `_waiting` stays 0 — without this a fleet whose pods
        # are merely saturated would never grow at all. Single-flight, because
        # unlike the demand above this is a prediction, and a prediction should
        # not rent four cards.
        threshold = max(1, int(self.mcfg.scale_up_inflight_per_pod or 1))
        if self._booting:
            return ""
        least = min(p.inflight for p in serving)
        if least >= threshold:
            # NOT a demand figure. Demand is already met — one pod is exactly
            # right for `threshold` in flight — and this rents ahead of the
            # next request rather than for one that exists.
            return (f"every pod is at its {threshold}-in-flight threshold "
                    f"(least loaded carries {least}) — renting one more for "
                    f"the next burst, not for work already queued")
        return ""

    def _grow_in_background(self):
        """Provision another pod without making the triggering request wait for it.

        Scale-up is anticipatory: this request is already being served by an
        existing pod, and the new one absorbs the NEXT burst. A cold pod takes
        minutes, so blocking here would punish exactly the request that proved
        the pool needs to grow.
        """
        def _run():
            try:
                # Adopt before renting, here too. Only the cold paths did —
                # `ensure_ready`, `acquire` with nothing healthy, and the warm
                # floor — so GROWTH always rented, even with a healthy pod for
                # this very pool sitting in the registry under another process.
                # Seen live: a run adopted one surviving pod, grew by one, and
                # rented a card beside a second idle pod it already owned, which
                # then billed for nothing. Adoption refuses by itself when it
                # would mix two tokens in one pool, so this cannot make that
                # worse.
                if not self._adopt_shared_pod():
                    self._provision_one(cancellable=True)
            except BootCancelled:
                pass          # said its piece already, and it is not a failure
            except Exception as exc:
                print(f"[runpod] scale-up for {self.model_key!r} failed: {exc}", flush=True)
            finally:
                with self._cv:
                    # Decrement, not reset: several boots may be on the way, and
                    # zeroing here would let the grower start the whole fleet
                    # again for demand that is already being provisioned for.
                    self._booting = max(0, self._booting - 1)
                    self._cv.notify_all()
        threading.Thread(target=_run, daemon=True,
                         name=f"runpod-grow-{self.model_key}").start()

    def acquire(self, timeout: float = 1200.0, affinity: str = "",
                grow_inline: bool = True):
        """Return (PodHandle, url) for a ready pod, provisioning on demand.

        Bumps in-flight; the caller MUST pair it with release(pod). Picks the pod
        holding this conversation's cache when ``affinity`` is given, else the
        least-loaded one, and grows the pool in the background once the least
        loaded pod is at ``scale_up_inflight_per_pod``.

        ``grow_inline=False`` means "answer now": trigger growth in the
        background and raise rather than renting a pod in the caller's thread
        or waiting on the condition variable. For a caller that can wait
        somewhere cheaper than a thread -- a coroutine with its own retry loop
        -- this is the only safe mode, and the reason is arithmetic:

        A page's lease ran with a 20 s bounded attempt in a 256-thread pool, so
        the lease STAGE could pass at most ~13 pages a second however many pods
        were up, and a thread that drew the growth branch sat through an entire
        ~9-minute boot. Against 811 pages, each retrying every couple of
        seconds, nothing ever got through: a live run dispatched ZERO pages to
        50 healthy pods in 41 minutes, every pod's log showing not one
        `POST /v1/ocr`, while the queue the scaler could see made it rent more.
        Waiting is free on an event loop and costs a whole thread here.
        """
        from codai.api.runpod_client import RunpodError
        deadline = time.time() + timeout
        while True:
            grow_now = False
            with self._cv:
                healthy = [p for p in self.pods if p.healthy]
                chosen = self._pick(healthy, affinity) if healthy else None
                # Asked ONCE, and the answer carries its own reason. Deciding
                # here and describing it separately below is what let the
                # message name a trigger that had not fired.
                why = self._grow_reason(healthy)
                if why:
                    self._booting += 1
                    grow_now = True
                if chosen is not None:
                    chosen.inflight += 1
                    _slots_of(chosen).append(time.time())
                    chosen.last_used = time.time()
                    self._remember_affinity(affinity, chosen)
                    url = chosen.url
            if chosen is not None:
                if grow_now:
                    print(f"[runpod] {self.model_key!r}: growing — {why} "
                          f"[{len(self.pods)} up, {self._booting} booting]",
                          flush=True)
                    self._grow_in_background()      # absorbs the NEXT request
                return chosen, url

            if grow_now and not grow_inline:
                # Growth belongs in the background; the caller waits on its own
                # loop. Released before raising, or the booting count leaks and
                # `_grow_reason` stops firing for everyone.
                with self._cv:
                    self._booting = max(0, self._booting - 1)
                    self._cv.notify_all()
                self._note_denied()
                self._grow_in_background()
                raise NoPodRightNow(
                    f"no pod free for {self.model_key!r} right now; growth "
                    f"started in the background")

            if grow_now:
                # Before renting: is a sibling process already running a pod for
                # this exact pool? One engine per GPU means the same capability
                # was being rented a card each.
                if sibling_is_provisioning(str(self.model_key)):
                    # Another engine is already booting one for this pool. Wait
                    # for it rather than renting a second card for the same work.
                    print(f"[runpod] another engine is booting a pod for "
                          f"{self.model_key!r} — waiting for it", flush=True)
                    with self._cv:
                        self._booting = max(0, self._booting - 1)
                        self._cv.notify_all()
                        # Never past the caller's deadline. This waited a flat
                        # 15s and looped straight back into _should_grow, which
                        # answers the same way while the sibling is still
                        # booting — so the deadline was never reached and the
                        # caller's timeout was ignored for as long as a sibling
                        # entry survived (up to REAP_GRACE_SECONDS).
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            raise RunpodError(
                                f"RunPod: another process is still booting a pod "
                                f"for {self.model_key!r} after {timeout}s.")
                        self._cv.wait(min(remaining, 15.0))
                    continue
                if self._adopt_shared_pod():
                    with self._cv:
                        self._booting = max(0, self._booting - 1)
                        self._cv.notify_all()
                    continue
                # Nothing to serve this request yet — provision inline and retry.
                try:
                    self._provision_one()
                finally:
                    with self._cv:
                        self._booting = max(0, self._booting - 1)
                        self._cv.notify_all()
                continue

            if not grow_inline:
                # Nothing to hand out and growth is someone else's job: say so
                # at once instead of holding this thread on the condvar. The
                # count still rises, because the scaler reads it as demand --
                # it is just no longer a thread each.
                self._note_denied()
                raise NoPodRightNow(
                    f"no pod slot for {self.model_key!r} right now "
                    f"(max_pods={self.effective_max_pods()})")

            # At max_pods with every pod at its in-flight ceiling, or someone else
            # is provisioning: wait for a slot rather than piling on. Counted
            # while waiting, so the next pass through _should_grow can see how
            # much work is actually queued rather than only that the fleet is
            # full.
            with self._cv:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise RunpodError(
                        f"RunPod: no pod slot available for {self.model_key!r} "
                        f"(max_pods={self.effective_max_pods()}, "
                        f"max_inflight_per_pod={self.mcfg.max_inflight_per_pod}) "
                        f"within {timeout}s.")
                self._waiting += 1
                try:
                    self._cv.wait(min(remaining, 10.0))
                finally:
                    self._waiting = max(0, self._waiting - 1)

    def _add_pod(self, handle: PodHandle) -> bool:
        """Put a pod in the pool, at most ONCE per pod id.

        Two paths append pods -- a fresh rent and an adoption -- and neither
        used to check whether the pod was already there. On a restart both can
        fire for the same pod in the same second (`reclaim: took back a pod`
        and `adopting pod <id>'s bearer token` are logged a moment apart), and
        the pool then holds two handles for one machine.

        That would not be a cosmetic duplicate. Every capacity question is
        answered by walking `self.pods`, so one A40 counted twice would claim
        2 x max_inflight_per_pod slots, and `max_pods` -- enforced against that
        same count -- could be satisfied by two entries for one card.

        Precautionary rather than a diagnosed fault, and worth keeping the
        distinction straight. A snapshot was once seen carrying two rows for
        one pod id, but the model key was not recorded, so two POOLS sharing
        one pod was never ruled out -- and that case this does not fix. The
        separate occasion when a max_pods=1 pool really did bill for two A40s
        was two DISTINCT pod ids, caused by the warm-floor top-up counting only
        healthy pods against the cap. What justifies this guard by itself is
        that two paths appended pods with no identity check between them.

        Returns True if the pod was added, False if it was already here. The
        EXISTING handle is kept rather than replaced: it carries the in-flight
        accounting for work already dispatched to that pod, and throwing it away
        would strand every slot it holds. What can legitimately change -- the
        address, after RunPod remaps a port -- is refreshed onto it.
        """
        with self._cv:
            for p in self.pods:
                if p.pod_id != handle.pod_id:
                    continue
                if handle.url and handle.url != p.url:
                    print(f"[runpod] pod {p.pod_id} is already in the pool for "
                          f"{self.model_key!r}; keeping the handle that holds "
                          f"its {p.inflight} in-flight slot(s) and updating its "
                          f"address to {handle.url}", flush=True)
                    p.url = handle.url
                else:
                    print(f"[runpod] pod {p.pod_id} is already in the pool for "
                          f"{self.model_key!r} — not adding it twice (a second "
                          f"entry would claim a second pod's worth of capacity "
                          f"for one card)", flush=True)
                p.healthy = True
                p.last_used = time.time()
                self._cv.notify_all()
                return False
            self.pods.append(handle)
            self._bump_cards()
            self._cv.notify_all()
            return True

    def release(self, pod: PodHandle):
        with self._cv:
            pod.inflight = max(0, pod.inflight - 1)
            # Oldest first: the slots are interchangeable (nothing identifies
            # which page holds which), and dropping the oldest keeps the list a
            # true picture of how long the REMAINING ones have been held, which
            # is what the staleness sweep reads.
            slots = _slots_of(pod)
            if slots:
                slots.pop(0)
            pod.last_used = time.time()
            self._cv.notify_all()

    def discard(self, pod: PodHandle, reason: str = "unreachable") -> None:
        """Drop a pod that refused a connection, now, not at the next sweep.

        The maintenance loop probes on a timer; between two probes a pod that
        was terminated behind the pool's back (the test harness reaping by
        API, RunPod reclaiming a spot machine) is still handed out, and the
        request fails with "connection refused" for something the pool could
        have known. The caller retries on a fresh acquire.
        """
        with self._cv:
            if pod not in self.pods:
                return
            pod.healthy = False
            self.pods.remove(pod)
            self._bump_cards()
        print(f"[runpod] pod {pod.pod_id} dropped from the pool: {reason}", flush=True)
        threading.Thread(target=self._terminate, args=(pod, reason), daemon=True).start()

    def forget(self, pod: PodHandle, reason: str = "gone") -> bool:
        """Drop a pod that no longer exists. Like ``discard``, minus the kill.

        ``discard`` terminates, because it is called when OUR use of a pod
        failed and the pod may well still be running and billing. This one is
        for a pod RunPod has positively told us is gone — terminated from the
        console, reclaimed as spot, lost with its host — where there is nothing
        left to terminate and an API call would only error. The bill still has
        to be closed, so the ledger entry is written here exactly as
        ``_terminate`` writes it.

        Returns True if this call is what removed it.
        """
        with self._cv:
            if pod not in self.pods:
                return False
            pod.healthy = False
            self.pods.remove(pod)
            self._bump_cards()
        if getattr(pod, "gpu", "") == "(shared)":
            # Another process owns the bill; it writes the ledger entry.
            print(f"[runpod] borrowed pod {pod.pod_id} is gone ({reason}) — "
                  f"dropped from {self.model_key!r}", flush=True)
            return True
        from codai.api import runpod_ledger
        # To NOW, which over-counts by however long it was dead before we
        # noticed. The same approximation _terminate makes, and the reconcile
        # runs every ~30s, so the error is bounded by that.
        cost = (time.time() - pod.started_at) / 3600.0 * pod.hourly_usd
        unregister_pod(pod.pod_id)
        try:
            runpod_ledger.record_spend(
                self.model_key, cost,
                {"pod_id": pod.pod_id, "gpu": pod.gpu, "is_spot": pod.is_spot})
        except Exception as exc:
            print(f"[runpod] ledger for gone pod {pod.pod_id}: {exc}", flush=True)
        print(f"[runpod] pod {pod.pod_id} for {self.model_key!r} is gone from "
              f"RunPod ({reason}) — dropped from the pool; billed ~${cost:.3f}. "
              f"The warm floor is re-checked on the next scaler pass.", flush=True)
        return True

    def _terminate(self, pod: PodHandle, reason: str):
        if getattr(pod, "gpu", "") == "(shared)":
            # Borrowed from another engine: it owns the lifecycle and the bill.
            # Terminating it here would kill a pod that process is still using.
            print(f"[runpod] releasing borrowed pod {pod.pod_id} ({reason}); its "
                  "owner reaps it", flush=True)
            with self._cv:
                if pod in self.pods:
                    self.pods.remove(pod)
                    self._bump_cards()
            return
        from codai.api.runpod_client import RunpodClient
        from codai.api import runpod_ledger
        cost = (time.time() - pod.started_at) / 3600.0 * pod.hourly_usd
        try:
            RunpodClient(self.account).terminate_pod(pod.pod_id)
        except Exception as exc:
            print(f"[runpod] terminate {pod.pod_id} failed: {exc}", flush=True)
        unregister_pod(pod.pod_id)
        runpod_ledger.record_spend(self.model_key, cost,
                                   {"pod_id": pod.pod_id, "gpu": pod.gpu, "is_spot": pod.is_spot})
        print(f"[runpod] pod {pod.pod_id} for {self.model_key!r} terminated ({reason}); "
              f"billed ~${cost:.3f}", flush=True)

    def client_floor(self) -> int:
        """The warm floor a client leased, 0 once the lease has run out.

        Read-only and cheap: the lease is not removed here, so the scaler, the
        status page and the endpoint all see the same number within a pass.
        """
        if self._client_floor <= 0:
            return 0
        if time.time() >= self._client_floor_until:
            return 0
        # Against the EFFECTIVE ceiling, so a lease that later narrowed its own
        # max_pods does not keep a floor above it -- a floor the pool would
        # have to grow to and a ceiling forbidding it is a fleet that oscillates
        # for as long as the lease lasts.
        return min(self._client_floor, self.effective_max_pods())

    def client_cap(self) -> int:
        """The pod CEILING a client leased, 0 once the lease has run out.

        Read-only and cheap, exactly like `client_floor`, so the scaler, the
        status page and the endpoint all see the same number within a pass.

        getattr, for the same reason `_slots_of` initialises on demand: not
        every pool object this is asked about went through `__init__`, and a
        READ-ONLY question about the ceiling must never be the thing that
        raises on the path that decides whether to rent a GPU. No lease is the
        normal answer anyway.
        """
        leased = int(getattr(self, "_client_cap", 0) or 0)
        if leased <= 0:
            return 0
        if time.time() >= float(getattr(self, "_client_cap_until", 0.0) or 0.0):
            return 0
        return min(leased, max(1, self.mcfg.max_pods))

    def effective_max_pods(self) -> int:
        """The ceiling on fleet size that applies right now.

        The configured `max_pods`, narrowed by a client lease if one is in
        force. Every path that decides whether to rent another pod reads THIS
        rather than `mcfg.max_pods`, because a ceiling only half of them honour
        is not a ceiling -- it is a race between the grower that respects it and
        the one that does not.

        A LEASE can never drive this to zero: "do not grow past N" is not a
        request to stop serving, and a lease asking for 0 is read as releasing
        its cap. A configured `max_pods` of 0 is different -- that is the
        operator having turned pod renting off for this model, and it stays 0.
        """
        configured = int(self.mcfg.max_pods or 0)
        capped = self.client_cap()
        if capped <= 0 or configured <= 0:
            return configured
        out = max(1, min(configured, capped))
        if out != getattr(self, "_client_cap_logged", -1):
            print(f"[runpod] {self.model_key!r}: fleet ceiling {out} from a "
                  f"client lease (configured max_pods {configured})", flush=True)
            try:
                self._client_cap_logged = out
            except Exception:
                pass
        return out

    def client_gpu_narrow(self) -> dict:
        """The GPU narrowing in force right now, or {} once the lease lapses.

        Tied to the lease rather than to the pool, so a caller's "A40 only"
        cannot outlive the 30 minutes it asked for, and the pool goes back to the
        operator's own preference by itself. Nothing persists it: there is
        deliberately no way for a request to change the model's configuration.
        """
        if self.client_floor() <= 0:
            return {}
        out = {}
        if self._client_gpu_types:
            out["gpu_types"] = list(self._client_gpu_types)
        if self._client_allow_other is not None:
            out["allow_other_gpus"] = bool(self._client_allow_other)
        return out

    def seed_region(self, dc: str) -> dict:
        """Fill one region's volume by booting a pod there once, then dropping it.

        No new pod shape: a normal pod for this model, pinned to ``dc``, with
        that region's volume attached. Booting it IS the download — the weights
        land on the volume because the volume is where the caches point — so
        the only thing this adds is doing it deliberately, before a user's
        request is waiting on it, and tearing the pod down afterwards.

        Blocking, and minutes long. Every later pod in that region starts from
        a warm volume instead of pulling the weights again.
        """
        dc = str(dc or "").strip().upper()
        vmap = volume_map_for(self.mcfg, self.account)
        vol = vmap.get(dc, "")
        if not vol:
            raise ValueError(
                f"RunPod: no volume configured for {dc} in network_volumes "
                f"({', '.join(sorted(vmap)) or 'empty'}) — there is nothing to "
                f"seed there. Create the volume and add it to the map first.")
        started = time.time()
        before = set(p.pod_id for p in list(self.pods))
        self._seed_dc = dc
        try:
            self._provision_one()
        finally:
            self._seed_dc = ""
        fresh = [p for p in list(self.pods) if p.pod_id not in before]
        out = {"region": dc, "volume_id": vol, "took_s": round(time.time() - started, 1),
               "pods": [p.pod_id for p in fresh]}
        for pod in fresh:
            # The seed's product is on the volume, not in the pod. Keeping it
            # would bill a card nobody asked to rent.
            self._terminate(pod, f"volume seed for {dc} finished")
        return out

    def set_client_floor(self, pods: int, gpu_types=None,
                         allow_other_gpus=None, max_pods=None) -> dict:
        """Lease (or release, with 0) a client-requested warm floor and ceiling.

        Clamped to max_pods: the configured ceiling is the budget, and a client
        asking for more gets the ceiling rather than an error — it asked for
        "as many as you can", and failing the call would leave it with none.

        ``max_pods`` narrows the CEILING for the life of the lease. It is a
        separate question from the floor and it needs its own answer: a client
        that knows it is running a staged batch -- 8 pods for the first stage,
        32 for the next -- can bound what the stage may cost, where asking for
        8 warm pods bounds nothing at all. Only ever narrower than the
        operator's configured `max_pods`; ``0`` drops the lease's own cap and
        ``None`` leaves it as it was.

        A floor above the ceiling is not refused: the ceiling wins, because it
        is the number that bounds spending and the floor is a convenience. The
        reply reports what was granted either way, so a caller that asked for
        something incoherent can see that it did.
        """
        configured = max(1, int(self.mcfg.max_pods))
        # The lease's own ceiling first: the floor is clamped to it below, so a
        # caller sending {"pods": 32, "max_pods": 8} gets 8 and not 32 warm
        # pods under an 8-pod cap, which is a fleet that can never settle.
        if max_pods is None:
            leased_cap = self.client_cap()
        else:
            leased_cap = max(0, int(max_pods))
            leased_cap = min(leased_cap, configured) if leased_cap else 0
        cap = min(configured, leased_cap) if leased_cap else configured
        want = max(0, int(pods))
        granted = min(want, cap)
        if granted:
            # Asking for warm pods cancels a pending release: the client changed
            # its mind, and retiring what it just asked for would be absurd.
            self._release_pending = False
        ttl = max(60, int(getattr(self.mcfg, "client_scale_ttl_s", 1800) or 1800))
        # A cap with no floor is a legitimate lease: "grow no further than 8,
        # but hold nothing warm for me" is what a client with a long tail of
        # work wants, and releasing the cap because the floor went to 0 would
        # take the ceiling away at exactly the moment it still matters. So the
        # cap lapses with its OWN deadline, and `pods: 0` clears it only when
        # the same call did not ask for one.
        keep_cap = leased_cap if (max_pods is not None or granted) else 0
        # Raises ValueError when the caller narrowed to nothing it is allowed to
        # have; the endpoint turns that into a 400 that names the allowed set.
        narrow = narrow_gpu_choice(self.mcfg, gpu_types, allow_other_gpus)
        with self._cv:
            self._client_floor = granted
            self._client_floor_until = (time.time() + ttl) if granted else 0.0
            self._client_floor_logged = -1
            self._client_cap = keep_cap
            self._client_cap_until = (time.time() + ttl) if keep_cap else 0.0
            self._client_cap_logged = -1
            if granted:
                self._client_gpu_types = (list(narrow["gpu_types"])
                                          if gpu_types else [])
                self._client_allow_other = (narrow["allow_other_gpus"]
                                            if allow_other_gpus is not None
                                            else None)
            else:
                # Releasing the lease releases its constraints too, or an
                # "A40 only" would quietly outlive the request that wanted it.
                self._client_gpu_types = []
                self._client_allow_other = None
        print(f"[runpod] {self.model_key!r}: client asked for {want} warm pod(s), "
              f"granted {granted} (max_pods {cap}) for {ttl}s", flush=True)
        if max_pods is not None:
            print(f"[runpod] {self.model_key!r}: this lease caps the fleet at "
                  f"{keep_cap or configured} pod(s) "
                  f"(configured max_pods {configured}"
                  + ("; the lease asked for more and was narrowed to it"
                     if int(max_pods or 0) > configured else "")
                  + (")" if keep_cap else "; the lease's own cap is released)"),
                  flush=True)
        if granted and (gpu_types or allow_other_gpus is not None):
            print(f"[runpod] {self.model_key!r}: this lease asks for "
                  f"{narrow['gpu_types'] or 'any card'}"
                  f"{'' if narrow['allow_other_gpus'] else ' and nothing else'}"
                  + ("".join(f"; {n}" for n in narrow["notes"])), flush=True)
        out = {"model": self.model_key, "requested": want, "granted": granted,
               "max_pods": cap, "configured_max_pods": configured,
               "client_max_pods": keep_cap,
               "expires_in_s": ttl if (granted or keep_cap) else 0,
               "clamped": want > cap}
        if max_pods is not None and int(max_pods or 0) > configured:
            # Reported for the same reason a narrowed gpu_types is: a caller
            # that believes it capped the fleet at 256 would read the bill as
            # coming from a cap it never had.
            out.setdefault("notes", []).append(
                f"max_pods {int(max_pods)} narrowed to the configured "
                f"ceiling {configured}")
        if granted and (gpu_types or allow_other_gpus is not None):
            # Reported back, because a constraint that was silently dropped is
            # worse than one refused: the caller would read the numbers as
            # having come from the card it asked for.
            out["gpu_types"] = narrow["gpu_types"]
            out["allow_other_gpus"] = narrow["allow_other_gpus"]
            if narrow["notes"]:
                # Extend: a narrowed max_pods may already have left a note
                # here, and replacing the list would hide whichever of the two
                # constraints was reported second.
                out.setdefault("notes", []).extend(narrow["notes"])
        return out

    def effective_min_pods(self) -> int:
        """The warm floor that applies right now.

        Outside a configured schedule window the floor is 0: the pool keeps
        serving (a request still cold-starts a pod) but stops paying to hold one
        ready. Inside the window, or with no schedule, it is the configured
        min_pods.

        A client lease (``/v1/runpod/scale``) is the HIGHER of the two, never the
        sum: it says "hold this many ready", which is already satisfied by a
        schedule holding that many. So a client cannot lower the configured
        floor, and the schedule cannot cancel a lease the client is relying on.
        """
        if not getattr(self.mcfg, "schedule_enabled", False):
            scheduled = self.mcfg.min_pods
        else:
            scheduled = (self.mcfg.min_pods
                         if schedule_state(self.mcfg)["in_window"] else 0)
        leased = self.client_floor()
        if leased > scheduled and leased != self._client_floor_logged:
            print(f"[runpod] {self.model_key!r}: warm floor {leased} from a client "
                  f"lease (configured {scheduled})", flush=True)
            self._client_floor_logged = leased
        return max(scheduled, min(leased, self.effective_max_pods()))

    def _pods_needed(self, serving: list) -> int:
        """How many pods the CURRENT concurrency justifies.

        The same arithmetic as scaling up, read backwards: one pod per
        ``scale_up_inflight_per_pod`` requests in flight. Never below the warm
        floor, never above the ceiling, and never zero while anything is in
        flight.
        """
        threshold = max(1, int(self.mcfg.scale_up_inflight_per_pod or 1))
        total = sum(p.inflight for p in serving)
        need = -(-total // threshold)          # ceil, in integers
        return max(self.effective_min_pods(),
                   min(self.effective_max_pods(), need))

    def _scale_down_victim(self, serving: list):
        """Which surplus pod to give back.

        By default the least loaded, and among equals the one used least recently:
        its prefix cache is the coldest, so retiring it throws away the least
        work. (That tie-break was inverted when this was written — ``-last_used``
        picked the pod used MOST recently, discarding the warmest cache in the
        set.)

        With ``scale_down_costliest_first`` price leads instead, which is the
        whole point on a mixed fleet: pods are rented at whatever the market
        offered, so one pool can hold a cheap card and an expensive one doing the
        same work. Cost still only decides WHICH pod goes, never whether one
        goes — and a dear pod with requests on it is drained, not dropped.
        """
        if getattr(self.mcfg, "scale_down_costliest_first", False):
            # Dearest first; then the idlest, then the coldest cache.
            return max(serving, key=lambda p: (float(p.hourly_usd or 0.0),
                                               -p.inflight, -p.last_used))
        return min(serving, key=lambda p: (p.inflight, p.last_used))

    def shed_now(self) -> dict:
        """Give back every pod the load no longer needs, immediately.

        The automatic path waits out ``scale_down_after_s`` and retires one pod
        per pass, because it is GUESSING from a momentary reading and a cold pod
        costs minutes to replace. A client calling this is not guessing: it has
        finished its batch and knows no more work is coming, which is knowledge
        the autoscaler does not have. So the window and the one-per-pass limit
        are both skipped.

        What is not skipped: the warm floor still holds, and a pod with requests
        on it drains rather than dying — paying for a pod for another minute is
        cheaper than making someone's page fail.
        """
        terminated, draining = [], []
        with self._cv:
            # Cancel first, and unconditionally. A boot that is on the way is a
            # pod this client asked for and has now given back; whether the
            # arithmetic below also finds surplus among the READY pods is a
            # separate question. Anticipatory boots see this and hand their pod
            # straight back instead of joining a fleet nobody wants.
            booting = int(self._booting)
            self._cancel_epoch += 1
            serving = [p for p in self.pods if p.healthy and not p.draining]
            # Booting pods count as capacity on the way, so the surplus is
            # measured against what the fleet will BE, not only what it is.
            # Without this a client holding four boots and no ready pod was
            # told "kept 0, needed 1" and kept paying for all four.
            need = self._pods_needed(serving)
            surplus = max(0, len(serving) + booting - need)
            order = sorted(serving, key=lambda p: (p.inflight, p.last_used))
            if getattr(self.mcfg, "scale_down_costliest_first", False):
                order = sorted(serving, key=lambda p: (-float(p.hourly_usd or 0.0),
                                                       p.inflight, p.last_used))
            for victim in order[:surplus]:
                if victim.inflight == 0:
                    victim.healthy = False
                    self.pods.remove(victim)
                    self._bump_cards()
                    terminated.append(victim.pod_id)
                    threading.Thread(target=self._terminate,
                                     args=(victim, "released by client"),
                                     daemon=True).start()
                else:
                    victim.draining = True
                    draining.append(victim.pod_id)
            # A boot cancelled here is capacity that will never arrive, so it
            # must not count against the ready pods we then decide to keep --
            # otherwise a client with one ready pod and three cancelled boots
            # would see its last serving pod shed too.
            self._surplus_since = 0.0
            kept = len([p for p in self.pods if p.healthy and not p.draining])
            # Anything still here above the operator's own floor is here because
            # it is WORKING, not because anyone wants it warm. Remember that, so
            # it goes the moment it falls idle instead of waiting out
            # idle_timeout_s.
            floor = self.effective_min_pods()
            self._release_pending = bool(kept > floor or draining)
        print(f"[runpod] {self.model_key!r}: client released the fleet — "
              f"{len(terminated)} terminated, {len(draining)} draining, "
              f"{booting} boot(s) cancelled, "
              f"{kept} kept ({need} still needed by work in flight, warm floor "
              f"{floor})"
              + (" — will retire as soon as they fall idle"
                 if self._release_pending else ""), flush=True)
        return {"terminated": terminated, "draining": draining, "kept": kept,
                # Boots in flight when the release arrived. They cannot be
                # named -- a pod that is still being created has no id here --
                # but they CAN be counted, and a count is the difference
                # between "nothing happened" and "four cards stopped".
                "cancelled_boots": booting,
                # `needed` is demand; `floor` is configuration. Reporting demand
                # AS "floor" read as though the operator had asked for the pod,
                # which is the opposite of what it meant.
                "needed": need, "floor": floor,
                "retire_when_idle": self._release_pending}

    def _reclaim_stranded_inflight(self, now: float) -> None:
        """Give back in-flight slots that no page can still be holding.

        `inflight` is a ratchet: raised in `_pick`, lowered only in `release`.
        Any dispatch path that fails to release raises it for the life of the
        pod, and at `max_inflight_per_pod` the pod is locked out while its card
        idles -- reported, misleadingly, as "at capacity". Two such leaks have
        been found and fixed (a lease dropped during acquisition, a page whose
        client went away); the reason this sweep exists is the third, which has
        not been found yet.

        So the pool stops trusting the counter indefinitely. A slot held longer
        than any page can legitimately take is not in flight, whatever the
        counter says, and is returned.

        Deliberately generous: this is a safety net, not a timeout. Cancelling
        real work would be far worse than holding a phantom slot a few minutes
        longer, so the bound is well past the slowest plausible page (a full
        token budget, its continuations and a retry, on a contended card).
        """
        ttl = INFLIGHT_TTL_S
        if ttl <= 0:
            return
        cutoff = now - ttl
        freed_total = 0
        with self._cv:
            for pod in self.pods:
                slots = _slots_of(pod)
                stale = sum(1 for t in slots if t < cutoff)
                if stale:
                    slots = [t for t in slots if t >= cutoff]
                    pod.inflight_since = slots
                # The LIST is the authority, always, not only when something
                # expired. `_pick` records a timestamp for every slot it takes,
                # so a counter standing above the recorded slots is counting
                # slots with no evidence that they were ever taken -- nothing
                # can attribute them to a live page, and they can only have
                # come from a path that raised the counter and lost the page.
                #
                # This also covers the upgrade itself: pods already in the pool
                # when this code arrives carry a counter and no timestamps, and
                # skipping them would leave exactly the locked-out pod this
                # sweep exists to free.
                unaccounted = max(0, pod.inflight - len(slots))
                if not stale and not unaccounted:
                    continue
                pod.inflight = len(slots)
                freed_total += stale + unaccounted
                detail = []
                if stale:
                    detail.append(f"{stale} held longer than {ttl:.0f}s")
                if unaccounted:
                    detail.append(f"{unaccounted} with no record of being taken")
                print(f"[runpod] pod {pod.pod_id}: returned "
                      f"{stale + unaccounted} in-flight slot(s) that cannot "
                      f"belong to a live page ("
                      + ", ".join(detail)
                      + f"). inflight is now {pod.inflight}. A slot leaks when "
                      f"a page is lost between being granted a pod and "
                      f"releasing it; until now the pod was being refused work "
                      f"it could serve.", flush=True)
            if freed_total:
                self._cv.notify_all()

    def _scale_down(self, now: float) -> None:
        """Retire one surplus pod per pass, once the surplus has persisted.

        One per pass, not all of them at once: each retirement changes the
        arithmetic, and a fleet that shed everything on a single quiet moment
        would then cold-start for minutes when the next burst arrived. The
        surplus has to hold for ``scale_down_after_s`` first, for the same
        reason.

        The pod picked is the least loaded; if it still has work it is marked
        draining and left to finish, so no request is ever thrown away to save a
        few cents.
        """
        if self._closed or int(getattr(self.mcfg, "scale_down_after_s", 0) or 0) <= 0:
            return
        with self._cv:
            serving = [p for p in self.pods if p.healthy and not p.draining]
            draining = [p for p in self.pods if p.healthy and p.draining]
            need = self._pods_needed(serving)
            surplus = len(serving) - need
            total = sum(p.inflight for p in serving)

            # A pod that was draining and has now emptied is done: terminate it
            # here rather than waiting for the idle reaper's timeout, which is
            # what it was taken out of service to avoid paying.
            for p in draining:
                if p.inflight == 0:
                    p.healthy = False
                    self.pods.remove(p)
                    self._bump_cards()
                    threading.Thread(target=self._terminate, args=(p, "scaled down"),
                                     daemon=True).start()

            if surplus <= 0 or total == 0:
                # Nothing in flight at all is NOT this routine's business. The
                # idle reaper already walks a quiet pool down to the warm floor,
                # on idle_timeout_s — a number the operator chose, weighing the
                # bill against a cold start. Shedding here too would quietly
                # replace 900s with scale_down_after_s and make that setting a
                # lie. This exists for the other case: pods still serving, but
                # more of them than the load needs.
                self._surplus_since = 0.0
                return
            if not self._surplus_since:
                self._surplus_since = now
                return
            if (now - self._surplus_since) < float(self.mcfg.scale_down_after_s):
                return
            # Hold the observation window again before the next one, so a pass
            # does not walk the fleet down in one sweep.
            self._surplus_since = now
            victim = self._scale_down_victim(serving)
            if victim.inflight == 0:
                victim.healthy = False
                self.pods.remove(victim)
                self._bump_cards()
                threading.Thread(target=self._terminate, args=(victim, "scaled down"),
                                 daemon=True).start()
                why = "terminated"
            else:
                victim.draining = True
                why = f"draining ({victim.inflight} in flight)"
        print(f"[runpod] {self.model_key!r}: {len(serving)} pod(s) for {total} "
              f"request(s) in flight, {need} needed — {why} {victim.pod_id}",
              flush=True)

    def maintain(self):
        """One scaler pass: shed pods the load no longer needs, reap idle/dead
        ones to min_pods, keep min warm."""
        now = time.time()
        # Before anything is torn down: a pod's log is only readable while the
        # pod exists, and the interesting lines are usually the last ones.
        with self._cv:
            alive_now = [p for p in self.pods if p.healthy]
        for p in alive_now:
            try:
                self._drain_pod_log(p)
            except Exception:
                pass
        self._reclaim_stranded_inflight(now)
        self._scale_down(now)
        with self._cv:
            pods = list(self.pods)
        # Idle teardown (down to min_pods), only pods with no in-flight work.
        idle_to = self.mcfg.idle_timeout_s
        keep_min = self.effective_min_pods()
        if keep_min < self.mcfg.min_pods:
            # The window just closed. Do not wait out idle_timeout_s before
            # releasing the pod that only existed to be warm — that is the whole
            # point of scheduling it — but still never kill one mid-request.
            idle_to = min(idle_to, 1) if idle_to else 1
            if not self._sched_closed_logged:
                print(f"[runpod] {self.model_key!r}: outside its warm-pod window "
                      f"({schedule_state(self.mcfg).get('window')}) — warm floor 0",
                      flush=True)
                self._sched_closed_logged = True
        elif self._release_pending:
            # A client asked for the fleet back while it was busy. Hold nothing
            # once the work drains — the same "do not wait out the timeout"
            # treatment a closed schedule window gets, for the same reason.
            idle_to = min(idle_to, 1) if idle_to else 1
        else:
            self._sched_closed_logged = False
        with self._cv:
            alive = [p for p in self.pods if p.healthy]
            for p in sorted(alive, key=lambda x: x.last_used):
                if len([q for q in self.pods if q.healthy]) <= keep_min:
                    break
                if p.inflight == 0 and idle_to > 0 and (now - p.last_used) > idle_to:
                    p.healthy = False
                    self.pods.remove(p)
                    self._bump_cards()
                    to_kill = p
                    # terminate outside the lock
                    threading.Thread(target=self._terminate, args=(to_kill, "idle"),
                                     daemon=True).start()
        # Keep min_pods warm.
        with self._cv:
            healthy_n = len([p for p in self.pods if p.healthy])
            if self._release_pending and healthy_n <= keep_min:
                self._release_pending = False          # satisfied
        if healthy_n < keep_min and not self._closed:
            # A pod that is ALREADY RUNNING for this pool beats waiting for one
            # that is booting, and beats renting a second. Tried before the
            # sibling check on purpose: after a restart the pods this deployment
            # rented a minute ago are running, registered and billing, and
            # deferring to a sibling's boot marker left them unowned -- the pool
            # empty, the account full. Adoption is the recovery path, so it must
            # not sit behind the thing it recovers from.
            #
            # In a LOOP: one adoption fills one slot, and a restart orphans the
            # whole fleet. Adopting eleven pods one scaler tick at a time took
            # six minutes where it should take one pass.
            # Bounded TWO ways, because `_adopt_shared_pod` returning True is a
            # report, not a guarantee: it can succeed without the pool gaining a
            # healthy pod (a pod registered but not yet probed, a race with the
            # reaper). `while ... and adopt()` spun forever the first time that
            # happened and took the scaler thread -- and with it the registry
            # heartbeat -- down with it.
            adopted = False
            try:
                for _ in range(max(0, keep_min - healthy_n)):
                    if healthy_n >= keep_min or not self._adopt_shared_pod():
                        break
                    adopted = True
                    with self._cv:
                        got = len([p for p in self.pods if p.healthy])
                    if got <= healthy_n:
                        print(f"[runpod] {self.model_key!r}: adoption reported "
                              f"success without adding a pod — not asking again "
                              f"this pass", flush=True)
                        break
                    healthy_n = got
            except Exception as exc:
                print(f"[runpod] maintain: adoption failed: {exc}", flush=True)
            # Adopted something? Then rent NOTHING this pass, even if the floor
            # is still short. The next pass is 15s away and will count what we
            # just took; renting beside a fresh adoption is how one job ends up
            # paying for two cards -- and a pool whose adoption reports success
            # without adding a pod must not be charged for the discrepancy.
            if healthy_n >= keep_min or adopted:
                return
            # One already on the way (from here, from ensure_ready, or from a
            # sibling engine sharing this pool) is one pod, not none: topping up
            # against a booting pod is how a warm floor of 1 became two.
            if sibling_is_provisioning(str(self.model_key)):
                return
            # Fill the WHOLE deficit, not one pod per tick. A pod needs ~5
            # minutes to have weights resident, and this used to rent exactly one
            # per pass and then BLOCK the scaler thread until it was ready, so a
            # client lease of 8 took ~40 minutes to become real -- measured on
            # 2026-10-14, 15:15 to 15:42 for four of them. Bounded by the model's
            # own `scale_up_max_parallel_boots`, which already means "how many
            # pods may be booting at once", and by max_pods.
            with self._cv:
                # Two different counts, because they answer two different
                # questions and conflating them rents cards nobody asked for.
                #
                # The DEFICIT is in pods that can serve, so it counts healthy
                # ones: an unhealthy pod does not satisfy a warm floor whose
                # whole purpose is that the next request finds a pod ready.
                #
                # The CAP is in pods that BILL, so it counts every pod in the
                # pool whatever its health, exactly as `_should_grow` does. A
                # pod that exists occupies a max_pods slot; it is charged for by
                # the minute while it boots and while it is sick.
                #
                # Counting only healthy pods against max_pods is how a pool with
                # max_pods=1 came to hold two A40s: after a restart the pool
                # ADOPTS the pod it had before, which is briefly not yet healthy
                # and was not booted by this process, so `_booting` is 0 and
                # `in_hand` computed 0 -- room under a cap of one, beside a pod
                # already running. Observed 2026-10-18: a pod adopted at 17:41,
                # a restart at 17:43, a second pod rented at 17:46, $1.18/hr for
                # a configuration that asked for $0.59.
                healthy_in_hand = len([p for p in self.pods if p.healthy]) + self._booting
                billing_in_hand = len(self.pods) + self._booting
                parallel = max(1, int(getattr(self.mcfg,
                                              "scale_up_max_parallel_boots", 4) or 4))
                want = min(keep_min - healthy_in_hand,
                           max(0, parallel - self._booting),
                           max(0, self.effective_max_pods() - billing_in_hand))
                if want <= 0:
                    return
                self._booting += want
            # Say so before spending. A warm floor silently failing to be
            # re-established is indistinguishable, from outside, from a warm
            # floor that is fine — which is how a pod terminated out of band
            # went unnoticed until the next request paid a cold boot.
            print(f"[runpod] {self.model_key!r}: warm floor {keep_min}, "
                  f"{healthy_n} pod(s) healthy — renting {want} "
                  f"(up to {parallel} may boot at once)", flush=True)
            for _ in range(want):
                # OFF the scaler thread. _provision_one blocks for minutes, and
                # this ran inline: one warm boot stalled the registry heartbeat
                # (same loop, every 60s) and every other pool's maintain() for
                # the whole boot. A stalled heartbeat is what makes a sibling
                # treat our booting pod as abandoned.
                threading.Thread(target=self._warm_provision_one, daemon=True,
                                 name=f"runpod-warm-{self.model_key}").start()

    def _warm_provision_one(self):
        """One warm-floor boot, on its own thread. Always gives `_booting` back:
        a leaked increment is a pool that can never grow again."""
        try:
            self._provision_one(cancellable=True)
        except BootCancelled:
            pass          # said its piece already, and it is not a failure
        except Exception as exc:
            print(f"[runpod] maintain: warm provision failed: {exc}", flush=True)
        finally:
            with self._cv:
                self._booting = max(0, self._booting - 1)
                self._cv.notify_all()

    def close(self):
        self._closed = True
        with self._cv:
            pods = list(self.pods)
            self.pods.clear()
        for p in pods:
            self._terminate(p, "shutdown")


# module-level pool registry + scaler ------------------------------------- #
#: Most denial timestamps a pool keeps. Only the newest `DENY_WINDOW_S` of
#: them is ever counted, so this bounds memory under a burst and nothing else.
_DENY_CAP = 4096


#: (gpu_type_id, data_center) -> when its "no capacity" verdict expires.
#:
#: RunPod's answer to "can I have this card here" is the same for every caller
#: for as long as the shortage lasts, but each boot thread re-asked
#: independently: 32 pods booting in parallel, each sweeping ~70 candidates,
#: produced 406 rejections in 120 seconds on a live run -- every GPU type
#: re-probed 33 to 72 times. That saturated a core (coderai-cpu at 60%), and
#: because CPython hands the GIL to whoever is running, it starved the event
#: loop the page forwards live on: 1,754 pages "in flight", not one socket
#: opened to a pod, and the undelivered queue made the scaler rent MORE pods.
#:
#: So the verdict is shared and short-lived. Short because capacity returns
#: without telling us, shared because asking again is what cost the run.
_unavail: dict = {}
_unavail_lock = threading.Lock()

#: How long a "no capacity" answer is trusted. Long enough to collapse a boot
#: storm's duplicate probes, short enough that a freed card is picked up on the
#: next rent rather than the next release.
UNAVAIL_TTL_S = 90.0


def _mark_unavailable(gpu_type_id: str, dc: str = "") -> None:
    if not gpu_type_id:
        return
    with _unavail_lock:
        _unavail[(str(gpu_type_id), str(dc or ""))] = time.time() + UNAVAIL_TTL_S
        if len(_unavail) > 4096:
            now = time.time()
            for k, v in list(_unavail.items()):
                if v <= now:
                    _unavail.pop(k, None)


def _is_unavailable(gpu_type_id: str, dc: str = "") -> bool:
    """Has this card already been refused here, recently, for anyone?"""
    if not gpu_type_id:
        return False
    now = time.time()
    with _unavail_lock:
        for key in ((str(gpu_type_id), str(dc or "")), (str(gpu_type_id), "")):
            until = _unavail.get(key)
            if until and until > now:
                return True
            if until:
                _unavail.pop(key, None)
    return False


def _forget_unavailable(gpu_type_id: str, dc: str = "") -> None:
    """A card that just deployed is available, whatever we thought."""
    with _unavail_lock:
        _unavail.pop((str(gpu_type_id), str(dc or "")), None)
        _unavail.pop((str(gpu_type_id), ""), None)


#: How long a READ-ONLY snapshot waits for a lock before giving up on it.
#:
#: A monitoring endpoint that blocks is worse than one that says "busy": during
#: a boot wave /v1/runpod/status stopped answering entirely, which a client
#: reasonably read as "the server is dead" when /healthz was answering in 2.9
#: ms the whole time -- and it is the one surface an operator has for seeing
#: WHY the server is busy, so it must not be the first thing a busy server
#: stops serving. Short, because a partial answer now beats a complete one
#: after the incident.
#: How long an in-flight slot may be held before the pool stops believing it.
#:
#: A safety net against leaks in the dispatch path, not a request timeout: see
#: `_reclaim_stranded_inflight`. Set well past the slowest page a pod can
#: legitimately be working on -- a full token budget plus its continuations and
#: a retry, on a card serving dozens of other pages -- because returning a slot
#: that IS in use would hand the same pod more work than its cap allows, while
#: holding a phantom a few minutes longer costs only those minutes.
INFLIGHT_TTL_S = 900.0


def _slots_of(pod) -> list:
    """A pod's in-flight timestamp list, created on demand.

    Not every object the pool is handed is a PodHandle: tests and duck-typed
    stand-ins carry only what they need, and a field added for bookkeeping
    should not force any of them to be rewritten. Initialised here rather than
    required at construction, as the per-card memo is.
    """
    slots = getattr(pod, "inflight_since", None)
    if slots is None:
        slots = []
        try:
            pod.inflight_since = slots
        except Exception:
            pass                # read-only stand-in: bookkeeping degrades, nothing breaks
    return slots

SNAPSHOT_LOCK_TIMEOUT_S = 2.0


@contextlib.contextmanager
def _lock_for_read(lock, timeout: float = None):
    """Hold `lock` if it can be had quickly; yield False if not.

    For snapshots only. A writer must never use this: skipping the lock to
    carry on would race. A reader that skips it reports less, which is the
    whole point.
    """
    want = SNAPSHOT_LOCK_TIMEOUT_S if timeout is None else float(timeout)
    got = False
    try:
        got = lock.acquire(timeout=want)
        yield got
    finally:
        if got:
            try:
                lock.release()
            except RuntimeError:
                pass


class NoPodRightNow(RuntimeError):
    """No pod is free THIS INSTANT and the caller asked not to wait here.

    Distinct from a pool that cannot serve at all: growth has been started and
    retrying shortly is the right response. Raised only for
    ``acquire(grow_inline=False)``.
    """


_pools_lock = threading.RLock()
_pools: dict = {}          # model_key -> RunpodPodPool
_scaler_started = False


#: Pod servers that are pinned to a single model at launch, so several models
#: can never share one. A coderai (or comparable) image picks the model per
#: request and can.
_SINGLE_MODEL_ENGINES = ("vllm", "llamacpp")


def canonical_model_key(model_key: str, entry: dict = None) -> str:
    """ONE pool key per set of weights, whatever name the caller used.

    A model answers to several names — its id, its HuggingFace path, any alias —
    and the pool key was simply whichever one the request happened to carry. So
    `bge-m3` and `BAAI/bge-m3` built two pools over the same weights and rented
    two A40s, one of them pure waste at $0.59/hr, with /v1/models listing both
    names and nothing anywhere saying they are the same thing.

    Keyed on the ENTRY, so every name for one model lands on one pool. A model
    with no entry keeps its name: there is nothing else to key on, and refusing
    to pool it would be worse than pooling it under the name given.
    """
    try:
        if entry is None:
            from codai.models.manager import _model_entry_for
            entry = _model_entry_for(model_key) or {}
    except Exception:
        entry = {}
    if not isinstance(entry, dict):
        entry = {}
    return (str(entry.get("id") or "").strip()
            or str(entry.get("path") or "").strip()
            or str(model_key))


def shared_pool_key(mcfg: "RunpodModelConfig", model_key: str,
                    model_path: str = "") -> str:
    """The pool key for this model: its own, or a shared one it opted into.

    Raises when the configured engine cannot serve more than one model — better
    a clear error now than a pod that answers every request with the wrong
    model's weights (or a 404 from a server that was launched for another one).
    """
    shared = (getattr(mcfg, "pool", "") or "").strip().lower()
    if not shared:
        return canonical_model_key(model_key)
    engine = resolve_pod_engine(mcfg, model_key, model_path)
    if engine in _SINGLE_MODEL_ENGINES:
        raise RuntimeError(
            f"RunPod: model {model_key!r} asks to share pod pool {shared!r}, but a "
            f"{engine} pod is launched for ONE model and cannot serve another. "
            "Share a pool only between models on a coderai (or custom multi-model) "
            "pod image, or drop the pool name so this model gets its own pods.")
    return f"pool:{shared}"


def shared_pool_seed_entries(pool_name: str, exclude_id: str = "") -> list:
    """The OTHER models that joined this named pool, ready to seed.

    A pod is seeded with the entry of whichever model happened to create the
    pool, and ``seed_entries`` is filled only for CAPABILITY pools — so the
    second model sharing a pool arrived at a pod that had never heard of it. The
    point of sharing is one pod serving several names (a base and its adapters,
    on one copy of the weights), which cannot work if only one name is known.

    Each entry's ``path`` is set to what the POD can fetch, because a seed with
    no path, or a local one, is dropped: these models are normally described by
    ``runpod.hf_repo`` and carry no path at all.
    """
    want = (pool_name or "").strip().lower()
    if not want:
        return []
    try:
        from codai.admin.routes import config_manager
        md = getattr(config_manager, "models_data", None) or {}
    except Exception:
        return []
    out = []
    for lst in md.values():
        if not isinstance(lst, list):
            continue
        for entry in lst:
            if not isinstance(entry, dict) or entry.get("enabled") is False:
                continue
            block = entry.get("runpod") if isinstance(entry.get("runpod"), dict) else {}
            if (block.get("pool") or "").strip().lower() != want:
                continue
            if str(entry.get("id") or "") == str(exclude_id or ""):
                continue
            try:
                xcfg = parse_model_runpod(block)
                kind, value = resolve_model_source(entry, xcfg)
            except Exception:
                continue
            if kind not in ("hf", "url") or not value:
                # Nothing the pod could fetch; registering it would turn a clear
                # "not available" into a load failure on a path nobody can see.
                continue
            copy = dict(entry)
            copy["path"] = value
            out.append(copy)
    return out


def get_pod_pool(model_key, account_cfg, mcfg: "RunpodModelConfig", served_name,
                 shared: bool = False, entry: dict = None) -> RunpodPodPool:
    """The pool for ``model_key``, created on first use.

    ``shared`` marks a pool several models joined by name. The first model to
    create it sets its shape (image, GPU class, budgets, scaling) and later
    joiners do NOT overwrite it — otherwise whichever model happened to load
    last would silently redefine the budget for everyone sharing the pod.
    """
    with _pools_lock:
        pool = _pools.get(model_key)
        if pool is None:
            pool = RunpodPodPool(model_key, account_cfg, mcfg, served_name, entry)
            _pools[model_key] = pool
        elif not shared:
            # refresh config each load so edits take effect
            pool.account, pool.mcfg, pool.served = account_cfg, mcfg, served_name
        if shared and not pool.seed_entries and str(model_key).startswith("pool:"):
            # Every name this pod must answer for, not just the one that rented
            # it: the adapters are registered when its engine STARTS, so a model
            # that arrives afterwards would need the engine restarted.
            pool.seed_entries = shared_pool_seed_entries(
                str(model_key)[5:], (entry or {}).get("id") or "")
            if pool.seed_entries:
                names = ", ".join(str(e.get("id")) for e in pool.seed_entries)
                print(f"[runpod] pool {model_key!r} is shared with: {names}", flush=True)
    _ensure_scaler()
    return pool


def get_model_pod_pool(model_name: str, entry: dict, block: dict):
    """A managed pod pool for ONE non-text model (an image/video/TTS model…).

    The model's own kind picks the pod image — a diffusion model cannot be served
    by vLLM — and the pod is told about the model at launch so it can serve a
    catalogue it would otherwise know nothing about. This is what lets two models
    of the same kind be placed differently: one video model local, another on its
    own pod, with its own budget.
    """
    from codai.models.manager import get_active_runpod_config
    acct = get_active_runpod_config()
    if acct is None or not getattr(acct, "enabled", False):
        raise RuntimeError(
            f"RunPod is not enabled — cannot provision a pod for {model_name!r}.")
    mcfg = parse_model_runpod(block or {})
    key = shared_pool_key(mcfg, model_name, str(entry.get("path") or ""))
    served = (mcfg.served_model or "").strip() or str(entry.get("path") or model_name)
    return get_pod_pool(key, acct, mcfg, served, shared=(key != model_name),
                        entry=entry)


def capability_seed_entries(capability: str, limit: int = 40) -> list:
    """The models a capability pod should know about.

    A pod starts with an EMPTY catalogue and refuses every model — observed
    live: a healthy embeddings pod answered "Model 'bge-m3' is not available.
    Use one of: " with nothing after the colon. A per-model pod is seeded with
    its one model; a capability pod serves whatever the capability serves, so it
    is seeded with this deployment's models OF THAT CAPABILITY.

    Only models the pod can actually fetch travel — a HuggingFace repo id or a
    URL. A local path would register a model the pod could never load, turning a
    clear "not available" into a confusing load failure.
    """
    try:
        from codai.admin.routes import config_manager
        md = getattr(config_manager, "models_data", None) or {}
    except Exception:
        return []
    out = []
    for section, lst in (md.items() if isinstance(md, dict) else []):
        if MODEL_TYPE_CAPABILITY.get(section) != capability:
            continue
        if not isinstance(lst, list):
            continue
        for entry in lst:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "")
            if not (_looks_like_repo_id(path) or path.startswith(("http://", "https://"))):
                continue                      # the pod could never fetch it
            # CARRY THE SECTION. In models.json the kind of a model is the dict
            # KEY it sits under, not a field on the entry -- and the entry is
            # all that travels. The pod's seeder reads `model_type` and falls
            # back to "text_models", so without this every seeded model was
            # registered on the pod as a TEXT model whatever it actually was.
            #
            # The symptom is not a routing error, which is what makes it
            # expensive: the pod boots, serves /v1/models, lists the model --
            # and then answers "Model 'BAAI/bge-m3' is a text model and cannot
            # be used for embedding generation. Use the text endpoint instead."
            # That reads as a configuration mistake on the caller's side.
            out.append(dict(entry, model_type=section))
            if len(out) >= limit:
                return out
    return out


def get_capability_pool(capability: str, block: dict):
    """A managed pod pool for a whole capability (images, video, tts, …).

    The pool machinery never cared that it was serving one model: it provisions,
    health-checks, scales and reaps whatever image it is given. Pointing it at a
    capability is what brings budgets, autoscaling and idle teardown to the
    subsystems that have no local engine to pin — the remote gateway asks for a
    URL here exactly the way the chat backend does.

    Defaults to `engine: coderai`, since a capability is served by a whole
    coderai on the far side rather than by an LLM server.

    A block may name a shared ``pool``: capabilities naming the same pool share
    one set of pods and one budget. That matters — a pod is a whole GPU, and one
    coderai can serve images, video and TTS from it; without this, three
    capabilities would rent three cards to do what one can.
    """
    from codai.models.manager import get_active_runpod_config
    acct = get_active_runpod_config()
    if acct is None or not getattr(acct, "enabled", False):
        raise RuntimeError(
            "RunPod is not enabled — cannot provision a pod for capability "
            f"{capability!r}. Enable it (and set an API key) or point the "
            "capability at a URL instead.")
    b = dict(block or {})
    b.setdefault("engine", "coderai")
    # A capability pod with no image uses the published one for that capability,
    # so enabling a capability is a single choice rather than an image name the
    # user has to know. A shared pool keeps whatever image it was created with.
    if not str(b.get("image") or "").strip() and b.get("engine") == "coderai":
        _img = default_capability_image(capability)
        if _img:
            b["image"] = _img
    shared = str(b.get("pool") or "").strip().lower()
    mcfg = parse_model_runpod(b)
    key = f"capability:{shared or capability}"
    pool = get_pod_pool(key, acct, mcfg, shared or capability, shared=bool(shared))
    # Tell the pod what it may serve; without this its catalogue is empty and it
    # refuses every request with "not available".
    if not getattr(pool, "seed_entries", None):
        pool.seed_entries = capability_seed_entries(capability)
    return pool


def warm_configured_pools() -> int:
    """Start the pods that are configured to stay warm, without waiting for a
    request to ask for them.

    A pool is normally created lazily on first use, which would mean the first
    request after a restart still pays the cold boot that `keep_warm` exists to
    avoid. Called at startup; returns how many pools it warmed.
    """
    warmed = 0
    try:
        from codai.admin.routes import config_manager
        md = getattr(config_manager, "models_data", None) or {}
        cfg = getattr(config_manager, "config", None)
    except Exception:
        return 0

    for section, lst in (md.items() if isinstance(md, dict) else []):
        if not isinstance(lst, list):
            continue
        for entry in lst:
            if not isinstance(entry, dict):
                continue
            block = entry.get("runpod")
            if not isinstance(block, dict) or not _as_bool(block.get("keep_warm"), False):
                continue
            # A disabled model must not rent a GPU. Nothing else consults
            # `enabled` here, so a catalogue entry parked for later with
            # keep_warm still set would have started billing at every boot.
            if entry.get("enabled") is False:
                continue
            # `id` is how a RunPod-only entry names itself: it has no local
            # `path` (there are no weights on this machine) and an `alias` only
            # when a second name is wanted. Without it such an entry warmed
            # under the key "" and the pool was never the one serving requests.
            name = entry.get("alias") or entry.get("path") or entry.get("id") or ""
            if not name:
                print("[runpod] a keep_warm entry has no id/alias/path — skipped",
                      flush=True)
                continue
            try:
                pool = get_model_pod_pool(name, entry, block)
                # ensure_ready() provisions unconditionally — it is the "a
                # request is waiting" path. At boot there is no request, so a
                # scheduled model must respect its window or a restart at 02:00
                # would rent the pod the schedule exists to avoid.
                if pool.effective_min_pods() < 1:
                    st = schedule_state(pool.mcfg)
                    print(f"[runpod] not warming {name!r} yet: outside its window "
                          f"({st.get('window')}) — a request still starts a pod",
                          flush=True)
                    continue
                pool.ensure_ready()
                warmed += 1
                print(f"[runpod] keeping a pod warm for {name!r}", flush=True)
            except Exception as exc:
                print(f"[runpod] could not warm {name!r}: {exc}", flush=True)

    remotes = getattr(cfg, "remotes", None)
    pods = getattr(remotes, "pods", None) if remotes is not None else None
    for cap, block in (pods.items() if isinstance(pods, dict) else []):
        if not isinstance(block, dict) or not _as_bool(block.get("keep_warm"), False):
            continue
        try:
            pool = get_capability_pool(str(cap), block)
            pool.ensure_ready()
            warmed += 1
            print(f"[runpod] keeping a pod warm for capability {cap!r}", flush=True)
        except Exception as exc:
            print(f"[runpod] could not warm capability {cap!r}: {exc}", flush=True)
    return warmed


def _all_pools_hourly_rate() -> float:
    with _pools_lock:
        return sum(p.hourly_rate() for p in _pools.values())


def _all_pools_live_cost() -> float:
    with _pools_lock:
        return sum(p.live_cost_usd() for p in _pools.values())


def fleet_summary(pods: list = None) -> dict:
    """"How many, where, which model" — the questions asked of a running fleet.

    Derived from pods_status() so there is one source of truth. Counts only
    pods this install rented: one created by hand in the RunPod console is not
    in the pools and is deliberately not counted here.
    """
    rows = pods_status() if pods is None else pods
    ready = [p for p in rows if p.get("state") == "ready" and p.get("healthy")]
    booting = [p for p in rows if p.get("state") != "ready"]

    def _tally(key, src):
        out = {}
        for r in src:
            k = str(r.get(key) or "") or "unknown"
            out[k] = out.get(k, 0) + 1
        return out

    return {
        "pods_total": len(rows),
        "pods_ready": len(ready),
        "pods_booting": len(booting),
        "pods_unhealthy": len([p for p in rows
                               if p.get("state") == "ready" and not p.get("healthy")]),
        "gpus_in_use": sum(int(p.get("gpu_count") or 1) for p in ready),
        "inflight": sum(int(p.get("inflight") or 0) for p in rows),
        "by_model": _tally("model", rows),
        "by_region": _tally("data_center", rows),
        "by_gpu": _tally("gpu", rows),
        "by_cloud_type": _tally("cloud_type", rows),
        "spot_pods": len([p for p in rows if p.get("is_spot")]),
        "hourly_usd": round(sum((p.get("hourly_usd") or 0.0) for p in ready), 4),
    }


def pools_schedule_status() -> list:
    """Per-pool warm-pod schedule: where each model stands right now.

    This is the monitoring surface for the schedule — what is warm, what the
    window is, whether we are inside it and when that next flips.

    Every lock here is taken with a timeout, exactly as `pods_status` does.
    This function was the half of the /v1/runpod/status snapshot that had NOT
    been bounded, so the endpoint could still hang on a pool whose condvar was
    busy however carefully the other half gave up -- and a monitoring endpoint
    that hangs during a boot wave is indistinguishable from a dead server.
    A pool that cannot be read contributes a row saying so.
    """
    out = []
    with _lock_for_read(_pools_lock) as got:
        pools = list(_pools.items()) if got else []
    if not pools:
        return [{"model": "", "enabled": False, "busy": True,
                 "note": "the pool registry was locked; this snapshot is "
                         "incomplete (a provisioning wave holds it)"}]
    for key, pool in pools:
        cfg = pool.mcfg
        st = schedule_state(cfg)
        with _lock_for_read(pool._cv) as got:
            if not got:
                out.append({"model": key, "enabled": st["enabled"],
                            "busy": True,
                            "note": f"pool {key!r} was locked when this "
                                    f"snapshot was taken"})
                continue
            healthy = len([p for p in pool.pods if p.healthy])
        out.append({
            "model": key,
            "enabled": st["enabled"],
            "in_window": st["in_window"],
            "window": st["window"],
            "days": st["days"],
            "tz": st["tz"] or "server local",
            "next_change": st["next_change"],
            "min_pods": cfg.min_pods,
            "effective_min_pods": pool.effective_min_pods(),
            "max_pods": cfg.max_pods,
            # The ceiling actually in force, and where it came from: a fleet
            # that stopped growing short of max_pods must say why.
            "effective_max_pods": pool.effective_max_pods(),
            "client_max_pods": pool.client_cap(),
            "healthy_pods": healthy,
            "data_center": cfg.data_center or "",
            # The client lease, so a bill nobody expected has a visible cause.
            "client_scale_allowed": bool(getattr(cfg, "allow_client_scale", False)),
            "client_pods": pool.client_floor(),
            "client_lease_expires_in_s": max(
                0, int(pool._client_floor_until - time.time())) if pool.client_floor() else 0,
            # What the lease narrowed the card choice to, if anything: a pod on
            # an unexpected card should have a visible cause too, not only a
            # bill.
            "client_gpu_narrow": pool.client_gpu_narrow(),
            # Which regions this pool may use and the volume each one attaches:
            # a slow region is only attributable if its storage is visible.
            # getattr: a status report must never be the thing that raises.
            "network_volumes": volume_map_for(getattr(pool, "mcfg", None),
                                              getattr(pool, "account", None)),
        })
    return out


def seed_region_volumes(model: str, regions=None) -> dict:
    """Pre-fill this model's regional volumes, one region at a time.

    An empty volume costs the first pod in that region the whole download —
    the cost the volume was added to remove. Seeding pays it once, on purpose,
    instead of making a user's first request wait for it.

    Sequential by design: N regions at once means N cards billing for the same
    minutes, and the point of the operation is that nobody is waiting on it.
    Blocking and minutes per region, so an HTTP caller should run it in the
    background and read the result from the log.
    """
    key = str(model or "").strip()
    if not key:
        raise LookupError("no model named")
    block = model_runpod_block(key)
    if not block:
        raise LookupError(f"{key!r} has no runpod block")
    mcfg = parse_model_runpod(block)
    if mcfg.is_serverless:
        raise PermissionError(
            f"{key!r} runs serverless — it has no volume of its own to seed")
    vmap = volume_map_for(mcfg, _account_hint())
    if not vmap:
        raise LookupError(
            f"{key!r} has no network_volumes map — a single network_volume_id "
            "needs no seeding across regions, since it only has one")
    want = [str(r).strip().upper() for r in (regions or list(vmap)) if str(r).strip()]
    unknown = [r for r in want if r not in vmap]
    if unknown:
        raise ValueError(
            f"no volume configured for {', '.join(unknown)} — network_volumes "
            f"has {', '.join(sorted(vmap))}")
    pool = get_model_pod_pool(key, _model_entry(key), block)
    out = {"model": key, "regions": [], "failed": []}
    for dc in want:
        print(f"[runpod] seeding {key!r} onto the volume in {dc}", flush=True)
        try:
            out["regions"].append(pool.seed_region(dc))
        except Exception as exc:
            # One region's capacity miss must not cost the others their turn.
            print(f"[runpod] seeding {dc} failed: {exc}", flush=True)
            out["failed"].append({"region": dc, "error": str(exc)})
    return out


def get_live_pool(key: str):
    """A pool that is running under exactly this key, or None.

    Matched on the registry key, which is what `pods_status` reports and
    therefore the only name a client can have seen.
    """
    k = str(key or "").strip()
    if not k:
        return None
    with _pools_lock:
        got = _pools.get(k)
        if got is not None:
            return got
        # An ALIAS of a pooled model. `status` reports the canonical key, but a
        # client holds whichever name it has been using all along, and being
        # told a running pool does not exist is how a pod ends up with no way
        # to turn it off.
        alt = canonical_model_key(k)
        return _pools.get(alt) if alt != k else None


def _scale_live_pool(pool, key: str, pods: int, teardown: bool = False,
                     gpu_types=None, allow_other_gpus=None,
                     max_pods=None) -> dict:
    """`set_client_pods` for a pool addressed directly rather than by model."""
    mcfg = getattr(pool, "mcfg", None)
    if mcfg is not None and not getattr(mcfg, "allow_client_scale", False):
        raise PermissionError(
            f"the pool {key!r} does not allow client scaling — set "
            '"allow_client_scale": true on the runpod block of a model that '
            "uses it")
    out = pool.set_client_floor(pods, gpu_types=gpu_types,
                                allow_other_gpus=allow_other_gpus,
                                max_pods=max_pods)
    if teardown:
        out["teardown"] = pool.shed_now()
    out["pool"] = out.get("model") or key
    out["model"] = key
    return out


def _already_gone(exc: BaseException) -> bool:
    """Does this error mean the pod does not exist any more?

    RunPod says so several ways depending on which API answered and how long
    ago the pod went, so the shapes are matched rather than one code. A false
    positive here reports a live pod as gone -- which the reconcile sweep then
    corrects from the account listing -- while a false negative answers 500 on
    a teardown that has in fact succeeded, which is the failure this exists to
    stop.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if "404" in text or "not found" in text:
        return True
    return any(h in text for h in (
        "does not exist", "no such pod", "already terminated",
        "already been terminated", "pod not found", "unknown pod",
        "invalid pod id", "has been removed"))


def release_pod(pod_id: str, reason: str = "released by client") -> dict:
    """Terminate ONE pod by id, wherever it is, and stop tracking it.

    The gap this fills: every other path here scales a POOL. A pod adopted from
    another process, a pod left behind by a crash, a pod whose pool was
    reconfigured underneath it -- none of those belong to a floor anyone can
    lower, and the only way to stop them billing was the RunPod console.

    Terminates whatever the account is actually running under that id, even
    when no pool here knows it: a pod this process cannot see is precisely the
    one that needs the button. Refuses a pod with requests on it unless
    ``force``-ed by the caller draining it first -- see `shed_now` for the
    graceful path.
    """
    pid = str(pod_id or "").strip()
    if not pid:
        raise LookupError("no pod id given")
    from codai.models.manager import get_active_runpod_config
    acct = get_active_runpod_config()
    if acct is None or not getattr(acct, "enabled", False):
        raise RuntimeError("RunPod is not enabled")
    out = {"pod_id": pid, "found_in_pool": None, "inflight": 0,
           "terminated": False}
    cost, gpu, spot = 0.0, "", False
    # Drop it from its pool FIRST, so nothing routes a new request onto a pod
    # that is about to go away.
    with _pools_lock:
        pools = list(_pools.items())
    for key, pool in pools:
        with pool._cv:
            victim = next((p for p in pool.pods if p.pod_id == pid), None)
            if victim is None:
                continue
            out["found_in_pool"] = key
            out["inflight"] = int(victim.inflight)
            # Attributed before it is dropped, or the money lands nowhere --
            # which is exactly how an untracked pod becomes an unexplained bill.
            cost = ((time.time() - victim.started_at) / 3600.0
                    * float(victim.hourly_usd or 0.0))
            gpu, spot = str(victim.gpu or ""), bool(victim.is_spot)
            victim.healthy = False
            pool.pods.remove(victim)
            pool._cv.notify_all()
        break
    from codai.api.runpod_client import RunpodClient
    try:
        RunpodClient(acct).terminate_pod(pid)
        out["terminated"] = True
    except Exception as exc:
        # ALREADY GONE IS THE GOAL, NOT A FAILURE. A client tearing down a
        # fleet releases pods by id, and by the time it reaches the last one
        # some are already gone -- reaped for idle, shed by a scale-down,
        # released by a retry of this same call. Raising for those answered 500
        # per pod, so a teardown that was working looked like a teardown that
        # had failed, and the operator was told the pods could not be stopped.
        # (Observed: five 500s in three seconds on a client's own release
        # sweep.) Terminating something that does not exist has nothing left to
        # do, so it is success, said plainly.
        if _already_gone(exc):
            out["terminated"] = False
            out["already_gone"] = True
            out["detail"] = (f"pod {pid} no longer exists on the account; "
                             f"nothing to terminate")
            print(f"[runpod] pod {pid} was already gone ({reason}) — release "
                  f"had nothing to do", flush=True)
        else:
            # Say which half happened. A pod dropped from the pool but still
            # billing is a worse state than either end, and the caller can only
            # act on it if it is told.
            out["error"] = str(exc)
            if out["found_in_pool"]:
                out["warning"] = (
                    f"pod {pid} was removed from pool {out['found_in_pool']!r} but "
                    f"RunPod did not confirm the termination — it may still be "
                    f"billing. Check the console.")
            raise RuntimeError(out.get("warning")
                               or f"could not terminate {pid}: {exc}")
    finally:
        with _pools_lock:
            _provisioning_ids.discard(pid)
            _provisioning_info.pop(pid, None)
        try:
            unregister_pod(pid)
        except Exception:
            pass
    if out["found_in_pool"]:
        try:
            from codai.api import runpod_ledger
            runpod_ledger.record_spend(out["found_in_pool"], cost,
                                       {"pod_id": pid, "gpu": gpu,
                                        "is_spot": spot, "reason": reason})
            out["billed_usd"] = round(cost, 4)
        except Exception:
            pass
    print(f"[runpod] pod {pid} terminated on request ({reason})"
          + (f" — it was serving {out['inflight']} request(s) in "
             f"{out['found_in_pool']!r}" if out["inflight"] else ""), flush=True)
    return out


def set_client_pods(model: str, pods: int, teardown: bool = False,
                    gpu_types=None, allow_other_gpus=None,
                    max_pods=None) -> dict:
    """Lease a warm floor for one model on a client's behalf.

    Raises LookupError when the model has no RunPod block, and PermissionError
    when it has one that did not opt in — told apart so the caller can answer
    404 and 403, which are different problems for whoever is integrating.

    ``teardown`` sheds the surplus straight away instead of leaving it to the
    scale-down window: the client that just finished a batch knows the work has
    stopped, which the autoscaler can only infer slowly. It cannot go below the
    operator's warm floor, and it drains rather than dropping work.

    ``gpu_types`` and ``allow_other_gpus`` constrain the cards this lease will
    accept, and may only NARROW what the operator configured -- see
    `narrow_gpu_choice`. They are the answer to a real need: a caller measuring
    throughput knows that a number taken on a different card is not comparable
    with yesterday's, and would rather wait for the right one than pay a
    fallback's price for a result it cannot use. Raises ValueError when the
    narrowing leaves nothing allowed, so the caller hears about a mistake
    instead of silently getting the operator's default.

    ``max_pods`` caps the fleet for the life of the lease, and may only narrow
    the operator's own ceiling. A warm floor says what to hold ready; it says
    nothing about how large the pool may become, so without this a client
    running a staged batch could ask for its first stage and still be grown to
    the configured maximum by the autoscaler.
    """
    key = str(model or "").strip()
    if not key:
        raise LookupError("no model named")
    # A LIVE POOL, addressed by the key this very function reports back as
    # `pool`. Several models share one pool, so `pool:digesta-qwen38` is not a
    # model and has no runpod block of its own -- and until now the only answer
    # to it was "has no runpod block", from code that already knew the caller
    # could not look the key up. A pool that exists can be addressed; its
    # permission and its clamp come from the config it was built with.
    existing = get_live_pool(key)
    if existing is not None:
        return _scale_live_pool(existing, key, pods, teardown=teardown,
                                gpu_types=gpu_types,
                                allow_other_gpus=allow_other_gpus,
                                max_pods=max_pods)
    block = model_runpod_block(key)
    if not block:
        raise LookupError(
            f"{key!r} has no runpod block, and no pod pool is running under "
            f"that name either. Pass a model id, or a pool key that "
            f"/v1/runpod/status is currently reporting.")
    mcfg = parse_model_runpod(block)
    if not mcfg.allow_client_scale:
        raise PermissionError(
            f"{key!r} does not allow client scaling — set "
            '"allow_client_scale": true on its runpod block')
    if mcfg.is_serverless:
        raise PermissionError(
            f"{key!r} runs serverless — RunPod owns its scaling; set "
            "min_workers/max_workers on the endpoint instead")
    # Creating the pool is the point: asking for a warm floor on a model nobody
    # has called yet is exactly when it is worth pre-warming.
    pool = get_model_pod_pool(key, _model_entry(key), block)
    out = pool.set_client_floor(pods, gpu_types=gpu_types,
                                allow_other_gpus=allow_other_gpus,
                                max_pods=max_pods)
    if teardown:
        # After the floor is lowered, so the shed measures against the new one.
        out["teardown"] = pool.shed_now()
    # Answer with the name the client asked about, not the internal pool key:
    # several models may share one pool, and "pool:digesta-qwen38" is not
    # something the caller sent or can look up.
    out["pool"] = out["model"]
    out["model"] = key
    return out


def pods_status() -> list:
    """Snapshot of all pods across pools for the stats/tasks pages.

    Every lock here is taken with a timeout: see `_lock_for_read`. A pool whose
    lock is busy contributes one `state: "busy"` row naming itself instead of
    hanging the whole snapshot, so the answer is always late-but-complete about
    what it could see and explicit about what it could not.
    """
    now = time.time()
    out = []
    with _lock_for_read(_pools_lock) as got:
        pools = list(_pools.items()) if got else []
    if not pools:
        out.append({"model": "", "pod_id": "", "gpu": "", "healthy": False,
                    "inflight": 0, "state": "busy",
                    "note": "the pool registry was locked; this snapshot is "
                            "incomplete (a provisioning wave holds it)"})
        return out
    for key, pool in pools:
        with _lock_for_read(pool._cv) as got:
            if not got:
                out.append({"model": key, "pod_id": "", "gpu": "",
                            "healthy": False, "inflight": 0, "state": "busy",
                            "note": f"pool {key!r} was locked when this "
                                    f"snapshot was taken"})
                continue
            for p in pool.pods:
                out.append({
                    "model": key, "pod_id": p.pod_id, "gpu": p.gpu,
                    "gpu_count": getattr(p, "gpu_count", 1),
                    "is_spot": p.is_spot, "healthy": p.healthy, "inflight": p.inflight,
                    "hourly_usd": p.hourly_usd, "uptime_s": int(now - p.started_at),
                    "live_cost_usd": round((now - p.started_at) / 3600.0 * p.hourly_usd, 4),
                    "console_url": p.console_url, "state": "ready",
                    "data_center": getattr(p, "data_center", "") or "",
                    "cloud_type": getattr(p, "cloud_type", "") or "",
                    "pool": (pool.mcfg.pool or "") if getattr(pool, "mcfg", None) else "",
                    "engine": (pool.mcfg.engine or "") if getattr(pool, "mcfg", None) else "",
                })
    # Booting pods (created, not yet serving) — visible so a slow/failing boot is
    # obvious and its vLLM log is one click away.
    with _lock_for_read(_pools_lock) as got:
        prov = list(_provisioning_info.items()) if got else []
    for pid, info in prov:
        out.append({
            "model": info.get("model"), "pod_id": pid, "gpu": info.get("gpu"),
            "is_spot": info.get("is_spot"), "healthy": False, "inflight": 0,
            "hourly_usd": info.get("hourly_usd"),
            "uptime_s": int(now - (info.get("started_at") or now)),
            "live_cost_usd": round((now - (info.get("started_at") or now)) / 3600.0
                                   * (info.get("hourly_usd") or 0), 4),
            "console_url": info.get("console_url"), "state": "provisioning",
        })
    return out


#: Logged once per process, so two processes disagreeing about where the
#: registry lives shows up in the log instead of as a terminated pod.
_REGISTRY_PATH_LOGGED = False

#: Pods this DEPLOYMENT owns, shared across processes.
#:
#: coderai runs a front plus one engine per GPU. Each engine has its own pool
#: registry AND its own reaper, so a process-local view of "our pods" means every
#: engine sees its siblings' pods as orphans. Observed live: the nvidia engine
#: created a pod and the radeon engine terminated it four seconds later, then the
#: reverse — a mutual kill loop that rented and destroyed pods until stopped.
def _pod_registry_path() -> str:
    """The one file every process must agree on.

    It used to fall back to ``~/.coderai`` when ``config_manager`` was not
    initialised in the calling process — and the container's launcher puts the
    real config dir at ``$CODERAI_CONFIG_DIR/coderai`` (it exports
    XDG_CONFIG_HOME, so ``user_config_dir()`` lands there too). So a process
    that had the config manager wrote one registry and a process that did not
    read ANOTHER, one directory up: the pods one owned were orphans to the
    other.

    That is what made the documented kill loop recur on the Digesta
    orchestrator: a pod created at 19:24:56 was terminated as untracked at
    19:24:57. The fallback is now the same XDG-aware path the launcher computes,
    so the two cannot diverge; the resolved path is logged once, because a
    registry that silently splits is invisible until it costs a GPU.
    """
    base = ""
    try:
        from codai.admin.routes import config_manager
        base = str(getattr(config_manager, "config_dir", "") or "")
    except Exception:
        base = ""
    if not base:
        try:
            from codai.platform_paths import (deployment_config_dir,
                                              user_config_dir)
            # The deployment's own answer first: XDG_CONFIG_HOME is exported by
            # the launcher and so exists only in the processes it started.
            dep = deployment_config_dir()
            base = str(dep) if dep else str(user_config_dir())
        except Exception:
            base = os.path.expanduser("~/.coderai")
    path = os.path.join(base, "runpod-pods.json")
    global _REGISTRY_PATH_LOGGED
    if not _REGISTRY_PATH_LOGGED:
        _REGISTRY_PATH_LOGGED = True
        print(f"[runpod] pod registry: {path}", flush=True)
    return path


def _registry_update(fn):
    """Read-modify-write the shared registry under an exclusive lock."""
    import fcntl
    import json as _json
    path = _pod_registry_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.seek(0)
            raw = fh.read().strip()
            try:
                data = _json.loads(raw) if raw else {}
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            if fn(data):
                fh.seek(0)
                fh.truncate()
                _json.dump(data, fh)
                fh.flush()
            return data
    except Exception as exc:
        print(f"[runpod] pod registry unavailable ({exc}) — falling back to this "
              "process's view", flush=True)
        return {}


#: This process, distinctly. os.getpid() is NOT enough: inside a container the
#: engine comes up as the same pid on every boot (74, here), so a pod registered
#: by the PREVIOUS container read as "our own" and was skipped — the restart
#: rented another A40 beside it, every time. Observed three times in a row.
_PROC_ID = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"


def _is_this_process(info: dict) -> bool:
    """Whether a registry entry was written by THIS process.

    Falls back to the pid for an entry written before the nonce existed, which
    is the old behaviour and no worse than it was.
    """
    proc = str(info.get("proc") or "")
    if proc:
        return proc == _PROC_ID
    return info.get("pid") == os.getpid()


def register_pod(pod_id: str, pool_key: str = "", url: str = "",
                 api_key: str = "", hourly_usd: float = 0.0,
                 image: str = "", gpu: str = "",
                 vram_gb: float = 0.0, gpu_count: int = 0) -> None:
    """Record a pod as owned by this deployment.

    Two reasons, both about processes that cannot see each other: no sibling's
    reaper may treat it as an orphan, and a sibling that needs the SAME pool can
    reuse this pod instead of renting a second GPU for the same work.

    The bearer token goes in too. A pool that was not given one generates a
    random token and launches its pods with it, so a pod that outlives the
    process that rented it — a restart, a crash — is adopted by the next pool
    with a DIFFERENT token and answers 401 to every request for the rest of its
    paid life. Observed live: a surviving pod reported "serves: (HTTP 401)".
    """
    if not pod_id:
        return

    def _write(d):
        # An ADOPTION re-registers a pod that already exists, so "created" must
        # survive the new owner: taking time.time() again would reset the age
        # the reaper's ledger attribution is computed from, and a pod adopted
        # every hour would look newly born every hour.
        was = d.get(str(pod_id))
        born = float((was or {}).get("created") or 0.0) if isinstance(was, dict) else 0.0
        d[str(pod_id)] = {"pool": str(pool_key), "pid": os.getpid(),
                          "proc": _PROC_ID, "at": time.time(),
                          # Distinct from "at", which is a heartbeat: the
                          # hours a reaped pod billed can only be computed from
                          # when it was CREATED, and that must survive a new
                          # owner.
                          "created": born or time.time(),
                          "url": str(url or ""), "key": str(api_key or ""),
                          # What it is RUNNING and what it is running ON.
                          # Adoption is only sound when both match what this
                          # pool would launch: the same model is not the same
                          # image (a pinned tag beside :latest serves different
                          # code, invisibly), and inheriting a fallback card
                          # would quietly break a lease that asked for one card
                          # and nothing else. Carried over on re-registration
                          # so an adoption does not erase them.
                          "image": str(image or (was or {}).get("image") or ""),
                          "gpu": str(gpu or (was or {}).get("gpu") or ""),
                          # ...and HOW BIG that card is, so a pod adopted after
                          # a restart can be given a share of the load matching
                          # its card without asking the account again. Carried
                          # over for the same reason the two above are.
                          "vram_gb": float(vram_gb or (was or {}).get("vram_gb") or 0.0),
                          "gpu_count": int(gpu_count or (was or {}).get("gpu_count") or 1),
                          # So a pod adopted after a restart is still attributed
                          # its real $/hr instead of looking free. Carried over
                          # like image/gpu: a re-registration that does not know
                          # the rate must not overwrite one that did, or an
                          # adoption turns a billing pod free.
                          "hourly_usd": float(hourly_usd
                                              or (was or {}).get("hourly_usd")
                                              or 0.0)}
        return True

    _registry_update(_write)


def touch_registered_pods() -> int:
    """Refresh this process's registry entries. Returns how many were touched.

    ``at`` is what tells a sibling's reaper that an entry still has an owner.
    Without a heartbeat it only ever said when the pod was REGISTERED, so an
    entry outlived its process forever — and because the reaper skips anything
    "known", the pod it named was shielded from reaping for the life of the
    install while no pool was managing it either. Exactly what happened to a
    qwen38 pod across an unattended upgrade: its pool had a warm floor of 0
    (outside its schedule window), so nothing adopted it, nothing applied
    ``idle_timeout_s`` to it, and the reaper left it alone because the dead
    process's entry still vouched for it. It billed on.

    So: a live owner says so every pass, and silence past REAP_GRACE_SECONDS
    lets the pod be reaped like any other leftover.
    """
    touched = [0]

    def _fn(d):
        now = time.time()
        for info in d.values():
            if isinstance(info, dict) and _is_this_process(info):
                info["at"] = now
                touched[0] += 1
        return bool(touched[0])

    _registry_update(_fn)
    return touched[0]


#: How long a URL-less registry entry may go un-heartbeated before it stops
#: meaning "a sibling is booting this". The heartbeat runs every 60s, so three
#: missed beats is a dead owner -- not a slow one.
#:
#: This used to be REAP_GRACE_SECONDS (1200s), which conflated two unrelated
#: questions: "is a sibling still booting?" (answered by the heartbeat) and "how
#: long before the reaper may kill an unowned pod?" (deliberately generous). The
#: cost was paid live on 2026-10-14: a deploy restarted coderai mid-boot, the
#: previous incarnation's URL-less entry looked like a live sibling for twenty
#: minutes, and because this check runs BEFORE adoption the pool could neither
#: rent nor adopt -- six pods billing on the account and none in the interface.
PROVISION_STALE_SECONDS = 180.0


def sibling_is_provisioning(pool_key: str) -> bool:
    """True when another LIVE process has a pod for this pool that is booting.

    Registered at creation with no URL yet. Without this the sibling rents its
    own during the ~2.5 minutes a pod takes to boot — which is exactly what
    happened live: two pods for one capability, 45 seconds apart.
    """
    data = _registry_update(lambda d: False)
    if not isinstance(data, dict):
        return False
    now = time.time()
    for info in data.values():
        if not isinstance(info, dict) or info.get("pool") != str(pool_key):
            continue
        if _is_this_process(info) or info.get("url"):
            continue
        # Only while its owner is still TALKING. A heartbeat that stopped means
        # the process that was booting this pod is gone, and waiting on it is
        # waiting forever -- while the pod it rented bills.
        if now - float(info.get("at") or 0) < PROVISION_STALE_SECONDS:
            return True
    return False


def _gpu_name_matches(name: str, want: str) -> bool:
    """Whether a recorded GPU name is the card ``want`` names."""
    return _gpu_matches({"id": name, "display_name": name}, want)


def find_shared_pod(pool_key: str, health_path: str = "/v1/models",
                    api_key: str = "", image: str = "",
                    gpus: Optional[list] = None) -> tuple:
    """A pod already running for this pool: (pod_id, url, key, hourly_usd).

    ``key`` is the token that pod was launched with, which is not necessarily
    ours — see register_pod. Empty when unknown (an entry from before tokens were
    recorded), in which case the caller's own token is tried.

    coderai runs one engine per GPU, and each has its own pools — so the same
    capability was rented a pod PER ENGINE, paying twice for one job while
    max_pods said 1. Checked against the shared registry and health-probed, so a
    dead entry is never handed out.

    It also covers OUR OWN pod from before a restart: the pid no longer matches,
    so the entry is adoptable. That matters most for a warm floor, which is
    re-established at every boot — an hourly unattended upgrade would otherwise
    rent a fresh A40 each time and leave the last one billing until the reaper's
    grace ran out.

    Serving the same model is not enough to inherit a pod, so ``image`` and
    ``gpus`` are checked when given:

    * **the image** must be the one this pool would launch. The same model on a
      pinned tag and on ``:latest`` is different CODE, and adopting the wrong
      one is invisible from outside — a whole evening went into chasing an OCR
      result that came from a pod nobody believed was running;
    * **the card** must be one the pool prefers (``gpu_types``, as narrowed by a
      client lease). A pod rented as a FALLBACK is a card the configuration
      accepts only when the preferred one is unavailable; inheriting it would
      quietly serve a lease that asked for one card and nothing else on another.

    An entry that records neither is refused rather than trusted: it was written
    before this check existed, and the cost of refusing is one extra pod once
    (the old one is reaped), while the cost of trusting is a silent answer from
    the wrong code.
    """
    data = _registry_update(lambda d: False)
    if not isinstance(data, dict):
        return None, "", "", 0.0
    for pod_id, info in data.items():
        if not isinstance(info, dict) or info.get("pool") != str(pool_key):
            continue
        if _is_this_process(info):
            continue                       # our own; the local pool handles it
        url = str(info.get("url") or "")
        if not url:
            # A pod that was still BOOTING when its owner died. register_pod
            # writes the url only once the pod is ready, so a restart landing
            # inside a boot — and these images take 6-14 minutes to boot — left
            # an entry nothing could ever adopt: skipped here in silence, then
            # billed until the reaper's grace expired. The pod id is enough to
            # ask RunPod where it is.
            url = _recover_pod_url(pod_id, info)
            if not url:
                print(f"[runpod] not adopting pod {pod_id}: it has no address "
                      f"recorded and RunPod does not give one yet (it was "
                      f"still booting when its owner stopped)", flush=True)
                continue
            print(f"[runpod] pod {pod_id} was mid-boot when its owner stopped; "
                  f"recovered its address from RunPod: {url}", flush=True)
        if image:
            theirs = str(info.get("image") or "")
            if theirs != str(image):
                print(f"[runpod] not adopting pod {pod_id}: it runs "
                      f"{theirs or '(an unrecorded image)'}, this pool runs "
                      f"{image}", flush=True)
                continue
        if gpus:
            card = str(info.get("gpu") or "")
            if not card or not any(_gpu_name_matches(card, w) for w in gpus):
                print(f"[runpod] not adopting pod {pod_id}: it runs on "
                      f"{card or '(an unrecorded card)'}, this pool asks for "
                      f"{', '.join(gpus)}", flush=True)
                continue
        key = str(info.get("key") or "") or api_key
        if _pod_health_ok(url, path=health_path, api_key=key):
            return pod_id, url, key, float(info.get("hourly_usd") or 0.0)
        # The last silent refusal. A pod that does not answer may be dead, may
        # be mid-warm-up, or may want a token we do not have — and renting
        # beside it looks identical from the log either way.
        print(f"[runpod] not adopting pod {pod_id}: it did not answer "
              f"{health_path} at {url} (dead, still loading, or it wants a "
              f"token this pool does not have)", flush=True)
    return None, "", "", 0.0


def _recover_pod_url(pod_id: str, info: dict) -> str:
    """Where a registered pod is now, asked of RunPod rather than remembered.

    Only for an entry with no url: the owner died between creating the pod and
    it becoming ready. Best effort — a pod that still has no port gives nothing
    and the caller says so.
    """
    try:
        from codai.models.manager import get_active_runpod_config
        from codai.api.runpod_client import RunpodClient, direct_tcp_url, pod_proxy_url
        acct = get_active_runpod_config()
        if acct is None or not getattr(acct, "api_key", ""):
            return ""
        live = RunpodClient(acct).get_pod(str(pod_id))
    except Exception as exc:
        print(f"[runpod] could not ask RunPod where pod {pod_id} is: {exc}",
              flush=True)
        return ""
    if str(live.get("status") or "").upper() in ("TERMINATED", "EXITED", "FAILED"):
        return ""
    ports = live.get("ports") or []
    if not ports:
        return ""
    # The same two shapes _provision_one chooses between, decided the same way:
    # a recorded direct-TCP address is https to a bare IP.
    try:
        port = int(info.get("port") or 0) or 8000
    except Exception:
        port = 8000
    direct = direct_tcp_url(ports, port, tls=True)
    return direct or pod_proxy_url(str(pod_id), port)


def unregister_pod(pod_id: str) -> None:
    """Forget a pod we terminated, so the registry does not grow without bound."""
    if not pod_id:
        return
    _registry_update(lambda d: d.pop(str(pod_id), None) is not None)


def registered_pod_ids() -> set:
    data = _registry_update(lambda d: False)
    return set(data.keys()) if isinstance(data, dict) else set()


def live_registered_pod_ids() -> set:
    """Registered pods whose owner is still around to manage them.

    Ours count unconditionally. A sibling's count while its heartbeat is fresh.
    Everything else is a pod with no manager, which is the reaper's job.
    """
    data = _registry_update(lambda d: False)
    if not isinstance(data, dict):
        return set()
    now, out = time.time(), set()
    for pod_id, info in data.items():
        if not isinstance(info, dict):
            continue
        if _is_this_process(info) or (now - float(info.get("at") or 0)
                                      < REAP_GRACE_SECONDS):
            out.add(pod_id)
    return out


def registry_entry(pod_id: str) -> dict:
    """One registry entry, or {}."""
    data = _registry_update(lambda d: False)
    info = (data or {}).get(str(pod_id))
    return dict(info) if isinstance(info, dict) else {}


#: Pod ids a successful listing did not mention, from the PREVIOUS sweep. A pod
#: has to be missing from two listings in a row before the pool gives up on it:
#: the RunPod API is eventually consistent, a partial page or a brief backend
#: hiccup can omit a pod that is running perfectly well, and the cost of
#: believing one such listing is a second rented GPU. Guarded by _pools_lock.
_absent_before: set = set()

#: Statuses that are RunPod telling us, in so many words, that the machine is
#: finished. No second opinion needed for these.
_DEAD_STATUSES = ("TERMINATED", "EXITED", "DEAD", "FAILED")


def close_bill_from_registry(pod_id: str, reason: str) -> float:
    """Ledger the hours of a pod NO POOL was holding. Returns what was recorded.

    A pool closes its own pods' bills — ``_terminate`` for one it retires,
    ``forget`` for one RunPod says is gone. Neither runs for a pod that only the
    registry knew about: one orphaned by a restart, or killed from the console or
    straight through the RunPod API. Those hours were billed and went into the
    ledger as nothing at all, which is the one place they must appear: a cost
    that is invisible cannot be budgeted against, and the surprise always
    arrives later, on the invoice.

    Computed from the registry's own ``created`` — the only record of when the
    pod started once no handle exists — so it survives the owner that rented it.
    """
    info = registry_entry(pod_id)
    rate = float(info.get("hourly_usd") or 0.0)
    began = float(info.get("created") or info.get("at") or 0.0)
    if not (rate and began):
        print(f"[runpod] cannot attribute {pod_id}'s spend: the registry records "
              f"{'no rate' if not rate else 'no start time'} for it", flush=True)
        return 0.0
    cost = (time.time() - began) / 3600.0 * rate
    try:
        from codai.api import runpod_ledger
        runpod_ledger.record_spend(
            str(info.get("pool") or "(orphan)"), cost,
            {"pod_id": str(pod_id), "gpu": str(info.get("gpu") or "(orphan)"),
             "reaped": True, "reason": reason})
    except Exception as exc:
        print(f"[runpod] could not write {pod_id}'s spend to the ledger: {exc}",
              flush=True)
        return 0.0
    print(f"[runpod] pod {pod_id} ({reason}) billed ~${cost:.3f} — recorded "
          f"against {info.get('pool') or '(orphan)'}", flush=True)
    return cost


def reconcile_pools(live_status: dict, candidates=None) -> set:
    """Reconcile pools (and the registry) against what RunPod actually has.

    Returns the set of pod ids now CONFIRMED gone — the caller unregisters
    those; any that were in a pool have already been forgotten here.

    The bug this exists for: ``keep_warm`` would not re-rent after a pod was
    terminated out of band, and nothing was logged. Both halves had the same
    cause — a pool only learns a pod is bad by FAILING to use it, and a
    warm-floor pod by definition has no traffic to fail. ``maintain()`` counted
    it as healthy, so ``healthy_n < keep_min`` was never true, so it never
    topped up and never had anything to say. The floor stayed "satisfied" by a
    machine that had not existed for hours, and the first real request paid a
    full cold boot — the exact cost keep_warm is there to avoid.

    ``live_status`` maps pod id -> status string, from one successful account
    listing. Only positive evidence acts: a dead status immediately, an absence
    only when the previous sweep saw it absent too. An empty or failed listing
    must never reach this function — see the caller.
    """
    global _absent_before
    with _pools_lock:
        pools = list(_pools.values())
        previously_absent = set(_absent_before)
        shielded = set(_provisioning_ids)
    #: id -> the pool handle holding it, for the ids we are judging.
    owned = {}
    for pool in pools:
        with pool._cv:
            for h in pool.pods:
                owned[str(h.pod_id)] = (pool, h)
    ids = {str(c) for c in (candidates or ())} | set(owned)
    absent_now, confirmed = set(), set()
    for pid in sorted(ids):
        if pid in shielded:
            continue
        ph = owned[pid][1] if pid in owned else None
        status = live_status.get(pid)
        if status is None:
            absent_now.add(pid)
            if pid not in previously_absent:
                # First listing without it. Say so when it is a pod we are
                # actively holding — one that flickers in and out of the
                # account listing is itself worth knowing about — but never act
                # on one data point.
                if pid in owned:
                    print(f"[runpod] pod {pid} ({owned[pid][0].model_key!r}) is not "
                          f"in the RunPod account listing; confirming on the next "
                          f"sweep before dropping it", flush=True)
                continue
            # A pod that served a request seconds ago exists, whatever the
            # listing says — traffic is better evidence than an API page. Only
            # the absence branch gets this: an explicit dead status is RunPod
            # telling us directly, and is trusted even mid-request. A warm-floor
            # pod, the case this whole function is for, has no traffic to speak
            # for it, so the guard never masks it.
            if (time.time() - float(getattr(ph, "last_used", 0) or 0)) < 60.0:
                continue
            reason = "not in the account listing twice running"
        elif str(status).upper() in _DEAD_STATUSES:
            reason = f"status {str(status).upper()}"
        else:
            continue
        confirmed.add(pid)
        if pid in owned:
            pool, h = owned[pid]
            pool.forget(h, reason)
    with _pools_lock:
        _absent_before = absent_now
    return confirmed


def _known_pod_ids() -> set:
    """Every pod id this DEPLOYMENT owns — not just this process's.

    A registry entry counts only while it still has an owner: ours always, and
    a sibling's for as long as it keeps heartbeating (see
    touch_registered_pods). An entry whose process is gone stops vouching for
    its pod after REAP_GRACE_SECONDS, because otherwise nothing ever reaps it
    and nothing manages it either — the pod simply bills.
    """
    ids = set()
    with _pools_lock:
        for pool in _pools.values():
            with pool._cv:
                ids.update(p.pod_id for p in pool.pods)
        ids.update(_provisioning_ids)
    ids.update(live_registered_pod_ids())
    return ids


#: Where RunPod mounts a global volume when it is the only one attached. With a
#: network volume beside it, RunPod moves the global one to /workspace-global.
GLOBAL_VOLUME_MOUNT = "/workspace"


#: Never reap a pod younger than this, whatever the registry says. Longer than
#: any boot budget, so a pod still pulling its image cannot be mistaken for a
#: leftover — the backstop for the race a registry cannot close (a process that
#: died between creating a pod and recording it).
REAP_GRACE_SECONDS = float(os.environ.get("CODERAI_RUNPOD_REAP_GRACE", "1200"))

#: Bounds on the kept-after-death pod log. Generous enough for a boot plus a
#: failure, small enough that a forgotten file cannot fill the config volume.
_POD_LOG_MAX_BYTES = int(os.environ.get("CODERAI_POD_LOG_MAX_BYTES", 8 * 1024 * 1024))
_POD_LOG_KEEP_LINES = int(os.environ.get("CODERAI_POD_LOG_KEEP_LINES", 20000))


def _rental_age_seconds(pod: dict):
    """Seconds since the pod was RENTED, from RunPod's own timestamps, or None.

    ``lastStatusChange`` is a sentence, not a field: "Rented by User: Thu Oct 15
    2026 19:36:58 GMT+0000 (Coordinated Universal Time)". The date after the
    colon is what we want, and only when the sentence is about the rental —
    a later status change would date the change, not the pod.
    """
    from datetime import datetime, timezone
    for key in ("createdAt", "created_at"):
        val = pod.get(key)
        if not val:
            continue
        try:
            return (datetime.now(timezone.utc) - datetime.fromisoformat(
                str(val).replace("Z", "+00:00"))).total_seconds()
        except Exception:
            continue
    text = str(pod.get("last_change") or pod.get("lastStatusChange") or "")
    if "rented" not in text.lower() or ":" not in text:
        return None
    stamp = text.split(":", 1)[1].split("(")[0].strip()
    for fmt in ("%a %b %d %Y %H:%M:%S GMT%z", "%a %b %d %Y %H:%M:%S %Z",
                "%a %b %d %Y %H:%M:%S"):
        try:
            when = datetime.strptime(stamp, fmt)
        except ValueError:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - when).total_seconds()
        # A clock that disagrees would otherwise reap on arrival.
        return age if age >= 0 else None
    return None


def _pod_age_seconds(pod: dict):
    """Seconds since the pod was created, or None when RunPod does not say.

    The RENTAL time first, uptime only as a floor. uptime is how long the
    CURRENT run has lasted, and it resets whenever the container restarts: a
    pod that fails and restarts every few minutes reported 234s while it was 11
    minutes old, so it could never reach REAP_GRACE_SECONDS and the reaper
    skipped it on every sweep, forever. Observed live — a pod abandoned
    mid-boot billed at $1.09/hr with nothing tracking it and no sweep that
    would ever have caught it.

    They are combined rather than swapped: uptime cannot exceed the age, so the
    larger of the two is still a true lower bound on how long this pod has
    existed, and keeping it means a pod whose rental stamp is unparsable is no
    worse off than before.
    """
    ages = []
    rental = _rental_age_seconds(pod)
    if rental is not None:
        ages.append(rental)
    for key in ("uptime_s", "uptimeSeconds", "runtimeSeconds"):
        val = pod.get(key)
        if isinstance(val, (int, float)) and val >= 0:
            ages.append(float(val))
            break
    return max(ages) if ages else None


def reap_orphans() -> int:
    """Terminate RunPod pods tagged as OURS that no live pool is tracking — stale
    pods from a crash/restart that would otherwise bill silently. Only touches pods
    named ``coderai-<our deployment_id>-*`` (never another deployment's), and never
    a pod currently mid-provision. Returns the number reaped."""
    try:
        from codai.models.manager import get_active_runpod_config
        acct = get_active_runpod_config()
    except Exception:
        acct = None
    if acct is None or not getattr(acct, "enabled", False) or not getattr(acct, "api_key", ""):
        return 0
    from codai.api.runpod_client import RunpodClient, RunpodError
    prefix = f"coderai-{_deploy_tag(acct)}-"
    # Pods this deployment recorded, whatever they are CALLED. Ownership was
    # decided by the name alone, so renaming `deployment_id` instantly orphaned
    # every pod already running under the old prefix: still billing, and no
    # longer anything's to reap. The registry is the better authority — it is
    # written when we create a pod — and the prefix stays as the fallback for a
    # pod this process never recorded.
    _ours = set()
    try:
        _ours = {str(k) for k in (_registry_update(lambda d: False) or {})}
    except Exception:
        _ours = set()
    try:
        client = RunpodClient(acct)
        pods = client.list_pods()
    except RunpodError as exc:
        print(f"[runpod-reaper] list failed: {exc}", flush=True)
        return 0
    known = _known_pod_ids()
    reaped = 0
    # Reconcile what we BELIEVE we have against what the account actually has,
    # before reaping anything. This listing is the only place a pod that died
    # out of band becomes visible: a pool never probes its own pods, so a
    # warm-floor pod with no traffic on it is never found dead by use. Pools
    # drop such a handle (and close its bill) here; registry entries for pods
    # RunPod no longer has go the same way — they are shielded from reaping by
    # being "known", so they would otherwise accumulate for the life of the
    # install. An empty listing is not evidence of an empty account, so a
    # successful-but-empty response is left to the two-sweep rule inside.
    live_status = {str(p.get("id")): str(p.get("status") or "")
                   for p in pods if p.get("id")}
    try:
        # Whose bill a POOL will close, read before the reconcile removes them
        # from their pools: `forget` ledgers those, and attributing them again
        # here would double-count every pod RunPod reclaims.
        with _pools_lock:
            _pooled = {str(h.pod_id) for p in _pools.values()
                       for h in list(p.pods)}
        # Judged against the WHOLE registry, not just the entries that still
        # have an owner. `known` is about shielding a pod from the reaper;
        # deciding whether a pod still exists is this function's own job, and
        # when 0.3.56 made a stale entry stop counting as known it also stopped
        # it being reconciled — so an entry for a pod that no longer exists was
        # never cleaned up and its hours were never attributed.
        for gone in reconcile_pools(live_status, known | registered_pod_ids()):
            if str(gone) not in _pooled:
                close_bill_from_registry(gone, "gone from the account listing")
            unregister_pod(gone)
    except Exception as exc:
        print(f"[runpod-reaper] reconcile: {exc}", flush=True)
    for p in pods:
        pid, name = p.get("id"), (p.get("name") or "")
        status = str(p.get("status") or "").upper()
        if not pid or not (name.startswith(prefix) or str(pid) in _ours):
            continue                      # not ours (or another deployment's)
        if pid in known:
            continue                      # tracked by a live pool / provisioning
        if status in ("TERMINATED", "EXITED"):
            continue
        age = _pod_age_seconds(p)
        if age is None:
            # Fail CLOSED. This guard is documented as holding "whatever the
            # registry says", and it was written as `age is not None and
            # age < GRACE` — so an unknown age skipped it entirely. list_pods()
            # returned no timestamp at all until the query below was widened,
            # which means the backstop never protected a single pod: a pod one
            # second old, invisible to a sibling's registry, was terminated as a
            # leftover. Observed live on the Digesta orchestrator — created
            # 19:24:56, reaped 19:24:57, and OCR then waited out the boot
            # timeout on a pod that no longer existed.
            #
            # Not reaping costs one more cycle. Reaping a live pod costs the work
            # on it and, when the pool immediately rents a replacement, bills in
            # a loop that never serves a request.
            print(f"[runpod-reaper] pod {pid} ({name}) has no age from RunPod — "
                  f"leaving it alone rather than guessing it is a leftover",
                  flush=True)
            continue
        if age < REAP_GRACE_SECONDS:
            continue                      # too young to be a leftover
        try:
            client.terminate_pod(pid)
            reaped += 1
            # A pod the reaper kills billed real money, and it used to leave no
            # trace: the spend was recorded by _terminate, which only runs for a
            # pod a pool was tracking. Same helper as the vanished case, so the
            # two paths cannot drift apart.
            close_bill_from_registry(pid, "reaped as a stale pod")
            unregister_pod(pid)
            print(f"[runpod-reaper] terminated STALE pod {pid} ({name}) — "
                  "not tracked by any pool and its owner stopped reporting",
                  flush=True)
        except Exception as exc:
            print(f"[runpod-reaper] failed to terminate {pid}: {exc}", flush=True)
    return reaped


def reclaim_registered_pods() -> int:
    """Take this deployment's surviving pods back into their pools, at startup.

    Adoption existed, but only ON DEMAND: it runs inside ``acquire`` when a
    request needs a pod. Nothing reconciled the registry at boot, so after a
    restart the pods this deployment was already paying for sat unowned until
    either a request happened to want that pool or the reaper's 20-minute grace
    expired. "Every restart leaves orphaned pods" was an accurate description
    of the design rather than a bug in one branch: an hourly unattended upgrade
    could pay for a fleet twice over, once for the pods it abandoned and again
    for the ones it rented beside them.

    Reclaiming here costs one registry read and one health probe per pod, and
    saves both the double billing and a cold boot the fleet had already paid
    for. Returns how many were taken back.

    Only pods whose pool key names a model this install configures: a shared
    pool key (``pool:…``) has no single entry to rebuild its shape from, and
    guessing would be worse than the on-demand path those already have.
    """
    try:
        from codai.models.manager import get_active_runpod_config
        acct = get_active_runpod_config()
    except Exception:
        acct = None
    if acct is None or not getattr(acct, "enabled", False):
        return 0
    try:
        data = _registry_update(lambda d: False) or {}
    except Exception as exc:
        print(f"[runpod] reclaim: cannot read the registry ({exc})", flush=True)
        return 0
    keys = {}
    for pod_id, info in data.items():
        if not isinstance(info, dict) or _is_this_process(info):
            continue
        key = str(info.get("pool") or "")
        if key:
            keys[key] = keys.get(key, 0) + 1
    if not keys:
        return 0
    taken = 0
    for key in keys:
        if key.startswith("pool:"):
            print(f"[runpod] reclaim: leaving shared pool {key!r} to the "
                  f"on-demand path — its shape comes from whichever model "
                  f"loads first, not from the registry", flush=True)
            continue
        entry = _model_entry(key)
        if not entry:
            print(f"[runpod] reclaim: {key!r} has pods registered but this "
                  f"install no longer configures it — leaving them to the "
                  f"reaper", flush=True)
            continue
        try:
            mcfg = parse_model_runpod(entry.get("runpod") or {})
            served = str((entry.get("runpod") or {}).get("served_model")
                         or entry.get("id") or key)
            pool = get_pod_pool(key, acct, mcfg, served, entry=entry)
            # EVERY pod of this pool, not one. A fleet of sixteen reclaimed one
            # pod and left fifteen billing, which is the bug with extra steps.
            # Bounded by how many the registry actually holds for this key, so
            # a pool that keeps answering "yes" cannot spin: adoption removes
            # nothing by itself, and the loop must end whatever it returns.
            for _ in range(keys[key]):
                if not pool._adopt_shared_pod():
                    break
                taken += 1
                print(f"[runpod] reclaim: took back a pod for {key!r} that "
                      f"outlived the process that rented it", flush=True)
        except Exception as exc:
            print(f"[runpod] reclaim: {key!r}: {exc}", flush=True)
    return taken


def _scaler_loop():
    tick = 0
    # Take our own pods back BEFORE reaping: a pod this deployment is already
    # paying for should rejoin its pool, not be killed and rented again.
    try:
        reclaim_registered_pods()
    except Exception as exc:
        print(f"[runpod] reclaim: startup: {exc}", flush=True)
    # Reap once promptly on start to catch orphans left by a crash/restart.
    try:
        reap_orphans()
    except Exception as exc:
        print(f"[runpod-reaper] startup: {exc}", flush=True)
    while True:
        time.sleep(15)
        tick += 1
        # Say we are still here, for every pod we own. Cheap (one locked file
        # write), and what keeps a sibling's reaper off our pods — and, when
        # this process dies, what stops ours being shielded forever.
        if tick % 4 == 1:
            try:
                touch_registered_pods()
            except Exception as exc:
                print(f"[runpod] registry heartbeat: {exc}", flush=True)
        try:
            with _pools_lock:
                pools = list(_pools.values())
            for pool in pools:
                try:
                    pool.maintain()
                except Exception as exc:
                    print(f"[runpod] scaler: {exc}", flush=True)
        except Exception:
            pass
        # Independent stale-pod safety sweep every ~30s, even with no local pools.
        if tick % 2 == 0:
            try:
                reap_orphans()
            except Exception as exc:
                print(f"[runpod-reaper] {exc}", flush=True)


def _ensure_scaler():
    global _scaler_started
    with _pools_lock:
        if _scaler_started:
            return
        _scaler_started = True
    threading.Thread(target=_scaler_loop, daemon=True, name="runpod-scaler").start()


def start_runpod_maintenance():
    """Start the scaler + stale-pod reaper independently of any loaded model — call
    at engine startup when RunPod is enabled so orphaned pods from a previous run are
    reaped even before the first request."""
    _ensure_scaler()


def stop_all_pods():
    with _pools_lock:
        pools = list(_pools.values())
        _pools.clear()
    for p in pools:
        try:
            p.close()
        except Exception:
            pass


import atexit as _atexit
_atexit.register(stop_all_pods)
