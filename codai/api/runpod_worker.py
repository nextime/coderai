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
    min_vram_gb: float = 0.0
    max_hourly_usd: float = 0.0              # per-model $/hr GPU ceiling (0 = none)
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
    # Hard ceiling on concurrent requests per pod (0 = no ceiling, let the remote
    # server's own batching absorb them). Above it, a request waits for a slot
    # instead of being piled on — use it for engines that degrade badly under
    # concurrency rather than queueing internally.
    max_inflight_per_pod: int = 0
    # Send a conversation back to the pod that already served it, so the remote's
    # prefix/KV cache still holds its context. Off = always pick the least-loaded.
    sticky_sessions: bool = True
    idle_timeout_s: int = 300                # destroy a pod this long after its last request
    # Boot budgets — bigger models need longer (image pull + weight download).
    boot_timeout_s: int = 300                # until the pod exposes its port
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
)

#: Capabilities served by a published image other than their own name.
_CAPABILITY_IMAGE_ALIASES = {
    "rerank": "embeddings",      # same sentence-transformers stack
    "speaker": "stt",            # diarization/voiceprints ride the STT deps
    "stems": "audio",
    "audio_clean": "audio",
    "audio_gen": "audio",
}


#: models.json section -> the capability whose pod image serves it. This is what
#: lets a NON-TEXT model be sent to RunPod individually: the pod image follows
#: from what the model is, not from whether its path ends in .gguf.
MODEL_TYPE_CAPABILITY = {
    "image_models": "images",
    "video_models": "video",
    "audio_models": "stt",
    "tts_models": "tts",
    "embedding_models": "embeddings",
    "spatial_models": "spatial",
    "audio_gen_models": "audio_gen",
}

#: Sections served by an LLM pod (vLLM / llama.cpp) rather than a capability pod.
_LLM_MODEL_TYPES = ("text_models", "gguf_models", "vision_models")


def model_capability(entry: dict) -> str:
    """The capability a model entry belongs to, or '' for a text/LLM model."""
    if not isinstance(entry, dict):
        return ""
    types = entry.get("model_types") or [entry.get("model_type") or ""]
    for mt in types:
        if mt in _LLM_MODEL_TYPES:
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
    if want in ("vllm", "llamacpp", "coderai", "custom"):
        return want
    # A non-text model is served by a whole coderai carrying that capability.
    if entry is None:
        entry = _model_entry(model_key)
    if model_capability(entry or {}):
        return "coderai"
    if mcfg.hf_gguf:
        return "llamacpp"
    if _looks_like_gguf(mcfg.served_model) or _looks_like_gguf(model_path) \
            or _looks_like_gguf(model_key):
        return "llamacpp"
    return "vllm"


#: Readiness probe per engine. A coderai pod answers /healthz long before any
#: model is loaded, which is exactly what we want: it is ready to be asked.
_HEALTH_PATHS = {"vllm": "/v1/models", "llamacpp": "/v1/models",
                 "coderai": "/healthz", "custom": "/healthz"}


def pod_plan(mcfg: "RunpodModelConfig", served: str, model_key: str = "",
             model_path: str = "", api_key: str = "", entry: dict = None) -> dict:
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
        cap = model_capability(entry or {})
        if not mcfg.image and cap:
            # A model of a known kind gets that capability's published image.
            image = default_capability_image(cap)
            if not image:
                raise RuntimeError(
                    f"RunPod: no published pod image for {cap!r} models — set "
                    "`image` on this model's runpod block to one you have built.")
            args = mcfg.docker_args
            return _plan(engine, image, args, mcfg, api_key, entry, served)
        if not mcfg.image:
            raise RuntimeError(
                f"RunPod {engine} pod: set `image` on the runpod block to the image "
                "to run. Capability pods default to the published "
                f"{CAPABILITY_IMAGE_REPO}-<capability> image; this one has none, so "
                "name it explicitly.")
        image = mcfg.image
        args = mcfg.docker_args          # usually blank: the image's entrypoint serves
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
    return {"engine": engine, "image": image, "args": args, "env": env,
            "health_path": mcfg.health_path or _HEALTH_PATHS.get(engine, "/v1/models")}


def _plan(engine, image, args, mcfg, api_key, entry, served) -> dict:  # noqa: D401
    """Finish a plan for a coderai pod: auth plus the model it must serve."""
    env = {}
    if api_key:
        env["CODERAI_API_TOKEN"] = api_key
    kind, value = resolve_model_source(entry or {}, mcfg)
    seed = seed_model_env(entry, served, value if kind in ("hf", "url") else "")
    if seed:
        env["CODERAI_SEED_MODELS"] = seed
    return {"engine": engine, "image": image, "args": args, "env": env,
            "health_path": mcfg.health_path or _HEALTH_PATHS.get(engine, "/healthz")}


def seed_model_env(entry: dict, served: str = "", source: str = "") -> str:
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
    elif not path or path.startswith("/"):
        # A local path cannot be resolved on a rented machine, and nothing said
        # where else to get it.
        return ""
    keep = ("path", "model_type", "model_types", "video_subtypes", "capabilities",
            "alias", "config_name", "load_in_4bit", "load_in_8bit", "n_ctx",
            "flash_attention", "model_template", "acceleration", "component_quantization",
            "languages", "supports_translation", "parser", "max_instances")
    out = {k: entry[k] for k in keep if k in entry and entry[k] is not None}
    out["path"] = path
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
    return " ".join(args)


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
    cfg.allow_spot = _as_bool(b.get("allow_spot"), cfg.allow_spot)
    cfg.image = (b.get("image") or "").strip()
    cfg.engine = (b.get("engine") or cfg.engine).strip().lower() or "auto"
    cfg.health_path = (b.get("health_path") or "").strip()
    cfg.docker_args = (b.get("docker_args") or "").strip()
    cfg.registry_auth_id = (b.get("registry_auth_id") or "").strip()
    cfg.pool = (b.get("pool") or "").strip().lower()
    cfg.quantization = (b.get("quantization") or "").strip()
    cfg.source = (b.get("source") or cfg.source).strip().lower() or "auto"
    cfg.hf_repo = (b.get("hf_repo") or "").strip()
    cfg.model_url = (b.get("model_url") or "").strip()
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
    cfg.scale_up_inflight_per_pod = max(1, _as_int(b.get("scale_up_inflight_per_pod"),
                                                   cfg.scale_up_inflight_per_pod))
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
        r = requests.get(url.rstrip("/") + "/" + path.lstrip("/"), timeout=timeout,
                         headers=headers)
        return r.status_code == 200
    except Exception:
        return False


def _rank_gpus(client, mcfg: "RunpodModelConfig", account_cfg) -> list:
    """Ranked list of GPU options across the model's allowed pools per its selection
    criteria (best first). Each item: {gpu_type_id, display_name, memory_gb,
    cloud_type, price, is_spot, bid}. Raises RunpodError if nothing fits the
    constraints. _provision_one tries them in order so a capacity miss on the top
    pick falls through to the next."""
    from codai.api.runpod_client import RunpodError
    catalog = client.list_gpu_types()
    allowed = [c.upper() for c in (mcfg.cloud_types or ["SECURE"])] or ["SECURE"]
    ceiling = mcfg.max_hourly_usd or 0.0
    explicit = (mcfg.gpu_type or getattr(account_cfg, "default_gpu_type", "") or "").strip()

    cands = []   # (price, is_spot, cloud_type, gpu, on_demand_price)
    for g in catalog:
        if explicit and g["id"] != explicit and g["display_name"] != explicit:
            continue
        mem = g.get("memory_gb") or 0
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
                if ceiling > 0 and price > ceiling:
                    continue
                cands.append((price, is_spot, ct, g, on_demand or price))
    if not cands:
        raise RunpodError(
            f"RunPod: no GPU in pools {allowed} with ≥{mcfg.min_vram_gb}GB under "
            f"${ceiling}/hr (spot={'on' if mcfg.allow_spot else 'off'}"
            + (f", gpu={explicit}" if explicit else "") + ").")

    if (mcfg.selection_criteria or "cheaper").lower() == "faster":
        # Reliability + throughput: SECURE before COMMUNITY, on-demand before spot,
        # more VRAM (throughput proxy) first, then cheapest.
        ct_rank = {"SECURE": 0, "COMMUNITY": 1}
        cands.sort(key=lambda c: (ct_rank.get(c[2], 9), c[1] is True,
                                  -(c[3].get("memory_gb") or 0), c[0]))
    else:  # cheaper
        cands.sort(key=lambda c: (c[0], c[1] is False))
    return [{"gpu_type_id": g["id"], "display_name": g["display_name"],
             "memory_gb": g.get("memory_gb"), "cloud_type": ct, "price": price,
             "is_spot": is_spot, "bid": on_demand if is_spot else 0.0}
            for (price, is_spot, ct, g, on_demand) in cands]


def _select_gpu(client, mcfg: "RunpodModelConfig", account_cfg) -> dict:
    """The single best GPU option (top of _rank_gpus)."""
    return _rank_gpus(client, mcfg, account_cfg)[0]


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
    last_used: float = 0.0
    console_url: str = ""


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
        elif kind in ("url", "upload"):
            raise RuntimeError(
                f"RunPod vLLM pod: vLLM is launched with `--model` and can only take "
                f"a HuggingFace repo id — it cannot fetch {kind!r}. Set `hf_repo`, or "
                "serve this model on a coderai pod (engine: coderai), which accepts a "
                "URL or an upload.")
        else:
            raise RuntimeError(
                f"RunPod vLLM pod: {name or '(nothing)'} is not a HuggingFace repo id, "
                "and nothing else says where the pod should get the weights. Set "
                "`hf_repo` (e.g. \"Qwen/Qwen3.5-9B\") on the model's runpod block.")
    args = ["--host", "0.0.0.0", "--port", str(mcfg.port or 8000),
            "--model", name, "--served-model-name", name]
    if mcfg.ctx and mcfg.ctx > 0:
        args += ["--max-model-len", str(mcfg.ctx)]
    if mcfg.quantization:
        args += ["--quantization", mcfg.quantization]
    return " ".join(args)


class RunpodPodPool:
    """A pool of coderai-managed RunPod pods for one model.

    * ``acquire()`` returns a ready pod URL (provisioning one on demand, up to
      ``max_pods``, subject to the cost caps) and bumps its in-flight count.
    * ``release()`` drops the in-flight count and stamps last-used.
    * the shared scaler thread (``_scaler_loop``) tears down idle pods to
      ``min_pods`` after ``idle_timeout_s``, re-checks health, reaps dead pods,
      and keeps ``min_pods`` warm.
    """

    def __init__(self, model_key, account_cfg, mcfg: "RunpodModelConfig", served_name,
                 entry: dict = None):
        #: The model's models.json entry, when this pool serves one specific
        #: model: it decides the pod image and is seeded into the pod.
        self.entry = entry if isinstance(entry, dict) else None
        self.model_key = model_key
        self.account = account_cfg
        self.mcfg = mcfg
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
        self._provisioning = False
        self._closed = False

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
    def _create_with_fallback(self, client, ranked, start=0):
        """Try ranked GPU candidates from ``start``; skip capacity misses. Returns
        (pod_id, sel, next_index). Raises if all fail."""
        from codai.api.runpod_client import RunpodError
        port = self.mcfg.port or 8000
        plan = pod_plan(self.mcfg, self.served, str(self.model_key),
                        _model_path_for(self.model_key), api_key=self.api_key,
                        entry=self.entry)
        image, args = plan["image"], plan["args"]
        self.health_path = plan["health_path"]
        env = dict(self.mcfg.env or {})
        env.update(plan.get("env") or {})
        dc = getattr(self.account, "data_center", "") or ""
        last = None
        for i in range(start, len(ranked)):
            sel = ranked[i]
            block = self._budget_blocks(sel["price"])
            if block:
                raise RunpodError(f"RunPod budget cap hit — not provisioning ({block}).")
            name = (f"coderai-{_deploy_tag(self.account)}-"
                    + str(self.model_key)[:20].replace("/", "_").replace(" ", "_")
                    + "-" + uuid.uuid4().hex[:6])
            print(f"[runpod] provisioning pod for {self.model_key!r}: {sel['display_name']} "
                  f"({sel['cloud_type']}{'/spot' if sel['is_spot'] else ''}) ${sel['price']}/hr",
                  flush=True)
            try:
                pod_id = client.create_pod(
                    name=name, image=image, gpu_type_id=sel["gpu_type_id"], port=port,
                    cloud_type=sel["cloud_type"], container_disk_gb=self.mcfg.container_disk_gb,
                    volume_gb=self.mcfg.volume_gb, env=env, docker_args=args,
                    is_spot=sel["is_spot"], bid_per_gpu=sel["bid"], data_center_id=dc,
                    registry_auth_id=(self.mcfg.registry_auth_id
                                      or getattr(self.account, "registry_auth_id", "") or ""))
                return pod_id, sel, i + 1
            except RunpodError as exc:
                msg = str(exc).lower()
                if "no longer any instances" in msg or "no instances" in msg or "capacity" in msg:
                    print(f"[runpod] {sel['display_name']} ({sel['cloud_type']}) unavailable — "
                          "trying next candidate", flush=True)
                    last = exc
                    continue
                raise
        raise last or RunpodError("RunPod: no candidate GPU could be deployed (capacity).")

    # Port should appear once the pod's container is running; a much longer wait
    # only bills for a pod that failed to start. The OpenAI server (image pull +
    # model load) gets a longer, separate budget.
    PORT_TIMEOUT_S = 300.0
    HEALTH_TIMEOUT_S = 600.0

    def _dump_pod_logs(self, client, pod_id, why):
        """Fetch + log the pod's container/vLLM output so a failed boot is
        self-diagnosing (the user's #1 cause: vLLM OOM / launch error)."""
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

    def _provision_one(self):
        """Create + boot one pod; append it healthy. Blocking (minutes).

        Retries on a different machine when a pod is accepted but never boots."""
        from codai.api.runpod_client import RunpodClient, RunpodError, pod_console_url
        client = RunpodClient(self.account)
        ranked = _rank_gpus(client, self.mcfg, self.account)
        port = self.mcfg.port or 8000
        idx, attempts, last_exc = 0, 0, None

        while idx < len(ranked) and attempts < self.MAX_PROVISION_ATTEMPTS:
            pod_id, sel, idx = self._create_with_fallback(client, ranked, idx)
            attempts += 1
            console = pod_console_url(pod_id)
            with _pools_lock:
                _provisioning_ids.add(pod_id)   # shield from the reaper during boot
                _provisioning_info[pod_id] = {
                    "model": self.model_key, "gpu": sel["display_name"],
                    "is_spot": sel["is_spot"], "hourly_usd": sel["price"],
                    "console_url": console, "started_at": time.time()}
            print(f"[runpod] pod {pod_id} created for {self.model_key!r} "
                  f"({sel['display_name']} {sel['cloud_type']}) — booting; logs: {console}",
                  flush=True)
            boot_to = float(self.mcfg.boot_timeout_s or self.PORT_TIMEOUT_S)
            load_to = float(self.mcfg.load_timeout_s or self.HEALTH_TIMEOUT_S)
            try:
                url = client.wait_ready(pod_id, port, ready_timeout=boot_to)
                # Wait for the OpenAI server inside the pod (image pull + model load).
                deadline = time.time() + load_to
                while time.time() < deadline:
                    if _pod_health_ok(url, path=self.health_path,
                                      api_key=self.api_key):
                        break
                    time.sleep(5)
                else:
                    raise RunpodError(f"pod {pod_id} OpenAI server did not answer "
                                      f"/v1/models within {int(load_to)}s")
            except Exception as exc:
                # Self-diagnose: pull the vLLM/container log before tearing down, so
                # the cause (OOM / bad args / model gate / stuck machine) is in the log.
                last_exc = exc
                print(f"[runpod] pod {pod_id} boot FAILED for {self.model_key!r}: {exc}",
                      flush=True)
                self._dump_pod_logs(client, pod_id, "boot failed")
                try:
                    client.terminate_pod(pod_id)
                except Exception:
                    pass
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
                          console_url=console)
            with self._cv:
                self.pods.append(h)
                self._cv.notify_all()
            with _pools_lock:
                _provisioning_ids.discard(pod_id)
                _provisioning_info.pop(pod_id, None)
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
        for _ in range(want):
            try:
                self._provision_one()
            except Exception as exc:
                print(f"[runpod] ensure_ready: {exc}", flush=True)
                raise

    def _pick(self, healthy: list, affinity: str):
        """Which pod serves this request: the one that already holds the
        conversation's cache, else the least loaded."""
        cap = int(getattr(self.mcfg, "max_inflight_per_pod", 0) or 0)
        free = [p for p in healthy if cap <= 0 or p.inflight < cap]
        if not free:
            return None
        if affinity and getattr(self.mcfg, "sticky_sessions", True):
            pid = self._affinity.get(affinity)
            for p in free:
                if p.pod_id == pid:
                    return p
        return min(free, key=lambda x: x.inflight)

    def _remember_affinity(self, affinity: str, pod: "PodHandle") -> None:
        if not affinity or not getattr(self.mcfg, "sticky_sessions", True):
            return
        self._affinity[affinity] = pod.pod_id
        self._affinity.move_to_end(affinity)
        while len(self._affinity) > 4096:
            self._affinity.popitem(last=False)

    def _should_grow(self, healthy: list) -> bool:
        """True when real concurrency justifies another pod.

        The pool used to grow only when it had NO healthy pod, so every request
        beyond the first piled onto pod #1 and max_pods never came into play.
        """
        if len(self.pods) >= self.mcfg.max_pods or self._provisioning:
            return False
        if not healthy:
            return True
        threshold = max(1, int(self.mcfg.scale_up_inflight_per_pod or 1))
        return min(p.inflight for p in healthy) >= threshold

    def _grow_in_background(self):
        """Provision another pod without making the triggering request wait for it.

        Scale-up is anticipatory: this request is already being served by an
        existing pod, and the new one absorbs the NEXT burst. A cold pod takes
        minutes, so blocking here would punish exactly the request that proved
        the pool needs to grow.
        """
        def _run():
            try:
                self._provision_one()
            except Exception as exc:
                print(f"[runpod] scale-up for {self.model_key!r} failed: {exc}", flush=True)
            finally:
                with self._cv:
                    self._provisioning = False
                    self._cv.notify_all()
        threading.Thread(target=_run, daemon=True,
                         name=f"runpod-grow-{self.model_key}").start()

    def acquire(self, timeout: float = 1200.0, affinity: str = ""):
        """Return (PodHandle, url) for a ready pod, provisioning on demand.

        Bumps in-flight; the caller MUST pair it with release(pod). Picks the pod
        holding this conversation's cache when ``affinity`` is given, else the
        least-loaded one, and grows the pool in the background once the least
        loaded pod is at ``scale_up_inflight_per_pod``.
        """
        from codai.api.runpod_client import RunpodError
        deadline = time.time() + timeout
        while True:
            grow_now = False
            with self._cv:
                healthy = [p for p in self.pods if p.healthy]
                chosen = self._pick(healthy, affinity) if healthy else None
                if self._should_grow(healthy):
                    self._provisioning = True
                    grow_now = True
                if chosen is not None:
                    chosen.inflight += 1
                    chosen.last_used = time.time()
                    self._remember_affinity(affinity, chosen)
                    url = chosen.url
            if chosen is not None:
                if grow_now:
                    self._grow_in_background()      # absorbs the NEXT request
                return chosen, url

            if grow_now:
                # Nothing to serve this request yet — provision inline and retry.
                try:
                    self._provision_one()
                finally:
                    with self._cv:
                        self._provisioning = False
                        self._cv.notify_all()
                continue

            # At max_pods with every pod at its in-flight ceiling, or someone else
            # is provisioning: wait for a slot rather than piling on.
            with self._cv:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise RunpodError(
                        f"RunPod: no pod slot available for {self.model_key!r} "
                        f"(max_pods={self.mcfg.max_pods}, "
                        f"max_inflight_per_pod={self.mcfg.max_inflight_per_pod}) "
                        f"within {timeout}s.")
                self._cv.wait(min(remaining, 10.0))

    def release(self, pod: PodHandle):
        with self._cv:
            pod.inflight = max(0, pod.inflight - 1)
            pod.last_used = time.time()
            self._cv.notify_all()

    def _terminate(self, pod: PodHandle, reason: str):
        from codai.api.runpod_client import RunpodClient
        from codai.api import runpod_ledger
        cost = (time.time() - pod.started_at) / 3600.0 * pod.hourly_usd
        try:
            RunpodClient(self.account).terminate_pod(pod.pod_id)
        except Exception as exc:
            print(f"[runpod] terminate {pod.pod_id} failed: {exc}", flush=True)
        runpod_ledger.record_spend(self.model_key, cost,
                                   {"pod_id": pod.pod_id, "gpu": pod.gpu, "is_spot": pod.is_spot})
        print(f"[runpod] pod {pod.pod_id} for {self.model_key!r} terminated ({reason}); "
              f"billed ~${cost:.3f}", flush=True)

    def maintain(self):
        """One scaler pass: reap idle/dead pods to min_pods, keep min warm."""
        now = time.time()
        with self._cv:
            pods = list(self.pods)
        # Idle teardown (down to min_pods), only pods with no in-flight work.
        idle_to = self.mcfg.idle_timeout_s
        keep_min = self.mcfg.min_pods
        with self._cv:
            alive = [p for p in self.pods if p.healthy]
            for p in sorted(alive, key=lambda x: x.last_used):
                if len([q for q in self.pods if q.healthy]) <= keep_min:
                    break
                if p.inflight == 0 and idle_to > 0 and (now - p.last_used) > idle_to:
                    p.healthy = False
                    self.pods.remove(p)
                    to_kill = p
                    # terminate outside the lock
                    threading.Thread(target=self._terminate, args=(to_kill, "idle"),
                                     daemon=True).start()
        # Keep min_pods warm.
        with self._cv:
            healthy_n = len([p for p in self.pods if p.healthy])
        if healthy_n < keep_min and not self._closed:
            try:
                self._provision_one()
            except Exception as exc:
                print(f"[runpod] maintain: warm provision failed: {exc}", flush=True)

    def close(self):
        self._closed = True
        with self._cv:
            pods = list(self.pods)
            self.pods.clear()
        for p in pods:
            self._terminate(p, "shutdown")


# module-level pool registry + scaler ------------------------------------- #
_pools_lock = threading.RLock()
_pools: dict = {}          # model_key -> RunpodPodPool
_scaler_started = False


#: Pod servers that are pinned to a single model at launch, so several models
#: can never share one. A coderai (or comparable) image picks the model per
#: request and can.
_SINGLE_MODEL_ENGINES = ("vllm", "llamacpp")


def shared_pool_key(mcfg: "RunpodModelConfig", model_key: str,
                    model_path: str = "") -> str:
    """The pool key for this model: its own, or a shared one it opted into.

    Raises when the configured engine cannot serve more than one model — better
    a clear error now than a pod that answers every request with the wrong
    model's weights (or a 404 from a server that was launched for another one).
    """
    shared = (getattr(mcfg, "pool", "") or "").strip().lower()
    if not shared:
        return str(model_key)
    engine = resolve_pod_engine(mcfg, model_key, model_path)
    if engine in _SINGLE_MODEL_ENGINES:
        raise RuntimeError(
            f"RunPod: model {model_key!r} asks to share pod pool {shared!r}, but a "
            f"{engine} pod is launched for ONE model and cannot serve another. "
            "Share a pool only between models on a coderai (or custom multi-model) "
            "pod image, or drop the pool name so this model gets its own pods.")
    return f"pool:{shared}"


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
    return get_pod_pool(key, acct, mcfg, shared or capability, shared=bool(shared))


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
            name = entry.get("alias") or entry.get("path") or ""
            try:
                pool = get_model_pod_pool(name, entry, block)
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


def pods_status() -> list:
    """Snapshot of all pods across pools for the stats/tasks pages."""
    now = time.time()
    out = []
    with _pools_lock:
        pools = list(_pools.items())
    for key, pool in pools:
        with pool._cv:
            for p in pool.pods:
                out.append({
                    "model": key, "pod_id": p.pod_id, "gpu": p.gpu,
                    "is_spot": p.is_spot, "healthy": p.healthy, "inflight": p.inflight,
                    "hourly_usd": p.hourly_usd, "uptime_s": int(now - p.started_at),
                    "live_cost_usd": round((now - p.started_at) / 3600.0 * p.hourly_usd, 4),
                    "console_url": p.console_url, "state": "ready",
                })
    # Booting pods (created, not yet serving) — visible so a slow/failing boot is
    # obvious and its vLLM log is one click away.
    with _pools_lock:
        prov = list(_provisioning_info.items())
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


def _known_pod_ids() -> set:
    """All pod ids this process is responsible for: live pool pods + mid-provision."""
    ids = set()
    with _pools_lock:
        for pool in _pools.values():
            with pool._cv:
                ids.update(p.pod_id for p in pool.pods)
        ids.update(_provisioning_ids)
    return ids


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
    try:
        client = RunpodClient(acct)
        pods = client.list_pods()
    except RunpodError as exc:
        print(f"[runpod-reaper] list failed: {exc}", flush=True)
        return 0
    known = _known_pod_ids()
    reaped = 0
    for p in pods:
        pid, name = p.get("id"), (p.get("name") or "")
        status = str(p.get("status") or "").upper()
        if not pid or not name.startswith(prefix):
            continue                      # not ours (or another deployment's)
        if pid in known:
            continue                      # tracked by a live pool / provisioning
        if status in ("TERMINATED", "EXITED"):
            continue
        try:
            client.terminate_pod(pid)
            reaped += 1
            print(f"[runpod-reaper] terminated STALE pod {pid} ({name}) — "
                  "not tracked by any pool", flush=True)
        except Exception as exc:
            print(f"[runpod-reaper] failed to terminate {pid}: {exc}", flush=True)
    return reaped


def _scaler_loop():
    tick = 0
    # Reap once promptly on start to catch orphans left by a crash/restart.
    try:
        reap_orphans()
    except Exception as exc:
        print(f"[runpod-reaper] startup: {exc}", flush=True)
    while True:
        time.sleep(15)
        tick += 1
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
