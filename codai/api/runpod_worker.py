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
    volume_mount_path: str = ""              # blank = the account default
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
    # Text: a coderai pod that serves LLMs. Unlike a vLLM/llama.cpp pod it can be
    # SENT a local LoRA adapter, which is the only way a local text adapter runs
    # on RunPod. Build and push it with packaging/runpod before using it.
    "text",
)

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
    if want in ("vllm", "llamacpp", "coderai", "custom"):
        return want
    # A non-text model is served by a whole coderai carrying that capability.
    if entry is None:
        entry = _model_entry(model_key)
    if model_capability(entry or {}):
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
        image = mcfg.image
        capability_hint = model_capability(entry or {}, include_text=True)
        if not image:
            image = default_capability_image(capability_hint) if capability_hint else ""
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


def volume_for(mcfg: "RunpodModelConfig", account) -> tuple:
    """(volume_id, mount_path) for this pool, or ('', '')."""
    vol = (getattr(mcfg, "network_volume_id", "") or "").strip() \
        or (getattr(account, "network_volume_id", "") or "").strip()
    if not vol:
        return "", ""
    mount = (getattr(mcfg, "volume_mount_path", "") or "").strip() \
        or (getattr(account, "volume_mount_path", "") or "").strip() or "/workspace"
    return vol, mount.rstrip("/") or "/workspace"


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
        f'exec "$VENV/bin/python" -m uvicorn codai.api.app:app --host 0.0.0.0 '
        f'--port {int(port)}',
    ])


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
    }


def pod_ocr_engine(requested: str) -> str:
    """The OCR engine a pod will actually serve for a request naming ``requested``.

    Mirrors the env a pod is launched with: surya needs a VLM server the image
    does not carry, so a pod serves docTR for it. The gateway uses this to
    rewrite the request's own `engine` field, because the pod honours what the
    request asks for over its configured default.
    """
    r = (requested or "").strip().lower()
    if r == "surya":
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


def _plan(engine, image, args, mcfg, api_key, entry, served,
          seed_entries: list = None, capability: str = "") -> dict:  # noqa: D401
    """Finish a plan for a coderai pod: auth plus the models it must serve."""
    import json as _json
    env = {}
    if api_key:
        env["CODERAI_API_TOKEN"] = api_key
    seeds = []
    if entry:
        kind, value = resolve_model_source(entry, mcfg)
        one = seed_model_env(entry, served, value if kind in ("hf", "url") else "")
        if one:
            seeds.extend(_json.loads(one))
    for extra in (seed_entries or []):
        one = seed_model_env(extra, "", "")
        if one:
            seeds.extend(_json.loads(one))
    if seeds:
        # De-duplicate by path: a capability pod's list can overlap the one model
        # a per-model pool also seeds.
        seen, unique = set(), []
        for s in seeds:
            if s.get("path") in seen:
                continue
            seen.add(s.get("path"))
            unique.append(s)
        env["CODERAI_SEED_MODELS"] = _json.dumps(unique)
        print(f"[runpod] pod will be told about {len(unique)} model(s)", flush=True)
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

    if cap == "ocr":
        env["CODERAI_OCR_ENABLED"] = "1"
        wanted = str((entry or {}).get("path") or "").strip().lower()
        if wanted == "surya":
            # Surya 0.22 is a VLM ("surya-ocr-2"), not a self-contained OCR
            # library: it needs a server — vLLM, which it spawns IN DOCKER (a
            # pod cannot), or llama-server on a GGUF, which the image does not
            # ship. Locally it rides coderai's own vLLM engine; a pod has none.
            # So a pod asked for surya serves the request with docTR, which is
            # in the image and has passed — rather than loading surya, reaching
            # inference, and dying on "docker binary not found".
            print("[runpod] surya on a pod needs a VLM server the image does not "
                  "carry — this OCR request will be served by docTR", flush=True)
            wanted = "doctr"
        if wanted not in ("paddle", "doctr", "surya"):
            # The pod images ship docTR, the only engine that runs in-process.
            wanted = "doctr"
        env["CODERAI_OCR_DEFAULT_ENGINE"] = wanted
        # The subsystem flag is not enough: each engine has a gate of its own,
        # and doctr and surya both default to off.
        env[f"CODERAI_OCR_{wanted.upper()}_ENABLED"] = "1"
        if wanted == "paddle":
            # The paddle image bakes it, because paddlepaddle brings its own
            # CUDA runtime. Point the engine at that interpreter rather than
            # letting it try to build a venv on a rented machine.
            env["CODERAI_OCR_PADDLE_VENV"] = "/opt/coderai/venvs/paddleocr"
        if wanted == "surya" or _surya_accepted():
            # Surya is GPL and gated on an explicit acceptance. The pod inherits
            # the decision made HERE — it cannot make it for itself, and asking
            # for surya by name is that decision.
            env["CODERAI_OCR_SURYA_ACCEPT_LICENSE"] = "1"

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
                    rewritten.append({"path": src, "weight": spec.get("weight", 1.0),
                                      "name": spec.get("name")})
                    continue
                local = local_path(src)
                bid = blob_id(local) if local and os.path.isfile(local) else ""
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
    cfg.gpu_count = max(1, min(8, _as_int(b.get("gpu_count"), cfg.gpu_count)))
    cfg.allow_spot = _as_bool(b.get("allow_spot"), cfg.allow_spot)
    cfg.image = (b.get("image") or "").strip()
    cfg.engine = (b.get("engine") or cfg.engine).strip().lower() or "auto"
    cfg.health_path = (b.get("health_path") or "").strip()
    cfg.docker_args = (b.get("docker_args") or "").strip()
    cfg.registry_auth_id = (b.get("registry_auth_id") or "").strip()
    cfg.network_volume_id = (b.get("network_volume_id") or "").strip()
    if cfg.network_volume_id:
        # RunPod offers network volumes on Secure Cloud only; leaving COMMUNITY
        # in the list just produces candidates the attach would reject.
        cfg.cloud_types = ["SECURE"]
    cfg.volume_mount_path = (b.get("volume_mount_path") or "").strip()
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
    n = max(1, int(getattr(mcfg, "gpu_count", 1) or 1))
    # The catalogue prices ONE card. A pod with N of them costs N times that and
    # holds N times the VRAM, so both the ceiling and the VRAM floor are checked
    # against the pod, not the card — or a 4-GPU pod would pass a $1/hr ceiling
    # at $0.90 a card and bill $3.60.
    explicit = (mcfg.gpu_type or getattr(account_cfg, "default_gpu_type", "") or "").strip()

    cands = []   # (price, is_spot, cloud_type, gpu, on_demand_price)
    for g in catalog:
        if explicit and g["id"] != explicit and g["display_name"] != explicit:
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
                cands.append((price, is_spot, ct, g, (on_demand or price) * n))
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
    return [{"gpu_type_id": g["id"],
             "display_name": (f"{n}x {g['display_name']}" if n > 1 else g["display_name"]),
             "memory_gb": (g.get("memory_gb") or 0) * n, "cloud_type": ct, "price": price,
             "is_spot": is_spot, "bid": on_demand if is_spot else 0.0, "gpu_count": n}
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
    return " ".join(args)


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
      ``min_pods`` after ``idle_timeout_s``, re-checks health, reaps dead pods,
      and keeps ``min_pods`` warm.
    """

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
    def _log_pod_catalogue(self, url: str) -> None:
        """Log what the pod says it can serve, the moment it is up.

        The one check that would have turned a confusing "Model 'x' is not
        available. Use one of: " into an obvious "the pod registered nothing",
        minutes earlier and without reading a single line of our own code.
        """
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = requests.get(url.rstrip("/") + "/v1/models", headers=headers, timeout=20)
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
        vol_id, vol_mount = volume_for(self.mcfg, self.account)
        if vol_id:
            env.update(volume_env(vol_mount))
            if getattr(self, "_volume_dc", None) is None:
                self._volume_dc = self._data_center_for_volume(vol_id)
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
                    data_center_id=(getattr(self, "_volume_dc", "") or dc),
                    network_volume_id=vol_id,
                    volume_mount_path=(vol_mount or "/workspace"),
                    registry_auth_id=(self.mcfg.registry_auth_id
                                      or getattr(self.account, "registry_auth_id", "") or ""),
                    entrypoint=plan.get("entrypoint"), start_cmd=plan.get("start_cmd"))
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

    def _dump_pod_logs(self, client, pod_id, why, url: str = ""):
        """Say what the pod was doing, from whichever source can still answer.

        RunPod publishes no pod-log API — the REST OpenAPI spec lists 23 routes
        and none of them serve logs, which is why every attempt came back 400.
        The console is a human surface. So a coderai pod keeps its own boot
        record and serves it at /boot: if the port ever opened, that is the real
        log, and it covers the container phases too (boot.sh writes them to the
        same file). Tried first, because it is the only one that ever works.
        """
        if url:
            try:
                import requests
                headers = ({"Authorization": f"Bearer {self.api_key}"}
                           if self.api_key else {})
                r = requests.get(url.rstrip("/") + "/boot", headers=headers, timeout=15)
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
                _provisioning_ids.add(pod_id)   # shield from THIS process's reaper
                _provisioning_info[pod_id] = {
                    "model": self.model_key, "gpu": sel["display_name"],
                    "is_spot": sel["is_spot"], "hourly_usd": sel["price"],
                    "console_url": console, "started_at": time.time()}
            # …and from every SIBLING engine's reaper, which cannot see the set
            # above: each engine has its own pools and its own reaper.
            register_pod(pod_id, str(self.model_key), api_key=self.api_key)  # url once ready
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
                url = client.wait_ready(pod_id, port, ready_timeout=boot_to)
                t_port = time.time()
                print(f"[runpod] pod {pod_id}: port open after "
                      f"{t_port - t_created:.0f}s (image pull + container start)",
                      flush=True)
                # Wait for the OpenAI server inside the pod (image pull + model load).
                deadline = time.time() + load_to
                while time.time() < deadline:
                    if _pod_health_ok(url, path=self.health_path,
                                      api_key=self.api_key):
                        print(f"[runpod] pod {pod_id}: serving after "
                              f"{time.time() - t_port:.0f}s more "
                              f"({time.time() - t_created:.0f}s total)", flush=True)
                        self._log_pod_catalogue(url)
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
                self._dump_pod_logs(client, pod_id, "boot failed", url)
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
            # Publish the URL so a sibling engine needing the same pool reuses
            # this pod instead of renting a second GPU for the same work.
            register_pod(pod_id, str(self.model_key), url, self.api_key)
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
                # Before renting: is a sibling process already running a pod for
                # this exact pool? One engine per GPU means the same capability
                # was being rented a card each.
                if sibling_is_provisioning(str(self.model_key)):
                    # Another engine is already booting one for this pool. Wait
                    # for it rather than renting a second card for the same work.
                    print(f"[runpod] another engine is booting a pod for "
                          f"{self.model_key!r} — waiting for it", flush=True)
                    with self._cv:
                        self._provisioning = False
                        self._cv.notify_all()
                        self._cv.wait(15.0)
                    continue
                shared_id, shared_url, shared_key = find_shared_pod(
                    str(self.model_key), self.health_path, self.api_key)
                if shared_url and shared_key and shared_key != self.api_key:
                    # That pod only answers to the token it was launched with.
                    # Adopt the token so this pool — and the pods it rents next —
                    # speak the same one. Only safe while we hold no pod of our
                    # own; otherwise leave it to be reaped rather than end up
                    # with a pool whose pods want two different tokens.
                    with self._cv:
                        adoptable = not self.pods
                    if not adoptable:
                        shared_id, shared_url = None, ""
                    else:
                        self.api_key = shared_key
                        print(f"[runpod] adopting pod {shared_id}'s bearer token",
                              flush=True)
                if shared_url:
                    print(f"[runpod] reusing pod {shared_id} from another engine "
                          f"for {self.model_key!r}", flush=True)
                    handle = PodHandle(pod_id=shared_id, url=shared_url,
                                       hourly_usd=0.0,  # billed by its owner
                                       started_at=time.time(), gpu="(shared)",
                                       healthy=True, last_used=time.time())
                    with self._cv:
                        self.pods.append(handle)
                        self._provisioning = False
                        self._cv.notify_all()
                    continue
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
        if getattr(pod, "gpu", "") == "(shared)":
            # Borrowed from another engine: it owns the lifecycle and the bill.
            # Terminating it here would kill a pod that process is still using.
            print(f"[runpod] releasing borrowed pod {pod.pod_id} ({reason}); its "
                  "owner reaps it", flush=True)
            with self._cv:
                if pod in self.pods:
                    self.pods.remove(pod)
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
            out.append(entry)
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


#: Pods this DEPLOYMENT owns, shared across processes.
#:
#: coderai runs a front plus one engine per GPU. Each engine has its own pool
#: registry AND its own reaper, so a process-local view of "our pods" means every
#: engine sees its siblings' pods as orphans. Observed live: the nvidia engine
#: created a pod and the radeon engine terminated it four seconds later, then the
#: reverse — a mutual kill loop that rented and destroyed pods until stopped.
def _pod_registry_path() -> str:
    try:
        from codai.admin.routes import config_manager
        base = str(getattr(config_manager, "config_dir", "") or "")
    except Exception:
        base = ""
    base = base or os.path.expanduser("~/.coderai")
    return os.path.join(base, "runpod-pods.json")


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


def register_pod(pod_id: str, pool_key: str = "", url: str = "",
                 api_key: str = "") -> None:
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
    _registry_update(lambda d: d.__setitem__(
        str(pod_id), {"pool": str(pool_key), "pid": os.getpid(), "at": time.time(),
                      "url": str(url or ""), "key": str(api_key or "")})
        or True)


def sibling_is_provisioning(pool_key: str) -> bool:
    """True when another process has a pod for this pool that is still booting.

    Registered at creation with no URL yet. Without this the sibling rents its
    own during the ~2.5 minutes a pod takes to boot — which is exactly what
    happened live: two pods for one capability, 45 seconds apart.
    """
    data = _registry_update(lambda d: False)
    if not isinstance(data, dict):
        return False
    mine = os.getpid()
    now = time.time()
    for info in data.values():
        if not isinstance(info, dict) or info.get("pool") != str(pool_key):
            continue
        if info.get("pid") == mine or info.get("url"):
            continue
        # Only while it could plausibly still be booting; a stale entry from a
        # dead process must not block provisioning forever.
        if now - float(info.get("at") or 0) < REAP_GRACE_SECONDS:
            return True
    return False


def find_shared_pod(pool_key: str, health_path: str = "/v1/models",
                    api_key: str = "") -> tuple:
    """A pod another process already has for this pool: (pod_id, url, key).

    ``key`` is the token that pod was launched with, which is not necessarily
    ours — see register_pod. Empty when unknown (an entry from before tokens were
    recorded), in which case the caller's own token is tried.

    coderai runs one engine per GPU, and each has its own pools — so the same
    capability was rented a pod PER ENGINE, paying twice for one job while
    max_pods said 1. Checked against the shared registry and health-probed, so a
    dead entry is never handed out.
    """
    data = _registry_update(lambda d: False)
    if not isinstance(data, dict):
        return None, "", ""
    mine = os.getpid()
    for pod_id, info in data.items():
        if not isinstance(info, dict) or info.get("pool") != str(pool_key):
            continue
        if info.get("pid") == mine:
            continue                       # our own; the local pool handles it
        url = str(info.get("url") or "")
        if not url:
            continue
        key = str(info.get("key") or "") or api_key
        if _pod_health_ok(url, path=health_path, api_key=key):
            return pod_id, url, key
    return None, "", ""


def unregister_pod(pod_id: str) -> None:
    """Forget a pod we terminated, so the registry does not grow without bound."""
    if not pod_id:
        return
    _registry_update(lambda d: d.pop(str(pod_id), None) is not None)


def registered_pod_ids() -> set:
    data = _registry_update(lambda d: False)
    return set(data.keys()) if isinstance(data, dict) else set()


def _known_pod_ids() -> set:
    """Every pod id this DEPLOYMENT owns — not just this process's."""
    ids = set()
    with _pools_lock:
        for pool in _pools.values():
            with pool._cv:
                ids.update(p.pod_id for p in pool.pods)
        ids.update(_provisioning_ids)
    ids.update(registered_pod_ids())
    return ids


#: Never reap a pod younger than this, whatever the registry says. Longer than
#: any boot budget, so a pod still pulling its image cannot be mistaken for a
#: leftover — the backstop for the race a registry cannot close (a process that
#: died between creating a pod and recording it).
REAP_GRACE_SECONDS = float(os.environ.get("CODERAI_RUNPOD_REAP_GRACE", "1200"))


def _pod_age_seconds(pod: dict):
    """Seconds since the pod was created, or None when RunPod does not say."""
    for key in ("uptime_s", "uptimeSeconds", "runtimeSeconds"):
        val = pod.get(key)
        if isinstance(val, (int, float)) and val >= 0:
            return float(val)
    for key in ("createdAt", "created_at"):
        val = pod.get(key)
        if not val:
            continue
        try:
            from datetime import datetime, timezone
            return (datetime.now(timezone.utc) - datetime.fromisoformat(
                str(val).replace("Z", "+00:00"))).total_seconds()
        except Exception:
            continue
    return None


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
        age = _pod_age_seconds(p)
        if age is not None and age < REAP_GRACE_SECONDS:
            continue                      # too young to be a leftover
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
