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

"""Configuration management for coderai."""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from codai.broker.config import BrokerConfig


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "127.0.0.1"
    port: int = 8776
    https: bool = False
    https_key_path: Optional[str] = None
    https_cert_path: Optional[str] = None
    queue_max_size: int = 6
    max_parallel_requests: int = 2
    # Per-engine overrides for max_parallel_requests, keyed by engine name
    # (e.g. {"nvidia": 4, "radeon": 1}). Each engine is a separate process and
    # enforces this on itself, so the default already applies per-engine; the
    # override lets a bigger card run more concurrently than a smaller one. Blank =
    # use the default above.
    max_parallel_requests_overrides: dict = field(default_factory=dict)
    # Per-engine AMD GPU power/clock lock (engine name → level, e.g.
    # {"radeon": "high"}). Applied at engine startup: writes the level to each
    # amdgpu card's power_dpm_force_performance_level. Stabilises Polaris/GCN
    # cards that hang under sustained Vulkan compute due to DPM clock switching.
    # Levels: auto | low | high | manual | profile_standard | profile_peak.
    # Best-effort — an unprivileged container logs the manual host command.
    dpm_force_performance_level_overrides: dict = field(default_factory=dict)
    # Extra environment variables injected per engine (engine name → {VAR: val}),
    # merged into that engine's process env at spawn. For low-level driver tuning
    # without hardcoding — e.g. {"radeon": {"RADV_DEBUG": "syncshaders"}} to
    # serialize RADV shader dispatch and avoid Polaris async compute-ring hangs.
    engine_env_overrides: dict = field(default_factory=dict)
    # Per-engine minimum interval between inference request DISPATCHES, in
    # milliseconds (engine name → ms). 0 / unset = no throttle. Spaces request
    # starts by at least this gap, capping the request rate and inserting idle
    # time between GPU submissions — a stability lever for a marginal card that
    # wedges under sustained back-to-back compute (e.g. {"radeon": 250}).
    engine_request_min_interval_ms: dict = field(default_factory=dict)
    # ─── Frontend/engine split ───────────────────────────────────────────────
    # coderai boots a thin, always-responsive *front* reverse proxy on the public
    # host/port and supervises one or more *engine* subprocesses (which do all
    # GPU/model work) on internal localhost ports. This keeps the web UI responsive
    # while a model loads or generates.
    internal_port_base: int = 8780      # first engine binds here; +1 per extra engine
    engines: int = 0                    # 0 = auto (one per detected GPU, min 1)
    engine_gpus: Optional[list] = None  # explicit GPU indices, e.g. [0, 1]; None = auto
    proxy_status_timeout: float = 4.0   # short timeout for UI/status proxying (seconds);
                                        # generous enough that a GIL-busy engine's
                                        # health poll doesn't time out mid-generation
    proxy_max_inflight: int = 64        # max concurrent proxied requests through the front
    gpu_swap_batch: int = 10            # on a shared GPU (GGUF-isolation split), serve up to
                                        # this many requests for the model that currently owns
                                        # the card before swapping to a queued different model
                                        # (then round-robin back). Prevents cross-engine VRAM
                                        # contention while avoiding per-request model thrash.
    engine_restart_drain_grace: float = 30.0  # on engine restart, wait this many seconds
                                              # for in-flight requests to finish before
                                              # killing the process (0 = bounce immediately)
    # Process-isolate GGUF (llama.cpp) inference from torch/diffusers on NVIDIA.
    # llama.cpp's CUDA backend and PyTorch sharing one process corrupts the CUDA
    # context — after a GGUF model runs, the next torch kernel (e.g. a diffusers
    # text-encoder) dies with "CUDA error: invalid argument". When True (default) and
    # GPUs are auto-detected, each NVIDIA torch engine gets a co-located sibling
    # *gguf* engine on the SAME card (own process → own CUDA context): the torch
    # engine drops the `gguf` capability and serves transformers/diffusers, while the
    # sibling serves GGUF via llama.cpp. Both are real engine subprocesses, so the
    # front's routing, VRAM/eviction and thermal pause/SIGSTOP all apply unchanged.
    # Ignored when `engine_specs` is set (declare the split yourself there).
    isolate_gguf_engine: bool = True
    # Explicit, heterogeneous engine declarations. Auto GPU detection only finds
    # NVIDIA cards and assumes one backend, and CUDA vs Vulkan device enumeration is
    # inconsistent — so for mixed setups (e.g. an NVIDIA + a Radeon card, where the
    # NVIDIA engine also serves GGUF via Vulkan) declare each engine with its own
    # env block. When non-empty this overrides `engines`/`engine_gpus`. Each item:
    #   {
    #     "name": "nvidia",          # label for logs
    #     "backend": "nvidia",       # nvidia | vulkan (forces this engine's backend)
    #     "capabilities": [...],     # optional; defaults from backend (see below)
    #     "env": { "CUDA_VISIBLE_DEVICES": "0", "GGML_VK_VISIBLE_DEVICES": "0",
    #              "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json" }
    #   }
    # Default capabilities: nvidia → ["transformers","gguf"]; vulkan → ["gguf"].
    engine_specs: Optional[list] = None
    # Preferred engine (by name or backend) when a model is compatible with more
    # than one — e.g. a GGUF that could run on either an NVIDIA or a Radeon engine.
    # None = spread to the least-loaded compatible engine. A per-model "engine" set
    # in models.json overrides this for that model.
    default_engine: Optional[str] = None


@dataclass
class BackendConfig:
    """Backend configuration."""
    type: str = "auto"
    image_backend: str = "auto"
    audio_backend: str = "auto"
    tts_backend: str = "auto"


@dataclass
class ModelsConfig:
    """Models configuration."""
    default_load_mode: str = "ondemand"
    hf_cache_dir: Optional[str] = None
    gguf_cache_dir: Optional[str] = None
    max_model_instances: int = 1  # max concurrent instances per model (global default; overridable per-model via "max_instances")
    # Per-engine overrides for max_model_instances, keyed by engine name
    # (e.g. {"nvidia": 2, "radeon": 1}). Applied per-engine process; blank = default.
    max_model_instances_overrides: dict = field(default_factory=dict)
    # Node-wide cap on a reply's max_tokens. A client's requested max_tokens is
    # honored only when it is SMALLER than this; a larger (or absent) request is
    # clamped to this value. None = no cap (use the client's value, or the 2048
    # fallback). Overridable per-model via the models.json entry's "max_tokens".
    max_tokens: Optional[int] = None
    # While a model is loading / not ready, signal "still working" so the channel
    # stays alive and a watching client can show progress: out-of-band broker
    # `pending` keepalives + a non-content SSE status chunk (no message pollution).
    # Global default on; override per-model via the models.json entry's
    # "load_status_updates" (set false to deactivate for that model).
    load_status_updates: bool = True
    # Keepalive sent on the DIRECT streaming API path while a request waits for a
    # front queue slot or the engine's model load, so the client doesn't time out:
    #   "invisible" (default) — empty-content SSE chunk + x_queue_info metadata
    #                           (holds the connection; no message-content pollution)
    #   "visible"             — short visible status text (appears in the content)
    #   "silent"              — keep the connection alive via SSE comments only
    #                           (no chunk, no content, no status) — still prevents a
    #                           client/proxy idle-timeout disconnect while the engine
    #                           is stuck loading
    # When the request has thinking enabled the keepalive is sent on the reasoning
    # channel instead (no pollution), unless the mode is "silent". Global default;
    # override per-model via the models.json entry's "wait_status_mode".
    wait_status_mode: str = "invisible"
    # Degenerate tool-call loop guard: when the incoming history shows the same
    # tool invoked with the same arguments repeatedly (and each attempt failed or
    # the call spilled as un-parsed markup), inject a one-shot system reminder
    # before generation telling the model to stop repeating it. coderai sees the
    # whole history each request, so it can break a loop the agent's own runner
    # didn't. Set repeats<=0 to disable. Overridable per-model via models.json
    # "tool_loop_guard" / "tool_loop_repeats".
    tool_loop_guard: bool = True
    tool_loop_repeats: int = 3
    # gemma-4 native tool-call heuristic (`call:NAME{…}` / `<|tool_call>` markup):
    #   "full"       — parse & strip every call:/response: span (max recall, may
    #                  eat legit `call:foo{…}` text in coding/prose replies);
    #   "restricted" — only treat a span as a call when NAME is a declared tool
    #                  (real calls work; prose/code is left intact) [default];
    #   "off"        — disable the gemma heuristic entirely (for bigger models
    #                  that emit standard structured tool calls).
    # Overridable per-model via the models.json entry's "gemma_tool_parser".
    gemma_tool_parser: str = "restricted"


@dataclass
class OffloadConfig:
    """Offload configuration."""
    directory: str = "./offload"
    strategy: str = "auto"
    max_gpu_percent: Optional[float] = None
    no_ram: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    manual_ram_gb: Optional[float] = None
    flash_attention: bool = False
    # Server-wide ceiling on host RAM (process-tree RSS) the server may use, in GB.
    # None = no global cap (per-load budget = available RAM, as before). When set, new
    # model loads get a CPU-offload budget clamped to the remaining headroom so the
    # overflow spills to the offload directory (disk), and idle models can be evicted.
    max_ram_gb: Optional[float] = None
    evict_idle_on_ram: bool = True   # unload idle LRU models when over the RAM cap
    ram_leak_watch: bool = True      # background watcher samples RSS + auto-mitigates
    # Leak-watch mitigation tuning. The watcher runs a mitigation ladder when RSS
    # crosses ram_watch_soft_fraction of the cap (or a leak is suspected). On a
    # marginal GPU the cross-thread CUDA call in that ladder can be undesirable, so
    # ram_watch_cuda gates whether mitigation is allowed to call torch.cuda.empty_cache().
    ram_watch_poll_seconds: float = 15.0    # how often the watcher samples RSS
    ram_watch_soft_fraction: float = 0.90   # mitigate at/above this fraction of the cap
    ram_watch_cuda: bool = True             # allow mitigation to call CUDA empty_cache()
    # Cross-backend GPU pooling. OFF by default: each engine uses only its own
    # backend's GPUs (CUDA for an NVIDIA engine, Vulkan for a Radeon engine) and a
    # model still splits across multiple SAME-backend cards (e.g. 2× 3090). When ON,
    # an engine may ALSO pool a model across a different backend's card (e.g. NVIDIA
    # 3090 via CUDA + Radeon RX 580 via Vulkan) for more total VRAM — slower, since
    # the weakest card bottlenecks each token.
    gpu_split: bool = False
    # llama.cpp per-device layer ratio when a model is split, in llama.cpp device
    # order (CUDA devices first, then Vulkan). e.g. "0.8,0.2" = 80% on the first GPU
    # (3090), 20% on the second (RX 580). Blank = even split across the devices.
    tensor_split: Optional[str] = None
    # Auto-split strategy when no explicit tensor_split is given:
    #   "vram"        → proportional to each card's free VRAM (max capacity)
    #   "performance" → fill the fast lead card first, spill only the overflow to the
    #                   slower card(s) so the weak GPU gates throughput the least.
    split_strategy: str = "vram"
    # Per-MODEL cap (GB) on how much VRAM the auto-split may place on the secondary
    # (non-main) card. For a 2-card split one cap is enough — which card is secondary
    # is implicit. Lives in each model's config; None/0 = no cap.
    split_secondary_cap_gb: Optional[float] = None
    # GLOBAL per-CARD VRAM caps: {card_key: gb}. Unlike the per-model scalar above,
    # this caps a SPECIFIC physical card by a stable key (nvidia:<uuid> / amd:<pci>)
    # regardless of which engine/model uses it — so you can independently limit, say,
    # the Radeon to 4 GB and the NVIDIA to 20 GB. Effective cap on a device = lowest
    # of its global per-card cap and the per-model secondary cap (when it applies).
    split_card_caps_gb: Dict[str, float] = field(default_factory=dict)


@dataclass
class VulkanConfig:
    """Vulkan backend configuration."""
    n_gpu_layers: int = -1
    n_ctx: int = 2048
    device_id: int = 0
    single_gpu: bool = False


@dataclass
class ImageConfig:
    """Image generation configuration."""
    llm_path: Optional[str] = None
    vae_path: Optional[str] = None
    sample_method: str = "res_multistep"
    steps: int = 4
    width: int = 512
    height: int = 512
    cfg_scale: float = 1.0
    precision: str = "f32"
    cpu_offload: bool = False
    seed: Optional[int] = None
    vae_tiling: bool = False
    clip_on_cpu: bool = False


@dataclass
class WhisperConfig:
    """Whisper ASR configuration."""
    server_path: Optional[str] = None
    server_port: int = 8744


@dataclass
class ArchiveConfig:
    """Generation archive configuration."""
    enabled: bool = True
    directory: str = ""        # empty = <config_dir>/archive; relative paths resolve from config_dir
    retention: str = "never"   # one of: 1h 1d 2d 1w 1m 3m 6m 1y never


@dataclass
class ThermalConfig:
    """Thermal-protection configuration.

    Before running a request against a loaded model, wait until CPU/GPU
    temperatures are within safe limits so a long sequence of heavy
    generations can't overheat the machine and trip its power-off protection.
    Thresholds are in degrees Celsius. CPU and GPU can be toggled separately.
    """
    cpu_enabled: bool = True
    gpu_enabled: bool = True
    cpu_high: float = 90.0      # pause when CPU reaches this temperature
    cpu_resume: float = 87.0    # resume once CPU drops back to/below this
    gpu_high: float = 90.0      # pause when GPU reaches this temperature
    gpu_resume: float = 87.0    # resume once GPU drops back to/below this
    # Per-vendor GPU threshold overrides, e.g. {"amd": {"high": 95, "resume": 92}}.
    # A card uses its vendor's override when present, else the gpu_high/gpu_resume
    # defaults above — so e.g. a Radeon that runs hotter can have a higher limit
    # than an NVIDIA card. Keyed by vendor: "nvidia" | "amd" | "intel".
    gpu_overrides: dict = field(default_factory=dict)
    poll_seconds: float = 5.0   # how often to re-check while cooling down
    # Proactive soft-throttle: before a hard pause, when a sensor enters the warm
    # band [soft_throttle_temp, *_high) insert a short per-step sleep (scaled by
    # how close to the pause threshold) so the temperature climbs slower and the
    # hard cooldown is rarely hit. Caps the heat-rate of a single pegged core.
    soft_throttle_enabled: bool = False
    soft_throttle_temp: float = 80.0       # engage at/above this temperature (°C)
    soft_throttle_max_sleep: float = 3.0   # max seconds to sleep/checkpoint at the limit
    # Front-driven thermal supervision. The front proxy monitors temperatures
    # centrally (it stays responsive even while an engine is GIL-blocked in a long
    # native call) and tells engines to pause/resume cooperatively. When an engine
    # ignores a pause — stuck in a native call it can't interrupt — for
    # `stop_escalate_checks` consecutive monitor checks, the front escalates to an
    # OS-level SIGSTOP (and SIGCONT to resume after cooldown).
    supervisor_enabled: bool = True
    stop_escalate_checks: int = 3


@dataclass
class JobsConfig:
    """Background-job (LoRA training) configuration."""
    # When True, an interrupted training job (process restart) is left
    # 'interrupted' so it can resume from its on-disk checkpoint. When False,
    # such jobs are marked 'cancelled' on startup and not auto-resumed (their
    # checkpoints are kept, so they can be restarted manually from the Tasks
    # page). The --no-resume-jobs CLI flag forces this off for one run.
    resume_on_restart: bool = True


@dataclass
class EnhanceConfig:
    """Video enhancement (upscale / FPS interpolation) tool policy.

    By default these run fully in-process on torch models (ESRGAN upscaler, RIFE/
    FILM interpolator) — no subprocess, no ffmpeg. The flags below OPT IN to the
    external tools as alternatives when no model is configured/preferred."""
    allow_ffmpeg: bool = False        # allow ffmpeg (frame I/O / minterpolate) instead of PyAV+model
    allow_rife_ncnn: bool = False     # allow the external rife-ncnn-vulkan binary instead of a torch model


@dataclass
class CompactionConfig:
    """Global defaults for auto-compaction of an over-long chat history.

    Per-model settings in a models.json entry (``auto_compact``,
    ``auto_compact_pct``, ``auto_compact_strategy``, ``auto_compact_model``,
    ``auto_compact_tolerance_pct``) OVERRIDE the values here; when a model leaves
    one unset, the global default below applies. ``model`` selects which model
    performs the summarization for the ``summarize`` strategy — empty means use
    the same model that serves the request. Pointing it at a smaller/faster model
    lets that model summarize the old turns while the big model answers; the
    dropped history is chunked to fit the chosen summarizer's own context window
    before it is summarized.

    The over-context check counts the prompt PLUS the request's ``max_tokens``
    (the reply is generated into the same window). When that total overflows
    ``n_ctx`` by no more than ``tolerance_pct`` we leave history alone and just
    trim ``max_tokens`` to fit (lossless); beyond that we compact, and as a last
    resort slice any single message still larger than the target. ``min_output``
    is the smallest reply (in tokens) we always try to leave room for."""
    enabled: bool = False
    pct: int = 85                        # compact when the prompt reaches this % of n_ctx
    strategy: str = "drop_oldest"        # drop_oldest | keep_head_tail | summarize
    model: str = ""                      # model id/alias that summarizes; "" = same as request
    tolerance_pct: int = 20              # accept prompt+max_tokens up to this % over n_ctx (trim instead of compact)
    min_output: int = 512                # always try to leave at least this many tokens for the reply
    estimate_safety: float = 1.15        # inflate the chars/4 prompt estimate by this factor for fit/trim (it undercounts code/JSON)


@dataclass
class RemotesConfig:
    """Serve whole capabilities from another coderai instead of locally.

    Every subsystem that loads its model in-process — images, video, embeddings,
    rerank, OCR, TTS, STT, voice cloning, audio generation, stems, 3D, pipelines
    — has no service of its own to redirect. But another coderai exposes the same
    API, so it can BE the service: map a capability to a base URL here and
    :class:`~codai.api.remote_gateway.RemoteGatewayMiddleware` replays matching
    /v1 requests there and streams the answer back.

    A model's own ``service_url`` (models.json) wins over this map. Chat and
    completions are not routed here — they go through RemoteOpenAIBackend.

    Example::

        "remotes": {"enabled": true,
                    "endpoints": {"images": "http://gpu-box:8000",
                                  "video": "https://pod-xyz-8000.proxy.runpod.net"}}
    """
    enabled: bool = True
    api_key: str = ""                        # bearer token for the remote(s)
    max_body_mb: int = 512                   # bigger requests are served locally
    endpoints: dict = field(default_factory=dict)   # capability -> base URL
    # capability -> RunPod pod block (same fields as a model's `runpod` block).
    # coderai then provisions, health-checks, scales and reaps that pod itself,
    # so an image or video capability gets the budgets and idle teardown a
    # RunPod-served LLM has. `"endpoints": {"images": "runpod"}` is shorthand
    # for "use the pod block for images" (defaults apply when there is none).
    pods: dict = field(default_factory=dict)


@dataclass
class Ds4Config:
    """DeepSeek V4 via ds4 (antirez/DwarfStar) external-worker configuration.

    ds4 is a native inference engine built specifically for DeepSeek V4 that exposes
    an OpenAI-compatible HTTP server (``ds4-server``). When ``enabled``, coderai owns
    the whole lifecycle: on first use it clones + builds ds4, downloads the chosen
    GGUF weight variant, launches ``ds4-server`` as a managed subprocess, and proxies
    text requests to it. Any requested model whose name matches ``model_id`` (or
    contains ``deepseek-v4``) is routed to ds4 instead of the normal backends.
    """
    enabled: bool = False
    # Point this at an engine already running somewhere else (another host, a
    # container, a rented pod) and coderai proxies to it instead of building,
    # downloading or launching anything locally. Blank = manage it here.
    service_url: str = ""
    repo_url: str = "https://github.com/antirez/ds4"
    install_dir: Optional[str] = None      # None = ~/.coderai/ds4
    build_target: str = "auto"             # auto|cuda-generic|cuda-spark|metal|cpu
    # The model ds4-server loads. Preferred: serve a deepseek4 GGUF the user
    # already has — the requested model's own path is used when it resolves to a
    # local .gguf, else `model_path` (an explicit override), else the variant is
    # downloaded as a last resort. So you normally DON'T set model_variant at all.
    model_path: str = ""                   # explicit GGUF for ds4-server -m (overrides the download)
    auto_download: bool = False            # OFF by default: only download a variant when explicitly opted in
    model_variant: str = "q4-imatrix"      # download_model.sh variant (used only when auto_download is on)
    model_id: str = "deepseek-v4"          # model id/alias that routes to ds4
    host: str = "127.0.0.1"
    port: int = 0                          # 0 = auto-pick a free port
    ctx: int = 100000                      # ds4-server --ctx context window
    ssd_streaming: bool = False            # ds4-server --ssd-streaming: stream experts from SSD/disk
    extra_args: str = ""                   # extra flags passed to ds4-server
    # VRAM (GiB) ds4-server keeps free for non-cache use on CUDA, exported as
    # DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB. ds4 defaults this to half the
    # card, which over-reserves for small-weight MoE models and starves the
    # streaming expert cache. 0 = leave ds4's default. Set just above the model's
    # resident weights (+~2 GiB headroom) to hand the rest to the expert cache.
    expert_cache_reserve_gb: int = 0
    # Free-form environment for ds4-server, as whitespace/newline-separated
    # KEY=VALUE pairs. ds4 exposes many CUDA tunables only via env, e.g.
    # DS4_CUDA_WEIGHT_ARENA_CHUNK_MB (default 1792) — lower it (e.g. 512) so the
    # model-weight arena allocates in smaller chunks that fit a heap fragmented by
    # the streaming expert cache, avoiding "model arena alloc failed … OOM".
    extra_env: str = ""
    auto_build: bool = True                # clone+build the binary if it's missing
    # On-disk KV-checkpoint cache (ds4-server --kv-disk-dir, defaulted to
    # <offload>/ds4-kv). ds4 writes prompt KV checkpoints here to reuse across
    # requests; abandoned sessions accumulate and never self-prune. When enabled, a
    # background janitor deletes cache entries whose newest file is older than
    # kv_cache_max_age_hours every kv_cache_cleanup_interval_minutes. Active
    # sessions (recently touched) are spared (age is by newest mtime).
    kv_cache_cleanup_enabled: bool = False
    kv_cache_max_age_hours: float = 168.0          # 7 days
    kv_cache_cleanup_interval_minutes: float = 360.0  # 6 hours


@dataclass
class ColibriConfig:
    """GLM-5.2 via colibri (JustVugg/colibri) embedded-engine configuration.

    colibri is a pure-C MoE inference engine for GLM-5.2 that streams experts from
    disk. Unlike ds4 it ships *no* server we keep running — its Python launcher is
    only a thin gateway. So coderai drives the C engine binary (``colibri``) DIRECTLY
    over its stdin/stdout "mux" wire protocol (SUBMIT/DATA/DONE, see
    ``docs/serve_protocol.md``): we own the build, the process, the GLM-5.2 chat
    template and the protocol client — no colibri Python at runtime.

    The model is a *directory* container (int4 g64 + int8 MTP, ~372 GB), NOT a
    single file — so routing matches by ``model_id``/alias/``model_path`` (the
    container dir), not by GGUF architecture the way ds4 does. When ``enabled``, any
    requested model whose name matches ``model_id`` (or contains ``glm-5.2`` /
    ``colibri``) is routed to the colibri engine instead of the normal backends.
    """
    enabled: bool = False
    # Point this at an engine already running somewhere else (another host, a
    # container, a rented pod) and coderai proxies to it instead of building,
    # downloading or launching anything locally. Blank = manage it here.
    service_url: str = ""
    repo_url: str = "https://github.com/JustVugg/colibri"
    install_dir: Optional[str] = None      # None = ~/.coderai/colibri
    build_target: str = "auto"             # auto|cuda|hip|cpu (auto: CUDA if nvcc present)
    # The GLM-5.2 int4 container directory colibri loads (engine env SNAP=<dir>).
    # Preferred: point the requested model's own path at the container; else this
    # explicit override is used. There is no auto-download of the 372 GB container.
    model_path: str = ""                   # explicit container dir (overrides per-model path)
    model_id: str = "glm-5.2-colibri"      # model id/alias that routes to colibri
    ctx: int = 100000                      # advisory context window (engine NGEN/KV sizing)
    kv_slots: int = 1                       # engine KV_SLOTS (1–16): concurrent cached conversations
    cap: int = 8                            # engine positional "cap" arg (worker thread cap)
    # VRAM (GiB) of resident experts colibri pins on CUDA, exported as CUDA_EXPERT_GB.
    # "" = leave colibri's default (auto). "all" pins every expert it can fit.
    cuda_expert_gb: str = ""
    extra_args: str = ""                   # reserved: extra positional/flag args to the engine
    # Free-form environment for the colibri engine, whitespace/newline-separated
    # KEY=VALUE pairs. colibri exposes its tuning ONLY via env (see docs/ENVIRONMENT.md):
    # COLI_MODEL_MIRROR, COLI_DISK_WEIGHTS, COLI_NUMA, COLI_CUDA_PIPE, DIRECT, PIPE,
    # PILOT, DRAFT, SPEC_PIN, GRAMMAR, etc.
    extra_env: str = ""
    auto_build: bool = True                # clone+build the binary if it's missing


@dataclass
class K3Config:
    """Kimi-K3 via kimi-k3-in-c (FareedKhan-dev) native-engine configuration.

    kimi-k3-in-c is a portable-C CPU inference engine for Kimi-K3 (2.78T params, 16/896
    experts active) that streams the always-on dense trunk + routed experts from disk,
    running the full model in as little as ~8 GB RAM. Upstream it is a one-shot batch
    CLI; coderai applies ``packaging/patch-k3.py`` to add a resident serve loop that
    speaks the SAME mux stdin/stdout protocol as colibri, so the existing
    :class:`~codai.api.colibri_worker.MuxEngine` drives it. coderai owns the build, the
    process, the Kimi-K3 chat template (rendered XTML string) and the protocol client.

    The model is a *directory* (the HF checkpoint, ~1.56 TB) plus a packed dense
    ``trunk`` produced by the repo's ``scripts/pack-trunk.sh`` (~109 GB). Routing
    matches by ``model_id``/alias/``model_path`` or a ``backend: "k3"`` pin. There is no
    auto-download of the multi-TB checkpoint. CPU-only: needs AVX2 + FMA and ~1.7 TB of
    fast local storage.
    """
    enabled: bool = False
    # Point this at an engine already running somewhere else (another host, a
    # container, a rented pod) and coderai proxies to it instead of building,
    # downloading or launching anything locally. Blank = manage it here.
    service_url: str = ""
    repo_url: str = "https://github.com/FareedKhan-dev/kimi-k3-in-c"
    install_dir: Optional[str] = None      # None = ~/.coderai/kimi-k3-in-c
    model_path: str = ""                   # the Kimi-K3 checkpoint directory (config + tokenizer + shards)
    trunk_dir: str = ""                    # packed dense trunk dir (scripts/pack-trunk.sh); "" = resident (needs ~114 GB RAM)
    tok_dir: str = ""                      # tokenizer dir; "" = use the checkpoint (model_path) dir
    model_id: str = "kimi-k3"              # model id/alias that routes to k3
    preset: str = ""                       # laptop|desktop|workstation|server|max (blank = use trunk_gb/cache_gb)
    trunk_gb: float = 16.0                 # GiB of the packed trunk pinned resident (--trunk-gb)
    cache_gb: float = 64.0                 # GiB expert LRU cache (--cache-gb)
    ctx: int = 4096                        # serve context capacity (env K3_MAXT)
    extra_args: str = ""                   # extra flags passed to the k3 binary
    extra_env: str = ""                    # free-form KEY=VALUE env (K3_EXPERT_GB, K3_BITS, K3_DIRECT, K3_PIPE, …)
    auto_build: bool = True                # clone+patch+build the binary if it's missing


@dataclass
class KtransformersConfig:
    """ktransformers (kvcache-ai) via SGLang — CPU+GPU heterogeneous engine config.

    ktransformers is a CPU+GPU heterogeneous MoE inference engine (Intel AMX/AVX512/AVX2
    CPU kernels for quantized experts + GPU for the dense trunk/attention). It exposes an
    OpenAI-compatible HTTP server through SGLang, so — like ds4 — coderai owns the whole
    lifecycle: it launches ``python -m sglang.launch_server`` as a managed subprocess and
    proxies ``/v1/chat/completions`` to it. It can serve many families (DeepSeek-V3/R1/V4,
    Kimi-K2/K2.5, Qwen3, GLM-5/5.2, MiniMax), so it is selected PER MODEL via an explicit
    ``backend: "kt"`` pin or the configured ``model_id`` alias — never by a broad name
    marker (it would collide with every other engine).

    Heavy dependencies (SGLang + kt-kernel, built once) — install them out of band and
    point ``model_path`` at the HF model dir and ``kt_weight_path`` at the KT quantized
    weights. Best throughput needs AMX/AVX-512 CPUs.
    """
    enabled: bool = False
    # Point this at an engine already running somewhere else (another host, a
    # container, a rented pod) and coderai proxies to it instead of building,
    # downloading or launching anything locally. Blank = manage it here.
    service_url: str = ""
    repo_url: str = "https://github.com/kvcache-ai/ktransformers"
    install_dir: Optional[str] = None      # None = ~/.coderai/ktransformers (kt-kernel build)
    model_path: str = ""                   # HF model directory (--model)
    kt_weight_path: str = ""               # KT quantized weights directory (--kt-weight-path)
    model_id: str = "ktransformers"        # model id/alias that routes to kt (and --served-model-name)
    host: str = "127.0.0.1"                # SGLang bind host
    port: int = 0                          # 0 = auto-pick a free port
    ctx: int = 32768                       # context length (--context-length)
    extra_args: str = ""                   # extra flags for sglang.launch_server (e.g. --tp-size, KT knobs)
    extra_env: str = ""                    # free-form KEY=VALUE env for the subprocess
    auto_build: bool = False               # pip-install SGLang+kt-kernel if missing (heavy; off by default)


@dataclass
class VllmConfig:
    """vLLM high-concurrency backend — the vLLM OpenAI server, driven as a subprocess.

    vLLM (https://github.com/vllm-project/vllm) provides continuous batching + paged KV for
    far higher aggregate throughput than serialized single-instance backends. Like ds4/kt,
    coderai launches ``python -m vllm.entrypoints.openai.api_server`` as a managed
    subprocess and proxies ``/v1/chat/completions`` to it (:mod:`codai.backends.vllm`).

    vLLM pins its own torch/CUDA (e.g. torch 2.13 / cu13), which conflicts with the main
    coderai venv, so it runs in an ISOLATED venv (``venv``; blank → baked
    /opt/coderai/vllm_venv, else the /cache mount, else ~/.coderai/vllm_venv), built from
    requirements-vllm.txt.

    Like the nvidia/radeon engines, vLLM serves MODELS FROM THE MODEL LIST: tag a model
    entry with ``backend: "vllm"`` and vLLM serves that model using its own ``path`` (each
    served under its own name). ``model_id``/``model_path`` below are OPTIONAL — only a
    convenience for serving a single model that has no model-list entry (ds4/kt style), and
    the OCR subsystem passes its own model. Leave ``model_id`` blank for the model-list flow.
    Also reused by the OCR subsystem to serve Surya2 (a VLM) with continuous batching.
    """
    enabled: bool = False
    # Point this at an engine already running somewhere else (another host, a
    # container, a rented pod) and coderai proxies to it instead of building,
    # downloading or launching anything locally. Blank = manage it here.
    service_url: str = ""
    venv: str = ""                       # isolated venv dir; blank = auto (baked/cache/home)
    model_path: str = ""                 # OPTIONAL single-model HF dir/repo id (--model); blank = use the model list
    model_id: str = ""                   # OPTIONAL single-model alias/served-name; blank = per-model from the list
    gpu: str = ""                        # CUDA device(s) for this vLLM instance (CUDA_VISIBLE_DEVICES,
                                         # e.g. "0" or "0,1"); blank = all visible NVIDIA GPUs. CUDA-only.
    host: str = "127.0.0.1"
    port: int = 0                        # 0 = auto-pick a free port
    ctx: int = 32768                     # --max-model-len
    gpu_memory_utilization: float = 0.90  # --gpu-memory-utilization
    tensor_parallel_size: int = 1        # --tensor-parallel-size
    max_num_seqs: int = 0                # --max-num-seqs (0 = vLLM default)
    dtype: str = ""                      # --dtype (blank = auto; e.g. bfloat16/float16)
    quantization: str = ""               # --quantization (blank = none; e.g. awq/gptq/fp8)
    extra_args: str = ""                 # extra flags for the api_server
    extra_env: str = ""                  # free-form KEY=VALUE env for the subprocess
    auto_build: bool = False             # create the isolated venv + pip install vllm if missing


@dataclass
class RunpodConfig:
    """RunPod remote-GPU backend — account-level settings.

    RunPod (https://runpod.io) rents GPUs by the second, either as **Pods** (full GPU
    containers coderai provisions, load-balances and scales itself) or **Serverless**
    endpoints (RunPod autoscales; coderai just proxies). A model is sent to RunPod by
    tagging its models.json entry with ``backend: "runpod"`` plus a per-model ``runpod``
    block (mode/gpu/price/scaling — see codai.api.runpod_worker). This object holds only
    the account-wide settings shared by every RunPod model.

    The RunPod backend uses NO local VRAM: it runs as a thin HTTP proxy inside the
    primary engine (like ds4/kt) and forwards OpenAI requests to the remote pod/endpoint
    (:mod:`codai.backends.runpod`, :mod:`codai.api.runpod_worker`).
    """
    enabled: bool = False
    api_key: str = ""                     # RunPod API key (masked in the UI, never logged)
    cloud_type: str = "SECURE"            # SECURE | COMMUNITY (default; per-model can override)
    api_base: str = "https://api.runpod.io/graphql"   # GraphQL endpoint (override for testing)
    rest_base: str = "https://rest.runpod.io/v1"      # REST endpoint (pod lifecycle)
    serverless_base: str = "https://api.runpod.ai/v2"  # serverless invoke base
    default_gpu_type: str = ""            # fallback RunPod gpuTypeId when a model gives none
    data_center: str = ""                 # optional data-center id filter (blank = any)
    # RunPod "Container Registry Credentials" id, for pods whose image lives in a
    # private registry (your own coderai image, typically). Per-model/capability
    # `registry_auth_id` overrides this. Blank = public images only.
    registry_auth_id: str = ""
    # A RunPod network volume (created by hand in the RunPod console) that pods
    # attach at launch. Weights downloaded onto it survive the pod, so the second
    # pod does not re-download a 30 GB model — and it doubles as the place to put
    # uploaded models and LoRA adapters, since every pod sees the same files.
    # Constraints RunPod imposes: Secure Cloud only, and the pod MUST run in the
    # volume's own data center.
    network_volume_id: str = ""
    volume_mount_path: str = "/workspace"    # where pods mount it
    # Stable tag baked into every pod name (coderai-<deployment_id>-<model>-<rand>) so the
    # reaper can identify OUR pods across restarts and never touch another deployment's.
    deployment_id: str = "default"
    # Account-wide safety net: the scaler will not let the SUM of all running RunPod
    # pods' hourly cost exceed this. 0 = no global cap (per-model $/hr still applies).
    global_max_hourly_usd: float = 0.0
    # Account-wide CUMULATIVE spend cap over a rolling period (distinct from the $/hr
    # rate cap above): the scaler refuses to provision when trailing spend across ALL
    # RunPod models would exceed this. 0 / "unlimited" = no cap.
    global_cost_limit_usd: float = 0.0
    global_cost_period: str = "unlimited"   # hour | day | week | month | unlimited


@dataclass
class OcrConfig:
    """Dedicated OCR subsystem configuration.

    coderai's OCR is done by PURPOSE-BUILT OCR engines (detection + recognition),
    NOT a vision LLM — far faster, GPU-batchable, and it returns faithful text with
    line/word bounding boxes and layout instead of hallucinating. Three engines are
    supported and selectable per request (``engine`` field on ``/v1/ocr``), falling
    back to ``default_engine``:

    - ``paddle``  — PaddleOCR + PP-Structure (Apache-2.0; layout + tables; recommended)
    - ``doctr``   — Mindee docTR (Apache-2.0; pure-PyTorch; easy install)
    - ``surya``   — Surya (best layout/reading-order; LICENSE-GATED — GPL/commercial:
      only loaded when ``surya_accept_license`` is True)

    Heavy OCR dependencies are installed out of band (optional) so the base image
    stays lean. Concurrency on a single GPU comes from loading multiple instances of
    an engine (``*_instances``) and fanning requests across them (``max_concurrency``);
    a continuous-batching path via vLLM is a deferred option (see docs/vllm.md).

    Per document coderai can produce: (1) plain-text transcription, (2) structured
    JSON of fields via an existing coderai text model (``extract_*``), and (3)
    stamp/signature flags (``detect_*``).
    """
    enabled: bool = False
    default_engine: str = "paddle"        # paddle|doctr|surya
    dpi: int = 200                        # PDF rasterisation DPI
    max_concurrency: int = 4             # global cap on in-flight OCR pages
    lang: str = "it"                     # default document language

    # --- PaddleOCR / PP-Structure ---
    paddle_enabled: bool = True
    paddle_use_gpu: bool = True
    paddle_instances: int = 2            # copies loaded for concurrent OCR
    paddle_structure: bool = True        # PP-Structure layout + tables + reading order
    paddle_lang: str = "it"             # PaddleOCR lang code (it / latin / …)
    paddle_det_model_dir: str = ""       # override detection model dir (blank = default)
    paddle_rec_model_dir: str = ""       # override recognition model dir (blank = default)
    # PaddleOCR runs in an ISOLATED venv subprocess (opencv-contrib clash + bundled CUDA).
    paddle_venv: str = ""               # isolated venv dir; blank = ~/.coderai/paddle_venv
    paddle_auto_build: bool = False     # create the venv + pip install requirements-ocr-paddle.txt on first use

    # --- docTR (Mindee) --- runs IN-PROCESS (uses the main venv's torch)
    doctr_enabled: bool = False
    doctr_use_gpu: bool = True
    doctr_instances: int = 1
    doctr_det_arch: str = "db_resnet50"
    doctr_reco_arch: str = "crnn_vgg16_bn"

    # --- Surya (LICENSE-GATED) --- runs in an ISOLATED venv subprocess (pillow<11 clash)
    surya_enabled: bool = False
    surya_accept_license: bool = False   # MUST be explicitly set (GPL — compatible with coderai GPLv3)
    surya_instances: int = 1
    surya_langs: str = "it"
    surya_venv: str = ""               # isolated venv dir; blank = ~/.coderai/surya_venv
    surya_auto_build: bool = False     # create the venv + pip install requirements-surya.txt on first use
    # Surya serving mode: "local" = classic det+recognition on torch in the isolated venv
    # (surya-ocr <=0.17); "vllm"/"llamacpp" = the latest "Surya2" VLM served by an external
    # OpenAI server that Surya attaches to (SURYA_INFERENCE_URL). "vllm" reuses coderai's
    # vLLM backend to serve `surya_model` (continuous batching); "llamacpp" attaches to a
    # llama.cpp server at `surya_server_url`.
    surya_serve: str = "local"         # local | vllm | llamacpp
    surya_model: str = "datalab-to/surya-ocr-2"   # HF checkpoint for the served (vllm) backend
    surya_server_url: str = ""         # external OpenAI server URL (llamacpp/manual); blank = auto

    # --- stamp / signature detection ---  [O3]
    detect_mode: str = "off"             # off|layout|detector|both
    detect_model_path: str = ""          # YOLO signature/stamp weights (detector/both modes)
    detect_conf: float = 0.35            # detector confidence threshold

    # --- structured field extraction (via an existing coderai text model) ---  [O4]
    extract_enabled: bool = False
    extract_model_id: str = ""           # coderai text model id/alias used for extraction
    extract_max_tokens: int = 2048
    # Default extraction schema SPEC — not a hardcoded schema. Accepts a named schema
    # (e.g. "italian_sentenza"; files in <config_dir>/ocr_schemas/ + built-in seeds),
    # inline JSON (starts with '{'/'['), or blank/"auto" for generic key/value
    # extraction. Overridable per request via the `schema` field. See codai/ocr/schemas.py.
    extract_schema: str = ""
    extract_validate: bool = True        # validate output against JSON Schema (jsonschema-typed schemas only)


@dataclass
class Config:
    """Main configuration class."""
    version: str = "1.0"
    server: ServerConfig = field(default_factory=ServerConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    vulkan: VulkanConfig = field(default_factory=VulkanConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    jobs: JobsConfig = field(default_factory=JobsConfig)
    enhance: EnhanceConfig = field(default_factory=EnhanceConfig)
    remotes: RemotesConfig = field(default_factory=RemotesConfig)
    ds4: Ds4Config = field(default_factory=Ds4Config)
    colibri: ColibriConfig = field(default_factory=ColibriConfig)
    k3: K3Config = field(default_factory=K3Config)
    ktransformers: KtransformersConfig = field(default_factory=KtransformersConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    runpod: RunpodConfig = field(default_factory=RunpodConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    system_prompt: Optional[str] = None
    tools_closer_prompt: bool = False
    grammar_guided: bool = False
    file_path: Optional[str] = None
    # Base directory for temporary working files (frame extraction, upscaling,
    # interpolation, etc.). None/empty = the OS default (usually /tmp). Point it at
    # a large-capacity volume when /tmp is small — 4× upscaling extracts many large
    # frames and can exhaust a small /tmp ("No space left on device").
    tmp_dir: Optional[str] = None
    # Periodic cleanup of the temporary-working dir (above). A background janitor
    # deletes entries older than tmp_cleanup_max_age_hours every
    # tmp_cleanup_interval_minutes. Guards against runaway tmp growth from
    # delete=False temp files left by interrupted generations. Only runs when a
    # dedicated tmp_dir is configured (never prunes a bare system /tmp).
    tmp_cleanup_enabled: bool = True
    tmp_cleanup_max_age_hours: float = 24.0
    tmp_cleanup_interval_minutes: float = 60.0
    hf_chat_templates: list = field(default_factory=list)
    reasoning_options: list = field(default_factory=list)
    parser: str = "auto"


class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    def __init__(self, config_dir: str):
        """Initialize the configuration manager.
        
        Args:
            config_dir: Path to the configuration directory
        """
        self.config_dir = Path(config_dir).expanduser()
        self.config_path = self.config_dir / "config.json"
        self.models_path = self.config_dir / "models.json"
        self.auth_path = self.config_dir / "auth.json"
        self.pipelines_path = self.config_dir / "pipelines.json"
        
        self.config: Optional[Config] = None
        self.models_data: Dict[str, Any] = {}
        self.auth_data: Dict[str, Any] = {}
        self.pipelines_data: list = []
    
    def ensure_config_dir(self):
        """Create configuration directory if it doesn't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def create_default_configs(self):
        """Create default configuration files."""
        self.ensure_config_dir()
        
        # Create default config.json
        if not self.config_path.exists():
            default_config = {
                "version": "1.0",
                "server": {
                    "host": "127.0.0.1",
                    "port": 8776,
                    "https": False,
                    "https_key_path": None,
                    "https_cert_path": None
                },
                "backend": {
                    "type": "auto",
                    "image_backend": "auto",
                    "audio_backend": "auto",
                    "tts_backend": "auto"
                },
                "models": {
                    "default_load_mode": "ondemand"
                },
                "offload": {
                    "directory": "./offload"
                },
                "broker": {
                    "enabled": False,
                    "base_url": "",
                    "scope": "user",
                    "username": "",
                    "provider_id": "",
                    "client_id": "",
                    "registration_token": "",
                    "advertised_endpoint": "",
                    "transport": "websocket",
                    "heartbeat_interval_seconds": 30,
                    "connect_timeout_seconds": 10,
                    "request_timeout_seconds": 30,
                    "reconnect_initial_delay_seconds": 1,
                    "reconnect_max_delay_seconds": 60
                },
                "system_prompt": None,
                "tools_closer_prompt": False,
                "grammar_guided": False,
                "file_path": None,
                "hf_chat_templates": [],
                "reasoning_options": [],
                "parser": "auto"
            }
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"Created default config: {self.config_path}")
        
        # Create default models.json
        if not self.models_path.exists():
            default_models = {
                "text_models": [],
                "image_models": [],
                "audio_models": [],
                "vision_models": [],
                "tts_models": [],
                "gguf_models": [],
                "loaded": [],
                "preload": [],
                "unloaded": [],
                "aliases": {}
            }
            with open(self.models_path, 'w') as f:
                json.dump(default_models, f, indent=2)
            print(f"Created default models config: {self.models_path}")
        
        # Create default auth.json
        if not self.auth_path.exists():
            from codai.admin.auth import hash_password
            default_auth = {
                "users": [{
                    "id": 1,
                    "username": "admin",
                    "password_hash": hash_password("admin"),
                    "role": "admin",
                    "created_at": "2026-05-03T00:00:00Z",
                    "must_change_password": True
                }],
                "tokens": [],
                "sessions": {}
            }
            with open(self.auth_path, 'w') as f:
                json.dump(default_auth, f, indent=2)
            print(f"Created default auth config: {self.auth_path}")
            print(f"\nDefault credentials: admin / admin")
            print("IMPORTANT: Change this password immediately after first login.\n")
    
    def load(self) -> Config:
        """Load configuration from files.
        
        Returns:
            Config object with loaded settings
        """
        # Create defaults if config directory is empty or doesn't exist
        self.create_default_configs()
        
        # Load config.json
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Parse into Config dataclass. Use a tolerant constructor (_dc) that
            # drops unknown keys: a stale or newer-version config.json must NEVER
            # crash the whole load, which would silently reset ALL settings to
            # defaults (the "had to reconfigure everything" bug).
            import dataclasses as _dataclasses

            def _dc(cls, data):
                if not isinstance(data, dict):
                    return cls()
                known = {f.name for f in _dataclasses.fields(cls)}
                extra = [k for k in data if k not in known]
                if extra:
                    print(f"[config] ignoring unknown {cls.__name__} keys: {extra}")
                return cls(**{k: v for k, v in data.items() if k in known})

            self.config = Config(
                version=config_data.get("version", "1.0"),
                server=_dc(ServerConfig, config_data.get("server", {})),
                backend=_dc(BackendConfig, config_data.get("backend", {})),
                models=_dc(ModelsConfig, config_data.get("models", {})),
                offload=_dc(OffloadConfig, config_data.get("offload", {})),
                vulkan=_dc(VulkanConfig, config_data.get("vulkan", {})),
                image=_dc(ImageConfig, config_data.get("image", {})),
                whisper=_dc(WhisperConfig, config_data.get("whisper", {})),
                archive=_dc(ArchiveConfig, config_data.get("archive", {})),
                thermal=_dc(ThermalConfig, config_data.get("thermal", {})),
                jobs=_dc(JobsConfig, config_data.get("jobs", {})),
                enhance=_dc(EnhanceConfig, config_data.get("enhance", {})),
                remotes=_dc(RemotesConfig, config_data.get("remotes", {})),
                ds4=_dc(Ds4Config, config_data.get("ds4", {})),
                colibri=_dc(ColibriConfig, config_data.get("colibri", {})),
                k3=_dc(K3Config, config_data.get("k3", {})),
                ktransformers=_dc(KtransformersConfig, config_data.get("ktransformers", {})),
                vllm=_dc(VllmConfig, config_data.get("vllm", {})),
                runpod=_dc(RunpodConfig, config_data.get("runpod", {})),
                ocr=_dc(OcrConfig, config_data.get("ocr", {})),
                compaction=_dc(CompactionConfig, config_data.get("compaction", {})),
                broker=_dc(BrokerConfig, config_data.get("broker", {})),
                system_prompt=config_data.get("system_prompt"),
                tools_closer_prompt=config_data.get("tools_closer_prompt", False),
                grammar_guided=config_data.get("grammar_guided", False),
                file_path=config_data.get("file_path"),
                tmp_dir=config_data.get("tmp_dir"),
                tmp_cleanup_enabled=config_data.get("tmp_cleanup_enabled", True),
                tmp_cleanup_max_age_hours=config_data.get("tmp_cleanup_max_age_hours", 24.0),
                tmp_cleanup_interval_minutes=config_data.get("tmp_cleanup_interval_minutes", 60.0),
                hf_chat_templates=config_data.get("hf_chat_templates", []),
                reasoning_options=config_data.get("reasoning_options", []),
                parser=config_data.get("parser", "auto")
            )
        else:
            self.config = Config()
        
        # Load models.json
        if self.models_path.exists():
            with open(self.models_path, 'r') as f:
                self.models_data = json.load(f)
        else:
            self.models_data = {
                "text_models": [],
                "image_models": [],
                "audio_models": [],
                "vision_models": [],
                "tts_models": [],
                "gguf_models": [],
                "loaded": [],
                "preload": [],
                "unloaded": [],
                "aliases": {}
            }
        
        # A pod rented for one model starts with an empty catalogue and would
        # refuse every request for it. CODERAI_SEED_MODELS carries the entries it
        # must know about, merged in at startup (existing entries win, so this
        # can never overwrite a real deployment's catalogue).
        self._seed_models_from_env()
        self._enable_subsystems_from_env()

        # Load auth.json
        if self.auth_path.exists():
            with open(self.auth_path, 'r') as f:
                self.auth_data = json.load(f)
        else:
            self.auth_data = {
                "users": [],
                "tokens": [],
                "sessions": {}
            }

        # Load pipelines.json
        if self.pipelines_path.exists():
            with open(self.pipelines_path, 'r') as f:
                self.pipelines_data = json.load(f)
        else:
            self.pipelines_data = []
        
        return self.config
    
    def save_config(self):
        """Save config.json to disk."""
        config_dict = {
            "version": self.config.version,
            "server": {
                "host": self.config.server.host,
                "port": self.config.server.port,
                "https": self.config.server.https,
                "https_key_path": self.config.server.https_key_path,
                "https_cert_path": self.config.server.https_cert_path,
                "queue_max_size": self.config.server.queue_max_size,
                "max_parallel_requests": self.config.server.max_parallel_requests,
                "max_parallel_requests_overrides": self.config.server.max_parallel_requests_overrides,
                "dpm_force_performance_level_overrides": self.config.server.dpm_force_performance_level_overrides,
                "internal_port_base": self.config.server.internal_port_base,
                "engines": self.config.server.engines,
                "engine_gpus": self.config.server.engine_gpus,
                "proxy_status_timeout": self.config.server.proxy_status_timeout,
                "proxy_max_inflight": self.config.server.proxy_max_inflight,
                "gpu_swap_batch": self.config.server.gpu_swap_batch,
                "engine_restart_drain_grace": self.config.server.engine_restart_drain_grace,
                "isolate_gguf_engine": self.config.server.isolate_gguf_engine,
                "engine_specs": self.config.server.engine_specs,
                "default_engine": self.config.server.default_engine,
            },
            "backend": {
                "type": self.config.backend.type,
                "image_backend": self.config.backend.image_backend,
                "audio_backend": self.config.backend.audio_backend,
                "tts_backend": self.config.backend.tts_backend
            },
            "models": {
                "default_load_mode": self.config.models.default_load_mode,
                "hf_cache_dir": self.config.models.hf_cache_dir,
                "gguf_cache_dir": self.config.models.gguf_cache_dir,
                "max_model_instances": self.config.models.max_model_instances,
                "max_model_instances_overrides": self.config.models.max_model_instances_overrides,
                "load_status_updates": self.config.models.load_status_updates,
                "wait_status_mode": self.config.models.wait_status_mode,
            },
            "offload": {
                "directory": self.config.offload.directory,
                "strategy": self.config.offload.strategy,
                "max_gpu_percent": self.config.offload.max_gpu_percent,
                "no_ram": self.config.offload.no_ram,
                "load_in_4bit": self.config.offload.load_in_4bit,
                "load_in_8bit": self.config.offload.load_in_8bit,
                "manual_ram_gb": self.config.offload.manual_ram_gb,
                "flash_attention": self.config.offload.flash_attention,
                "max_ram_gb": self.config.offload.max_ram_gb,
                "evict_idle_on_ram": self.config.offload.evict_idle_on_ram,
                "ram_leak_watch": self.config.offload.ram_leak_watch,
                "ram_watch_poll_seconds": self.config.offload.ram_watch_poll_seconds,
                "ram_watch_soft_fraction": self.config.offload.ram_watch_soft_fraction,
                "ram_watch_cuda": self.config.offload.ram_watch_cuda,
                "gpu_split": self.config.offload.gpu_split,
                "tensor_split": self.config.offload.tensor_split,
                "split_strategy": self.config.offload.split_strategy,
                "split_secondary_cap_gb": self.config.offload.split_secondary_cap_gb,
                "split_card_caps_gb": self.config.offload.split_card_caps_gb
            },
            "vulkan": {
                "n_gpu_layers": self.config.vulkan.n_gpu_layers,
                "n_ctx": self.config.vulkan.n_ctx,
                "device_id": self.config.vulkan.device_id,
                "single_gpu": self.config.vulkan.single_gpu
            },
            "image": {
                "llm_path": self.config.image.llm_path,
                "vae_path": self.config.image.vae_path,
                "sample_method": self.config.image.sample_method,
                "steps": self.config.image.steps,
                "width": self.config.image.width,
                "height": self.config.image.height,
                "cfg_scale": self.config.image.cfg_scale,
                "precision": self.config.image.precision,
                "cpu_offload": self.config.image.cpu_offload,
                "seed": self.config.image.seed,
                "vae_tiling": self.config.image.vae_tiling,
                "clip_on_cpu": self.config.image.clip_on_cpu
            },
            "archive": {
                "enabled": self.config.archive.enabled,
                "directory": self.config.archive.directory,
                "retention": self.config.archive.retention,
            },
            "thermal": {
                "cpu_enabled": self.config.thermal.cpu_enabled,
                "gpu_enabled": self.config.thermal.gpu_enabled,
                "cpu_high": self.config.thermal.cpu_high,
                "cpu_resume": self.config.thermal.cpu_resume,
                "gpu_high": self.config.thermal.gpu_high,
                "gpu_resume": self.config.thermal.gpu_resume,
                "gpu_overrides": self.config.thermal.gpu_overrides,
                "poll_seconds": self.config.thermal.poll_seconds,
                "soft_throttle_enabled": self.config.thermal.soft_throttle_enabled,
                "soft_throttle_temp": self.config.thermal.soft_throttle_temp,
                "soft_throttle_max_sleep": self.config.thermal.soft_throttle_max_sleep,
                "supervisor_enabled": self.config.thermal.supervisor_enabled,
                "stop_escalate_checks": self.config.thermal.stop_escalate_checks,
            },
            "jobs": {
                "resume_on_restart": self.config.jobs.resume_on_restart,
            },
            "enhance": {
                "allow_ffmpeg": self.config.enhance.allow_ffmpeg,
                "allow_rife_ncnn": self.config.enhance.allow_rife_ncnn,
            },
            "remotes": {
                "enabled": self.config.remotes.enabled,
                "api_key": self.config.remotes.api_key,
                "max_body_mb": self.config.remotes.max_body_mb,
                "endpoints": dict(self.config.remotes.endpoints or {}),
                "pods": dict(self.config.remotes.pods or {}),
            },
            "ds4": {
                "enabled": self.config.ds4.enabled,
                "service_url": self.config.ds4.service_url,
                "repo_url": self.config.ds4.repo_url,
                "install_dir": self.config.ds4.install_dir,
                "build_target": self.config.ds4.build_target,
                "model_path": self.config.ds4.model_path,
                "auto_download": self.config.ds4.auto_download,
                "model_variant": self.config.ds4.model_variant,
                "model_id": self.config.ds4.model_id,
                "host": self.config.ds4.host,
                "port": self.config.ds4.port,
                "ctx": self.config.ds4.ctx,
                "ssd_streaming": self.config.ds4.ssd_streaming,
                "extra_args": self.config.ds4.extra_args,
                "expert_cache_reserve_gb": self.config.ds4.expert_cache_reserve_gb,
                "extra_env": self.config.ds4.extra_env,
                "auto_build": self.config.ds4.auto_build,
                "kv_cache_cleanup_enabled": self.config.ds4.kv_cache_cleanup_enabled,
                "kv_cache_max_age_hours": self.config.ds4.kv_cache_max_age_hours,
                "kv_cache_cleanup_interval_minutes": self.config.ds4.kv_cache_cleanup_interval_minutes,
            },
            "colibri": {
                "enabled": self.config.colibri.enabled,
                "service_url": self.config.colibri.service_url,
                "repo_url": self.config.colibri.repo_url,
                "install_dir": self.config.colibri.install_dir,
                "build_target": self.config.colibri.build_target,
                "model_path": self.config.colibri.model_path,
                "model_id": self.config.colibri.model_id,
                "ctx": self.config.colibri.ctx,
                "kv_slots": self.config.colibri.kv_slots,
                "cap": self.config.colibri.cap,
                "cuda_expert_gb": self.config.colibri.cuda_expert_gb,
                "extra_args": self.config.colibri.extra_args,
                "extra_env": self.config.colibri.extra_env,
                "auto_build": self.config.colibri.auto_build,
            },
            "k3": {
                "enabled": self.config.k3.enabled,
                "service_url": self.config.k3.service_url,
                "repo_url": self.config.k3.repo_url,
                "install_dir": self.config.k3.install_dir,
                "model_path": self.config.k3.model_path,
                "trunk_dir": self.config.k3.trunk_dir,
                "tok_dir": self.config.k3.tok_dir,
                "model_id": self.config.k3.model_id,
                "preset": self.config.k3.preset,
                "trunk_gb": self.config.k3.trunk_gb,
                "cache_gb": self.config.k3.cache_gb,
                "ctx": self.config.k3.ctx,
                "extra_args": self.config.k3.extra_args,
                "extra_env": self.config.k3.extra_env,
                "auto_build": self.config.k3.auto_build,
            },
            "ktransformers": {
                "enabled": self.config.ktransformers.enabled,
                "service_url": self.config.ktransformers.service_url,
                "repo_url": self.config.ktransformers.repo_url,
                "install_dir": self.config.ktransformers.install_dir,
                "model_path": self.config.ktransformers.model_path,
                "kt_weight_path": self.config.ktransformers.kt_weight_path,
                "model_id": self.config.ktransformers.model_id,
                "host": self.config.ktransformers.host,
                "port": self.config.ktransformers.port,
                "ctx": self.config.ktransformers.ctx,
                "extra_args": self.config.ktransformers.extra_args,
                "extra_env": self.config.ktransformers.extra_env,
                "auto_build": self.config.ktransformers.auto_build,
            },
            "vllm": {
                "enabled": self.config.vllm.enabled,
                "service_url": self.config.vllm.service_url,
                "venv": self.config.vllm.venv,
                "model_path": self.config.vllm.model_path,
                "model_id": self.config.vllm.model_id,
                "gpu": self.config.vllm.gpu,
                "host": self.config.vllm.host,
                "port": self.config.vllm.port,
                "ctx": self.config.vllm.ctx,
                "gpu_memory_utilization": self.config.vllm.gpu_memory_utilization,
                "tensor_parallel_size": self.config.vllm.tensor_parallel_size,
                "max_num_seqs": self.config.vllm.max_num_seqs,
                "dtype": self.config.vllm.dtype,
                "quantization": self.config.vllm.quantization,
                "extra_args": self.config.vllm.extra_args,
                "extra_env": self.config.vllm.extra_env,
                "auto_build": self.config.vllm.auto_build,
            },
            "runpod": {
                "enabled": self.config.runpod.enabled,
                "api_key": self.config.runpod.api_key,
                "cloud_type": self.config.runpod.cloud_type,
                "api_base": self.config.runpod.api_base,
                "rest_base": self.config.runpod.rest_base,
                "serverless_base": self.config.runpod.serverless_base,
                "default_gpu_type": self.config.runpod.default_gpu_type,
                "data_center": self.config.runpod.data_center,
                "registry_auth_id": self.config.runpod.registry_auth_id,
                "network_volume_id": self.config.runpod.network_volume_id,
                "volume_mount_path": self.config.runpod.volume_mount_path,
                "deployment_id": self.config.runpod.deployment_id,
                "global_max_hourly_usd": self.config.runpod.global_max_hourly_usd,
                "global_cost_limit_usd": self.config.runpod.global_cost_limit_usd,
                "global_cost_period": self.config.runpod.global_cost_period,
            },
            "ocr": {
                "enabled": self.config.ocr.enabled,
                "default_engine": self.config.ocr.default_engine,
                "dpi": self.config.ocr.dpi,
                "max_concurrency": self.config.ocr.max_concurrency,
                "lang": self.config.ocr.lang,
                "paddle_enabled": self.config.ocr.paddle_enabled,
                "paddle_use_gpu": self.config.ocr.paddle_use_gpu,
                "paddle_instances": self.config.ocr.paddle_instances,
                "paddle_structure": self.config.ocr.paddle_structure,
                "paddle_lang": self.config.ocr.paddle_lang,
                "paddle_det_model_dir": self.config.ocr.paddle_det_model_dir,
                "paddle_rec_model_dir": self.config.ocr.paddle_rec_model_dir,
                "paddle_venv": self.config.ocr.paddle_venv,
                "paddle_auto_build": self.config.ocr.paddle_auto_build,
                "doctr_enabled": self.config.ocr.doctr_enabled,
                "doctr_use_gpu": self.config.ocr.doctr_use_gpu,
                "doctr_instances": self.config.ocr.doctr_instances,
                "doctr_det_arch": self.config.ocr.doctr_det_arch,
                "doctr_reco_arch": self.config.ocr.doctr_reco_arch,
                "surya_enabled": self.config.ocr.surya_enabled,
                "surya_accept_license": self.config.ocr.surya_accept_license,
                "surya_instances": self.config.ocr.surya_instances,
                "surya_langs": self.config.ocr.surya_langs,
                "surya_venv": self.config.ocr.surya_venv,
                "surya_auto_build": self.config.ocr.surya_auto_build,
                "surya_serve": self.config.ocr.surya_serve,
                "surya_model": self.config.ocr.surya_model,
                "surya_server_url": self.config.ocr.surya_server_url,
                "detect_mode": self.config.ocr.detect_mode,
                "detect_model_path": self.config.ocr.detect_model_path,
                "detect_conf": self.config.ocr.detect_conf,
                "extract_enabled": self.config.ocr.extract_enabled,
                "extract_model_id": self.config.ocr.extract_model_id,
                "extract_max_tokens": self.config.ocr.extract_max_tokens,
                "extract_schema": self.config.ocr.extract_schema,
                "extract_validate": self.config.ocr.extract_validate,
            },
            "compaction": {
                "enabled": self.config.compaction.enabled,
                "pct": self.config.compaction.pct,
                "strategy": self.config.compaction.strategy,
                "model": self.config.compaction.model,
                "tolerance_pct": self.config.compaction.tolerance_pct,
                "min_output": self.config.compaction.min_output,
                "estimate_safety": self.config.compaction.estimate_safety,
            },
            "broker": {
                "enabled": self.config.broker.enabled,
                "base_url": self.config.broker.base_url,
                "scope": self.config.broker.scope,
                "username": self.config.broker.username,
                "provider_id": self.config.broker.provider_id,
                "client_id": self.config.broker.client_id,
                "registration_token": self.config.broker.registration_token,
                "advertised_endpoint": self.config.broker.advertised_endpoint,
                "websocket_path": self.config.broker.websocket_path,
                "transport": self.config.broker.transport,
                "heartbeat_interval_seconds": self.config.broker.heartbeat_interval_seconds,
                "connect_timeout_seconds": self.config.broker.connect_timeout_seconds,
                "request_timeout_seconds": self.config.broker.request_timeout_seconds,
                "reconnect_initial_delay_seconds": self.config.broker.reconnect_initial_delay_seconds,
                "reconnect_max_delay_seconds": self.config.broker.reconnect_max_delay_seconds,
                "websocket_ping_interval": self.config.broker.websocket_ping_interval,
            },
            "system_prompt": self.config.system_prompt,
            "tools_closer_prompt": self.config.tools_closer_prompt,
            "grammar_guided": self.config.grammar_guided,
            "file_path": self.config.file_path,
            "tmp_dir": self.config.tmp_dir,
            "tmp_cleanup_enabled": self.config.tmp_cleanup_enabled,
            "tmp_cleanup_max_age_hours": self.config.tmp_cleanup_max_age_hours,
            "tmp_cleanup_interval_minutes": self.config.tmp_cleanup_interval_minutes,
            "hf_chat_templates": self.config.hf_chat_templates,
            "reasoning_options": self.config.reasoning_options,
            "parser": self.config.parser
        }

        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _enable_subsystems_from_env(self) -> None:
        """Switch on a subsystem a pod was rented to provide.

        Some subsystems are off by default because they are heavy or
        license-gated, which is right for a fresh install and wrong for a pod
        rented specifically to serve them: an OCR pod booted, took the request,
        and answered "OCR subsystem is disabled (enable it in Settings → OCR)"
        — a setting screen nobody is going to open on a machine that exists for
        the next four minutes.

        Env only, never written to disk: like the seed list, this describes one
        disposable instance, not a deployment.
        """
        import os as _os

        def _flag(name: str) -> bool:
            return str(_os.environ.get(name, "")).strip().lower() in (
                "1", "true", "yes", "on")

        if _flag("CODERAI_OCR_ENABLED"):
            self.config.ocr.enabled = True
            engine = (_os.environ.get("CODERAI_OCR_DEFAULT_ENGINE") or "").strip()
            if engine:
                self.config.ocr.default_engine = engine
            # Each engine has its OWN gate on top of the subsystem's, and two of
            # the three default to off. Enabling the subsystem alone got a pod as
            # far as "OCR engine 'surya' is not enabled" — a second refusal from
            # a second flag, one round later.
            for name in ("paddle", "doctr", "surya"):
                if _flag(f"CODERAI_OCR_{name.upper()}_ENABLED"):
                    setattr(self.config.ocr, f"{name}_enabled", True)
            if _flag("CODERAI_OCR_SURYA_ACCEPT_LICENSE"):
                # Surya is GPL and gated on an explicit acceptance. The pod
                # inherits the decision made here; it cannot make it itself.
                self.config.ocr.surya_accept_license = True

    def _seed_models_from_env(self) -> None:
        """Register models named by CODERAI_SEED_MODELS (JSON list of entries).

        Used by RunPod capability pods, which are rented for a specific model and
        have no models.json of their own. Never persisted: a pod is disposable,
        and writing it would surprise anyone who set the variable on a real
        installation.
        """
        raw = os.environ.get("CODERAI_SEED_MODELS", "").strip()
        if not raw:
            return
        try:
            entries = json.loads(raw)
        except Exception as exc:
            print(f"[seed] CODERAI_SEED_MODELS is not valid JSON: {exc}", flush=True)
            return
        if isinstance(entries, dict):
            entries = [entries]
        if not isinstance(entries, list):
            return
        for e in entries:
            if not isinstance(e, dict) or not e.get("path"):
                continue
            section = e.get("model_type") or "text_models"
            lst = self.models_data.setdefault(section, [])
            if not isinstance(lst, list):
                continue
            path = str(e["path"])
            if any(isinstance(m, dict) and m.get("path") == path for m in lst):
                continue            # a real entry already covers it — leave it alone
            lst.append(dict(e))
            print(f"[seed] registered {path} in {section}", flush=True)
        # The catalogue alone is not enough: request validation asks the MODEL
        # MANAGER for its allowed identifiers, and a seeded pod whose manager was
        # never told still answers "not available" with an empty list. The
        # manager is set up after this runs, so the work is deferred to it.
        self._pending_seed_entries = [e for e in entries if isinstance(e, dict)
                                      and e.get("path")]

    def save_models(self):
        """Save models.json to disk."""
        with open(self.models_path, 'w') as f:
            json.dump(self.models_data, f, indent=2)

    def persist_model_field(self, model_path: str, key: str, value, config_id: str = None) -> bool:
        """Set a SINGLE field on the matching model entry by RE-READING models.json
        from disk first, then writing back — never dumping this process's whole
        in-memory models_data.

        This avoids a multi-process clobber: each engine (front + every backend
        engine) loads models.json at its own boot, so a secondary engine's
        in-memory copy is stale w.r.t. a UI edit (e.g. n_ctx) made afterward. When
        that engine later auto-persists a runtime field (e.g. measured_vram_gb), a
        plain save_models() would write its stale full state and revert the edit.
        Read-modify-write of only the intended field keeps every other key intact.
        Also refreshes self.models_data so this process sees the merged result."""
        try:
            on_disk = {}
            if self.models_path.exists():
                with open(self.models_path, 'r') as f:
                    on_disk = json.load(f)
        except Exception:
            on_disk = dict(self.models_data)
        bare = model_path.split(":", 1)[1] if ":" in model_path else model_path
        changed = False
        for cat, lst in on_disk.items():
            if not isinstance(lst, list):
                continue
            for entry in lst:
                if not isinstance(entry, dict):
                    continue
                # When a config_id is given, target ONLY that exact entry — so a
                # same-path sibling config (multi-config) doesn't inherit the other's
                # measured footprint (which would make e.g. a small-ctx variant offload
                # using the large-ctx variant's VRAM figure). Fall back to path/basename
                # matching for legacy entries without a config_id.
                if config_id:
                    if entry.get("config_id") != config_id:
                        continue
                else:
                    epath = entry.get("path") or entry.get("id") or ""
                    if not (epath == bare or epath.split("/")[-1] == bare.split("/")[-1]):
                        continue
                if entry.get(key) != value:
                    entry[key] = value
                    changed = True
        if changed:
            tmp = str(self.models_path) + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(on_disk, f, indent=2)
            os.replace(tmp, self.models_path)
        # Keep our in-memory copy consistent with the merged on-disk file.
        self.models_data = on_disk
        return changed
    
    def save_auth(self):
        """Save auth.json to disk."""
        with open(self.auth_path, 'w') as f:
            json.dump(self.auth_data, f, indent=2)

    def save_pipelines(self):
        """Save pipelines.json to disk."""
        with open(self.pipelines_path, 'w') as f:
            json.dump(self.pipelines_data, f, indent=2)
    
    def reload(self):
        """Reload all configuration files."""
        return self.load()
