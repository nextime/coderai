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
    # GET /metrics (Prometheus) without a credential. Off = a session cookie,
    # an API token or the cluster token is required, as for the other telemetry.
    metrics_public: bool = False
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
    # Release an in-flight request the front has held for longer than this, so a
    # leaked one cannot wedge the install. The count is what the hourly
    # autoupgrade's idle guard reads, and with MAX_DEFERRALS=0 a non-zero count
    # stops upgrading ALTOGETHER -- a leak is therefore not a cosmetic metric
    # problem, it silently freezes the host on old code. Well above the slowest
    # legitimate request: a cold RunPod pod can make one OCR page take 14 minutes,
    # so this is an hour, not a minute. Training is unaffected (it is tracked as a
    # JOB; the request that starts it returns at once). 0 disables the reaper.
    inflight_max_age_s: float = 3600.0
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
class AdminForwardAuthConfig:
    """Trust an identity a reverse proxy has already authenticated (SSO).

    OFF by default and GUI-ONLY: it never changes how /v1/* authenticates.
    When enabled, a request that comes from a trusted proxy AND carries the
    right shared secret is logged in as the user named in ``user_header``,
    with no password form. Both checks are required — a trusted header is a
    forged-identity hole the moment it is honoured from anywhere else — and
    anything missing fails CLOSED.
    """
    enabled: bool = False
    user_header: str = "X-Forwarded-User"
    groups_header: str = "X-Forwarded-Groups"
    admin_group: str = "admins"
    shared_secret_header: str = "X-Coderai-Proxy-Secret"
    #: The expected value. Blank = the feature stays off however `enabled` is
    #: set: without a secret there is nothing to distinguish the proxy from a
    #: client that simply sends the header.
    shared_secret: str = ""
    trusted_proxies: list = field(default_factory=lambda: ["127.0.0.1", "::1"])
    auto_create_users: bool = True
    #: Leave the password form working (a fallback for loopback/direct access).
    #: False = the proxy is the only way in.
    keep_local_login: bool = False
    #: What to do when the proxy sends no groups header at all. False (default)
    #: = the user is a non-admin; True = every forward-authed user is an admin.
    #: Explicit on purpose: guessing here either locks the operator out or hands
    #: admin to everyone upstream lets through.
    admin_without_groups: bool = False
    #: Where a logout goes. The real session lives upstream, so returning to our
    #: own login would just auto-log-in again.
    post_logout_url: str = ""


@dataclass
class AdminConfig:
    """Admin GUI settings."""
    forward_auth: AdminForwardAuthConfig = field(default_factory=AdminForwardAuthConfig)


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
    #: Where Civitai single-file checkpoints are kept. Configured like the two
    #: above because they are the same KIND of thing -- multi-gigabyte weights
    #: that belong on whichever disk holds the weights, not on the partition
    #: the server runs from. Left unset they land beside the other coderai
    #: caches, which on a host whose caches are a small volume fills it: seen
    #: on 2026-10-28, a 13.5 GB checkpoint took the cache volume to 99% and the
    #: next model could not even write its config fetch ("No space left on
    #: device"). CODERAI_CIVITAI_CACHE still wins over this.
    civitai_cache_dir: Optional[str] = None
    max_model_instances: int = 1  # max concurrent instances per model (global default; overridable per-model via "max_instances")
    # Per-engine overrides for max_model_instances, keyed by engine name
    # (e.g. {"nvidia": 2, "radeon": 1}). Applied per-engine process; blank = default.
    max_model_instances_overrides: dict = field(default_factory=dict)
    #: Repositories whose OWN CODE may be executed when loading a model
    #: (``custom_pipeline`` / ``trust_remote_code``). An ALLOW-LIST, because
    #: the alternative is a server that runs arbitrary Python from any
    #: checkpoint an operator happens to name -- and the published pipelines
    #: that need it (MOSS, Kromoss) ship the pipeline class in the repo.
    #:
    #: Entries are exact repo ids or an ``org/*`` prefix::
    #:
    #:     "trust_remote_code": ["OpenMOSS-Team/MOSS-Image-SFT",
    #:                           "lodestones/*"]
    #:
    #: Empty (the default) means no model may run remote code, and a model
    #: configured to want it is refused by name rather than loaded quietly
    #: without the pipeline it asked for.
    trust_remote_code: list = field(default_factory=list)
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
    # Which CLIP layer the text embedding comes from, counted from the end.
    # None = the pipeline's own choice. The booru-tag SDXL fine-tunes (Pony,
    # Illustrious, NoobAI) are trained at 2; a model can override this.
    clip_skip: Optional[int] = None
    # Parse the A1111 prompt grammar -- (tag:1.3), [tag], BREAK. Off keeps
    # brackets literal; the prompt is windowed past 77 tokens either way.
    prompt_weighting: bool = True
    # Request-size ceiling. A size above it is a 400 naming the limit, which
    # is a better answer than the CUDA OOM an absurd size used to produce.
    max_side: int = 4096
    max_pixels: int = 4096 * 1024      # 4.2 MP -- 2048x2048, 2560x1440, …
    # A flat directory of textual-inversion embeddings, loaded by filename so
    # a prompt copied off a model card finds its trigger word. Empty = none.
    embeddings_dir: str = ""


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
class ChunkingConfig:
    """Retrieval chunking (`POST /v1/chunk`).

    NOT embedding windowing. `ocr`/`embeddings` settings make one input fit a
    model window and pool it into ONE vector; this splits a document into MANY
    separately-indexable chunks, each with its section, parent and references.

    Token counts are approximated as characters / `chars_per_token`, so the
    endpoint loads no tokenizer and needs no model, no GPU and no pod -- which
    is what lets every image serve it, light-core ones included.
    """
    # The defaults are the ones a production index was built with: changing them
    # changes every chunk boundary, which makes a re-chunked index incomparable
    # with the one it replaces and retrieval quality unmeasurable across the
    # change. Treat them as a format, not a preference.
    chunk_tokens: int = 800              # target size, approximated tokens
    overlap: int = 120                   # carried tail, approximated tokens
    chars_per_token: int = 4             # the approximation divisor
    structural: bool = True              # False = flat length chunking
    structure: str = "it-legal"          # default template name
    require_text: bool = True            # refuse container bytes (see below)
    # A chunker that accepts a PDF or an OOXML container embeds garbage that the
    # index cannot afterwards distinguish from content. Measured on a production
    # corpus: 0.59% of chunks held control characters, 0.04% raw OOXML --
    # documents whose container was chunked instead of its extracted text.
    # `require_text: false` opts out deliberately.
    max_chars: int = 8_000_000           # above this -> 413 too_large
    # Operator-supplied structure templates, same JSON shape as the built-in
    # `it-legal`, selected by name. An unknown name is an ERROR, never a silent
    # fallback: a caller who asked for their own template and silently got
    # Italian legal sectioning would index the wrong boundaries and never know.
    templates: dict = field(default_factory=dict)


@dataclass
class CompactionConfig:
    """Global defaults for auto-compaction of an over-long chat history.

    Per-model settings in a models.json entry (``auto_compact``,
    ``auto_compact_pct``, ``auto_compact_strategy``, ``auto_compact_model``,
    ``auto_compact_tolerance_pct``, ``auto_compact_head_messages``,
    ``auto_compact_head_tokens``, ``auto_compact_tail_tokens``,
    ``auto_compact_target_pct``, ``auto_compact_retry_on_overflow``) OVERRIDE the
    values here; when a model leaves one unset, the global default below applies.

    The strategies are ``drop_oldest`` (keep the system messages and the recent
    tail), ``keep_head_tail`` (also keep the opening turns, drop the centre) and
    ``summarize`` (keep both ends and replace the centre with an LLM summary of
    exactly the messages being dropped -- "keep the start and the end, compact
    the middle"; ``summarize_middle`` is accepted as a name for it). An unknown
    name is read as ``keep_head_tail`` rather than silently dropping the head.

    ``model`` selects which model
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
    # What the kept ends are, for keep_head_tail and summarize. The head is the
    # opening instructions a long conversation keeps drifting away from, so it is
    # sized in MESSAGES (one turn is the anchor; a handful is a briefing), while
    # the tail is sized in TOKENS because what matters there is how much recent
    # conversation survives verbatim, not how many turns it took. 0 means "as
    # much as the target allows": the whole remaining budget for the tail, half
    # of it for the head.
    head_messages: int = 1               # leading turns kept verbatim (from the first user turn)
    head_tokens: int = 0                 # cap on the head, in tokens (0 = half the target)
    tail_tokens: int = 0                 # recent turns kept verbatim, in tokens (0 = the rest)
    target_pct: int = 65                 # compact DOWN to this % of n_ctx (leaves room to grow)
    # Retry a request an engine refused for length, compacting against the window
    # and prompt size the refusal reports. On by default: without it a caller is
    # told to shorten a conversation coderai was configured to shorten for them,
    # and the correction is learned too late to help that request.
    retry_on_overflow: bool = True


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
class ClusterConfig:
    """Other coderai installs on your network, used as engines of this one.

    A **node** is a whole coderai running somewhere else — its own front, its
    own cards, its own admin — that this front treats exactly like one of the
    engine processes it spawns: it polls the node's aggregated engine state,
    assigns models to it, pins a model to it by name from the Models page, and
    routes requests there by capability and load. The node needs nothing but an
    API token; the head needs the node's URL and that token.

    The same section also declares the **llama.cpp RPC servers** THIS machine
    contributes (`rpc_servers`), so a GGUF model on another node can be spread
    over this machine's cards too (per-model ``rpc_servers`` on the model). The
    binary is built by ``packaging/build-rpc-server.sh`` from the very same
    llama.cpp the bundled llama-cpp-python was built from — the RPC protocol is
    versioned and the two must match.

    Nodes: ``[{"name": "box2", "url": "https://box2:8776", "api_key": "…",
              "verify": "system" | "off" | "pem", "ca_pem": "-----BEGIN…",
              "capabilities": [...], "enabled": true}]``
    ``capabilities`` blank = whatever the node reports (union of its engines).
    """
    enabled: bool = False
    nodes: list = field(default_factory=list)
    # What THIS install calls itself when it is a node (matches the head's
    # ``cluster.nodes[].name`` and a model's ``node_paths`` key). Blank = hostname.
    node_name: str = ""
    # How often (s) the head polls each node for state — the same short timeout
    # the local engines get, but a node is a network hop away.
    poll_timeout_s: float = 4.0
    # Answer /cluster/* to callers holding a valid API token of this install.
    # Off = this install can be a head but never a node.
    serve: bool = True
    # llama.cpp RPC servers this machine runs (one process each):
    # [{"name": "3090", "host": "0.0.0.0", "port": 50052, "device": "CUDA0",
    #   "mem_gb": 0, "threads": 0, "enabled": true}]
    # ``device`` names a ggml device (CUDA0, Vulkan1 …; blank = the server's
    # default); ``mem_gb`` caps what it advertises (0 = all).
    rpc_servers: list = field(default_factory=list)
    rpc_bin: str = ""                        # path to rpc-server; blank = auto-find
    # The address other machines reach THIS one at (for the RPC endpoints it
    # advertises and the ray head it starts). Blank = best guess from the
    # default route.
    advertise_host: str = ""
    # --- zero-config membership (codai/cluster/discovery.py) ---
    # One secret shared by every box of the cluster. A node accepts it on its
    # /cluster/* endpoints (besides its own API tokens); a head uses it as the
    # API key of every node it discovers, and of a listed node whose api_key
    # is blank. The mDNS announcement carries only an HMAC fingerprint of it.
    token: str = ""
    # Announce this install as _coderai._tcp on the LAN and listen for the
    # others (needs the zeroconf package; link-local only).
    discovery: bool = False
    # Turn every discovered member (same token, serve on) into a cluster node.
    auto_join: bool = True
    # --- routing ---
    # Prefer, for a follow-up turn of a conversation, the engine or node that
    # served its earlier turns (its KV/prefix cache is warm there) when the
    # model is resident in more than one place.
    prefix_affinity: bool = True
    prefix_affinity_ttl_s: float = 1800.0


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
    venv: str = ""                         # venv holding SGLang + kt-kernel (they pin their own torch); blank = main
    model_path: str = ""                   # HF model directory (--model)
    kt_weight_path: str = ""               # KT quantized weights directory (--kt-weight-path)
    model_id: str = "ktransformers"        # model id/alias that routes to kt (and --served-model-name)
    host: str = "127.0.0.1"                # SGLang bind host
    port: int = 0                          # 0 = auto-pick a free port
    ctx: int = 32768                       # context length (--context-length)
    extra_args: str = ""                   # extra flags for sglang.launch_server (e.g. --tp-size, KT knobs)
    extra_env: str = ""                    # free-form KEY=VALUE env for the subprocess
    auto_build: bool = False               # pip-install SGLang+kt-kernel if missing (heavy; off by default)
    # Multi-node SGLang: this process is rank 0 of ``nnodes``; every other rank is
    # started by its ``start_cmd`` (templated with {rank}, {nnodes},
    # {dist_init_addr}, {model}) — ssh, docker, a systemd unit, whatever starts
    # the same SGLang on that machine. Per-model ``kt`` blocks override these.
    nnodes: int = 1
    dist_init_addr: str = ""               # host:port of rank 0 as the others reach it; blank = advertise_host:20000
    tp_size: int = 0                       # --tp-size across ALL nodes' GPUs (0 = SGLang default)
    nodes: list = field(default_factory=list)   # [{"name","start_cmd","stop_cmd"}] for ranks 1..n-1


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
    # Ask vLLM to report logprob tokens as `token_id:{id}` instead of rendered
    # strings. Off by default because it changes what that field looks like, and
    # only one consumer reads it.
    #
    # It exists for a measured dead end: on vLLM 0.27.1 serving
    # `datalab-to/surya-ocr-2`, whose vocabulary is BYTE-level, an accented letter
    # comes back as U+FFFD in `content` AND as an empty `bytes` list in the
    # logprobs -- 257 such tokens on one page, with no byte >= 0x80 anywhere. So
    # there is nothing to recover from either output. The token IDS are intact,
    # and surya's own tokenizer decodes them correctly, so with this on the OCR
    # worker can rebuild the text itself (`ocr.surya_repair_accents`) instead of
    # waiting for an upstream fix or pinning vLLM back to 0.20.1.
    return_token_ids: bool = False
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
    # How the share learner decides it may give memory back. It measures an
    # engine's real footprint from the cache a successful boot reported, then
    # proposes the smallest share that still leaves `kv_target_gb` of cache --
    # but a cache that cannot hold the window the engine serves is useless, and
    # a vLLM whose cache cannot hold `ctx` refuses to start. So the target is a
    # floor, raised to `kv_min_concurrency` full-length requests of `ctx`
    # (bytes-per-token measured from the boot report, since it depends on the
    # model's attention shape). 2 GiB suits a page-at-a-time VLM; an LLM serving
    # long conversations needs the window-derived figure. Both are overridable
    # per model in its `vllm` block.
    kv_target_gb: float = 2.0            # smallest cache worth keeping, before the floor
    kv_min_concurrency: float = 2.0      # full-length requests the cache must hold
    tensor_parallel_size: int = 1        # --tensor-parallel-size
    max_num_seqs: int = 0                # --max-num-seqs (0 = vLLM default)
    dtype: str = ""                      # --dtype (blank = auto; e.g. bfloat16/float16)
    # --quantization. Blank means AUTO-DETECT, which is almost always what you
    # want: vLLM reads the checkpoint's own quantization_config and picks the
    # fastest kernel it has for it (a compressed-tensors W4A16 checkpoint gets
    # Marlin). This field DECLARES how the weights on disk are packed -- it does
    # not convert anything -- so naming the wrong format moves the engine onto a
    # slower kernel or stops it booting. A checkpoint whose name says AWQ can
    # still be compressed-tensors; believe the config, not the repo name.
    quantization: str = ""
    # How many tokens may be prefilled in ONE engine step (--max-num-batched-tokens).
    # vLLM's own default is 2048, so a 60k-token prompt is prefilled as ~30
    # sequential chunks of small, tensor-core-starved GEMMs. Raising it fills the
    # cards better and cuts time-to-first-token on long prompts; the cost is a
    # higher activation peak at profiling, which is taken out of the KV cache --
    # so raise it watching `Available KV cache memory` and the concurrency the
    # boot report prints. 0 = leave vLLM's default.
    max_num_batched_tokens: int = 0
    # Reuse the KV of a prompt prefix already computed (--enable-prefix-caching).
    # "auto" leaves vLLM's per-model default, which is ON for ordinary decoder
    # models but OFF for hybrid-attention ones (supported there, but opt-in while
    # the feature matures upstream). "on" forces it, which is the one setting that
    # can take a repeated-document workload -- the same file asked about again and
    # again -- from a full prefill per question to almost none. "off" forces it off.
    prefix_caching: str = "auto"         # auto | on | off
    # Speculative decoding (--speculative-config). Decode on one GPU is bound by
    # weight bandwidth, not arithmetic, so drafting several tokens and checking
    # them in one pass is the only way to beat that ceiling. "ngram" needs no
    # draft model and no VRAM: it drafts by finding the current suffix earlier in
    # the prompt, which is why it pays off exactly on answers that quote their
    # source (extraction, citation, summarising a document) and does nothing for
    # free-form prose. Anything else is passed through as raw JSON, so a draft
    # model can be configured without a code change.
    speculative: str = ""                # "" | "ngram" | raw JSON for --speculative-config
    speculative_tokens: int = 0          # draft length for "ngram" (0 = built-in default)
    # Tool calling. vLLM parses a model's tool-call syntax with a named parser
    # and REFUSES a request carrying `tools` unless it was launched with one:
    #     "auto" tool choice requires --enable-auto-tool-choice and
    #      --tool-call-parser to be set
    # -- a 400 at request time, from a server that started cleanly, for a
    # feature the catalogue says the model has. The syntax belongs to the model
    # family, so "auto" picks it from the model name and an explicit value
    # (here, or per model in its `vllm` block) wins. "off" disables it.
    tool_call_parser: str = "auto"
    # Which vLLM parser separates a thinking model's reasoning from its answer.
    # "auto" reads the model's chat template; "off" leaves the two joined.
    reasoning_parser: str = "auto"
    extra_args: str = ""                 # extra flags for the api_server
    extra_env: str = ""                  # free-form KEY=VALUE env for the subprocess
    auto_build: bool = False             # create the isolated venv + pip install vllm if missing
    # Multi-node vLLM over Ray: tensor parallel inside a node, pipeline parallel
    # across nodes. With ``nodes`` set (or ``pipeline_parallel_size`` > 1, or the
    # executor forced to ray) coderai starts a ray head here, runs each node's
    # ``start_cmd`` (templated with {ray_address} — the coderai-vllm image joins
    # with CODERAI_RAY_ADDRESS), waits until ray sees tp*pp GPUs, then launches
    # vLLM with --distributed-executor-backend ray. Per-model ``vllm`` blocks
    # override all of these.
    pipeline_parallel_size: int = 1      # --pipeline-parallel-size
    distributed_executor_backend: str = ""   # "" auto | "mp" | "ray"
    ray_address: str = ""                # join an EXISTING ray cluster instead of starting a head
    ray_port: int = 6379                 # the head's port when coderai starts it
    nodes: list = field(default_factory=list)   # [{"name","start_cmd","stop_cmd","gpus"}]
    nodes_ready_timeout_s: int = 600     # how long to wait for the nodes' GPUs to join


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
    # Account-wide region allow-list, tried in order on a capacity miss. The way
    # to say "anywhere in the EU": `data_center` pins ONE region, and leaving it
    # blank lets RunPod pick any region on earth — not what you want for data
    # that may not leave the EU. A per-model `data_centers` overrides this, and a
    # network volume overrides both (it exists in one region only).
    data_centers: list = field(default_factory=list)
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
    # One volume PER REGION, which is how a multi-region install keeps both.
    # RunPod ties a network volume to a single data centre, so the id above
    # pins every pod to that volume's region and the `data_centers` fallback
    # list is discarded — an install had to choose between renting wherever a
    # card is free and not re-downloading the weights onto every cold pod.
    # A map {"EU-SE-1": "vol-a", "EU-RO-1": "vol-b"} lets the scaler pick the
    # region first and attach that region's own volume, so the fallback list
    # keeps working. Regions with no volume here are dropped from the list
    # rather than rented without storage, because a pod that silently has none
    # is the slow boot this exists to prevent. Supersedes network_volume_id
    # when set; a one-entry map behaves exactly like the single id.
    network_volumes: dict = field(default_factory=dict)
    volume_mount_path: str = "/workspace"    # where pods mount it
    # A RunPod GLOBAL volume (beta, Sept 2026): region-INDEPENDENT storage, so
    # unlike network_volume_id above it does NOT pin pods to one data centre —
    # which is the whole reason to use it. Store the weights once and rent a GPU
    # in whichever region has one. Backed by object storage: fine for weights,
    # adapters and tokenizers, wrong for checkpoints (no file locking, no atomic
    # rename, eventual consistency, last-write-wins). RunPod mounts it at
    # /workspace on its own, or /workspace-global when a network volume is
    # attached too — so do not set both unless you mean that.
    # Must be created in the RunPod console: the public API still refuses the
    # type (POST /v2/network-volumes allows only STANDARD|HIGH_PERFORMANCE and
    # demands a dataCenter).
    global_volume_id: str = ""
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
class LongcatConfig:
    """LongCat-Video (Meituan, 13.6B dense, MIT) — driven as a managed subprocess.

    LongCat does text-to-video, image-to-video, video-continuation, long-video,
    interactive and audio-driven avatar generation, in three coarse-to-fine stages
    (base 480p → distilled LoRA → 720p refinement LoRA). There is no diffusers pipeline
    for it (diffusers merged LongCat-*Image*, not the video model), so coderai drives the
    upstream repo's own ``LongCatVideoPipeline`` out of an ISOLATED venv.

    That venv is the strictest in the tree: Python 3.10, torch 2.6+cu124,
    transformers 4.41 and numpy 1.26 — four simultaneous conflicts with the main venv
    (3.13 / 2.11+cu130 / 5.x / 2.4). See requirements-longcat.txt.

    coderai CANNOT build this venv by itself: every venv bootstrap in ``codai/`` uses
    ``sys.executable -m venv``, which is 3.13. The image bundles a standalone 3.10
    (shared with the lip-sync tools, at /opt/coderai/py310) and the built venv; on a
    source install, point ``python`` at a 3.10 interpreter or build the venv by hand.

    Like the other engines this serves MODELS FROM THE MODEL LIST: tag an entry
    ``backend: "longcat"``. Per-model settings (variant, quantisation, segments,
    context-parallel size) live on the entry; what is here is how to reach the runtime.
    """
    enabled: bool = False
    # Point this at a LongCat service already running elsewhere (another host, a
    # container, a rented pod) and coderai proxies to it instead of building,
    # downloading or launching anything locally. Blank = manage it here.
    service_url: str = ""
    venv: str = ""              # isolated venv dir; blank = auto (baked/cache/home)
    python: str = ""            # a Python 3.10 interpreter for building the venv;
                                # blank = the bundled /opt/coderai/py310, else PATH
    auto_build: bool = False    # build the venv on first use (needs `python` above).
                                # OFF by default: it pulls torch 2.6 + its CUDA runtime,
                                # which is minutes and gigabytes on a first request.
    host: str = "127.0.0.1"
    port: int = 0               # 0 = pick a free port per service
    gpu: str = ""               # CUDA device(s) (CUDA_VISIBLE_DEVICES); blank = all
    # FlashAttention-2 is upstream's default but is a source build needing ninja and a
    # CUDA toolchain; xformers ships a wheel and is what requirements-longcat.txt
    # installs. "flash" assumes you installed flash-attn into the venv yourself.
    attention: str = "xformers"         # xformers | flash | sdpa
    ready_timeout: float = 1800.0       # a 13.6B DiT load is not quick
    # When another model needs the card, how long to let an in-flight generation finish
    # before stopping the service anyway. A segment is ~80 new frames and minutes of
    # work, and the manager's own busy signal cannot see it (the worker holds no model
    # pool), so this is what makes the handover a clean swap instead of a lost request.
    evict_drain_timeout_s: float = 300.0
    # LoRA / QLoRA training. coderai does not implement the loop — SimpleTuner does, and
    # it supports LongCat-Video natively including a quantised base — so it is driven out
    # of a venv of its own (SimpleTuner pins its own torch and would fight the inference
    # venv's). Blank = beside the inference venv as <venv>-train.
    train_venv: str = ""
    # SimpleTuner needs Python >=3.12 — the OPPOSITE of the 3.10 the inference venv
    # pins — so the training venv is built with a different interpreter. Blank uses
    # coderai's own (3.13), which satisfies it; set this only to override.
    train_python: str = ""
    train_auto_build: bool = False
    # int8-quanto | int4-quanto | fp8-torchao | "" (bf16). This is the QLoRA switch: the
    # 13.6B transformer does not fit a consumer card at bf16 with optimiser state.
    train_base_precision: str = "int8-quanto"
    train_lora_rank: int = 8            # SimpleTuner's guidance for this model is 4-8
    train_gradient_checkpointing: bool = True
    extra_args: str = ""                # appended to the service command line
    extra_env: str = ""                 # KEY=VALUE pairs for the service process


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
    - ``olmocr``  — olmOCR-2 (AllenAI, Apache-2.0): a document VLM, not a detector —
      best transcription fidelity on hard pages (old scans, maths, tables, multi-column),
      but NO bounding boxes. Served through coderai's own model manager, its vLLM
      backend, or any OpenAI-compatible endpoint (``olmocr_serve``).
    - ``chandra`` — Chandra (Datalab; LICENSE-GATED — code Apache-2.0 but the WEIGHTS are
      a modified OpenRAIL-M): also a document VLM, but one that returns layout blocks WITH
      boxes and labels in a single pass, and from a different model family than surya-2 —
      the second opinion on a page surya reads badly (``chandra_serve``).

    Heavy OCR dependencies are installed out of band (optional) so the base image
    stays lean. Concurrency on a single GPU comes from loading multiple instances of
    an engine (``*_instances``) and fanning requests across them (``max_concurrency``);
    a continuous-batching path via vLLM is a deferred option (see docs/vllm.md).

    Per document coderai can produce: (1) plain-text transcription, (2) structured
    JSON of fields via an existing coderai text model (``extract_*``), and (3)
    stamp/signature flags (``detect_*``).
    """
    enabled: bool = False
    default_engine: str = "paddle"        # paddle|doctr|surya|olmocr|chandra
    dpi: int = 200                        # PDF rasterisation DPI
    # Per-engine PDF rasterisation DPI (0 = use `dpi` above). Surya-2 gets its own
    # because its input is capped at 3072x2048 = 6.29 Mpixels: `scale_to_fit` in
    # surya/inference/util.py is called with DEFAULTS ONLY, so no surya setting can
    # raise it, and anything larger is LANCZOS-downscaled to fit. An A4 page needs
    # ~255 DPI to reach that ceiling, so the shared 200 was handing the model 3.89 of
    # the 6.29 Mpixels it will accept -- 62% -- and an Italian accent is ~3 pixels
    # tall at 200 DPI. That is why accented letters came back as the unreadable-glyph
    # marker (see base.UNREADABLE). 250 saturates A4 at 1.5x the pixels of 200;
    # higher still works (it supersamples, which tends to help glyphs) but costs
    # rasterisation time and memory for no extra pixels at the model.
    surya_dpi: int = 250
    # Global cap on in-flight OCR pages: a SAFETY CEILING, not a throughput knob.
    #
    # It used to ship at 4, which made it the binding limit for every install:
    # the real limiter is meant to be each engine's own `*_instances` (the pool
    # hands out one instance per page and an engine is not reentrant), and a
    # RunPod fleet can hold `max_pods` x `scale_up_inflight_per_pod` pages in
    # flight -- 64 x 16 = 1024 on the Digesta install. A ceiling of 4 capped that
    # at a quarter of ONE pod, measured: one A40 sustains 16 concurrent pages for
    # surya (7.1 req/s) and for chandra (1.9 pages/s), with both peaking at 16 and
    # failing outright by 32.
    #
    # 64 is four pods' worth at that knee -- far above what a single-box install
    # can feed, so it stops being the limiter, while staying finite. Not higher,
    # because every admitted page holds a RASTERISED image: an A4 at 250 DPI is
    # ~6 Mpixels, ~18 MB in memory, so 64 in flight is ~1.2 GB and 1024 would be
    # ~18 GB of page bitmaps before any engine does work.
    #
    # Raise it when a fleet genuinely needs more, together with the per-engine
    # instance counts -- which on this design are what the box can host (a surya
    # worker is a SUBPROCESS at ~750 MB, so 64 of them is ~45 GB).
    #
    # CORRECTION (0.3.65): the paragraph above reasons as if resident == admitted.
    # It was not. `ocr_document` rasterised EVERY page of a document up front and
    # held the list for the whole request, so the pages waiting on this semaphore
    # were resident too and the real figure was (in-flight documents x their page
    # count), which this setting does not bound at all. A production install was
    # OOM-killed at 90.7 GB with this at 1024 and again at 92.7 GB with it at 128 --
    # eight times smaller, half a gigabyte worse. Memory is bounded by
    # `max_resident_mb` below; this stays what its first line always said it was,
    # a ceiling on concurrent page WORK.
    max_concurrency: int = 64            # global ceiling on in-flight OCR pages
    # How many pages may be ACCEPTED AND WAITING for a slot before the server
    # starts refusing new requests with 429. 0 = follow `max_concurrency`.
    #
    # The queue is deliberate, not an overflow: when a pod finishes a page there
    # must already be another page ready to hand it, or the fleet idles between
    # every dispatch and the pods are paid for the gap. What the queue must not
    # do is grow without limit -- a client that offers ten times the capacity
    # then waits longer and longer for each reply, which it reads as "the pages
    # are getting slower" rather than "stop sending". Past this depth the
    # honest answer is 429 with a Retry-After, and a client that obeys it gets
    # its work done sooner than one that keeps pushing.
    max_queued_pages: int = 0
    # Stop a generation the moment it is provably repeating itself, instead of
    # paying its whole token budget and then rejecting the result.
    #
    # A VLM that loses its place on a page repeats the last thing it saw until
    # it runs out of tokens. Every engine here already runs the same detector
    # over the FINISHED reply and treats a looping one as retryable -- so this
    # changes nothing about which pages are rejected, only how much of each one
    # is paid for. On a 100-page sample of a real judicial corpus (surya, A40,
    # cap 4096 / 1 continuation) 4 pages looped and accounted for 18.9% of the
    # run's entire generation time; one spent 450 s emitting 17,586 characters
    # of the same clause.
    #
    # On by default BECAUSE it is the same judgement, reached sooner: a page it
    # abandons is a page the post-hoc check would have rejected anyway.
    early_loop_abort: bool = True
    # Output before which nothing is judged. The largest genuine page in the
    # measured corpus is 5,054 characters and the median ~1,900, so 2,000 is
    # comfortably past where a real page has shown its character.
    early_loop_min_chars: int = 2000
    # How often to re-run the detector, in characters of new output.
    early_loop_step: int = 400
    # Consecutive checks that must all say "repeating" before abandoning the
    # page. A real loop never recovers; a repetitive passage -- a form, a table
    # of dotted leaders, a column of repeated dates -- does. Swept over 393 page
    # reads: at min_chars 1200 with persistence 1 there was one false positive
    # (a 1,444-character page with a repetitive middle and genuine text after
    # it); at 2000/2 there were none, and 12 of 12 real loops were still caught.
    early_loop_persistence: int = 2
    # Ceiling on the bytes that in-flight documents may hold in RASTERISED pages,
    # charged before a page exists (codai/ocr/budget.py prices a document from its
    # page tree). 0 = derive it from the cgroup limit or host RAM, which is the
    # shipped default because the installs that most need a ceiling are the ones
    # that will never set one. A document bigger than the whole budget still runs,
    # alone -- a gate that can refuse forever is just a deadlock.
    max_resident_mb: int = 0
    # Refuse a document longer than this many pages instead of reading it.
    # 0 = no ceiling, which is the shipped default: page count is not a fault,
    # and an install that batch-reads book scans is as legitimate as one that
    # reads two-page letters.
    #
    # It exists because length is the one property that is knowable BEFORE any
    # work is done (page_estimate reads the page tree) and the one that scales
    # every cost at once -- time, memory, tokens and, where pages are read on
    # rented cards, money. An operator who knows their corpus has a tail of
    # 1800-page outliers that will never finish inside a client's timeout is
    # better served refusing them in milliseconds, with a reason, than having
    # them time out expensively an hour later.
    #
    # Refusing LOUDLY is the whole point: a skipped document answers 413 with
    # its page count and this limit, so a caller can route it somewhere else.
    # It must never look like an empty or failed read -- see the "a degraded
    # page still answers 200" trap this subsystem already carries.
    max_pages: int = 0
    # Ceiling on ONE upload, in MB. A scan is as big as its DPI and its page
    # count make it -- a 300 DPI colour filing passes 100 MB without being
    # unusual -- so this is a setting rather than the constant it used to be.
    # 0 = no ceiling here (the reverse proxy's body limit still applies, and
    # nginx's own default of 1 MB is the usual reason a large page is refused
    # before coderai ever sees it: see docs/ocr.md).
    max_upload_mb: int = 100
    # What to do when the model's pods are the ones that run the OCR pipeline
    # (their image carries the rasteriser) but no pod can take a page.
    #
    #   auto    -- read it here at once. What 0.4.1-0.4.5 did, kept because an
    #              install with a GPU of its own may genuinely prefer it.
    #   prefer  -- wait up to `pod_pipeline_wait_s` for a pod, then read it
    #              here. The default, because a pool with nothing free this
    #              second is usually a pool whose pods are still booting.
    #   require -- never rasterise on this box: answer 503 and say why.
    #
    # The default is not a performance setting, it is a memory one. A GPU-less
    # orchestrator that quietly takes the pipeline back holds every page it
    # renders: measured at ~0.8 GB/min and 36 GB on a 125 GB box, with sixteen
    # rented cards idle, because a caller named the model instead of the engine
    # and the gate said nothing. Waiting a minute is cheaper than that in every
    # install where the pods exist at all.
    pod_pipeline: str = "prefer"
    # How long `prefer` (and `require`) wait for a pod before giving up on one.
    #
    # This was 60 s through 0.4.15, deliberately too short to cover a cold pool,
    # because a waiting page held a POD-POOL THREAD for the whole wait: a long
    # ceiling meant thousands of parked threads, so the cheaper mistake was to
    # stop waiting and read the page here.
    #
    # It no longer costs a thread. The lease waits on the event loop and only
    # each attempt touches a thread (`POD_LEASE_ATTEMPT_S`), so a page can wait
    # out an actual boot for free -- and a boot IS minutes: image pull, vLLM
    # load, weights. At 60 s a cold start therefore handed the whole first wave
    # of pages to this box, one GPU-less orchestrator rasterising for a fleet it
    # was already paying for, which is the failure mode `prefer` exists to
    # prevent. Ten minutes covers a boot with margin; an install whose pods are
    # always warm never reaches it, because a free pod answers at once.
    pod_pipeline_wait_s: float = 600.0
    # --- how a page is handed to the model ---
    # Measured on A4 at 200 DPI (3.4-3.9 Mpix): rendering a page costs 21-46 ms
    # and ENCODING it 86-353 ms, so three quarters of what is called
    # rasterisation is zlib, and on the surya path it was paid twice per page.
    #
    #   auto   (default) a page that already IS an image -- a scan held as one
    #          full-page JPEG -- is sent to the model exactly as the PDF holds
    #          it: no render, no encode, no resample (0.1 ms and 147 KB against
    #          46 + 352 ms and 1846 KB). Everything else is encoded as PNG at
    #          zlib level 1, which is lossless and 2.6x faster than PIL's
    #          default level 6.
    #   png    always re-encode. The one setting under which two runs of the
    #          same document produce byte-identical requests whatever the PDF
    #          held, which is what a controlled comparison needs.
    #   jpeg   lossy, 8 ms and a third of the bytes. Whether it costs accuracy
    #          is a question about a corpus rather than about a codec, so it is
    #          opt-in and not a default.
    # Forward a page to a pod on the EVENT LOOP rather than on a thread.
    #
    # The sync path costs one thread per page in flight, and a thread is the
    # wrong unit for a ~60 s wait on a socket: 15 documents a second at ~3.5
    # pages each needs ~3,200 outstanding, against a thread-pool ceiling of
    # 2048. On by default; off puts the forward back on the pod thread pool,
    # which is kept working so an operator who hits something odd on a new
    # transport can change one setting instead of waiting for a release.
    pod_http_async: bool = True
    page_image_format: str = "auto"
    page_image_jpeg_quality: int = 95
    # An ADDITIONAL floor on passing a scan through, in real DPI. 0 means the
    # floor is the engine's own target DPI, which is the only setting under
    # which a passed-through scan cannot change what the model sees.
    #
    # It shipped as an absolute 150 and that was wrong: a 150-DPI scan then
    # reached a 250-DPI engine at native size -- 40 % less linear resolution
    # than the render it replaced -- and surya-ocr-2 lost Italian accents on
    # those pages. Reported from production the same afternoon. The point of
    # the passthrough was "the same pixels without the work"; fewer pixels is
    # not a saving, it is a different page. Raise this only to demand MORE than
    # the target.
    page_image_min_dpi: float = 0.0
    page_image_max_dpi_factor: float = 3.0
    # How many pages of one document are rasterised at a time. The document is read
    # in windows, so residency is O(window) instead of O(document): a 400-page
    # filing costs the same as an 8-page one. 0 = follow `max_concurrency` (capped
    # at 32), so a window is never smaller than the work the engines can take nor
    # large enough to reintroduce the problem.
    page_window: int = 0
    # How many PROCESSES may rasterise PDFs at once. pdfium permits one caller at
    # a time, so in-process rasterisation is single-threaded however many cores
    # the box has -- which 0.3.66 made explicit with a lock after concurrent
    # access corrupted pdfium and reported "Data format error" on valid files.
    # pdfium IS safe one-document-per-process, so the throughput comes back from
    # workers (codai/ocr/workers/raster_worker.py), one document each.
    # 0 = auto (half the cores, at most 4). 1 = rasterise in-process under the
    # lock: no IPC and no extra processes, which a small host may prefer.
    raster_workers: int = 0
    # How many PROCESSES forward /v1/ocr pages to rented pods. One Python process
    # forwards at most ~16-23 pages/s however many pods it has (measured: one
    # core at 100%, the rest of the box idle, pod GPUs mostly idle), and 128 pods
    # need ~85. With N > 0, coderai-ocr-workers starts N extra engine processes
    # and nginx sends /v1/ocr and /v1/ocr/batch straight to them; the main
    # engine still owns the fleet (renting, scaling, reaping). 0 = off: every
    # page goes through the front and one engine, as before. Rule of thumb: one
    # worker per ~20 pods, at most the number of cores minus two.
    workers: int = 0
    # Default document language when a request names none. "auto" (the shipped
    # default) means let the engine recognise what it finds: surya-2, olmOCR and
    # docTR are multilingual and have no language input at all, and PaddleOCR --
    # the one engine built for a single language -- gets `paddle_auto_lang`.
    lang: str = "auto"
    # What "auto" becomes for PaddleOCR, which has no auto mode. A SCRIPT family,
    # not detection: `latin` covers Italian, French, Spanish, German, Portuguese
    # and the rest of the Latin alphabet. PaddleOCR's own default is `ch`, a worse
    # guess for most documents than no guess. Another script needs naming
    # (lang=arabic|japan|cyrillic|devanagari) or one of the multilingual engines.
    paddle_auto_lang: str = "latin"
    # After an engine fails to come up, refuse that engine FAST for this long instead of
    # re-attempting the load on every request (a vLLM-backed engine burns ~35s per failed
    # boot; Surya-2 once did that 338 times in a row on a card that was 11 GB short).
    build_retry_cooldown_s: float = 60.0
    # When a model load evicts OCR, wait up to this long for in-flight pages to finish
    # before tearing the pool down — the clean swap the model manager does for a busy
    # model, so OCR and models take turns at a page boundary instead of OCR losing the
    # page it was mid-way through. 0 = tear down immediately.
    evict_drain_timeout_s: float = 60.0
    # The STARTING share of the card a VLM OCR engine's own vLLM instance asks for
    # (surya, olmocr or chandra in vllm mode); 0 = use vllm.gpu_memory_utilization.
    # It is a fraction of the card's TOTAL memory and has to cover the engine's whole
    # footprint, not just its weights: measured live, surya-ocr-2 on a 24 GB 3090 needs
    # ~10.9 GB before a single KV block (weights 1.37 + CUDA graphs 0.05 + ~9.4 GB of
    # vision-encoder profiling peak), so the 0.35 this once shipped (8.4 GB) was under
    # the floor and could never start. 0.60 is a reasonable opening bid for a 7B VLM —
    # but it is NOT a measurement of your model on your card, and it does not have to be:
    # when vLLM reports the budget cannot hold the cache, the launcher escalates the
    # share, re-evicts for the larger figure and remembers what booted, keyed by
    # model+card+limits (see vllm_worker.ensure_service). Any model, any card, any
    # request mix converges on a share that fits.
    vlm_gpu_memory_utilization: float = 0.60
    # …and these bring the footprint itself down, which is the better lever. A VLM OCR
    # engine otherwise inherits the LLM backend's ctx (18432 here), and the activation
    # peak vLLM profiles for that size IS the 9.4 GB above. Chunked prefill is on, so
    # capping the tokens per batch shrinks the peak WITHOUT shrinking the usable context.
    # 0 = inherit the backend's value.
    vlm_max_num_batched_tokens: int = 4096
    vlm_max_num_seqs: int = 16
    vlm_max_model_len: int = 0            # 0 = inherit vllm.ctx (a page needs far less)
    # When the chosen engine cannot serve a document, try these engines instead rather
    # than failing the request: a 503 means a document nobody read, and that is strictly
    # worse than a transcription from the second-best engine. "auto" = every other
    # ENABLED engine, fewest-ways-to-fail first (paddle, doctr, olmocr, chandra, surya);
    # "off"/"none" = no fallback (fail with the original error); or an explicit
    # comma/space separated order. Under `pod_pipeline: require` the chain keeps
    # only engines that can run on a pod -- paddle and docTR have no pod-side
    # pipeline, and offering them would be this box reading after all.
    fallback_engines: str = "auto"
    # A page can come back DEGRADED without the engine ever failing: the reply
    # looped until it hit the token cap, or it stopped at the cap and could not
    # be resumed. `fallback_engines` never fires for those -- the call returned
    # 200 -- so a run can come out almost entirely truncated with no error
    # anywhere. These re-read ONLY the degraded pages on another engine and keep
    # whichever answer is not degraded. "auto" = the fallback order; "off" = keep
    # what the first engine produced; or an explicit comma/space separated order.
    #
    # NOTE what "auto" inherits: with `fallback_engines: "off"` the chain is
    # empty and this recovery does nothing. The two are different events -- one
    # engine FAILING is not a page coming back ruined -- so an install that
    # wants no engine substitution on failure and does want a looping page
    # re-read must name the engines here rather than leave this on "auto".
    retry_degraded: str = "auto"
    # Stop re-reading after this many pages of one document, 0 = no limit. A
    # scan the whole fleet reads badly would otherwise pay twice for every page
    # to arrive at the same place.
    retry_degraded_max_pages: int = 0
    # Re-read a page whose text is below this FRACTION of the document's median
    # page, even though the engine flagged nothing. 0 = off, which is the
    # default, because unlike a looping or truncated page this one is INFERRED:
    # the engine says the page is fine and we are disagreeing with it from the
    # shape of the document around it. A real early stop looks exactly like a
    # title page or a signature page, and only the operator knows which their
    # corpus has. 0.25 catches an engine that returned an eighth of what its
    # neighbours did without touching a document of genuinely uneven pages.
    retry_short_pages_below: float = 0.0
    # ...and only in a document with at least this many pages, since a median
    # taken over two pages is not a median.
    retry_short_pages_min_doc: int = 4
    # The status a document gets when EVERY page with ink on it came back
    # empty. 200 with no text is indistinguishable from a blank document, and a
    # client that treats an empty read as an unreadable scan then writes the
    # document off; this says plainly that the read did not happen. 0 keeps the
    # old 200. A partially read document is always 200, with `unread_pages`.
    unread_status: int = 502
    # Wall-clock ceiling for ONE /v1/ocr request, 0 = no limit. A read that
    # dribbles bytes never trips a client's READ timeout -- that timeout is
    # per-read, so a response trickling for nineteen minutes looks healthy to
    # httpx the whole way. A client cannot bound the call without a stopwatch
    # of its own, so the bound belongs here as well.
    request_timeout_s: float = 0.0

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
    surya_instances: int = 1             # WORKER PROCESSES (~600 MB each)
    #: Pages in flight PER worker process, in the served modes only. The
    #: protocol used to be one page per process, so concurrency was the process
    #: count: `surya_instances: 64` meant 64 processes and 37 GB, and on
    #: 2026-10-12 it meant a nine-minute stall with the GPU idle. In a served
    #: mode the process only rasterises a page and waits on a socket, so lanes
    #: buy the same concurrency for one torch import instead of N. Effective
    #: concurrency is instances x lanes. In `local` mode this is ignored (a lane
    #: would be a second copy of the model on the card).
    surya_lanes: int = 4
    #: How many times surya may RE-run a page whose generation repeated itself.
    #: surya's own number is 3, and it is unreachable from its settings (the vLLM
    #: backend never passes `max_retries`), so this is a ceiling we apply by
    #: hooking the decision. Each retry is another full generation at a higher
    #: temperature: measured on the Digesta orchestrator on 2026-10-13, a batch
    #: ran at 2,644 generated tokens per request against pages whose text is
    #: ~400, with a 25-27% multimodal cache hit rate on the pod — a quarter of
    #: all requests were re-sends of an image it had already read. The GPU was
    #: never the limit (18 concurrent, `Waiting: 0`, 5-7% KV cache); the tokens
    #: were being spent re-reading pages the model had already failed to read.
    #: 1 keeps the rescue a retry does buy and drops the long tail; 0 disables
    #: it; -1 means "whatever surya decides".
    surya_max_retries: int = 1
    #: Run surya's LAYOUT predictor beside recognition, filling each page's
    #: ``regions`` with labelled blocks in reading order.
    #:
    #: Off by default because it is a second model (~0.25 GB) and a second pass
    #: over every page, and most callers want text. It is not free and it is
    #: not nothing: ``/v1/ocr/engines`` advertised ``regions: true`` for surya
    #: for a long time while the worker returned a hardcoded empty list, so a
    #: client that chose surya BECAUSE the catalogue said it returned regions
    #: got none, with no error to ask about. The row is now read from this
    #: setting, so the claim is true by construction either way.
    surya_layout: bool = False
    #: Run surya's TABLE RECOGNITION predictor over the regions layout labelled
    #: as tables, filling ``tables`` with cell structure rather than a bbox
    #: alone. Needs ``surya_layout`` (it works from the layout's table regions).
    surya_table_rec: bool = False
    #: Cap on the tokens surya may generate for one page, or 0 for surya's own
    #: 2048 (`chat_completions_batch(max_tokens_default=2048)`, which its vLLM
    #: backend never overrides, so 2048 was the only value available).
    #:
    #: It matters because surya-ocr-2 emits BLOCK MARKUP -- a category and a
    #: bbox per block, before the text -- so a reply stopped at the cap is
    #: incomplete markup, and incomplete markup parses to a few blocks or to
    #: none. The page then arrives as a short fragment or as no text at all,
    #: which is indistinguishable in the finished text from a page the model
    #: could not read. A dense page can legitimately need several thousand
    #: tokens. 0 by default because raising it costs tokens on every page and
    #: the right value depends on the documents; `model_attempts.truncated` in a
    #: page's result says when this is what went wrong.
    surya_max_tokens: int = 0
    #: How many times a page whose reply hit the token cap may be RESUMED from
    #: where it stopped, before the page is returned partial and marked
    #: `model_attempts.complete = false`.
    #:
    #: Raising `surya_max_tokens` only moves the cliff -- a dense enough page
    #: overflows any fixed number, and what comes back is incomplete block
    #: markup that parses to a fragment or to nothing. Resumption is the only
    #: answer that holds for every document: the text so far goes back as a
    #: prefilled assistant turn (`continue_final_message`) so the model carries
    #: on the exact string rather than answering afresh about the same image.
    #:
    #: PER PAGE, not per document: a fifty-page document is fifty independent
    #: pages, each with its own budget, so what this number has to cover is the
    #: densest SINGLE page, not the longest document.
    #:
    #: 8 by default, because returning a page known to be incomplete is a
    #: correctness failure rather than a speed/cost trade-off, and because the
    #: real ceiling is the model's context rather than this number: each
    #: resumption re-sends the text so far as input, so with surya-ocr-2's
    #: 32,768-token context and ~4,000 of it spent on the page image, roughly
    #: thirteen 2,048-token segments exhaust the window whatever is set here.
    #: Raising `surya_max_tokens` lowers that count in proportion. A page that
    #: stops making progress stops early regardless, so a generous budget costs
    #: nothing on a page that cannot use it. 0 disables resumption.
    surya_max_continuations: int = 8
    #: FLOOR on a caller-supplied page budget. 0 = off, and off is the default:
    #: a per-request `max_tokens` exists so the caller can choose, and an
    #: install that silently overrode it would be lying about an API.
    #:
    #: Why it exists at all. A client sent `max_tokens: 3072` on every page of a
    #: corpus whose pages need just over that. Each page therefore generated to
    #: the cap, stopped mid-text, and paid a continuation that RE-SENT the first
    #: 3072 tokens to retrieve the last few hundred -- and by the second cut-off
    #: the model was often repeating, so the page was refused a resume and
    #: re-run whole at a higher temperature. Measured across 594 continuations:
    #: 532 pages reached segment 2, which is the signature of a budget that is
    #: marginally too small rather than far too small -- the most expensive
    #: case, because nearly every page pays double. The cards sat at 1.4% KV
    #: cache while the fleet grew to 65 pods, and the operator paying for them
    #: had no way to decline.
    #:
    #: So: an operator may state a minimum, and a request below it is raised to
    #: it and SAID ONCE per engine. Never lowered -- a caller asking for more
    #: than the floor knows something this setting does not.
    surya_max_tokens_floor: int = 0
    #: surya's repetition detector, retuned. 0 on all three leaves its own
    #: defaults (4 repeats, a 500-character window, scaling 3.0).
    #:
    #: A page it judges degenerate is re-run up to `surya_max_retries` times at
    #: rising temperature and then parsed from whatever came back, so a FALSE
    #: positive costs several full generations and usually returns less text
    #: than the first reply had. Whether a repetition is degeneration depends on
    #: the documents — a corpus of tables, dotted leaders or form fields repeats
    #: short strings legitimately — which is why this is an operator setting and
    #: not a per-request one: it describes the corpus, not the page.
    #:
    #: Raise `surya_repeat_max_repeats` to be more forgiving (more repetitions
    #: before a reply counts as looping), lower it to be stricter.
    surya_repeat_max_repeats: int = 0
    surya_repeat_window: int = 0
    surya_repeat_scaling: float = 0.0
    #: The serving model's context window, or 0 to ask the server for its own
    #: `max_model_len` (vLLM reports it on `/v1/models`).
    #:
    #: This is the ceiling resumption actually runs into, and the one it used to
    #: hit the hard way: each round re-sends the text so far as INPUT, so asking
    #: for another full `surya_max_tokens` once the window is nearly full is not
    #: a bigger segment, it is an HTTP 400 -- and a page that was nearly whole
    #: gets reported as a page that could not be resumed at all. Knowing the
    #: number lets a page stop one segment early, keep what it has, and say so.
    #:
    #: Set it when the server does not report a window, or when it is served
    #: with a shorter `--max-model-len` than the checkpoint advertises.
    surya_context_tokens: int = 0
    #: A page whose text is this short gets its ink measured, so a short result
    #: can be explained rather than guessed at. 0 disables the measurement.
    blank_page_max_chars: int = 200
    #: Below this fraction of non-background pixels a page is reported blank.
    #: 0.5% is well under any page carrying text (a typical text page at 200 DPI
    #: is several per cent) and above the scan edges, specks and watermarks that
    #: an empty scanned page still carries -- a real one measured 0.13%, all of
    #: it a page edge and a diagonal "copia non ufficiale".
    blank_page_ink: float = 0.005
    surya_langs: str = "it"
    # Return the model's own per-token bytes for the places a page came back
    # unreadable (see base.UNREADABLE). Off by default: it is a diagnostic, it
    # costs response weight, and it reaches into surya's internals to get data
    # surya itself constructs and then drops. Turn it on when a document is
    # losing characters and you need to know WHOSE fault it is -- the model
    # emitting a broken byte sequence, or our serving stack mis-assembling a
    # correct one. The two are indistinguishable in the finished text, because
    # surya-ocr-2's vocabulary is byte-level and decoding either byte of an
    # accented letter alone yields exactly one replacement character.
    surya_token_trace: bool = False
    # Rebuild a surya reply's text from the model's OWN per-token bytes when the
    # text we are handed has lost characters. On by default, because the
    # alternative is shipping text measured to be corrupt: on 2026-10-12 a page of
    # crisp synthetic Italian came back from surya-ocr-2 with one U+FFFD per
    # accented letter while the model's byte stream decoded cleanly and it
    # reported ~96% confidence, and chandra read the identical pixels perfectly.
    # The correct bytes arrive in the same response's logprobs, so the reply is
    # simply rebuilt from them. Guarded three ways (strict decode, no marker in
    # the result, and an identical ASCII skeleton) so it can only ever change
    # non-ASCII characters, and it does nothing at all when the text is clean.
    # Set false to see exactly what the server sent.
    surya_repair_accents: bool = True
    surya_venv: str = ""               # isolated venv dir; blank = ~/.coderai/surya_venv
    surya_auto_build: bool = False     # create the venv + pip install requirements-surya.txt on first use
    # Surya serving mode: "local" = classic det+recognition on torch in the isolated venv
    # (surya-ocr <=0.17); the rest serve the latest "Surya2" VLM from an OpenAI server that
    # Surya attaches to (SURYA_INFERENCE_URL), which is where the work then happens:
    #   "model"    — the VLM is a MODEL in models.json (`surya_model_id`), served however
    #                that entry says: a RunPod pod this install rents on demand, another
    #                server via service_url, or the local vLLM backend. The only served
    #                mode that works with no GPU on this box — which is precisely the box
    #                whose job is to rent one.
    #   "vllm"     — coderai's OWN isolated vLLM venv serves `surya_model` here (CUDA).
    #   "llamacpp" — attach to a llama.cpp/OpenAI server already running, at
    #                `surya_server_url`. ("server" is accepted as a synonym.)
    # `local` runs detection and recognition in Surya's own venv, on this
    # machine's GPU. Like paddle and docTR, that is a REQUIREMENT and not a
    # preference: Surya's own settings fall back to CPU whenever torch sees no
    # CUDA device (surya/settings.py TORCH_DEVICE_MODEL: cuda -> mps -> cpu), and
    # a VLM pipeline on a CPU is not slow, it is unusable — while looking
    # perfectly healthy. Set it to false to run on the CPU deliberately.
    surya_use_gpu: bool = True
    # NOT "local": an install that says nothing would get the mode that runs the
    # model here, which on a box with no GPU is a refusal (and used to be a
    # silent CPU run). "model" adapts to wherever the Surya-2 checkpoint is
    # actually served, and asks for ocr.surya_model_id when it is not told.
    surya_serve: str = "model"         # model | local | vllm | llamacpp
    surya_model: str = "datalab-to/surya-ocr-2"   # HF checkpoint for the served (vllm) backend
    surya_model_id: str = ""           # ("model" mode) the models.json id serving Surya2
    surya_server_url: str = ""         # external OpenAI server URL (llamacpp/manual); blank = auto

    # --- how a served engine reaches its model ("model" mode, both engines) --- #
    # An OCR pipeline drives ONE server address itself, which is what keeps its
    # images, json_schema and logprobs intact. One address cannot spread load,
    # though: an attached engine pins to the pod it was given, and a pool that
    # counts its own in-flight requests never sees OCR traffic at all, so it never
    # scales. With fan-out on, the engine is handed an address of OURS that
    # forwards each request — byte for byte, deciding nothing — to the least busy
    # of the local server, the cluster nodes holding the model, and the pod pool.
    # Turn it off to go back to one direct address (one pod, no growth).
    # How busy is too busy comes from the MODEL (its own
    # `scale_up_inflight_per_pod`), not from here — one knob per model, already on
    # the model page, instead of a second OCR-shaped copy of it.
    serve_fanout: bool = True

    # --- olmOCR-2 (AllenAI, Apache-2.0) --- a document VLM: whole-page transcription in
    # reading order, equations as LaTeX, tables as HTML, no bounding boxes. Driven over an
    # OpenAI chat endpoint with olmOCR's own prompt; see codai/ocr/olmocr.py.
    olmocr_enabled: bool = False
    olmocr_instances: int = 1            # in-flight pages against the serving model
    # How it is served:
    #   "model"  — through coderai's OWN model manager: olmocr_model_id names a VISION
    #              model in models.json (GGUF+mmproj on llama.cpp, or HF/qwenvl). Gets
    #              VRAM accounting, eviction and the thermal governor for free, and is the
    #              only mode that works without CUDA.
    #   "vllm"   — coderai's vLLM backend serves olmocr_model on its own instance.
    #   "server" — attach to an OpenAI-compatible server already running it.
    olmocr_serve: str = "model"          # model | vllm | server
    olmocr_model_id: str = ""            # ("model" mode) a vision model id from models.json
    olmocr_model: str = "allenai/olmOCR-2-7B-1025-FP8"   # ("vllm"/"server" mode) HF id / served name
    olmocr_server_url: str = ""          # ("server" mode) OpenAI base URL of that model
    olmocr_api_key: str = ""             # ("server" mode) bearer token, if it wants one
    olmocr_longest_side: int = 1288      # page render size the model was trained on
    olmocr_max_tokens: int = 4096        # a dense page runs long; this is the page budget
    #: Resumptions allowed for a page whose reply hit the cap. The reply is sent
    #: back as a prefilled assistant turn and continued from where it stopped,
    #: so a page longer than one budget still arrives whole. 0 disables it, and
    #: a request may raise or lower it for one document (`max_continuations`).
    #: olmOCR needs this more than most: its reply opens with YAML front matter,
    #: so a generation cut off before the body closes parses to metadata and no
    #: text at all.
    olmocr_max_continuations: int = 8
    #: See `surya_max_tokens_floor`. 0 = off.
    olmocr_max_tokens_floor: int = 0
    #: The serving model's context window, or 0 to leave resumption uncapped by
    #: one. Resumption re-sends everything so far as INPUT, so the window — not
    #: the budget — is what a long page runs out of.
    olmocr_context_tokens: int = 0
    olmocr_temperature: float = 0.1
    olmocr_timeout: float = 300.0        # per-page ceiling on the model call
    olmocr_retry_rotation: bool = True   # re-ask once on a page the model reports rotated
    # As chandra_logprobs, for olmOCR. It returns no lines, so its figure lives on
    # the page's meta — there is nowhere per-line to put one and nothing per-line
    # to measure.
    olmocr_logprobs: bool = True

    # --- Chandra (Datalab; code Apache-2.0, WEIGHTS modified OpenRAIL-M) --- a document
    # VLM like olmOCR, but its `ocr_layout` prompt returns one <div> per layout block with
    # data-bbox (normalised 0-1000) and data-label, so unlike olmOCR it gives boxes AND
    # labelled regions in a single pass. Qwen3-VL family, so a different tokenizer and
    # different failure modes from surya-2 — which makes it the second opinion worth
    # having on a page surya cannot read. See codai/ocr/chandra.py.
    chandra_enabled: bool = False
    # LICENSE-GATED, and on NARROWER terms than Surya's GPL: the weights are free for
    # research, personal use and companies under $2M funding/revenue, and may not be used
    # competitively with Datalab's own API. Commercial self-hosting needs a licence from
    # Datalab. Nothing loads until this is set deliberately.
    chandra_accept_license: bool = False
    chandra_instances: int = 1           # in-flight pages against the serving model
    # Served exactly like olmOCR: "model" (coderai's own model manager — the only mode
    # that works on a GPU-less box, including one whose job is renting a pod), "vllm"
    # (coderai's vLLM backend on its own instance), "server" (an endpoint already up).
    chandra_serve: str = "model"         # model | vllm | server
    chandra_model_id: str = ""           # ("model" mode) a vision model id from models.json
    chandra_model: str = "datalab-to/chandra-ocr-2"   # ("vllm"/"server") HF id / served name
    chandra_server_url: str = ""         # ("server" mode) OpenAI base URL of that model
    chandra_api_key: str = ""            # ("server" mode) bearer token, if it wants one
    # Chandra's input is capped at 3072x2048 = 6.29 Mpixels — the SAME ceiling as surya-2
    # (chandra/model/util.py `scale_to_fit`, same numbers), reached by an A4 page at about
    # 255 DPI. Above it the page is downscaled before the model sees it, so rasterising
    # higher costs time and buys nothing. 250 saturates A4; see `surya_dpi`.
    chandra_dpi: int = 250
    # Ask for layout blocks (the `ocr_layout` prompt) rather than plain HTML. On by
    # default because it is the whole reason to choose this engine over olmOCR: with it
    # off the reply is one unlocated blob and every bbox comes back empty.
    chandra_layout: bool = True
    # Keep Page-Header/Page-Footer blocks in the text. On: a judicial document's header
    # carries the court, the section and the case number.
    chandra_include_furniture: bool = True
    chandra_max_tokens: int = 12384      # chandra's own MAX_OUTPUT_TOKENS; HTML runs long
    #: Resumptions allowed for a page whose reply hit the cap (see
    #: olmocr_max_continuations). chandra emits HTML, so a cut-off reply is
    #: unclosed markup — which parses to a fragment of the page or to nothing,
    #: and reads in the finished text exactly like a page the model could not
    #: transcribe. 0 disables it; a request may override it per document.
    chandra_max_continuations: int = 8
    #: See `surya_max_tokens_floor`. 0 = off.
    chandra_max_tokens_floor: int = 0
    #: The serving model's context window, or 0 for no ceiling of our own.
    chandra_context_tokens: int = 0
    chandra_temperature: float = 0.0     # chandra's own default: greedy
    chandra_top_p: float = 0.1
    # A VLM that loses its place on a page does not fail — it repeats the last thing it
    # saw until it runs out of tokens, and answers 200 with a plausible-looking page. Each
    # retry raises the temperature (a greedy decode falls into the same loop twice), which
    # is what chandra's own client does. 0 = accept the first reply whatever it looks like.
    chandra_max_retries: int = 2
    chandra_timeout: float = 300.0       # per-page ceiling on the model call
    # Ask the server for per-token logprobs, which is what makes the reported
    # `conf` a MEASUREMENT (the mean of exp(logprob), the same computation surya
    # does) rather than a placeholder. Cheap: one float per token. Chandra
    # generates a whole page in one call, so the figure is page-level and every
    # line carries it, with meta.confidence_scope saying so; a server that
    # declines the field leaves conf at 0.0 and confidence_measured false.
    chandra_logprobs: bool = True

    # --- several published ids for one engine ---
    #: Named configurations, each served under its own id and reported as its
    #: own row by ``/v1/ocr/engines``::
    #:
    #:     "engine_configs": [
    #:       {"id": "surya-layout", "engine": "surya",
    #:        "overrides": {"surya_layout": true, "surya_table_rec": true}},
    #:       {"id": "chandra-plain", "engine": "chandra",
    #:        "overrides": {"chandra_layout": false}}
    #:     ]
    #:
    #: The same thing image models have had: one set of weights, several
    #: published ids, each with its own settings and its own line in the
    #: catalogue. An id gets its own engine pool, so two ids of one engine that
    #: differ in what they LOAD are two pools, while ids that differ only in
    #: what they ask of a loaded model still cost one.
    #:
    #: Every key in ``overrides`` must be a real OcrConfig field; a typo is
    #: refused at resolution rather than silently serving the base settings.
    engine_configs: list = field(default_factory=list)

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
    admin: AdminConfig = field(default_factory=AdminConfig)
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
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    ds4: Ds4Config = field(default_factory=Ds4Config)
    colibri: ColibriConfig = field(default_factory=ColibriConfig)
    k3: K3Config = field(default_factory=K3Config)
    ktransformers: KtransformersConfig = field(default_factory=KtransformersConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    longcat: LongcatConfig = field(default_factory=LongcatConfig)
    runpod: RunpodConfig = field(default_factory=RunpodConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
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


def seed_entry_name(entry: dict) -> str:
    """The name a seeded entry is asked for by — not necessarily its path."""
    if not isinstance(entry, dict):
        return ""
    return (str(entry.get("alias") or "").strip()
            or str(entry.get("id") or "").strip())


def _same_model(known: dict, seed: dict) -> bool:
    """Is ``seed`` the model ``known`` already is?

    Not "same path". A LoRA and the base it adapts share one path BY DEFINITION
    — the adapter is a few MB of deltas over the base weights — so keying
    identity on the path alone made the second of the two vanish into the first.
    A pod seeded with both was told about one, and answered 404 under the other
    name for the rest of its paid life. Which one survived depended on the order
    they arrived in, so it looked intermittent rather than wrong.

    Identity is (weights, name): the same path under a DIFFERENT name is another
    model that happens to load from the same place.
    """
    if str(known.get("path") or "") != str(seed.get("path") or ""):
        return False
    a, b = seed_entry_name(known), seed_entry_name(seed)
    return not a or not b or a == b


def _is_ocr_worker() -> bool:
    """True in a process started by coderai-ocr-workers (CODERAI_OCR_WORKER=i/N)."""
    try:
        from codai.api.runpod_fleet_share import is_follower
        return is_follower()
    except Exception:
        return False


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

            # Which server keys the operator actually WROTE, as opposed to
            # which ones have dataclass defaults. Needed because some defaults
            # are only right for some deployments: the admission gate of 2 is
            # correct for one local card answering chat and crippling for an
            # OCR install dispatching pages to a rented fleet, and the two
            # cases are indistinguishable from the parsed value alone. An
            # explicit 2 must be obeyed; an absent 2 may be derived.
            self.explicit_server_keys = set(
                (config_data.get("server") or {}).keys())

            self.config = Config(
                version=config_data.get("version", "1.0"),
                server=_dc(ServerConfig, config_data.get("server", {})),
                backend=_dc(BackendConfig, config_data.get("backend", {})),
                admin=AdminConfig(forward_auth=_dc(
                    AdminForwardAuthConfig,
                    (config_data.get("admin") or {}).get("forward_auth", {}))),
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
                cluster=_dc(ClusterConfig, config_data.get("cluster", {})),
                ds4=_dc(Ds4Config, config_data.get("ds4", {})),
                colibri=_dc(ColibriConfig, config_data.get("colibri", {})),
                k3=_dc(K3Config, config_data.get("k3", {})),
                ktransformers=_dc(KtransformersConfig, config_data.get("ktransformers", {})),
                vllm=_dc(VllmConfig, config_data.get("vllm", {})),
                longcat=_dc(LongcatConfig, config_data.get("longcat", {})),
                runpod=_dc(RunpodConfig, config_data.get("runpod", {})),
                ocr=_dc(OcrConfig, config_data.get("ocr", {})),
                compaction=_dc(CompactionConfig, config_data.get("compaction", {})),
                chunking=_dc(ChunkingConfig, config_data.get("chunking", {})),
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
        # A capability image has no config to edit: its cluster membership
        # comes from the environment it was launched with (docker -e …).
        _env = __import__("os").environ
        if _env.get("CODERAI_CLUSTER_TOKEN"):
            self.config.cluster.token = _env["CODERAI_CLUSTER_TOKEN"].strip()
        if _env.get("CODERAI_DISCOVERY", "").strip().lower() in ("1", "true", "yes", "on"):
            self.config.cluster.discovery = True
        if _env.get("CODERAI_NODE_NAME"):
            self.config.cluster.node_name = _env["CODERAI_NODE_NAME"].strip()
        if _env.get("CODERAI_ADVERTISE_HOST"):
            self.config.cluster.advertise_host = _env["CODERAI_ADVERTISE_HOST"].strip()

        if not self.config_path.exists():
            # No config.json at all: nothing was explicitly set.
            self.explicit_server_keys = set()

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
        if _is_ocr_worker():
            # A worker's ocr ceilings are its 1/N share, not the configured
            # values; writing them back would shrink the real config by N on
            # every save. The front and the main engine own this file.
            return
        config_dict = {
            "version": self.config.version,
            "server": {
                "host": self.config.server.host,
                "port": self.config.server.port,
                "https": self.config.server.https,
                "https_key_path": self.config.server.https_key_path,
                "https_cert_path": self.config.server.https_cert_path,
                "metrics_public": self.config.server.metrics_public,
                "queue_max_size": self.config.server.queue_max_size,
                "max_parallel_requests": self.config.server.max_parallel_requests,
                "max_parallel_requests_overrides": self.config.server.max_parallel_requests_overrides,
                "dpm_force_performance_level_overrides": self.config.server.dpm_force_performance_level_overrides,
                "internal_port_base": self.config.server.internal_port_base,
                "engines": self.config.server.engines,
                "engine_gpus": self.config.server.engine_gpus,
                "proxy_status_timeout": self.config.server.proxy_status_timeout,
                "proxy_max_inflight": self.config.server.proxy_max_inflight,
                "inflight_max_age_s": self.config.server.inflight_max_age_s,
                "gpu_swap_batch": self.config.server.gpu_swap_batch,
                "engine_restart_drain_grace": self.config.server.engine_restart_drain_grace,
                "isolate_gguf_engine": self.config.server.isolate_gguf_engine,
                "engine_specs": self.config.server.engine_specs,
                "default_engine": self.config.server.default_engine,
            },
            "admin": {
                "forward_auth": {
                    "enabled": self.config.admin.forward_auth.enabled,
                    "user_header": self.config.admin.forward_auth.user_header,
                    "groups_header": self.config.admin.forward_auth.groups_header,
                    "admin_group": self.config.admin.forward_auth.admin_group,
                    "shared_secret_header": self.config.admin.forward_auth.shared_secret_header,
                    "shared_secret": self.config.admin.forward_auth.shared_secret,
                    "trusted_proxies": self.config.admin.forward_auth.trusted_proxies,
                    "auto_create_users": self.config.admin.forward_auth.auto_create_users,
                    "keep_local_login": self.config.admin.forward_auth.keep_local_login,
                    "admin_without_groups": self.config.admin.forward_auth.admin_without_groups,
                    "post_logout_url": self.config.admin.forward_auth.post_logout_url,
                },
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
                "clip_on_cpu": self.config.image.clip_on_cpu,
                "clip_skip": self.config.image.clip_skip,
                "prompt_weighting": self.config.image.prompt_weighting,
                "max_side": self.config.image.max_side,
                "max_pixels": self.config.image.max_pixels,
                "embeddings_dir": self.config.image.embeddings_dir
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
            "cluster": __import__("dataclasses").asdict(self.config.cluster),
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
                "nnodes": self.config.ktransformers.nnodes,
                "dist_init_addr": self.config.ktransformers.dist_init_addr,
                "tp_size": self.config.ktransformers.tp_size,
                "nodes": list(self.config.ktransformers.nodes or []),
            },
            "longcat": {
                "enabled": self.config.longcat.enabled,
                "service_url": self.config.longcat.service_url,
                "venv": self.config.longcat.venv,
                "python": self.config.longcat.python,
                "auto_build": self.config.longcat.auto_build,
                "host": self.config.longcat.host,
                "port": self.config.longcat.port,
                "gpu": self.config.longcat.gpu,
                "attention": self.config.longcat.attention,
                "ready_timeout": self.config.longcat.ready_timeout,
                "evict_drain_timeout_s": self.config.longcat.evict_drain_timeout_s,
                "train_venv": self.config.longcat.train_venv,
                "train_auto_build": self.config.longcat.train_auto_build,
                "train_base_precision": self.config.longcat.train_base_precision,
                "train_lora_rank": self.config.longcat.train_lora_rank,
                "train_gradient_checkpointing": self.config.longcat.train_gradient_checkpointing,
                "extra_args": self.config.longcat.extra_args,
                "extra_env": self.config.longcat.extra_env,
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
                "return_token_ids": self.config.vllm.return_token_ids,
                "tensor_parallel_size": self.config.vllm.tensor_parallel_size,
                "max_num_seqs": self.config.vllm.max_num_seqs,
                "dtype": self.config.vllm.dtype,
                "quantization": self.config.vllm.quantization,
                "extra_args": self.config.vllm.extra_args,
                "extra_env": self.config.vllm.extra_env,
                "auto_build": self.config.vllm.auto_build,
                "pipeline_parallel_size": self.config.vllm.pipeline_parallel_size,
                "distributed_executor_backend": self.config.vllm.distributed_executor_backend,
                "ray_address": self.config.vllm.ray_address,
                "ray_port": self.config.vllm.ray_port,
                "nodes": list(self.config.vllm.nodes or []),
                "nodes_ready_timeout_s": self.config.vllm.nodes_ready_timeout_s,
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
                "network_volumes": dict(self.config.runpod.network_volumes or {}),
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
                "max_resident_mb": self.config.ocr.max_resident_mb,
                "max_pages": self.config.ocr.max_pages,
                "max_upload_mb": self.config.ocr.max_upload_mb,
                "pod_pipeline": self.config.ocr.pod_pipeline,
                "pod_pipeline_wait_s": self.config.ocr.pod_pipeline_wait_s,
                "pod_http_async": self.config.ocr.pod_http_async,
                "page_image_format": self.config.ocr.page_image_format,
                "page_image_jpeg_quality": self.config.ocr.page_image_jpeg_quality,
                "page_image_min_dpi": self.config.ocr.page_image_min_dpi,
                "page_image_max_dpi_factor": self.config.ocr.page_image_max_dpi_factor,
                "page_window": self.config.ocr.page_window,
                "raster_workers": self.config.ocr.raster_workers,
                "workers": self.config.ocr.workers,
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
                "surya_lanes": getattr(self.config.ocr, "surya_lanes", 4),
                "surya_max_retries": getattr(self.config.ocr, "surya_max_retries", 1),
                "surya_max_tokens": getattr(self.config.ocr, "surya_max_tokens", 0),
                "surya_max_continuations": getattr(self.config.ocr,
                                                   "surya_max_continuations", 2),
                "surya_repeat_max_repeats": getattr(
                    self.config.ocr, "surya_repeat_max_repeats", 0),
                "surya_repeat_window": getattr(self.config.ocr,
                                               "surya_repeat_window", 0),
                "surya_repeat_scaling": getattr(self.config.ocr,
                                                "surya_repeat_scaling", 0.0),
                "surya_langs": self.config.ocr.surya_langs,
                "surya_token_trace": self.config.ocr.surya_token_trace,
                "surya_repair_accents": self.config.ocr.surya_repair_accents,
                "surya_venv": self.config.ocr.surya_venv,
                "surya_auto_build": self.config.ocr.surya_auto_build,
                "surya_use_gpu": self.config.ocr.surya_use_gpu,
                "surya_serve": self.config.ocr.surya_serve,
                "surya_model": self.config.ocr.surya_model,
                "surya_model_id": self.config.ocr.surya_model_id,
                "surya_server_url": self.config.ocr.surya_server_url,
                "serve_fanout": self.config.ocr.serve_fanout,
                "build_retry_cooldown_s": self.config.ocr.build_retry_cooldown_s,
                "evict_drain_timeout_s": self.config.ocr.evict_drain_timeout_s,
                "vlm_gpu_memory_utilization": self.config.ocr.vlm_gpu_memory_utilization,
                "vlm_max_num_batched_tokens": self.config.ocr.vlm_max_num_batched_tokens,
                "vlm_max_num_seqs": self.config.ocr.vlm_max_num_seqs,
                "vlm_max_model_len": self.config.ocr.vlm_max_model_len,
                "fallback_engines": self.config.ocr.fallback_engines,
                "retry_degraded": self.config.ocr.retry_degraded,
                "retry_degraded_max_pages": self.config.ocr.retry_degraded_max_pages,
                "unread_status": self.config.ocr.unread_status,
                "request_timeout_s": self.config.ocr.request_timeout_s,
                "retry_short_pages_below": self.config.ocr.retry_short_pages_below,
                "retry_short_pages_min_doc": self.config.ocr.retry_short_pages_min_doc,
                "olmocr_enabled": self.config.ocr.olmocr_enabled,
                "olmocr_instances": self.config.ocr.olmocr_instances,
                "olmocr_serve": self.config.ocr.olmocr_serve,
                "olmocr_model_id": self.config.ocr.olmocr_model_id,
                "olmocr_model": self.config.ocr.olmocr_model,
                "olmocr_server_url": self.config.ocr.olmocr_server_url,
                "olmocr_api_key": self.config.ocr.olmocr_api_key,
                "olmocr_longest_side": self.config.ocr.olmocr_longest_side,
                "olmocr_max_tokens": self.config.ocr.olmocr_max_tokens,
                "olmocr_max_continuations": self.config.ocr.olmocr_max_continuations,
                "olmocr_context_tokens": self.config.ocr.olmocr_context_tokens,
                "olmocr_temperature": self.config.ocr.olmocr_temperature,
                "olmocr_timeout": self.config.ocr.olmocr_timeout,
                "olmocr_retry_rotation": self.config.ocr.olmocr_retry_rotation,
                "olmocr_logprobs": self.config.ocr.olmocr_logprobs,
                "chandra_enabled": self.config.ocr.chandra_enabled,
                "chandra_accept_license": self.config.ocr.chandra_accept_license,
                "chandra_instances": self.config.ocr.chandra_instances,
                "chandra_serve": self.config.ocr.chandra_serve,
                "chandra_model_id": self.config.ocr.chandra_model_id,
                "chandra_model": self.config.ocr.chandra_model,
                "chandra_server_url": self.config.ocr.chandra_server_url,
                "chandra_api_key": self.config.ocr.chandra_api_key,
                "chandra_dpi": self.config.ocr.chandra_dpi,
                "chandra_layout": self.config.ocr.chandra_layout,
                "chandra_include_furniture": self.config.ocr.chandra_include_furniture,
                "chandra_max_tokens": self.config.ocr.chandra_max_tokens,
                "chandra_max_continuations": self.config.ocr.chandra_max_continuations,
                "chandra_context_tokens": self.config.ocr.chandra_context_tokens,
                "chandra_temperature": self.config.ocr.chandra_temperature,
                "chandra_top_p": self.config.ocr.chandra_top_p,
                "chandra_max_retries": self.config.ocr.chandra_max_retries,
                "chandra_timeout": self.config.ocr.chandra_timeout,
                "chandra_logprobs": self.config.ocr.chandra_logprobs,
                "detect_mode": self.config.ocr.detect_mode,
                "detect_model_path": self.config.ocr.detect_model_path,
                "detect_conf": self.config.ocr.detect_conf,
                "extract_enabled": self.config.ocr.extract_enabled,
                "extract_model_id": self.config.ocr.extract_model_id,
                "extract_max_tokens": self.config.ocr.extract_max_tokens,
                "extract_schema": self.config.ocr.extract_schema,
                "extract_validate": self.config.ocr.extract_validate,
            },
            "chunking": {
                "chunk_tokens": self.config.chunking.chunk_tokens,
                "overlap": self.config.chunking.overlap,
                "chars_per_token": self.config.chunking.chars_per_token,
                "structural": self.config.chunking.structural,
                "structure": self.config.chunking.structure,
                "require_text": self.config.chunking.require_text,
                "max_chars": self.config.chunking.max_chars,
                "templates": self.config.chunking.templates,
            },
            "compaction": {
                "enabled": self.config.compaction.enabled,
                "pct": self.config.compaction.pct,
                "strategy": self.config.compaction.strategy,
                "model": self.config.compaction.model,
                "tolerance_pct": self.config.compaction.tolerance_pct,
                "min_output": self.config.compaction.min_output,
                "estimate_safety": self.config.compaction.estimate_safety,
                "head_messages": self.config.compaction.head_messages,
                "head_tokens": self.config.compaction.head_tokens,
                "tail_tokens": self.config.compaction.tail_tokens,
                "target_pct": self.config.compaction.target_pct,
                "retry_on_overflow": self.config.compaction.retry_on_overflow,
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
            for name in ("paddle", "doctr", "surya", "olmocr", "chandra"):
                if _flag(f"CODERAI_OCR_{name.upper()}_ENABLED"):
                    setattr(self.config.ocr, f"{name}_enabled", True)
                # Where that engine's interpreter lives. A pod image that ships
                # the engine in its OWN venv points this at that interpreter, so
                # the subprocess engine runs it instead of trying to build a venv
                # on a machine rented by the second.
                venv = (_os.environ.get(f"CODERAI_OCR_{name.upper()}_VENV") or "").strip()
                if venv and hasattr(self.config.ocr, f"{name}_venv"):
                    setattr(self.config.ocr, f"{name}_venv", venv)
            # olmOCR needs to be told WHERE its model is; a pod that carries no vLLM
            # is given a server URL (or the id of a vision model it does hold).
            # Surya's served mode is seeded the same way, and so is the fallback
            # chain: on a pod "auto" would hand a failed page to an engine the image
            # cannot serve, so the renting install pins it to what is present.
            for _env, _attr in (("CODERAI_OCR_OLMOCR_SERVE", "olmocr_serve"),
                                ("CODERAI_OCR_OLMOCR_MODEL", "olmocr_model"),
                                ("CODERAI_OCR_OLMOCR_MODEL_ID", "olmocr_model_id"),
                                ("CODERAI_OCR_OLMOCR_SERVER_URL", "olmocr_server_url"),
                                ("CODERAI_OCR_SURYA_SERVE", "surya_serve"),
                                ("CODERAI_OCR_SURYA_MODEL", "surya_model"),
                                ("CODERAI_OCR_SURYA_MODEL_ID", "surya_model_id"),
                                ("CODERAI_OCR_SURYA_SERVER_URL", "surya_server_url"),
                                ("CODERAI_OCR_CHANDRA_SERVE", "chandra_serve"),
                                ("CODERAI_OCR_CHANDRA_MODEL", "chandra_model"),
                                ("CODERAI_OCR_CHANDRA_MODEL_ID", "chandra_model_id"),
                                ("CODERAI_OCR_CHANDRA_SERVER_URL", "chandra_server_url"),
                                ("CODERAI_OCR_FALLBACK_ENGINES", "fallback_engines"),
                                ("CODERAI_OCR_RETRY_DEGRADED", "retry_degraded")):
                _val = (_os.environ.get(_env) or "").strip()
                if _val:
                    setattr(self.config.ocr, _attr, _val)
            # The TUNING, not just the wiring. A pod that runs the OCR pipeline
            # itself (runpod `pod_ocr_pipeline`) must read a page the same way
            # the orchestrator would, and every one of these was set in response
            # to a measured failure: the 3072-token page budget that surya's own
            # per-item 12288 used to beat, the continuation count, the repeat
            # detector's thresholds, the DPI that saturates surya's pixel cap.
            # Seeding the serve URL but not the budgets would give a pod running
            # surya's defaults -- which is indistinguishable, from the outside,
            # from the "pre-fix pod image" that cost three rounds of blame.
            for _env, _attr, _cast in (
                    ("CODERAI_OCR_DPI", "dpi", int),
                    ("CODERAI_OCR_SURYA_DPI", "surya_dpi", int),
                    ("CODERAI_OCR_SURYA_MAX_TOKENS", "surya_max_tokens", int),
                    ("CODERAI_OCR_SURYA_MAX_CONTINUATIONS",
                     "surya_max_continuations", int),
                    ("CODERAI_OCR_SURYA_CONTEXT_TOKENS",
                     "surya_context_tokens", int),
                    ("CODERAI_OCR_SURYA_MAX_TOKENS_FLOOR",
                     "surya_max_tokens_floor", int),
                    ("CODERAI_OCR_CHANDRA_MAX_TOKENS_FLOOR",
                     "chandra_max_tokens_floor", int),
                    ("CODERAI_OCR_OLMOCR_MAX_TOKENS_FLOOR",
                     "olmocr_max_tokens_floor", int),
                    ("CODERAI_OCR_SURYA_MAX_RETRIES", "surya_max_retries", int),
                    ("CODERAI_OCR_SURYA_REPEAT_MAX_REPEATS",
                     "surya_repeat_max_repeats", int),
                    ("CODERAI_OCR_SURYA_REPEAT_WINDOW",
                     "surya_repeat_window", int),
                    ("CODERAI_OCR_SURYA_REPEAT_SCALING",
                     "surya_repeat_scaling", float),
                    ("CODERAI_OCR_SURYA_INSTANCES", "surya_instances", int),
                    ("CODERAI_OCR_SURYA_LANES", "surya_lanes", int),
                    ("CODERAI_OCR_SURYA_LANGS", "surya_langs", str),
                    # How many pages the engine serves AT ONCE. These were sent
                    # to every pod and read for surya only, so a pod running
                    # chandra or olmOCR fell back to the dataclass default of
                    # one instance however many the operator configured -- and
                    # a pod has no config.json, so env is the only channel it
                    # has. Measured consequence on one A40: 32 pages in flight,
                    # `[ocr] engine 'chandra': 1 instance(s) ready`, vLLM
                    # reporting `Running: 1, Waiting: 0` at 1.6% KV cache and
                    # 55 tokens/s, against surya's ~2000 on the same card. It
                    # read as "chandra is 36x slower"; it was one page at a time.
                    ("CODERAI_OCR_CHANDRA_INSTANCES", "chandra_instances", int),
                    ("CODERAI_OCR_OLMOCR_INSTANCES", "olmocr_instances", int),
                    # The rest of each engine's page budget, for the same
                    # reason: seeding the wiring and not the tuning gives a pod
                    # that runs the engine's defaults, which is
                    # indistinguishable from the wrong image.
                    ("CODERAI_OCR_CHANDRA_MAX_TOKENS", "chandra_max_tokens", int),
                    ("CODERAI_OCR_CHANDRA_MAX_CONTINUATIONS",
                     "chandra_max_continuations", int),
                    ("CODERAI_OCR_CHANDRA_MAX_RETRIES", "chandra_max_retries", int),
                    ("CODERAI_OCR_CHANDRA_DPI", "chandra_dpi", int),
                    ("CODERAI_OCR_OLMOCR_MAX_TOKENS", "olmocr_max_tokens", int),
                    ("CODERAI_OCR_OLMOCR_MAX_CONTINUATIONS",
                     "olmocr_max_continuations", int),
                    ("CODERAI_OCR_MAX_CONCURRENCY", "max_concurrency", int),
                    ("CODERAI_OCR_MAX_QUEUED_PAGES", "max_queued_pages", int),
                    ("CODERAI_OCR_EARLY_LOOP_MIN_CHARS", "early_loop_min_chars", int),
                    ("CODERAI_OCR_EARLY_LOOP_STEP", "early_loop_step", int),
                    ("CODERAI_OCR_EARLY_LOOP_PERSISTENCE", "early_loop_persistence", int),
                    ("CODERAI_OCR_MAX_RESIDENT_MB", "max_resident_mb", int),
                    ("CODERAI_OCR_PAGE_WINDOW", "page_window", int),
                    ("CODERAI_OCR_RASTER_WORKERS", "raster_workers", int),
                    ("CODERAI_OCR_BLANK_PAGE_MAX_CHARS",
                     "blank_page_max_chars", int),
                    ("CODERAI_OCR_DETECT_MODE", "detect_mode", str),
                    ("CODERAI_OCR_DETECT_MODEL_PATH", "detect_model_path", str),
                    ("CODERAI_OCR_DETECT_CONF", "detect_conf", float)):
                _val = (_os.environ.get(_env) or "").strip()
                if not _val:
                    continue
                try:
                    setattr(self.config.ocr, _attr, _cast(_val))
                except (TypeError, ValueError):
                    print(f"[config] {_env}={_val!r} is not a {_cast.__name__}; "
                          f"leaving ocr.{_attr} alone", flush=True)
            for _env, _attr in (
                    ("CODERAI_OCR_SURYA_REPAIR_ACCENTS", "surya_repair_accents"),
                    ("CODERAI_OCR_SURYA_TOKEN_TRACE", "surya_token_trace"),
                    ("CODERAI_OCR_SURYA_USE_GPU", "surya_use_gpu"),
                    # Whether this pod's surya worker loads the layout
                    # predictor and the table recogniser. Both are models, so
                    # "off" has to be expressible: see the note below for why
                    # a boolean cannot travel through the typed loop.
                    ("CODERAI_OCR_SURYA_LAYOUT", "surya_layout"),
                    ("CODERAI_OCR_SURYA_TABLE_REC", "surya_table_rec"),
                    # HERE, not in the typed loop above. That loop casts with
                    # the declared type, and `bool("0")` is True in Python -- so
                    # a pod sent CODERAI_OCR_EARLY_LOOP_ABORT=0 turned the
                    # feature ON. `_flag` reads the STRING, which is the only
                    # way an env var can express false.
                    ("CODERAI_OCR_EARLY_LOOP_ABORT", "early_loop_abort")):
                if _os.environ.get(_env) is not None:
                    setattr(self.config.ocr, _attr, _flag(_env))
            # vLLM settings a POD needs but cannot inherit: it launches its own
            # engine from its own config. `return_token_ids` is the one the
            # accent repair depends on -- see vllm_worker._wants_token_ids for
            # what its absence cost.
            if _os.environ.get("CODERAI_VLLM_RETURN_TOKEN_IDS") is not None:
                self.config.vllm.return_token_ids = _flag(
                    "CODERAI_VLLM_RETURN_TOKEN_IDS")
            if _flag("CODERAI_OCR_SURYA_ACCEPT_LICENSE"):
                # Surya is GPL and gated on an explicit acceptance. The pod
                # inherits the decision made here; it cannot make it itself.
                self.config.ocr.surya_accept_license = True
            if _flag("CODERAI_OCR_CHANDRA_ACCEPT_LICENSE"):
                # Chandra's weights are a modified OpenRAIL-M, and on narrower
                # terms than Surya's GPL. Same rule: the pod inherits the
                # decision made here and cannot make it itself.
                self.config.ocr.chandra_accept_license = True

        # An OCR worker (CODERAI_OCR_WORKER=i/N, started by coderai-ocr-workers)
        # is one of N processes that nginx spreads /v1/ocr over. Every per-process
        # ceiling in its config was written for ONE process, so N copies of it
        # would admit N times the configured queue and spawn N times the local
        # engine instances. Each worker takes its share instead, rounded up.
        try:
            from codai.api.runpod_fleet_share import worker_slot as _wslot
            _slot = _wslot()
        except Exception:
            _slot = None
        if _slot and _slot[1] > 1:
            _n = _slot[1]
            for _attr in ("max_concurrency", "max_queued_pages", "surya_instances",
                          "chandra_instances", "olmocr_instances"):
                _v = getattr(self.config.ocr, _attr, None)
                if isinstance(_v, int) and _v > 0:
                    setattr(self.config.ocr, _attr, max(1, -(-_v // _n)))

        # The native engines — ds4, colibri, k3 — ship disabled for the same
        # reason OCR does, and an engines pod is rented to run exactly one of
        # them. CODERAI_<ENGINE>_ENABLED switches it on; CODERAI_<ENGINE>_CONFIG
        # carries the renting install's settings for it (context, expert
        # cache, download variant…) as JSON, applied field by field onto the
        # dataclass — only fields it already has, so a stray key cannot plant
        # anything. Where the binary lives is the image's business
        # (CODERAI_<ENGINE>_DIR, read by the worker itself), not this block's.
        for name, attr in (("ds4", "ds4"), ("colibri", "colibri"), ("k3", "k3"),
                           ("kt", "ktransformers"), ("vllm", "vllm")):
            if not _flag(f"CODERAI_{name.upper()}_ENABLED"):
                continue
            eng = getattr(self.config, attr, None)
            if eng is None:
                continue
            eng.enabled = True
            raw = (_os.environ.get(f"CODERAI_{name.upper()}_CONFIG") or "").strip()
            if not raw:
                continue
            try:
                forwarded = json.loads(raw)
            except Exception as exc:
                print(f"[seed] CODERAI_{name.upper()}_CONFIG is not valid JSON: {exc}",
                      flush=True)
                continue
            if not isinstance(forwarded, dict):
                continue
            for k, v in forwarded.items():
                if k in ("enabled", "install_dir", "repo_url", "service_url",
                         "host", "port", "model_path", "auto_build", "venv"):
                    continue            # local to whichever machine set them
                if hasattr(eng, k):
                    setattr(eng, k, v)
            print(f"[seed] {name} enabled with {sorted(forwarded)}", flush=True)

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
            if any(isinstance(m, dict) and _same_model(m, e) for m in lst):
                continue            # a real entry already covers it — leave it alone
            lst.append(dict(e))
            print(f"[seed] registered {seed_entry_name(e) or path} in {section}",
                  flush=True)
        # The catalogue alone is not enough: request validation asks the MODEL
        # MANAGER for its allowed identifiers, and a seeded pod whose manager was
        # never told still answers "not available" with an empty list. The
        # manager is set up after this runs, so the work is deferred to it.
        self._pending_seed_entries = [e for e in entries if isinstance(e, dict)
                                      and e.get("path")]

    def save_models(self):
        """Save models.json to disk."""
        if _is_ocr_worker():
            return
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
