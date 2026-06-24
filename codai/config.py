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
    engine_restart_drain_grace: float = 30.0  # on engine restart, wait this many seconds
                                              # for in-flight requests to finish before
                                              # killing the process (0 = bounce immediately)
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
    ds4: Ds4Config = field(default_factory=Ds4Config)
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
                ds4=_dc(Ds4Config, config_data.get("ds4", {})),
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
                "internal_port_base": self.config.server.internal_port_base,
                "engines": self.config.server.engines,
                "engine_gpus": self.config.server.engine_gpus,
                "proxy_status_timeout": self.config.server.proxy_status_timeout,
                "proxy_max_inflight": self.config.server.proxy_max_inflight,
                "engine_restart_drain_grace": self.config.server.engine_restart_drain_grace,
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
            "ds4": {
                "enabled": self.config.ds4.enabled,
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
    
    def save_models(self):
        """Save models.json to disk."""
        with open(self.models_path, 'w') as f:
            json.dump(self.models_data, f, indent=2)

    def persist_model_field(self, model_path: str, key: str, value) -> bool:
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
                epath = entry.get("path") or entry.get("id") or ""
                if epath == bare or epath.split("/")[-1] == bare.split("/")[-1]:
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
