# CoderAI Configuration & Admin Dashboard Design

## Overview

Refactor coderai from a complex CLI-driven application to a configuration-file-based system with a comprehensive web administration dashboard. All command-line options (except `--debug` and `--config`) are replaced by JSON configuration files stored in `~/.coderai/` by default.

## 1. CLI Changes

### Removed CLI Options
All existing options in `codai/cli.py` (446 lines of arguments) are removed except:

### Retained CLI Options
- `--debug`: Enable debug output (default: false)
- `--config DIR`: Set configuration directory (default: `~/.coderai`)

### Initialization Flow
1. Parse `--config` (default: `~/.coderai/`)
2. Create config directory if it doesn't exist
3. If config directory is empty, create default minimal config files
4. Load configuration from JSON files
5. Start server with settings from config

---

## 2. Configuration File Structure

All configuration stored as JSON in the config directory:

### `config.json` - Main Configuration
```json
{
  "version": "1.0",
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "https": false,
    "https_key_path": null,
    "https_cert_path": null
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
  "system_prompt": null,
  "tools_closer_prompt": false,
  "grammar_guided": false,
  "file_path": null,
  "hf_chat_templates": [],
  "reasoning_options": [],
  "parser": "auto"
}
```

**Note**: All model-specific settings (GPU layers, quantization, context size, image generation parameters, etc.) are now stored per-model in `models.json` rather than as global defaults in `config.json`. This allows different models to have different configurations even if they share the same backend or capability type.

### `models.json` - Model Registry & Configurations
```json
{
  "text_models": [
    {
      "id": "microsoft/DialoGPT-medium",
      "backend": "nvidia",
      "context_size": 0,
      "n_gpu_layers": -1,
      "load_in_4bit": false,
      "load_in_8bit": false,
      "flash_attn": false,
      "offload_strategy": "auto",
      "manual_ram_gb": null,
      "max_gpu_percent": null,
      "no_ram": false,
      "enabled": true
    }
  ],
  "image_models": [
    {
      "id": "stable-diffusion-xl-base-1.0",
      "backend": "nvidia",
      "llm_path": null,
      "vae_path": null,
      "sample_method": "res_multistep",
      "steps": 4,
      "width": 512,
      "height": 512,
      "cfg_scale": 1.0,
      "precision": "f32",
      "cpu_offload": false,
      "seed": null,
      "vae_tiling": false,
      "clip_on_cpu": false,
      "enabled": true
    }
  ],
  "audio_models": [
    {
      "id": "openai/whisper-1",
      "backend": "nvidia",
      "context_ms": 0,
      "offload": null,
      "vulkan_device": 0,
      "enabled": true
    }
  ],
  "vision_models": [
    {
      "id": "llava-1.5",
      "backend": "nvidia",
      "context_size": 0,
      "offload": null,
      "n_gpu_layers": -1,
      "enabled": true
    }
  ],
  "tts_models": [
    {
      "id": "kokoro",
      "backend": "nvidia",
      "voice": "af",
      "speed": 1.0,
      "enabled": true
    }
  ],
  "gguf_models": [
    {
      "id": "llama-2-7b.Q4_K_M.gguf",
      "backend": "vulkan",
      "context_size": 2048,
      "n_gpu_layers": 35,
      "vulkan_device": 0,
      "vulkan_single_gpu": false,
      "enabled": true
    }
  ],
  "loaded": [
    "microsoft/DialoGPT-medium"
  ],
  "preload": [
    "stable-diffusion-xl-base-1.0"
  ],
  "unloaded": [],
  "aliases": {
    "default": "microsoft/DialoGPT-medium",
    "code": "microsoft/DialoGPT-medium",
    "sdxl": "stable-diffusion-xl-base-1.0"
  }
}
```

### `auth.json` - User Accounts & Tokens
```json
{
  "users": [
    {
      "id": 1,
      "username": "admin",
      "password_hash": "$argon2id$...",
      "role": "admin",
      "created_at": "2026-05-03T00:00:00Z"
    }
  ],
  "tokens": [
    {
      "id": 1,
      "name": "OpenAI Compatible",
      "token": "sk-coderai-...",
      "provider": "openai",
      "created_at": "2026-05-03T00:00:00Z",
      "last_used": null
    }
  ],
  "sessions": []
}
```

---

## 3. Web Administration Dashboard

### Layout & Theme
- **Dark theme**: #0d1117 background, #161b22 cards, #21262d borders
- **Accent colors**: #58a6ff (blue), #3fb950 (green), #f85149 (red)
- **Modern fonts**: system-ui, -apple-system, Segoe UI
- **Responsive**: works on desktop and tablet

### Authentication
- Login page at `/login`
- Session-based cookies with CSRF protection
- Default credentials: `admin` / `admin` (forced change on first login)
- Password hashing with Argon2
- Sessions stored in `auth.json` (in-memory hot cache, persisted to disk)

### Pages

#### 1. **Overview Dashboard** (`/admin`)
- System status: uptime, backend type (NVIDIA/Vulkan/OpenCL), GPU info
- Active models: currently loaded, preload queue, memory usage (VRAM/RAM)
- Request stats: total, active, queued
- Quick actions: restart server, clear cache
- Line charts for request volume and latency

#### 2. **Models** (`/admin/models`)
- **Sub-tabs**:
  - **Local Models**: List all downloaded GGUF and HuggingFace models, size, format, status
  - **Download**: Search HuggingFace with filters (model type, size, license, language)
  - **Configuration**: Set loaded models, preload models, backend options per model
  - **Model Details**: Click a model to see specs, performance, edit context size, GPU layers

#### 3. **API Tokens** (`/admin/tokens`)
- List all tokens with name, provider, last used
- Generate new token (random 32-char hex, prefixed `sk-coderai-`)
- Revoke/delete tokens
- Copy token to clipboard (one-time reveal)

#### 4. **Users** (`/admin/users`)
- Admin can change own password
- CRUD for other users (username, password, role)
- Role-based: `admin`, `user`, `readonly`

#### 5. **Chat Interface** (`/chat`)
- OpenAI-compatible chat UI
- Model selector dropdown (all available models)
- Streaming responses
- File attachments (images, documents)
- Export conversation

### Routes & Middleware
- Static files: `/static/` (CSS, JS, images)
- Admin routes: `/admin/*` (require admin role)
- Auth routes: `/login`, `/logout`, `/auth/check`
- API routes (FastAPI): `/v1/*` (require bearer token or session auth)
- Web UI routes: Jinja2 templates for admin and chat

---

## 4. Model Management & Loading Strategy

### Model Types & Backend Mapping
| Model Type | Backends | Format | Per-Model Config Fields |
|------------|----------|--------|------------------------|
| Text LLM | NVIDIA (Transformers), Vulkan (llama-cpp) | HF safetensors / GGUF | backend, context_size, n_gpu_layers, load_in_4bit, load_in_8bit, flash_attn, offload_strategy, manual_ram_gb, max_gpu_percent, no_ram |
| Image Generation | NVIDIA (Diffusers), Vulkan (sd.cpp) | HF Diffusers / GGUF-SD | backend, llm_path, vae_path, sample_method, steps, width, height, cfg_scale, precision, cpu_offload, seed, vae_tiling, clip_on_cpu |
| Audio Transcription | NVIDIA (Transformers), Vulkan (whisper.cpp) | HF / GGUF | backend, context_ms, offload, vulkan_device |
| TTS | NVIDIA/Kokoro, Vulkan/kokoro | Kokoro models | backend, voice, speed |
| Vision | NVIDIA (LLaVA), Vulkan (llava.cpp) | HF / GGUF | backend, context_size, offload, n_gpu_layers |

**Key Design Principle**: Each model entry in `models.json` contains ALL configuration specific to that model. This allows:
- Multiple text models with different quantization settings (one 4-bit, one 8-bit)
- Multiple image models with different resolutions (512x512 for speed, 1024x1024 for quality)
- Multiple GGUF models with different GPU layer counts (35 layers for one, all layers for another)
- Same model with different backends (e.g., GGUF on Vulkan for one instance, HF on NVIDIA for another)

### Loading Modes
- **ondemand** (default): Only one model resident in VRAM at a time. Unload on switch.
- **loadall**: All models try to load into VRAM, OOM → CPU RAM offload.
- **loadswap**: First model in VRAM, others in CPU RAM. Swap on demand.

### Pre-load vs Loaded Status
- **Loaded**: Model actively in VRAM (or CPU RAM for loadswap)
- **Preload**: Model configured to be loaded at startup (into VRAM or CPU RAM depending on mode)
- **Unloaded**: Model not loaded; will be loaded on first request if available

### Request Queue & Smart Reordering
1. Request arrives for model X
2. If model X already in VRAM → serve immediately
3. If model X in CPU RAM → move to VRAM (evict current if needed)
4. If model X unloaded → load from disk
5. **Smart reorder**: Queue grouped by model state:
   - Requests for currently loaded models served first (preserve order within group)
   - Then requests for CPU RAM resident models (FIFO)
   - Finally requests for unloaded models (FIFO)
6. **Starvation prevention**: If a model hasn't been served in N requests, boost its priority

### Model Lifecycle
```
Startup:
  └─> Load models in "loaded" list (respecting load_mode)
  └─> Pre-load "preload" models (into CPU RAM if loadswap)
  
Runtime:
  └─> On API request: check queue → load/swap if needed → serve request
  └─> Queue management: group by model availability, preserve FIFO within groups
  └─> Periodic cleanup: keep only "loaded" count of models in VRAM
```

---

## 5. Database & Persistence

All data persisted to JSON files in config directory:

| File | Purpose |
|------|---------|
| `config.json` | Server and backend settings |
| `models.json` | Model registry, aliases, per-model config |
| `auth.json` | Users, tokens, active sessions |
| `cache.db` (optional) | Model download cache metadata (existing system) |

---

## 6. API Changes

### Token-Based Authentication
All API endpoints require a bearer token:
```
Authorization: Bearer sk-coderai-<32hex>
```
Tokens validated against `auth.json` tokens list.

### New Admin API Endpoints (FastAPI)
- `GET /admin/api/models` - list all models
- `POST /admin/api/models/download` - download from HuggingFace
- `POST /admin/api/models/remove` - delete local model
- `POST /admin/api/models/configure` - update model settings
- `GET /admin/api/tokens` - list tokens
- `POST /admin/api/tokens` - create token
- `DELETE /admin/api/tokens/{id}` - revoke token
- `GET /admin/api/users` - list users
- `POST /admin/api/users` - create user
- `PUT /admin/api/users/{id}` - update user
- `DELETE /admin/api/users/{id}` - delete user
- `POST /admin/api/system/reload` - reload config without restart
- `GET /admin/api/system/status` - system health

### WebSocket for Real-time Updates
- `/ws/admin` - admin dashboard live updates (requests, model status, VRAM)
- `/ws/chat` - chat streaming (SSE compatible)

---

## 7. Security Considerations

- Session cookies: `HttpOnly`, `Secure` (if HTTPS), `SameSite=strict`
- CSRF tokens for all POST/PUT/DELETE admin forms
- Passwords: Argon2id with salt
- Token generation: cryptographically secure random (32+ bytes)
- Rate limiting: admin endpoints (10 req/s), API (100 req/s per token)
- Input validation: model IDs, file paths sanitized
- File serving: restrict to config directory, no path traversal

---

## 8. Implementation Phases

### Phase 1: Configuration Foundation
1. Refactor `cli.py` → only `--debug` and `--config`
2. Create `ConfigManager` class (load/save/validate JSON)
3. Migrate all CLI defaults to `config.json`
4. Auto-create default configs on first run
5. Update `main.py` to read from config

### Phase 2: Admin Dashboard (FastAPI + Jinja2)
1. Create `admin/` package structure:
   - `admin/routes.py` - admin page routes
   - `admin/models.py` - model management logic
   - `admin/users.py` - user/API token logic
   - `admin/dashboard.py` - overview stats
   - `admin/templates/` - Jinja2 templates
   - `admin/static/` - CSS, JS, images
2. Implement authentication middleware
3. Build login page + session management
4. Build overview page with stats
5. Build models page (list, card grid)

### Phase 3: Models CRUD & Search
1. Integrate `codai/models/cache.py` for download/list
2. Build HuggingFace search API integration
3. Create download/remove model forms
4. Model configuration form (backend, context, GPU layers, quantization)
5. Implement model aliases system
6. Model status polling (WebSocket)

### Phase 4: Users & Tokens
1. User CRUD with Argon2 password hashing
2. Token generation (random secure, `sk-coderai-*` prefix)
3. Token usage tracking (last_used timestamp)
4. Session management (store in auth.json)
5. First-run setup wizard (force password change)

### Phase 5: Chat Interface
1. Chat page template (similar to OpenAI ChatGPT UI)
2. Model selector dropdown
3. Chat history (localStorage)
4. Streaming response handling (SSE)
5. Export/conversation management

### Phase 6: Model Loading & Queue
1. Refactor `MultiModelManager` to respect config (loaded/preload/unloaded)
2. Implement smart request queue with same-model clustering
3. WebSocket updates for model status
4. Graceful degradation (fallback models)
5. Cache management (auto-clean old models if disk full)

### Phase 7: Polish & Testing
1. Dark theme CSS polish
2. Error pages and handling
3. Responsive design
4. Accessibility (ARIA labels, keyboard navigation)
5. Integration tests for API endpoints
6. Load testing with multiple models

---

## 9. Web Interface Pages (Jinja2 Templates)

### Base Layout
- Dark sidebar navigation
- Top bar: server status, user menu, logout
- Main content area (responsive)

### Login Page
```
+-------------------------------------------+
|  CoderAI Admin                            |
|  [Logo]                                   |
|                                           |
|  Username: [________]                     |
|  Password: [________]                     |
|                                           |
|  [Login]                                  |
|                                           |
|  Default: admin / admin                   |
+-------------------------------------------+
```

### Overview Dashboard
```
+---------------------------------------------------+
| Models | Tokens | Users | Chat   [Reload] [Logout]|
+---------------------------------------------------+
| System Status          | Active Models            |
| - Backend: NVIDIA      | - phi-3 (VRAM)           |
| - GPU: RTX 4090 24GB  | - Llama-2 (CPU RAM)      |
| - Uptime: 3d 12h      | [Manage Models]          |
|                       |                          |
| Request Stats         | VRAM Usage               |
| - Total: 12,453       | [██████████░░░░] 68%     |
| - Queued: 3           |                          |
| - Last hour: 234      | System Health: OK        |
+---------------------------------------------------+
| Recent Activity (table)                           |
+---------------------------------------------------+
```

### Models Page
```
+---------------------------------------------------+
| [Local Models] [Download] [Config] [Search]      |
+---------------------------------------------------+
| Local Models:                                     |
|  [ ] phi-3-mini.q4.gguf  3.2GB   VRAM  [Load]    |
|  [x] Llama-2-7B.Q4_K_M.gguf  4.1GB   CPU  [Load] |
|  [ ] mistral-7b.gguf     4.5GB   Cached [Load]  |
|                                                   |
| Download from HuggingFace:                        |
|  Search: [_____________] [Filters▼] [Search]     |
|  Results:                                         |
|    - model1 (4.2GB, NVIDIA, MIT) [Download]      |
|    - model2 (3.8GB, Vulkan, Apache) [Download]   |
+---------------------------------------------------+
```

### Chat Interface
```
+---------------------------------------------------+
|  Models: [phi-3-mini ▼]  New Chat  History       |
+---------------------------------------------------+
| Chat:                                             |
|  User: Explain transformers                      |
|  AI:  [streaming response...]                    |
|                                                   |
|  [Input...]  [Send] [Attach]                     |
+---------------------------------------------------+
```

---

## 10. Data Flow

### Startup Sequence
```
1. main.py: parse --debug, --config
2. ConfigManager.load() → loads config.json, models.json, auth.json
3. Auto-create defaults if missing
4. Initialize ModelManager with settings from config
5. Load models listed in "loaded" and "preload" (respecting load_mode)
6. Start FastAPI server with:
   - Static file serving /templates
   - Admin routes (with session auth)
   - API routes (with token auth)
   - WebSocket routes
7. Print startup info (backends, loaded models, URL)
```

### Request Handling
```
1. Request arrives at /v1/chat/completions
2. Auth middleware: check Bearer token or session
3. Extract model from request body
4. MultiModelManager.request_model(model):
   - Check if model allowed in config
   - Check if already loaded in VRAM → return
   - Check if in CPU RAM → move to VRAM (evict if needed)
   - If unloaded → load from disk
   - Apply smart queue reordering
5. Pass to backend for inference
6. Stream/return response
```

### Admin Dashboard
```
1. User visits /admin → redirect to /login if not authenticated
2. POST /login → validate credentials → set session cookie
3. SPA-style navigation via sidebar (full page reloads, no JS framework)
4. Each admin page fetches data via FastAPI endpoints (JSON)
5. Forms POST to endpoints, redirect back with flash messages
6. WebSocket updates push live stats to dashboard
```

---

## 11. File Structure After Refactor

```
coderai/
├── codai/
│   ├── main.py              # Entry point (simplified)
│   ├── cli.py               # Only --debug, --config parsing
│   ├── config.py            # NEW: ConfigManager class
│   ├── api/
│   │   ├── app.py           # FastAPI app + routes
│   │   ├── state.py         # Global state (reduced)
│   │   ├── text.py
│   │   ├── images.py
│   │   ├── transcriptions.py
│   │   └── tts.py
│   ├── models/
│   │   ├── manager.py       # MultiModelManager (updated)
│   │   ├── cache.py         # Model download/caching
│   │   ├── parser.py
│   │   └── backends/
│   │       ├── base.py
│   │       ├── nvidia.py
│   │       └── vulkan.py
│   ├── admin/               # NEW: Admin dashboard
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── auth.py          # Authentication, sessions, passwords
│   │   ├── models.py        # Model CRUD, search, download
│   │   ├── tokens.py        # API token management
│   │   ├── users.py         # User management
│   │   ├── dashboard.py     # Overview stats
│   │   ├── templates/
│   │   │   ├── base.html
│   │   │   ├── login.html
│   │   │   ├── dashboard.html
│   │   │   ├── models.html
│   │   │   ├── tokens.html
│   │   │   ├── users.html
│   │   │   └── chat.html
│   │   └── static/
│   │       ├── style.css    # Dark theme
│   │       └── app.js
│   └── pydantic/
│       ├── textrequest.py
│       ├── imagerequest.py
│       └── transcriptionrequest.py
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-05-03-coderai-config-admin-dashboard-design.md  ← THIS FILE
├── requirements.txt
├── README.md (updated)
└── AGENTS.md (updated)
```

---

## 12. Benefits

- **Simplified CLI**: Only 2 flags to remember
- **Centralized config**: All settings in one place, version-controllable
- **Visual management**: No need to edit CLI flags or restart manually
- **User management**: Multiple users with roles and tokens
- **Model discovery**: Built-in HuggingFace search
- **Runtime control**: Change settings via dashboard, reload without restart
- **History & monitoring**: See requests, errors, usage stats
- **Backup/restore**: Config files are portable

---

## 13. Backwards Compatibility

- Old CLI command-line will fail with helpful message
- Migration script can convert existing args → config file
- Existing model cache locations preserved
- API endpoints remain compatible (only auth added)

---

## 14. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Significant rewrite, regression bugs | Comprehensive testing, phased rollout |
| Existing users lose configs | Provide migration tool, document manual migration |
| Security vulnerabilities (auth, tokens) | Use proven libraries (passlib, secrets), security audit |
| Web UI becomes maintenance burden | Keep it simple (Jinja2, no heavy JS framework) |
| Model loading complexity breaks | Maintain existing `MultiModelManager` logic, wrap in config layer |

---

## 15. Open Questions & Decisions Needed

1. **Should model search show ONLY GGUF models, or all HF models?**  
   → Recommend: filter by GGUF for Vulkan, all for NVIDIA with format indicator

2. **Should admin be able to delete models from disk, or just unregister?**  
   → Recommend: delete from cache directory with confirmation

3. **Should chat interface support advanced parameters (temp, top_p, etc)?**  
   → Recommend: collapsible advanced panel in chat UI

4. **Should config support environment variable substitution?** (e.g., `${HOME}`)  
   → Recommend: yes, for paths

5. **Should there be a "safe mode" if config.json is corrupt?**  
   → Recommend: fall back to hardcoded minimal defaults, rebuild default config

6. **Should we keep command-line flag to bypass config entirely for debugging?**  
   → Recommend: `--force-cli` flag (hidden/undocumented) for dev use
