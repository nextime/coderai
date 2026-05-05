# CoderAI

An OpenAI-compatible API server with web administration dashboard, supporting multiple GPU backends: NVIDIA (CUDA), AMD (Vulkan), and Intel (Vulkan). Configuration-driven architecture with per-model settings and multi-modal support (text, image, audio, TTS).

## Features

### Core Capabilities
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's API endpoints
- **Web Admin Dashboard**: Modern UI for model management, user authentication, and API tokens
- **Configuration-Based**: JSON config files for all settings - no complex CLI arguments
- **Multi-Modal Support**: Text generation, image generation, audio transcription, text-to-speech
- **Per-Model Configuration**: Individual settings for each model (GPU layers, quantization, context size)
- **On-Demand Loading**: Models load automatically when requested, unload when idle

### GPU Backend Support
- **NVIDIA (CUDA)**: PyTorch + Transformers for HuggingFace models
- **AMD GPUs**: llama-cpp-python + Vulkan for GGUF models
- **Intel GPUs**: iGPU/Arc support via Vulkan
- **Auto-Detection**: Automatically selects best available backend
- **Multi-GPU**: Automatic distribution across multiple devices

### Advanced Features
- **Memory Management**: Smart VRAM → RAM → Disk offloading (NVIDIA)
- **Quantization**: 4-bit/8-bit via bitsandbytes (NVIDIA) or GGUF quantization (Vulkan)
- **Flash Attention 2**: Optional faster inference for supported NVIDIA GPUs
- **Streaming**: Server-sent events for real-time token generation
- **Tool Calling**: Function calling and tool use support
- **Authentication**: Session-based auth with API token support

## Installation

### Prerequisites

- Python 3.8+
- For NVIDIA GPUs: CUDA toolkit (11.8+ recommended)
- For AMD/Intel GPUs (Vulkan): Vulkan drivers and SDK
- For CPU-only: No additional requirements

**Note**: The Vulkan backend works with:
- AMD GPUs (RX 400 series and newer) - **Recommended**
- Intel integrated GPUs (HD 600 series and newer) and Intel Arc GPUs
- NVIDIA GPUs (GTX 900 series and newer) - *CUDA backend preferred*

Any GPU with Vulkan 1.2+ driver support should work with the Vulkan backend.

### Quick Install with Build Script

The easiest way to install is using the provided build script:

```bash
# Clone the repository
git clone git@git.nexlab.net:nexlab/coderai.git
cd coderai

# Install all backends (recommended)
./build.sh all

# Or install specific backend:
./build.sh nvidia   # NVIDIA GPUs only
./build.sh vulkan   # AMD/Intel GPUs only
```

**Note**: The `all` option installs support for all backends, allowing you to switch between them via configuration. The `vulkan` option works for both AMD and Intel GPUs.

The build script will:
- Create a virtual environment
- Install the appropriate dependencies for your GPU
- Set up the correct backend(s)

### Manual Installation

If you prefer manual installation:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# For NVIDIA GPUs
pip install torch torchvision torchaudio
pip install -r requirements-nvidia.txt

# For AMD GPUs with Vulkan
CMAKE_ARGS="-DGGML_VULKAN=ON" pip install llama-cpp-python --no-cache-dir
pip install -r requirements-vulkan.txt
```

### Platform-Specific Requirements

#### NVIDIA (CUDA)

Requires:
- NVIDIA GPU with CUDA support
- CUDA toolkit (11.8+ or 12.1+)
- PyTorch with CUDA

Models: HuggingFace format (safetensors/pytorch)

#### AMD and Intel (Vulkan)

Requires:
- GPU with Vulkan 1.2+ support:
  - AMD: RX 400 series and newer (recommended)
  - Intel: HD 600 series integrated graphics or newer, Intel Arc GPUs
  - NVIDIA: GTX 900 series and newer (but CUDA backend preferred)
- Vulkan drivers and SDK

**Install Vulkan drivers and tools:**
```bash
# Debian/Ubuntu
sudo apt install libvulkan-dev vulkan-tools mesa-vulkan-drivers glslc glslang-tools glslang-dev

# Fedora
sudo dnf install vulkan-loader-devel vulkan-tools mesa-vulkan-drivers glslang

# Arch Linux
sudo pacman -S vulkan-headers vulkan-icd-loader vulkan-radeon glslang
```

**Note:** The shader compiler `glslc` is required to build llama-cpp-python with Vulkan support. On Debian/Ubuntu, it's provided by the `glslc` package. If `glslc` is not found after installing, try:

```bash
# Check if glslc exists somewhere
find /usr -name "glslc" 2>/dev/null

# If found in a non-standard location, add to PATH
export PATH=$PATH:/usr/lib/shaderc/bin

# Or create a symlink if glslangValidator exists
sudo ln -s $(which glslangValidator) /usr/local/bin/glslc
```

Models: GGUF format (from HuggingFace or local files)

**Note**: The Vulkan backend uses llama-cpp-python with GGUF models, which provides excellent performance on AMD and Intel GPUs without requiring vendor-specific SDKs (ROCm/OneAPI).

### Optional Dependencies

#### bitsandbytes (Quantization)

For 4-bit and 8-bit quantization support (reduces VRAM requirements):

```bash
# CUDA
pip install "bitsandbytes>=0.41.0"

# ROCm support may require building from source
# See: https://github.com/TimDettmers/bitsandbytes
```

#### Flash Attention 2

For significantly faster inference on supported GPUs (requires specific CUDA/ROCm versions):

```bash
# Requires CUDA 11.6+ or ROCm 5.4+
pip install flash-attn --no-build-isolation
```

**Note**: Flash Attention 2 requires:
- CUDA 11.6+ or ROCm 5.4+
- Linux OS (Windows support is experimental)
- Specific GPU architectures (Ampere, Ada Lovelace, Hopper for NVIDIA)

## Usage

### Quick Start

```bash
# Activate the virtual environment
source venv_all/bin/activate  # or venv/bin/activate

# Start the server (uses default config at ~/.coderai/)
python coderai

# Or specify a custom config directory
python coderai --config /path/to/config

# Enable debug mode for troubleshooting
python coderai --debug
```

The server will start on `http://0.0.0.0:8000` by default.

### Access Points

- **Admin Dashboard**: http://localhost:8000/admin
- **Chat Interface**: http://localhost:8000/chat
- **API Endpoints**: http://localhost:8000/v1/*
- **API Documentation**: http://localhost:8000/docs

### First Login

Default credentials (you'll be prompted to change the password):
- **Username**: `admin`
- **Password**: `admin`

### Configuration Files

CoderAI uses JSON configuration files stored in `~/.coderai/` (or custom directory via `--config`):

```
~/.coderai/
├── config.json       # Server, backend, and global settings
├── models.json       # Model registry and per-model configurations
├── auth.json         # Users, API tokens, and sessions
└── secret_key        # Session signing key (auto-generated)
```

These files are automatically created with sensible defaults on first run.

### Command-Line Options

```
usage: coderai [-h] [--config CONFIG] [--debug] [--dump]
               [--list-cached-models] [--remove-all-models]
               [--remove-model REMOVE_MODEL] [--download-model DOWNLOAD_MODEL]
               [--download-file-pattern DOWNLOAD_FILE_PATTERN]
               [--vulkan-list-devices]

OpenAI-compatible API server supporting NVIDIA (CUDA) and Vulkan backends

options:
  -h, --help            show this help message and exit
  --config CONFIG       Configuration directory (default: ~/.coderai/)
  --debug               Enable debug mode - dumps full request/response to stdout
  --dump                Dump model output: raw output, parsed output, and debug info
  --list-cached-models  List all cached models in the model cache directory
  --remove-all-models   Remove all cached models from the model cache directory
  --remove-model NAME   Remove a specific cached model by name or hash
  --download-model ID   Download a model to cache (URL or HuggingFace model ID)
  --download-file-pattern PATTERN
                        File pattern for HuggingFace downloads (e.g., .gguf, .safetensors)
  --vulkan-list-devices List available Vulkan GPU devices and exit
```

## API Documentation

The API is compatible with OpenAI's REST API. Interactive documentation is available at `http://localhost:8000/docs` when the server is running.

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completions (ChatGPT-style) |
| `POST /v1/completions` | Text completions (GPT-style) |

### Example curl Commands

#### List Models

```bash
curl http://localhost:8000/v1/models
```

#### Chat Completion (Non-Streaming)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-medium",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }'
```

#### Chat Completion (Streaming)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-medium",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true,
    "max_tokens": 200
  }'
```

#### Text Completion

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-medium",
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.8
  }'
```

#### Chat Completion with Tools

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-medium",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

## Configuration

### Configuration Files

All settings are managed through JSON files in the configuration directory (`~/.coderai/` by default):

#### config.json - Server and Backend Settings

```json
{
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
    "default_load_mode": "ondemand",
    "hf_cache_dir": null,
    "gguf_cache_dir": null
  },
  "offload": {
    "directory": "./offload",
    "strategy": "auto",
    "max_gpu_percent": null,
    "no_ram": false,
    "load_in_4bit": false,
    "load_in_8bit": false,
    "manual_ram_gb": null,
    "flash_attention": false
  },
  "vulkan": {
    "n_gpu_layers": -1,
    "n_ctx": 2048,
    "device_id": 0,
    "single_gpu": false
  },
  "image": {
    "steps": 4,
    "width": 512,
    "height": 512,
    "cfg_scale": 1.0,
    "precision": "f32",
    "cpu_offload": false
  },
  "whisper": {
    "server_path": null,
    "server_port": 8744
  }
}
```

#### models.json - Model Registry

```json
{
  "text_models": [
    {
      "id": "microsoft/DialoGPT-medium",
      "backend": "nvidia",
      "context_size": 2048,
      "n_gpu_layers": -1,
      "load_in_4bit": false,
      "load_in_8bit": false,
      "flash_attention": false,
      "enabled": true
    },
    {
      "id": "phi-3-mini-4k-instruct-q4_k_m.gguf",
      "backend": "vulkan",
      "context_size": 4096,
      "n_gpu_layers": -1,
      "enabled": true
    }
  ],
  "image_models": [
    {
      "id": "stable-diffusion-xl-base-1.0",
      "backend": "nvidia",
      "steps": 4,
      "width": 512,
      "height": 512,
      "cfg_scale": 1.0,
      "enabled": true
    }
  ],
  "audio_models": [],
  "vision_models": [],
  "tts_models": [],
  "loaded": [],
  "preload": [],
  "aliases": {
    "default": "microsoft/DialoGPT-medium"
  }
}
```

#### auth.json - Users and API Tokens

```json
{
  "users": [
    {
      "id": "admin",
      "username": "admin",
      "password_hash": "$argon2id$...",
      "role": "admin",
      "created_at": "2026-05-05T00:00:00Z"
    }
  ],
  "tokens": [
    {
      "id": "tok_abc123",
      "token": "sk-coderai-abc123...",
      "name": "Production API",
      "created_at": "2026-05-05T00:00:00Z",
      "last_used": null
    }
  ],
  "sessions": {}
}
```

### Managing Configuration

#### Via Web Dashboard

The easiest way to manage configuration is through the web dashboard at `http://localhost:8000/admin`:

- **Models**: Add, remove, enable/disable models; configure per-model settings
- **Users**: Create users, change passwords, manage roles
- **Tokens**: Generate API tokens for programmatic access
- **Settings**: Adjust server, backend, and global settings

#### Via Configuration Files

You can also edit the JSON files directly. Changes take effect after restarting the server or using the reload endpoint:

```bash
curl -X POST http://localhost:8000/admin/api/system/reload
```

### Per-Model Configuration

Each model can have its own settings that override global defaults:

**Text Models (NVIDIA backend):**
- `backend`: "nvidia" or "vulkan"
- `context_size`: Context window size
- `n_gpu_layers`: Number of layers on GPU (-1 = all)
- `load_in_4bit`: Enable 4-bit quantization
- `load_in_8bit`: Enable 8-bit quantization
- `flash_attention`: Enable Flash Attention 2

**Text Models (Vulkan backend):**
- `backend`: "vulkan"
- `context_size`: Context window size
- `n_gpu_layers`: Number of layers on GPU (-1 = all)

**Image Models:**
- `backend`: "nvidia" or "vulkan"
- `steps`: Number of diffusion steps
- `width`: Image width
- `height`: Image height
- `cfg_scale`: Classifier-free guidance scale
- `precision`: "f32" or "f16"

### Backend Selection

Backends can be configured globally in `config.json` or per-model in `models.json`:

- **`auto`**: Automatically detect and use best available backend
- **`nvidia`**: Use CUDA backend (PyTorch + Transformers)
- **`vulkan`**: Use Vulkan backend (llama-cpp-python)

### Model Loading Modes

Configure in `config.json` under `models.default_load_mode`:

- **`ondemand`** (default): Load models when first requested, unload when idle
- **`preload`**: Load models listed in `models.json` → `preload` array at startup
- **`lazy`**: Never preload, always load on-demand

## Backend-Specific Setup

### NVIDIA (CUDA)

```bash
# Using build script
./build.sh nvidia

# Or manually install CUDA-enabled PyTorch
pip install "torch>=2.0.0" "torchvision>=0.15.0" "torchaudio>=2.0.0"
pip install -r requirements-nvidia.txt
```

**Configuration in models.json:**
```json
{
  "text_models": [
    {
      "id": "meta-llama/Llama-2-7b-chat-hf",
      "backend": "nvidia",
      "context_size": 4096,
      "n_gpu_layers": -1,
      "load_in_4bit": false,
      "load_in_8bit": false,
      "flash_attention": false,
      "enabled": true
    }
  ]
}
```

### AMD and Intel (Vulkan)

```bash
# Install Vulkan drivers first
# Debian/Ubuntu (AMD and Intel):
sudo apt install libvulkan-dev vulkan-tools mesa-vulkan-drivers intel-media-va-driver

# Fedora:
sudo dnf install vulkan-loader-devel vulkan-tools mesa-vulkan-drivers intel-gpu-tools

# Using build script
./build.sh vulkan

# List available Vulkan GPU devices
python coderai --vulkan-list-devices
```

**Vulkan Backend Notes:**
- Uses GGUF format models (much smaller than full HuggingFace models)
- Q4_K_M quantization recommended for 4GB+ VRAM GPUs
- Q5_K_M or Q6_K for higher quality
- Works on:
  - AMD RX 400 series and newer (**recommended**)
  - Intel integrated graphics (HD 600 series+) and Intel Arc GPUs
  - NVIDIA GTX 900 series and newer (but CUDA backend is preferred)
- Any GPU with Vulkan 1.2+ driver support should work
- **Update llama-cpp-python** for newer model support: `pip install --upgrade llama-cpp-python --no-cache-dir`

**Intel GPU Specific Notes:**
- Intel integrated GPUs have limited VRAM (shared with system RAM), so use smaller models
- Recommended for Intel iGPUs: `Q4_K_M` quantized models under 2GB file size
- Intel Arc GPUs work well with the same settings as AMD GPUs

**Configuration in models.json:**
```json
{
  "text_models": [
    {
      "id": "phi-3-mini-4k-instruct-q4_k_m.gguf",
      "backend": "vulkan",
      "context_size": 4096,
      "n_gpu_layers": -1,
      "enabled": true
    }
  ]
}
```

**Vulkan Configuration in config.json:**
```json
{
  "vulkan": {
    "n_gpu_layers": -1,
    "n_ctx": 2048,
    "device_id": 0,
    "single_gpu": false
  }
}
```

### CPU-Only

While not recommended for performance, you can run on CPU:

```bash
# NVIDIA backend on CPU
pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-nvidia.txt

# Or Vulkan backend on CPU (llama-cpp supports CPU fallback)
CMAKE_ARGS="-DGGML_VULKAN=OFF" pip install llama-cpp-python
```

Configure in `config.json`:
```json
{
  "backend": {
    "type": "nvidia"
  },
  "vulkan": {
    "n_gpu_layers": 0
  }
}
```

### ROCm Alternative (deprecated)

While the Vulkan backend is now recommended for AMD GPUs, ROCm support is still available through the NVIDIA backend if you have ROCm-enabled PyTorch installed.

### Low VRAM Configuration

For GPUs with limited VRAM (4-8GB), configure in `config.json` or per-model in `models.json`:

**Global configuration (config.json):**
```json
{
  "offload": {
    "load_in_4bit": true,
    "directory": "/path/to/fast/storage"
  }
}
```

**Per-model configuration (models.json):**
```json
{
  "text_models": [
    {
      "id": "meta-llama/Llama-2-7b-chat-hf",
      "backend": "nvidia",
      "load_in_4bit": true,
      "enabled": true
    }
  ]
}
```

### Using Vulkan with Multiple GPUs (NVIDIA + AMD)

If your system has both NVIDIA and AMD GPUs, llama.cpp's Vulkan backend will automatically distribute layers across all visible GPUs for performance. To force Vulkan to use **only** the AMD GPU and prevent VRAM allocation on the NVIDIA GPU, configure in `config.json`:

**Configuration in config.json:**
```json
{
  "vulkan": {
    "device_id": 1,
    "single_gpu": true
  }
}
```

**Alternative: Environment variables**
```bash
# List available Vulkan devices first
python coderai --vulkan-list-devices

# Then use VK_DEVICE_SELECT_DEVICE to force a specific device
# For example, if device 1 is your AMD GPU:
VK_DEVICE_SELECT_DEVICE=1 python coderai

# Or hide NVIDIA GPU from CUDA (prevents any CUDA usage)
CUDA_VISIBLE_DEVICES="" python coderai
```

**Understanding the Issue:**
When you have multiple Vulkan-compatible GPUs, llama.cpp automatically distributes model layers across them (shown in logs as "layer X assigned to device VulkanY"). The `single_gpu: true` setting prevents this by using the `tensor_split` parameter with a value of `[0.0, 1.0]` (or similar depending on device count), which tells llama.cpp to put 0% of layers on some GPUs and 100% on the selected GPU.

**Notes:**
- The `device_id` setting maps to `main_gpu` in llama-cpp-python
- The `single_gpu` flag builds a `tensor_split` array to force single GPU usage
- Vulkan enumerates all GPUs in your system, so device IDs may differ from CUDA device IDs
- The `vulkaninfo` command shows all GPUs visible to Vulkan

### Multi-GPU Setup

Multiple GPUs are automatically detected and utilized. The model will be distributed across available devices based on memory availability.

```bash
# Set visible GPUs (optional)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run - model will be distributed across all visible GPUs
python coderai
```

## Model Recommendations

### NVIDIA Backend (HuggingFace Models)

#### Small Models (For Testing)

- `microsoft/DialoGPT-medium` (~345M parameters)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~1.1B parameters)
- `facebook/blenderbot-400M-distill` (~400M parameters)

#### Medium Models (4-8GB VRAM with 4-bit)

- `meta-llama/Llama-2-7b-chat-hf` (~7B parameters)
- `mistralai/Mistral-7B-Instruct-v0.2` (~7B parameters)
- `HuggingFaceH4/zephyr-7b-beta` (~7B parameters)

#### Large Models (Multiple GPUs or High VRAM)

- `meta-llama/Llama-2-13b-chat-hf` (~13B parameters)
- `meta-llama/Llama-2-70b-chat-hf` (~70B parameters) - requires multiple GPUs or disk offload
- `bigscience/bloom-7b1` (~7B parameters)

### Vulkan Backend (GGUF Models)

#### Small Models (2-4GB VRAM)

- `TheBloke/phi-2-GGUF` - phi-2.Q4_K_M.gguf (~1.6B parameters, ~1GB file)
- `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` - tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

#### Medium Models (4-8GB VRAM)

- `TheBloke/Llama-2-7B-GGUF` - llama-2-7b.Q4_K_M.gguf (~4GB file)
- `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` - mistral-7b-instruct-v0.2.Q4_K_M.gguf
- `microsoft/Phi-3-mini-4k-instruct-gguf` - Phi-3-mini-4k-instruct-q4.gguf

#### Large Models (8GB+ VRAM)

- `TheBloke/Llama-2-13B-GGUF` - llama-2-13b.Q4_K_M.gguf (~7.5GB file)
- `TheBloke/deepseek-coder-6.7B-base-GGUF` - deepseek-coder-6.7b-base.Q4_K_M.gguf

**GGUF Quantization Guide:**
- `Q4_K_M` - Best balance of speed/quality (recommended)
- `Q5_K_M` - Higher quality, slightly slower
- `Q6_K` - Near-unquantized quality
- `Q8_0` - Maximum quality, largest size

**Download Example:**
```bash
# Using huggingface-cli
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf --local-dir ./models

# Or let coderai download automatically
python coderai --model TheBloke/Llama-2-7B-GGUF --backend vulkan
```

## Troubleshooting

### Shell Redirection Error: "No such file or directory: '0.0'"

**Problem**: Running `pip install torch>=2.0.0` fails with an error about file "0.0" or "=2.0.0" not found.

**Cause**: The shell interprets `>` as output redirection. The command creates a file named "=2.0.0" and installs an unversioned torch package.

**Solutions**:
1. **Use quotes** (recommended): `pip install "torch>=2.0.0"`
2. **Use exact versions**: `pip install torch==2.0.0`
3. **Use requirements.txt**: Add exact versions to requirements.txt and run `pip install -r requirements.txt`

### Out of Memory Errors

**Problem**: `CUDA out of memory` or system RAM exhausted

**Solutions**:
1. Use quantization: `--load-in-4bit` or `--load-in-8bit`
2. Enable disk offload: `--offload-dir /path/to/storage`
3. Use a smaller model
4. Reduce batch size in client requests

### Flash Attention Installation Fails

**Problem**: `pip install flash-attn` fails to build

**Solutions**:
1. Ensure CUDA/ROCm is properly installed
2. Install build dependencies: `pip install packaging ninja`
3. Try without build isolation: `pip install flash-attn --no-build-isolation`
4. Check GPU compatibility (Ampere, Ada Lovelace, Hopper for NVIDIA)
5. Skip Flash Attention - the server works without it

### Flash Attention: No module named 'torch' during build

**Problem**: Flash Attention build fails with `ModuleNotFoundError: No module named 'torch'` even though PyTorch is installed (e.g., PyTorch 2.9.1+rocm6.4).

**Cause**: pip uses isolated build environments by default, which prevents flash-attention from seeing the installed torch package during compilation.

**Solutions**:
1. **Use --no-build-isolation flag** (recommended):
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **For ROCm systems**, you may also need to limit parallel jobs to avoid resource exhaustion:
   ```bash
   MAX_JOBS=4 pip install flash-attn --no-build-isolation
   ```

3. **Use pre-built wheels** if available for your platform (check https://github.com/Dao-AILab/flash-attention/releases)

4. **ROCm 6.4 compatibility note**: Flash Attention may not officially support ROCm 6.4 yet (it was primarily built for ROCm 6.0). If build fails on ROCm 6.4, you can run without Flash Attention:
   ```bash
   python coderai --model meta-llama/Llama-2-7b-chat-hf
   # (omit the --flash-attn flag)
   ```

5. **Fallback**: The server works perfectly without Flash Attention - simply omit the `--flash-attn` flag when starting the server.

### bitsandbytes Not Working on ROCm

**Problem**: Quantization fails on AMD GPUs

**Solutions**:
1. bitsandbytes has limited ROCm support
2. Use disk offload instead: `--offload-dir /path/to/storage`
3. Build bitsandbytes from source with ROCm support

### Model Download Stuck or Slow

**Problem**: HuggingFace model download is slow or fails

**Solutions**:
1. Set HuggingFace cache directory: `export HF_HOME=/path/to/cache`
2. Use mirror: `export HF_ENDPOINT=https://hf-mirror.com` (for China)
3. Download model manually with `git-lfs` and use local path

### Auto-Detection Issues in Containers

**Problem**: Wrong memory detection in Docker/Podman containers

**Solutions**:
1. Specify RAM manually: `--ram 16`
2. Pass through GPU devices properly
3. For Docker: `--gpus all` flag for NVIDIA, or proper device mapping for ROCm

### API Returns 503 Errors

**Problem**: `Model not loaded` error

**Solutions**:
1. Ensure model name is correct and accessible
2. Check model requires authentication: `huggingface-cli login`
3. Verify internet connection for first-time model download

### ROCm Not Detected

**Problem**: ROCm GPU not detected, falling back to CPU

**Solutions**:
1. Verify ROCm installation: `rocminfo`
2. Check PyTorch ROCm build: `python -c "import torch; print(torch.version.hip)"`
3. Set HIP visible devices: `export HIP_VISIBLE_DEVICES=0`

### Import Errors

**Problem**: `ModuleNotFoundError` for various packages

**Solutions**:
1. Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
2. Check Python version: `python --version` (should be 3.8+)
3. Verify virtual environment is activated

### Vulkan-Specific Issues

**Problem**: "Vulkan backend not available" or llama-cpp fails to load

**Solutions**:
1. **Verify Vulkan drivers and shader compiler are installed:**
   ```bash
   # Check Vulkan installation
   vulkaninfo | grep "deviceName"
   
   # Check glslc (shader compiler) - REQUIRED for building
   glslc --version
   
   # Or install if missing
   # Debian/Ubuntu:
   sudo apt install libvulkan-dev vulkan-tools mesa-vulkan-drivers glslang-tools
   
   # Fedora:
   sudo dnf install vulkan-loader-devel vulkan-tools mesa-vulkan-drivers glslang
   ```
   
   **Note:** `glslc` is required to compile llama-cpp-python with Vulkan support. If you see "Could NOT find Vulkan (missing: glslc)", install the `glslc` package:
   ```bash
   sudo apt install glslc glslang-tools glslang-dev
   
   # If glslc still not found, check location and symlink:
   find /usr -name "glslc" 2>/dev/null
   sudo ln -s /usr/lib/shaderc/bin/glslc /usr/local/bin/glslc 2>/dev/null || sudo ln -s $(which glslangValidator) /usr/local/bin/glslc 2>/dev/null || echo "glslc not found, please install glslc package"
   ```

2. **Reinstall llama-cpp-python with Vulkan:**
   ```bash
   pip uninstall llama-cpp-python -y
   CMAKE_ARGS="-DGGML_VULKAN=ON" pip install llama-cpp-python --no-cache-dir
   ```

3. **Check GPU compatibility:**
    - **AMD**: RX 400 series and newer (best experience)
    - **Intel**: HD 600 series integrated graphics or newer, all Intel Arc GPUs
    - **NVIDIA**: GTX 900 series and newer (but CUDA backend preferred for NVIDIA)
    - Any GPU with Vulkan 1.2+ driver support should work

**Performance expectations by GPU:**
- AMD dedicated GPUs: Full performance, all layer offloading supported
- Intel Arc GPUs: Good performance, similar to AMD
- Intel integrated GPUs: Limited by shared system RAM, use smaller models (Q4_K_M under 2GB)

**Problem**: GGUF model fails to load or produces garbled output

**Solutions**:
1. **Verify model format**: Must be GGUF format, not regular HuggingFace format
   ```bash
   # Check file extension
   ls -la model.gguf  # Should end in .gguf
   ```

2. **Try different quantization**: Some GGUF files may be incompatible
   - Q4_K_M is most compatible (recommended)
   - Q5_K_M or Q6_K for higher quality
   - Avoid IQ quants if having issues

3. **Check model architecture**: Some very new models may need updated llama-cpp
   ```bash
   pip install --upgrade llama-cpp-python
   ```

**Problem**: Vulkan backend runs on CPU instead of GPU

**Solutions**:
1. **Check layer offloading**: Verify layers are being offloaded
   ```bash
   # Check GPU layers parameter (default -1 = all layers)
   python coderai --model model.gguf --backend vulkan --n-gpu-layers 35
   ```

2. **Check verbose output**: Look for Vulkan device initialization in logs
   ```bash
   # Run with verbose logging
   python coderai --model model.gguf --backend vulkan 2>&1 | grep -i vulkan
   ```

3. **Verify GPU visibility**: Check that Vulkan sees your GPU
   ```bash
   vulkaninfo | grep -A 5 "GPU0\|GPU1"
   ```

### Backend Not Detected

**Problem**: "No suitable backend found" error

**Solutions**:
1. **Check which backends are available:**
   ```bash
   python -c "import coderai; print(coderai.detect_available_backends())"
   ```

2. **For NVIDIA**: Ensure PyTorch with CUDA is installed
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **For Vulkan**: Ensure llama-cpp-python is installed with Vulkan support
   ```bash
   python -c "from llama_cpp import Llama; print('llama-cpp available')"
   ```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a merge request.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [HuggingFace Transformers](https://huggingface.co/docs/transformers/) (NVIDIA backend)
- Powered by [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) with Vulkan support (AMD/Intel backend)
- Inspired by the OpenAI API specification

---

**Note on AI.PROMPT**: This project was enhanced following instructions to add Vulkan support for AMD and Intel GPUs alongside the existing NVIDIA/CUDA support. The implementation uses llama-cpp-python for Vulkan/GGUF model support while maintaining full compatibility with the existing HuggingFace/Transformers backend for NVIDIA GPUs.
