# CoderAI

An OpenAI-compatible API server supporting both NVIDIA (CUDA) and AMD (Vulkan) GPUs. Uses HuggingFace Transformers for NVIDIA GPUs and llama-cpp-python with Vulkan for AMD GPUs.

## Features

- **Dual Backend Support**: NVIDIA (CUDA) via PyTorch + Transformers, AMD (Vulkan) via llama-cpp-python
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's API endpoints
- **Memory-Aware Model Loading**: Automatically determines optimal loading strategy based on available VRAM and RAM (NVIDIA)
- **Sequential Offloading**: Smart offload from VRAM → RAM → Disk when needed (NVIDIA)
- **Multi-GPU Support**: Automatic distribution across multiple CUDA devices (NVIDIA)
- **GPU Auto-Detection**: Automatically detects available backends
- **Quantization Support**: 4-bit and 8-bit quantization via bitsandbytes (NVIDIA) or built-in GGUF quantization (Vulkan)
- **Flash Attention 2**: Optional faster attention implementation for supported NVIDIA GPUs
- **Streaming Responses**: Server-sent events for real-time token generation
- **Tool Calling**: Support for function calling and tool use
- **Multiple Endpoints**: `/v1/chat/completions`, `/v1/completions`, and `/v1/models`

## Installation

### Prerequisites

- Python 3.8+
- For NVIDIA GPUs: CUDA toolkit (11.8+ recommended)
- For AMD GPUs (Vulkan): Vulkan drivers and SDK
- For CPU-only: No additional requirements

### Quick Install with Build Script

The easiest way to install is using the provided build script:

```bash
# Clone the repository
git clone git@git.nexlab.net:nexlab/coderai.git
cd coderai

# For NVIDIA GPUs (default)
./build.sh nvidia

# For AMD GPUs with Vulkan support
./build.sh vulkan
```

The build script will:
- Create a virtual environment
- Install the appropriate dependencies for your GPU
- Set up the correct backend

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

#### AMD (Vulkan)

Requires:
- AMD GPU with Vulkan support (RX 400 series and newer)
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

**Note**: The Vulkan backend uses llama-cpp-python with GGUF models, which provides excellent performance on AMD GPUs without requiring ROCm.

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

### Basic Usage

```bash
# Activate the virtual environment created by build.sh
source venv/bin/activate

# Run with NVIDIA backend (HuggingFace models)
python coderai --model microsoft/DialoGPT-medium --backend nvidia

# Run with Vulkan backend (GGUF models)
python coderai --model ./phi-3-mini-4k-instruct-q4_k_m.gguf --backend vulkan

# The server will start on http://0.0.0.0:8000 by default
```

### Command-Line Options

```
usage: coderai [-h] [--model MODEL] [--backend {auto,nvidia,vulkan}] [--host HOST]
               [--port PORT] [--offload-dir OFFLOAD_DIR] [--load-in-4bit]
               [--load-in-8bit] [--ram RAM] [--flash-attn] [--n-gpu-layers N]
               [--n-ctx N]

OpenAI-compatible API server supporting NVIDIA (CUDA) and Vulkan backends

options:
  -h, --help            show this help message and exit
  --model MODEL         Model name or path. For NVIDIA: HuggingFace model.
                        For Vulkan: GGUF file path or HF repo
  --backend {auto,nvidia,vulkan}
                        Backend to use: auto (detect), nvidia (CUDA), or
                        vulkan (AMD GPUs)
  --host HOST           Host to bind to (default: 0.0.0.0)
  --port PORT           Port to bind to (default: 8000)
  --offload-dir OFFLOAD_DIR
                        Directory for disk offload (NVIDIA only, default: ./offload)
  --load-in-4bit        Load model in 4-bit precision (NVIDIA only, requires bitsandbytes)
  --load-in-8bit        Load model in 8-bit precision (NVIDIA only, requires bitsandbytes)
  --ram RAM             Manually specify available RAM in GB (NVIDIA only)
  --flash-attn          Use Flash Attention 2 (NVIDIA only, requires flash-attn)
  --n-gpu-layers N      Number of layers to offload to GPU (Vulkan only,
                        default: -1 = all layers)
  --n-ctx N             Context window size (Vulkan only, default: 2048)
  --vulkan-device N     Vulkan GPU device ID to use (Vulkan only, default: 0)
  --vulkan-list-devices List available Vulkan GPU devices and exit
```

### Backend Selection

The `--backend` option controls which backend to use:

- **`auto`** (default): Automatically detects available backends, preferring NVIDIA if available
- **`nvidia`**: Use PyTorch + Transformers with CUDA (for NVIDIA GPUs)
- **`vulkan`**: Use llama-cpp-python with Vulkan (for AMD GPUs)

### Model Formats by Backend

#### NVIDIA Backend
Uses HuggingFace Transformers format:
```bash
python coderai --model microsoft/DialoGPT-medium --backend nvidia
python coderai --model meta-llama/Llama-2-7b-chat-hf --backend nvidia
```

#### Vulkan Backend
Uses GGUF format (can be local files or downloaded from HuggingFace):
```bash
# Local GGUF file
python coderai --model ./phi-3-mini-4k-instruct-q4_k_m.gguf --backend vulkan

# Download from HuggingFace (auto-selects GGUF file)
python coderai --model microsoft/Phi-3-mini-4k-instruct-gguf --backend vulkan

# Specific GGUF file from repo
python coderai --model TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf --backend vulkan
```

**Finding GGUF models:**
- Search on HuggingFace: https://huggingface.co/models?search=gguf
- Popular collections: TheBloke, unsloth, bartowski
- Recommended quantization: Q4_K_M for best speed/quality balance

### Examples

#### Run with 4-bit Quantization (Low VRAM)

```bash
python coderai --model meta-llama/Llama-2-7b-chat-hf --load-in-4bit
```

#### Run with Custom Offload Directory

```bash
python coderai --model bigscience/bloom-7b1 --offload-dir /path/to/fast/storage
```

#### Run on Specific Host/Port

```bash
python coderai --model microsoft/DialoGPT-medium --host 127.0.0.1 --port 8080
```

#### Specify Available RAM Manually

Useful for containerized environments where auto-detection may not work:

```bash
python coderai --model meta-llama/Llama-2-13b-chat-hf --ram 32
```

#### Enable Flash Attention 2

```bash
python coderai --model meta-llama/Llama-2-7b-chat-hf --flash-attn
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

## Configuration for Different Setups

### NVIDIA (CUDA)

```bash
# Using build script
./build.sh nvidia

# Or manually install CUDA-enabled PyTorch
pip install "torch>=2.0.0" "torchvision>=0.15.0" "torchaudio>=2.0.0"
pip install -r requirements-nvidia.txt

# Run with GPU acceleration
python coderai --model meta-llama/Llama-2-7b-chat-hf --backend nvidia

# Optional: Enable Flash Attention 2 for faster inference
python coderai --model meta-llama/Llama-2-7b-chat-hf --backend nvidia --flash-attn
```

### AMD (Vulkan)

```bash
# Install Vulkan drivers first
# Debian/Ubuntu:
sudo apt install libvulkan-dev vulkan-tools mesa-vulkan-drivers

# Using build script
./build.sh vulkan

# Run with GGUF model
python coderai --model ./phi-3-mini-4k-instruct-q4_k_m.gguf --backend vulkan

# Or download automatically from HuggingFace
python coderai --model TheBloke/Llama-2-7B-GGUF --backend vulkan

# Control GPU layer offloading (default: -1 = all layers)
python coderai --model model.gguf --backend vulkan --n-gpu-layers 35

# Adjust context window (default: 2048)
python coderai --model model.gguf --backend vulkan --n-ctx 4096

# Select specific GPU device (if you have multiple GPUs - e.g., NVIDIA + AMD)
python coderai --model model.gguf --backend vulkan --vulkan-device 1

# List available Vulkan GPU devices
python coderai --vulkan-list-devices
```

**Vulkan Backend Notes:**
- Uses GGUF format models (much smaller than full HuggingFace models)
- Q4_K_M quantization recommended for 4GB+ VRAM GPUs
- Q5_K_M or Q6_K for higher quality
- Works on AMD RX 400 series and newer
- Also works on NVIDIA GPUs but CUDA backend is preferred for NVIDIA
- **Update llama-cpp-python** for newer model support: `pip install --upgrade llama-cpp-python --no-cache-dir`

### CPU-Only

While not recommended for performance, you can run on CPU:

```bash
# NVIDIA backend on CPU
pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-nvidia.txt
python coderai --model microsoft/DialoGPT-medium --backend nvidia

# Or Vulkan backend on CPU (llama-cpp supports CPU fallback)
CMAKE_ARGS="-DGGML_VULKAN=OFF" pip install llama-cpp-python
python coderai --model model.gguf --backend vulkan
```

### ROCm Alternative (deprecated)

While the Vulkan backend is now recommended for AMD GPUs, ROCm support is still available through the NVIDIA backend if you have ROCm-enabled PyTorch installed.

### Low VRAM Configuration

For GPUs with limited VRAM (4-8GB):

```bash
# Option 1: Use 4-bit quantization
python coderai --model meta-llama/Llama-2-7b-chat-hf --load-in-4bit

# Option 2: Use 8-bit quantization
python coderai --model meta-llama/Llama-2-13b-chat-hf --load-in-8bit

# Option 3: Enable disk offload for very large models
python coderai --model bigscience/bloom-7b1 --offload-dir /path/to/fast/storage
```

### Multi-GPU Setup

Multiple GPUs are automatically detected and utilized. The model will be distributed across available devices based on memory availability.

```bash
# Set visible GPUs (optional)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run - model will be distributed across all visible GPUs
python coderai --model meta-llama/Llama-2-70b-chat-hf --load-in-8bit
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
   - AMD RX 400 series and newer
   - NVIDIA GTX 900 series and newer (but CUDA backend preferred for NVIDIA)
   - Intel Arc GPUs (experimental)

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
- Powered by [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) with Vulkan support (AMD backend)
- Inspired by the OpenAI API specification

---

**Note on AI.PROMPT**: This project was enhanced following instructions to add Vulkan support for AMD GPUs alongside the existing NVIDIA/CUDA support. The implementation uses llama-cpp-python for Vulkan/GGUF model support while maintaining full compatibility with the existing HuggingFace/Transformers backend for NVIDIA GPUs.
