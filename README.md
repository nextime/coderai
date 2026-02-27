# CoderAI

An OpenAI-compatible API server for HuggingFace models with intelligent memory management, GPU auto-detection, and advanced features like tool calling and streaming.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's API endpoints
- **Memory-Aware Model Loading**: Automatically determines optimal loading strategy based on available VRAM and RAM
- **Sequential Offloading**: Smart offload from VRAM → RAM → Disk when needed
- **Multi-GPU Support**: Automatic distribution across multiple CUDA/ROCm devices
- **GPU Auto-Detection**: Automatically detects CUDA (NVIDIA) or ROCm (AMD) GPUs
- **Quantization Support**: 4-bit and 8-bit quantization via bitsandbytes for reduced memory usage
- **Flash Attention 2**: Optional faster attention implementation for supported GPUs
- **Streaming Responses**: Server-sent events for real-time token generation
- **Tool Calling**: Support for function calling and tool use
- **Multiple Endpoints**: `/v1/chat/completions`, `/v1/completions`, and `/v1/models`

## Installation

### Prerequisites

- Python 3.8+
- For NVIDIA GPUs: CUDA toolkit (11.8+ recommended)
- For AMD GPUs: ROCm (5.4+ recommended)
- For CPU-only: No additional requirements

### Basic Installation

```bash
# Clone the repository
git clone git@git.nexlab.net:nexlab/coderai.git
cd coderai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install base requirements
pip install -r requirements.txt
```

### Platform-Specific PyTorch Installation

PyTorch installation varies by platform. Uncomment the appropriate section in [`requirements.txt`](requirements.txt) or install manually:

#### NVIDIA (CUDA)

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4 (latest)
pip install torch torchvision torchaudio
```

#### AMD (ROCm)

```bash
# For ROCm 5.4.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# For ROCm 5.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# For ROCm 6.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

#### CPU Only

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Optional Dependencies

#### bitsandbytes (Quantization)

For 4-bit and 8-bit quantization support (reduces VRAM requirements):

```bash
# CUDA
pip install bitsandbytes>=0.41.0

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
# Run with a specific model
python coderai --model microsoft/DialoGPT-medium

# The server will start on http://0.0.0.0:8000 by default
```

### Command-Line Options

```
usage: coderai [-h] [--model MODEL] [--host HOST] [--port PORT]
               [--offload-dir OFFLOAD_DIR] [--load-in-4bit] [--load-in-8bit]
               [--ram RAM] [--flash-attn]

OpenAI-compatible API server with memory-aware model loading

options:
  -h, --help            show this help message and exit
  --model MODEL         HuggingFace model name or path
  --host HOST           Host to bind to (default: 0.0.0.0)
  --port PORT           Port to bind to (default: 8000)
  --offload-dir OFFLOAD_DIR
                        Directory for disk offload when model doesn't fit in
                        VRAM+RAM (default: ./offload)
  --load-in-4bit        Load model in 4-bit precision (requires bitsandbytes)
  --load-in-8bit        Load model in 8-bit precision (requires bitsandbytes)
  --ram RAM             Manually specify available RAM in GB (bypasses auto-
                        detection)
  --flash-attn          Use Flash Attention 2 for faster inference (requires
                        flash-attn package and compatible GPU)
```

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

### CUDA (NVIDIA GPU)

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Run with GPU acceleration (automatic)
python coderai --model meta-llama/Llama-2-7b-chat-hf

# Optional: Enable Flash Attention 2 for faster inference
python coderai --model meta-llama/Llama-2-7b-chat-hf --flash-attn
```

### ROCm (AMD GPU)

```bash
# Install ROCm-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# Run with GPU acceleration (automatic)
python coderai --model meta-llama/Llama-2-7b-chat-hf

# Check ROCm detection in output
```

### CPU-Only

```bash
# Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Run on CPU (automatic fallback)
python coderai --model microsoft/DialoGPT-medium
```

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

### Small Models (For Testing)

- `microsoft/DialoGPT-medium` (~345M parameters)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~1.1B parameters)
- `facebook/blenderbot-400M-distill` (~400M parameters)

### Medium Models (4-8GB VRAM with 4-bit)

- `meta-llama/Llama-2-7b-chat-hf` (~7B parameters)
- `mistralai/Mistral-7B-Instruct-v0.2` (~7B parameters)
- `HuggingFaceH4/zephyr-7b-beta` (~7B parameters)

### Large Models (Multiple GPUs or High VRAM)

- `meta-llama/Llama-2-13b-chat-hf` (~13B parameters)
- `meta-llama/Llama-2-70b-chat-hf` (~70B parameters) - requires multiple GPUs or disk offload
- `bigscience/bloom-7b1` (~7B parameters)

## Troubleshooting

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

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a merge request.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- Inspired by the OpenAI API specification
