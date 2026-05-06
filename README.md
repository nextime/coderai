# CoderAI

An OpenAI-compatible API server with web administration dashboard, supporting multiple GPU backends: NVIDIA (CUDA), AMD (Vulkan), and Intel (Vulkan). Configuration-driven architecture with per-model settings and full multi-modal support.

## Features

### Core Capabilities
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's API endpoints
- **Web Studio**: Modern UI for all generation tasks — chat, image, video, audio, pipelines
- **Configuration-Based**: JSON config files for all settings — no complex CLI arguments
- **Multi-Modal**: Text, image, video, audio, TTS, STT, embeddings
- **Per-Model Configuration**: Individual settings for each model (GPU layers, quantization, context size)
- **On-Demand Loading**: Models load automatically when requested, unload when idle

### GPU Backend Support
- **NVIDIA (CUDA)**: PyTorch + Transformers for HuggingFace models
- **AMD GPUs**: llama-cpp-python + Vulkan for GGUF models
- **Intel GPUs**: iGPU/Arc support via Vulkan
- **Auto-Detection**: Automatically selects best available backend
- **Multi-GPU**: Automatic distribution across multiple devices

### Image Generation
- **Text-to-Image**: Stable Diffusion, SDXL, Flux, and GGUF image models (via stable-diffusion.cpp)
- **Image-to-Image**: Style transfer and image editing
- **Inpainting**: Fill masked regions with AI-generated content
- **Upscaling**: Real-ESRGAN super-resolution (2×/4×/8×)
- **Deblur**: Wiener deconvolution + unsharp masking
- **Unpixelate**: Real-ESRGAN restoration of pixelated/compressed images
- **Outfit Change**: Auto-generated clothing mask + inpainting for wardrobe changes
- **Face Swap**: InsightFace INSwapper — swap faces in images and videos
- **Depth Estimation**: Monocular depth maps
- **Segmentation**: SAM-based object segmentation

### Video Generation
- **Text-to-Video**: Generate video from text prompts
- **Image-to-Video**: Animate a still image
- **Video-to-Video**: Transform existing video
- **Ti2V**: Text + image → video with camera motion control
- **Frame Interpolation**: Increase FPS via RIFE or ffmpeg minterpolate
- **Upscaling**: Real-ESRGAN video upscaling
- **Subtitles**: Whisper transcription + optional translation + burn-in
- **Dubbing**: Transcribe → translate → TTS → replace audio track

### Audio
- **Text-to-Speech**: Kokoro TTS with voice selection and speed control
- **Speech-to-Text**: Whisper transcription (faster-whisper / whispercpp)
- **Music/SFX Generation**: MusicGen, AudioGen, AudioLDM2
- **Voice Cloning**: F5-TTS zero-shot voice cloning from a reference audio clip
- **Voice Conversion (SVC)**: Seed-VC — converts timbre while preserving pitch, melody and expression; **singing mode** for music
- **Voice Profiles**: Save named voice profiles (reference audio + transcript) for reuse

### Pipelines
Built-in multi-step pipelines callable from the API or web UI:

| Endpoint | Description |
|---|---|
| `POST /v1/pipelines/image-to-video` | Generate image → animate → optional audio |
| `POST /v1/pipelines/video-dub` | Transcribe → translate → TTS dub → burn subtitles |
| `POST /v1/pipelines/story` | LLM script → images per scene → video → TTS narration |
| `POST /v1/pipelines/audio-dub` | Transcribe audio/video → translate → clone voice → replace audio |

**Custom Pipeline Builder**: Create, save and run your own multi-step pipelines from the web UI or API. Chain any combination of 18 step types with `{{input}}` and `{{stepN.output}}` template variables.

### Advanced Features
- **Memory Management**: Smart VRAM → RAM → Disk offloading (NVIDIA)
- **Quantization**: 4-bit/8-bit via bitsandbytes (NVIDIA) or GGUF quantization (Vulkan)
- **Flash Attention 2**: Optional faster inference for supported NVIDIA GPUs
- **Streaming**: Server-sent events for real-time token generation
- **Tool Calling**: Function calling and tool use support
- **Authentication**: Session-based auth with API token support
- **Webcam/Microphone**: Capture directly from browser for face swap and voice cloning

---

## Installation

### Prerequisites

- Python 3.8+
- For NVIDIA GPUs: CUDA toolkit (11.8+ recommended)
- For AMD/Intel GPUs (Vulkan): Vulkan drivers and SDK
- For CPU-only: No additional requirements

### Quick Install with Build Script

```bash
git clone git@git.nexlab.net:nexlab/coderai.git
cd coderai

./build.sh all      # All backends (recommended)
./build.sh nvidia   # NVIDIA only
./build.sh vulkan   # AMD/Intel only
```

The build script creates a virtual environment, installs dependencies, and builds GPU-accelerated backends including `stable-diffusion-cpp-python` with CUDA+Vulkan support.

### Manual Installation

```bash
python -m venv venv
source venv/bin/activate

# NVIDIA
pip install torch torchvision torchaudio
pip install -r requirements-nvidia.txt

# AMD/Intel (Vulkan)
CMAKE_ARGS="-DGGML_VULKAN=ON" pip install llama-cpp-python --no-cache-dir
pip install -r requirements-vulkan.txt
```

### Stable Diffusion GGUF (CUDA + Vulkan)

```bash
CMAKE_ARGS="-DSD_WEBM=OFF -DSD_CUDA=ON -DSD_VULKAN=ON" \
  pip install stable-diffusion-cpp-python --no-cache-dir --force-reinstall
```

### Voice Cloning and Voice Conversion

```bash
pip install f5-tts    # Voice cloning (F5-TTS)
pip install seed-vc   # Voice conversion / singing SVC
```

### Full-Quality Audio ML Stack

```bash
pip install demucs deepfilternet rnnoise voicefixer
```

Use this stack when you want:
- real ML stem separation for `/v1/audio/stems`
- learned restoration for `/v1/audio/cleanup`
- the strongest available backend path for `/v1/pipelines/audio-music-dub`

Notes:
- `demucs` is the primary separator for vocals/instrumental and multi-stem workflows.
- `deepfilternet` is the primary learned cleanup backend.
- `rnnoise` and `voicefixer` are optional alternates / complements.
- Full music-dub quality depends on separation plus singing-capable conversion; even with this stack, output quality still depends heavily on source material and model/runtime availability.

### Face Swap

```bash
pip install insightface onnxruntime-gpu
# inswapper_128.onnx downloads automatically on first use
```

---

## Usage

```bash
source venv_all/bin/activate

python coderai                          # Default config at ~/.coderai/
python coderai --config /path/to/cfg   # Custom config directory
python coderai --debug                 # Debug mode
```

Server starts on `http://0.0.0.0:8000`.

### Access Points

| URL | Description |
|---|---|
| `http://localhost:8000/admin` | Admin dashboard |
| `http://localhost:8000/chat` | Web Studio (generation UI) |
| `http://localhost:8000/v1/*` | OpenAI-compatible API |
| `http://localhost:8000/docs` | Interactive API docs |

Default credentials: `admin` / `admin` (prompted to change on first login).

---

## Configuration

Config files live in `~/.coderai/` (or `--config` path):

```
~/.coderai/
├── config.json      # Server, backend, global settings
├── models.json      # Model registry and per-model config
├── auth.json        # Users, API tokens, sessions
├── pipelines.json   # Custom pipeline definitions
└── secret_key       # Session signing key (auto-generated)
```

### config.json

```json
{
  "server": { "host": "0.0.0.0", "port": 8000 },
  "backend": { "type": "auto" },
  "models": { "default_load_mode": "ondemand" },
  "offload": { "load_in_4bit": false, "flash_attention": false },
  "vulkan": { "n_gpu_layers": -1, "n_ctx": 2048, "device_id": 0 }
}
```

### models.json

```json
{
  "text_models":  [{ "id": "Qwen/Qwen3.5-9B", "backend": "nvidia", "enabled": true }],
  "image_models": [{ "id": "z_image_turbo-Q2_K.gguf", "backend": "auto", "enabled": true }],
  "tts_models":   [{ "id": "kokoro-v1.0.onnx", "enabled": true }],
  "audio_models": [],
  "video_models": []
}
```

---

## API Reference

### Text

| Endpoint | Description |
|---|---|
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completions (streaming supported) |
| `POST /v1/completions` | Text completions |
| `POST /v1/embeddings` | Text embeddings |

### Image

| Endpoint | Description |
|---|---|
| `POST /v1/images/generations` | Text-to-image |
| `POST /v1/images/edits` | Image-to-image |
| `POST /v1/images/inpaint` | Inpainting |
| `POST /v1/images/upscale` | Real-ESRGAN upscaling |
| `POST /v1/images/deblur` | Deblur / sharpen |
| `POST /v1/images/unpixelate` | Remove pixelation |
| `POST /v1/images/outfit` | Change clothing/outfit |
| `POST /v1/images/faceswap` | Face swap (image or video) |
| `POST /v1/images/depth` | Depth estimation |
| `POST /v1/images/segment` | Object segmentation |

### Video

| Endpoint | Description |
|---|---|
| `POST /v1/video/generations` | Generate video (t2v/i2v/v2v/ti2v/interp) |
| `POST /v1/video/upscale` | Upscale video |
| `POST /v1/video/subtitle` | Generate/burn subtitles |
| `POST /v1/video/interpolate` | Frame interpolation |
| `POST /v1/video/dub` | Dub video to another language |

### Audio

| Endpoint | Description |
|---|---|
| `POST /v1/audio/speech` | Text-to-speech |
| `POST /v1/audio/transcriptions` | Speech-to-text (Whisper) |
| `POST /v1/audio/generate` | Music/SFX generation |
| `POST /v1/audio/clone` | Voice cloning TTS (F5-TTS) |
| `POST /v1/audio/convert` | Voice conversion / SVC (Seed-VC) |
| `GET /v1/audio/voices` | List saved voice profiles |
| `POST /v1/audio/voices` | Save a voice profile |
| `DELETE /v1/audio/voices/{name}` | Delete a voice profile |

### Pipelines

| Endpoint | Description |
|---|---|
| `POST /v1/pipelines/image-to-video` | Image gen → video animation |
| `POST /v1/pipelines/video-dub` | Full video dubbing pipeline |
| `POST /v1/pipelines/story` | LLM → images → video → TTS |
| `POST /v1/pipelines/audio-dub` | Audio/video dub with voice cloning |
| `GET /v1/pipelines/custom` | List custom pipelines |
| `POST /v1/pipelines/custom` | Create custom pipeline |
| `PUT /v1/pipelines/custom/{id}` | Update custom pipeline |
| `DELETE /v1/pipelines/custom/{id}` | Delete custom pipeline |
| `POST /v1/pipelines/custom/{id}/run` | Run a saved custom pipeline |
| `POST /v1/pipelines/run` | Run an inline pipeline definition |
| `GET /v1/pipelines/step-types` | List available step types |

### Custom Pipeline Definition

```json
{
  "name": "My Pipeline",
  "steps": [
    {
      "type": "text_gen",
      "label": "Write scene description",
      "params": {
        "model": "Qwen/Qwen3.5-9B",
        "prompt": "Describe a visual scene for: {{input}}"
      }
    },
    {
      "type": "image_gen",
      "params": {
        "model": "z_image_turbo-Q2_K.gguf",
        "prompt": "{{step0.output}}"
      }
    },
    {
      "type": "video_gen",
      "params": {
        "model": "wan-model",
        "mode": "i2v",
        "init_image": "{{step1.url}}"
      }
    }
  ]
}
```

Template variables: `{{input}}`, `{{stepN.output}}`, `{{stepN.url}}`.

Available step types: `text_gen`, `image_gen`, `image_edit`, `image_inpaint`, `image_upscale`, `image_deblur`, `image_unpix`, `image_outfit`, `image_faceswap`, `video_gen`, `video_upscale`, `video_sub`, `video_interp`, `video_dub`, `tts`, `audio_gen`, `voice_clone`, `voice_convert`.

---

## Backend-Specific Notes

### NVIDIA (CUDA)

- HuggingFace format models (safetensors/pytorch)
- GGUF text models via llama-cpp-python with CUDA
- Stable Diffusion GGUF via stable-diffusion.cpp with CUDA
- Optional: bitsandbytes (4-bit/8-bit quantization), Flash Attention 2

### AMD / Intel (Vulkan)

- GGUF format models via llama-cpp-python with Vulkan
- Stable Diffusion GGUF via stable-diffusion.cpp with Vulkan
- No ROCm/OneAPI required
- Intel iGPUs: use Q4_K_M models under 2GB

### Multi-GPU (NVIDIA + AMD)

To force Vulkan to use only the AMD GPU:

```json
{ "vulkan": { "device_id": 1, "single_gpu": true } }
```

### Low VRAM

```json
{ "offload": { "load_in_4bit": true } }
```

---

## Troubleshooting

### numpy ABI mismatch after installing new packages

```bash
pip install --force-reinstall --no-cache-dir --no-deps realesrgan insightface
```

### stable-diffusion.cpp: "get sd version from file failed"

The model architecture is not recognized. Update stable-diffusion-cpp-python:

```bash
CMAKE_ARGS="-DSD_WEBM=OFF -DSD_CUDA=ON -DSD_VULKAN=ON" \
  pip install stable-diffusion-cpp-python --upgrade --no-cache-dir
```

### stable-diffusion.cpp using CPU instead of GPU

Reinstall with GPU flags:

```bash
CMAKE_ARGS="-DSD_WEBM=OFF -DSD_CUDA=ON -DSD_VULKAN=ON" \
  pip install stable-diffusion-cpp-python --no-cache-dir --force-reinstall
```

### Vulkan backend not available

```bash
# Install Vulkan drivers and shader compiler
sudo apt install libvulkan-dev vulkan-tools mesa-vulkan-drivers glslc glslang-tools

# Rebuild llama-cpp-python
CMAKE_ARGS="-DGGML_VULKAN=ON" pip install llama-cpp-python --no-cache-dir --force-reinstall
```

### Flash Attention build fails

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### Model not loading (503 errors)

- Verify model name matches exactly what's in `models.json`
- Check HuggingFace authentication: `huggingface-cli login`
- Ensure the model type matches the endpoint (image models cannot be used via `/v1/chat/completions`)

---

## License

GNU General Public License v3.0 — see [LICENSE.md](LICENSE.md).

## Contributing

Merge requests welcome.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) — NVIDIA text backend
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — Vulkan/CUDA GGUF text backend
- [stable-diffusion-cpp-python](https://github.com/william-murray1204/stable-diffusion-cpp-python) — GGUF image backend
- [InsightFace](https://github.com/deepinsight/insightface) — face swap
- [F5-TTS](https://github.com/SWivid/F5-TTS) — voice cloning
- [Seed-VC](https://github.com/Plachta/Seed-VC) — singing voice conversion
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — image/video upscaling
