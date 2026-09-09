# CoderAI

**Repository:** [https://git.nexlab.net/nexlab/coderai](https://git.nexlab.net/nexlab/coderai)

![CoderAI](CoderAI.gif)

A multimodal and multi-backend local model orchestrator with an OpenAI-compatible API server to run models on local GPUs, supporting multiple GPU backends: NVIDIA (CUDA), AMD (Vulkan), and Intel (Vulkan). Configuration-driven architecture with per-model settings and full multi-modal support.

Text, images, video, speech, embeddings, OCR and LoRA training behind one API — served by
whichever runtime each model actually needs, from PyTorch to native C MoE engines that
stream a multi-terabyte model off disk, to a GPU **rented by the second** when the local
one isn't enough.

## Features

### Core Capabilities
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's API endpoints
- **Web Studio**: Modern UI for all generation tasks — chat, image, video, audio, pipelines
- **Configuration-Based**: JSON config files for all settings — no complex CLI arguments
- **Multi-Modal**: Text, image, video, audio, TTS, STT, embeddings
- **Per-Model Configuration**: Individual settings for each model (GPU layers, quantization, context size)
- **On-Demand Loading**: Models load automatically when requested, unload when idle
- **Memory Management**: Smart VRAM → RAM → Disk offloading for efficient resource usage
- **Parallel Execution**: Run multiple models simultaneously (VRAM permitting)
- **Auto-Swap**: Automatic model switching on request — load what's needed, unload what's idle
- **Request Queue**: Concurrent requests are queued and processed in order per model
- **Prompt Caching**: Reuse KV cache across requests to reduce latency and computation
- **Prompt Aggregation**: Batch concurrent requests into a single inference pass for higher throughput
- **Custom Pipelines**: Create and save multi-step workflows combining any generation tasks
- **Pre-Built Pipelines**: Ready-to-use pipelines for common workflows (image-to-video, dubbing, story generation)

### GPU Backend Support
- **NVIDIA (CUDA)**: PyTorch + Transformers for HuggingFace models
- **AMD GPUs**: llama-cpp-python + Vulkan for GGUF models
- **Intel GPUs**: iGPU/Arc support via Vulkan
- **Auto-Detection**: Automatically selects best available backend
- **Multi-GPU**: Automatic distribution across multiple devices
- **Front / Engine Split**: A torch-free front proxy supervises one engine subprocess per
  GPU, so the UI stays responsive while an engine is loading or generating

### Inference Engines

Beyond the built-in `transformers` and `gguf` paths, CoderAI can drive external native
engines — each selected **per model** with a `backend` pin in `models.json`. They exist so a
frontier-size MoE model can run on hardware that "shouldn't" be able to hold it, or so a
model can be served with far higher concurrency.

| `backend` | Engine | What it's for |
|---|---|---|
| `transformers` | PyTorch + HF Transformers | safetensors models on CUDA |
| `gguf` | llama.cpp | GGUF models on CUDA or Vulkan |
| `ds4` | [ds4 / DwarfStar](https://github.com/antirez/ds4) | DeepSeek-V4, native C/CUDA engine with its own OpenAI server |
| `colibri` | [colibri](https://github.com/JustVugg/colibri) | GLM-5.2 / DeepSeek-V4 / Kimi-K3 — pure-C MoE engine streaming experts from disk; driven directly over its stdin/stdout mux protocol |
| `k3` | [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) | Kimi-K3 (2.78T params) on **CPU**, streaming trunk + experts from disk in as little as ~8 GB RAM |
| `kt` | [ktransformers](https://github.com/kvcache-ai/ktransformers) via SGLang | CPU+GPU heterogeneous MoE (DeepSeek / Kimi / Qwen / GLM / MiniMax) |
| `vllm` | [vLLM](https://github.com/vllm-project/vllm) | continuous batching + paged KV for high aggregate throughput; runs in an isolated venv |
| `runpod` | RunPod | a **remote rented GPU** — see below |

CoderAI owns the full lifecycle of each: build, weight download, process supervision,
health, VRAM co-tenancy and teardown. `ds4`, `kt`, `vllm` and `runpod` are HTTP-proxy
backends; `colibri` and `k3` are driven over a native wire protocol. `vllm` also appears as
a **first-class engine node** on the engines/tasks pages, alongside `nvidia` and `radeon`.

See [`docs/`](docs/) for a per-engine guide.

### Remote GPUs (RunPod)

A model can be served on a **rented cloud GPU** instead of local hardware — same
`/v1/models` entry, same `/v1/chat/completions`, same Tasks page, but the weights never
touch your disk. Two per-model modes:

- **Pods** — CoderAI provisions, health-checks, load-balances, scales and destroys the GPU
  containers itself, with GPU/price/VRAM selection, spot support, capacity fallback and
  stuck-boot retry.
- **Serverless** — CoderAI proxies to a RunPod serverless endpoint you already created.

Plus:
- **Cost controls**: per-model and global `$/hr` rate caps, and cumulative spend budgets over
  a trailing `hour`/`day`/`week`/`month` window, on a persistent ledger
- **Cold start & idle teardown**: pods serve many requests, then self-destruct a configurable
  time after the last one
- **Stale-pod reaper**: a maintenance loop identifies pods this deployment created and
  terminates any that shouldn't exist, so a crash can never leave a GPU billing
- **Spillover**: a *local* model can burst to RunPod only when the local GPU is full or absent
- **No local model required**: an instance with no GPU and no local weights can run purely as
  a RunPod orchestrator

Full guide: [`docs/runpod.md`](docs/runpod.md).

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
- **Character Profiles**: Apply up to 6 named character profiles (IP-Adapter) to anchor appearance across generations
- **Environment Profiles**: Apply up to 6 named environment profiles to condition scene/background style
- **Generation Progress**: Real-time per-step progress tracking via `GET /v1/images/progress`

### Video Generation
- **Text-to-Video**: Generate video from text prompts
- **Image-to-Video**: Animate a still image
- **Video-to-Video**: Transform existing video
- **Ti2V**: Text + image → video with camera motion control
- **Frame Interpolation**: Increase FPS via RIFE or ffmpeg minterpolate
- **Upscaling**: Real-ESRGAN video upscaling
- **Subtitles**: Whisper transcription + optional translation + burn-in
- **Dubbing**: Transcribe → translate → TTS → replace audio track
- **Character Profiles**: Apply up to 6 saved character profiles for visual consistency. Identity is conditioned via IP-Adapter on pipelines that support it (AnimateDiff and other SD-based video models); pipelines without IP-Adapter (Wan, CogVideoX, LTX, SVD) fall back to the prompt hint plus any per-character LoRA, and say so in the log
- **Environment Profiles**: Apply up to 6 saved environment profiles to condition the scene
- **Multi-Character Dialog**: Assign spoken lines to individual characters — each line picks a character profile, voice profile or TTS voice ID, and text; lines are **voice-cloned with F5-TTS** from the named profile, mixed with correct timing (sequential or manual), and lip-synced via Wav2Lip
- **Lip Sync**: Wav2Lip is a **managed install** — the code and weights are fetched and patched on first use, so no manual setup. If it can't run, the response carries a `warnings` entry saying the audio was muxed without mouth sync rather than passing off an unsynced video as a success

### Audio
- **Text-to-Speech**: Kokoro TTS with voice selection and speed control
- **Music/SFX Generation**: MusicGen, AudioGen, AudioLDM2
- **Voice Cloning**: F5-TTS zero-shot voice cloning from a reference audio clip
- **Voice Conversion (SVC)**: Seed-VC — converts timbre while preserving pitch, melody and expression; **singing mode** for music
- **Voice Profiles**: Save named voice profiles (reference audio + transcript) for reuse; voice profiles can be extracted directly from video files and updated via PATCH
- **Stem Separation**: Demucs vocals/instrumental or 4-stem split (`/v1/audio/stems`)
- **Restoration**: DeepFilterNet denoise, normalise, de-hum, de-click (`/v1/audio/cleanup`)

### Speech-to-Text and Speaker Recognition

`/v1/audio/transcriptions` is not tied to one Whisper implementation. The **STT backend is
chosen per model** with the model entry's `backend` key:

| `backend` | Engine | Notes |
|---|---|---|
| *(unset)* | whisper.cpp server → faster-whisper → whispercpp | the default fallback chain |
| `whisper-server` | whisper.cpp GGUF | per-model runner; counts as a model load and is VRAM-evictable |
| `whisper-hf` | CrisperWhisper (HF) | verbatim transcription with word timestamps; isolated venv |
| `crisperwhisper` | CrisperWhisper worker | isolated venv, long-form overlap windowing |
| `wav2vec2` | Wav2Vec2 (HF) | FP16 on GPU |
| `vosk` | Vosk | CPU, per-language model directory |
| `nemo` | NVIDIA NeMo (Canary / Parakeet) | isolated venv; the **translation-capable** family |

- **Per-model languages**: a model entry can declare a `languages` list; a request for an
  unlisted language is rejected with the allowed list, and `/v1/models` surfaces it
- **Translation**: `target_language` on a model with `supports_translation: true`
- **Word timestamps**: `timestamp_granularities=word` → a top-level `words[]`
- **Diarization**: `diarize=true` tags every segment with a `speaker`, or call
  `/v1/audio/diarization` directly (pyannote in an isolated venv, ungated-first with an
  `HF_TOKEN` fallback)
- **Speaker recognition**: enrol named voiceprints and then **identify** or **verify** a
  voice (`ecapa` / `pyannote` / `wespeaker` backends). Diarization with `identify=true`
  relabels the turns with the enrolled names instead of `SPEAKER_00`
- **VRAM-aware**: every STT backend participates in VRAM eviction; `keep_resident: true`
  keeps a small model co-resident

### Embeddings and Reranking

`/v1/embeddings` covers far more than text. The backend is inferred from the model itself:

| Family | Serves |
|---|---|
| sentence-transformers / transformers | general text embeddings |
| **BGE-M3** | native multi-vector: `dense`, `sparse` (lexical weights) and `colbert` (per-token) in one pass |
| CLIP / vision (DINOv2, ViT) | image embeddings |
| **GME-Qwen2-VL** | text and image in one shared space (native loader, no `trust_remote_code`) |
| GGUF via llama.cpp | `llama` and `llama-vl` (with an `mmproj`) |
| dinov2.cpp | GGUF DINOv2 through a `dinov2-embed` subprocess (Vulkan/CPU) |
| **GeoCLIP** | `geoclip` (image) and `geoclip-location` (`"lat,lon"`) in one shared 512-d space — image geolocation |
| **VPR** | `vpr` / EigenPlaces (2048-d) and `salad` / DINOv2-SALAD (8448-d, isolated venv) for visual place recognition |

Request extras: `image` (URL, data URI, path or base64), `embedding_types` to pick
dense/sparse/colbert, `dimensions` to truncate, and `quantization` (TurboQuant `turbo8` …
`turbo2`) for packed low-bit vectors.

**`POST /v1/rerank`** adds cross-encoder reranking (e.g. `bge-reranker-v2-m3`) — query +
documents in, `relevance_score` per document out. No extra dependency; rerankers are
registered in the embedding model category.

### Document OCR

A dedicated OCR subsystem (`/v1/ocr`) — real OCR engines, **not** a VLM prompted to read:

- **Engines**: `doctr` (in-process), `paddle` (PaddleOCR + PP-Structure layout/tables, isolated
  venv) and `surya` (opt-in, GPL; local, vLLM-served Surya-2, or an external llama-server)
- **Input**: images or PDFs (rasterised at a configurable DPI), single or batch
- **Output**: full text plus per-page `lines[]` with bounding boxes and confidence,
  `regions[]` (layout) and `tables[]`
- **Structured extraction**: `structured=true` runs a text model against a **schema** —
  a JSON document, not code — to return typed fields. Schemas are data-driven and stored in
  `<config>/ocr_schemas/`; built-ins cover `italian_sentenza`, `generic_document` and an
  `invoice` JSON Schema (optionally validated)
- **Stamp / signature detection**: `detect=layout|detector|both` — layout-marker matching
  (EN + IT) and/or a small YOLO detector. It **locates** stamps and signatures; it does not
  verify them
- Also available as an `ocr` pipeline step, so OCR → LLM chains in one call

### LoRA Training

Train a LoRA on your own GPU from the API or the web UI (`/v1/loras/train`):

- **Image or video targets** — SD1.x/SDXL UNet, or a Wan video DiT with 4-bit QLoRA
- **Source images** from a saved character/environment profile, or uploaded inline
- **Content-addressed blob store** (`/v1/loras/upload`, `/v1/loras/blob/{hash}`) so a client
  can skip re-uploading a LoRA it already sent
- **Resumable and non-blocking**: `wait=false` returns a `job_id`; progress is pollable by
  job or by session, jobs survive a client disconnect and are restartable from the Tasks page
- Scheduled centrally (one training at a time) and holds the GPU reservation, so a
  concurrent model load can't OOM the trainer

### Character Profiles
Named collections of reference images used to condition character appearance via IP-Adapter across any image or video generation. Up to 6 profiles can be selected per generation.

- **Extract from images/video**: Automatic face detection and cropping to build a reference set
- **Generate from text prompt**: Create reference images from a visual description (no source images needed) using any registered image model
- **Reuse across generations**: Select saved profiles by name in the API or web UI
- **CRUD API**: Create, list, view, patch (add/remove images), and delete profiles

### Environment Profiles
Named collections of reference images that condition scene background and environment style via IP-Adapter. Same multi-slot selection (up to 6) as character profiles.

- **Extract from images/video**: Selects the sharpest frames/images (no face crop — full image used)
- **Generate from text prompt**: Create reference images from a scene description using any registered image model
- **Reuse across generations**: Select saved profiles by name in image and video generation

### 2D ↔ 3D Conversion
- **Image → 3D**: Convert any image to stereo pair, anaglyph, depth map, or mesh (GLB/OBJ) via depth estimation + optional 3D reconstruction
- **3D → Image**: Render a GLB/OBJ model to a 2D image from a specified viewpoint
- **Video → 3D**: Apply frame-by-frame depth processing to produce a 3D video
- **3D → Video**: Render a 3D model as a turntable rotation video
- **Text/Image → 3D Model**: Generate a 3D GLB model from a text prompt or reference image (requires a compatible 3D generation model)

### Generation Archive
- **Auto-save**: Every generated file (image, video, audio) is optionally saved to a configurable local directory
- **Configurable retention**: `1h`, `1d`, `2d`, `1w`, `1m`, `3m`, `6m`, `1y`, or `never` — automatic hourly cleanup
- **Browse & delete**: View and delete archived files from the web UI Archive tab or via the API
- **Settings page**: Archive directory and retention are configurable from the web UI Settings page without a restart

### Pipelines
Built-in multi-step pipelines callable from the API or web UI:

| Endpoint | Description |
|---|---|
| `POST /v1/pipelines/image-to-video` | Generate image → animate → optional audio |
| `POST /v1/pipelines/video-dub` | Transcribe → translate → TTS dub → burn subtitles |
| `POST /v1/pipelines/story` | LLM script → images per scene → video → TTS narration |
| `POST /v1/pipelines/audio-dub` | Transcribe audio/video → translate → clone voice → replace audio |

#### Music dubbing

`POST /v1/pipelines/audio-music-dub` sings a song in another language over its own
backing track. Six real stages:

1. **Separate** — Demucs splits the track; the vocal is isolated and drums/bass/other are
   summed back into an instrumental bed
2. **Transcribe** — STT runs on the *isolated vocal*, not the full mix, which is what makes
   the lyrics legible
3. **Adapt** — with a `text_model`, an LLM adapts the lyrics to be **singable**: same
   syllable count per line, rhyme preserved where possible. Without one it falls back to a
   literal argostranslate pass
4. **Re-sing** — F5-TTS synthesises the new lyrics using the isolated original vocal as the
   cloning reference, so the dub keeps the original singer's timbre with no voice profile to
   set up (or pass `voice_name` to use a saved one)
5. **Match the singing** — an optional Seed-VC pass with `f0_condition` on, conditioned on
   the original vocal, to pull the result toward sung rather than spoken delivery
6. **Remix** — the new vocal is mixed back over the instrumental and limited

```json
{
  "audio": "<base64 or data URI>",
  "audio_model": "whisper0",
  "text_model": "my-llm",
  "source_lang": "en",
  "target_lang": "it",
  "notes": "keep it playful"
}
```

Every stage that needs an optional dependency degrades to a reported `skipped` state with
a reason instead of failing or faking it. The response carries `status` (`complete` or
`degraded`), a `warnings` list, per-step statuses, and all four artifacts — `vocals`,
`instrumental`, `converted_vocals` and `final_mix`.

**Custom Pipeline Builder**: Create, save and run your own multi-step pipelines from the web UI or API. Chain any combination of 18 step types with `{{input}}` and `{{stepN.output}}` template variables.

### Advanced Features
- **Memory Management**: Smart VRAM → RAM → Disk offloading (NVIDIA)
- **VRAM Eviction**: Every backend — including the external engines, OCR and STT workers —
  is eviction-tracked, so a new load reclaims VRAM from an idle tenant instead of OOM-ing
- **Global RAM Cap**: A server-wide host-RAM ceiling with a leak watcher and LRU
  disk-offload eviction
- **Thermal Protection**: The front supervises GPU temperature and cooperatively pauses
  (then, if needed, SIGSTOPs) an engine that is cooking the card
- **GPU Swap Gate**: When engines share a GPU, same-model requests are batched before the
  card is handed over, so two engines don't thrash
- **Quantization**: 4-bit/8-bit via bitsandbytes, GPTQModel/Marlin fast kernels, or GGUF
- **Flash Attention 2**: Optional faster inference for supported NVIDIA GPUs
- **Streaming**: Server-sent events for real-time token generation
- **Tool Calling**: Function calling and tool use support
- **Authentication**: Session-based auth with API token support
- **Webcam/Microphone**: Capture directly from browser for face swap and voice cloning

---

## Quick Start

```bash
git clone git@git.nexlab.net:nexlab/coderai.git
cd coderai
./build.sh all          # build all backends (recommended)
source venv_all/bin/activate
python coderai          # starts on http://127.0.0.1:8776
```

macOS:

```bash
./osxbuild.sh all
source venv_osx_all/bin/activate
python coderai
```

Windows PowerShell:

```powershell
.\build.ps1 -Backend all
.\venv_win_all\Scripts\Activate.ps1
python coderai
```

That's it. Open `http://127.0.0.1:8776/admin` and log in with `admin` / `admin`.

---

## Installation

### Prerequisites

- Python 3.8+
- For NVIDIA GPUs: CUDA toolkit (11.8+ recommended)
- For AMD/Intel GPUs (Vulkan): Vulkan drivers and SDK
- For CPU-only: No additional requirements

### Build Script

```bash
git clone git@git.nexlab.net:nexlab/coderai.git
cd coderai

./build.sh all      # All backends (recommended)
./build.sh nvidia   # NVIDIA only
./build.sh vulkan   # AMD/Intel only
```

Platform-specific alternatives:

```bash
./osxbuild.sh all   # macOS, prefers Metal-backed builds when available
```

```powershell
.\build.ps1 -Backend all   # Windows, prefers CUDA-backed builds when available
```

Packaging options:

```bash
./build.sh all --package
./osxbuild.sh all --package
```

```powershell
.\build.ps1 -Backend all -Package
```

`--package` installs PyInstaller into the build virtual environment and produces a self-contained distributable from the venv that was just created or updated.

Packaging outputs:
- Linux: `dist-package/coderai`
- macOS: `dist-package/coderai` and `dist-package/CoderAI.app`
- Windows: `dist-package/coderai.exe`

Packaging notes:
- macOS does have an equivalent to a standalone packaged app: a `.app` bundle. `osxbuild.sh --package` now builds both a single CLI binary and a macOS app bundle.
- These packages bundle the Python interpreter and Python modules from the venv, but they do not eliminate the need for compatible external GPU/runtime drivers on the target machine.
- CUDA builds on Linux and Windows still require matching NVIDIA driver/runtime support on the destination system.
- Metal builds on macOS still require a compatible macOS system with Metal support.

The build script creates a virtual environment, installs dependencies, and builds GPU-accelerated backends including `stable-diffusion-cpp-python` with CUDA+Vulkan support.

Platform backend notes:
- Linux: CUDA for NVIDIA, Vulkan for AMD/Intel/NVIDIA, OpenCL fallback where supported.
- macOS: Metal is the correct GPU acceleration path instead of CUDA. `osxbuild.sh` uses PyTorch MPS plus `GGML_METAL` / `SD_METAL` builds where available.
- Windows: CUDA remains the primary NVIDIA acceleration path. `build.ps1` focuses on CUDA or CPU installs.
- There is no general-purpose CUDA workflow for current macOS systems; Apple GPU acceleration uses Metal.

### Platform Support Matrix

| Capability | Linux | macOS | Windows |
|---|---|---|---|
| Core server / admin UI | Yes | Yes | Yes |
| Default path handling | Yes | Yes | Yes |
| PyTorch GPU acceleration | CUDA | Metal (MPS) | CUDA |
| `llama-cpp-python` GPU path | CUDA / Vulkan | Metal | CUDA |
| `stable-diffusion-cpp-python` GPU path | CUDA / Vulkan / OpenCL | Metal | CUDA |
| `whisper.cpp` accelerated path | Vulkan / CPU fallback | Metal / CPU fallback | CPU fallback |
| InsightFace / ONNX runtime | `onnxruntime-gpu` | `onnxruntime-silicon` or CPU | `onnxruntime-gpu` |
| Build script included in repo | `build.sh` | `osxbuild.sh` | `build.ps1` |

Notes:
- "Yes" means CoderAI has an intended path for that platform, not that every optional dependency is guaranteed to install on every machine.
- macOS GPU acceleration is Metal-based; there is no standard modern CUDA path for macOS.
- Windows currently uses CUDA as the main NVIDIA acceleration path; Vulkan/OpenCL build flows are not the primary Windows setup in this repository.
- Some optional audio and media packages may still vary by Python version, hardware, and upstream wheel availability.

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

Without this stack, `/v1/audio/stems` and `/v1/audio/cleanup` return `501` with the
package to install — or accept `fallback_mode: true` for the best-effort ffmpeg path.

### Face Swap

```bash
pip install insightface onnxruntime-gpu
# inswapper_128.onnx downloads automatically on first use
```

### Optional subsystems

Several subsystems pin dependencies that conflict with the main environment, so they live in
their own requirements files and — where the conflict is unavoidable — their own virtualenv,
built on demand from the admin UI:

| File | Subsystem | Isolated venv |
|---|---|---|
| `requirements-ocr.txt` | OCR core + docTR | no |
| `requirements-ocr-paddle.txt` | PaddleOCR + PP-Structure | yes |
| `requirements-surya.txt` | Surya OCR (GPL, opt-in) | yes |
| `requirements-vllm.txt` | vLLM engine (pins its own torch/CUDA) | yes |
| `requirements-nemo.txt` | NVIDIA NeMo Canary/Parakeet STT | yes |
| `requirements-crisperwhisper.txt` | CrisperWhisper verbatim STT | yes |
| `requirements-pyannote.txt` | Speaker diarization | yes |

---

## Documentation

The [`docs/`](docs/) directory carries the deep dives — one per engine and subsystem:

| Doc | Subject |
|---|---|
| [`frontend-engine-split.md`](docs/frontend-engine-split.md) | Front proxy, engine subprocesses, routing |
| [`runpod.md`](docs/runpod.md) | Renting remote GPUs: pods, serverless, budgets, the reaper |
| [`vllm.md`](docs/vllm.md) | vLLM as a first-class engine node |
| [`deepseek-ds4.md`](docs/deepseek-ds4.md) · [`glm-colibri.md`](docs/glm-colibri.md) · [`kimi-k3.md`](docs/kimi-k3.md) · [`ktransformers.md`](docs/ktransformers.md) | The native MoE engines |
| [`ocr.md`](docs/ocr.md) | The OCR subsystem and schema store |
| [`zimage-lora-training.md`](docs/zimage-lora-training.md) | LoRA training |
| [`expressive-tts.md`](docs/expressive-tts.md) · [`dtype-auto-selection.md`](docs/dtype-auto-selection.md) · [`gguf-process-isolation.md`](docs/gguf-process-isolation.md) | Subsystem notes |
| [`reverse-proxy-nginx.md`](docs/reverse-proxy-nginx.md) | Deploying behind nginx |

---

## Usage

```bash
source venv_all/bin/activate

python coderai                          # Default config at ~/.coderai/
python coderai --config /path/to/cfg   # Custom config directory
python coderai --debug                 # Debug mode
```

Server starts on `http://127.0.0.1:8776` by default.

### Access Points

| URL | Description |
|---|---|
| `http://127.0.0.1:8776/admin` | Admin dashboard |
| `http://127.0.0.1:8776/chat` | Web Studio (generation UI) |
| `http://127.0.0.1:8776/v1/*` | OpenAI-compatible API |
| `http://127.0.0.1:8776/docs` | Interactive API docs |

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

### AISBF Broker Client

CoderAI includes an AISBF broker websocket client that can register this instance
with a broker and receive brokered requests.

You can configure it either by editing `config.json` directly or from the admin
Settings page under `AISBF Broker`.

Example broker configuration:

```json
{
  "broker": {
    "enabled": true,
    "base_url": "https://broker.example.com",
    "scope": "user",
    "username": "alice",
    "provider_id": "coderai-local",
    "client_id": "workstation-01",
    "registration_token": "your-registration-token",
    "advertised_endpoint": "http://127.0.0.1:8776",
    "transport": "websocket",
    "heartbeat_interval_seconds": 30,
    "connect_timeout_seconds": 10,
    "request_timeout_seconds": 30,
    "reconnect_initial_delay_seconds": 1,
    "reconnect_max_delay_seconds": 60
  }
}
```

Broker notes:
- `base_url` accepts `http`, `https`, `ws`, or `wss`; the websocket route is derived automatically.
- `scope: "user"` requires a non-global `username`.
- `scope: "global"` requires `username: "global"`.
- When `enabled` is `true`, `provider_id`, `client_id`, and `registration_token` are required.
- `advertised_endpoint` is optional and is sent to the broker as the externally reachable endpoint for this instance.
- Restart CoderAI after changing broker settings so the background broker service reconnects with the new configuration.

### config.json

```json
{
  "server": { "host": "127.0.0.1", "port": 8776 },
  "backend": { "type": "auto" },
  "models": { "default_load_mode": "ondemand" },
  "offload": { "load_in_4bit": false, "flash_attention": false },
  "vulkan": { "n_gpu_layers": -1, "n_ctx": 2048, "device_id": 0 },
  "archive": {
    "enabled": true,
    "directory": "",
    "retention": "1w"
  }
}
```

`archive.directory` — absolute path, or empty to use `<config_dir>/archive`.
`archive.retention` — one of: `1h`, `1d`, `2d`, `1w`, `1m`, `3m`, `6m`, `1y`, `never`.

Other top-level blocks, each with its own tab on the Settings page:

| Block | Purpose |
|---|---|
| `server.engine_specs` / `engines` / `engine_gpus` | The front/engine split — how many engines, which GPU each owns, and their capabilities |
| `ds4`, `colibri`, `k3`, `ktransformers`, `vllm` | Per-engine enablement, install dirs, model ids and build settings |
| `runpod` | RunPod account settings and global cost caps — see [`docs/runpod.md`](docs/runpod.md) |
| `ocr` | OCR engines, DPI, concurrency, detection mode, structured-extraction model and venv paths |
| `broker` | AISBF broker client (below) |

### Front proxy and engines

By default `coderai` starts a **torch-free front proxy** on the public port plus one
**engine subprocess per GPU** on `127.0.0.1:8780+`. The front routes each request to an
engine that can serve the model, aggregates status and tasks, and keeps the UI responsive
while an engine is busy loading or generating.

```bash
coderai                                     # front + auto-detected engines (default)
coderai --single-process                    # legacy single process
coderai --engine-only --internal-port 8780  # an engine (normally spawned by the front)
```

Engine placement precedence: a per-model `engine` pin → an engine that already has the
model resident → `server.default_engine` → the least-loaded compatible engine. Engines bind
localhost only and require an internal token from the front. See
[`docs/frontend-engine-split.md`](docs/frontend-engine-split.md).

### models.json

Models are grouped by category — `text_models`, `image_models`, `video_models`,
`audio_models`, `audio_gen_models`, `tts_models`, `vision_models`, `embedding_models`,
`gguf_models`, `spatial_models`:

```json
{
  "text_models":  [{ "id": "Qwen/Qwen3.5-9B", "backend": "nvidia", "enabled": true }],
  "image_models": [{ "id": "z_image_turbo-Q2_K.gguf", "backend": "auto", "enabled": true }],
  "tts_models":   [{ "id": "kokoro-v1.0.onnx", "enabled": true }],
  "audio_models": [],
  "video_models": []
}
```

Useful per-entry keys:

| Key | Meaning |
|---|---|
| `backend` | Compute backend / engine pin: `auto`, `nvidia`, `vulkan`, `opencl`, `cpu`, or an engine (`colibri`, `ds4`, `k3`, `kt`, `vllm`, `runpod`), or an STT family (`whisper-server`, `whisper-hf`, `crisperwhisper`, `wav2vec2`, `vosk`, `nemo`) |
| `engine` / `engine_fallback` | Pin the model to a named engine node (e.g. `nvidia`, `radeon`) |
| `alias` | The name clients use — **required** to tell sibling configs apart |
| `config_id` / `config_name` | Identity and label of one config of a model (see below) |
| `capabilities` | e.g. `["reranking"]`, `["embeddings"]` |
| `languages` / `supports_translation` | STT language allow-list and translation support |
| `keep_resident` | Keep a small model co-resident instead of evicting it |
| `embedding_types` | Default vector types for a multi-vector embedder |
| `lora_train_base_model` | The UNet model this model's LoRAs are trained against |
| `runpod` / `runpod_spillover` | RunPod placement and cloud-burst config |

`measured_vram_gb`, `measured_ram_gb` and `measured_n_gpu_layers` are written back at
runtime — CoderAI learns each model's real footprint and reuses it on the next load.

#### Multiple configurations of one model

The same weights can be registered more than once with different settings. Each entry gets
a `config_id`, an optional `config_name` label, and a distinct **`alias`** — the alias is
what makes the sibling addressable, so two configs without distinct aliases collapse into
one. A real example: one `whisper-large-v3-q8_0.gguf` file registered three times as
`whisper0`, `whisper1` and `whisper2`, pinned to different engine nodes so transcription
runs in parallel across two GPUs. Or one GGUF LLM registered twice with different context
sizes (`lisa`, `lisa-32k`).

Create one from the Models page with the "new config" action; runtime-measured values are
persisted per `config_id`, so the configs don't overwrite each other.

---

## API Reference

### Text

| Endpoint | Description |
|---|---|
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completions (streaming supported) |
| `POST /v1/completions` | Text completions |
| `POST /v1/embeddings` | Embeddings — text, image, multi-vector (dense/sparse/colbert), geolocation, VPR |
| `POST /v1/rerank` | Cross-encoder reranking of documents against a query |
| `GET /v1/files/{filename}` | Fetch a generated file |

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
| `POST /v1/images/to3d` | Image → stereo / anaglyph / depth map / mesh |
| `POST /v1/images/from3d` | 3D model → rendered 2D image |
| `GET /v1/images/progress` | Current generation progress (step/total) |

### Video

| Endpoint | Description |
|---|---|
| `POST /v1/video/generations` | Generate video (t2v/i2v/v2v/ti2v/interp) — supports `character_profiles`, `environment_profiles`, `dialogs` |
| `POST /v1/video/upscale` | Upscale video |
| `POST /v1/video/subtitle` | Generate/burn subtitles |
| `POST /v1/video/interpolate` | Frame interpolation |
| `POST /v1/video/dub` | Dub video to another language |
| `POST /v1/video/to3d` | Video → 3D video (frame-by-frame depth) |
| `POST /v1/video/from3d` | 3D model → turntable video |

### Audio

| Endpoint | Description |
|---|---|
| `POST /v1/audio/speech` | Text-to-speech (supports `voice_profile` for F5-TTS cloning) |
| `POST /v1/audio/transcriptions` | Speech-to-text — `language`, `target_language`, `timestamp_granularities`, `diarize` |
| `POST /v1/audio/diarization` | Who spoke when — optionally `identify` against enrolled speakers |
| `POST /v1/audio/speaker-embeddings` | Voiceprint vectors (`ecapa`/`pyannote`/`wespeaker`, optional sliding `window`) |
| `GET \| POST /v1/audio/speakers` | List or enrol named speakers |
| `DELETE /v1/audio/speakers/{name}` | Remove an enrolled speaker |
| `POST /v1/audio/speaker-identify` | Best-matching enrolled speaker, or `unknown` |
| `POST /v1/audio/speaker-verify` | Verify a clip against one enrolled speaker |
| `POST /v1/audio/stems` | Stem separation (Demucs) — vocals/instrumental or 4-stem |
| `POST /v1/audio/cleanup` | Denoise / normalise / de-hum / de-click (DeepFilterNet) |
| `POST /v1/audio/generate` | Music/SFX generation |
| `GET /v1/audio/progress` | Audio-generation progress |
| `POST /v1/audio/clone` | Voice cloning TTS (F5-TTS) |
| `POST /v1/audio/convert` | Voice conversion / SVC (Seed-VC) |
| `GET /v1/audio/voices` | List saved voice profiles |
| `POST /v1/audio/voices` | Save a voice profile |
| `GET /v1/audio/voices/{name}` | Get a specific voice profile |
| `PATCH /v1/audio/voices/{name}` | Update a voice profile |
| `POST /v1/audio/voices/extract` | Extract voice profile from audio/video |
| `DELETE /v1/audio/voices/{name}` | Delete a voice profile |

### Character Profiles

| Endpoint | Description |
|---|---|
| `GET /v1/characters` | List all character profiles |
| `POST /v1/characters` | Save a character profile from images/videos |
| `POST /v1/characters/extract` | Extract a profile from source images/videos (face-crop + sharpness ranking) |
| `POST /v1/characters/generate` | Generate reference images from a text prompt and save as profile |
| `GET /v1/characters/{name}` | Get a character profile with images |
| `PATCH /v1/characters/{name}` | Update description or add/remove reference images |
| `DELETE /v1/characters/{name}` | Delete a character profile |

### Environment Profiles

| Endpoint | Description |
|---|---|
| `GET /v1/environments` | List all environment profiles |
| `POST /v1/environments` | Save an environment profile from images/videos |
| `POST /v1/environments/extract` | Extract a profile from source images/videos (sharpness ranking, full image) |
| `POST /v1/environments/generate` | Generate reference images from a scene description and save as profile |
| `GET /v1/environments/{name}` | Get an environment profile with images |
| `PATCH /v1/environments/{name}` | Update description or add/remove reference images |
| `DELETE /v1/environments/{name}` | Delete an environment profile |

### OCR

| Endpoint | Description |
|---|---|
| `POST /v1/ocr` | Transcribe one image or PDF — `engine`, `dpi`, `structured`, `schema`, `detect` |
| `POST /v1/ocr/batch` | Same, for many files; per-file errors are reported inline |
| `GET \| POST /v1/ocr/schemas` | List or create structured-extraction schemas |
| `GET \| PUT \| DELETE /v1/ocr/schemas/{name}` | Read, replace or delete a schema |

### LoRA Training

| Endpoint | Description |
|---|---|
| `POST /v1/loras/train` | Train a LoRA (image or video target); `wait=false` for a job id |
| `GET /v1/loras/progress` | Progress by `job`, by `session`, or a global snapshot |
| `POST /v1/loras/upload` | Upload a LoRA into the content-addressed blob store |
| `GET /v1/loras/blob/{hash}` | Check whether a blob is already stored |
| `GET \| DELETE /v1/loras/{name}` | Read or delete a registered LoRA |
| `GET /v1/loras` | List registered LoRAs |

### 3D Generation

| Endpoint | Description |
|---|---|
| `POST /v1/3d/generate` | Text or image → 3D model (GLB) |

### Archive

| Endpoint | Description |
|---|---|
| `GET /v1/archive` | List all archived generated files |
| `DELETE /v1/archive/{filename}` | Delete an archived file |

### Pipelines

| Endpoint | Description |
|---|---|
| `POST /v1/pipelines/image-to-video` | Image gen → video animation |
| `POST /v1/pipelines/video-dub` | Full video dubbing pipeline |
| `POST /v1/pipelines/story` | LLM → images → video → TTS |
| `POST /v1/pipelines/audio-dub` | Audio/video dub with voice cloning |
| `POST /v1/pipelines/audio-understand` | Transcribe audio, then answer a question about it with a text model |
| `POST /v1/pipelines/audio-music-dub` | Dub a song into another language over its original backing track |
| `GET /v1/pipelines/custom` | List custom pipelines |
| `POST /v1/pipelines/custom` | Create custom pipeline |
| `PUT /v1/pipelines/custom/{id}` | Update custom pipeline |
| `DELETE /v1/pipelines/custom/{id}` | Delete custom pipeline |
| `POST /v1/pipelines/custom/{id}/run` | Run a saved custom pipeline |
| `POST /v1/pipelines/run` | Run an inline pipeline definition |
| `GET /v1/pipelines/step-types` | List available step types |

### Character & Environment Profile Usage

Profiles are selected by name in any image or video generation request. Up to 6 of each type can be combined:

```json
{
  "model": "wan-model",
  "prompt": "Alice and Bob walking in a forest",
  "character_profiles": ["Alice", "Bob"],
  "character_strength": 0.8,
  "environment_profiles": ["forest-summer"],
  "environment_strength": 0.6
}
```

### Multi-Character Dialog

Add spoken dialog to a video generation. Each line specifies a character, a voice, and text. Lines are TTS-synthesised, timed, mixed, and lip-synced:

```json
{
  "model": "wan-model",
  "prompt": "Two characters having a conversation",
  "character_profiles": ["Alice", "Bob"],
  "lip_sync_method": "wav2lip",
  "dialogs": [
    {
      "character": "Alice",
      "voice": "alice-voice-profile",
      "text": "Hello, how are you today?",
      "lip_sync": true
    },
    {
      "character": "Bob",
      "voice": "en-US-GuyNeural",
      "text": "I'm doing great, thanks for asking!",
      "lip_sync": true,
      "start_time": 3.5
    }
  ]
}
```

Fields per dialog line: `character` (profile name for lip-sync face selection), `voice` (saved voice profile name or TTS voice ID), `text`, `start_time` (seconds; omit for sequential auto-timing), `lip_sync` (bool), `lang`, `speed`.

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

Available step types: `text_gen`, `image_gen`, `image_edit`, `image_inpaint`, `image_upscale`, `image_deblur`, `image_unpix`, `image_outfit`, `image_faceswap`, `image_to3d`, `video_gen`, `video_upscale`, `video_sub`, `video_interp`, `video_dub`, `video_to3d`, `tts`, `audio_gen`, `voice_clone`, `voice_convert`, `ocr`.

Call `GET /v1/pipelines/step-types` for the authoritative list on your build.

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

## Developer

**Stefy Lanza** &lt;stefy@nexlab.net&gt;

## License

GNU General Public License v3.0 — see [LICENSE.md](LICENSE.md).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

## Donations

If you find CoderAI useful, consider supporting development:

- **Bitcoin (BTC):** `bc1qcpt2uutqkz4456j5r78rjm3gwq03h5fpwmcc5u`
- **Ethereum (ETH):** `0xdA6dAb526515b5cb556d20269207D43fcc760E51`

## Contributing

Merge requests welcome.

## Acknowledgments

CoderAI stands on the shoulders of remarkable open-source work. Special, heartfelt
thanks to the developers of the native inference engines that let it punch far above
its hardware — running models that otherwise simply couldn't run on a single GPU:

- **[colibri](https://github.com/JustVugg/colibri)** by **JustVugg** — a brilliant
  pure-C multi-family MoE engine that streams **GLM-5.2, DeepSeek-V4 and Kimi-K3**
  across VRAM/RAM/disk to run them on a single consumer GPU (or CPU).
- **[ds4 / DwarfStar](https://github.com/antirez/ds4)** by **Salvatore Sanfilippo
  (antirez)** — a superb from-scratch **DeepSeek-V4** inference engine.
- **[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)** by
  **Fareed Khan** — a portable-C engine that runs **Kimi-K3 (2.78T params)** on CPU by
  streaming the dense trunk + routed experts from disk in as little as ~8 GB RAM.
- **[ktransformers](https://github.com/kvcache-ai/ktransformers)** by **KVCache.AI** —
  a CPU+GPU heterogeneous engine (served via SGLang) for large MoE models
  (DeepSeek / Kimi / Qwen / GLM / MiniMax).
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** &
  **[whisper.cpp](https://github.com/ggml-org/whisper.cpp)** by **Georgi Gerganov**
  and contributors — the foundational GGUF LLM inference and Whisper STT engines.
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** (Baidu),
  **[docTR](https://github.com/mindee/doctr)** (Mindee) and
  **[Surya](https://github.com/VikParuchuri/surya)** — the dedicated OCR engines behind
  the `/v1/ocr` document-transcription subsystem (text + layout + structured extraction).

And the libraries, models and research CoderAI builds on:

- [FastAPI](https://fastapi.tiangolo.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) & [Diffusers](https://github.com/huggingface/diffusers) — NVIDIA text/image backends
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — Vulkan/CUDA GGUF text backend
- [stable-diffusion-cpp-python](https://github.com/william-murray1204/stable-diffusion-cpp-python) — GGUF image backend
- [InsightFace](https://github.com/deepinsight/insightface) — face swap
- [F5-TTS](https://github.com/SWivid/F5-TTS) — voice cloning
- [Seed-VC](https://github.com/Plachta/Seed-VC) — singing voice conversion
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — image/video upscaling
- [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) — PDF rasterisation for OCR; [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — stamp/signature detection
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) — audio-driven lip sync
- [SadTalker](https://github.com/OpenTalker/SadTalker) — talking head / lip sync generation
- Visual place recognition & geolocation research — [EigenPlaces](https://github.com/gmberton/EigenPlaces) (Gabriele Berton et al.), [DINOv2-SALAD](https://github.com/serizba/salad) (Sergio Izquierdo, Javier Civera), [GeoCLIP](https://github.com/VicenteVivan/geo-clip) (Vicente Vivanco et al.), [DINOv2](https://github.com/facebookresearch/dinov2) (Meta AI)
