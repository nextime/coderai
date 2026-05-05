# Multimodal Capability Indicators - UI Examples

## Search Results (HuggingFace)

### Before
```
stable-diffusion-xl-base-1.0
  text-to-image  ↓ 2.5M  ♥ 15k
  [Info] [▾ Files] [Download]
```

### After
```
stable-diffusion-xl-base-1.0
  text-to-image  [Text] [T2I] [I2I] [Inpaint]  ↓ 2.5M  ♥ 15k
  [Info] [▾ Files] [Download]
```

## Local Models (HuggingFace Cache)

### Before
| Model | Size | Files | Config | Actions |
|-------|------|-------|--------|---------|
| meta-llama/Llama-2-7b-chat-hf | 13.5 GB | 42 | enabled | [Load now] [Configure] [Remove] [Delete] |

### After
| Model | Size | Files | Capabilities | Config | Actions |
|-------|------|-------|--------------|--------|---------|
| meta-llama/Llama-2-7b-chat-hf | 13.5 GB | 42 | [Text] | enabled | [Load now] [Configure] [Remove] [Delete] |
| stabilityai/stable-diffusion-xl-base-1.0 | 6.9 GB | 28 | [Text] [T2I] [I2I] [Inpaint] | enabled | [Load now] [Configure] [Remove] [Delete] |
| llava-hf/llava-v1.5-7b-hf | 13.1 GB | 35 | [Text] [I2T] | enabled | [Load now] [Configure] [Remove] [Delete] |

## Local Models (GGUF Cache)

### Before
| File | Size | Config | Actions |
|------|------|--------|---------|
| llama-2-7b-chat.Q4_K_M.gguf | 4.1 GB | enabled | [Load now] [Configure] [Remove] [Delete] |

### After
| File | Size | Capabilities | Config | Actions |
|------|------|--------------|--------|---------|
| llama-2-7b-chat.Q4_K_M.gguf | 4.1 GB | [Text] | enabled | [Load now] [Configure] [Remove] [Delete] |
| stable-diffusion-xl.Q4_K_M.gguf | 3.8 GB | [Text] [T2I] [I2I] | enabled | [Load now] [Configure] [Remove] [Delete] |

## Chat Sidebar

### Before
```
[LLM] llama-2-7b-chat
[IMG] stable-diffusion-xl
[VLM] llava-v1.5-7b
```

### After
```
[LLM] llama-2-7b-chat
[IMG] stable-diffusion-xl T+I+I
[VLM] llava-v1.5-7b T+V
```

## Search Filters

### New Capability Chips (in addition to existing filters)
```
Cap: [Text] [T2I] [I2T] [T2V] [I2V] [T2A] [STT] [TTS] [Embed] [Tool calling] [Vision] [Reasoning] [Code] [Multilingual] [Roleplay] [Math]
```

### Usage
- Click chips to filter models by capability
- Multiple chips = AND filter (model must have all selected capabilities)
- Works with existing filters (size, quant, pipeline, etc.)

## Capability Badge Legend

| Badge | Full Name | Description |
|-------|-----------|-------------|
| Text | Text Generation | LLM chat/completion |
| T2I | Text-to-Image | Generate images from text |
| I2T | Image-to-Text | Vision models, VQA, captioning |
| I2I | Image-to-Image | Transform/edit images |
| T2V | Text-to-Video | Generate videos from text |
| I2V | Image-to-Video | Animate images into videos |
| V2V | Video-to-Video | Transform/edit videos |
| T2A | Text-to-Audio | Generate music/audio from text |
| A2A | Audio-to-Audio | Transform/edit audio |
| STT | Speech-to-Text | Transcribe audio to text |
| TTS | Text-to-Speech | Synthesize speech from text |
| Embed | Embeddings | Generate text/image embeddings |
| Inpaint | Inpainting | Fill masked regions in images |
| ControlNet | ControlNet | Guided image generation |
| Depth | Depth Estimation | Estimate depth from images |
| Segment | Image Segmentation | Segment objects in images |
| Upscale | Image Upscaling | Enhance image resolution |
| Face | Face Restoration | Restore/enhance faces |
| Detect | Object Detection | Detect objects in images |
| Interp | Video Interpolation | Generate intermediate frames |
| V-Upscale | Video Upscaling | Enhance video resolution |
| Lip-sync | Lip Sync | Sync lips to audio |
| Subs | Subtitle Generation | Generate subtitles from audio |
| Dub | Video Dubbing | Translate and dub videos |

## Example Searches

### Find Text-to-Image Models
1. Go to Models → Find on HuggingFace
2. Click "T2I" chip
3. Results show only T2I models (Stable Diffusion, FLUX, etc.)

### Find Vision LLMs (Multimodal)
1. Click both "Text" and "I2T" chips
2. Results show models that can do both text generation and image understanding (LLaVA, Qwen-VL, etc.)

### Find Text-to-Video Models
1. Click "T2V" chip
2. Results show T2V models (CogVideoX, LTX-Video, etc.)

### Find Models with Multiple Capabilities
1. Click multiple capability chips
2. Only models with ALL selected capabilities are shown
3. Great for finding truly multimodal models
