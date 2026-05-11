# TODO: Multi-backend support for face swap, deblur, and unpixelate

We are working on **CoderAI** (`/storage/coderai`), a local AI inference server with a web studio UI. The codebase is Python/FastAPI on the backend and vanilla JS/HTML on the frontend (`codai/admin/templates/chat.html`).

We want to expand the **face swap** feature (`codai/api/faceswap.py`) to support multiple backends and models, so users can choose the best option for their hardware and use case. The same philosophy should apply to **deblur** (`/v1/images/deblur`) and **unpixelate/upscale** (`/v1/images/unpixelate`) — currently both use single hardcoded approaches (OpenCV Wiener and Real-ESRGAN respectively), but users should be able to pick between alternatives there too.

**The goal is maximum user choice** across all three features:

For **face swap**, consider offering at minimum:
- The current insightface + inswapper_128 path
- SimSwap or other ONNX-compatible swapper models
- Optional post-processing enhancers (CodeFormer, GFPGAN, or similar) that can be toggled on top of any swapper
- A facefusion-based path if it simplifies supporting multiple models via subprocess

For **deblur**, alternatives to pure OpenCV signal processing could include ML-based blind deblurring models (e.g. NAFNet, Restormer, or similar lightweight restoration networks).

For **unpixelate/upscale**, alternatives to Real-ESRGAN could include ESRGAN variants, SwinIR, HAT, or any other super-resolution model the user has downloaded.

The API should accept a `model` or `backend` parameter so the caller can select which implementation to use. Missing models should fail gracefully with a clear error rather than silently falling back. The web UI should expose the available options (discovered at runtime based on what's installed/downloaded) as a selector in the relevant panels.

Read the existing implementations before proposing changes to understand the current structure, file layout, and how the UI communicates with the backend.
