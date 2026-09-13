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

---

# TODO: review MoneyPrinterTurbo for ideas worth borrowing

https://github.com/harry0703/MoneyPrinterTurbo/blob/main/README-en.md

An automated short-video generation stack (script → stock/generated footage →
subtitles → TTS → assembled clip). Worth reading against what we already have in
`tools/videogen.py`, `tools/video_editor.py` and `tools/character_studio.py`:
where does it do something we don't, and is any of it worth lifting? Points of
comparison: how it plans a video from a single prompt, how it sources and cuts
footage, subtitle generation/burn-in, the TTS/voice pipeline, and how its web UI
is organised. Not yet evaluated — this entry is only a reminder to look.

# TODO: OpenVoice features missing from our voice cloning

https://github.com/myshell-ai/OpenVoice (MIT). We clone with F5-TTS
(`codai/api/voice_clone.py`, in-context: reference audio + its transcript) and
convert with seed-vc (`/v1/audio/convert`). OpenVoice splits the job — a base TTS
speaks the text, then a tone-colour converter re-paints the timbre — which buys
four things we lack, in rough order of value:

1. **Transcript-free cloning.** `extract_se()` works from audio alone. Today a
   profile without a transcript cannot clone at all (`video.py` falls back to a
   generic voice), and our Whisper `base` fallback is the least accurate tier — a
   wrong `ref_text` visibly degrades F5. Store a tone-colour embedding on the
   profile at extract time and reuse it per line.
2. **Cross-lingual voice.** The base TTS owns the language, so the accent doesn't
   follow the reference. `tools/video_dubber.py` currently translates the text but
   keeps the reference speaker's accent.
3. **Multi-clip embedding averaging.** `extract_se()` averages several references;
   we store exactly one WAV per profile. Cheapest quality win available, and it
   fits the existing `PATCH /v1/audio/voices/{name}` shape.
4. **Watermarking.** `add_watermark()` / `detect_watermark()` on synthesised
   audio. We generate cloned speech of real people at scale and mark none of it.

Integration shape: OpenVoice pins older torch/librosa, so it belongs in an
isolated venv worker like pyannote / NeMo-Canary / H3 —
`requirements-openvoice.txt` + `tools/openvoice_service.py` +
`codai/api/openvoice_worker.py`. Then `/v1/audio/clone` gains an `engine` field
(`f5` default, `openvoice`), auto-selecting OpenVoice when a profile has no
transcript or the target language differs from the reference. Its converter sits
beside seed-vc; both are audio→audio timbre transfer.

Already done (2026-09-13, commit 45cc730): F5-TTS engine caching and reference
trimming to 15 s — those were our own bugs, not missing OpenVoice features.
