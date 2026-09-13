# Reference projects

Outside projects we've studied, what each does that we don't, and what was taken
from them. Kept as a record because several of these are meant to seed **promo
tools** built on coderai — the point is to know what exists before rebuilding it.

Each entry says plainly whether it was *evaluated* (read the code / the paper) or
only *noted* (a link to look at later), so nothing here reads as a conclusion it
hasn't earned.

---

## OpenVoice — myshell-ai/OpenVoice · MIT · EVALUATED 2026-09-13

https://github.com/myshell-ai/OpenVoice · paper: arXiv:2312.01479

Instant voice cloning. Read from `openvoice/api.py`, the README and the paper
abstract.

**Architecture.** Two stages rather than one. `BaseSpeakerTTS.tts()` speaks the
text in a *base* voice with the chosen language and style (V2 uses **MeloTTS**:
English, Spanish, French, Chinese, Japanese, Korean). Then
`ToneColorConverter.convert(source, src_se, tgt_se, tau=0.3)` re-paints only the
timbre. The target embedding comes from `extract_se(audio_paths)` — **audio only,
no transcript anywhere in the pipeline** — and it averages across several clips.
`add_watermark()` / `detect_watermark()` mark the output.

**Why the split matters.** Language and delivery belong to the base TTS, timbre to
the converter, so cross-lingual cloning needs no multilingual data for the target
speaker. The cost is prosody: it copies how someone *sounds*, not how they *speak*.

**How it compares to what we run.** We clone with F5-TTS (in-context: reference
audio + its transcript, one model, so accent and prosody come along) and convert
with seed-vc. At the model level we were already ahead — F5-TTS over MeloTTS for
naturalness, XTTS-v2 over their base for language coverage (17 vs 6), seed-vc over
their tone-colour converter. What we lacked was the *wiring*, plus watermarking.

**Taken from it** (commits `45cc730`, and the engine work that followed):
- `engine` on `/v1/audio/clone`: `f5` | `xtts` | `chain` | `melotts`, plus `auto`,
  which picks a transcript-free engine when the profile has no transcript or the
  target language differs from the reference.
- `chain` = base TTS → seed-vc. That IS OpenVoice's architecture, with seed-vc in
  place of its converter and any of our TTS engines as the base.
- MeloTTS itself, as an isolated-venv worker (`requirements-melotts.txt`,
  `tools/melotts_service.py`, `codai/api/melotts_worker.py`) — so the exact
  OpenVoice V2 pairing is reproducible here, and MeloTTS is also a TTS engine in
  its own right.
- Multi-clip voice profiles: `PATCH /v1/audio/voices/{name}` takes `add_clips` /
  `remove_clips`, and a clone prompts on the cleanest take (scored on clipping,
  level and crest factor — deliberately not "loudest wins", since a clipped take
  has the highest RMS and is the worst prompt).
- Watermarking, via **AudioSeal** (MIT, Meta) rather than their silentcipher:
  on by default, `CODERAI_AUDIO_WATERMARK=0` or `watermark=false` to disable,
  detection at `POST /v1/audio/watermark/detect`.

**Not taken.** Their base TTS as a default (F5/XTTS are better) and their
converter (seed-vc is newer and stronger).

**Still open.** True embedding *averaging* across clips — we pick the best clip
rather than average, because neither XTTS nor seed-vc exposes an averaged speaker
embedding. Worth revisiting if we ever store our own embeddings on the profile.

---

## MoneyPrinterTurbo — harry0703/MoneyPrinterTurbo · MIT · EVALUATED 2026-09-13 (README level)

https://github.com/harry0703/MoneyPrinterTurbo/blob/main/README-en.md
MIT, Python, ~123k stars, actively pushed. Read at README level — not the source.

**Its pipeline**, one topic in, a finished short out:

    topic -> LLM script -> LLM search terms -> stock/AI footage -> TTS voiceover
          -> subtitles -> background music -> assemble 9:16 / 16:9 / 1:1 -> publish

Everything heavy is somebody's cloud: a dozen LLM vendors for the script, Pexels /
Pixabay / Coverr for footage, Edge/Azure/ElevenLabs/Fish for voice, with Whisper
run locally for subtitles. `config.toml` is mostly API keys. Streamlit UI, FastAPI
service with Swagger, a CLI with batch manifests, and direct upload to TikTok,
Instagram and YouTube Shorts.

**What it has that we don't:**

1. **A planner.** One topic becomes a script, and the script becomes per-scene
   search terms. Our video tools all start from prompts *you* write —
   `gen_township_fighters.py` is the only one that asks an LLM to plan anything.
   We have local LLMs on tap, so this is the cheapest gap to close and the one the
   promo tools actually need.
2. **Stock footage.** Pexels/Pixabay/Coverr. We generate every frame, which is
   minutes of GPU per clip; for promo B-roll a stock lookup is seconds and often
   looks better. Nothing in coderai sources stock media today.
3. **Vertical as a first-class format.** 9:16 / 1:1 presets throughout. Our tools
   default to 16:9 (768x432) and `video_editor.py`'s "vertical" control is a zoom
   axis, not an aspect preset.
4. **Publishing.** Direct upload to TikTok / Instagram / YouTube Shorts. We have
   `tools/township_upload.py` for one site and nothing social.
5. Batch manifests and task history as a first-class CLI concern.

**What we have that it doesn't:** all of it local and key-free; character identity
(profiles, IP-Adapter, per-character LoRA); voice cloning of a *specific* person
plus conversion and lip sync; dubbing with diarisation and speaker ID; MiniMax H3
with natively synchronised audio. Its quality ceiling is whatever its vendors give
it; ours is the models we run.

Already covered on our side: subtitles (`generate_subtitles`, `burn_subtitles`,
`subtitle_style` = default/karaoke/minimal), background music with volume control
(`videogen.py`), and assembly/concat (`video_editor.py`).

**Proposal for the promo tool** — take its *shape*, not its dependencies:

    topic -> our LLM writes script + scene plan -> per scene: stock lookup OR
    generate (character studio / H3 / Wan) -> our TTS or a cloned voice ->
    our subtitles -> assemble 9:16 -> review -> publish

Which needs, in order: an LLM planner endpoint or tool step; an optional stock
provider (Pexels/Pixabay keys, degrading to generation when absent); aspect presets
including 9:16; and publishing adapters. The first two are where the leverage is —
the rest we already have.

## Playtime-AI (Hugging Face) — EVALUATED 2026-09-10

https://huggingface.co/Playtime-AI

72 LoRA repos, catalogued via the HF API: 33 for LTX-2.3, 24 for MiniMax H3,
7 for Krea 2, plus Wan/Z-Image collections. Studied for **how** the LoRAs are
trained, which is not secret — two public recipes match the catalogue:

- **Stills path** (ai-toolkit, 12 GB VRAM): ~32 face crops, captions reduced to
  `(trigger), 1girl` so identity collapses into the trigger token, 512×512,
  rank/alpha 16, LR 1e-4 adamw8bit, 1000 steps, INT8 quantised transformer with
  CPU offload, `flowmatch` + `timestep_type: shift`, `audio_loss_multiplier: 0.0`,
  ~6-7 h per LoRA.
- **Clips path** (fal's H3 trainer): 50-200 clips of 3-15 s at exactly 24.000 fps,
  bucketed to multiples of 32, `frames % 17 == 5`, rank 16, 1500-5000 steps, audio
  kept. Their finding: training resolution matters more than rank or step count.

Recorded because the *technique* is what we need for per-character identity LoRAs
(`codai/api/loras.py`, the character studio's `train` step). The catalogue itself
pairs named-celebrity likenesses with explicit concept LoRAs; we build the generic
trainer for characters we have rights to and don't reproduce that.

Related: [minimax-h3.md](minimax-h3.md) for the H3 facts gathered at the same time.
