# Expressive TTS (emotion / delivery)

The video editor shows **Emotion** and **Delivery** dropdowns whenever the
configured TTS model advertises them (`codai/api/tts_backends.py`:
`family_emotions` / `family_styles`). Two engines support expressive control.

## Bark — in-stack, no extra deps

Works with the server's current `transformers`. Configure a Bark model as the
TTS model, e.g. `--tts-model suno/bark` (or `suno/bark-small`).

- **Delivery**: `normal`, `whispering` (`[whispers] …`), `singing` (`♪ … ♪`),
  `emphasis` (UPPERCASE).
- **Emotion**: inserts a matching non-verbal cue — `laughter`→`[laughs]`,
  `sigh`→`[sighs]`, `gasp`→`[gasps]`.
- **Voice**: a Bark preset like `v2/en_speaker_6`. The editor's Kokoro voice ids
  don't apply and fall back to the default preset (set `voice_preset` in the
  model config to change it). Speed isn't controllable in Bark.

## Parler — fully managed by coderai (no setup)

`parler-tts` pins an old `transformers`/`tokenizers`/`huggingface-hub` that
**conflict with this server** — never `pip install` it into the coderai venv.
coderai handles this for you: just use a Parler model as the TTS model
(e.g. `parler-tts/parler-tts-mini-multilingual`). The worker is launched lazily —
only when a request for that model actually arrives — and shut down when the
model is evicted, exactly like loading/unloading any other model. On first use it

1. creates a dedicated venv at `~/.coderai/parler_venv`
   (override with `CODERAI_PARLER_VENV`), built `--system-site-packages` so the
   base torch/numpy are reused and only the conflicting packages land in it;
2. `pip install`s parler-tts there;
3. launches `tools/parler_tts_service.py` in that venv on a local port, pointing
   `HF_HUB_CACHE` at coderai's own cache and forcing **offline mode**
   (`HF_HUB_OFFLINE=1`) so it loads strictly the model you **already downloaded
   via the model interface** — the worker never downloads anything itself;
4. health-checks it and routes synthesis to it.

The worker is owned by `codai/api/parler_worker.py`; the backend's `cleanup()`
calls `stop_service()`, so the model manager's normal eviction tears the process
down. The first request blocks while the venv builds, then it's cached.

If the model isn't in coderai's cache, the worker fails fast with a clear error
("download '<model>' from the model interface first") instead of fetching it.
Download the Parler model through the normal HF download UI first.

The editor's **Emotion**/**Delivery** dropdowns drive it: coderai POSTs
`{text, voice, speed, emotion, style}` to the worker, which maps them into a
natural-language delivery description (whisper / shout / monotone / expressive +
emotion + pace). A fixed `description` in the model config overrides the
auto-built one. An explicit `service_url` in the config bypasses management and
talks to an externally-run service instead.

> The model must still be in the server's allowed-models registry to be
> selectable — that's the only configuration; the worker itself needs none.
