# Composing a finished video — `POST /v1/video/compose`

CoderAI can generate every *piece* of a short video: narration
(`/v1/audio/speech`, `/v1/audio/clone`), stills (`/v1/images/generations`),
clips (`/v1/video/generations`), music (`/v1/audio/generate`). Nothing turned the
pieces into a reel, and a client on another machine cannot do it: assembly needs
ffmpeg and two facts only this server has — how long the narration it just
synthesised actually is, and when each word falls.

So composition happens here. You send an ordered list of **scenes**; you get back
an uploadable MP4, and optionally a thumbnail, subtitles and the narration track.

```
POST   /v1/video/compose              → 202 {"id": "cmp_…", "status": "queued"}
GET    /v1/video/compose/{id}         → status, stage, progress, warnings, result
POST   /v1/video/compose/{id}/cancel  → stop it (the ffmpeg child is killed)
GET    /v1/video/compose              → every job this front still remembers
POST   /v1/files/upload               → store your own footage/music once
GET    /v1/files/blob/{sha256}        → "do you already have these bytes?"
```

Everything is bearer-authenticated like the rest of the API, media fields accept
the usual four forms (a URL, a `/v1/files/...` path, a `data:` URI, raw base64),
and every artefact is served from `/v1/files/`.

## The shape of a request

```jsonc
{
  "canvas": { "width": 1080, "height": 1920, "fps": 30 },

  // Voice-over defaults for every scene that gives `text` without `audio`.
  "voice": { "engine": "tts", "model": "kokoro", "voice": "af_sarah",
             "language": "en-us", "speed": 1.0 },

  "scenes": [
    {
      "text": "Octopuses have three hearts and blue blood.",  // narration + caption
      "audio": null,          // pre-rendered narration instead; overrides TTS
      "min_duration": 2.0,    // floor, seconds
      "padding": 0.15,        // silence after the narration
      "audio_mix": "voice",   // voice | clip | both  (see "The clip's own sound")
      "clip_volume": 0.35,    // the clip's level under the voice, for "both"
      "speakers": [],         // two characters talking (see "A dialogue")
      "visuals": [            // split evenly across the scene, in order
        {"type": "video", "src": "https://…/clip.mp4", "trim_start": 0.0,
         "keep_audio": false,   // keep THIS clip's own sound (see below)
         "volume": null},       // its level under the voice; else clip_volume
        {"type": "image", "src": "/v1/files/abc.png",
         "ken_burns": {"zoom_from": 1.0, "zoom_to": 1.2, "pan": "auto"}},
        {"type": "gradient", "colors": ["#6D28D9", "#DB2777"]},
        {"type": "color", "color": "#111827"}
      ]
    }
  ],

  "visual_fit": "cover",                                   // cover | contain | blur_pad
  "transition": {"type": "none", "duration": 0.3},         // none|fade|crossfade|slide|wipe|dissolve
  "music": {"src": "/v1/files/track.mp3", "volume": 0.14, "loop": true,
            "duck": true, "fade_in": 1.0, "fade_out": 2.0},
  "captions": {"enabled": true, "timing": "auto", "preset": "karaoke",
               "font": "Montserrat", "position": "center",
               "max_words_per_line": 4, "max_lines": 2},
  "overlays": [{"type": "text", "text": "made with motus",
                "position": "bottom-right", "opacity": 0.6, "size": 0.025},
               {"type": "image", "src": "https://…/logo.png",
                "position": "top-left", "width": 0.15}],
  "thumbnail": {"at": 0.25, "text": "3 hearts?!"},
  "output": {"format": "mp4", "video_codec": "h264", "crf": 20,
             "audio_bitrate": "192k"},
  "outputs": ["video", "thumbnail", "srt", "vtt", "narration"],
  "async": true
}
```

A scene needs either narration (`text`, `audio` or `speakers`) or an explicit
`"duration"`, and at least one visual. Anything else has a default. A dialogue
scene needs neither `text` nor visuals: its lines are the narration and its
characters are the picture.

## The clip's own sound

Every visual is rendered silent, which is right for a stock shot under a
voice-over and wrong for footage whose sound **is** the content — a crowd, a
product demo, a piece to camera that was filmed with its audio. Two fields:

```jsonc
{
  "audio_mix": "both",             // voice | clip | both
  "clip_volume": 0.35,
  "visuals": [{"type": "video", "src": "/v1/files/demo.mp4", "keep_audio": true}]
}
```

* `"voice"` (the default) — the narration only. Unchanged from before; a
  request written without these fields renders exactly as it did.
* `"clip"` — the clip's sound **instead of** narration. No TTS call is made for
  this scene, and the scene lasts as long as the clip: cutting it at a
  narration's length would truncate the content itself. An explicit `duration`
  still wins.
* `"both"` — the narration over the clip, with the clip ducked while the voice
  is speaking and brought back up between lines.

The clip's sound is folded into the scene's narration track, so the music bed
ducks under it exactly as it ducks under a voice. With one video visual in the
scene, `keep_audio` is optional — asking for the clip's sound and not saying
which clip can only mean the one clip. With two, mark the one you mean.

Silent footage with `keep_audio: true` is a **warning, not a failure**: the
scene renders with its narration (or in silence for `"clip"`) and the job says
so. Only the first marked clip in a scene contributes; a second clip's sound
belongs in its own scene, where it also gets its own duration.

## A dialogue — two characters in one scene

`presenter` is one character talking to camera, which is a monologue. For an
exchange, give the scene `speakers`:

```jsonc
{
  "speakers": [
    {"character": "alice", "text": "Did you see the new build?",
     "voice": {"model": "kokoro", "voice": "af_sarah", "language": "en-gb"},
     "layout": "left", "gap": 0.25},
    {"character": "bob", "text": "It shipped this morning.",
     "voice": {"model": "kokoro", "voice": "am_adam"}, "layout": "right"}
  ]
}
```

The lines are spoken in order, each in its own voice, and each character is
lip-synced in their own region of the frame. A character may speak several
times; their lines group into one region in first-speaking order.

* `layout: "left" | "right"` (the default) — every character stays on screen in
  their own column for the whole scene. A listener's reaction is half of a
  dialogue, so this is the default.
* `layout: "full"` — the frame cuts to whoever is speaking.

All the speakers in one scene must use columns or all must use `full`; mixing
them is a 400, because a character cannot both hold a column and be off screen.

Each character needs a `character` profile or an `image` (or a request-level
`presenter` to inherit a portrait from). A profile name that does not exist is a
400 **before** the render starts, so a typo does not cost you the narration.

Captions follow the speaker: a caption never mixes two characters' words, and
each event carries a `speaker` so the `.srt`/`.vtt` sidecars keep the
attribution.

One lip-sync run happens per **character**, not per line: each character is
given a track holding only their own lines, so their mouth moves on their words
and stays closed while the other one talks.

A dialogue replaces `text`, `voice` and `presenter` for that scene. Keep an
exchange to a beat — a longer conversation is several scenes, which is also how
it gets its own visuals.

## What the server decides

**Timing.** A scene lasts *narration + padding*, at least `min_duration`; with no
narration it lasts its `duration`. The scene's visuals split that time equally,
in order. This is the whole reason composition is server-side: the client would
have to probe the audio it cannot see.

**Visuals.** Each is scaled to the canvas per `visual_fit` — `cover` crops to
fill (what a reel wants), `contain` letterboxes, `blur_pad` letterboxes onto a
blurred blow-up of the same frame — normalised to `canvas.fps` and `yuv420p`. A
video shorter than its slot loops; a longer one is trimmed from `trim_start`. A
still gets a slow Ken Burns move, alternating direction per image when
`pan: "auto"`. **A visual that cannot be fetched or decoded never fails the
job**: it becomes a gradient and a line in `warnings`.

**Audio.** Narration is synthesised per scene (so the timing is exact), padded,
concatenated and loudness-normalised to about −16 LUFS, which is why two reels
made a week apart are as loud as each other. Music is looped or trimmed to the
video's length, faded in and out, and — with `duck: true` — pushed under the
narration with a sidechain compressor. The mix is limited to −0.3 dBFS so no
platform re-encodes it for loudness.

**Captions.** Burned in, and **the text shown is the text you sent**. Speech
recognition is used only for *when* each word is spoken, never for what it says,
so a mis-heard word cannot reach the screen. `timing`:

* `auto` (default) — word timestamps from a speech model when one is configured,
  estimated from word length otherwise (and a warning saying so);
* `stt` — require the model; `501` at submission time if there is none;
* `estimate` — never load a speech model.

Not every speech backend reports *per-word* timings: the HF pipelines
(`wav2vec2`, `whisper_hf`, CrisperWhisper), `vosk` and plain faster-whisper do
— measured, on one install: CrisperWhisper 14 words for a 4.3 s Italian line,
wav2vec2 the same 14 — while a `whisper-server` (whisper.cpp) model and the
NeMo worker behind canary return text only. `auto` therefore picks a model that can, whatever it is called, and
`captions.stt_model` overrides the pick. A model that declares
`languages` refuses the rest with a `400`, so the pick also skips a
per-language model when the narration is in a language it does not declare. A model that returns no timings is not
an error — the render continues with estimated ones — but it is reported in the
job's `warnings`, because the captions will drift against the voice.

Presets `karaoke`, `bold`, `neon` highlight the active word; `classic`, `boxed`,
`minimal` draw whole lines. Lines break at `max_words_per_line` and stay inside a
safe area of 10% top/bottom and 6% at the sides, clear of the TikTok/Reels/Shorts
furniture. The returned SRT/VTT carry the same text and the same timings.

**Output.** H.264 High + AAC 48 kHz stereo, `yuv420p`, `+faststart` — uploadable
to YouTube, TikTok, Instagram and Facebook as it is.

## Polling

```jsonc
{
  "id": "cmp_8f2c1a",
  "status": "running",              // queued | running | done | failed | cancelled
  "stage": "narration",             // narration | captions | visuals | music | render | thumbnail
  "progress": 42.0,
  "message": "scene 3/8: narration",
  "warnings": ["scene 2 visual 1: download failed, used gradient"],
  "error": null,
  "result": null
}
```

When `status` is `done`:

```jsonc
"result": {
  "video":     {"url": "https://…/v1/files/cmp_8f2c1a.mp4", "path": "/v1/files/cmp_8f2c1a.mp4",
                "duration": 47.3, "width": 1080, "height": 1920, "size": 18234567},
  "thumbnail": {"url": "…", "path": "/v1/files/cmp_8f2c1a.jpg", "size": 84213},
  "srt":       {"url": "…", "path": "/v1/files/cmp_8f2c1a.srt", "size": 1204},
  "vtt":       {"url": "…", "path": "/v1/files/cmp_8f2c1a.vtt", "size": 1250},
  "narration": {"url": "…", "path": "/v1/files/cmp_8f2c1a_voice.wav", "size": 4500044},
  "scenes":    [{"index": 0, "start": 0.0, "end": 4.12}],
  "duration":  47.3
}
```

`url` is absolute (built from the request that created the job) and `path` is the
relative form; either can be fetched, the relative one survives a change of
hostname.

Jobs **queue** and run one at a time — a render is ffmpeg-bound and two at once
finish later than one after the other — and a waiting job's `message` says how
many are ahead of it. Poll every 2–3 s. With `"async": false` the call blocks and
returns the finished document, which is useful in tests and a bad idea for a real
render.

Errors: `400` for an invalid spec, naming the scene and field
(`scenes[2].visuals[0].src: required for type 'video'`); `404` for an unknown job
id; `501` when a requested sub-feature has no model configured (`music.generate`
without an audio-generation model, `captions.timing: "stt"` without a speech
model, narration without a TTS model) or when the install has no ffmpeg.

## Uploading your own media

Inlining a 40 MB clip as base64 in every render is the kind of waste that makes
an API unusable, so upload it once:

```bash
curl -H "Authorization: Bearer $TOKEN" -F file=@footage.mp4 \
     https://coderai.example/v1/files/upload
# {"id":"sha256:9f86d0…","url":"https://…/v1/files/up-9f86d0….mp4",
#  "path":"/v1/files/up-9f86d0….mp4","bytes":41234567,"mime":"video/mp4",
#  "kind":"video","existed":false}
```

Then name `path` (or `url`) as a visual's `src`, or as `music.src`. Storage is
content-addressed by sha256, so re-uploading the same bytes is free (`existed:
true`) and a client can ask first:

```bash
curl -H "Authorization: Bearer $TOKEN" https://coderai.example/v1/files/blob/9f86d0…
# 200 with the URL, or 404 — upload only on 404
```

The body may also be JSON (`{"file": "<base64 or data: URI>"}`) or the raw bytes
with a `Content-Type`. Video, image and audio only; the type is decided by the
content, not by the client's claim.

Uploading does not need an output directory. With `--file-path` set, an upload
lands there beside the generated files; without one it goes to the install's
cache directory instead, and both the `sha256:` id and the `/v1/files/<name>`
URL work the same way. (Composing a *new* video still needs an output
directory — there has to be somewhere to write the result.)

## The talking presenter

A person talking to camera is the format most short videos use, and it is
something only this server can assemble: composition synthesises the narration,
so it holds the exact audio of every scene, and the lip-sync engines and
identity-locked video generation are here too.

```jsonc
{
  "presenter": {
    "character": "nova",            // a saved character profile (its front reference is used)
    "image": null,                  // or an explicit portrait (URL / /v1/files / data URI / base64)
    "engine": "auto",               // auto | latentsync | musetalk | wav2lip | sadtalker | <any installed name>
    "motion": "generate",           // generate: a short identity-locked clip per scene
                                    // still: animate the portrait   src: use base_src
    "base_src": null,
    "prompt": "talking to the camera, natural head movement, soft studio light",
    "layout": "full",               // full | pip | split
    "position": "bottom-right",     // pip: top/bottom + left/right/center
    "size": 0.38,                   // pip: bubble width as a fraction of the canvas
    "shape": "circle",              // pip: circle | rounded | rect
    "split": "bottom",              // split: which half the presenter takes
    "border": {"color": "#FFFFFF", "width": 6},
    "remove_background": false      // ignored (with a warning) when no matting is installed
  },
  "scenes": [
    {"text": "Hi! Today: three hearts.", "visuals": [...]},                      // the default presenter
    {"text": "Here's how they work.",    "visuals": [...], "presenter": false},  // B-roll only
    {"text": "Follow for more!",         "visuals": [], "presenter": {"layout": "full"}}
  ]
}
```

* **Timing** is the scene's: the engine is driven by *that scene's* narration —
  synthesised or supplied — and the clip is held to narration + padding (or
  `min_duration`). Captions, music and ducking are unchanged.
* **`engine: "auto"`** takes a video-driven engine when there is a clip to drive
  and a portrait engine otherwise. The video-driven order is **latentsync,
  musetalk, wav2lip**: wav2lip works at 96x96 and upscales, which is where the
  soft, smeared mouth on a 1080p reel comes from, while LatentSync (Apache-2.0)
  and MuseTalk (MIT) run at 256-512 px natively. For a still portrait
  **sadtalker** still leads — it is the only one that invents head motion
  instead of replacing a mouth. Any other engine installed
  under its own name can be named instead — no API change for a new one. Ask
  `GET /v1/video/presenter/engines` what this install has.
* **`layout`**: `full` replaces the scene's visuals (a scene may then have none);
  `pip` composites a bubble over them inside the caption safe area — and captions
  at the bottom move above it automatically; `split` stacks presenter and B-roll,
  each filling its half. With a presenter, a scene's visuals are joined without a
  transition *between them* and composited as one clip; without one, nothing
  changes.
* **Nothing fails the render**: no engine, no face, a crash, a failed composite —
  each falls back to the scene's visuals (or a gradient when there are none) and
  adds a warning naming the scene. `stage` is `presenter`, with
  `message: "scene 2/6: lip-sync"`, and `result.scenes[i].presenter` is
  `{"engine": …, "layout": …}` or `null`.
* **Errors at submission**: `400` for an unknown character, an invalid layout,
  position, size, shape, motion or a `pip` scene with no visuals; `501` when a
  presenter is asked for and no lip-sync engine is installed, naming what to
  install.
* **GPU**: lip-sync is the only GPU step in a composition (the rest is ffmpeg), so
  it takes a lease from the same scheduler ordinary model requests use instead of
  fighting a generation for the card.

### `POST /v1/video/talking-head`

The presenter on its own, for a preview on a character page: a character (or a
portrait), `text` synthesised with `voice` — or `audio` you already have — and the
same job semantics as compose (same job store, so either status URL works). Capped
at 15 s of audio. `result.video` as usual.

```bash
curl -H "Authorization: Bearer $TOKEN" -X POST …/v1/video/talking-head \
  -d '{"character":"nova","text":"Hi, I am Nova.","canvas":{"width":720,"height":720}}'
# {"id":"cmp_…","status":"queued"}
```

## The same references elsewhere

The extractors that build characters and voice profiles from a user's own media
resolve their media fields through the same resolver, so they accept everything
compose does — a `/v1/files/...` path, an absolute URL of one, a `sha256:` upload
id, a `data:` URI, raw base64, a remote URL:

* `POST /v1/characters` (each reference image), `PATCH /v1/characters/{name}`
  (`add_images`)
* `POST /v1/characters/extract` (`images`, `videos`)
* `POST /v1/audio/voices`, `POST /v1/audio/voices/extract` (`audio`, `video`)

Upload the clip once and pass the path; a 100 MB source no longer has to travel
as base64 (a third larger) on every call. A file this install holds is read from
disk rather than fetched over HTTP — a loopback request would need a token and
would fail when the front is bound elsewhere. A bare path on the server is still
refused: a client cannot name files on this machine. Audio clips are now stored
under the extension their *content* implies rather than the one the caller
claimed.

## Notes for operators

* **ffmpeg is required** (the published images have it, with libass for the
  captions). Without it every compose request answers `501`.
* **Fonts**: captions ask for a family by name. The images ship Montserrat,
  Inter, DejaVu and Noto (including CJK); a family that is not installed falls
  back to the closest one and adds a warning, it does not fail. Treat that
  warning as an error in disguise when the layout matters: the substitute is a
  different width, and a wider one pushes lines towards the frame edge.

  **Adding your own**: drop font files (`.ttf`, `.otf`, `.ttc`) into
  `<config>/fonts` — `~/.coderai/fonts`, or `/config/coderai/fonts` in the
  images — and restart. They are registered with fontconfig at startup, so the
  ffmpeg/libass child that burns the captions resolves them too, and because
  the directory lives on the config volume they survive a restart and a new
  image. `CODERAI_FONT_DIR` points somewhere else (several paths, `:`-separated).
  Check with `fc-match -f '%{family}\n' 'Your Family:style=Bold'` inside the
  container: it must echo the family you asked for.

* **Caption size** is derived from the SHORT side of the frame, times `0.052`,
  times the preset's scale, times `captions.font_scale`. A 16:9 frame is
  unchanged by this (there the short side is the height); a vertical frame is
  sized as if it were that square, rather than from its 1920px height, which
  used to put a ~100px font in a 1080px-wide frame. On top of that, any single
  line still too wide for the text area (`width - 2 * 6%`) is measured against
  the real font file and given its own smaller size, so one long line shrinks
  instead of running off the edge. Clients that were scaling `font_scale` down
  by hand for 9:16 or 4:5 should drop that factor.
* **Where the files go**: the ordinary output directory (`--file-path`), served
  by `/v1/files/`, under normal retention. A composed video is typically read
  once by the client and copied to its own storage.
* **Cost**: a render is CPU-bound. One job at a time, each ffmpeg child capped at
  half the cores, so composition never starves the GPU engines.
* **Multi-engine installs**: composition runs in the engine the front routes it
  to (the primary), and it synthesises narration and runs speech recognition
  *in that process* — so the TTS and STT models it needs must be loadable there.
  Leave them unpinned, or pin them to the primary; a model pinned to another
  engine makes the job fail with that engine's own "model not allowed here"
  message. Supplying each scene's `audio` and `captions.timing: "estimate"`
  needs no model at all.
* **Polling is free**: `GET /v1/video/compose/{id}` and
  `GET /v1/video/talking-head/{id}` are exempt from the `/v1/video/` rate limit
  (10 requests a minute per IP by default), so polling a long render does not
  spend the budget its own renders need. They still require a Bearer token --
  the document names the job's output -- and the sibling `POST .../cancel` is
  limited like any other action.
* **Jobs live in memory** (the last 100, plus whatever is running): a front
  restart loses them, and a client polling a lost id gets `404` — treat that as
  "restart the render", not as an error in the request.

## Code

| Piece | Where |
|---|---|
| Endpoints, validation, job queue, orchestration | `codai/api/compose.py` |
| Uploads (`/v1/files/upload`, `/v1/files/blob/{hash}`) | `codai/api/uploads.py` |
| The talking presenter (engines, face, base clip) | `codai/compose/presenter.py` |
| Reference resolving, probing, fonts | `codai/compose/media.py` |
| Word timings, line breaking, ASS/SRT/VTT | `codai/compose/captions.py` |
| The ffmpeg pipeline | `codai/compose/render.py` |
| Tests (real ffmpeg, no model) | `tests/test_compose.py` |
