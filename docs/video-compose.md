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
      "visuals": [            // split evenly across the scene, in order
        {"type": "video", "src": "https://…/clip.mp4", "trim_start": 0.0},
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

A scene needs either narration (`text` and/or `audio`) or an explicit
`"duration"`, and at least one visual. Anything else has a default.

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
  back to the closest one and adds a warning, it does not fail.
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
* **Jobs live in memory** (the last 100, plus whatever is running): a front
  restart loses them, and a client polling a lost id gets `404` — treat that as
  "restart the render", not as an error in the request.

## Code

| Piece | Where |
|---|---|
| Endpoints, validation, job queue, orchestration | `codai/api/compose.py` |
| Uploads (`/v1/files/upload`, `/v1/files/blob/{hash}`) | `codai/api/uploads.py` |
| Reference resolving, probing, fonts | `codai/compose/media.py` |
| Word timings, line breaking, ASS/SRT/VTT | `codai/compose/captions.py` |
| The ffmpeg pipeline | `codai/compose/render.py` |
| Tests (real ffmpeg, no model) | `tests/test_compose.py` |
