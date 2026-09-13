# Running the bundled tools as separate services

Audit of whether each bundled tool can run **detached** — its own container,
reaching coderai only over the HTTP API (assumed proxied 1:1 by aisbf).

Verdict: **every client tool is detachable.** One coupling existed and is now
fixed; the rest is deployment plumbing (volumes, ffmpeg, timeouts).

## Client tools — detachable

| tool | reaches coderai via | needs |
|---|---|---|
| `character_studio.py` | `--base-url` / `CODERAI_BASE_URL`, `/v1/*` only | ffmpeg, own out-dir volume |
| `videogen.py` | same | ffmpeg, own out-dir volume |
| `video_dubber.py` | same (multipart uploads) | ffmpeg + ffprobe |
| `gen_township_fighters.py` | same | ffmpeg, own out-dir volume |
| `video_editor.py` | same | ffmpeg, own media-dir + output-dir volumes |
| `manual_multimodal_test_client.py` | `--url` | — |

Checked for, and clean:
- **No tool imports `codai.*` any more.** `video_editor.py` used to import
  `codai.api.tts_backends.family_emotions/family_styles` to fill its emotion and
  style pickers — the one thing that only worked inside the server's process tree.
  It now asks `GET /v1/audio/speech/capabilities` (added for this), falling back to
  the local import when the server is older, and to empty lists when neither works
  — the pickers just don't appear.
- **No tool sends a local path into the API** expecting the server to open it.
  Media goes as base64 or multipart. The `path` fields that do appear are the
  tools' own paths, returned to their own browser UIs.
- **No tool reads coderai's own state** — no `models.json`, no `/AI`, no
  `~/.coderai`. (`gen_township_fighters.py` mentions `models.json` only in
  `--help` text; `video_editor.py`'s `/cache/` is its own config path.)
- **Generated media is fetched over HTTP.** The clients resolve `b64_json` /
  `b64_mp4` / URLs / `/v1/...` paths, never a filesystem path from the server.

## Not API clients (different class)

- `review_outputs.py` — a review UI over the *tools'* output directories, not a
  coderai client. Detaching it means mounting those output volumes, not proxying.
- `*_service.py` (`canary`, `crisperwhisper`, `parler_tts`, `pyannote`, `melotts`,
  `h3`), `h3_common.py`, `vpr_salad_server.py` — these are coderai's own
  subprocesses, launched inside its isolated venvs by the matching `*_worker.py`.
  They are not detachable tools and must stay with the server.

## What each container needs

1. **ffmpeg + ffprobe** on PATH — every media tool shells out to them.
2. **Its own writable volume** for output; `video_editor.py` also needs its
   `media_dir` (its source library, which today is a local directory).
3. **`CODERAI_BASE_URL` + `CODERAI_API_KEY`** — all of them accept both as flags
   and as env vars.
4. **Generous proxy timeouts.** Video generation and LoRA training are single
   requests that run for minutes; the studio and videogen poll progress endpoints,
   but the generation call itself blocks. The nginx block for `/character/` uses
   `proxy_read_timeout 3600s` — detached tools need the same on their own routes.
5. **Large request bodies.** Uploads are base64 or multipart; the container nginx
   already allows 4 GB globally, and the `/character/` proxy on lisa sets
   `client_max_body_size 1024m` with `proxy_request_buffering off`.
6. **Sub-path awareness, if proxied under a prefix.** `character_studio.py` and
   `video_editor.py` honour `X-Forwarded-Prefix`; `videogen.py`,
   `gen_township_fighters.py` and `review_outputs.py` assume they own `/` (see
   reverse-proxy-nginx.md). Detaching those three behind a prefix needs the same
   `ROOT_PATH` treatment the character studio got.

## Not verified

The assumption that aisbf proxies the coderai API 1:1 was taken as given, not
tested. Nothing here has been run as an actual separate container either — this is
a source-level audit plus the one fix it turned up.
