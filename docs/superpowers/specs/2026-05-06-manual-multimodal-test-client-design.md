# Manual Multimodal Test Client Design

## Overview

Add a standalone Python test client script for manual, one-request-at-a-time smoke testing of key CoderAI API flows.

The script should support these test modes:

- normal LLM prompt
- transcription
- audio generation
- video generation
- video doubt
- music audio doubt

The user should be able to run exactly one mode per invocation, either by passing the mode directly on the command line or by launching the script without a mode and choosing from an interactive menu.

The script must be usable with minimal input by providing meaningful defaults for prompt, model, endpoint URL, and media files where needed, while still allowing explicit overrides.

## Goals

- Provide one convenient manual smoke-test client for multiple API capabilities.
- Support one test at a time, not batch orchestration.
- Allow both scripted CLI use and interactive ad-hoc use.
- Allow overriding endpoint URL, bearer token, model name, prompt text, and media file inputs.
- Save generated artifacts locally for generation modes and print any text reply to stdout.
- Keep the implementation standalone and easy to run without touching the server code paths.

## Non-Goals

- Build an automated benchmark suite.
- Add server endpoints or alter server request/response contracts.
- Guarantee that every configured model supports every test mode.
- Replace unit/integration tests with this script.
- Support running multiple modes in a single invocation.

## Current Context

The current repository already exposes the following relevant endpoints:

- `POST /v1/chat/completions` in `codai/api/text.py`
- `POST /v1/audio/transcriptions` in `codai/api/transcriptions.py`
- `POST /v1/audio/generate` in `codai/api/audio_gen.py`
- `POST /v1/video/generations` in `codai/api/video.py`

The Pydantic request models show:

- `AudioGenerationRequest` accepts prompt-based generation and returns audio-oriented response data.
- `VideoGenerationRequest` accepts prompt-based generation plus optional media inputs and returns video-oriented response data.
- `ChatCompletionRequest` is permissive on extra fields but `ChatMessage.content` normalization currently appears text-centric, converting multipart content arrays into flattened text or placeholders.

This means the client can confidently target the transcription, audio generation, video generation, and plain LLM paths directly. The two “doubt” modes must be designed around the request shapes the backend currently accepts rather than assuming a richer multimodal chat transport that is not yet proven here.

## Recommended Approach

Create a single standalone Python CLI script with:

- one positional `mode` argument for direct execution
- an interactive menu fallback when `mode` is omitted
- shared global options for connection/auth/model overrides
- mode-specific request builders
- shared response handling utilities
- local artifact saving for generation outputs

This keeps the user experience simple while avoiding duplicated logic across multiple small scripts.

## Script Behavior

### Invocation styles

The script should support both of these patterns:

- direct mode invocation, for example `python <script> llm`
- interactive menu invocation, for example `python <script>` then select one mode

Exactly one mode should execute per run.

### Supported modes

The script should expose these user-facing mode names:

- `llm`
- `transcription`
- `audio-generation`
- `video-generation`
- `video-doubt`
- `music-audio-doubt`

Aliases are optional but not required in the first version.

### Global options

Every mode should support these shared overrides:

- `--url` for the base server URL
- `--token` for bearer authentication
- `--model` for model name or alias override
- `--prompt` for prompt override when the mode uses prompt text
- `--output-dir` for local artifact save location

The script may also expose mode-specific file flags such as `--file`, `--audio-file`, `--video-file`, or similarly explicit names, but the shared defaults must make a basic smoke test possible without forcing the user to provide every input.

## Mode Design

### LLM mode

Purpose:
- send a normal text-only chat request

Endpoint:
- `POST /v1/chat/completions`

Default inputs:
- default URL
- default text-capable model alias/name
- default prompt such as a short deterministic question

Success behavior:
- print assistant text reply to stdout

### Transcription mode

Purpose:
- upload one audio file for speech-to-text

Endpoint:
- `POST /v1/audio/transcriptions`

Default inputs:
- default URL
- default audio transcription model alias/name
- default sample audio file path
- optional default language/prompt values if useful

Request format:
- multipart form with `model` and uploaded `file`

Success behavior:
- print returned transcription text to stdout

### Audio generation mode

Purpose:
- generate an audio artifact from a text prompt

Endpoint:
- `POST /v1/audio/generate`

Default inputs:
- default URL
- default audio-generation model alias/name
- default audio-generation prompt
- default response format chosen so the client can resolve an artifact reliably

Success behavior:
- save the resulting audio artifact locally
- print the local output path to stdout
- print any returned text metadata or summary if present

### Video generation mode

Purpose:
- generate a video artifact from a text prompt

Endpoint:
- `POST /v1/video/generations`

Default inputs:
- default URL
- default video-generation model alias/name
- default prompt
- conservative default generation parameters suitable for smoke testing

Success behavior:
- save the resulting video artifact locally
- print the local output path to stdout
- print any returned text metadata or summary if present

### Video doubt mode

Purpose:
- ask a textual question about a supplied video input

Endpoint strategy:
- use the currently supported server request shape rather than inventing a new dedicated endpoint

Design constraint:
- the current visible chat request normalization in `codai/pydantic/textrequest.py` suggests text-centric handling of multipart content arrays, so this mode must be implemented only according to request formats actually supported by the current codebase.

First-version design:
- treat this as a smoke-test mode that combines:
  - a user prompt describing the question about the video
  - a required or default video input reference
  - the best-supported request format discovered in the implementation phase

Success behavior:
- print the resulting text answer to stdout
- if the server returns an artifact in addition to text, save it and print the path as well

### Music audio doubt mode

Purpose:
- ask a textual question about a supplied audio/music input

Endpoint strategy:
- same constraint and approach as `video-doubt`

First-version design:
- combine:
  - a user prompt describing the question about the audio/music input
  - a required or default audio input reference
  - the best-supported request format discovered in the implementation phase

Success behavior:
- print the resulting text answer to stdout
- if the server returns an artifact in addition to text, save it and print the path as well

## Defaults Strategy

The script should be useful without a long flag list.

Each mode should define a default bundle containing:

- default base URL
- default model
- default prompt if applicable
- default sample file path if required
- default output extension or expected artifact type

These defaults should be easy to inspect and adjust inside the script.

For file-backed defaults:
- use predictable local sample asset paths
- do not embed binary sample files into the script itself
- keep the default sample paths stable and predictable, specifically under `samples/`
- maintain a fixed internal manifest of vetted public sample URLs for any required default media files
- when a required default sample file is missing, the client should first attempt to download it into the expected `samples/` path and only fail if that download attempt does not succeed
- dynamic web search at runtime is out of scope; sample sourcing should remain deterministic and testable

## Output Handling

### Stdout behavior

The script should always print the text reply/result to stdout when one exists.

Examples:
- LLM reply text
- transcription text
- text answer from video/audio doubt mode
- metadata/summary text accompanying a generation response

### Artifact behavior

For generation-oriented responses, the script should save the generated artifact locally by default.

This includes:
- audio file for `audio-generation`
- video file for `video-generation`
- any returned artifact from other modes, if present

The script should then print the resolved saved path to stdout.

### Output directory

The script should save files to a predictable output directory, configurable through `--output-dir`.

File naming should avoid collisions by including mode and a timestamp or similarly unique suffix.

### URL and base64 handling

If an endpoint returns:
- a downloadable URL, the client should fetch it and save the binary locally
- base64 content, the client should decode and save it locally

The script should hide those mechanics from the user and present a final local artifact path.

## Error Handling

The script should fail clearly when:

- a required input file is missing and no default sample exists
- the server returns a non-2xx response
- a generation mode succeeds structurally but does not actually contain a usable artifact
- the response shape is incompatible with the selected mode
- a URL-based artifact cannot be downloaded
- base64 artifact decoding fails

Error messages should explain:
- which mode failed
- which endpoint was called
- which file/model/prompt inputs were resolved
- what the server or client-side parsing error was

## File and Code Organization

A focused standalone script is preferred over scattering logic across multiple files unless the script becomes too large.

Internally, the script should still be organized into clear units such as:

- argument parsing and interactive selection
- mode default resolution
- request builders per mode
- HTTP execution helpers
- response parsing and artifact saving helpers

If helper extraction becomes necessary, it should remain narrowly scoped to this client tooling rather than affecting server modules.

## Testing Strategy

Add automated tests for the client script logic without depending on live model inference.

Recommended test coverage:

- mode selection logic for CLI vs interactive fallback
- default resolution per mode
- override precedence for URL/token/model/prompt/file inputs
- default sample auto-download behavior when expected media files are missing
- request construction for each supported endpoint type
- response parsing for:
  - plain text responses
  - generation responses returning URLs
  - generation responses returning base64 payloads
- error handling for failed default-sample downloads, missing override files, and malformed responses

Use mocked HTTP responses rather than live network calls.

## Open Implementation Constraint

The exact implementation of `video-doubt` and `music-audio-doubt` must be finalized by confirming which request shape the current backend really supports for video/audio-question-style prompts.

The implementation should not invent a new server contract. It should instead:

- inspect current supported request structures in the repo
- choose the best existing format
- document any limitations explicitly in the client behavior and tests

## Files Likely to Change

Likely additions:
- one standalone test client script in a project-appropriate scripts/tools location
- one or more tests for that script
- internal sample-source manifest and download helper logic for `samples/transcription.wav`, `samples/question-video.mp4`, and `samples/question-audio.wav`

Likely no server changes are required for the first version.

## Design Decisions Finalized

- One standalone Python script handles all supported manual smoke-test modes.
- The script runs exactly one mode per invocation.
- It supports both direct mode invocation and interactive selection fallback.
- Users can override endpoint URL, bearer token, model name, prompt, and file inputs.
- Defaults are built in so the script remains usable with minimal arguments.
- Required default media files continue to resolve to stable `samples/` paths and are auto-downloaded from fixed public URLs when absent.
- Generation modes save artifacts locally and print the saved path.
- Textual replies are printed to stdout whenever present.
- `video-doubt` and `music-audio-doubt` must be implemented against proven existing backend request formats, not assumed new multimodal contracts.
