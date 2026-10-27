# Text to music: ACE-Step

`POST /v1/audio/generate` with an ACE-Step model writes a song — a style prompt,
optional sung lyrics, a length in seconds. It joins MusicGen, AudioGen,
AudioLDM2 and Stable Audio on the same endpoint; the model's name selects the
branch, so nothing about the request shape is ACE-Step-specific.

```bash
curl -s http://localhost:8000/v1/audio/generate \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"ACE-Step",
       "prompt":"slow synthwave, warm analog pads, 90 bpm",
       "lyrics":"[verse]\nthe tide goes out\n[chorus]\nand takes the light",
       "duration":60, "steps":60, "seed":42}'
```

| field | ACE-Step default | notes |
|---|---|---|
| `prompt` | — | the *style*, not the words: genre, instruments, tempo, mood |
| `lyrics` | `""` | supports its `[verse]` / `[chorus]` tags; `[inst]` for an instrumental |
| `duration` | 60 s | the whole track is generated at once, not stitched |
| `steps` | 60 | diffusion steps |
| `cfg_coef` | 15.0 | guidance; ACE-Step wants far more of it than MusicGen |
| `seed` | random | reproducible with everything else held equal |

Output is 48 kHz stereo WAV.

`duration` and `cfg_coef` are shared across every model on this endpoint and
their field defaults are MusicGen's (10 s, 3.0). An ACE-Step call that leaves
them out gets **ACE-Step's** defaults instead, because a 10-second song at a
guidance it was not tuned for is not a useful reading of "unspecified". A value
you actually send is always used, including `duration: 10`.

## Why it runs in its own interpreter

ACE-Step ships its own `acestep` package, which pins `transformers==4.50` and
`spacy==3.8.4`. The server runs transformers 5.x, and spacy 3.8.4 publishes no
cp313 wheel at all — so this is not merely a dependency the server would rather
not hold, it is one that **cannot be installed on the server's interpreter**.

So it runs the way CrisperWhisper, Canary, pyannote and MeloTTS do: a python
**3.12** venv of its own, with `tools/acestep_service.py` inside it speaking
HTTP on a local port, driven by `codai/api/acestep_worker.py`.

- The venv is built on the **first request** to an ACE-Step model and reused
  afterwards. A model that is configured but never called costs nothing.
- `python3.12 -m venv` is tried first; on hosts that have `/usr/bin/python3.12`
  without the `python3.12-venv` package (common on Debian) it falls back to
  `uv venv --python 3.12`, which will also fetch a 3.12 if the host has none.
- `CODERAI_ACESTEP_VENV` overrides the location; `service_url` (or
  `CODERAI_ACESTEP_SERVICE_URL`) points at a worker running elsewhere instead,
  and then nothing is built, spawned or downloaded locally.

The weights come from whatever the model's `path` resolves to. An HF snapshot of
`ACE-Step/ACE-Step-v1-3.5B` already has the four subdirectories the pipeline
expects (`ace_step_transformer`, `music_dcae_f8c8`, `music_vocoder`,
`umt5-base`), so the worker points the pipeline straight at the snapshot
coderai already has rather than letting it fetch a second copy.

### It evicts like any other model

The handle cached in the model manager exposes `cleanup()`, which is what the
manager calls when it evicts a model to make room for the next one — and here
that stops the worker process, which is the only thing that frees its VRAM.

The service also loads the checkpoints **before** it answers its health check.
That is not tidiness: coderai measures a model's footprint as the device-wide
free-VRAM delta across the load, and that window closes when the worker reports
healthy. Constructing `ACEStepPipeline` loads nothing — upstream reads the
checkpoints on the first `__call__` — so a lazy load would measure ~0 GB and
eviction would treat a resident 3.5B model as free. Measured on a 3090:
**~8.5 GB** at bf16.

### Precision and quantization

`precision` is honoured exactly. Upstream resolves its own `dtype` argument as
`torch.bfloat16 if dtype == "bfloat16" else torch.float32`, so asking it for
float16 silently yields float32 — twice the VRAM, no error. The worker sets
`ACE_PIPELINE_DTYPE`, which is read after that switch, instead.

The quantization switches (`load_in_8bit` / `load_in_4bit`) ask upstream for
`quantized=True`, which does **not** quantize anything: it loads ACE-Step's
separate int4wo export (`*_int4wo.bin`) from the same directory layout. If those
files are not in the checkpoint directory the worker says so at load time,
naming the missing file, rather than failing with `FileNotFoundError` minutes
in. Point the model's path at the quantized checkpoint, or leave the switches
off.

## The two image variants

The venv is ~6.4 GB, so the **default image does not carry it** — it builds on
first use, like every other isolated worker. A second variant bakes it in, for
installs that would rather pay the size than the first-call wait:

```bash
packaging/linux/build_oci_image.sh                  # default: no ACE-Step venv
packaging/linux/build_oci_image.sh --with-acestep   # venv pre-built, baked in
```

`--with-acestep` **copies the development venv** when there is one
(`$CODERAI_ACESTEP_VENV`, else `~/.coderai/acestep_venv`) — it is already built
and already the thing that was tested. Only when there is none does it build one
with `uv`. `--acestep-venv PATH` bundles a specific venv and implies
`--with-acestep`.

Because the base image is Ubuntu 22.04, whose system python is 3.10, a baked
3.12 venv also needs a **relocatable 3.12 interpreter** bundled beside it; the
build fetches one with `uv python install 3.12` if the host has none, and the
image re-points the venv's `pyvenv.cfg` at `/opt/coderai/py312` — the same
arrangement the 3.10 lip-sync and LongCat venvs use.

Both variants resolve the venv the same way at runtime, so neither needs an
environment variable: the worker uses `/opt/coderai/acestep_venv` (local image)
or `/opt/coderai/venvs/acestep` (pod images) when one is there, and
`~/.coderai/acestep_venv` otherwise.
