# Plan: pod images for the models added this round

**Status: plan only. Nothing here is built.**

The models added in this round run locally today. This is what it would take to
rent them, written against `packaging/runpod/` as it stands — the profile
mechanism, the two cores, and the size line that constrains both.

## What the existing images already cover

| Model | Pod image today | Gap |
|---|---|---|
| Z-Image, Wan2.2-TI2V-5B | `coderai-video` (`video.txt`: diffusers, torchvision, imageio) | none known — same diffusers path as the other video models |
| LTX-2.3 (10Eros, PinkCherry) | `coderai-video` | **the GGUF single-file path and its LoRAs**, see below |
| MusicGen, AudioGen, AudioLDM2, Stable Audio | `coderai-audio` (full core + audiocraft) | none |
| ACE-Step | — | **no image can run it**, see below |
| MiniMax H3 | — | isolated venv, never run on real weights locally either |

## 1. ACE-Step: a new profile on the light core

The one real finding: **the pod cores are `python:3.12-slim`**, which is exactly
the interpreter ACE-Step needs. The local image is on Ubuntu 24.04, whose system python is
also 3.12, so neither place needs a bundled interpreter — only a base older
than 24.04 would. `Dockerfile.capability`'s venv loop already does
`python -m venv /opt/coderai/venvs/<name>` with the image's own python, so a
3.12 venv is what it produces by default.

It must **not** be added to the `audio` profile. `audio` has no `.light` marker,
so it is built on the full core — 7 GB of CUDA and torch in the main venv for
audiocraft. A 6.4 GB ACE-Step venv on top of that lands far above the measured
RunPod ceiling (10.1 GB boots; 14.5 GB is "Exited by Runpod" two seconds after
rent, with no other reason given). So:

```
packaging/runpod/profiles/
  acestep.light                  # empty marker -> build on the LIGHT core
  acestep.txt                    # main venv: nothing GPU, just the API's deps
  acestep.venv-acestep.txt       # = requirements-acestep.txt, verbatim
  acestep.venv-acestep.uv        # marker: install with uv, not pip
  acestep.venv-acestep.check     # deep check, below
```

- The `.uv` marker is **required**, not a preference: `requirements-acestep.txt`
  carries the PyTorch extra index, and uv's default first-index-wins resolves
  unrelated names off it (it pinned `requests==2.28.1` and the install
  collapsed). The loop's uv branch already passes
  `--index-strategy unsafe-best-match`.
- The loop's own import probe (`import acestep`) is satisfied as-is: the
  distribution is `ace-step`, the module is `acestep`, and the loop derives the
  probe from the venv name, so the file must be named `venv-acestep`.
- `CODERAI_ACESTEP_VENV=/opt/coderai/venvs/acestep` is **already forwarded** to
  pods (`runpod_worker.py`, beside pyannote/nemo/crisperwhisper/TTS), so the
  worker finds the baked venv instead of building a 6.4 GB one on a machine
  rented by the second.

Estimated image size: light core + the venv, so ~7–8 GB. That is under the line
but not comfortably — worth measuring before publishing, and worth trimming
`torchvision` from the requirements if it turns out unused (ACE-Step declares
it; the audio path has no reason to touch it).

**The deep check should load weights, not just import.** The lesson from the
embeddings image was 303 s of boot paid before anyone could discover that every
request failed; here the equivalent trap is that constructing
`ACEStepPipeline` succeeds while loading nothing — it reads the checkpoints on
the first call. So `acestep.venv-acestep.check` should at minimum assert that
`load_checkpoint` exists and that `ACE_PIPELINE_DTYPE` is honoured, and the
smoke test should generate ~2 seconds of audio on a real pod before the image is
signed.

**Weights.** ~7.8 GB across four subdirectories. The pod should fetch
`ACE-Step/ACE-Step-v1-3.5B` into the HF cache on the volume, like every other
model; `checkpoint_dir` then points at the snapshot, which already has the
layout the pipeline wants. Without a network volume it downloads to container
disk on every rent.

## 2. LTX-2.3 on a pod: the open question

The local LTX-2.3 models load a **GGUF single file** plus a diffusers `config=`,
with converter key-alias repair, and their DMD LoRAs are in ComfyUI layout. On a
pod, `coderai-video` has diffusers and torch, so the pipeline class is there —
but none of this has been run on a pod, and two things are unverified:

- `gguf` (the python package the single-file loader needs) is **not in the pod
  images at all**: they install the curated `core.txt`, not `requirements.txt`,
  and the package appears in neither that nor `video.txt`. It has to be added —
  checked, not assumed.
- how the weights get there: these are 20+ GB single files, which means a
  network volume and `volume_path`, not a cold download per rent.

This is a measurement task, not a design one. It should be settled on one rented
pod before anything is built.

## 3. What not to do

- **No per-model images.** 10Eros and PinkCherry are two checkpoints on one
  pipeline; they differ by weights, which travel on the volume. One `video`
  image serves both — the sibling-path and cache-key fixes that let them
  coexist locally are server-side, not image-side.
- **No ACE-Step in the `audio` image.** Size, above.
- **No new core.** Both existing cores are reused by every profile and the
  registry stores them once; a third core would be carried for one image.

## Order of work

1. One rented pod, by hand: does `coderai-video` run an LTX-2.3 GGUF at all?
   That answers §2 before any image is built.
2. Build `coderai-acestep` from the four profile files above, one image at a
   time (build → push → sign → delete the local copy; the docker disk runs
   90–98% full).
3. Smoke-test it on a real pod with a 2-second generation, not just `/healthz`.
4. Add `gguf` to `video.txt` and rebuild `coderai-video` — needed regardless of
   what §1 finds, since no pod image carries it — plus whatever else §1 turns up.
