# Capabilities that need an interpreter of their own

Some engines cannot share the server's interpreter. coqui-tts imports a
symbol transformers 5 removed; parler-tts, NeMo and pyannote pin an old
transformers/tokenizers; MeloTTS the same. Pinning any of them in the server
would break every other capability, so each gets a venv of its own, built on
first use, and coderai drives it as a subprocess.

That part worked. What did not was the environment those subprocesses were
started in.

## The one variable that broke all of them

Measured on the engine process, 2026-10-29:

```
HOME=/home/ubuntu
PYTHONHOME=/opt/coderai/python
```

Every worker was launched with `env=dict(os.environ)`, so every one of them
inherited that `PYTHONHOME`. A venv interpreter started under a `PYTHONHOME`
that points at a *different* installation takes its stdlib and its
`site-packages` from there — the venv's own packages are not on `sys.path` at
all.

What the operator saw instead:

| reported as | actually |
|---|---|
| `501 … no venv containing it was found` (XTTS) | the venv was there and populated; the `import TTS` probe ran against the server's site-packages |
| `Command '[… -m venv …]' returned non-zero exit status 1` (parler) | `python -m venv` run under a foreign `PYTHONHOME` |
| `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` | the same shape one level down: the worker's torch resolving shared libraries through another installation's directories |

Three different-looking failures, one cause — which is why each was chased
separately and none was fixed.

`codai/api/venv_env.py` is now the only place a worker's environment is
built. It drops what redirects an interpreter (`PYTHONHOME`, `PYTHONPATH`,
`PYTHONSTARTUP`, `PYTHONEXECUTABLE`, `VIRTUAL_ENV`, …).

`LD_LIBRARY_PATH` is decided per venv, because the two kinds of worker need
opposite things:

* a **foreign** interpreter (ACE-Step's 3.12, LatentSync's 3.11, the lipsync
  3.10) must not see coderai's library directories, or it loads this python's
  extension modules — `ImportError: libtorch_python.so: undefined symbol:
  PyObject_GetOptionalAttrString`, a CPython 3.13 API, from a 3.10 process;
* a venv built from **the server's own** interpreter (parler, CrisperWhisper,
  NeMo, pyannote) runs that very python and must find its `libpython`, so
  stripping those directories would stop it starting at all.

`env_for_venv` reads the venv's `pyvenv.cfg` and answers for that venv.
Everything outside the server's installation is kept either way: a worker that
cannot see `/usr/local/nvidia/lib` has no GPU, which is a worse failure than
the one being fixed. (And `sys.prefix` is ignored when it IS the system
prefix — a pip install puts it at `/usr`, where treating it as "ours" would
strip the driver from every worker.)

It applies to **every** call into a venv: the import probe, `python -m venv`,
each `pip install` and the service process itself. A clean service process with
a probe run under the poisoned environment still reports the venv as broken,
and a clean probe with a poisoned `pip` installs into the wrong tree. A test
enforces that no isolated worker builds its own environment dict.

It does **not** apply to coderai's own children — the engine supervisor,
cluster RPC — which are the same interpreter, and for which `PYTHONHOME` is how
they find it.

## When a venv that is there cannot be used

`python -m venv` over a broken venv dies on the very thing that broke it, so a
venv is never repaired in place. `venv_env.create_venv` rebuilds it, and
`venv_usable` is the single decision about when:

* **a dangling interpreter.** `bin/python` is a symlink to the python that
  built it. Build it on the host, run it in the container — or build it before
  the image moved its interpreter — and the link dangles. `Path.exists()`
  follows symlinks and answers False, so the old code concluded "missing" and
  ran `venv` over the top: `Errno 2 … '/…/parler_venv/bin/python3'`, on every
  request, for ever, with 7 GB of packages sitting in it.
* **a minor-version mismatch.** Measured the same day: an `xtts_venv` whose
  `bin/python3` pointed at `/usr/bin/python3`, built by the host's 3.13 while
  the container's is 3.12. The interpreter runs perfectly and imports none of
  the 875 MB under `lib/python3.13/site-packages`. "Does it run" cannot see
  this.
* **`--system-site-packages` where isolation was wanted** — the server's
  incompatible stack leaks straight back in. The reverse counts too: the
  MiniMax-H3 overlay inherits the server's torch *on purpose*, so an isolated
  venv found where an overlay is wanted is also rebuilt.

## Importable is not working

With the environment fixed, Parler got further: the venv built, the service
started, the model loaded — and the first synthesis died in the audio
decoder's `conv1d`:

```
RuntimeError: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed
ptrDesc->finalize() cudnn_status: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
```

Every cudnn sublibrary was present in the venv. The stack was a torch 2.15
with the CUDA-13 wheels, which `pip install git+…parler-tts` had resolved on
its own, because nothing pinned it. And `import parler_tts, soundfile` — the
only health check — answered True for that venv for ever, so nothing ever
rebuilt it.

Two changes:

* **the stack is pinned.** `requirements-parler.txt` names the torch build,
  the way every other isolated worker's file does. Pip resolving a CUDA stack
  on its own is how a venv ends up newer than the fleet it runs on.
* **the health check runs the op that fails.** The decoder is convolutional,
  so the probe is one `conv1d` on the GPU. A venv that imports but cannot run
  it is rebuilt from the pinned file; a CPU-only host passes, because there is
  no GPU stack to be broken.

And the service itself latches cudnn off if it still fails there: torch has
native kernels for those convolutions, the decoder is a fraction of the work,
and the alternative is no speech at all.

The general rule, which cost two bring-ups here: **a venv's health check must
exercise what the worker actually does.** An import proves the files are in
the right directory and nothing else.

## Knowing before you ask

A venv is built on first use, and that build downloads several GB. Discovering
that inside a synthesis request is a surprise, so
`GET /v1/audio/speech/capabilities?model=…` answers it up front:

```json
{
  "model": "coqui/XTTS-v2",
  "family": "coqui",
  "voices": [{"id": "Claribel Dervla", "gender": null, "languages": []}],
  "voice_selection": "builtin|clone",
  "needs_venv": true,
  "env_ready": false,
  "env_detail": "/home/ubuntu/.coderai/xtts_venv: it does not exist"
}
```

`env_ready: false` does not mean unusable — it means the first request will
spend minutes building the environment, which is what a client wants to know
before offering the voice.

Answered from the filesystem alone: the venv's `pyvenv.cfg`, its recorded
interpreter and the package directory. Never by importing into the venv, and
never by loading a model — a catalogue call that costs seconds is how a listing
route turns into a loader.

## The engines

| engine | needs a venv | package |
|---|---|---|
| Kokoro, Bark, transformers TTS | no | — |
| XTTS (coqui) | yes | `coqui-tts` |
| Parler | yes | `parler-tts` |
| MeloTTS | yes | `melo` |
| CrisperWhisper | yes | pinned `transformers` + `torch` |
| NeMo / Canary | yes | `nemo_toolkit[asr]` |
| pyannote (diarization) | yes | `pyannote.audio` |
| MiniMax-H3 | overlay | `diffusers>=0.40` |

`CODERAI_XTTS_VENV`, `CODERAI_PARLER_VENV` and the equivalents point at a venv
built elsewhere — a pod image that bakes one, for instance. A `service_url` on
the model entry points at a worker running on another host entirely, and then
nothing is built, spawned or downloaded locally.
