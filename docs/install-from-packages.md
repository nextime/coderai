# Installing from the published packages

Everything CoderAI ships is a container image on GitHub Container Registry,
public, pulled without credentials. There is nothing to build.

## The packages

| Package | What it is | Pull |
|---|---|---|
| `ghcr.io/nextime/coderai` | **The full CoderAI** — every capability, every engine, the Web Studio and admin UI, the cluster head/node, llama.cpp with the RPC backend and `rpc-server`. One layer, ~28 GB. Tags: `latest`, `0.2.20`, … | `docker pull ghcr.io/nextime/coderai:latest` |
| `ghcr.io/nextime/coderai-images` | Image generation only (SDXL, Flux, Z-Image …) | `docker pull ghcr.io/nextime/coderai-images:latest` |
| `ghcr.io/nextime/coderai-video` | Video generation (Wan, LTX-2 …), upscale, interpolation | `…/coderai-video:latest` |
| `ghcr.io/nextime/coderai-text` | LLMs with transformers / llama.cpp, incl. local LoRA adapters | `…/coderai-text:latest` |
| `ghcr.io/nextime/coderai-llama` | llama.cpp GGUF server as a CoderAI pod (light core, CUDA for Ampere→Blackwell, RPC) | `…/coderai-llama:latest` |
| `ghcr.io/nextime/coderai-vllm` | vLLM as a CoderAI pod (its own venv, `+cu129` build, Ray worker mode) | `…/coderai-vllm:latest` |
| `ghcr.io/nextime/coderai-engines` | ds4, colibri, kimi-k3-in-c compiled for sm_80…sm_120a | `…/coderai-engines:latest` |
| `ghcr.io/nextime/coderai-engines-kt` | ktransformers (SGLang + kt-kernel) | `…/coderai-engines-kt:latest` |
| `ghcr.io/nextime/coderai-embeddings` | Embeddings and reranking | `…/coderai-embeddings:latest` |
| `ghcr.io/nextime/coderai-ocr` | Document OCR (docTR, Surya) | `…/coderai-ocr:latest` |
| `ghcr.io/nextime/coderai-ocr-paddle` | PaddleOCR (own CUDA runtime) | `…/coderai-ocr-paddle:latest` |
| `ghcr.io/nextime/coderai-tts` | Text-to-speech (Kokoro, F5-TTS …) | `…/coderai-tts:latest` |
| `ghcr.io/nextime/coderai-tts-xtts` | Coqui XTTS | `…/coderai-tts-xtts:latest` |
| `ghcr.io/nextime/coderai-stt` | Transcription (whisper.cpp, Wav2Vec2, Vosk) | `…/coderai-stt:latest` |
| `ghcr.io/nextime/coderai-stt-nemo` | NVIDIA NeMo Canary / Parakeet | `…/coderai-stt-nemo:latest` |
| `ghcr.io/nextime/coderai-stt-crisper` | CrisperWhisper | `…/coderai-stt-crisper:latest` |
| `ghcr.io/nextime/coderai-speaker` | Diarization and speaker recognition (pyannote) | `…/coderai-speaker:latest` |
| `ghcr.io/nextime/coderai-voice` | Voice cloning and conversion | `…/coderai-voice:latest` |
| `ghcr.io/nextime/coderai-audio` | Music / sound generation, stems | `…/coderai-audio:latest` |
| `ghcr.io/nextime/coderai-faceswap` | Face swap | `…/coderai-faceswap:latest` |

The capability images are the same codebase with only one capability's
dependencies (5–12 GB). They are what a RunPod pod, a `host` machine or a
cluster node that only needs one thing runs; the full image is what a
workstation runs. All of them expose the same API on port 8000 (capability
images) or 8776 (full image), and all of them can be a cluster node.

Every tag is also published with its version (`:0.2.20`); `:latest` moves
only after the image has completed a real request on real hardware.

**Every published image is signed** with [cosign](https://github.com/sigstore/cosign)
against the key pair in `packaging/cosign.pub` (the private half never
leaves the release machine). Check before you run:

```bash
cosign verify --key https://raw.githubusercontent.com/nextime/coderai/master/packaging/cosign.pub \
       ghcr.io/nextime/coderai:latest
```

It prints the signed digest and the `version` / `git` annotations of the
build; a tag whose digest does not verify is not ours (or was re-pushed and
not yet re-signed — `packaging/sign-images.sh` is the release step that
signs, `… verify` the check across all images). Docker can enforce this:
a `policy-controller` / `cosign` admission rule on a cluster, or simply the
command above in whatever pulls.

For an offline machine, pull the image where there is a connection and
move it: `docker save ghcr.io/nextime/coderai:0.2.20 | gzip > coderai.tar.gz`
there, `docker load < coderai.tar.gz` here.

## Linux

Requirements: Docker (or Podman) and, for NVIDIA cards, the NVIDIA
container toolkit. AMD/Intel cards use Vulkan (`--vulkan`).

```bash
docker pull ghcr.io/nextime/coderai:latest
curl -fsSLO https://raw.githubusercontent.com/nextime/coderai/master/packaging/linux/run_oci.sh
chmod +x run_oci.sh
./run_oci.sh --nvidia --data-dir ~/coderai-runtime -d      # first run creates config/, models/, cache/
```

Then open `http://localhost:8776/admin` (first-run admin login is printed in
the log; `./run_oci.sh --help` lists every option: `--vulkan`, `--all`,
`--local` to use an existing `~/.coderai`, `--map` for model directories,
`--upgrade` to pull the production branch in place).

`packaging/linux/run_oci.sh` installs itself as `coderai-docker`
(`~/.local/usr/bin`) on first use, so afterwards it is simply
`coderai-docker --nvidia -d` and `coderai-docker --upgrade`. Add
`--host-network` on a box that should find, or be found by, the others over
mDNS (Settings → Cluster → discovery + a shared cluster token): link-local
multicast never crosses Docker's bridge.

Without the script, the plain `docker run` it builds is:

```bash
docker run --rm --name coderai --ipc=host --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e CODERAI_HOST=0.0.0.0 -e CODERAI_PORT=8776 \
  -p 8776:8776 \
  -v ~/coderai-runtime/config:/config -v ~/coderai-runtime/models:/models -v ~/coderai-runtime/cache:/cache \
  ghcr.io/nextime/coderai:latest
```

## Windows (NVIDIA, via Docker Desktop + WSL2)

CoderAI on Windows is the same Linux image under Docker Desktop's WSL2
backend with CUDA-on-WSL: text, images, video, audio, cluster head or node,
RunPod escalation — everything except the AMD/Intel Vulkan path, which WSL2
does not expose to Linux containers.

`packaging/windows/` holds the installer:

* `CoderAI-Setup.exe` (attached to each GitHub release) — installs the
  launcher, adds it to PATH and the Start menu, and on first run enables
  WSL2, installs Docker Desktop through `winget` if it is missing, checks the
  NVIDIA driver, and pulls the image.
* Or, without the .exe, in an elevated PowerShell:

  ```powershell
  irm https://raw.githubusercontent.com/nextime/coderai/master/packaging/windows/install-coderai.ps1 | iex
  ```

Afterwards: `coderai` (starts, opens the admin page), `coderai -Stop`,
`coderai -Upgrade`, `coderai -Logs`. Data lives under
`%LOCALAPPDATA%\CoderAI` (config, models, cache); keep it on the WSL/NTFS
drive with the most space — `coderai -DataDir D:\coderai`.

Windows notes: `.wslconfig` must give the WSL2 VM enough memory
(`memory=32GB` or more for video models); a Windows box works as a cluster
*head* out of the box, and as a *node*, RPC server or Ray worker once the
ports are published (Docker Desktop is NAT'd; `coderai -Port` and the
node's advertise host are the two settings).

## Other machines' capability images as hosts and nodes

A capability image on a machine you own is a `host` backend or a cluster
node exactly like a pod would be:

```bash
docker run -d --name coderai-images --gpus all -p 8000:8000 \
  -e CODERAI_API_TOKEN=some-token ghcr.io/nextime/coderai-images:latest
```

then on the head: model → *Runs on: a machine of yours*, URL
`http://thatbox:8000`, token `some-token` — or Settings → Cluster → add it
as a node — or give both the same cluster token with discovery on and let
them find each other: on the capability container `--network host -e
CODERAI_CLUSTER_TOKEN=… -e CODERAI_DISCOVERY=1` (and `-e CODERAI_NODE_NAME=…`
to name it; `CODERAI_ADVERTISE_HOST` when it has several addresses). `CODERAI_RPC_SERVER_PORT=50052` on the `coderai-llama` image also
lends its card to a GGUF loaded elsewhere; `CODERAI_RAY_ADDRESS=…` turns the
`coderai-vllm` image into a Ray worker for a multi-node vLLM launch. See
[`cluster.md`](cluster.md) and [`remote-execution.md`](remote-execution.md).
