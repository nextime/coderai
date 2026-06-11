# CoderAI Distribution Plan

> **Purpose of this document.** This is an implementation-ready specification for
> distributing CoderAI as **prebuilt, no-compile, no-venv-recreation artifacts** that run
> on a freshly installed machine after a simple download/extract/install. It is detailed
> enough that a future request of *"implement DISTRIBUTION.md"* can be executed end to end
> without re-deriving the design. When implementing, treat the **file layouts, commands,
> version pins, and CI skeletons below as the contract**; deviate only where a TODO or an
> "Open parameters" item explicitly leaves a choice.

---

## 1. Goals and non-goals

### Goals
- **No compilation on the user's machine.** All native modules are prebuilt in CI.
- **No venv creation / dependency resolution on install.** Ship a self-contained Python
  interpreter with every package already installed.
- **Runs on "any" reasonably modern Linux** (≈ 2021+) and on **Windows 10/11**, after a
  download + extract (Linux tarball), `docker run` (OCI image), or a double-click installer
  (Windows).
- **Three artifacts from one CI pipeline:**
  1. Linux **relocatable tarball** (`.tar.zst`) — extract and run, no Docker.
  2. Linux **OCI image** (Docker/Podman) — most robust GPU path.
  3. Windows **installer** (`.exe`, Inno Setup) — easy install + launcher.
- **Backend coverage: NVIDIA (CUDA) + Vulkan + CPU** (matches today's `all` venv).

### Non-goals
- **Bundling models.** Models are large and `township_output/`-style generated media are
  gitignored. Models are downloaded at runtime via the admin UI / Hugging Face. The
  artifacts ship with an **empty `models/` dir**.
- **Bundling GPU drivers.** The host must provide the GPU **driver** (NVIDIA driver, or a
  Vulkan ICD via mesa/AMDVLK/Intel). See the GPU contract (§4).
- **macOS** (tracked separately; `osxbuild.sh` exists). This doc may be extended later.
- **ROCm/HIP** as a first-class bundled backend (AMD users go through Vulkan).

---

## 2. Why the current artifacts are not portable (constraints)

Established by inspecting the live build environment:

| Fact | Value | Consequence |
|---|---|---|
| Build-host glibc | **2.42** (Debian testing) | Anything compiled/linked here needs glibc ≥ 2.42. Ubuntu 24.04 = 2.39, Debian 12 = 2.36, RHEL 9 = 2.34 → **current binaries won't run there.** Fix: build in an **old-glibc container**. |
| `venv_all` size | **~11 GB** | nvidia wheels 2.7G + torch 1.2G + flash-attn 0.9G + triton 0.6G + onnxruntime 0.43G + llama_cpp 0.3G. Download size is inherent. |
| Python | `/usr/bin/python3.13`, `pyvenv.cfg home=/usr/bin` | A venv has **no stdlib of its own**; it points back at system Python. Copying a venv to a machine lacking that exact 3.13 fails. Fix: ship a **standalone interpreter** (its own stdlib). |
| CUDA runtime | bundled in `site-packages/nvidia/*` (2.7G) | Host needs **only the NVIDIA driver**, not the CUDA toolkit. ✓ |
| Compiled native modules | `llama_cpp`, `stable_diffusion_cpp`, `whispercpp`, `flash_attn`, `causal_conv1d` | These are glibc/ABI-bound and link CUDA/Vulkan. They are what CI must prebuild. |

**The core principle:** portability comes from *where and how you build* (old glibc, standalone
Python), not from a packing trick applied to the current env. `conda-pack`/`venv-pack` of the
current 2.42 env would only run on equally-new distros and is therefore rejected as the primary
mechanism.

---

## 3. Architecture: one pipeline, three artifacts

```
                       ┌─────────────────────────────────────────────┐
                       │  Native-module build (per-OS, old toolchain) │
                       │  → wheels: llama_cpp, stable_diffusion_cpp,  │
                       │           whispercpp  (CUDA + Vulkan)        │
                       └───────────────┬──────────────┬──────────────┘
                                       │ wheels        │ wheels
              ┌────────────────────────┴───┐     ┌────┴───────────────────────┐
   LINUX  →  │ assemble standalone Python  │     │  WINDOWS embeddable Python  │  ← WINDOWS
              │ (python-build-standalone)   │     │  + all wheels (uv)          │
              │ + all wheels (uv)           │     └────┬────────────────────────┘
              └───────┬───────────┬─────────┘          │
                      │           │                    │
            ① tar.zst │   ② OCI image (buildx)         │ ③ Inno Setup .exe
              (extract  │   (docker/podman)              │ (installer + launcher)
               & run)   │                                │
```

Shared design across all three:
- **Standalone Python** (no system Python, no venv recreation).
- **CUDA via pip `nvidia-*` wheels** → host needs only the driver.
- **Vulkan via the host loader** (`libvulkan.so.1` / `vulkan-1.dll`) → host needs a GPU ICD.
- **Models never bundled.**
- The existing `codai/platform_paths.py` already abstracts per-OS paths — rely on it; do not
  hardcode paths in launchers beyond the install root.

---

## 4. GPU / runtime contract (document this for users verbatim)

| Backend | What ships in the artifact | What the host must provide |
|---|---|---|
| **CUDA (NVIDIA)** | `torch` cu-wheels + `nvidia-*` runtime wheels + `llama_cpp`/`sd_cpp`/`whispercpp` compiled with CUDA | **NVIDIA driver only** (no CUDA toolkit). OCI: NVIDIA Container Toolkit. |
| **Vulkan (any GPU)** | native modules compiled with `GGML_VULKAN=ON` / `SD_VULKAN` | Host **Vulkan loader + ICD**: Linux `libvulkan1` + `mesa-vulkan-drivers`/AMDVLK/NVIDIA; Windows `vulkan-1.dll` (driver-provided, always present). |
| **CPU** | CPU paths of every module | Nothing beyond glibc / VC++ runtime. Always the fallback. |

Never bundle the GPU driver or the host Vulkan ICD. Bundling the Vulkan **loader** is allowed
but discouraged (prefer dlopen of the host's, so the host's ICD is found).

---

## 5. Shared building blocks

### 5.1 Standalone Python (interpreter source of truth)
- Use **`python-build-standalone`** (astral / `indygreg`) release matching the repo's Python
  minor (currently **3.13**). Use the **`install_only`** variant.
  - Linux x86_64: `cpython-3.13.<patch>+<date>-x86_64-unknown-linux-gnu-install_only.tar.gz`
  - Windows x86_64: `cpython-3.13.<patch>+<date>-x86_64-pc-windows-msvc-install_only.tar.gz`
- These are **relocatable** and contain their own stdlib. Pin the exact release in
  `packaging/versions.env` (see §9).
- Install packages into it with **`uv pip install`** (fast, deterministic, offline-capable
  from a wheel cache). `uv` is fetched as a standalone binary in CI.

### 5.2 Native-module wheels (the only things CI compiles)
Compile each into a wheel (`uv pip wheel` / `pip wheel`) using the same CMAKE flags the current
`build.sh` uses:

| Module | Linux CMAKE_ARGS | Windows CMAKE_ARGS | Notes |
|---|---|---|---|
| `llama-cpp-python` | `-DGGML_VULKAN=ON -DGGML_CUDA=ON` | `-DGGML_VULKAN=ON -DGGML_CUDA=ON` | one wheel covers CUDA+Vulkan+CPU |
| `stable-diffusion-cpp-python` | `-DSD_VULKAN=ON -DSD_CUDA=ON -DSD_WEBM=OFF` (or `-DSD_USE_SYSTEM_WEBM=ON`) | `-DSD_CUDA=ON -DSD_VULKAN=ON -DSD_WEBM=OFF` | the libwebm submodule fix is mandatory (see build.sh:94-101) |
| `whispercpp` | `-DWHISPER_VULKAN=ON -DGGML_VULKAN=ON` | `-DWHISPER_VULKAN=ON -DGGML_VULKAN=ON` | built from source in build.sh:288 |

**Optional/extras (NOT in base bundle):** `flash-attn` (~0.9G, CUDA-arch-specific, slow/fragile
on old glibc), `bitsandbytes`, `causal_conv1d`, `turboquant-py[torch]` (optional TurboQuant
embedding-quantization backend; the built-in NumPy backend works without it). Ship these as a
separate **"cuda-extras"** download/layer the user can opt into; the app must already degrade
gracefully without them.

### 5.3 Pure/prebuilt wheels (no compilation)
`torch`, `torchvision`, `torchaudio`, `nvidia-*`, `triton`, `transformers`, `diffusers`,
`accelerate`, `onnxruntime`(-gpu), `insightface`, `sentence-transformers`, plus everything in
`requirements.txt` / `requirements-nvidia.txt` / `requirements-vulkan.txt`. These resolve to
manylinux/win wheels from PyPI + the PyTorch index. No build step.

### 5.4 Build base images (old glibc = forward compatibility)
- **Primary build base: glibc 2.31** (Debian 11 "bullseye" or `manylinux_2_31` equivalent).
  Runs on Ubuntu 20.04+, Debian 11+, RHEL 9+ — ≈ everything from 2021. Easier to install the
  Vulkan SDK + CUDA toolkit than older bases.
- **Fallback for wider reach: `manylinux_2_28`** (glibc 2.28). Older toolchain; use only if a
  user needs pre-2021 distros.
- Pin the chosen base digest in `packaging/versions.env`.

---

## 6. Repository layout to create

```
packaging/
  versions.env                # pinned versions (python-build-standalone, uv, cuda, vulkan sdk, base image digests)
  common/
    requirements.lock         # fully pinned, hash-locked resolution (uv pip compile output)
    assemble_env.sh           # shared: lay down standalone python + uv pip install wheels + app  (Linux/macOS)
    assemble_env.ps1          # shared: Windows equivalent
    app_payload.txt           # include/exclude globs for the codai/ app copy (excludes models, __pycache__, township_output, venv_*)
  linux/
    Dockerfile.build          # old-glibc + CUDA + Vulkan SDK → compiles native wheels into /wheels
    build_native_wheels.sh    # invoked inside Dockerfile.build
    make_tarball.sh           # assemble standalone-python bundle → coderai-linux-x64.tar.zst
    launcher/coderai          # runtime launcher (sets PYTHONHOME/LD_LIBRARY_PATH, execs python coderai)
    Dockerfile.runtime        # slim runtime image (artifact ②)
  windows/
    build_native_wheels.ps1   # MSVC + CUDA + Vulkan SDK → native wheels
    make_bundle.ps1           # assemble embeddable python + wheels + app  → staging dir
    installer.iss             # Inno Setup script (artifact ③)
    launcher/coderai-launcher.ps1   # starts server, opens browser to admin UI
  ci/
    .github/workflows/release.yml   # orchestrates all three (copy into .github/workflows/ when implementing)
DISTRIBUTION.md               # this file
```

> When implementing, **create `packaging/` and move the new scripts there**; do not bloat the
> existing top-level `build.sh`/`build.ps1` (those remain the *developer* build path). The
> distribution scripts are a separate, CI-oriented track that consumes prebuilt wheels.

---

## 7. Artifact ① — Linux relocatable tarball

### 7.1 Final layout (what the user extracts)
```
coderai/
  python/        # python-build-standalone tree: bin/python3, lib/python3.13/, incl. all site-packages
  app/           # copy of repo (codai/, coderai launcher, templates, static) minus excludes
  models/        # empty; CODERAI_MODELS_DIR points here by default
  bin/coderai    # launcher (chmod +x)
  VERSION
  README-RUN.txt
```

### 7.2 Launcher `bin/coderai` (exact behavior)
```sh
#!/bin/sh
# Resolve the install root from this script's location (handles symlinks).
HERE="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
export PYTHONHOME="$HERE/python"
# Bundled CUDA runtime libs ship inside the nvidia/* wheels:
NV="$HERE/python/lib/python3.13/site-packages/nvidia"
LIBS="$HERE/python/lib"
if [ -d "$NV" ]; then
  for d in "$NV"/*/lib; do LIBS="$LIBS:$d"; done
fi
export LD_LIBRARY_PATH="$LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Keep all state inside the bundle by default. codai/platform_paths.py honors the XDG
# vars (user_config_dir/user_data_dir/user_cache_dir), so point them at the install root
# unless the user already set them. (If a unified CODERAI_HOME is added app-side later,
# set that instead — see §12.)
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HERE/config}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$HERE/data}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HERE/cache}"
exec "$HERE/python/bin/python3" "$HERE/app/coderai" "$@"
```
- Do **not** rely on `activate`; call the interpreter directly with `PYTHONHOME` set.
- RPATHs of the compiled native `.so`s should be `$ORIGIN`-relative (patchelf in CI if needed)
  so they find their sibling ggml/CUDA libs without `LD_LIBRARY_PATH` gymnastics; the
  `LD_LIBRARY_PATH` above is the belt-and-suspenders for the nvidia wheels.

### 7.3 Build steps (`packaging/linux/make_tarball.sh`)
1. `docker build -f packaging/linux/Dockerfile.build` → produces `/wheels/*.whl` (native) +
   exports them to the host (`docker create` + `docker cp`, or a buildx `--output`).
2. Download + extract `python-build-standalone` into `coderai/python/`.
3. Fetch `uv` static binary.
4. `uv pip install --python coderai/python/bin/python3 -r packaging/common/requirements.lock
   --find-links /wheels` (native wheels resolved locally; the rest from PyPI + torch index).
5. Copy app via `packaging/common/app_payload.txt` includes into `coderai/app/`.
6. Drop in `bin/coderai`, `VERSION`, `README-RUN.txt`; `mkdir coderai/models`.
7. **Prune**: `__pycache__`, `*.pyi` test dirs, `pip`, `*.dist-info/RECORD` optional, `.a`
   static libs, duplicate `nvidia/*` headers — to shrink. (Document expected size ≈ 4–5 GB
   zstd.)
8. `tar --zstd -cf dist/coderai-linux-x64.tar.zst coderai/` + `sha256sum`.

### 7.4 User experience
```sh
tar --zstd -xf coderai-linux-x64.tar.zst
./coderai/bin/coderai            # starts server; prints http://127.0.0.1:<port>/admin
```
README-RUN.txt documents: NVIDIA → install driver; AMD/Intel Vulkan →
`sudo apt install libvulkan1 mesa-vulkan-drivers`; no GPU → works on CPU.

---

## 8. Artifact ② — Linux OCI image

### 8.1 Dockerfile.runtime (multi-stage; reuse the wheels from §7 step 1)
```dockerfile
# ---- stage 1: wheels come from Dockerfile.build (or COPY --from a wheels image) ----
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      libvulkan1 mesa-vulkan-drivers libgomp1 ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# standalone python + installed packages assembled exactly like the tarball's coderai/python:
COPY --from=assembler /opt/coderai /opt/coderai
ENV PYTHONHOME=/opt/coderai/python \
    PATH=/opt/coderai/python/bin:$PATH \
    XDG_CONFIG_HOME=/config \
    XDG_DATA_HOME=/models \
    XDG_CACHE_HOME=/cache
# (If a unified CODERAI_HOME override is added app-side per §12, set it here instead.)
EXPOSE 8776
VOLUME ["/models", "/config", "/cache"]
ENTRYPOINT ["/opt/coderai/python/bin/python3", "/opt/coderai/app/coderai"]
```
- Base is plain Debian slim (CUDA runtime arrives via pip nvidia wheels). NVIDIA Container
  Toolkit injects the driver at `--gpus all`.
- Provide CPU/Vulkan-only and CUDA variants if size matters (tag suffixes `:cpu`, `:cuda`).

### 8.2 Run
```sh
# NVIDIA
docker run --gpus all -p 8776:8776 -v $PWD/models:/models -v $PWD/config:/config ghcr.io/<org>/coderai:latest
# AMD/Intel via Vulkan
docker run --device /dev/dri -p 8776:8776 -v $PWD/models:/models ghcr.io/<org>/coderai:latest
# CPU
docker run -p 8776:8776 -v $PWD/models:/models ghcr.io/<org>/coderai:latest
```
Publish to **GHCR**; also `docker save | zstd` a loadable tarball attached to the Release for
offline/Podman use.

---

## 9. Artifact ③ — Windows installer

### 9.1 Strategy
Prebuild native wheels on a `windows-latest` CI runner (MSVC Build Tools + CUDA toolkit +
Vulkan SDK), assemble a standalone-Python bundle, wrap in an **Inno Setup** installer. No
compiler on the user's machine.

### 9.2 Bundle assembly (`packaging/windows/make_bundle.ps1`)
1. `build_native_wheels.ps1` compiles `llama-cpp-python`, `stable-diffusion-cpp-python`,
   `whispercpp` with the CMAKE flags from §5.2 → `wheels\`.
2. Extract `python-build-standalone` Windows `install_only` into `staging\python\`.
3. `uv pip install --python staging\python\python.exe -r packaging\common\requirements.lock
   --find-links wheels`.
4. Copy app (per `app_payload.txt`) into `staging\app\`.
5. Add `staging\models\` (empty), `staging\coderai-launcher.ps1`, icon, VERSION.

### 9.3 Launcher (`coderai-launcher.ps1` → wrapped as `coderai.exe` via a tiny shim or `ps2exe`)
- Sets `PYTHONHOME=<install>\python`, `PATH` to include `python` + bundled `nvidia\*\bin`.
- Starts `python.exe app\coderai` minimized (or as a background process / tray icon).
- Waits for the port, then `Start-Process http://127.0.0.1:<port>/admin`.
- A second shortcut "Stop CoderAI" kills the process.

### 9.4 Inno Setup (`installer.iss`) requirements
- Install to `{localappdata}\Programs\CoderAI` (no admin) **or** `{autopf}\CoderAI` (admin) —
  default to per-user to avoid UAC.
- Bundle + silently install the **VC++ 2015-2022 redistributable** if absent (check the
  registry key; run `vc_redist.x64.exe /quiet /norestart`).
- Create Start-Menu group + desktop shortcut → the launcher.
- Register an uninstaller (Inno does this automatically); clean removal of the whole tree.
- Two installer SKUs:
  - **Offline** `CoderAI-Setup-x64.exe` (~5–8 GB) — everything embedded.
  - **Web/stub** `CoderAI-WebSetup-x64.exe` (small) — downloads the GPU payload (the big
    `nvidia-*`/torch wheels) on first run from the Release assets; good for users who don't
    need GPU or want a small download. Implemented via Inno's `[Code]` + `idpDownload` or a
    first-run step in the launcher.
- GPU notes shown on the finish page: NVIDIA driver for CUDA; Vulkan works via any modern GPU
  driver (vulkan-1.dll is driver-provided).

### 9.5 Windows GPU coverage (confirmed feasible)
- **CUDA**: identical to Linux — CUDA DLLs ship in the pip `nvidia-*`/torch cu-wheels; host
  needs only the NVIDIA driver.
- **Vulkan**: `vulkan-1.dll` is provided by every GPU driver (NVIDIA/AMD/Intel) in System32 →
  the Vulkan-compiled `.pyd` works across vendors.
- **CPU**: always.

### 9.6 Rejected Windows options (record so we don't revisit)
- **PyInstaller `--onefile`** (current `build.ps1 --package`): fragile with torch/diffusers,
  ~11 GB self-extract per launch, AV false positives. Keep only as a fallback build.
- **MSIX**: sandbox fights the large ML stack + GPU access.
- **WSL2 + reuse Linux artifacts**: easiest to produce, GPU works (CUDA in WSL2), but "enable
  WSL2" is not double-click-easy → document as a power-user path, not the default.

---

## 10. CI pipeline (`packaging/ci/.github/workflows/release.yml`)

Trigger: `on: push: tags: ['v*']` + manual `workflow_dispatch`.

Jobs:
1. **lock** — `uv pip compile` the merged requirements (`requirements.txt` +
   `requirements-nvidia.txt` + `requirements-vulkan.txt` + the native modules as `--find-links`
   placeholders) into `packaging/common/requirements.lock` (hash-pinned). Cache it.
2. **linux-native-wheels** — build `packaging/linux/Dockerfile.build` (glibc 2.31 + CUDA +
   Vulkan SDK); export `/wheels`. Cache by hash of the three native module versions.
3. **linux-tarball** — needs (1)(2); run `make_tarball.sh`; upload `*.tar.zst` + `.sha256`.
4. **linux-image** — needs (1)(2); `docker buildx build --push` to GHCR (`:latest`, `:vX.Y.Z`,
   `:cpu`, `:cuda`); also `docker save | zstd` artifact.
5. **windows-native-wheels** — `windows-latest`; install MSVC + CUDA + Vulkan SDK (chocolatey /
   official installers, pinned); run `build_native_wheels.ps1`; upload `wheels\`.
6. **windows-installer** — needs (5); `make_bundle.ps1` then compile `installer.iss` with the
   Inno Setup CLI (`iscc`); upload both installer SKUs.
7. **release** — needs (3)(4)(6); create/attach all assets to the GitHub Release; publish
   checksums + this doc's "GPU contract" as release notes.

Caching: key the native-wheel jobs on `(module versions, base image digest, python version)` so
they only rebuild when those change — the expensive compile happens rarely.

---

## 11. Version pins (`packaging/versions.env`)

Fill these at implementation time; keep them the single source of truth referenced by every
script.
```
PYTHON_VERSION=3.13                # must match repo (currently 3.13.12)
PBS_RELEASE=                       # python-build-standalone tag, e.g. 20250xxx
UV_VERSION=                        # astral uv pinned version
LINUX_BUILD_BASE=                  # e.g. debian:11-slim@sha256:...  (glibc 2.31)
CUDA_VERSION=                      # toolkit used to compile native wheels (match torch cuXX)
VULKAN_SDK_VERSION=                # LunarG SDK pinned
LLAMA_CPP_PYTHON_VERSION=
SD_CPP_PYTHON_VERSION=
WHISPERCPP_REF=                    # git ref for whisper.cpp python bindings
INNO_SETUP_VERSION=
```
Keep `CUDA_VERSION` consistent with the `torch` cuXX wheels (e.g. torch cu124 ↔ CUDA 12.4) so
the bundled `nvidia-*` runtime matches what the native modules were compiled against.

---

## 12. App-side changes likely needed (verify during implementation)

- **`coderai` entry** already works as `python coderai` (it imports `codai.main.main`). Confirm
  it runs headless with no extra args and binds `config.server.host:port`.
- **Config/models dir.** Today `codai/platform_paths.py` derives locations from the **XDG**
  vars on Linux (`XDG_CONFIG_HOME`/`XDG_DATA_HOME`/`XDG_CACHE_HOME`, with
  `user_config_dir()`/`user_data_dir()`/`user_cache_dir()`) and from Windows dir vars via
  `_windows_dir(env_var, fallback)`. `codai/config.py` builds everything under a single
  `config_dir` (`config.json`, `models.json`, `auth.json`, `pipelines.json`). The launchers
  in §7/§9 therefore point the **XDG vars** (Linux) / the Windows dir vars at the install
  root — this works **without any app change**. *Optional improvement:* add a single
  `CODERAI_HOME` env var that overrides all of them at once (cleaner for the container's
  `/config` + `/models` volumes); if added, update the launchers to set it instead. Verify
  how `main.py` chooses `config_dir` (CLI arg vs default) so the env override actually wins.
- **First-run UX**: with empty `models/`, the server must start and the admin UI must let the
  user download a model (it already has `/admin/api/model-download`). Verify no model is
  required at boot.
- **Graceful absence of extras** (`flash-attn`, `bitsandbytes`): confirm import guards so the
  base bundle runs without them.
- **Whisper server binary**: build.sh also compiles a standalone `whisper-server`; decide
  whether the bundle ships it or uses the in-process `whispercpp` wheel. Prefer the wheel to
  avoid shipping a second native binary.

---

## 13. Verification matrix (acceptance criteria)

Run after building each artifact. "Pass" = server boots, `/v1/models` responds, a tiny CPU
text generation and a small image generation succeed.

| Target | Host | GPU path to verify |
|---|---|---|
| tarball | Ubuntu 22.04 clean | CPU; NVIDIA (driver only); AMD Vulkan (mesa) |
| tarball | Debian 12 clean | CPU; NVIDIA |
| tarball | Fedora/RHEL 9 clean | CPU (glibc 2.34 ≥ build glibc) |
| OCI image | Docker + NVIDIA Container Toolkit | `--gpus all` CUDA |
| OCI image | Podman | CPU + `--device /dev/dri` Vulkan |
| Windows installer | Win 11 clean, NVIDIA | CUDA + browser auto-open |
| Windows installer | Win 10, AMD | Vulkan + CPU |

Each must require **no compiler, no pip, no manual venv** on the target. Capture install size
and cold-start time.

---

## 14. Open parameters (decide at implementation, defaults in **bold**)

1. Build glibc floor: **2.31 (Debian 11)** vs 2.28 (manylinux_2_28, wider/older).
2. `flash-attn`/`bitsandbytes`: **separate cuda-extras download** vs in-base (bigger).
3. Windows installer SKUs: **ship both** offline + web-stub vs offline only.
4. OCI variants: **single `all` image** first; add `:cpu`/`:cuda` slims if size complaints.
5. Compression: **zstd** (fast, good ratio) for the Linux tarball.
6. Registry: **GHCR**; mirror to Docker Hub optional.
7. Models dir default: **inside install root** (`coderai/models`) overridable by env/volume.

---

## 15. Out-of-scope follow-ups (note for later)
- macOS `.dmg` / notarized app (Metal backend; `osxbuild.sh` is the starting point).
- ARM64 builds (Linux aarch64 tarball + image) — same design, different base + wheels.
- Auto-update channel for the tarball/installer.
- Signed Windows installer (code-signing cert) to avoid SmartScreen warnings.
