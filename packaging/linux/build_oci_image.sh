#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSIONS_FILE="$ROOT_DIR/packaging/versions.env"
if [[ -f "$VERSIONS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$VERSIONS_FILE"
fi

DOCKER_BIN="${DOCKER:-docker}"
read -r -a DOCKER_CMD <<< "$DOCKER_BIN"
IMAGE_TAG="${OCI_IMAGE:-coderai:local}"
BUILD_MODE="from-scratch"
VENV_PATH=""
INCLUDE_LOCAL_LIBS=1
AUTO_LOCAL_BINS=1
LOCAL_BINARIES=()
LOCAL_BINARY_DIRS=()
# Optional second venv for Parler-TTS (pinned transformers 4.46). Bundled as an
# overlay whose site-packages is prepended to PYTHONPATH at runtime, shadowing the
# main stack while torch/etc resolve from it underneath.
PARLER_VENV="${CODERAI_PARLER_VENV:-$HOME/.coderai/parler_venv}"
INCLUDE_PARLER=1
# Isolated lip-sync tools (Python 3.10 venvs + repos + weights) and the ds4 native
# engine. Bundled so the image replicates the local install. ds4 DeepSeek-V4 GGUF
# weights are NOT bundled (multi-GB, runtime-downloaded into a volume).
INCLUDE_TOOLS=1
# One shared Python 3.10 venv serves both wav2lip and sadtalker (identical torch),
# halving the torch footprint. Repo code is bundled WITHOUT model weights — those
# download on first lip-sync use into the /cache volume.
LIPSYNC_VENV="${CODERAI_LIPSYNC_VENV:-$HOME/.coderai/lipsync_venv}"
WAV2LIP_DIR="${CODERAI_WAV2LIP_SRC:-$HOME/.coderai/Wav2Lip}"
SADTALKER_DIR="${CODERAI_SADTALKER_SRC:-$HOME/.coderai/SadTalker}"
DS4_DIR="${CODERAI_DS4_DIR:-$HOME/.coderai/ds4}"
# After a successful build, export the image and assemble the final distribution
# bundle (image tarball + install.sh + coderai-docker runner). Disable with
# --no-dist (just builds the image).
MAKE_DIST=1
# After a successful build (and dist export), prune the dangling/intermediate
# images this build left behind (untagged <none> layers from the multi-stage
# build). Off with --no-prune. Only dangling images are removed — never tagged
# images (the final image, the pulled CUDA base, or anything else you have).
PRUNE_BUILD_IMAGES=1
# --versioned auto-derives a structured image tag from the app version + build
# mode + target GPU: full_<gpu>_<version> for a local-venv build, base_<gpu>_<version>
# for a from-scratch build (e.g. coderai:full_all_0.1.0). GPU_SEL is a label only —
# the image bundles CUDA + Vulkan regardless — so pick what you intend to ship.
VERSIONED=0
GPU_SEL="all"   # all | nvidia | vulkan

usage() {
  cat <<'EOF'
Usage:
  ./build-oci.sh [IMAGE_TAG]
  ./build-oci.sh --from-venv [IMAGE_TAG]
  ./build-oci.sh --venv PATH [IMAGE_TAG]

Options:
  --from-scratch          Build the full OCI image from pinned packages and native wheels (default).
  --from-venv             Build from the currently activated virtualenv ($VIRTUAL_ENV).
  --venv PATH             Build from a specific local virtualenv.
  --no-local-libs         Do not copy ldd-discovered local native libraries from the venv.
  --no-auto-local-bins    Do not auto-include known locally compiled helper binaries.
  --include-local-bin PATH
                          Copy an extra tested local binary into the image, including its ldd libs.
                          Can be repeated. Useful for tools such as whisper-server.
  --include-local-dir PATH
                          Copy executable files from a local build directory, including ldd libs.
                          Can be repeated. Useful for local whisper.cpp build/bin directories.
  --parler-venv PATH      Bundle this Parler-TTS venv as an overlay (default:
                          $CODERAI_PARLER_VENV or ~/.coderai/parler_venv if present).
  --no-parler             Do not bundle the Parler-TTS venv overlay.
  --no-tools              Do not bundle the lip-sync (wav2lip/sadtalker) venvs or ds4.
  -t, --tag TAG           Image tag to create (default: coderai:local or OCI_IMAGE from versions.env).
  --versioned             Auto-derive the tag from the app version + build mode + GPU:
                            local-venv build -> coderai:full_<gpu>_<version>
                            from-scratch     -> coderai:base_<gpu>_<version>
                          (overrides -t/--tag and the default). <version> comes from
                          codai.__version__; <gpu> from --gpu (default all).
  --gpu all|nvidia|vulkan Target-GPU label used by --versioned (default: all). Label
                          only — the image always bundles both CUDA and Vulkan.
  --no-dist               Just build the image; skip exporting + bundling for distribution.
  --dist                  Force building the distribution bundle (default ON).
  --no-prune              Keep the dangling/intermediate images the build created.
  --prune                 Prune dangling build images after building (default ON).
  -h, --help              Show this help.

Examples:
  ./build-oci.sh
  source venv_all/bin/activate && ./build-oci.sh --from-venv coderai:venv
  ./build-oci.sh --venv ./venv_all -t coderai:venv
  ./build-oci.sh --venv ./venv_all --include-local-bin /usr/local/bin/whisper-server -t coderai:venv
  ./build-oci.sh --venv ./venv_all --include-local-dir ~/whisper.cpp/build/bin -t coderai:venv
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-scratch)
      BUILD_MODE="from-scratch"
      shift
      ;;
    --from-venv)
      BUILD_MODE="venv"
      VENV_PATH="${VIRTUAL_ENV:-}"
      shift
      ;;
    --venv)
      BUILD_MODE="venv"
      if [[ $# -lt 2 ]]; then
        echo "Error: --venv requires a path" >&2
        exit 2
      fi
      VENV_PATH="$2"
      shift 2
      ;;
    --no-local-libs)
      INCLUDE_LOCAL_LIBS=0
      shift
      ;;
    --no-auto-local-bins)
      AUTO_LOCAL_BINS=0
      shift
      ;;
    --parler-venv)
      BUILD_MODE="venv"
      if [[ $# -lt 2 ]]; then
        echo "Error: --parler-venv requires a path" >&2
        exit 2
      fi
      PARLER_VENV="$2"
      INCLUDE_PARLER=1
      shift 2
      ;;
    --no-parler)
      INCLUDE_PARLER=0
      shift
      ;;
    --no-tools)
      INCLUDE_TOOLS=0
      shift
      ;;
    --include-local-bin)
      BUILD_MODE="venv"
      if [[ $# -lt 2 ]]; then
        echo "Error: --include-local-bin requires a path" >&2
        exit 2
      fi
      LOCAL_BINARIES+=("$2")
      shift 2
      ;;
    --include-local-dir)
      BUILD_MODE="venv"
      if [[ $# -lt 2 ]]; then
        echo "Error: --include-local-dir requires a path" >&2
        exit 2
      fi
      LOCAL_BINARY_DIRS+=("$2")
      shift 2
      ;;
    -t|--tag)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires an image tag" >&2
        exit 2
      fi
      IMAGE_TAG="$2"
      shift 2
      ;;
    --versioned)
      VERSIONED=1
      shift
      ;;
    --gpu)
      if [[ $# -lt 2 ]]; then
        echo "Error: --gpu requires one of: all nvidia vulkan" >&2
        exit 2
      fi
      case "$2" in
        all|nvidia|vulkan) GPU_SEL="$2" ;;
        *) echo "Error: --gpu must be one of: all nvidia vulkan (got '$2')" >&2; exit 2 ;;
      esac
      shift 2
      ;;
    --no-dist)
      MAKE_DIST=0
      shift
      ;;
    --no-prune)
      PRUNE_BUILD_IMAGES=0
      shift
      ;;
    --prune)
      PRUNE_BUILD_IMAGES=1
      shift
      ;;
    --dist)
      MAKE_DIST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      IMAGE_TAG="$1"
      shift
      ;;
  esac
done

# --versioned: derive coderai:<prefix>_<gpu>_<version>. prefix=full for a local
# venv build (the running stack), base for a from-scratch build. Overrides any
# -t/--tag or positional tag, since the whole point is a deterministic name.
if [[ "$VERSIONED" == "1" ]]; then
  APP_VERSION="$(cd "$ROOT_DIR" && python3 - <<'PY'
import re, pathlib, sys
txt = pathlib.Path("codai/__init__.py").read_text()
m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', txt)
print(m.group(1) if m else "")
PY
)"
  if [[ -z "$APP_VERSION" ]]; then
    echo "Error: --versioned could not read __version__ from codai/__init__.py" >&2
    exit 2
  fi
  if [[ "$BUILD_MODE" == "venv" ]]; then PREFIX="full"; else PREFIX="base"; fi
  IMAGE_TAG="coderai:${PREFIX}_${GPU_SEL}_${APP_VERSION}"
  echo "[build] --versioned -> $IMAGE_TAG"
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.13.5}"
PBS_RELEASE="${PBS_RELEASE:-20250612}"
UV_VERSION="${UV_VERSION:-0.7.13}"
CUDA_VERSION="${CUDA_VERSION:-12.4.1}"
UBUNTU_VERSION="${UBUNTU_VERSION:-22.04}"
WHISPERCPP_REF="${WHISPERCPP_REF:-master}"
LLAMA_CPP_PYTHON_VERSION="${LLAMA_CPP_PYTHON_VERSION:-}"
SD_CPP_PYTHON_VERSION="${SD_CPP_PYTHON_VERSION:-}"
# SageAttention (INT8 video attention): built from source (CUDA-arch-sensitive).
# Default ON for the RTX 3090 arch (8.6); non-fatal in the Dockerfile so a build
# on another arch/CUDA just falls back to flash/SDPA at runtime.
BUILD_SAGEATTENTION="${BUILD_SAGEATTENTION:-1}"
SAGEATTENTION_REF="${SAGEATTENTION_REF:-main}"
SAGEATTENTION_ARCH="${SAGEATTENTION_ARCH:-8.6}"

write_build_manifest() {
  local mode="$1"
  local manifest_python="${2:-}"
  local venv_path="${3:-}"
  local local_bins_joined=""
  if [[ ${#LOCAL_BINARIES[@]} -gt 0 ]]; then
    local IFS=:
    local_bins_joined="${LOCAL_BINARIES[*]}"
  fi
  MANIFEST_OUT="$ROOT_DIR/.packaging-cache/build-manifest.json" \
  PROJECT_ROOT="$ROOT_DIR" \
  MANIFEST_ARTIFACT="oci-image" \
  MANIFEST_BUILD_MODE="$mode" \
  MANIFEST_PYTHON="$manifest_python" \
  MANIFEST_VENV="$venv_path" \
  MANIFEST_LOCAL_BINS="$local_bins_joined" \
  PYTHON_VERSION="$PYTHON_VERSION" \
  PBS_RELEASE="$PBS_RELEASE" \
  UV_VERSION="$UV_VERSION" \
  CUDA_VERSION="$CUDA_VERSION" \
  UBUNTU_VERSION="$UBUNTU_VERSION" \
    python3 "$ROOT_DIR/packaging/common/write_manifest.py"
}

add_local_binary() {
  local path="$1"
  local abs_path
  if [[ ! -x "$path" || ! -f "$path" ]]; then
    return 0
  fi
  abs_path="$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
  for existing in "${LOCAL_BINARIES[@]}"; do
    [[ "$existing" == "$abs_path" ]] && return 0
  done
  LOCAL_BINARIES+=("$abs_path")
}

discover_local_binaries() {
  [[ "$AUTO_LOCAL_BINS" == "1" ]] || return 0

  # Common locations used by the existing developer build path. Python extension
  # modules are already copied with the venv; this targets standalone helper
  # binaries that were compiled locally and tested outside Docker.
  local candidates=(
    "/usr/local/bin/whisper-server"
    "/usr/local/bin/whisper-cli"
    "$HOME/whisper.cpp/build/bin/whisper-server"
    "$HOME/whisper.cpp/build/bin/whisper-cli"
    "$HOME/whisper.cpp/build/bin/main"
    "$HOME/whisper.cpp/build/bin/server"
    "/usr/local/bin/ds4-server"
    "${CODERAI_DS4_DIR:-$HOME/.coderai/ds4}/ds4-server"
    "/usr/local/bin/rife-ncnn-vulkan"
    "$HOME/.local/bin/rife-ncnn-vulkan"
  )
  local path
  for path in "${candidates[@]}"; do
    add_local_binary "$path"
  done
}

prepare_venv_bundle() {
  local venv="$1"
  local bundle="$2"
  local include_libs="$3"
  rm -rf "$bundle"
  mkdir -p "$bundle/venv" "$bundle/local-libs" "$bundle/local-bin"

  echo "Preparing venv bundle: $bundle"
  cp -a "$venv/." "$bundle/venv/"

  discover_local_binaries

  local dir_path
  for dir_path in "${LOCAL_BINARY_DIRS[@]}"; do
    if [[ ! -d "$dir_path" ]]; then
      echo "Error: local binary directory does not exist: $dir_path" >&2
      exit 2
    fi
    while IFS= read -r -d '' found_bin; do
      add_local_binary "$found_bin"
    done < <(find "$dir_path" -maxdepth 2 -type f -perm -111 -print0)
  done

  for bin_path in "${LOCAL_BINARIES[@]}"; do
    if [[ ! -x "$bin_path" ]]; then
      echo "Error: local binary is not executable: $bin_path" >&2
      exit 2
    fi
    dest_path="$bundle/local-bin/$(basename "$bin_path")"
    if [[ -e "$dest_path" && "$bin_path" -ef "$dest_path" ]]; then
      continue
    fi
    # -L: dereference symlinks so the REAL binary is bundled. /usr/local/bin
    # entries are often symlinks to a build dir (e.g. ~/whisper.cpp/build/bin);
    # copying the link verbatim leaves a dangling symlink in the image.
    cp -aL --remove-destination "$bin_path" "$dest_path"
  done

  if [[ ${#LOCAL_BINARIES[@]} -gt 0 ]]; then
    printf 'Included local compiled binaries:\n'
    printf '  %s\n' "${LOCAL_BINARIES[@]}"
  fi

  # Parler-TTS overlay venv. Only its site-packages is needed: it was created with
  # --system-site-packages, so it holds just the pinned overrides (transformers
  # 4.46, parler_tts, tokenizers, ...); torch/numpy resolve from the main venv.
  if [[ "$INCLUDE_PARLER" == "1" && -n "$PARLER_VENV" && -d "$PARLER_VENV" ]]; then
    local parler_sp
    parler_sp="$PARLER_VENV/lib/python${VENV_PYTHON_MINOR}/site-packages"
    if [[ -d "$parler_sp" ]]; then
      mkdir -p "$bundle/parler-venv/site-packages"
      rsync -a --delete "$parler_sp/" "$bundle/parler-venv/site-packages/"
      echo "Bundled Parler-TTS overlay from: $parler_sp"
    else
      echo "Warning: --parler-venv given but no site-packages at $parler_sp (skipping)" >&2
    fi
  fi

  # Isolated lip-sync tools + ds4 engine. The two venvs share one standalone
  # Python 3.10 (read from a venv's pyvenv.cfg `home`); it's bundled once and the
  # venvs are re-pointed at it during the image build.
  if [[ "$INCLUDE_TOOLS" == "1" ]]; then
    local py310_dir=""
    if [[ -f "$LIPSYNC_VENV/pyvenv.cfg" ]]; then
      local home_bin
      home_bin="$(sed -n 's/^home *= *//p' "$LIPSYNC_VENV/pyvenv.cfg" | head -1)"
      [[ -n "$home_bin" ]] && py310_dir="$(dirname "$home_bin")"
    fi
    if [[ -n "$py310_dir" && -d "$py310_dir" ]]; then
      mkdir -p "$bundle/py310"
      rsync -a "$py310_dir/" "$bundle/py310/"
      echo "Bundled standalone Python 3.10 from: $py310_dir"
    else
      echo "Warning: could not locate the py3.10 interpreter for the lip-sync venv" >&2
    fi
    local _venv_excl=(--exclude '__pycache__' --exclude '*.pyc' --exclude 'pip/' --exclude '*.dist-info/RECORD')
    if [[ -d "$LIPSYNC_VENV" ]]; then rsync -a "${_venv_excl[@]}" "$LIPSYNC_VENV/" "$bundle/lipsync_venv/"; echo "Bundled shared lip-sync venv"; fi
    # Repo CODE ONLY — checkpoints/weights are excluded and download at runtime.
    if [[ -d "$WAV2LIP_DIR" ]]; then rsync -a --exclude 'checkpoints/' --exclude 'face_detection/detection/sfd/*.pth' "$WAV2LIP_DIR/" "$bundle/Wav2Lip/"; echo "Bundled Wav2Lip code (no weights)"; fi
    if [[ -d "$SADTALKER_DIR" ]]; then rsync -a --exclude 'checkpoints/*' --exclude 'gfpgan/weights/*' "$SADTALKER_DIR/" "$bundle/SadTalker/"; echo "Bundled SadTalker code (no weights)"; fi
    # ds4: binary + scripts, minus any downloaded multi-GB GGUF weights.
    if [[ -d "$DS4_DIR" ]]; then
      rsync -a --exclude 'gguf/' --exclude '*.gguf' --exclude '*.gguf.*' "$DS4_DIR/" "$bundle/ds4/"
      echo "Bundled ds4 (binary + scripts, no weights)"
    fi
  fi

  if [[ "$include_libs" != "1" ]]; then
    return 0
  fi

  VENV_BUNDLE="$bundle" VENV_PATH_FOR_LDD="$venv" python3 - <<'PY'
import os
import shutil
import subprocess
from pathlib import Path

bundle = Path(os.environ["VENV_BUNDLE"])
venv = Path(os.environ["VENV_PATH_FOR_LDD"])
local_libs = bundle / "local-libs"
local_bin = bundle / "local-bin"
parler_sp = bundle / "parler-venv" / "site-packages"

skip_prefixes = (
    "/lib/ld-linux",
    "/lib64/ld-linux",
)
skip_names = {
    "linux-vdso.so.1",
    "libc.so.6",
    "libdl.so.2",
    "libm.so.6",
    "libpthread.so.0",
    "librt.so.1",
    "libutil.so.1",
    "libresolv.so.2",
    "libselinux.so.1",
    "libpcre2-8.so.0",
    "libacl.so.1",
    "libattr.so.1",
    "libz.so.1",
    "libzstd.so.1",
    "liblzma.so.5",
    "libbz2.so.1.0",
    "libssl.so.3",
    "libcrypto.so.3",
    "libgcc_s.so.1",
    "libstdc++.so.6",
}
skip_starts = (
    "libcuda.so",       # host driver, injected by NVIDIA Container Toolkit
    "libnvidia-",       # host driver stack
)

candidates = []
for root in (venv / "lib", venv / "bin", local_bin, parler_sp):
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if path.is_file() and (os.access(path, os.X_OK) or ".so" in path.name):
            candidates.append(path)

libs = set()
for path in candidates:
    try:
        proc = subprocess.run(["ldd", str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        continue
    if proc.returncode not in (0, 1):
        continue
    for line in proc.stdout.splitlines():
        line = line.strip()
        dep = None
        if "=>" in line:
            rhs = line.split("=>", 1)[1].strip()
            if rhs.startswith("/"):
                dep = rhs.split(" (", 1)[0]
        elif line.startswith("/"):
            dep = line.split(" (", 1)[0]
        if not dep:
            continue
        dep_path = Path(dep)
        name = dep_path.name
        dep_str = str(dep_path)
        if name in skip_names or any(name.startswith(s) for s in skip_starts):
            continue
        if dep_str.startswith(("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")) and "site-packages" not in dep_str:
            continue
        if any(dep_str.startswith(p) for p in skip_prefixes):
            continue
        if dep_path.exists():
            libs.add(dep_path.resolve())

for src in sorted(libs):
    dest = local_libs / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
    # Also create the originally requested soname if it differs from the real path name.
    # ldd usually resolves symlinks; native loaders often ask for the symlink soname.
    for requested in [p for p in libs if p.name != src.name and p.resolve() == src]:
        link = local_libs / requested.name
        if not link.exists():
            try:
                link.symlink_to(src.name)
            except OSError:
                shutil.copy2(src, link)

print(f"Copied {len(libs)} ldd-discovered native librar{'y' if len(libs) == 1 else 'ies'}")
PY
}

if [[ "$BUILD_MODE" == "venv" ]]; then
  if [[ -z "$VENV_PATH" ]]; then
    echo "Error: --from-venv requires an activated virtualenv, or use --venv PATH" >&2
    exit 2
  fi
  VENV_PATH="$(cd "$VENV_PATH" && pwd)"
  if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    echo "Error: '$VENV_PATH' does not look like a Python virtualenv (missing bin/python)" >&2
    exit 2
  fi
  VENV_PYTHON_MINOR="$($VENV_PATH/bin/python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  PBS_PYTHON_MINOR="${PYTHON_VERSION%.*}"
  if [[ "$VENV_PYTHON_MINOR" != "$PBS_PYTHON_MINOR" ]]; then
    cat >&2 <<EOF
Error: venv Python minor ($VENV_PYTHON_MINOR) does not match standalone Python minor ($PBS_PYTHON_MINOR).
Set PYTHON_VERSION in packaging/versions.env to a python-build-standalone release with the same minor,
or use a matching virtualenv.
EOF
    exit 2
  fi
  VENV_CONTEXT="$ROOT_DIR/.packaging-cache/oci-venv-context"
  prepare_venv_bundle "$VENV_PATH" "$VENV_CONTEXT" "$INCLUDE_LOCAL_LIBS"
  write_build_manifest "venv" "$VENV_PATH/bin/python" "$VENV_PATH"
else
  mkdir -p "$ROOT_DIR/.packaging-cache"
  write_build_manifest "from-scratch" "" ""
fi

cat <<EOF
Building CoderAI OCI image locally
  image:        $IMAGE_TAG
  mode:         $BUILD_MODE
  python:       $PYTHON_VERSION
  PBS release:  $PBS_RELEASE
  uv:           $UV_VERSION
  CUDA:         $CUDA_VERSION
  Ubuntu base:  $UBUNTU_VERSION
EOF
if [[ "$BUILD_MODE" == "venv" ]]; then
  cat <<EOF
  venv:         $VENV_PATH
  venv python:  $VENV_PYTHON_MINOR
  local libs:   $([[ "$INCLUDE_LOCAL_LIBS" == "1" ]] && echo copied || echo disabled)
  local bins:   ${#LOCAL_BINARIES[@]}
EOF
fi

if [[ "$BUILD_MODE" == "venv" ]]; then
  "${DOCKER_CMD[@]}" build \
    -f "$ROOT_DIR/packaging/linux/Dockerfile.oci-venv" \
    --build-context "local_bundle=$VENV_CONTEXT" \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    --build-arg PBS_RELEASE="$PBS_RELEASE" \
    --build-arg CUDA_VERSION="$CUDA_VERSION" \
    --build-arg UBUNTU_VERSION="$UBUNTU_VERSION" \
    --build-arg VENV_PYTHON_MINOR="$VENV_PYTHON_MINOR" \
    -t "$IMAGE_TAG" \
    "$ROOT_DIR"
else
  "${DOCKER_CMD[@]}" build \
    -f "$ROOT_DIR/packaging/linux/Dockerfile.oci" \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    --build-arg PBS_RELEASE="$PBS_RELEASE" \
    --build-arg UV_VERSION="$UV_VERSION" \
    --build-arg CUDA_VERSION="$CUDA_VERSION" \
    --build-arg UBUNTU_VERSION="$UBUNTU_VERSION" \
    --build-arg WHISPERCPP_REF="$WHISPERCPP_REF" \
    --build-arg LLAMA_CPP_PYTHON_VERSION="$LLAMA_CPP_PYTHON_VERSION" \
    --build-arg SD_CPP_PYTHON_VERSION="$SD_CPP_PYTHON_VERSION" \
    --build-arg BUILD_SAGEATTENTION="$BUILD_SAGEATTENTION" \
    --build-arg SAGEATTENTION_REF="$SAGEATTENTION_REF" \
    --build-arg SAGEATTENTION_ARCH="$SAGEATTENTION_ARCH" \
    -t "$IMAGE_TAG" \
    "$ROOT_DIR"
fi

cat <<EOF

Built $IMAGE_TAG

Run examples (run as your own UID; create+own the dirs first):
  mkdir -p coderai-config coderai-models coderai-cache
  sudo chown -R "\$(id -u):\$(id -g)" coderai-config coderai-models coderai-cache

  NVIDIA:
    $DOCKER_BIN run --gpus all --ipc=host -p 8776:8776 --user "\$(id -u):\$(id -g)" -v "\$PWD/coderai-config:/config" -v "\$PWD/coderai-models:/models" -v "\$PWD/coderai-cache:/cache" $IMAGE_TAG

  AMD/Intel Vulkan:
    $DOCKER_BIN run --device /dev/dri --ipc=host -p 8776:8776 --user "\$(id -u):\$(id -g)" -v "\$PWD/coderai-config:/config" -v "\$PWD/coderai-models:/models" -v "\$PWD/coderai-cache:/cache" $IMAGE_TAG

  CPU:
    $DOCKER_BIN run --ipc=host -p 8776:8776 --user "\$(id -u):\$(id -g)" -v "\$PWD/coderai-config:/config" -v "\$PWD/coderai-models:/models" -v "\$PWD/coderai-cache:/cache" $IMAGE_TAG

(Drop --user to run as container-root, or use rootless/userns-remap Docker.)

One published port (8776) fronts everything via nginx:
  /  server+API+admin   /editor/  video editor   /videogen/  studio   /township/  fighters

External storage: point /models and /cache at a big disk or NFS volume —
  -v /mnt/bigstorage/coderai/models:/models -v /mnt/bigstorage/coderai/cache:/cache
Non-root: add  --user "\$(id -u):\$(id -g)"  (mounts must be owned by that UID),
  or use rootless/userns-remap Docker with no extra flags.
See packaging/linux/README-RUN.txt (also at /opt/coderai/README-RUN.txt in the image).
EOF

# Export the image + assemble the final distribution bundle (image tarball +
# install.sh + coderai-docker runner) unless disabled with --no-dist.
if [[ "$MAKE_DIST" == "1" ]]; then
  echo ""
  echo "=== Building distribution bundle (use --no-dist to skip) ==="
  DOCKER="$DOCKER_BIN" IMAGE="$IMAGE_TAG" REFRESH=1 \
    "$ROOT_DIR/packaging/linux/make_dist_bundle.sh"
fi

# Clean up the dangling/intermediate images this build produced (untagged <none>
# layers from the multi-stage build). `image prune -f` only removes dangling
# images — the final image ($IMAGE_TAG), the CUDA base, and any other tagged
# images are kept. Disable with --no-prune.
if [[ "$PRUNE_BUILD_IMAGES" == "1" ]]; then
  echo ""
  echo "=== Pruning dangling build images (use --no-prune to skip) ==="
  "${DOCKER_CMD[@]}" image prune -f 2>/dev/null || true
fi
