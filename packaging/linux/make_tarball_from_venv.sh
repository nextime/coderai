#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSIONS_FILE="$ROOT_DIR/packaging/versions.env"
if [[ -f "$VERSIONS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$VERSIONS_FILE"
fi

VENV_PATH="${VIRTUAL_ENV:-}"
OUT_DIR="$ROOT_DIR/dist"
OUT_NAME=""
INCLUDE_LOCAL_LIBS=1
AUTO_LOCAL_BINS=1
LOCAL_BINARIES=()
LOCAL_BINARY_DIRS=()
PYTHON_VERSION="${PYTHON_VERSION:-3.13.5}"
PBS_RELEASE="${PBS_RELEASE:-20250612}"
UV_VERSION="${UV_VERSION:-0.7.13}"
CUDA_VERSION="${CUDA_VERSION:-12.4.1}"
UBUNTU_VERSION="${UBUNTU_VERSION:-22.04}"

usage() {
  cat <<'EOF'
Usage:
  packaging/linux/make_tarball_from_venv.sh --venv PATH
  source venv_all/bin/activate && packaging/linux/make_tarball_from_venv.sh

Options:
  --venv PATH             Source virtualenv to package. Defaults to activated $VIRTUAL_ENV.
  -o, --output PATH       Output .tar.zst path. Defaults to dist/coderai-linux-x64-venv.tar.zst.
  --no-local-libs         Do not copy ldd-discovered native libraries from the venv.
  --no-auto-local-bins    Do not auto-include known locally compiled helper binaries.
  --include-local-bin PATH
                          Copy an extra tested local binary into bin/, including its ldd libs.
  --include-local-dir PATH
                          Copy executable files from a local build directory into bin/.
  -h, --help              Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      [[ $# -ge 2 ]] || { echo "Error: --venv requires a path" >&2; exit 2; }
      VENV_PATH="$2"; shift 2 ;;
    -o|--output)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a path" >&2; exit 2; }
      OUT_NAME="$2"; shift 2 ;;
    --no-local-libs) INCLUDE_LOCAL_LIBS=0; shift ;;
    --no-auto-local-bins) AUTO_LOCAL_BINS=0; shift ;;
    --include-local-bin)
      [[ $# -ge 2 ]] || { echo "Error: --include-local-bin requires a path" >&2; exit 2; }
      LOCAL_BINARIES+=("$2"); shift 2 ;;
    --include-local-dir)
      [[ $# -ge 2 ]] || { echo "Error: --include-local-dir requires a path" >&2; exit 2; }
      LOCAL_BINARY_DIRS+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "Error: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$VENV_PATH" ]]; then
  echo "Error: pass --venv PATH or activate a virtualenv first" >&2
  exit 2
fi
VENV_PATH="$(cd "$VENV_PATH" && pwd)"
[[ -x "$VENV_PATH/bin/python" ]] || { echo "Error: missing venv python: $VENV_PATH/bin/python" >&2; exit 2; }
VENV_PYTHON_MINOR="$($VENV_PATH/bin/python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
PBS_PYTHON_MINOR="${PYTHON_VERSION%.*}"
if [[ "$VENV_PYTHON_MINOR" != "$PBS_PYTHON_MINOR" ]]; then
  echo "Error: venv Python minor ($VENV_PYTHON_MINOR) does not match standalone Python minor ($PBS_PYTHON_MINOR)" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$ROOT_DIR/.packaging-cache"
[[ -n "$OUT_NAME" ]] || OUT_NAME="$OUT_DIR/coderai-linux-x64-venv.tar.zst"
STAGE="$ROOT_DIR/.packaging-cache/tarball/coderai"
rm -rf "$ROOT_DIR/.packaging-cache/tarball"
mkdir -p "$STAGE/python" "$STAGE/app" "$STAGE/bin" "$STAGE/local-libs" "$STAGE/config" "$STAGE/models" "$STAGE/cache"

curl -fsSL -o "$ROOT_DIR/.packaging-cache/python.tar.gz" \
  "https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/cpython-${PYTHON_VERSION}+${PBS_RELEASE}-x86_64-unknown-linux-gnu-install_only.tar.gz"
tar -xzf "$ROOT_DIR/.packaging-cache/python.tar.gz" -C "$STAGE"

source_sp="$VENV_PATH/lib/python${VENV_PYTHON_MINOR}/site-packages"
target_sp="$STAGE/python/lib/python${VENV_PYTHON_MINOR}/site-packages"
[[ -d "$source_sp" ]] || { echo "Error: venv site-packages not found: $source_sp" >&2; exit 2; }
rsync -a --delete "$source_sp/" "$target_sp/"
if [[ -d "$VENV_PATH/bin" ]]; then
  find "$VENV_PATH/bin" -maxdepth 1 -type f ! -name 'python*' ! -name 'activate*' -exec cp -a '{}' "$STAGE/python/bin/" \;
fi

add_local_binary() {
  local path="$1"
  local abs_path
  [[ -x "$path" && -f "$path" ]] || return 0
  abs_path="$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
  for existing in "${LOCAL_BINARIES[@]}"; do
    [[ "$existing" == "$abs_path" ]] && return 0
  done
  LOCAL_BINARIES+=("$abs_path")
}

if [[ "$AUTO_LOCAL_BINS" == "1" ]]; then
  for p in "/usr/local/bin/whisper-server" "/usr/local/bin/whisper-cli" "$HOME/whisper.cpp/build/bin/whisper-server" "$HOME/whisper.cpp/build/bin/whisper-cli" "$HOME/whisper.cpp/build/bin/main" "$HOME/whisper.cpp/build/bin/server"; do
    add_local_binary "$p"
  done
fi
for d in "${LOCAL_BINARY_DIRS[@]}"; do
  [[ -d "$d" ]] || { echo "Error: local binary directory does not exist: $d" >&2; exit 2; }
  while IFS= read -r -d '' found_bin; do add_local_binary "$found_bin"; done < <(find "$d" -maxdepth 2 -type f -perm -111 -print0)
done
for b in "${LOCAL_BINARIES[@]}"; do
  [[ -x "$b" ]] || { echo "Error: local binary is not executable: $b" >&2; exit 2; }
  cp -a "$b" "$STAGE/bin/"
done

if [[ "$INCLUDE_LOCAL_LIBS" == "1" ]]; then
  VENV_PATH_FOR_LDD="$VENV_PATH" LOCAL_BIN_DIR="$STAGE/bin" LOCAL_LIB_DIR="$STAGE/local-libs" python3 - <<'PY'
import os, shutil, subprocess
from pathlib import Path
venv=Path(os.environ['VENV_PATH_FOR_LDD'])
local_bin=Path(os.environ['LOCAL_BIN_DIR'])
local_lib=Path(os.environ['LOCAL_LIB_DIR'])
skip_names={'linux-vdso.so.1','libc.so.6','libdl.so.2','libm.so.6','libpthread.so.0','librt.so.1','libutil.so.1','libresolv.so.2','libselinux.so.1','libpcre2-8.so.0','libacl.so.1','libattr.so.1','libz.so.1','libzstd.so.1','liblzma.so.5','libbz2.so.1.0','libssl.so.3','libcrypto.so.3','libgcc_s.so.1','libstdc++.so.6'}
skip_starts=('libcuda.so','libnvidia-')
candidates=[]
for root in (venv/'lib', venv/'bin', local_bin):
    if root.exists():
        for p in root.rglob('*'):
            if p.is_file() and (os.access(p, os.X_OK) or '.so' in p.name):
                candidates.append(p)
libs=set()
for p in candidates:
    proc=subprocess.run(['ldd', str(p)], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    for line in proc.stdout.splitlines():
        line=line.strip(); dep=None
        if '=>' in line:
            rhs=line.split('=>',1)[1].strip()
            if rhs.startswith('/'): dep=rhs.split(' (',1)[0]
        elif line.startswith('/'):
            dep=line.split(' (',1)[0]
        if not dep: continue
        dp=Path(dep); name=dp.name
        if name in skip_names or any(name.startswith(s) for s in skip_starts): continue
        if str(dp).startswith(('/lib/','/lib64/','/usr/lib/','/usr/lib64/')) and 'site-packages' not in str(dp): continue
        if str(dp).startswith(('/lib/ld-linux','/lib64/ld-linux')): continue
        if dp.exists(): libs.add(dp.resolve())
for src in sorted(libs):
    dest=local_lib/src.name
    if not dest.exists(): shutil.copy2(src,dest)
print(f"Copied {len(libs)} ldd-discovered native libraries")
PY
fi

rsync -a --delete \
  --exclude '.git' --exclude 'venv*' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'models' --exclude 'offload' --exclude 'township_output' --exclude 'dist' --exclude '.packaging-cache' \
  "$ROOT_DIR/" "$STAGE/app/"
cp "$ROOT_DIR/packaging/linux/launcher/coderai-tarball" "$STAGE/bin/coderai"
cp "$ROOT_DIR/packaging/linux/README-RUN.txt" "$STAGE/README-RUN.txt"
chmod +x "$STAGE/bin/coderai"

MANIFEST_OUT="$STAGE/BUILD-MANIFEST.json" PROJECT_ROOT="$ROOT_DIR" MANIFEST_ARTIFACT="linux-tarball" MANIFEST_BUILD_MODE="venv" MANIFEST_PYTHON="$VENV_PATH/bin/python" MANIFEST_VENV="$VENV_PATH" MANIFEST_LOCAL_BINS="$(IFS=:; echo "${LOCAL_BINARIES[*]:-}")" PYTHON_VERSION="$PYTHON_VERSION" PBS_RELEASE="$PBS_RELEASE" UV_VERSION="$UV_VERSION" CUDA_VERSION="$CUDA_VERSION" UBUNTU_VERSION="$UBUNTU_VERSION" python3 "$ROOT_DIR/packaging/common/write_manifest.py"

find "$STAGE" -type d -name __pycache__ -prune -exec rm -rf '{}' +
tar --zstd -cf "$OUT_NAME" -C "$ROOT_DIR/.packaging-cache/tarball" coderai
sha256sum "$OUT_NAME" > "$OUT_NAME.sha256"

cat <<EOF
Created Linux tarball:
  archive:  $OUT_NAME
  checksum: $OUT_NAME.sha256

Extract and run:
  tar --zstd -xf "$OUT_NAME"
  ./coderai/bin/coderai
EOF
