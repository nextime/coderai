#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSIONS_FILE="$ROOT_DIR/packaging/versions.env"
if [[ -f "$VERSIONS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$VERSIONS_FILE"
fi

ENGINE="${CONTAINER_ENGINE:-docker}"
IMAGE_TAG="${OCI_IMAGE:-coderai:local}"
MODE="cpu"
PORT="${CODERAI_PORT:-8776}"
DATA_ROOT="$PWD/coderai-runtime"
DETACH=0
NAME="coderai"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  packaging/linux/run_oci.sh [--cpu|--nvidia|--vulkan] [IMAGE_TAG]

Options:
  --docker            Use docker (default).
  --podman            Use podman.
  --cpu               CPU-only run mode (default).
  --nvidia            NVIDIA CUDA mode; adds --gpus all for Docker.
  --vulkan            Vulkan mode; adds --device /dev/dri.
  -p, --port PORT     Host port to expose (default: 8776).
  --data-dir PATH     Directory for config/models/cache (default: ./coderai-runtime).
  --name NAME         Container name (default: coderai).
  -d, --detach        Run in background.
  -- ARGS             Extra args passed to the container engine before the image name.
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker) ENGINE=docker; shift ;;
    --podman) ENGINE=podman; shift ;;
    --cpu) MODE=cpu; shift ;;
    --nvidia|--cuda) MODE=nvidia; shift ;;
    --vulkan) MODE=vulkan; shift ;;
    -p|--port)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a port" >&2; exit 2; }
      PORT="$2"; shift 2 ;;
    --data-dir)
      [[ $# -ge 2 ]] || { echo "Error: --data-dir requires a path" >&2; exit 2; }
      DATA_ROOT="$2"; shift 2 ;;
    --name)
      [[ $# -ge 2 ]] || { echo "Error: --name requires a value" >&2; exit 2; }
      NAME="$2"; shift 2 ;;
    -d|--detach) DETACH=1; shift ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) IMAGE_TAG="$1"; shift ;;
  esac
done

mkdir -p "$DATA_ROOT/config" "$DATA_ROOT/models" "$DATA_ROOT/cache"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"

args=(run --rm --name "$NAME" --ipc=host -p "$PORT:8776" -e CODERAI_HOST=0.0.0.0 -e CODERAI_PORT=8776)
if [[ "$DETACH" == "1" ]]; then
  args+=(-d)
fi

case "$MODE" in
  nvidia)
    if [[ "$ENGINE" == "docker" ]]; then
      args+=(--gpus all)
    else
      args+=(--hooks-dir=/usr/share/containers/oci/hooks.d)
    fi
    ;;
  vulkan)
    args+=(--device /dev/dri)
    ;;
  cpu) ;;
esac

volume_suffix=""
if [[ "$ENGINE" == "podman" ]]; then
  volume_suffix=":Z"
fi
args+=(-v "$DATA_ROOT/config:/config$volume_suffix" -v "$DATA_ROOT/models:/models$volume_suffix" -v "$DATA_ROOT/cache:/cache$volume_suffix")
args+=("${EXTRA_ARGS[@]}" "$IMAGE_TAG")

cat <<EOF
Starting CoderAI OCI container
  engine:  $ENGINE
  image:   $IMAGE_TAG
  mode:    $MODE
  url:     http://127.0.0.1:$PORT/admin
  data:    $DATA_ROOT
EOF

exec "$ENGINE" "${args[@]}"
