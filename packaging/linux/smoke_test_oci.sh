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
MODE="cpu"
PORT="${CODERAI_PORT:-18776}"
KEEP=0
TIMEOUT=45
CONTAINER_NAME="coderai-smoke-$$"

usage() {
  cat <<'EOF'
Usage:
  packaging/linux/smoke_test_oci.sh [IMAGE_TAG]

Options:
  -t, --tag TAG       Image tag to test (default: coderai:local or OCI_IMAGE).
  --mode MODE         cpu, nvidia, or vulkan (default: cpu).
  --port PORT         Host port for boot test (default: 18776).
  --timeout SECONDS   Server boot timeout (default: 45).
  --keep              Keep the container after failure for inspection.
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--tag)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a tag" >&2; exit 2; }
      IMAGE_TAG="$2"
      shift 2
      ;;
    --mode)
      [[ $# -ge 2 ]] || { echo "Error: --mode requires cpu, nvidia, or vulkan" >&2; exit 2; }
      MODE="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "Error: --port requires a value" >&2; exit 2; }
      PORT="$2"
      shift 2
      ;;
    --timeout)
      [[ $# -ge 2 ]] || { echo "Error: --timeout requires seconds" >&2; exit 2; }
      TIMEOUT="$2"
      shift 2
      ;;
    --keep)
      KEEP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
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

case "$MODE" in
  cpu|nvidia|vulkan) ;;
  *) echo "Error: --mode must be cpu, nvidia, or vulkan" >&2; exit 2 ;;
esac

cleanup() {
  if [[ "$KEEP" != "1" ]]; then
    "${DOCKER_CMD[@]}" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if ! "${DOCKER_CMD[@]}" image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "Error: image not found: $IMAGE_TAG" >&2
  exit 1
fi

IMPORT_CHECK='import importlib.util, json
mods=["fastapi","uvicorn","torch","transformers","diffusers","accelerate","llama_cpp","PIL"]
optional=["stable_diffusion_cpp","whispercpp","bitsandbytes","onnxruntime"]
out={"required":{},"optional":{}}
missing=[]
for m in mods:
    ok=importlib.util.find_spec(m) is not None
    out["required"][m]=ok
    if not ok: missing.append(m)
for m in optional:
    out["optional"][m]=importlib.util.find_spec(m) is not None
try:
    import torch
    out["torch_cuda_available"]=bool(torch.cuda.is_available())
    out["torch_cuda_device_count"]=int(torch.cuda.device_count())
except Exception as e:
    out["torch_cuda_error"]=str(e)
print(json.dumps(out, sort_keys=True))
if missing:
    raise SystemExit("missing required imports: "+", ".join(missing))'

run_args=(--rm)
case "$MODE" in
  nvidia) run_args+=(--gpus all) ;;
  vulkan) run_args+=(--device /dev/dri) ;;
esac

echo "Checking imports in $IMAGE_TAG..."
"${DOCKER_CMD[@]}" run "${run_args[@]}" --entrypoint /opt/coderai/python/bin/python3 "$IMAGE_TAG" -c "$IMPORT_CHECK"

tmp_dir="$ROOT_DIR/.packaging-cache/smoke-$MODE-$$"
rm -rf "$tmp_dir" 2>/dev/null || true
mkdir -p "$tmp_dir/config" "$tmp_dir/models" "$tmp_dir/cache"

container_args=(-d --name "$CONTAINER_NAME" -p "$PORT:8776" -e CODERAI_HOST=0.0.0.0 -e CODERAI_PORT=8776 -v "$tmp_dir/config:/config" -v "$tmp_dir/models:/models" -v "$tmp_dir/cache:/cache")
case "$MODE" in
  nvidia) container_args+=(--gpus all --ipc=host) ;;
  vulkan) container_args+=(--device /dev/dri --ipc=host) ;;
  cpu) container_args+=(--ipc=host) ;;
esac

echo "Starting boot test container on http://127.0.0.1:$PORT ..."
"${DOCKER_CMD[@]}" run "${container_args[@]}" "$IMAGE_TAG" >/dev/null

start=$SECONDS
until status=$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/admin" 2>/dev/null) && [[ "$status" =~ ^(200|301|302|401|403)$ ]]; do
  if (( SECONDS - start > TIMEOUT )); then
    echo "Server did not respond within ${TIMEOUT}s" >&2
    "${DOCKER_CMD[@]}" logs "$CONTAINER_NAME" >&2 || true
    exit 1
  fi
  sleep 1
done

models_status=$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || true)
if [[ ! "$models_status" =~ ^(200|401|403)$ ]]; then
  echo "Unexpected /v1/models status: $models_status" >&2
  "$DOCKER_BIN" logs "$CONTAINER_NAME" >&2 || true
  exit 1
fi

echo "Smoke test passed for $IMAGE_TAG ($MODE)."
