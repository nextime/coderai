#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSIONS_FILE="$ROOT_DIR/packaging/versions.env"
if [[ -f "$VERSIONS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$VERSIONS_FILE"
fi

ENGINE="${CONTAINER_ENGINE:-docker}"
IMAGE_TAG="${OCI_IMAGE:-coderai:dist}"
MODE="cpu"
PORT="${CODERAI_PORT:-8776}"
DATA_ROOT="$PWD/coderai-runtime"
DETACH=0
NAME="coderai"
EXTRA_ARGS=()
# Optional: map an EXISTING local config dir + real data dirs so the image runs
# against your live config/models without a rebuild (an image is immutable; this
# is purely run-time bind-mounts). See --config-dir / --local / --map below.
CONFIG_DIR_SRC=""
INPLACE_CONFIG=0
MAPS=()
# Optional debug logging: CODERAI_DEBUG selects coderai's --debug* flags inside
# the container; LOG_FILE_CONT is the in-container log path (under a mounted
# volume so it's tailable on the host).
DEBUG_SPEC=""
LOG_FILE_CONT=""

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
  --config-dir PATH   Use an EXISTING config dir (with config.json/models.json),
                      mounted at /config/coderai. Copied to a temp dir by default
                      so the image's host/port rewrite leaves your dir untouched.
  --local             Shortcut for --config-dir ~/.coderai.
  --inplace-config    Mount --config-dir in place (the image WILL edit host/port).
  --map HOST[:CONT]   Bind-mount a host dir at the SAME path (or HOST:CONT) inside
                      the container, so absolute paths in models.json resolve
                      (e.g. --map /AI/guffcache). Repeatable.
  --debug[=SPEC]      Run coderai with debug flags. SPEC (default 'all'):
                        all | engine,requests,ws,web,thermal,lora,engine-web
                      Also writes a host-tailable file log (see --log-file).
  --log-file PATH     In-container log path (default /cache/logs/coderai.log,
                      visible on the host under the cache mount). Implies a file
                      log even without --debug. tee'd, so `docker logs` still works.
  -- ARGS             Extra args passed to the container engine before the image name.
  -h, --help          Show this help.

Test against your live config + data (no rebuild):
  packaging/linux/run_oci.sh --nvidia --local \
    --map /AI/guffcache --map /AI/huggingface --map /AI/offloads
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
    --config-dir)
      [[ $# -ge 2 ]] || { echo "Error: --config-dir requires a path" >&2; exit 2; }
      CONFIG_DIR_SRC="$2"; shift 2 ;;
    --local) CONFIG_DIR_SRC="$HOME/.coderai"; shift ;;
    --inplace-config) INPLACE_CONFIG=1; shift ;;
    --map)
      [[ $# -ge 2 ]] || { echo "Error: --map requires HOST[:CONT]" >&2; exit 2; }
      MAPS+=("$2"); shift 2 ;;
    --debug) DEBUG_SPEC="all"; shift ;;
    --debug=*) DEBUG_SPEC="${1#*=}"; shift ;;
    --log-file)
      [[ $# -ge 2 ]] || { echo "Error: --log-file requires a path" >&2; exit 2; }
      LOG_FILE_CONT="$2"; shift 2 ;;
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

# Config mount: either the fresh scratch dir, or an EXISTING local config dir
# mounted at /config/coderai (where the image launcher reads config.json).
CONFIG_NOTE="$DATA_ROOT/config (fresh)"
if [[ -n "$CONFIG_DIR_SRC" ]]; then
  [[ -d "$CONFIG_DIR_SRC" ]] || { echo "Error: --config-dir '$CONFIG_DIR_SRC' not found" >&2; exit 2; }
  CONFIG_DIR_SRC="$(cd "$CONFIG_DIR_SRC" && pwd)"
  if [[ "$INPLACE_CONFIG" == "1" ]]; then
    CFG_MOUNT="$CONFIG_DIR_SRC"
    CONFIG_NOTE="$CONFIG_DIR_SRC (in place — image rewrites host/port!)"
  else
    # Copy ONLY the json config files to a throwaway dir so the image's host/port
    # rewrite never touches your real config, and we don't copy big subdirs
    # (e.g. ~/.coderai/ds4 weights).
    CFG_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/coderai-cfg.XXXXXX")"
    CFG_MOUNT="$CFG_PARENT/coderai"
    mkdir -p "$CFG_MOUNT"
    cp -a "$CONFIG_DIR_SRC"/*.json "$CFG_MOUNT/" 2>/dev/null || true
    [[ -f "$CFG_MOUNT/config.json" ]] || { echo "Error: no config.json in '$CONFIG_DIR_SRC'" >&2; exit 2; }
    CONFIG_NOTE="$CONFIG_DIR_SRC → $CFG_MOUNT (copy; original untouched)"
  fi
  args+=(-v "$CFG_MOUNT:/config/coderai$volume_suffix" \
         -v "$DATA_ROOT/models:/models$volume_suffix" -v "$DATA_ROOT/cache:/cache$volume_suffix")
else
  args+=(-v "$DATA_ROOT/config:/config$volume_suffix" -v "$DATA_ROOT/models:/models$volume_suffix" -v "$DATA_ROOT/cache:/cache$volume_suffix")
fi

# 1:1 (or HOST:CONT) data mounts so absolute paths in models.json resolve.
for m in "${MAPS[@]:-}"; do
  [[ -n "$m" ]] || continue
  host="${m%%:*}"; cont="${m#*:}"; [[ "$m" == *:* ]] || cont="$host"
  if [[ -d "$host" ]]; then
    args+=(-v "$host:$cont$volume_suffix")
  else
    echo "Warning: --map source '$host' not found; skipping" >&2
  fi
done

# Debug flags + host-tailable file log. A file log is enabled by --debug or
# --log-file; default path lives under /cache so it lands on the host mount.
LOG_HOST_NOTE="(none)"
if [[ -n "$DEBUG_SPEC" || -n "$LOG_FILE_CONT" ]]; then
  : "${LOG_FILE_CONT:=/cache/logs/coderai.log}"
  [[ -n "$DEBUG_SPEC" ]] && args+=(-e "CODERAI_DEBUG=$DEBUG_SPEC")
  args+=(-e "CODERAI_LOG_FILE=$LOG_FILE_CONT")
  # Translate the in-container path to the host path for the banner, for the
  # standard /config|/models|/cache mounts.
  case "$LOG_FILE_CONT" in
    /cache/*)  LOG_HOST_NOTE="$DATA_ROOT/cache/${LOG_FILE_CONT#/cache/}" ;;
    /models/*) LOG_HOST_NOTE="$DATA_ROOT/models/${LOG_FILE_CONT#/models/}" ;;
    /config/*) LOG_HOST_NOTE="$DATA_ROOT/config/${LOG_FILE_CONT#/config/}" ;;
    *)         LOG_HOST_NOTE="$LOG_FILE_CONT (in-container; mount it to see it on the host)" ;;
  esac
fi

args+=("${EXTRA_ARGS[@]}" "$IMAGE_TAG")

cat <<EOF
Starting CoderAI OCI container
  engine:  $ENGINE
  image:   $IMAGE_TAG
  mode:    $MODE
  url:     http://127.0.0.1:$PORT/admin
  data:    $DATA_ROOT
  config:  $CONFIG_NOTE
  debug:   ${DEBUG_SPEC:-off}
  log:     $LOG_HOST_NOTE
EOF

if [[ "$LOG_HOST_NOTE" != "(none)" ]]; then
  echo "  tail it:  tail -F '$LOG_HOST_NOTE'"
fi

exec "$ENGINE" "${args[@]}"
