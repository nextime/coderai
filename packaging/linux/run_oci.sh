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
# Selected GPU backends. ADDITIVE: --nvidia --vulkan enables BOTH, so the
# container gets the NVIDIA driver libs (libcuda.so.1 — needed even by a
# CUDA-built llama-cpp running under Vulkan) AND /dev/dri. CPU always works.
declare -A MODES=()
# Bind-mount the host's libcuda.so.1 into the container (for Vulkan/CPU runs of a
# CUDA-built llama-cpp on a host that has the driver but where you don't want the
# full --gpus all). "auto" = detect via ldconfig; or an explicit path.
WITH_LIBCUDA=""
PORT="${CODERAI_PORT:-8776}"
# Host interface the published port binds to. Empty = Docker's default (all
# interfaces, 0.0.0.0). Set e.g. 127.0.0.1 to expose only on localhost.
HOST_BIND="${CODERAI_HOST_BIND:-}"
# Extra CLI flags passed straight through to the coderai server inside the
# container (via CODERAI_EXTRA_ARGS, appended by the in-image coderai launcher).
# Built from --coderai-arg (repeatable, one token each) and --coderai-args "...".
CODERAI_EXTRA_ARGS="${CODERAI_EXTRA_ARGS:-}"
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
# Demo tool web UIs (video editor, videogen, township, parler). Empty = image
# default (the three UIs on, parler off). Keyed by CODERAI_TOOL_* env var.
declare -A TOOL_STATE=()
DISABLE_ALL_TOOLS=0

# Map a friendly tool name to its CODERAI_TOOL_* env var (or fail).
tool_env_var() {
  case "$1" in
    video-editor|video_editor|editor) echo CODERAI_TOOL_VIDEO_EDITOR ;;
    videogen|video-gen)               echo CODERAI_TOOL_VIDEOGEN ;;
    township|fighters)                echo CODERAI_TOOL_TOWNSHIP ;;
    parler|tts)                       echo CODERAI_TOOL_PARLER ;;
    *) return 1 ;;
  esac
}

usage() {
  cat <<'EOF'
Usage:
  packaging/linux/run_oci.sh [--cpu|--nvidia|--vulkan] [IMAGE_TAG]

Options:
  --docker            Use docker (default).
  --podman            Use podman.
  --cpu               Enable the CPU backend (always available; default if none).
  --nvidia            Enable NVIDIA CUDA; adds --gpus all for Docker (maps the
                      driver incl. libcuda.so.1).
  --vulkan            Enable Vulkan; adds --device /dev/dri and auto bind-mounts
                      the host's libcuda.so.1 (the bundled llama-cpp is a CUDA
                      build). --nvidia and --vulkan are ADDITIVE — pass both to
                      enable both backends in one container.
  --all               Enable all GPU backends (nvidia + vulkan).
  --with-libcuda[=P]  Bind-mount libcuda.so.1 into the container so a CUDA-built
                      llama-cpp loads under --vulkan/--cpu on a driver-equipped
                      host. P is an explicit path; default auto-detects via
                      ldconfig. (Implied automatically when --nvidia is set.)
  -p, --port PORT     Host port to expose (default: 8776).
  --host ADDR         Host interface to bind the published port to (e.g.
                      127.0.0.1 for localhost-only, 0.0.0.0 for all interfaces).
                      Default: Docker's default (all interfaces).
  --coderai-arg ARG   Pass one extra flag straight through to the coderai server
                      (e.g. --coderai-arg --some-flag). Repeatable; each ARG is a
                      single token (no embedded spaces).
  --coderai-args STR  Pass a raw string of extra coderai flags (space-separated),
                      e.g. --coderai-args "--foo bar --baz". Appended after any
                      --coderai-arg values.
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
  --no-tools          Disable ALL bundled demo tool web UIs (video editor,
                      videogen, township). They're on by default.
  --enable-tool NAME  Force-enable a demo tool. Repeatable. NAME is one of:
                        video-editor | videogen | township | parler
                      (parler TTS is off by default; this turns it on.)
  --disable-tool NAME Disable a single demo tool. Repeatable. Same NAMEs as above.
                      Explicit --enable/--disable-tool overrides --no-tools.
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
    --cpu) MODES[cpu]=1; shift ;;
    --nvidia|--cuda) MODES[nvidia]=1; shift ;;
    --vulkan) MODES[vulkan]=1; shift ;;
    --all) MODES[nvidia]=1; MODES[vulkan]=1; shift ;;
    --with-libcuda) WITH_LIBCUDA="auto"; shift ;;
    --with-libcuda=*) WITH_LIBCUDA="${1#*=}"; shift ;;
    -p|--port)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a port" >&2; exit 2; }
      PORT="$2"; shift 2 ;;
    --host)
      [[ $# -ge 2 ]] || { echo "Error: --host requires an address" >&2; exit 2; }
      HOST_BIND="$2"; shift 2 ;;
    --coderai-arg)
      [[ $# -ge 2 ]] || { echo "Error: --coderai-arg requires a value" >&2; exit 2; }
      CODERAI_EXTRA_ARGS="${CODERAI_EXTRA_ARGS:+$CODERAI_EXTRA_ARGS }$2"; shift 2 ;;
    --coderai-args)
      [[ $# -ge 2 ]] || { echo "Error: --coderai-args requires a string" >&2; exit 2; }
      CODERAI_EXTRA_ARGS="${CODERAI_EXTRA_ARGS:+$CODERAI_EXTRA_ARGS }$2"; shift 2 ;;
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
    --no-tools) DISABLE_ALL_TOOLS=1; shift ;;
    --enable-tool)
      [[ $# -ge 2 ]] || { echo "Error: --enable-tool requires a tool name" >&2; exit 2; }
      _v="$(tool_env_var "$2")" || { echo "Error: unknown tool '$2' (video-editor|videogen|township|parler)" >&2; exit 2; }
      TOOL_STATE["$_v"]=true; shift 2 ;;
    --disable-tool)
      [[ $# -ge 2 ]] || { echo "Error: --disable-tool requires a tool name" >&2; exit 2; }
      _v="$(tool_env_var "$2")" || { echo "Error: unknown tool '$2' (video-editor|videogen|township|parler)" >&2; exit 2; }
      TOOL_STATE["$_v"]=false; shift 2 ;;
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

# Publish spec: bind to a specific host interface when --host is given, else let
# Docker use its default (all interfaces). CODERAI_HOST stays 0.0.0.0 so the
# server listens on all interfaces *inside* the container.
if [[ -n "$HOST_BIND" ]]; then
  PUBLISH="$HOST_BIND:$PORT:8776"
else
  PUBLISH="$PORT:8776"
fi
args=(run --rm --name "$NAME" --ipc=host -p "$PUBLISH" -e CODERAI_HOST=0.0.0.0 -e CODERAI_PORT=8776)
# Pass-through coderai server flags (appended by the in-image launcher's argv).
if [[ -n "$CODERAI_EXTRA_ARGS" ]]; then
  args+=(-e "CODERAI_EXTRA_ARGS=$CODERAI_EXTRA_ARGS")
fi
if [[ "$DETACH" == "1" ]]; then
  args+=(-d)
fi

# Default to CPU-only when no GPU backend was requested.
if [[ "${#MODES[@]}" -eq 0 ]]; then
  MODES[cpu]=1
fi

if [[ -n "${MODES[nvidia]:-}" ]]; then
  if [[ "$ENGINE" == "docker" ]]; then
    args+=(--gpus all)
  else
    args+=(--hooks-dir=/usr/share/containers/oci/hooks.d)
  fi
fi
if [[ -n "${MODES[vulkan]:-}" ]]; then
  args+=(--device /dev/dri)
  # The bundled llama-cpp is a CUDA build, so Vulkan GGUF still needs libcuda.so.1.
  # Auto-map it from the host (unless --nvidia already maps the whole driver, or
  # the user gave an explicit --with-libcuda path).
  [[ -z "$WITH_LIBCUDA" ]] && WITH_LIBCUDA="auto"
fi

# libcuda.so.1: the bundled llama-cpp-python is a CUDA build, so it needs the
# NVIDIA userspace driver lib even for Vulkan/CPU GGUF. --nvidia maps the whole
# driver via --gpus all already; --vulkan auto-enables a libcuda bind-mount (set
# just above); otherwise bind-mount just libcuda when asked via --with-libcuda,
# so a CUDA llama-cpp at least loads. Misses degrade gracefully now: the server
# starts and the Vulkan/GGUF backend is simply reported unavailable.
LIBCUDA_NOTE="none"
if [[ -n "${MODES[nvidia]:-}" ]]; then
  LIBCUDA_NOTE="via --gpus all (driver mapped)"
elif [[ -n "$WITH_LIBCUDA" ]]; then
  libcuda_path=""
  if [[ "$WITH_LIBCUDA" == "auto" ]]; then
    libcuda_path="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/ {print $NF; exit}')"
    [[ -n "$libcuda_path" ]] || for c in /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/libcuda.so.1 /usr/lib64/libcuda.so.1; do
      [[ -e "$c" ]] && { libcuda_path="$c"; break; }
    done
  else
    libcuda_path="$WITH_LIBCUDA"
  fi
  if [[ -n "$libcuda_path" && -e "$libcuda_path" ]]; then
    args+=(-v "$libcuda_path:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro")
    LIBCUDA_NOTE="$libcuda_path → /usr/lib/x86_64-linux-gnu/libcuda.so.1"
  else
    echo "Warning: --with-libcuda requested but libcuda.so.1 not found${WITH_LIBCUDA:+ ($WITH_LIBCUDA)}; skipping" >&2
    LIBCUDA_NOTE="requested but not found"
  fi
fi

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

# Demo tool toggles → CODERAI_TOOL_* env. --no-tools turns the three UIs off
# (unless a specific --enable-tool re-enabled one); explicit toggles always win.
if [[ "$DISABLE_ALL_TOOLS" == "1" ]]; then
  for _v in CODERAI_TOOL_VIDEO_EDITOR CODERAI_TOOL_VIDEOGEN CODERAI_TOOL_TOWNSHIP; do
    [[ -n "${TOOL_STATE[$_v]:-}" ]] || TOOL_STATE["$_v"]=false
  done
fi
TOOLS_NOTE="image defaults (video-editor, videogen, township on; parler off)"
if [[ "${#TOOL_STATE[@]}" -gt 0 ]]; then
  TOOLS_NOTE=""
  for _v in "${!TOOL_STATE[@]}"; do
    args+=(-e "$_v=${TOOL_STATE[$_v]}")
    _label="${_v#CODERAI_TOOL_}"
    TOOLS_NOTE+="$(echo "$_label" | tr 'A-Z_' 'a-z-')=${TOOL_STATE[$_v]} "
  done
  TOOLS_NOTE="${TOOLS_NOTE% }"
fi

args+=("${EXTRA_ARGS[@]}" "$IMAGE_TAG")

cat <<EOF
Starting CoderAI OCI container
  engine:  $ENGINE
  image:   $IMAGE_TAG
  mode:    $(echo "${!MODES[@]}" | tr ' ' '+' | tr 'A-Z' 'a-z')
  libcuda: $LIBCUDA_NOTE
  url:     http://${HOST_BIND:-127.0.0.1}:$PORT/admin
  data:    $DATA_ROOT
  config:  $CONFIG_NOTE
  debug:   ${DEBUG_SPEC:-off}
  log:     $LOG_HOST_NOTE
  tools:   $TOOLS_NOTE
  cdr-args:${CODERAI_EXTRA_ARGS:+ $CODERAI_EXTRA_ARGS}
EOF

if [[ "$LOG_HOST_NOTE" != "(none)" ]]; then
  echo "  tail it:  tail -F '$LOG_HOST_NOTE'"
fi

exec "$ENGINE" "${args[@]}"
