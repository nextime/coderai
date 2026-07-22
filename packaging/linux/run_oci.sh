#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSIONS_FILE="$ROOT_DIR/packaging/versions.env"
if [[ -f "$VERSIONS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$VERSIONS_FILE"
fi

ENGINE="${CONTAINER_ENGINE:-docker}"
# Default image tag (this literal is pinned by make_dist_bundle to the shipped
# tag). It's only a FALLBACK: when the user gives no explicit image (positional
# arg) and OCI_IMAGE is unset, we resolve the actual loaded coderai image below —
# auto-pick when there's exactly one, ask when there are several — so the runner
# always targets a real installed image instead of a possibly-wrong fixed tag.
IMAGE_TAG="${OCI_IMAGE:-coderai:dist}"
IMAGE_EXPLICIT=0
[[ -n "${OCI_IMAGE:-}" ]] && IMAGE_EXPLICIT=1
# --upgrade: instead of running the server, refresh the in-image application code
# from the git repo's production branch (see the upgrade block near the end).
# UPGRADE_FORCE re-installs even when not strictly newer; UPGRADE_REF/REPO/SSH_KEY
# override the source branch, URL and SSH identity used inside the container.
UPGRADE=0
UPGRADE_FORCE=0
UPGRADE_REF="${CODERAI_UPGRADE_REF:-production}"
UPGRADE_REPO="${CODERAI_UPGRADE_REPO:-}"
UPGRADE_SSH_KEY="${CODERAI_UPGRADE_SSH_KEY:-}"
# --no-pip: refresh code only, don't re-run pip even if dependencies changed.
UPGRADE_SKIP_PIP="${CODERAI_UPGRADE_SKIP_PIP:-0}"
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
# Runtime dir (models/cache, and config when not using --local). Defaults to
# $PWD/coderai-runtime, but for --local it moves to a stable per-user location
# (~/.config/coderai-runtime) so it doesn't depend on the current directory —
# unless the user pins it with --data-dir. DATA_ROOT_EXPLICIT tracks --data-dir.
DATA_ROOT="$PWD/coderai-runtime"
DATA_ROOT_EXPLICIT=0
IS_LOCAL=0
# Run the container as a specific user. DEFAULT: the invoking user (uid:gid) so
# nothing the container creates in the mounted /config|/models|/cache (logs,
# coderai-tmp, hf cache, …) is left root-owned — a single root run would
# otherwise poison those dirs so later --user runs can't write them (the
# "cannot write /cache/logs … logging to stdout only" symptom). Pass --root to
# opt back into the image's root default; pass --user UID[:GID] to pick another.
# When non-empty AND a config dir is used, the config is mounted IN PLACE so the
# app's edits persist to it (files stay owned by you); under --root we fall back
# to a throwaway temp copy so a root container can't leave root-owned config.
# Prefer the pre-sudo identity (SUDO_UID/GID) so `sudo coderai-docker` still runs
# the container as the real user, not root — same anti-footgun goal.
USER_SPEC="${SUDO_UID:-$(id -u)}:${SUDO_GID:-$(id -g)}"
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
# Extra CLI args appended to a tool's command line, keyed by CODERAI_*_ARGS env var.
declare -A TOOL_ARGS=()

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

# Map a friendly tool name to the env var holding EXTRA args for that tool's
# command (supervisord appends %(ENV_CODERAI_*_ARGS)s to each tool launcher).
tool_args_var() {
  case "$1" in
    video-editor|video_editor|editor) echo CODERAI_VIDEO_EDITOR_ARGS ;;
    videogen|video-gen)               echo CODERAI_VIDEOGEN_ARGS ;;
    township|fighters)                echo CODERAI_TOWNSHIP_ARGS ;;
    parler|tts)                       echo CODERAI_PARLER_ARGS ;;
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

Upgrade (refresh the in-image code instead of running the server):
  --upgrade           Fetch the git production branch and, if it is newer than
                      the version baked into the image, replace the in-image
                      application code and commit it back onto the SAME image tag
                      (no rebuild, no overlay image). The server is NOT started.
  --force             With --upgrade, reinstall the fetched code even if it is
                      not strictly newer than what's installed.
  --upgrade-ref REF   Branch/tag/commit to upgrade to (default: production).
  --upgrade-repo URL  Git URL to fetch from (default: the nexlab HTTPS repo, or
                      its SSH form when --ssh-key is given).
  --ssh-key PATH      Host path to an SSH private key; mounted into the upgrade
                      container so git can authenticate over SSH.
  --no-pip            With --upgrade, refresh the code only; do not re-run pip
                      even when the fetched code changed its dependencies.
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
  --data-dir PATH     Directory for config/models/cache. Default ./coderai-runtime,
                      or ~/.config/coderai-runtime when --local is used.
  --name NAME         Container name (default: coderai).
  -d, --detach        Run in background.
  --config-dir PATH   Use an EXISTING config dir (with config.json/models.json),
                      mounted at /config/coderai. Copied to a temp dir by default
                      so the image's host/port rewrite leaves your dir untouched.
  --local             Shortcut for --config-dir ~/.coderai. Also puts the runtime
                      dir under ~/.config/coderai-runtime (override with --data-dir).
                      Add --user to persist the app's config edits back to it.
  --user[=UID[:GID]]  Run the container as that user (no value = your uid:gid).
                      This is the DEFAULT (the invoking user) so the container
                      never leaves root-owned files in the mounts. With a config
                      dir, uses an IN-PLACE mount so config edits persist there,
                      owned by you.
  --root              Run the container as root (the image default). Opts out of
                      the default --user; anything written to the mounts will be
                      root-owned, and --config-dir/--local use a throwaway copy
                      (no persistence) to avoid root-owned config.
  --inplace-config    Mount --config-dir in place (config edits persist there).
  --map HOST[:CONT]   Bind-mount a host dir OR file at the SAME path (or HOST:CONT)
                      inside the container, so absolute paths in models.json (or a
                      tool's config) resolve (e.g. --map /AI/guffcache, or
                      --map /host/video_editor.config.json:/cache/video_editor/video_editor.config.json).
                      Repeatable.
  --debug[=SPEC]      Run coderai with debug flags. SPEC (default 'all'); may be
                      given as --debug=SPEC or --debug SPEC:
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
  --tool-arg TOOL VAL Append ONE extra CLI arg to a bundled tool's command line.
                      Repeatable. TOOL is one of video-editor|videogen|township|
                      parler. e.g. --tool-arg township --web-port --tool-arg township 9000
  --tool-args TOOL STR
                      Like --tool-arg but appends a whole whitespace-separated
                      string at once, e.g. --tool-args video-editor "--voice masculine".
                      Repeatable; combine with --tool-arg. (The video editor already
                      runs with --session on by default, persisting to
                      /cache/video_editor/sessions.)
  -- ARGS             Extra args passed to the container engine before the image name.
  -h, --help          Show this help.

Bring your bare-metal config + data into the container with --map (dir OR file):
  # Server config/models/caches (paths in models.json resolve 1:1):
  packaging/linux/run_oci.sh --nvidia --local \
    --map /AI/guffcache --map /AI/huggingface --map /AI/offloads
  # Township tool (one tree holds its config + all artifacts):
  ...  --map /storage/coderai/township_output:/cache/township_output
  # Video editor (config file + the media_dir/output_dir it points at):
  ...  --map /host/video_editor.config.json:/cache/video_editor/video_editor.config.json \
       --map /host/coderai_media --map /host/video_editor_output
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker) ENGINE=docker; shift ;;
    --podman) ENGINE=podman; shift ;;
    # In-image code upgrade (does not start the server). See the upgrade block
    # after argument parsing.
    --upgrade) UPGRADE=1; shift ;;
    --force) UPGRADE_FORCE=1; shift ;;
    --upgrade-ref)
      [[ $# -ge 2 ]] || { echo "Error: --upgrade-ref requires a branch/tag/commit" >&2; exit 2; }
      UPGRADE_REF="$2"; shift 2 ;;
    --upgrade-repo)
      [[ $# -ge 2 ]] || { echo "Error: --upgrade-repo requires a git URL" >&2; exit 2; }
      UPGRADE_REPO="$2"; shift 2 ;;
    --ssh-key)
      [[ $# -ge 2 ]] || { echo "Error: --ssh-key requires a path" >&2; exit 2; }
      UPGRADE_SSH_KEY="$2"; shift 2 ;;
    --no-pip) UPGRADE_SKIP_PIP=1; shift ;;
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
      DATA_ROOT="$2"; DATA_ROOT_EXPLICIT=1; shift 2 ;;
    --name)
      [[ $# -ge 2 ]] || { echo "Error: --name requires a value" >&2; exit 2; }
      NAME="$2"; shift 2 ;;
    --config-dir)
      [[ $# -ge 2 ]] || { echo "Error: --config-dir requires a path" >&2; exit 2; }
      CONFIG_DIR_SRC="$2"; shift 2 ;;
    --local) CONFIG_DIR_SRC="$HOME/.coderai"; IS_LOCAL=1; shift ;;
    --inplace-config) INPLACE_CONFIG=1; shift ;;
    # --user [UID[:GID]] runs the container as that user (default: your uid:gid).
    # With a config dir, this also switches the mount to IN PLACE so edits persist.
    --user)
      if [[ $# -ge 2 && "$2" =~ ^[0-9] ]]; then USER_SPEC="$2"; shift 2
      else USER_SPEC="$(id -u):$(id -g)"; shift; fi ;;
    --user=*) USER_SPEC="${1#*=}"; shift ;;
    # Opt back into the image's root default (the container runs as root). Anything
    # it writes into the mounts will be root-owned — only use when you really want
    # that (e.g. a shared root-managed data dir).
    --root) USER_SPEC=""; shift ;;
    --map)
      [[ $# -ge 2 ]] || { echo "Error: --map requires HOST[:CONT]" >&2; exit 2; }
      MAPS+=("$2"); shift 2 ;;
    # --debug accepts an optional SPEC as the next token (e.g. --debug engine,ws)
    # OR as --debug=SPEC. The next token is only taken as SPEC when it isn't
    # another option and doesn't look like an image ref (no ':' or '/'), so e.g.
    # `--debug coderai:tag` leaves the tag as the image, not the debug spec.
    --debug)
      if [[ $# -ge 2 && "$2" != -* && "$2" != *[:/]* ]]; then
        DEBUG_SPEC="$2"; shift 2
      else
        DEBUG_SPEC="all"; shift
      fi ;;
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
    # Append extra CLI args to a bundled tool's command. --tool-arg adds ONE token;
    # --tool-args adds a whitespace-separated string. Repeatable. e.g.
    #   --tool-arg township --web-port  --tool-arg township 9000
    #   --tool-args video-editor "--voice masculine"
    --tool-arg)
      [[ $# -ge 3 ]] || { echo "Error: --tool-arg requires TOOL and a value" >&2; exit 2; }
      _v="$(tool_args_var "$2")" || { echo "Error: unknown tool '$2' (video-editor|videogen|township|parler)" >&2; exit 2; }
      TOOL_ARGS["$_v"]="${TOOL_ARGS[$_v]:+${TOOL_ARGS[$_v]} }$3"; shift 3 ;;
    --tool-args)
      [[ $# -ge 3 ]] || { echo "Error: --tool-args requires TOOL and a string" >&2; exit 2; }
      _v="$(tool_args_var "$2")" || { echo "Error: unknown tool '$2' (video-editor|videogen|township|parler)" >&2; exit 2; }
      TOOL_ARGS["$_v"]="${TOOL_ARGS[$_v]:+${TOOL_ARGS[$_v]} }$3"; shift 3 ;;
    -d|--detach) DETACH=1; shift ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) IMAGE_TAG="$1"; IMAGE_EXPLICIT=1; shift ;;
  esac
done

# Resolve the image when the user didn't name one: prefer the single loaded
# coderai image; if several exist, ask (default = the pinned fallback if present,
# else the first). Keeps the runner pointed at a real installed image instead of
# a fixed tag that may not exist after a rebuild/retag.
if [[ "$IMAGE_EXPLICIT" -eq 0 ]]; then
  _imgs=()
  while IFS= read -r _l; do [[ -n "$_l" ]] && _imgs+=("$_l"); done < <(
    "$ENGINE" images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
      | grep -E '^coderai:' | grep -v ':<none>$' | sort -u)
  if [[ "${#_imgs[@]}" -eq 1 ]]; then
    IMAGE_TAG="${_imgs[0]}"
  elif [[ "${#_imgs[@]}" -gt 1 ]]; then
    # Default selection: the pinned fallback if it's one of them, else the first.
    _default="${_imgs[0]}"
    for _i in "${_imgs[@]}"; do [[ "$_i" == "$IMAGE_TAG" ]] && _default="$IMAGE_TAG"; done
    if [[ -t 0 ]]; then
      echo "Multiple coderai images found — choose one to run:" >&2
      _n=1; for _i in "${_imgs[@]}"; do
        _mark=""; [[ "$_i" == "$_default" ]] && _mark="  (default)"
        printf '  %d) %s%s\n' "$_n" "$_i" "$_mark" >&2; _n=$((_n+1))
      done
      printf 'Selection [1-%d, default %s]: ' "${#_imgs[@]}" "$_default" >&2
      read -r _sel </dev/tty || _sel=""
      if [[ -z "$_sel" ]]; then
        IMAGE_TAG="$_default"
      elif [[ "$_sel" =~ ^[0-9]+$ ]] && (( _sel >= 1 && _sel <= ${#_imgs[@]} )); then
        IMAGE_TAG="${_imgs[$((_sel-1))]}"
      else
        echo "Error: invalid selection: $_sel" >&2; exit 2
      fi
    else
      IMAGE_TAG="$_default"
      echo "Note: multiple coderai images; no TTY to choose — using $IMAGE_TAG" >&2
    fi
  fi
  # 0 images: keep the pinned fallback (docker run will report if it's missing).
fi

# ---------------------------------------------------------------------------
# --upgrade: refresh the in-image application code from git, then re-commit the
# SAME image tag. This does NOT start the server. We run the in-image upgrader
# (/usr/local/bin/coderai-upgrade) in a throwaway container as root (so it can
# write /opt/coderai/app), and commit only when it reports an actual update
# (exit 0). Exit 10 = already up to date (no commit); anything else = error.
# ---------------------------------------------------------------------------
if [[ "$UPGRADE" -eq 1 ]]; then
  UP_NAME="${NAME}-upgrade-$$"
  up_args=(run --name "$UP_NAME" --entrypoint /usr/local/bin/coderai-upgrade
           -e "CODERAI_UPGRADE_REF=$UPGRADE_REF"
           -e "CODERAI_UPGRADE_FORCE=$UPGRADE_FORCE"
           -e "CODERAI_UPGRADE_SKIP_PIP=$UPGRADE_SKIP_PIP")
  [[ -n "$UPGRADE_REPO" ]] && up_args+=(-e "CODERAI_UPGRADE_REPO=$UPGRADE_REPO")
  if [[ -n "$UPGRADE_SSH_KEY" ]]; then
    [[ -f "$UPGRADE_SSH_KEY" ]] || { echo "Error: --ssh-key '$UPGRADE_SSH_KEY' not found" >&2; exit 2; }
    _key_abs="$(cd "$(dirname "$UPGRADE_SSH_KEY")" && pwd)/$(basename "$UPGRADE_SSH_KEY")"
    up_args+=(-v "$_key_abs:/tmp/coderai_upgrade_key:ro" -e "CODERAI_UPGRADE_SSH_KEY=/tmp/coderai_upgrade_key")
  fi
  up_args+=("$IMAGE_TAG")

  echo "== CoderAI in-image upgrade =="
  echo "  engine:  $ENGINE"
  echo "  image:   $IMAGE_TAG"
  echo "  ref:     $UPGRADE_REF$([[ "$UPGRADE_FORCE" == "1" ]] && echo '   (force)')"
  echo "  auth:    $([[ -n "$UPGRADE_SSH_KEY" ]] && echo "ssh key $UPGRADE_SSH_KEY" || echo 'https/anonymous')"
  echo "  pip:     $([[ "$UPGRADE_SKIP_PIP" == "1" ]] && echo 'skipped (--no-pip)' || echo 'sync deps if changed')"

  # Make sure a stale upgrade container from an aborted run doesn't block us.
  "$ENGINE" rm -f "$UP_NAME" >/dev/null 2>&1 || true

  set +e
  "$ENGINE" "${up_args[@]}"
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    echo "== committing updated code back onto '$IMAGE_TAG' =="
    if "$ENGINE" commit "$UP_NAME" "$IMAGE_TAG" >/dev/null; then
      echo "== upgrade complete: '$IMAGE_TAG' now carries the new code. =="
      echo "   Restart the server to pick it up (e.g. stop the container and re-run coderai-docker)."
    else
      echo "Error: commit failed; image left unchanged." >&2
      "$ENGINE" rm -f "$UP_NAME" >/dev/null 2>&1 || true
      exit 1
    fi
  elif [[ "$rc" -eq 10 ]]; then
    echo "== no upgrade applied (already up to date); image unchanged. =="
  else
    echo "Error: upgrade failed (exit $rc); image left unchanged." >&2
    "$ENGINE" rm -f "$UP_NAME" >/dev/null 2>&1 || true
    exit "$rc"
  fi
  "$ENGINE" rm -f "$UP_NAME" >/dev/null 2>&1 || true
  exit 0
fi

# For --local, default the runtime dir to a stable per-user location under
# ~/.config instead of the current directory (unless --data-dir was given).
if [[ "$IS_LOCAL" -eq 1 && "$DATA_ROOT_EXPLICIT" -eq 0 ]]; then
  DATA_ROOT="${XDG_CONFIG_HOME:-$HOME/.config}/coderai-runtime"
fi

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
# Forward a HuggingFace token from the host env so the engines authenticate to the
# HF Hub (higher rate limits + gated models) instead of sending unauthenticated
# requests. huggingface_hub auto-reads these; no-op when unset.
for _hv in HF_TOKEN HUGGING_FACE_HUB_TOKEN; do
  if [[ -n "${!_hv:-}" ]]; then args+=(-e "$_hv=${!_hv}"); fi
done
# Pass-through coderai server flags (appended by the in-image launcher's argv).
if [[ -n "$CODERAI_EXTRA_ARGS" ]]; then
  args+=(-e "CODERAI_EXTRA_ARGS=$CODERAI_EXTRA_ARGS")
fi
if [[ -n "$USER_SPEC" ]]; then
  args+=(--user "$USER_SPEC")
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
    # Some hosts set `no-cgroups = true` in /etc/nvidia-container-runtime/config.toml.
    # Then --gpus all injects the device nodes + driver libs but does NOT add them to
    # the container's device-cgroup allowlist, so the kernel blocks GPU access and
    # NVML fails ("Failed to initialize NVML: Unknown Error") -> torch.cuda sees no
    # GPU -> coderai falls back to vulkan/cpu. Passing the nodes via --device adds
    # them to the cgroup allowlist. Harmless when cgroups are managed normally.
    for _dev in /dev/nvidia*; do
      [[ -c "$_dev" ]] && args+=(--device "$_dev")
    done
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
  # Mount IN PLACE (so the app's config edits persist back to the real dir) when
  # explicitly asked (--inplace-config) OR when running as a specific --user (so
  # the files written stay owned by you, not root). The in-image launcher no
  # longer rewrites host/port into config.json (it passes them on the CLI), so an
  # in-place mount is non-destructive. Without --user, fall back to a throwaway
  # copy so a root container can't leave root-owned files in your real config.
  if [[ "$INPLACE_CONFIG" == "1" || -n "$USER_SPEC" ]]; then
    CFG_MOUNT="$CONFIG_DIR_SRC"
    CONFIG_NOTE="$CONFIG_DIR_SRC (in place — edits persist here)"
  else
    # Copy ONLY the json config files to a throwaway dir so nothing in the
    # container can touch your real config, and we don't copy big subdirs
    # (e.g. ~/.coderai/ds4 weights).
    CFG_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/coderai-cfg.XXXXXX")"
    CFG_MOUNT="$CFG_PARENT/coderai"
    mkdir -p "$CFG_MOUNT"
    cp -a "$CONFIG_DIR_SRC"/*.json "$CFG_MOUNT/" 2>/dev/null || true
    [[ -f "$CFG_MOUNT/config.json" ]] || { echo "Error: no config.json in '$CONFIG_DIR_SRC'" >&2; exit 2; }
    CONFIG_NOTE="$CONFIG_DIR_SRC → $CFG_MOUNT (temporary copy; original untouched)"
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
  if [[ -e "$host" ]]; then
    # Accept a directory OR a single file (e.g. a tool's config.json that lives
    # loose on the host rather than in a dedicated dir). Docker bind-mounts both.
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
# Per-tool extra CLI args (CODERAI_*_ARGS) → supervisord appends them to the tool.
TOOLARGS_NOTE="(none)"
if [[ "${#TOOL_ARGS[@]}" -gt 0 ]]; then
  TOOLARGS_NOTE=""
  for _v in "${!TOOL_ARGS[@]}"; do
    args+=(-e "$_v=${TOOL_ARGS[$_v]}")
    TOOLARGS_NOTE+="${_v}='${TOOL_ARGS[$_v]}' "
  done
  TOOLARGS_NOTE="${TOOLARGS_NOTE% }"
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
  tool-args:$([[ "$TOOLARGS_NOTE" == "(none)" ]] && echo " (none)" || echo " $TOOLARGS_NOTE")
  user:    ${USER_SPEC:-container default (root)}
  cdr-args:${CODERAI_EXTRA_ARGS:+ $CODERAI_EXTRA_ARGS}
EOF

if [[ "$LOG_HOST_NOTE" != "(none)" ]]; then
  echo "  tail it:  tail -F '$LOG_HOST_NOTE'"
fi

exec "$ENGINE" "${args[@]}"
