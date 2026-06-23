#!/usr/bin/env bash
# CoderAI Docker distribution installer.
#
# Expand the distribution tarball and run this script. It:
#   1. Loads the bundled image into Docker (docker load).
#   2. Installs the `coderai-docker` runner:
#        - root  -> /usr/local/bin/coderai-docker
#        - user  -> ~/.local/usr/bin/coderai-docker  (and ensures it's on PATH,
#                   adding it to ~/.bashrc if missing).
#
# Usage:  ./install.sh [--yes]
# Env:    CONTAINER_ENGINE=docker|podman   (default docker)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAR="${IMAGE_TAR:-$HERE/coderai-dist.tar.gz}"
RUNNER_SRC="${RUNNER_SRC:-$HERE/coderai-docker}"
ENGINE="${CONTAINER_ENGINE:-docker}"
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) ASSUME_YES=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) printf 'Error: unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

say(){ printf '%s\n' "$*"; }
die(){ printf 'Error: %s\n' "$*" >&2; exit 1; }

# Tell the user what this will do, and confirm before proceeding.
if [ "$(id -u)" -eq 0 ]; then _bin="/usr/local/bin"; else _bin="$HOME/.local/usr/bin"; fi
say "This will load the CoderAI image into $ENGINE and install the 'coderai-docker'"
say "runner to $_bin. Your runtime data and config are not touched."
if [ "$ASSUME_YES" -ne 1 ]; then
  printf 'Proceed? [y/N] '
  read -r reply </dev/tty || reply=""
  case "$reply" in y|Y|yes|YES) ;; *) die "aborted by user." ;; esac
fi

command -v "$ENGINE" >/dev/null 2>&1 || die "'$ENGINE' not found in PATH — install Docker (or set CONTAINER_ENGINE=podman) first."
[ -f "$IMAGE_TAR" ]  || die "image tarball not found: $IMAGE_TAR"
[ -f "$RUNNER_SRC" ] || die "runner script not found: $RUNNER_SRC"

# Decide whether the engine needs sudo for daemon access.
DK=("$ENGINE")
if ! "$ENGINE" info >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    DK=(sudo "$ENGINE")
    say "[install] Docker daemon needs elevated access — using sudo (you may be prompted)."
  else
    die "cannot reach the Docker daemon (no permission and no sudo available)."
  fi
fi

say "[install] loading image from $IMAGE_TAR — this is large, please wait…"
LOAD_OUT="$("${DK[@]}" load -i "$IMAGE_TAR")"
say "$LOAD_OUT"
# e.g. "Loaded image: coderai:full_all_0.1.0" — the tag we just installed.
LOADED_TAG="$(printf '%s\n' "$LOAD_OUT" | sed -n 's/^Loaded image: //p' | head -1)"

# Pick the install dir by privilege.
if [ "$(id -u)" -eq 0 ]; then
  BIN_DIR="/usr/local/bin"
else
  BIN_DIR="$HOME/.local/usr/bin"
fi
mkdir -p "$BIN_DIR"
install -m 0755 "$RUNNER_SRC" "$BIN_DIR/coderai-docker"
say "[install] installed runner: $BIN_DIR/coderai-docker"

# Ensure a user install dir is on PATH (root's /usr/local/bin already is).
if [ "$(id -u)" -ne 0 ]; then
  case ":${PATH:-}:" in
    *":$BIN_DIR:"*)
      say "[install] $BIN_DIR is already on PATH." ;;
    *)
      RC="${HOME}/.bashrc"
      if [ -f "$RC" ] && grep -Fqs "$BIN_DIR" "$RC"; then
        say "[install] $RC already references $BIN_DIR; open a new shell or 'source $RC'."
      else
        printf '\n# Added by the CoderAI Docker installer\nexport PATH="%s:$PATH"\n' "$BIN_DIR" >> "$RC"
        say "[install] added $BIN_DIR to PATH in $RC."
        say "[install] run:  source \"$RC\"   (or open a new terminal) to pick it up now."
      fi
      ;;
  esac
fi

# Offer to remove OTHER coderai images (e.g. older versions) — default NO.
OTHERS=()
while IFS= read -r line; do
  [ -n "$line" ] && [ "$line" != "$LOADED_TAG" ] && OTHERS+=("$line")
done < <(
  "${DK[@]}" images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
    | grep -E '^coderai:' | grep -v ':<none>$' | sort -u)

if [ "${#OTHERS[@]}" -gt 0 ]; then
  say ""
  say "[install] other coderai image(s) already present:"
  for img in "${OTHERS[@]}"; do say "    $img"; done
  # Default NO — only remove on an explicit yes (and not under non-interactive --yes).
  if [ "$ASSUME_YES" -ne 1 ]; then
    printf '[install] remove the other image(s) above? [y/N] '
    read -r reply </dev/tty || reply=""
    case "$reply" in
      y|Y|yes|YES)
        for img in "${OTHERS[@]}"; do
          "${DK[@]}" image rm "$img" >/dev/null 2>&1 && say "[install] removed $img" \
            || say "[install] could not remove $img (in use?)"
        done ;;
      *) say "[install] keeping the other image(s)." ;;
    esac
  else
    say "[install] --yes given; keeping the other image(s) (default)."
  fi
fi

say ""
say "═══════════════════════════════════════════════════════════════════════════"
say " CoderAI installed."
say "═══════════════════════════════════════════════════════════════════════════"
[ -n "$LOADED_TAG" ] && say " Image:  $LOADED_TAG"
say " Runner: $BIN_DIR/coderai-docker"
if [ "$(id -u)" -ne 0 ] && ! case ":${PATH:-}:" in *":$BIN_DIR:"*) true;; *) false;; esac; then
  say " NOTE:   $BIN_DIR is not on PATH yet — open a new terminal or 'source ~/.bashrc'."
fi
say ""
say " QUICK START"
say "   coderai-docker --nvidia                 # GPU via NVIDIA; or --vulkan / --cpu"
say "   coderai-docker --nvidia -d              # run detached (background)"
say "   then open:  http://localhost:8776/admin # web UI (default port 8776)"
say ""
say " USE YOUR LOCAL CONFIG + DATA (recommended)"
say "   coderai-docker --nvidia --local \\"
say "       --map /AI/huggingface --map /AI/offloads"
say "   --local     reads config/models.json from ~/.coderai (edits persist there)."
say "   --map H[:C] bind-mounts a host dir so absolute paths in models.json resolve;"
say "               repeat it for each models/cache location you reference."
say ""
say " WHERE THINGS LIVE"
say "   Runtime data (models/cache): ./coderai-runtime, or ~/.config/coderai-runtime"
say "   with --local (override with --data-dir PATH)."
say "   The container runs as YOU by default (uid:gid), so nothing it writes to"
say "   those dirs is root-owned. Use --root only if you deliberately want root."
say ""
say " LOGS"
say "   coderai-docker --nvidia --debug         # tee a host-tailable file log"
say "   Default log path: <cache>/logs/coderai.log (also visible via 'docker logs')."
say "   tail -F ~/.config/coderai-runtime/cache/logs/coderai.log   # with --local"
say ""
say " MANAGE"
say "   docker logs -f coderai     # follow output of a detached run"
say "   docker stop coderai        # stop it (the container is --rm, so it's removed)"
say ""
say " ALL OPTIONS"
say "   coderai-docker --help      # port/host, tools, --root, --user, --config-dir, …"
say "═══════════════════════════════════════════════════════════════════════════"
