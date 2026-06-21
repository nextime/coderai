#!/usr/bin/env bash
# CoderAI Docker distribution uninstaller — reverses install.sh. It:
#   1. Removes the `coderai-docker` runner from both the root and user install
#      dirs (/usr/local/bin and ~/.local/usr/bin).
#   2. Removes the loaded image (unless --keep-image).
#   3. Points out the PATH line install.sh may have added to ~/.bashrc.
#
# It does NOT touch your runtime data (coderai-runtime/, ~/.coderai) — those are
# yours; delete them by hand if you want them gone.
#
# Usage:  ./uninstall.sh [--keep-image] [--image TAG] [--yes]
# Env:    CONTAINER_ENGINE=docker|podman   (default docker)
#         OCI_IMAGE=TAG                     image tag to remove (default coderai:dist)
set -euo pipefail

ENGINE="${CONTAINER_ENGINE:-docker}"
IMAGE="${OCI_IMAGE:-coderai:dist}"
KEEP_IMAGE=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-image) KEEP_IMAGE=1; shift ;;
    --image) [[ $# -ge 2 ]] || { echo "Error: --image requires a tag" >&2; exit 2; }; IMAGE="$2"; shift 2 ;;
    -y|--yes) ASSUME_YES=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Error: unknown option: $1" >&2; exit 2 ;;
  esac
done

say(){ printf '%s\n' "$*"; }
die(){ printf 'Error: %s\n' "$*" >&2; exit 1; }

# Tell the user what this will do, and confirm before proceeding.
say "This will remove the 'coderai-docker' runner and$([ "$KEEP_IMAGE" -eq 1 ] && echo ' keep' || echo ' (after a prompt) remove') the CoderAI image"
say "'$IMAGE' from $ENGINE. Your runtime data and config are not touched."
if [ "$ASSUME_YES" -ne 1 ]; then
  printf 'Proceed? [y/N] '
  read -r reply </dev/tty || reply=""
  case "$reply" in y|Y|yes|YES) ;; *) die "aborted by user." ;; esac
fi

# 1. Remove the runner from every place install.sh may have put it.
removed_any=0
for d in /usr/local/bin "$HOME/.local/usr/bin"; do
  f="$d/coderai-docker"
  [ -e "$f" ] || continue
  if [ -w "$d" ]; then
    rm -f "$f" && { say "[uninstall] removed runner: $f"; removed_any=1; }
  elif command -v sudo >/dev/null 2>&1; then
    sudo rm -f "$f" && { say "[uninstall] removed runner (sudo): $f"; removed_any=1; }
  else
    say "[uninstall] cannot remove $f (no write permission and no sudo) — remove it manually."
  fi
done
[ "$removed_any" -eq 1 ] || say "[uninstall] no coderai-docker runner found in the standard locations."

# 2. Remove the image unless asked to keep it.
if [ "$KEEP_IMAGE" -eq 0 ]; then
  DK=("$ENGINE")
  if ! "$ENGINE" info >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then DK=(sudo "$ENGINE"); fi
  fi
  if "${DK[@]}" image inspect "$IMAGE" >/dev/null 2>&1; then
    if [ "$ASSUME_YES" -ne 1 ]; then
      printf '[uninstall] remove image "%s"? [y/N] ' "$IMAGE"
      read -r reply </dev/tty || reply=""
      case "$reply" in y|Y|yes|YES) ;; *) say "[uninstall] keeping image $IMAGE."; KEEP_IMAGE=1 ;; esac
    fi
    if [ "$KEEP_IMAGE" -eq 0 ]; then
      "${DK[@]}" image rm "$IMAGE" && say "[uninstall] removed image: $IMAGE"
    fi
  else
    say "[uninstall] image '$IMAGE' not present (use --image TAG if you tagged it differently)."
  fi
else
  say "[uninstall] keeping image (--keep-image)."
fi

# 3. Note the PATH line install.sh may have appended (we don't edit ~/.bashrc).
RC="${HOME}/.bashrc"
if [ -f "$RC" ] && grep -Fqs "Added by the CoderAI Docker installer" "$RC"; then
  say ""
  say "[uninstall] $RC still has the PATH line install.sh added. Remove it if you like:"
  say "    # Added by the CoderAI Docker installer"
  say "    export PATH=\"\$HOME/.local/usr/bin:\$PATH\""
fi

say ""
say "Done. Runtime data (coderai-runtime/, ~/.coderai) was left untouched."
