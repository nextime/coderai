#!/usr/bin/env bash
# CoderAI Docker distribution uninstaller — reverses install.sh. It:
#   1. Removes the `coderai-docker` runner from both the root and user install
#      dirs (/usr/local/bin and ~/.local/usr/bin).
#   2. Removes the installed CoderAI image(s) (unless --keep-image). With no
#      --image, it resolves the actual loaded coderai image(s): one -> that one,
#      several -> asks which (or all). --image TAG targets a specific tag.
#   3. Points out the PATH line install.sh may have added to ~/.bashrc.
#
# It does NOT touch your runtime data (coderai-runtime/, ~/.coderai) — those are
# yours; delete them by hand if you want them gone.
#
# Usage:  ./uninstall.sh [--keep-image] [--image TAG] [--yes]
# Env:    CONTAINER_ENGINE=docker|podman   (default docker)
#         OCI_IMAGE=TAG                     specific image tag to remove
set -euo pipefail

ENGINE="${CONTAINER_ENGINE:-docker}"
IMAGE="${OCI_IMAGE:-}"
IMAGE_EXPLICIT=0
[ -n "$IMAGE" ] && IMAGE_EXPLICIT=1
KEEP_IMAGE=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-image) KEEP_IMAGE=1; shift ;;
    --image) [[ $# -ge 2 ]] || { echo "Error: --image requires a tag" >&2; exit 2; }; IMAGE="$2"; IMAGE_EXPLICIT=1; shift 2 ;;
    -y|--yes) ASSUME_YES=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "Error: unknown option: $1" >&2; exit 2 ;;
  esac
done

say(){ printf '%s\n' "$*"; }
die(){ printf 'Error: %s\n' "$*" >&2; exit 1; }

# Tell the user what this will do, and confirm before proceeding.
if [ "$KEEP_IMAGE" -eq 1 ]; then _img_note="keep the CoderAI image(s)";
elif [ "$IMAGE_EXPLICIT" -eq 1 ]; then _img_note="(after a prompt) remove the image '$IMAGE'";
else _img_note="(after a prompt) remove the installed CoderAI image(s)"; fi
say "This will remove the 'coderai-docker' runner and $_img_note from $ENGINE."
say "Your runtime data and config are not touched."
if [ "$ASSUME_YES" -ne 1 ]; then
  printf 'Proceed? [y/N] '
  read -r reply </dev/tty || reply=""
  case "$reply" in y|Y|yes|YES) ;; *) die "aborted by user." ;; esac
fi

# Engine access (sudo only if needed for the daemon).
DK=("$ENGINE")
if ! "$ENGINE" info >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then DK=(sudo "$ENGINE"); fi
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

# Remove one image tag (helper).
rm_image(){
  "${DK[@]}" image rm "$1" >/dev/null 2>&1 && say "[uninstall] removed image: $1" \
    || say "[uninstall] could not remove image: $1"
}

# 2. Remove the image(s) unless asked to keep them.
if [ "$KEEP_IMAGE" -eq 1 ]; then
  say "[uninstall] keeping image(s) (--keep-image)."
elif [ "$IMAGE_EXPLICIT" -eq 1 ]; then
  # Explicit tag: remove just that one.
  if "${DK[@]}" image inspect "$IMAGE" >/dev/null 2>&1; then
    if [ "$ASSUME_YES" -eq 1 ]; then rm_image "$IMAGE"; else
      printf '[uninstall] remove image "%s"? [y/N] ' "$IMAGE"
      read -r reply </dev/tty || reply=""
      case "$reply" in y|Y|yes|YES) rm_image "$IMAGE" ;; *) say "[uninstall] keeping image $IMAGE." ;; esac
    fi
  else
    say "[uninstall] image '$IMAGE' not present."
  fi
else
  # Resolve the installed coderai image(s).
  IMGS=()
  while IFS= read -r line; do [ -n "$line" ] && IMGS+=("$line"); done < <(
    "${DK[@]}" images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
      | grep -E '^coderai:' | grep -v ':<none>$' | sort -u)
  case "${#IMGS[@]}" in
    0) say "[uninstall] no coderai image found to remove." ;;
    1)
      if [ "$ASSUME_YES" -eq 1 ]; then rm_image "${IMGS[0]}"; else
        printf '[uninstall] remove image "%s"? [y/N] ' "${IMGS[0]}"
        read -r reply </dev/tty || reply=""
        case "$reply" in y|Y|yes|YES) rm_image "${IMGS[0]}" ;; *) say "[uninstall] keeping image ${IMGS[0]}." ;; esac
      fi
      ;;
    *)
      if [ "$ASSUME_YES" -eq 1 ]; then
        say "[uninstall] multiple coderai images; --yes -> removing all:"
        for img in "${IMGS[@]}"; do rm_image "$img"; done
      else
        say "Multiple coderai images found:"
        i=1; for img in "${IMGS[@]}"; do printf '  %d) %s\n' "$i" "$img"; i=$((i+1)); done
        printf 'Remove which? [1-%d / a=all / n=none, default n]: ' "${#IMGS[@]}"
        read -r sel </dev/tty || sel=""
        case "$sel" in
          a|A|all) for img in "${IMGS[@]}"; do rm_image "$img"; done ;;
          ""|n|N|no|NO) say "[uninstall] keeping all images." ;;
          *)
            if printf '%s' "$sel" | grep -qE '^[0-9]+$' && [ "$sel" -ge 1 ] && [ "$sel" -le "${#IMGS[@]}" ]; then
              rm_image "${IMGS[$((sel-1))]}"
            else
              die "invalid selection: $sel"
            fi
            ;;
        esac
      fi
      ;;
  esac
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
