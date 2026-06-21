#!/usr/bin/env bash
# Assemble a self-contained distribution bundle the recipient can expand and
# install with no repo checkout:
#
#   <NAME>/
#     install.sh            loads the image + installs the runner (see dist-bundle/)
#     uninstall.sh          removes the runner + (optionally) the image
#     coderai-docker        the run wrapper (run_oci.sh, image tag pinned)
#     coderai-dist.tar.gz   the image (gzip-compressed `docker save`)
#     README.txt / README.md
#
# Output: $OUT_DIR/<NAME>.tar  (not re-gzipped — the image is already compressed).
#
# Usage:
#   [DOCKER="sudo docker"] [IMAGE=coderai:dist] packaging/linux/make_dist_bundle.sh
#   REFRESH=1 ...   # force re-export of the image tarball even if present
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DOCKER_BIN="${DOCKER:-docker}"; read -r -a DK <<< "$DOCKER_BIN"
IMAGE="${IMAGE:-coderai:dist}"
OUT_DIR="${OUT_DIR:-$REPO/dist}"
NAME="${NAME:-coderai-docker-dist}"
IMAGE_TAR="$OUT_DIR/coderai-dist.tar.gz"

command -v "${DK[0]}" >/dev/null 2>&1 || { echo "Error: '${DK[0]}' not found" >&2; exit 1; }
"${DK[@]}" image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "Error: image '$IMAGE' not found — build it first (build_oci_image.sh)." >&2; exit 1; }

mkdir -p "$OUT_DIR"

# 1. Export the image tarball (skip if present unless REFRESH=1).
if [[ ! -f "$IMAGE_TAR" || "${REFRESH:-0}" == "1" ]]; then
  if command -v pigz >/dev/null 2>&1; then COMP=pigz; else COMP=gzip; fi
  echo "[bundle] exporting $IMAGE -> $IMAGE_TAR (via $COMP)…"
  "${DK[@]}" save "$IMAGE" | "$COMP" > "$IMAGE_TAR"
else
  echo "[bundle] reusing existing $IMAGE_TAR (set REFRESH=1 to re-export)"
fi

# 2. Stage the bundle tree. Stage UNDER $OUT_DIR (not /tmp, which is often a
# small tmpfs) and hardlink the multi-GB image tarball when possible so we don't
# duplicate it on disk.
TMP="$(mktemp -d "$OUT_DIR/.bundle.XXXXXX")"
STAGE="$TMP/$NAME"
mkdir -p "$STAGE"
ln "$IMAGE_TAR" "$STAGE/coderai-dist.tar.gz" 2>/dev/null \
  || cp "$IMAGE_TAR" "$STAGE/coderai-dist.tar.gz"
install -m 0755 "$HERE/run_oci.sh"            "$STAGE/coderai-docker"
install -m 0755 "$HERE/dist-bundle/install.sh"   "$STAGE/install.sh"
install -m 0755 "$HERE/dist-bundle/uninstall.sh" "$STAGE/uninstall.sh"
cp "$HERE/dist-bundle/README.txt"             "$STAGE/README.txt"
cp "$HERE/dist-bundle/README.md"              "$STAGE/README.md"

# Pin the runner's default image tag to exactly what we shipped, so it matches
# the tag restored by `docker load` regardless of the build's tag.
sed -i "s|\${OCI_IMAGE:-coderai:dist}|\${OCI_IMAGE:-$IMAGE}|" "$STAGE/coderai-docker"

# 3. Tar it (store, no extra gzip — coderai-dist.tar.gz is already compressed).
BUNDLE="$OUT_DIR/$NAME.tar"
tar -C "$TMP" -cf "$BUNDLE" "$NAME"
rm -rf "$TMP"

echo "[bundle] wrote $BUNDLE ($(du -h "$BUNDLE" | cut -f1))"
echo "[bundle] recipient:  tar -xf $(basename "$BUNDLE") && cd $NAME && ./install.sh"
