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
OUT_DIR="$ROOT_DIR/dist"
OUT_NAME=""
COMPRESS=1

usage() {
  cat <<'EOF'
Usage:
  packaging/linux/package_oci_image.sh [IMAGE_TAG]
  packaging/linux/package_oci_image.sh -t IMAGE_TAG -o dist/name.tar.zst

Options:
  -t, --tag TAG       Image tag to export (default: coderai:local or OCI_IMAGE).
  -o, --output PATH   Output archive path. Defaults to dist/coderai-oci-<tag>.tar.zst.
  --no-compress       Write an uncompressed docker-save .tar.
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
    -o|--output)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a path" >&2; exit 2; }
      OUT_NAME="$2"
      shift 2
      ;;
    --no-compress)
      COMPRESS=0
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

if ! "${DOCKER_CMD[@]}" image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "Error: image not found: $IMAGE_TAG" >&2
  echo "Build it first with ./build-oci.sh" >&2
  exit 1
fi

safe_tag="${IMAGE_TAG//[^A-Za-z0-9_.-]/-}"
mkdir -p "$OUT_DIR"
if [[ -z "$OUT_NAME" ]]; then
  if [[ "$COMPRESS" == "1" ]]; then
    OUT_NAME="$OUT_DIR/coderai-oci-${safe_tag}.tar.zst"
  else
    OUT_NAME="$OUT_DIR/coderai-oci-${safe_tag}.tar"
  fi
fi

if [[ "$COMPRESS" == "1" ]]; then
  if ! command -v zstd >/dev/null 2>&1; then
    echo "Error: zstd is required for compressed export. Use --no-compress or install zstd." >&2
    exit 1
  fi
  "${DOCKER_CMD[@]}" save "$IMAGE_TAG" | zstd -T0 -19 -o "$OUT_NAME"
else
  "${DOCKER_CMD[@]}" save -o "$OUT_NAME" "$IMAGE_TAG"
fi

sha256sum "$OUT_NAME" > "$OUT_NAME.sha256"

cat <<EOF
Exported OCI image artifact:
  archive:  $OUT_NAME
  checksum: $OUT_NAME.sha256

Load it with:
  $DOCKER_BIN load -i "$OUT_NAME"
EOF
