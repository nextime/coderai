#!/usr/bin/env bash
# Fast incremental image update: re-layer ONLY the coderai app code + launcher
# scripts + service configs on top of an already-built image. No 20 GB bundle
# recopy — seconds, not the ~15 min of a full build_oci_image.sh run.
#
# It keeps an immutable `coderai:base` tag (the heavy bundle) and rebuilds the
# shipped `coderai:dist` as base + a thin app layer. Because every update starts
# from the SAME base, app layers never stack up over repeated updates.
#
# Usage:
#   [DOCKER="sudo docker"] ./update_oci_image.sh
#   BASE_IMAGE=coderai:base TAG=coderai:dist DOCKER="sudo docker" ./update_oci_image.sh
#
# First run seeds coderai:base from the current coderai:dist. To re-baseline the
# bundle (new venv/libs/tools), run build_oci_image.sh and then:
#   docker rmi coderai:base   # drop the stale base; next update re-seeds it
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
DOCKER_BIN="${DOCKER:-docker}"
read -r -a DK <<< "$DOCKER_BIN"
BASE_IMAGE="${BASE_IMAGE:-coderai:base}"
TAG="${TAG:-coderai:dist}"
SEED_FROM="${SEED_FROM:-coderai:dist}"

img_exists(){ "${DK[@]}" image inspect "$1" >/dev/null 2>&1; }

# Seed the immutable base from a previously built full image if it doesn't exist.
if ! img_exists "$BASE_IMAGE"; then
  if img_exists "$SEED_FROM"; then
    echo "== seeding immutable base '$BASE_IMAGE' from '$SEED_FROM' =="
    "${DK[@]}" tag "$SEED_FROM" "$BASE_IMAGE"
  else
    echo "Base '$BASE_IMAGE' and seed '$SEED_FROM' both missing." >&2
    echo "Run packaging/linux/build_oci_image.sh for a full build first." >&2
    exit 1
  fi
fi

echo "== updating '$TAG' from base '$BASE_IMAGE' (app code only) =="
t0=$(date +%s)
"${DK[@]}" build \
  -f "$HERE/Dockerfile.update" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t "$TAG" "$REPO_ROOT"
echo "== done in $(( $(date +%s) - t0 ))s: '$TAG' (base '$BASE_IMAGE' unchanged) =="

# REBUILD_LLAMA=1: also rebuild llama-cpp-python (with the RPC backend) and
# rpc-server in a CUDA devel stage and lay them over '$TAG' — the one part of
# the bundle the app overlay cannot refresh (Dockerfile.update-llama).
if [ "${REBUILD_LLAMA:-0}" = "1" ]; then
  echo "== rebuilding llama-cpp-python + rpc-server over '$TAG' =="
  t1=$(date +%s)
  DOCKER_BUILDKIT=1 "${DK[@]}" build \
    -f "$HERE/Dockerfile.update-llama" \
    --build-arg BASE_IMAGE="$TAG" \
    ${CUDA_BUILDER:+--build-arg CUDA_BUILDER="$CUDA_BUILDER"} \
    ${CUDA_ARCHS:+--build-arg CUDA_ARCHS="$CUDA_ARCHS"} \
    -t "$TAG" "$REPO_ROOT"
  echo "== llama runtime rebuilt in $(( $(date +%s) - t1 ))s =="
fi
echo "   Tip: 'docker image prune -f' to drop the now-dangling previous '$TAG' layer."
