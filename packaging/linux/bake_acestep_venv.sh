#!/usr/bin/env bash
# Produce the ACE-Step variant of an already-built image: BASE_IMAGE + the
# python-3.12 venv, as one layer, in a couple of minutes.
#
# The slim image is the default because the venv is ~6.4 GB and the worker
# builds it on the first request to an ACE-Step model. This is for installs that
# would rather pay the size than the first-call wait, and for anyone who wants
# the two variants to be provably the same image.
#
# Usage:
#   ./bake_acestep_venv.sh                       # coderai:base -> coderai:base-acestep
#   ./bake_acestep_venv.sh coderai:full_all_0.5.10
#   BASE_IMAGE=... TAG=... ACESTEP_VENV=... ./bake_acestep_venv.sh
#
# The venv comes from the development install by default -- it is already built
# and already the thing that was tested. build_oci_image.sh --with-acestep is
# the other route: it bundles the same venv during a full build.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

BASE_IMAGE="${1:-${BASE_IMAGE:-coderai:base}}"
TAG="${TAG:-${BASE_IMAGE%%:*}:${BASE_IMAGE##*:}-acestep}"
ACESTEP_VENV="${ACESTEP_VENV:-${CODERAI_ACESTEP_VENV:-$HOME/.coderai/acestep_venv}}"
DOCKER_BIN="${DOCKER:-docker}"
read -r -a DK <<< "$DOCKER_BIN"

if ! "${DK[@]}" image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "Error: base image '$BASE_IMAGE' not found — build it first (build_oci_image.sh)." >&2
  exit 1
fi
if [[ ! -f "$ACESTEP_VENV/pyvenv.cfg" ]]; then
  echo "Error: no ACE-Step venv at '$ACESTEP_VENV' (no pyvenv.cfg)." >&2
  echo "       Build one with:  build_oci_image.sh --with-acestep   (it builds when none exists)" >&2
  echo "       or let the worker build it on the first request and re-run this." >&2
  exit 2
fi

# A venv is tied to its interpreter's minor version. The image supplies 3.12
# (Ubuntu 24.04) or carries a relocatable one; a venv on any other minor would
# import nothing, and the Dockerfile's check would be the first to notice --
# after copying 6.4 GB.
_minor="$(sed -n 's/^version_info *= *//p' "$ACESTEP_VENV/pyvenv.cfg" | cut -d. -f1,2)"
if [[ -n "$_minor" && "$_minor" != "3.12" ]]; then
  echo "Error: '$ACESTEP_VENV' is a python $_minor venv; ACE-Step needs 3.12." >&2
  exit 2
fi

echo "== baking ACE-Step venv into '$TAG' =="
echo "   base: $BASE_IMAGE"
echo "   venv: $ACESTEP_VENV ($(du -sh "$ACESTEP_VENV" | cut -f1))"
t0=$(date +%s)
DOCKER_BUILDKIT=1 "${DK[@]}" build \
  -f "$HERE/Dockerfile.acestep-overlay" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-context "acestep_venv=$ACESTEP_VENV" \
  -t "$TAG" \
  "$REPO_ROOT"
echo "== done in $(( $(date +%s) - t0 ))s: '$TAG' =="
"${DK[@]}" images --format '   {{.Repository}}:{{.Tag}}  {{.Size}}' "$TAG"
