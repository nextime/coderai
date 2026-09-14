#!/usr/bin/env bash
# Build (and optionally push) a trimmed coderai image for one capability.
#
#   ./packaging/runpod/build_capability_image.sh video
#   ./packaging/runpod/build_capability_image.sh video ghcr.io/me/coderai-video:0.1
#   PUSH=1 ./packaging/runpod/build_capability_image.sh video ghcr.io/me/coderai-video:0.1
#
# The image is what a RunPod capability pod runs (`engine: coderai`). Push it to
# a registry RunPod can pull from and put the tag in the pod block's `image`;
# for a private registry also set `registry_auth_id`. See
# docs/remote-execution.md.
set -euo pipefail

cd "$(dirname "$0")/../.."
PROFILE="${1:-}"
TAG="${2:-coderai-${PROFILE}:latest}"
PROFILES_DIR="packaging/runpod/profiles"

if [[ -z "$PROFILE" ]]; then
    echo "usage: $0 <profile> [tag]" >&2
    echo "profiles: $(ls "$PROFILES_DIR" | grep -v '^core.txt$' | sed 's/\.txt$//' | tr '\n' ' ')" >&2
    exit 2
fi
if [[ ! -f "$PROFILES_DIR/$PROFILE.txt" ]]; then
    echo "no such profile: $PROFILE (see $PROFILES_DIR/)" >&2
    exit 2
fi

# The shared core image. Built once and reused by every profile, so the nine
# images share those layers by construction — the registry stores and transfers
# the core once instead of nine near-identical ~7 GB copies. Rebuild it with
# REBUILD_CORE=1 (needed when core.txt changes).
CORE_BASE="${CORE_BASE:-coderai-capability-base:latest}"
if [[ "${REBUILD_CORE:-0}" == "1" ]] || ! docker image inspect "$CORE_BASE" >/dev/null 2>&1; then
    echo "==> building the shared core $CORE_BASE"
    DOCKER_BUILDKIT=1 docker build -f packaging/runpod/Dockerfile.capability-base \
        -t "$CORE_BASE" . || { echo "core build failed" >&2; exit 1; }
else
    echo "==> reusing the shared core $CORE_BASE"
fi

echo "==> building $TAG from profile '$PROFILE'"
# BuildKit for the pip cache mount: nine profiles, one torch download.
DOCKER_BUILDKIT=1 docker build -f packaging/runpod/Dockerfile.capability \
    --build-arg "PROFILE=$PROFILE" \
    --build-arg "CORE_BASE=$CORE_BASE" \
    -t "$TAG" .

# The Dockerfile already proves the app imports. This proves it SERVES: a pod is
# probed on /healthz, so a container that boots but never answers is the failure
# mode worth catching here rather than on a rented GPU.
echo "==> smoke-testing $TAG"
cid=$(docker run -d -P "$TAG")
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
port=$(docker port "$cid" 8000/tcp | head -1 | sed 's/.*://')
for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${port}/healthz" >/dev/null 2>&1; then
        echo "    /healthz OK"
        ok=1
        break
    fi
    sleep 2
done
if [[ "${ok:-0}" != "1" ]]; then
    echo "!!! $TAG never answered /healthz — logs follow" >&2
    docker logs --tail 60 "$cid" >&2
    exit 1
fi

size=$(docker image inspect "$TAG" --format '{{.Size}}')
echo "==> $TAG built and healthy ($(( size / 1000000 )) MB)"

if [[ "${PUSH:-0}" == "1" ]]; then
    echo "==> pushing $TAG"
    docker push "$TAG"
fi
