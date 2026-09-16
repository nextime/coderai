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

# The slim image carries no ML dependencies: it runs coderai from a venv on a
# network volume (see venv_on_volume). Built from its own Dockerfile, with no
# profile and no import smoke test — there is nothing to import yet.
if [[ "$PROFILE" == "slim" ]]; then
    TAG="${2:-coderai-slim:latest}"
    echo "==> building $TAG (no ML dependencies; runs from a venv on a volume)"
    DOCKER_BUILDKIT=1 docker build -f packaging/runpod/Dockerfile.capability-slim \
        -t "$TAG" . || exit 1
    size=$(docker image inspect "$TAG" --format '{{.Size}}')
    echo "==> $TAG built ($(( size / 1000000 )) MB)"
    if [[ "${PUSH:-0}" == "1" ]]; then docker push "$TAG"; fi
    exit 0
fi

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
# A profile that runs its capability in a venv of its own starts from the LIGHT
# core — coderai's deps and no GPU stack — marked by a `<profile>.light` file.
# RunPod refuses images above a size line (10.1 GB boots, 14.5 GB is refused),
# and a specialised image on the full core carried 7 GB of torch and CUDA its
# main venv never touched. See Dockerfile.capability-base-light.
if [[ -f "packaging/runpod/profiles/${PROFILE}.light" ]]; then
    CORE_BASE="${CORE_BASE:-coderai-capability-base-light:latest}"
    CORE_DOCKERFILE=packaging/runpod/Dockerfile.capability-base-light
else
    CORE_BASE="${CORE_BASE:-coderai-capability-base:latest}"
    CORE_DOCKERFILE=packaging/runpod/Dockerfile.capability-base
fi
if [[ "${REBUILD_CORE:-0}" == "1" ]] || ! docker image inspect "$CORE_BASE" >/dev/null 2>&1; then
    echo "==> building the shared core $CORE_BASE"
    DOCKER_BUILDKIT=1 docker build -f "$CORE_DOCKERFILE" \
        -t "$CORE_BASE" . || { echo "core build failed" >&2; exit 1; }
else
    echo "==> reusing the shared core $CORE_BASE"
fi

# A profile may name its own Dockerfile (`<profile>.dockerfile`) when the generic
# one cannot express it — the engines image compiles C/CUDA in a toolchain stage.
DOCKERFILE=packaging/runpod/Dockerfile.capability
if [[ -f "$PROFILES_DIR/$PROFILE.dockerfile" ]]; then
    DOCKERFILE="packaging/runpod/$(head -1 "$PROFILES_DIR/$PROFILE.dockerfile")"
fi

# The engine sources live outside the repo (the same trees the local install
# builds from: ~/.coderai/ds4, colibri, kimi-k3-in-c). Stage them into the build
# context without weights, objects or git history, so the image builds the SAME
# code the local engines run, for every pod GPU instead of this one.
if [[ "$PROFILE" == "engines" ]]; then
    SRC=packaging/runpod/engines-src
    rm -rf "$SRC"; mkdir -p "$SRC"
    for pair in "ds4:${CODERAI_DS4_DIR:-$HOME/.coderai/ds4}" \
                "colibri:${CODERAI_COLIBRI_DIR:-$HOME/.coderai/colibri}" \
                "kimi-k3-in-c:${CODERAI_K3_DIR:-$HOME/.coderai/kimi-k3-in-c}"; do
        name="${pair%%:*}"; dir="${pair#*:}"
        if [[ ! -d "$dir" ]]; then
            echo "engine source tree missing: $dir (set CODERAI_$(echo "$name" | tr a-z- A-Z_ | sed 's/KIMI_K3_IN_C/K3/')_DIR)" >&2
            exit 2
        fi
        rsync -a --exclude '.git/' --exclude 'build/' --exclude '*.o' --exclude '*.gguf' \
              --exclude '*.gguf.*' --exclude 'gguf/' --exclude '*.safetensors' --exclude '*.bin' \
              --exclude 'glm52_i4/' --exclude 'web/node_modules/' --exclude 'containers/' \
              --exclude '/ds4flash.gguf' --exclude '/ds4-server' --exclude '/ds4' --exclude '/ds4-agent' \
              --exclude '/ds4-bench' --exclude '/ds4-eval' --exclude '/speed-bench' \
              "$dir/" "$SRC/$name/"
        echo "    staged $name from $dir ($(du -sh "$SRC/$name" | cut -f1))"
    done
fi

echo "==> building $TAG from profile '$PROFILE' ($DOCKERFILE)"
# BuildKit for the pip cache mount: nine profiles, one torch download.
DOCKER_BUILDKIT=1 docker build -f "$DOCKERFILE" \
    --build-arg "PROFILE=$PROFILE" \
    --build-arg "IMAGE_TAG=$TAG" \
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
