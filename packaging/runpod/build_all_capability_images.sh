#!/usr/bin/env bash
# Build every capability image, then optionally push them.
#
#   ./packaging/runpod/build_all_capability_images.sh ghcr.io/nextime
#   PUSH=1 ./packaging/runpod/build_all_capability_images.sh ghcr.io/nextime
#
# Builds are sequential on purpose: they share a pip cache and a core layer, so
# running them in parallel would race on both and download torch several times.
# Each image is smoke-tested (it must answer /healthz) before the next starts.
set -uo pipefail

cd "$(dirname "$0")/../.."
NS="${1:-}"
VERSION="${VERSION:-$(python3 -c 'import re,pathlib; print(re.search(r"__version__ = \"([^\"]+)\"", pathlib.Path("codai/__init__.py").read_text()).group(1))')}"
PROFILES="${PROFILES:-$(ls packaging/runpod/profiles/*.txt | xargs -n1 basename | sed 's/\.txt$//' | grep -v '^core$' | tr '\n' ' ')}"

if [[ -z "$NS" ]]; then
    echo "usage: $0 <registry/namespace>   e.g. ghcr.io/nextime" >&2
    exit 2
fi

echo "== building ${VERSION} images for: ${PROFILES}"
ok=(); failed=()
for p in $PROFILES; do
    tag="${NS}/coderai-${p}:${VERSION}"
    echo
    echo "================ $p -> $tag ================"
    if PUSH=0 ./packaging/runpod/build_capability_image.sh "$p" "$tag"; then
        docker tag "$tag" "${NS}/coderai-${p}:latest"
        ok+=("$tag")
    else
        failed+=("$p")
    fi
done

echo
echo "== built: ${#ok[@]}   failed: ${#failed[@]} ${failed[*]:-}"
for t in "${ok[@]}"; do
    printf '   %-52s %s MB\n' "$t" "$(( $(docker image inspect "$t" --format '{{.Size}}') / 1000000 ))"
done

if [[ "${PUSH:-0}" == "1" && ${#ok[@]} -gt 0 ]]; then
    echo
    echo "== pushing ${#ok[@]} images to ${NS}"
    for t in "${ok[@]}"; do
        docker push "$t" || failed+=("push:$t")
        docker push "${t%:*}:latest" || failed+=("push:${t%:*}:latest")
    done
fi

[[ ${#failed[@]} -eq 0 ]] || { echo "FAILURES: ${failed[*]}" >&2; exit 1; }
