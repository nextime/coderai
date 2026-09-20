#!/usr/bin/env bash
# Sign (or verify) the published CoderAI images with cosign.
#
#   ./packaging/sign-images.sh                 # sign coderai:<version> + :latest
#   ./packaging/sign-images.sh all             # + the 19 capability images
#   ./packaging/sign-images.sh verify [ref…]   # check signatures (anyone can)
#   IMAGES="ghcr.io/nextime/coderai:0.2.20" ./packaging/sign-images.sh
#
# Key-pair signing (works headless; keyless Sigstore needs a browser login):
# the private key lives OUTSIDE the repo at $COSIGN_KEY
# (default ~/.coderai-signing/cosign.key, generated on first use, encrypted
# with $COSIGN_PASSWORD — blank = an unencrypted key file, mode 600); the
# public key is committed as packaging/cosign.pub and is what users verify
# against. Signatures are attached to the image DIGEST in the registry
# (an OCI artifact next to the image), so a re-pushed tag with a different
# digest is unsigned until this script runs again.
#
# Verify (no account needed):
#   cosign verify --key https://raw.githubusercontent.com/nextime/coderai/master/packaging/cosign.pub \
#          ghcr.io/nextime/coderai:latest
set -euo pipefail
cd "$(dirname "$0")/.."

NS="${NS:-ghcr.io/nextime}"
VERSION="${VERSION:-$(python3 -c 'import re,pathlib; print(re.search(r"__version__ = \"([^\"]+)\"", pathlib.Path("codai/__init__.py").read_text()).group(1))')}"
COSIGN_KEY="${COSIGN_KEY:-$HOME/.coderai-signing/cosign.key}"
PUB="packaging/cosign.pub"
export COSIGN_PASSWORD="${COSIGN_PASSWORD:-}"
export COSIGN_YES=true            # no interactive prompts

cosign_bin(){
    if command -v cosign >/dev/null 2>&1; then echo cosign; return; fi
    if [[ -x "$HOME/.local/bin/cosign" ]]; then echo "$HOME/.local/bin/cosign"; return; fi
    echo "installing cosign into ~/.local/bin" >&2
    mkdir -p "$HOME/.local/bin"
    curl -fsSL -o "$HOME/.local/bin/cosign" \
        "https://github.com/sigstore/cosign/releases/download/v3.1.3/cosign-linux-amd64"
    chmod +x "$HOME/.local/bin/cosign"
    echo "$HOME/.local/bin/cosign"
}
COSIGN="$(cosign_bin)"

capability_images(){
    ls packaging/runpod/profiles/*.txt | xargs -n1 basename | sed 's/\.txt$//' \
        | grep -v '^core' | grep -v '\.venv-' | sed "s#^#${NS}/coderai-#; s#\$#:latest#"
}

mode="${1:-sign}"
case "$mode" in
  verify)
    shift || true
    refs=("$@")
    [[ ${#refs[@]} -eq 0 ]] && refs=("${NS}/coderai:latest" "${NS}/coderai:${VERSION}")
    rc=0
    for ref in "${refs[@]}"; do
        if "$COSIGN" verify --key "$PUB" "$ref" >/dev/null 2>&1; then
            echo "OK       $ref"
        else
            echo "UNSIGNED $ref"; rc=1
        fi
    done
    exit $rc
    ;;
  sign|all) ;;
  *) echo "usage: $0 [sign|all|verify [ref…]]" >&2; exit 2 ;;
esac

# --- key ---------------------------------------------------------------
if [[ ! -f "$COSIGN_KEY" ]]; then
    mkdir -p "$(dirname "$COSIGN_KEY")"; chmod 700 "$(dirname "$COSIGN_KEY")"
    echo "generating a signing key pair at $COSIGN_KEY"
    ( cd "$(dirname "$COSIGN_KEY")" && "$COSIGN" generate-key-pair --output-key-prefix "$(basename "${COSIGN_KEY%.key}")" )
    chmod 600 "$COSIGN_KEY"
fi
pubfile="${COSIGN_KEY%.key}.pub"
if [[ -f "$pubfile" ]] && ! cmp -s "$pubfile" "$PUB"; then
    cp "$pubfile" "$PUB"
    echo "public key written to $PUB — commit it"
fi

# --- images ------------------------------------------------------------
if [[ -n "${IMAGES:-}" ]]; then
    mapfile -t images < <(tr ' ' '\n' <<<"$IMAGES" | sed '/^$/d')
else
    images=("${NS}/coderai:${VERSION}" "${NS}/coderai:latest")
    if [[ "$mode" == "all" ]]; then
        mapfile -t caps < <(capability_images)
        images+=("${caps[@]}")
    fi
fi

# Sign by digest: a tag can move, a digest is the image.
for ref in "${images[@]}"; do
    digest="$(docker buildx imagetools inspect "$ref" --format '{{json .Manifest.Digest}}' 2>/dev/null | tr -d '"' || true)"
    if [[ -z "$digest" ]]; then
        digest="$(docker manifest inspect -v "$ref" 2>/dev/null | python3 -c 'import sys,json
d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print(d.get("Descriptor",{}).get("digest",""))' || true)"
    fi
    if [[ -z "$digest" ]]; then
        echo "SKIP  $ref (not in the registry)"; continue
    fi
    name="${ref%%:*}"; name="${name%@*}"
    if "$COSIGN" verify --key "$PUB" "${name}@${digest}" >/dev/null 2>&1; then
        echo "OK    $ref already signed (${digest:7:12})"; continue
    fi
    echo "SIGN  $ref  ${digest:7:12}"
    "$COSIGN" sign --key "$COSIGN_KEY" \
        -a "version=${VERSION}" -a "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
        "${name}@${digest}"
done
echo "done. verify with: cosign verify --key $PUB <image>"
