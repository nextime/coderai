#!/usr/bin/env bash
# Flatten the shipped image into a single layer and re-tag it as the new base.
#
# update_oci_image.sh adds a thin app layer on top of coderai:base. Doing that
# repeatedly is fine (layers never stack — every update starts from the same
# base), but the SHIPPED image is nicer as one squashed layer: smaller to export,
# faster to load, and a clean starting point for the next update.
#
# `docker export | docker import` drops the image config entirely, so this script
# re-applies it from the source image and then VERIFIES the result matches. That
# check is the point: a silently lost ENTRYPOINT or PATH produces an image that
# builds, ships, and then fails to start.
#
#   ./packaging/linux/flatten_oci_image.sh                 # coderai:dist -> coderai:base
#   SRC=coderai:dist DEST=coderai:base ./flatten_oci_image.sh
#   RETAG=0 ./flatten_oci_image.sh                         # build the flat tag, don't move base
set -euo pipefail

DOCKER_BIN="${DOCKER:-docker}"
read -r -a DK <<< "$DOCKER_BIN"
SRC="${SRC:-coderai:dist}"
DEST="${DEST:-coderai:base}"
STAMP="${STAMP:-$(date +%Y%m%d%H%M)}"
FLAT_TAG="${FLAT_TAG:-coderai:flat-${STAMP}}"
RETAG="${RETAG:-1}"
# How many previous bases to keep as `<dest>-pre-<stamp>` rollback tags. Each one
# pins a whole ~26 GB image, so they pile up fast; 1 keeps a rollback without
# hoarding every build. 0 keeps none (nothing to roll back to but git).
KEEP_BACKUPS="${KEEP_BACKUPS:-1}"
# The intermediate `flat-<stamp>` tag is what DEST points at; it is the same
# image, so keeping it around just makes `docker image ls` noisy. 0 drops it.
KEEP_FLAT_TAG="${KEEP_FLAT_TAG:-0}"
# Drop secret-valued env vars from the flattened config. A token baked into an
# image is readable by anyone who can pull it — `docker image inspect` is enough,
# no need to run anything. These arrive by `docker commit` of a container that was
# started with -e, and every flatten since has carried them forward. run_oci.sh
# passes HF_TOKEN at runtime, so removing it from the image changes nothing.
STRIP_SECRET_ENV="${STRIP_SECRET_ENV:-1}"
SECRET_ENV_RE="${SECRET_ENV_RE:-(TOKEN|KEY|SECRET|PASSWORD|PASSWD)}"

"${DK[@]}" image inspect "$SRC" >/dev/null

echo "== flattening $SRC -> $FLAT_TAG =="
cid="$("${DK[@]}" create "$SRC")"
cleanup(){ "${DK[@]}" rm -f "$cid" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# Rebuild the config as `docker import` change directives. Everything is emitted
# in JSON form so values containing spaces (CODERAI_EXTRA_ARGS) survive intact —
# the one thing that silently breaks when this is done by hand.
mapfile -t CHANGES < <("${DK[@]}" image inspect "$SRC" --format '{{json .Config}}' \
    | STRIP_SECRET_ENV="$STRIP_SECRET_ENV" SECRET_ENV_RE="$SECRET_ENV_RE" python3 -c '
import json, os, re, sys
c = json.load(sys.stdin)
strip = os.environ.get("STRIP_SECRET_ENV") == "1"
pat = re.compile(os.environ.get("SECRET_ENV_RE", "(TOKEN|KEY|SECRET|PASSWORD)"), re.I)
dropped = []
out = []
for kv in c.get("Env") or []:
    k, _, v = kv.partition("=")
    if strip and v and pat.search(k):
        dropped.append(k)
        continue
    out.append("ENV " + k + "=" + json.dumps(v))
if dropped:
    print("DROPPED:" + ",".join(dropped), file=sys.stderr)
for field, word in (("Entrypoint", "ENTRYPOINT"), ("Cmd", "CMD")):
    val = c.get(field)
    if val:
        out.append(word + " " + json.dumps(val))
if c.get("WorkingDir"):
    out.append("WORKDIR " + c["WorkingDir"])
if c.get("User"):
    out.append("USER " + c["User"])
for port in (c.get("ExposedPorts") or {}):
    out.append("EXPOSE " + port.split("/")[0])
for vol in (c.get("Volumes") or {}):
    out.append("VOLUME " + json.dumps([vol]))
for k, v in (c.get("Labels") or {}).items():
    out.append("LABEL " + k + "=" + json.dumps(v))
print("\n".join(out))
')

args=()
for ch in "${CHANGES[@]}"; do args+=(-c "$ch"); done
echo "   re-applying ${#CHANGES[@]} config directives"

# Report what was stripped, by NAME only — never echo the value being removed.
mapfile -t DROPPED < <("${DK[@]}" image inspect "$SRC" --format '{{json .Config.Env}}' \
    | STRIP_SECRET_ENV="$STRIP_SECRET_ENV" SECRET_ENV_RE="$SECRET_ENV_RE" python3 -c '
import json, os, re, sys
if os.environ.get("STRIP_SECRET_ENV") != "1":
    raise SystemExit
pat = re.compile(os.environ.get("SECRET_ENV_RE", "(TOKEN|KEY|SECRET|PASSWORD)"), re.I)
for kv in json.load(sys.stdin) or []:
    k, _, v = kv.partition("=")
    if v and pat.search(k):
        print(k)
')
if [[ ${#DROPPED[@]} -gt 0 ]]; then
    echo "   NOTE: stripping baked-in secrets from the image config: ${DROPPED[*]}"
    echo "         (supply them at runtime with -e; run_oci.sh already does)"
fi

"${DK[@]}" export "$cid" | "${DK[@]}" import "${args[@]}" - "$FLAT_TAG"
cleanup
trap - EXIT

# Verify: the flattened image must be configured exactly like its source. An
# image that lost its ENTRYPOINT still builds and ships — and then won't start.
echo "== verifying config equality =="
if ! DROPPED_ENV="${DROPPED[*]:-}" python3 - "$SRC" "$FLAT_TAG" <<'PY'
import json, os, subprocess, sys

DROPPED = set((os.environ.get("DROPPED_ENV") or "").split())


def cfg(tag):
    raw = subprocess.check_output(["docker", "image", "inspect", tag,
                                   "--format", "{{json .Config}}"])
    c = json.loads(raw)
    return {
        # Deliberately stripped secrets are not a mismatch.
        "Env": sorted(e for e in (c.get("Env") or [])
                      if e.partition("=")[0] not in DROPPED),
        "Entrypoint": c.get("Entrypoint"),
        "Cmd": c.get("Cmd"),
        "WorkingDir": c.get("WorkingDir") or "",
        "User": c.get("User") or "",
        "ExposedPorts": sorted((c.get("ExposedPorts") or {}).keys()),
        "Volumes": sorted((c.get("Volumes") or {}).keys()),
        "Labels": c.get("Labels") or {},
    }

a, b = cfg(sys.argv[1]), cfg(sys.argv[2])
bad = [k for k in a if a[k] != b[k]]
if bad:
    for k in bad:
        print(f"   MISMATCH {k}:\n     src : {a[k]}\n     flat: {b[k]}")
    sys.exit(1)
print(f"   config matches ({len(a['Env'])} env vars, entrypoint + cmd + ports preserved)")
PY
then
    echo "!!! flattened image config does not match $SRC — NOT retagging $DEST" >&2
    exit 1
fi

if [[ "$RETAG" == "1" ]]; then
    repo="${DEST%%:*}"
    tag="${DEST#*:}"
    if "${DK[@]}" image inspect "$DEST" >/dev/null 2>&1 && [[ "$KEEP_BACKUPS" != "0" ]]; then
        backup="${repo}:${tag}-pre-${STAMP}"
        "${DK[@]}" tag "$DEST" "$backup"
        echo "== previous $DEST backed up as $backup =="
    fi
    "${DK[@]}" tag "$FLAT_TAG" "$DEST"
    echo "== $DEST now points at $("${DK[@]}" image inspect "$DEST" --format '{{.Id}}' | cut -c8-19) =="

    # Retire older rollback tags. Every one of these pins a whole ~26 GB image,
    # and a handful of builds is a quarter-terabyte of disk that nothing reads.
    # Sorted by name, which is chronological because the stamp is a timestamp.
    mapfile -t old_backups < <("${DK[@]}" image ls "$repo" --format '{{.Tag}}' \
        | grep -E "^${tag}-pre-" | sort -r | tail -n +$(( KEEP_BACKUPS + 1 )))
    for b in "${old_backups[@]:-}"; do
        [[ -n "$b" ]] || continue
        echo "   retiring old backup ${repo}:${b}"
        "${DK[@]}" rmi "${repo}:${b}" >/dev/null 2>&1 || true
    done

    if [[ "$KEEP_FLAT_TAG" == "0" && "$FLAT_TAG" != "$DEST" ]]; then
        # Same image as DEST — dropping the tag frees no space, just noise.
        "${DK[@]}" rmi "$FLAT_TAG" >/dev/null 2>&1 || true
    fi
fi

"${DK[@]}" image ls "${DEST%%:*}" --format '{{.Repository}}:{{.Tag}}  {{.Size}}  {{.ID}}' | head -5
