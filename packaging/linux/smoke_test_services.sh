#!/usr/bin/env bash
# Smoke test for the all-in-one CoderAI image: brings the container up and checks
# that nginx + the bundled services answer, and that every external binary/worker
# we rely on is present and runnable. Does NOT load models (no weights needed).
#
# Usage: [DOCKER="sudo docker"] [GPU=--gpus=all] ./smoke_test_services.sh [IMAGE]
set -uo pipefail

DOCKER_BIN="${DOCKER:-docker}"
read -r -a DK <<< "$DOCKER_BIN"
IMAGE="${1:-coderai:dist}"
PORT="${PORT:-18080}"
NAME="coderai-smoke-$$"
GPU="${GPU:-}"
TMP="$(mktemp -d)"
fails=0
note(){ printf '%-52s %s\n' "$1" "$2"; }
ok(){ note "$1" "OK"; }
bad(){ note "$1" "FAIL — $2"; fails=$((fails+1)); }

cleanup(){ "${DK[@]}" rm -f "$NAME" >/dev/null 2>&1 || true; rm -rf "$TMP"; }
trap cleanup EXIT

echo "== starting $IMAGE as $NAME (port $PORT) =="
mkdir -p "$TMP/config" "$TMP/models" "$TMP/cache"
# shellcheck disable=SC2086
"${DK[@]}" run -d --name "$NAME" $GPU --ipc=host \
  --user "$(id -u):$(id -g)" \
  -p "$PORT:8776" \
  -v "$TMP/config:/config" -v "$TMP/models:/models" -v "$TMP/cache:/cache" \
  "$IMAGE" >/dev/null || { echo "container failed to start"; exit 1; }

echo "== waiting for the front to answer =="
up=0
for _ in $(seq 1 60); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" || true)"
  # Any non-5xx HTTP code means the front + coderai are up (the root path itself
  # 404s — the UI lives at /admin); 502/503 means the upstream isn't ready yet.
  case "$code" in 200|301|302|307|401|403|404) up=1; break;; esac
  if ! "${DK[@]}" ps -q --filter "name=$NAME" | grep -q .; then
    echo "container exited early; logs:"; "${DK[@]}" logs "$NAME" 2>&1 | tail -40; exit 1
  fi
  sleep 3
done
[ "$up" = 1 ] && ok "front http://…:$PORT/ responds" || bad "front /" "no response"

echo "== sub-path mounts =="
for p in editor videogen township; do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/$p/" || true)"
  case "$code" in 200|301|302|307) ok "/$p/ ($code)";; *) bad "/$p/" "http $code";; esac
done

echo "== bundled binaries on PATH =="
for b in ffmpeg ffprobe vulkaninfo nginx supervisord whisper-server ds4-server wav2lip sadtalker lspci; do
  if "${DK[@]}" exec "$NAME" sh -lc "command -v $b >/dev/null 2>&1"; then ok "bin: $b"; else bad "bin: $b" "missing"; fi
done

echo "== ds4 seeded on the cache volume =="
if "${DK[@]}" exec "$NAME" sh -lc "test -x /cache/ds4/ds4-server"; then ok "/cache/ds4/ds4-server"; else bad "/cache/ds4/ds4-server" "missing"; fi

echo "== colibri seeded on the cache volume =="
if "${DK[@]}" exec "$NAME" sh -lc "test -x /cache/colibri/c/colibri"; then ok "/cache/colibri/c/colibri"; else bad "/cache/colibri/c/colibri" "missing (optional — bundle with build.sh --colibri)"; fi

echo "== shared lip-sync venv (py3.10 + torch) =="
if "${DK[@]}" exec "$NAME" /opt/coderai/lipsync_venv/bin/python -c "import torch,sys; print(sys.version.split()[0], torch.__version__)" >/dev/null 2>&1; then
  ok "lipsync venv imports torch"
else
  bad "lipsync venv" "python/torch import failed"
fi
# Repo code is bundled; weights are NOT (download on first lip-sync use).
if "${DK[@]}" exec "$NAME" sh -lc "test -f /opt/coderai/Wav2Lip/inference.py && test -f /opt/coderai/SadTalker/inference.py"; then
  ok "lip-sync repo code present"
else
  bad "lip-sync repo code" "missing"
fi

echo "== parler overlay present =="
if "${DK[@]}" exec "$NAME" sh -lc "test -d /opt/coderai/parler-venv/site-packages"; then ok "parler overlay"; else bad "parler overlay" "missing"; fi

echo
if [ "$fails" = 0 ]; then echo "SMOKE TEST PASSED"; else echo "SMOKE TEST: $fails failure(s)"; "${DK[@]}" logs "$NAME" 2>&1 | tail -30; fi
exit "$fails"
