#!/bin/sh
# Pod entrypoint: say what is happening, from the first instant there is a
# container to say it in.
#
# A rented pod is opaque in two separate windows, and they need different fixes.
# Before the image is pulled NOTHING of ours runs, so no amount of application
# logging helps — that window is diagnosed from outside, by polling the pod's
# status. But from container start to the first answered request there were
# minutes of total silence, which looks exactly like a hung pod; a boot that
# failed and a boot that was merely slow produced the same evidence, and the
# RunPod log API answered HTTP 400 when we went looking.
#
# So: timestamped phase lines on stdout (the pod console always has these, even
# when the log API does not), and the same phases written to a file the app
# serves at /boot, so the moment the port opens we can ask the pod where it is
# instead of guessing.
set -u

BOOT_LOG="${CODERAI_BOOT_LOG:-/tmp/coderai-boot.log}"
: > "$BOOT_LOG"

t0=$(date +%s)
phase() {
    now=$(date +%s)
    line="[boot +$((now - t0))s] $*"
    echo "$line"
    echo "$line" >> "$BOOT_LOG"
}

phase "container started"
phase "profile=${PROFILE:-unknown} image=${CODERAI_IMAGE_TAG:-unknown}"
phase "python=$(python -V 2>&1)"

# What the pod was actually told to serve. A pod that answers "not available"
# for its own model is the single most common failure, and the seed list is the
# first place to look — it is worth one line at boot rather than an exec into a
# machine that has since been reaped.
if [ -n "${CODERAI_SEED_MODELS:-}" ]; then
    phase "seed models: $(echo "$CODERAI_SEED_MODELS" | head -c 400)"
else
    phase "seed models: NONE (this pod will only serve what its image ships)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    phase "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
else
    phase "gpu: nvidia-smi not present"
fi

phase "starting uvicorn on 0.0.0.0:8000"
exec python -m uvicorn codai.api.app:app --host 0.0.0.0 --port 8000 \
     --log-level "${CODERAI_LOG_LEVEL:-info}"
