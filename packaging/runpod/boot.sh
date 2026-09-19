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
# NOT set -u, and NOT set -e. Every line before the exec is a diagnostic: it
# exists to tell us what the pod is, and a diagnostic must never be able to
# stop the boot it describes. A speaker pod died with EXITED at uptime 0s on
# three machines in a row — the image booted fine locally — because an
# environment difference tripped a strict-mode abort before uvicorn was ever
# reached. The one line that matters is the last one.

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
phase "user=$(id -un 2>/dev/null || echo unknown) cwd=$(pwd 2>/dev/null || echo unknown)"
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
    phase "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo 'nvidia-smi query failed')"
else
    phase "gpu: nvidia-smi not present"
fi

# TLS, when the renting coderai sent a certificate: it reaches this pod at a
# public ip:port with nothing in between (direct_tcp), so the pod terminates
# TLS itself. The cert is signed by that install's own CA and only it
# verifies it. PEM in env → files, because uvicorn wants paths. Absent env
# means the RunPod proxy is in front, which speaks plain HTTP to us.
TLS_ARGS=""
if [ -n "$CODERAI_TLS_CERT" ] && [ -n "$CODERAI_TLS_KEY" ]; then
    mkdir -p /run/coderai-tls && chmod 700 /run/coderai-tls
    printf '%s\n' "$CODERAI_TLS_CERT" > /run/coderai-tls/cert.pem
    printf '%s\n' "$CODERAI_TLS_KEY" > /run/coderai-tls/key.pem
    chmod 600 /run/coderai-tls/key.pem
    unset CODERAI_TLS_KEY
    TLS_ARGS="--ssl-certfile /run/coderai-tls/cert.pem --ssl-keyfile /run/coderai-tls/key.pem"
    phase "tls: serving https with the certificate the renting coderai sent"
fi

# A ray WORKER instead of a server: the vllm image joins a multi-node vLLM
# launch whose head is the coderai that started this container
# (codai/cluster/multinode.py). Nothing else runs; the head drives the GPUs.
if [ -n "${CODERAI_RAY_ADDRESS:-}" ]; then
    RAY_BIN="${CODERAI_VLLM_VENV:-/opt/coderai/venvs/vllm}/bin/ray"
    [ -x "$RAY_BIN" ] || RAY_BIN="ray"
    phase "ray worker: joining ${CODERAI_RAY_ADDRESS} with ${RAY_BIN}"
    exec "$RAY_BIN" start --address="${CODERAI_RAY_ADDRESS}" --block \
         ${CODERAI_RAY_NODE_IP:+--node-ip-address="$CODERAI_RAY_NODE_IP"} \
         --disable-usage-stats
fi

# A llama.cpp rpc-server beside the app: this pod's/host's cards become
# devices of a GGUF loaded elsewhere (codai/backends/ggml_rpc.py). The port
# must be exposed (direct TCP on a pod, -p on docker). Unauthenticated
# protocol — LAN, WireGuard or a pod's direct port with a firewall only.
if [ -n "${CODERAI_RPC_SERVER_PORT:-}" ]; then
    RPC_BIN="${CODERAI_RPC_SERVER_BIN:-/usr/local/bin/rpc-server}"
    if [ -x "$RPC_BIN" ]; then
        phase "rpc-server: listening on 0.0.0.0:${CODERAI_RPC_SERVER_PORT}${CODERAI_RPC_SERVER_DEVICE:+ device $CODERAI_RPC_SERVER_DEVICE}"
        "$RPC_BIN" -H 0.0.0.0 -p "$CODERAI_RPC_SERVER_PORT" \
            ${CODERAI_RPC_SERVER_DEVICE:+-d "$CODERAI_RPC_SERVER_DEVICE"} \
            ${CODERAI_RPC_SERVER_MEM_MB:+-m "$CODERAI_RPC_SERVER_MEM_MB"} &
        if [ "${CODERAI_RPC_SERVER_ONLY:-0}" = "1" ]; then
            phase "rpc-server only: no API server on this pod"
            wait
            exit $?
        fi
    else
        phase "rpc-server requested but $RPC_BIN is not in this image"
    fi
fi

phase "starting uvicorn on 0.0.0.0:8000${TLS_ARGS:+ (https)}"
exec python -m uvicorn codai.api.app:app --host 0.0.0.0 --port 8000 \
     --log-level "${CODERAI_LOG_LEVEL:-info}" $TLS_ARGS
