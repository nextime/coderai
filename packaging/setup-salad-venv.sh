#!/bin/bash
# Create the isolated venv for the DINOv2-SALAD VPR embedding server
# (codai/api/vpr_salad_server.py). It SHARES the main coderai torch/torchvision
# (--system-site-packages) and adds ONLY the heavy VPR-specific deps
# (pytorch_lightning, pytorch_metric_learning) so they never enter — and cannot
# break — the main engine venv. Idempotent; safe to re-run.
#
# The venv lives on a persistent volume (/cache) so it survives container
# restarts. On a fresh machine/image, run this once; the 'salad'/'dinov2-salad'
# embedding model errors with a clear message until it exists (EigenPlaces VPR
# needs no venv and works regardless).
set -e
MAIN_PY="${CODERAI_PYTHON:-/opt/coderai/python/bin/python3}"
VENV="${SALAD_VENV:-/cache/salad-venv}"

echo "== creating SALAD venv at $VENV (shares $MAIN_PY site-packages) =="
"$MAIN_PY" -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install --no-input --disable-pip-version-check \
    pytorch_lightning pytorch_metric_learning
echo "== salad-venv ready: $VENV =="
"$VENV/bin/python" -c "import pytorch_lightning, pytorch_metric_learning; print('  deps OK')"
