#!/usr/bin/env bash
set -euo pipefail

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/packaging/linux/make_tarball_from_venv.sh" "$@"
