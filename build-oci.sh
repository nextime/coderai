#!/usr/bin/env bash
set -euo pipefail

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/packaging/linux/build_oci_image.sh" "$@"
