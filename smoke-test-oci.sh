#!/usr/bin/env bash
set -euo pipefail

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/packaging/linux/smoke_test_oci.sh" "$@"
