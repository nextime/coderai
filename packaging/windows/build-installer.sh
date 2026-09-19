#!/usr/bin/env bash
# Build CoderAI-Setup-<version>.exe with Inno Setup 6 under wine (Linux), or
# fall back to a zip of the scripts when Inno Setup cannot be had.
#   packaging/windows/build-installer.sh [VERSION]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VER="${1:-$(python3 -c "import re;print(re.search(r'__version__ = \"([^\"]+)\"', open('$ROOT/codai/__init__.py').read()).group(1))")}"
OUT="$ROOT/dist"; mkdir -p "$OUT"
ISCC=""
for c in "$HOME/.wine/drive_c/InnoSetup/ISCC.exe" "$HOME/.wine/drive_c/Program Files (x86)/Inno Setup 6/ISCC.exe" "$HOME/.wine/drive_c/Program Files/Inno Setup 6/ISCC.exe"; do
  [ -f "$c" ] && ISCC="$c" && break
done
# wine wants a display even for /VERYSILENT installs and for ISCC; xvfb-run gives it one.
WINE="wine"; command -v xvfb-run >/dev/null 2>&1 && WINE="xvfb-run -a wine"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-$$}"; mkdir -p "$XDG_RUNTIME_DIR"
if [ -z "$ISCC" ] && command -v wine >/dev/null 2>&1 && [ "${INNO_DOWNLOAD:-1}" = 1 ]; then
  echo "== installing Inno Setup 6 under wine (once) =="
  tmp=$(mktemp -d); curl -fsSL -o "$tmp/is.exe" "${INNO_URL:-https://github.com/jrsoftware/issrc/releases/download/is-6_7_3/innosetup-6.7.3.exe}"
  WINEDEBUG=-all $WINE "$tmp/is.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- '/DIR=C:\InnoSetup' >/dev/null 2>&1 || true
  rm -rf "$tmp"
  for c in "$HOME/.wine/drive_c/InnoSetup/ISCC.exe" "$HOME/.wine/drive_c/Program Files (x86)/Inno Setup 6/ISCC.exe"; do
    [ -f "$c" ] && ISCC="$c" && break
  done
fi
if [ -n "$ISCC" ]; then
  echo "== building CoderAI-Setup-$VER.exe =="
  ( cd "$HERE" && WINEDEBUG=-all $WINE "$ISCC" "/DAppVersion=$VER" "/O$(winepath -w "$OUT")" CoderAI.iss )
  ls -la "$OUT"/CoderAI-Setup-"$VER".exe
else
  echo "Inno Setup not available: packaging the scripts as a zip instead" >&2
fi
( cd "$HERE" && zip -q -j "$OUT/coderai-windows-$VER.zip" coderai.ps1 coderai.cmd install-coderai.ps1 README.md CoderAI.iss )
ls -la "$OUT/coderai-windows-$VER.zip"
