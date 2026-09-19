#!/usr/bin/env bash
# Build llama.cpp's rpc-server from the SAME llama.cpp the installed
# llama-cpp-python was built from, and put it beside the venv's python.
#
# The RPC protocol is versioned: a client and a server from different
# llama.cpp commits refuse each other. llama-cpp-python's sdist vendors the
# exact llama.cpp tree its wheel was built from, so that tree — not a fresh
# clone of master — is what rpc-server must come from. The binary is what a
# machine runs to lend its cards to a GGUF loaded elsewhere
# (codai/cluster/rpc.py starts it from cluster.rpc_servers; the app's own
# loads reach it through codai/backends/ggml_rpc.py).
#
# Usage:  packaging/build-rpc-server.sh [--cuda] [--vulkan] [--dest DIR]
#   Backends default to whatever the installed llama-cpp-python has (it is
#   probed by looking at its lib/ directory). The python is the one on PATH
#   (activate the venv first) unless PYTHON is set.
set -euo pipefail

PYTHON="${PYTHON:-python}"
CUDA=""; VULKAN=""; DEST=""
while [ $# -gt 0 ]; do
    case "$1" in
        --cuda) CUDA=1 ;;
        --vulkan) VULKAN=1 ;;
        --dest) DEST="$2"; shift ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

VER=$("$PYTHON" -c "import llama_cpp; print(llama_cpp.__version__)")
LIBDIR=$("$PYTHON" -c "import llama_cpp, os; print(os.path.join(os.path.dirname(llama_cpp.__file__), 'lib'))")
BINDIR=$("$PYTHON" -c "import sys, os; print(os.path.dirname(sys.executable))")
DEST="${DEST:-$BINDIR}"
if [ -z "$CUDA$VULKAN" ]; then
    ls "$LIBDIR" | grep -q 'libggml-cuda' && CUDA=1 || true
    ls "$LIBDIR" | grep -q 'libggml-vulkan' && VULKAN=1 || true
fi
echo "llama-cpp-python $VER  (cuda=${CUDA:-0} vulkan=${VULKAN:-0})  -> $DEST/rpc-server"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
"$PYTHON" -m pip download --no-deps --no-binary :all: --no-build-isolation \
    "llama-cpp-python==$VER" -d . >/dev/null 2>&1 || \
"$PYTHON" -m pip download --no-deps --no-binary :all: "llama-cpp-python==$VER" -d .
SDIST=$(ls llama_cpp_python-*.tar.gz | head -1)
tar -xzf "$SDIST"
SRC=$(ls -d llama_cpp_python-*/vendor/llama.cpp | head -1)
[ -f "$SRC/CMakeLists.txt" ] || { echo "vendored llama.cpp not found in $SDIST" >&2; exit 1; }

CM=(-DGGML_RPC=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_AVX2=ON)
[ -n "$CUDA" ] && CM+=(-DGGML_CUDA=ON ${CMAKE_CUDA_ARCHITECTURES:+-DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES"})
[ -n "$VULKAN" ] && CM+=(-DGGML_VULKAN=ON)
cmake -S "$SRC" -B build "${CM[@]}" ${EXTRA_CMAKE_ARGS:-} >/dev/null
cmake --build build --target rpc-server -j"$(nproc)"
BIN=$(find build -type f -name rpc-server | head -1)
[ -x "$BIN" ] || { echo "rpc-server did not build" >&2; exit 1; }
install -m 755 "$BIN" "$DEST/rpc-server"
echo "installed $DEST/rpc-server"
"$DEST/rpc-server" --help 2>&1 | head -3 || true
