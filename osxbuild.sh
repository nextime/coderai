#!/bin/bash
# CoderAI macOS build script
# Usage: ./osxbuild.sh [metal|cpu|all] [--venv <venv>] [--package]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BACKEND="${1:-all}"
CUSTOM_VENV=""
PACKAGE=false

i=1
for arg in "$@"; do
    case $arg in
        --venv)
            i=$((i + 1))
            eval "CUSTOM_VENV=\${$i}"
            ;;
        --package)
            PACKAGE=true
            ;;
    esac
    i=$((i + 1))
done

BACKEND=$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')
if [[ "$BACKEND" != "metal" && "$BACKEND" != "cpu" && "$BACKEND" != "all" ]]; then
    echo -e "${RED}Error: Invalid backend '$BACKEND'${NC}"
    echo "Usage: ./osxbuild.sh [metal|cpu|all] [--venv <venv>]"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CoderAI macOS Build Script${NC}"
echo -e "${BLUE}  Backend: ${GREEN}$BACKEND${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_VERSION=$($PYTHON_BIN --version 2>&1 | python3 -c 'import sys,re; print(re.search(r"(\d+\.\d+)", sys.stdin.read()).group(1))')
echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"

if [ -n "$CUSTOM_VENV" ]; then
    VENV_DIR="$CUSTOM_VENV"
elif [ "$BACKEND" = "metal" ]; then
    VENV_DIR="venv_osx_metal"
elif [ "$BACKEND" = "cpu" ]; then
    VENV_DIR="venv_osx_cpu"
else
    VENV_DIR="venv_osx_all"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment: $VENV_DIR${NC}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo -e "${YELLOW}Using existing virtual environment: $VENV_DIR${NC}"
fi

source "$VENV_DIR/bin/activate"
export PIP_NO_INPUT=1
export PIP_REQUIRE_VIRTUALENV=1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

install_common_ml_stack() {
    python -m pip install "imageio[ffmpeg]" scipy soundfile sentence-transformers openai-whisper argostranslate edge-tts kokoro-tts timm || true
    python -m pip install realesrgan basicsr || true
    python -m pip install demucs deepfilternet rnnoise voicefixer || true
    python -m pip install insightface onnxruntime-silicon || python -m pip install insightface onnxruntime || true
    python -m pip install f5-tts seed-vc || true
    python -m pip install audiocraft || true
}

install_metal_stack() {
    echo -e "${YELLOW}Installing PyTorch with Apple Silicon / MPS support...${NC}"
    python -m pip install torch torchvision torchaudio

    echo -e "${YELLOW}Installing llama-cpp-python with Metal support...${NC}"
    CMAKE_ARGS="-DGGML_METAL=ON" python -m pip install --upgrade llama-cpp-python --no-cache-dir || {
        echo -e "${YELLOW}Warning: Metal build failed, installing CPU llama-cpp-python${NC}"
        python -m pip install --upgrade llama-cpp-python
    }

    echo -e "${YELLOW}Installing stable-diffusion-cpp-python with Metal support...${NC}"
    CMAKE_ARGS="-DSD_METAL=ON -DSD_WEBM=OFF" python -m pip install stable-diffusion-cpp-python --no-cache-dir || {
        echo -e "${YELLOW}Warning: Metal stable-diffusion-cpp-python not available${NC}"
    }

    echo -e "${YELLOW}Installing whispercpp with Metal support when possible...${NC}"
    python -m pip uninstall -y whispercpp >/dev/null 2>&1 || true
    TMP_DIR="${TMPDIR:-/tmp}/coderai-whispercpp"
    rm -rf "$TMP_DIR"
    git clone --depth 1 https://github.com/ggerganov/whisper.cpp "$TMP_DIR" >/dev/null 2>&1 || true
    if [ -d "$TMP_DIR/bindings/python" ]; then
        (cd "$TMP_DIR/bindings/python" && CMAKE_ARGS="-DWHISPER_METAL=ON -DGGML_METAL=ON" python -m pip install . --no-cache-dir --force-reinstall) || \
        (cd "$TMP_DIR/bindings/python" && python -m pip install . --no-cache-dir --force-reinstall) || true
    fi
}

install_cpu_stack() {
    echo -e "${YELLOW}Installing CPU-only runtime...${NC}"
    python -m pip install torch torchvision torchaudio
    python -m pip install --upgrade llama-cpp-python
    python -m pip install stable-diffusion-cpp-python || true
}

package_app() {
    echo -e "${YELLOW}Packaging CoderAI with PyInstaller...${NC}"
    python -m pip install pyinstaller
    mkdir -p dist-package
    pyinstaller --clean --noconfirm --onefile --name coderai \
        --collect-all codai \
        --collect-all fastapi \
        --collect-all uvicorn \
        --collect-all pydantic \
        --collect-all transformers \
        --collect-all diffusers \
        --collect-all sentence_transformers \
        --collect-all whispercpp \
        --collect-all insightface \
        --collect-all onnxruntime \
        --collect-all PIL \
        coderai
    pyinstaller --clean --noconfirm --windowed --name CoderAI \
        --collect-all codai \
        --collect-all fastapi \
        --collect-all uvicorn \
        --collect-all pydantic \
        --collect-all transformers \
        --collect-all diffusers \
        --collect-all sentence_transformers \
        --collect-all whispercpp \
        --collect-all insightface \
        --collect-all onnxruntime \
        --collect-all PIL \
        coderai
    cp dist/coderai dist-package/coderai
    if [ -d "dist/CoderAI.app" ]; then
        rm -rf dist-package/CoderAI.app
        cp -R dist/CoderAI.app dist-package/CoderAI.app
    fi
    echo -e "${GREEN}✓ Packaged CLI binary: dist-package/coderai${NC}"
    echo -e "${GREEN}✓ Packaged macOS app bundle: dist-package/CoderAI.app${NC}"
    echo -e "${YELLOW}Note: macOS equivalent packaging is a single CLI binary plus a .app bundle; target machines still need compatible GPU/runtime libraries.${NC}"
}

if [ "$BACKEND" = "metal" ]; then
    install_metal_stack
    install_common_ml_stack
elif [ "$BACKEND" = "cpu" ]; then
    install_cpu_stack
    install_common_ml_stack
else
    install_metal_stack
    install_common_ml_stack
fi

echo "$BACKEND" > .backend
if [ "$PACKAGE" = true ]; then
    package_app
fi
echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo "To activate the environment in the future, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Recommended runtime notes:"
echo "  - macOS uses Metal (MPS / GGML_METAL / SD_METAL) instead of CUDA"
echo "  - NVIDIA CUDA is not the standard macOS path"
