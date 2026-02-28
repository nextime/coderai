#!/bin/bash
# Build script for CoderAI - Supports NVIDIA (CUDA) and Vulkan (AMD GPUs) backends
# Usage: ./build.sh [nvidia|vulkan]
# Default: nvidia

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Determine backend
BACKEND="${1:-nvidia}"
BACKEND=$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')

if [[ "$BACKEND" != "nvidia" && "$BACKEND" != "vulkan" ]]; then
    echo -e "${RED}Error: Invalid backend '$BACKEND'${NC}"
    echo "Usage: ./build.sh [nvidia|vulkan]"
    echo "  nvidia  - Use PyTorch with CUDA for NVIDIA GPUs"
    echo "  vulkan  - Use llama-cpp-python with Vulkan for AMD GPUs"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CoderAI Build Script${NC}"
echo -e "${BLUE}  Backend: ${GREEN}$BACKEND${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python 3.8+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"

# Create virtual environment if it doesn't exist
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

echo ""
echo -e "${BLUE}Installing dependencies for $BACKEND backend...${NC}"
echo ""

if [ "$BACKEND" = "nvidia" ]; then
    # NVIDIA/CUDA backend
    echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
    pip install "torch>=2.0.0" "torchvision>=0.15.0" "torchaudio>=2.0.0"
    
    echo -e "${YELLOW}Installing NVIDIA-specific requirements...${NC}"
    pip install -r requirements-nvidia.txt
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  NVIDIA/CUDA build complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Usage:"
    echo "  source venv/bin/activate"
    echo "  python coderai --model <huggingface-model-name>"
    echo ""
    echo "Example:"
    echo "  python coderai --model microsoft/DialoGPT-medium"
    echo ""
    
elif [ "$BACKEND" = "vulkan" ]; then
    # Vulkan backend
    echo -e "${YELLOW}Installing llama-cpp-python with Vulkan support...${NC}"
    
    # Check for required Vulkan development libraries
    if ! pkg-config --exists vulkan 2>/dev/null; then
        echo -e "${YELLOW}Warning: Vulkan development libraries not found via pkg-config${NC}"
        echo -e "${YELLOW}You may need to install Vulkan drivers and SDK:${NC}"
        echo "  Debian/Ubuntu: sudo apt install libvulkan-dev vulkan-tools"
        echo "  Fedora: sudo dnf install vulkan-loader-devel vulkan-tools"
        echo "  Arch: sudo pacman -S vulkan-headers vulkan-icd-loader"
        echo ""
        echo -e "${YELLOW}Attempting installation anyway...${NC}"
    fi
    
    # Check for glslc (Vulkan shader compiler) which is required for building
    if ! command -v glslc &> /dev/null; then
        echo -e "${YELLOW}Warning: glslc (Vulkan shader compiler) not found${NC}"
        echo -e "${YELLOW}You need to install the shader compiler:${NC}"
        echo "  Debian/Ubuntu: sudo apt install glslang-tools"
        echo "  Fedora:        sudo dnf install glslang"
        echo "  Arch:          sudo pacman -S glslang"
        echo ""
        echo -e "${RED}Error: Cannot build llama-cpp-python with Vulkan without glslc${NC}"
        exit 1
    fi
    
    # Install llama-cpp-python with Vulkan support
    # CMAKE_ARGS is used to enable Vulkan during compilation
    echo -e "${YELLOW}Building llama-cpp-python with Vulkan support (this may take a few minutes)...${NC}"
    CMAKE_ARGS="-DGGML_VULKAN=ON" pip install llama-cpp-python --no-cache-dir
    
    echo -e "${YELLOW}Installing Vulkan-specific requirements...${NC}"
    pip install -r requirements-vulkan.txt
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Vulkan build complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Usage:"
    echo "  source venv/bin/activate"
    echo "  python coderai --model <path-to-gguf-model> --backend vulkan"
    echo ""
    echo "Example:"
    echo "  python coderai --model ./phi-3-mini-4k-instruct-q4_k_m.gguf --backend vulkan"
    echo ""
    echo "Note: For Vulkan, you need to use GGUF format models."
    echo "      Download from: https://huggingface.co/models?search=gguf"
    echo ""
fi

# Create .backend file to track which backend was used
echo "$BACKEND" > .backend

echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
