#!/bin/bash
# Build script for CoderAI - Supports NVIDIA (CUDA) and Vulkan backends
# Usage: ./build.sh [nvidia|vulkan|vulkan-nvidia]
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

if [[ "$BACKEND" != "nvidia" && "$BACKEND" != "vulkan" && "$BACKEND" != "vulkan-nvidia" && "$BACKEND" != "cuda" ]]; then
    echo -e "${RED}Error: Invalid backend '$BACKEND'${NC}"
    echo "Usage: ./build.sh [nvidia|vulkan|vulkan-nvidia|cuda]"
    echo "  nvidia       - Use PyTorch with CUDA for NVIDIA GPUs"
    echo "  vulkan      - Use llama-cpp-python with Vulkan for AMD GPUs"
    echo "  vulkan-nvidia - Use llama-cpp-python with Vulkan for NVIDIA GPU only"
    echo "  cuda        - Use llama-cpp-python with CUDA for NVIDIA GPUs"
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

# Determine venv directory based on backend
if [ "$BACKEND" = "nvidia" ]; then
    VENV_DIR="venv_nvidia"
elif [ "$BACKEND" = "vulkan" ]; then
    VENV_DIR="venv_vulkan"
elif [ "$BACKEND" = "vulkan-nvidia" ]; then
    VENV_DIR="venv_vulkan_nvidia"
elif [ "$BACKEND" = "cuda" ]; then
    VENV_DIR="venv_cuda"
fi

# Create virtual environment if it doesn't exist
echo -e "${YELLOW}Creating virtual environment: $VENV_DIR${NC}"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Created virtual environment: $VENV_DIR${NC}"
else
    echo -e "${YELLOW}Using existing virtual environment: $VENV_DIR${NC}"
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
    echo "  source $VENV_DIR/bin/activate"
    echo "  python coderai --model <huggingface-model-name>"
    echo ""
    echo "Example:"
    echo "  python coderai --model microsoft/DialoGPT-medium"
    echo ""
    
elif [ "$BACKEND" = "vulkan" ]; then
    # Vulkan backend (all GPUs)
    echo -e "${YELLOW}Installing llama-cpp-python with Vulkan support (all GPUs)...${NC}"
    
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
    
    # Check for glslc (Vulkan shader compiler)
    GLSLC_CMD=""
    if command -v glslc &> /dev/null; then
        GLSLC_CMD="glslc"
    elif command -v glslangValidator &> /dev/null; then
        GLSLC_CMD="glslangValidator"
    fi
    
    if [ -z "$GLSLC_CMD" ]; then
        echo -e "${YELLOW}Warning: glslc/glslangValidator not found in PATH${NC}"
    else
        echo -e "${GREEN}✓ Found Vulkan shader compiler: $GLSLC_CMD${NC}"
    fi
    
    # Build with Vulkan support
    echo -e "${YELLOW}Building llama-cpp-python with Vulkan support...${NC}"
    CMAKE_ARGS="-DGGML_VULKAN=ON" pip install --upgrade llama-cpp-python --no-cache-dir || {
        echo -e "${RED}Build failed!${NC}"
        exit 1
    }
    
    echo -e "${YELLOW}Installing Vulkan-specific requirements...${NC}"
    pip install -r requirements-vulkan.txt
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Vulkan build complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Usage:"
    echo "  python coderai --model <gguf-model> --backend vulkan"
    echo ""
    
elif [ "$BACKEND" = "vulkan-nvidia" ]; then
    # Vulkan backend (NVIDIA only)
    echo -e "${YELLOW}Installing llama-cpp-python with Vulkan support (NVIDIA-only)...${NC}"
    
    # Check for required Vulkan development libraries
    if ! pkg-config --exists vulkan 2>/dev/null; then
        echo -e "${YELLOW}Warning: Vulkan development libraries not found via pkg-config${NC}"
    fi
    
    # Check for glslc (Vulkan shader compiler)
    GLSLC_CMD=""
    if command -v glslc &> /dev/null; then
        GLSLC_CMD="glslc"
    elif command -v glslangValidator &> /dev/null; then
        GLSLC_CMD="glslangValidator"
    fi
    
    if [ -z "$GLSLC_CMD" ]; then
        echo -e "${YELLOW}Warning: glslc/glslangValidator not found in PATH${NC}"
    else
        echo -e "${GREEN}✓ Found Vulkan shader compiler: $GLSLC_CMD${NC}"
    fi
    
    # Build with Vulkan support
    # Note: llama.cpp doesn't have a compile-time option to disable specific GPUs
    # The device selection happens at runtime via environment variables
    echo -e "${YELLOW}Building llama-cpp-python with Vulkan support...${NC}"
    CMAKE_ARGS="-DGGML_VULKAN=ON" pip install --upgrade llama-cpp-python --no-cache-dir || {
        echo -e "${RED}Build failed!${NC}"
        exit 1
    }
    
    echo -e "${YELLOW}Installing Vulkan-specific requirements...${NC}"
    pip install -r requirements-vulkan.txt
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Vulkan (NVIDIA-only) build complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Usage:"
    echo "  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\"
    echo "  python coderai --model <gguf-model> --backend vulkan"
    echo ""
    echo "Note: This build includes both AMD and NVIDIA Vulkan support."
    echo "      At runtime, use VK_ICD_FILENAMES to select only NVIDIA."
    echo ""
 
elif [ "$BACKEND" = "cuda" ]; then
    # llama-cpp-python with CUDA backend (NVIDIA only)
    echo -e "${YELLOW}Installing llama-cpp-python with CUDA support...${NC}"
    
    # Check for CUDA toolkit
    if ! command -v nvcc &> /dev/null; then
        echo -e "${YELLOW}Warning: CUDA toolkit (nvcc) not found in PATH${NC}"
        echo -e "${YELLOW}You may need to install CUDA toolkit:${NC}"
        echo "  Download from: https://developer.nvidia.com/cuda-downloads"
    else
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
        echo -e "${GREEN}✓ Found CUDA $CUDA_VERSION${NC}"
    fi
    
    # Check for CUDA libraries
    if [ -d "/usr/local/cuda" ]; then
        echo -e "${GREEN}✓ Found CUDA at /usr/local/cuda${NC}"
    fi
    
    # Build llama-cpp-python with CUDA support
    echo -e "${YELLOW}Building llama-cpp-python with CUDA support...${NC}"
    echo -e "${YELLOW}This may take several minutes...${NC}"
    CMAKE_ARGS="-DGGML_CUDA=ON" pip install --upgrade llama-cpp-python --no-cache-dir || {
        echo ""
        echo -e "${RED}Build failed!${NC}"
        echo -e "${YELLOW}Make sure CUDA toolkit is installed:${NC}"
        echo "  sudo apt install cuda-toolkit-12"
        echo "  or"
        echo "  Download from: https://developer.nvidia.com/cuda-downloads"
        exit 1
    }
    
    echo -e "${YELLOW}Installing Vulkan-specific requirements...${NC}"
    pip install -r requirements-vulkan.txt
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  llama-cpp-python CUDA build complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Usage:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  python coderai --model <gguf-model> --backend vulkan --vulkan-device 0"
    echo ""
    echo "Note: With CUDA backend, llama-cpp-python will only use NVIDIA GPUs."
    echo ""
fi

# Create .backend file to track which backend was used
echo "$BACKEND" > .backend

echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo "To activate the environment in the future, run:"
echo "  source $VENV_DIR/bin/activate"
