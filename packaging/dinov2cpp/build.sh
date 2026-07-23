#!/bin/bash
# Build dinov2.cpp with ggml's Vulkan backend + the coderai `embed` server.
#
# Produces: <workdir>/dinov2.cpp/build/bin/embed
# Deps (debian/ubuntu): build-essential cmake pkg-config libvulkan-dev glslc
#                       libopencv-dev
#
# Usage: build.sh [workdir]   (default: /tmp/dinov2cpp-build)
set -euo pipefail

WORKDIR="${1:-/tmp/dinov2cpp-build}"
PIN=3d070782afc264b7d60aa5692c5b10cb79b9bd56
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$WORKDIR"
cd "$WORKDIR"
if [ ! -d dinov2.cpp ]; then
    git clone https://github.com/lavaman131/dinov2.cpp.git
fi
cd dinov2.cpp
git checkout "$PIN"
# submodule URL is ssh (git@github.com:) — force anonymous https
git -c url."https://github.com/".insteadOf="git@github.com:" \
    submodule update --init --depth 1

# ── coderai patches (idempotent) ─────────────────────────────────────────────
# 1) Vulkan backend init in dino_model_load + header include.
python3 - <<'EOF'
import re

src = open('dinov2.cpp').read()
if 'GGML_USE_VULKAN' not in src:
    src = src.replace(
        '#ifdef GGML_USE_CUDA\n#include "ggml-cuda.h"\n#endif',
        '#ifdef GGML_USE_VULKAN\n#include "ggml-vulkan.h"\n#endif\n\n'
        '#ifdef GGML_USE_CUDA\n#include "ggml-cuda.h"\n#endif', 1)
    vk_block = (
        '#ifdef GGML_USE_VULKAN\n'
        '    if (!getenv("DINOV2_FORCE_CPU")) {\n'
        '        fprintf(stderr, "%s: using Vulkan backend\\n", __func__);\n'
        '        model.backend = ggml_backend_vk_init(0);\n'
        '        if (!model.backend) {\n'
        '            fprintf(stderr, "%s: ggml_backend_vk_init() failed\\n", __func__);\n'
        '        }\n'
        '    }\n'
        '#endif\n'
        '#ifdef GGML_USE_CUDA\n')
    src = src.replace('#ifdef GGML_USE_CUDA\n    fprintf(stderr, "%s: using CUDA backend',
                      vk_block + '    fprintf(stderr, "%s: using CUDA backend', 1)
    open('dinov2.cpp', 'w').write(src)
    print('patched dinov2.cpp')

cm = open('CMakeLists.txt').read()
if 'add_executable(embed' not in cm:
    cm = cm.replace(
        'option(BUILD_QUANTIZE',
        'add_executable(embed embed.cpp dinov2.cpp)\n'
        'target_link_libraries(embed PRIVATE ${OpenCV_LIBS} PUBLIC ggml)\n'
        'target_include_directories(embed PUBLIC .)\n'
        'if (GGML_VULKAN)\n'
        '    target_compile_definitions(embed PRIVATE GGML_USE_VULKAN)\n'
        '    target_compile_definitions(inference PRIVATE GGML_USE_VULKAN)\n'
        'endif ()\n\n'
        'option(BUILD_QUANTIZE', 1)
    open('CMakeLists.txt', 'w').write(cm)
    print('patched CMakeLists.txt')
EOF
cp "$HERE/embed.cpp" .

mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON \
      -DBUILD_REALTIME=OFF -DBUILD_QUANTIZE=OFF ..
make -j"$(nproc)" embed
echo "BUILT: $(pwd)/bin/embed"
