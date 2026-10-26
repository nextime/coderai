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
      -DBUILD_SHARED_LIBS=OFF \
      -DBUILD_REALTIME=OFF -DBUILD_QUANTIZE=ON ..
make -j"$(nproc)" embed quantize

# Also leave it under the name the server looks for. The binary cmake produces
# is `embed`, but the image build stages a helper binary under its BASENAME and
# the embeddings backend resolves `dinov2-embed` -- so a copy built as `embed`
# installed as /usr/local/bin/embed and was never found. Both names, so neither
# convention breaks.
cp -f bin/embed bin/dinov2-embed

# ── private shared libraries ────────────────────────────────────────────────
# OpenCV is a build dep AND a runtime one, and the coderai image has no system
# OpenCV: only the venv's cv2 wheel, at a different version and with mangled
# sonames. So the libraries this binary needs and the image lacks are shipped
# BESIDE it, in `dinov2-embed-libs`, reached through the binary's own RPATH.
#
# Not /opt/coderai/local-libs, which is on the global LD_LIBRARY_PATH: OpenCV
# 4.10 placed there would be offered to every process in the image, cv2
# included. A private directory on one binary's RPATH cannot shadow anything.
#
# RPATH lists the sibling directory AND $ORIGIN, because the image stages
# helper binaries flat into /usr/local/bin while a developer keeps the
# directory next to the build -- both layouts then resolve.
if command -v patchelf >/dev/null 2>&1; then
    LIBDIR="$(pwd)/bin/dinov2-embed-libs"
    mkdir -p "$LIBDIR"
    # OpenCV is not the end of it: Debian's libopencv_imgcodecs links GDAL,
    # GDCM and OpenEXR, none of which this binary uses -- it decodes JPEG and
    # PNG -- but all of which the loader demands at startup. Copying only
    # libopencv_* therefore moved the error instead of removing it (measured:
    # three libraries copied, six more still "not found").
    #
    # So the whole dependency tail goes in, MINUS what any Linux has anyway:
    # libc, libm, libstdc++, libgcc, libgomp, pthread, dl and the Vulkan
    # loader, which must come from the host/image or the binary would be
    # running two C libraries at once.
    #
    # cp -L, because most of these paths are symlinks; copying the link gives a
    # 20-byte file and a mystery at load time.
    BASE_SYSTEM='^(linux-vdso|ld-linux|libc|libm|libdl|libpthread|librt|libstdc\+\+|libgcc_s|libgomp|libvulkan)\.'
    ldd bin/dinov2-embed \
      | sed -n 's/^[[:space:]]*\([^ ]*\) => \([^ ]*\).*/\1 \2/p' \
      | grep -Ev "$BASE_SYSTEM" \
      | while read -r soname path; do
            [ -e "$path" ] || continue
            cp -Lf "$path" "$LIBDIR/$soname"
        done
    patchelf --set-rpath '$ORIGIN/dinov2-embed-libs:$ORIGIN' bin/dinov2-embed
    # And every copied library needs its OWN RUNPATH. patchelf writes
    # DT_RUNPATH, and unlike the old DT_RPATH that is NOT inherited by a
    # dependency's dependencies: with it set only on the binary, the binary
    # found libopencv_imgcodecs and libopencv_imgcodecs then failed to find
    # libjpeg, GDAL and GDCM. $ORIGIN on each .so is what manylinux wheels do
    # for exactly this.
    for so in "$LIBDIR"/*.so*; do
        [ -e "$so" ] || continue
        patchelf --set-rpath '$ORIGIN' "$so" 2>/dev/null \
            || echo "  WARNING: could not set RUNPATH on $(basename "$so")"
    done
    echo "PRIVATE LIBS: $LIBDIR ($(find "$LIBDIR" -type f | wc -l) file(s), $(du -sh "$LIBDIR" | cut -f1))"
else
    echo "WARNING: patchelf not installed -- no private library directory was"
    echo "         made. The binary will need its OpenCV libraries resolvable"
    echo "         wherever it runs."
fi

echo "BUILT: $(pwd)/bin/embed"
echo "BUILT: $(pwd)/bin/dinov2-embed   (install this one as "
echo "       /usr/local/bin/dinov2-embed, where the image collects helpers;"
echo "       keep dinov2-embed-libs/ beside it -- the image build copies it)"
