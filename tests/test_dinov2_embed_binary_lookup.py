"""The DINOv2 GGUF embedder's binary has to be FOUND where it is installed.

Measured on a running install, 2026-10-26: 6,176 image-embedding requests
answered 500 over about twelve hours, every one of them
``dinov2-embed binary not found at /opt/coderai/bin/dinov2-embed``. Two
mistakes compounded:

* the image collects locally compiled helper binaries into ``/usr/local/bin``
  (whisper-server, ds4-server, colibri, rife), and ``/opt/coderai/bin`` does
  not exist in it at all -- so the one path it looked at could never hold the
  binary, and the error told whoever read it to install it there;
* ``dinov2-embed`` was not in the image build's discovery list, so it was
  dropped by a rebuild and nothing failed until a client asked.

This model is image-only and llama.cpp cannot load a ViT-only arch, so there
is no fallback runtime: a missing binary is the whole capability gone.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codai.api.embeddings as emb


@pytest.fixture(autouse=True)
def _no_inherited_env(monkeypatch):
    monkeypatch.delenv("DINOV2_EMBED_BIN", raising=False)


def test_the_install_location_is_looked_at_first(monkeypatch, tmp_path):
    """/usr/local/bin is where the image puts it; it must be tried."""
    assert emb._DINOV2_EMBED_CANDIDATES[0] == "/usr/local/bin/dinov2-embed"
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    found = {"/usr/local/bin/dinov2-embed"}
    monkeypatch.setattr(emb.os.path, "isfile", lambda p: p in found)
    assert emb._dinov2_embed_bin() == "/usr/local/bin/dinov2-embed"


def test_path_wins_over_the_hardcoded_list(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/somewhere/else/dinov2-embed"
                        if name == "dinov2-embed" else None)
    assert emb._dinov2_embed_bin() == "/somewhere/else/dinov2-embed"


def test_an_explicit_override_is_obeyed(monkeypatch, tmp_path):
    binary = tmp_path / "dinov2-embed"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv("DINOV2_EMBED_BIN", str(binary))
    assert emb._dinov2_embed_bin() == str(binary)


def test_an_explicit_override_that_is_missing_says_so(monkeypatch):
    monkeypatch.setenv("DINOV2_EMBED_BIN", "/nope/dinov2-embed")
    with pytest.raises(RuntimeError) as exc:
        emb._dinov2_embed_bin()
    assert "DINOV2_EMBED_BIN" in str(exc.value)


def test_the_error_names_every_place_it_looked(monkeypatch):
    """The old message named one directory, and the wrong one."""
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(emb.os.path, "isfile", lambda p: False)
    with pytest.raises(RuntimeError) as exc:
        emb._dinov2_embed_bin()
    msg = str(exc.value)
    for cand in emb._DINOV2_EMBED_CANDIDATES:
        assert cand in msg
    assert "PATH" in msg
    assert "packaging/dinov2cpp/build.sh" in msg
    # And that it is not a degraded mode: there is no other runtime.
    assert "no fallback" in msg


def test_the_image_build_collects_the_binary():
    """Otherwise the next rebuild drops the capability again, silently."""
    script = (Path(__file__).resolve().parents[1]
              / "packaging/linux/build_oci_image.sh").read_text()
    assert "/usr/local/bin/dinov2-embed" in script, (
        "dinov2-embed is not in discover_local_binaries, so an image built "
        "from this script serves the DINOv2 GGUF model as a 500")


def test_the_native_build_produces_that_name():
    build = (Path(__file__).resolve().parents[1]
             / "packaging/dinov2cpp/build.sh").read_text()
    assert "bin/dinov2-embed" in build


# ===========================================================================
#  the private library directory
# ===========================================================================
#  The binary links OpenCV, and the image has NO system OpenCV -- only the
#  venv's cv2 wheel, a different version with mangled sonames. So the
#  libraries it needs travel beside it, on its own RPATH. Not in
#  /opt/coderai/local-libs: that is on the global LD_LIBRARY_PATH, where
#  OpenCV 4.10 would be offered to every process in the image, cv2 included.

def _build_script() -> str:
    return (Path(__file__).resolve().parents[1]
            / "packaging/dinov2cpp/build.sh").read_text()


def _image_script() -> str:
    return (Path(__file__).resolve().parents[1]
            / "packaging/linux/build_oci_image.sh").read_text()


def test_the_build_makes_a_private_library_directory():
    src = _build_script()
    assert "dinov2-embed-libs" in src
    assert "patchelf --set-rpath" in src


def test_the_rpath_covers_both_layouts():
    """A sibling directory for a developer, flat for the image's /usr/local/bin."""
    src = _build_script()
    assert "'$ORIGIN/dinov2-embed-libs:$ORIGIN'" in src


def test_the_whole_dependency_tail_is_copied_not_just_opencv():
    """Copying only libopencv_* moved the error instead of removing it.

    Debian's libopencv_imgcodecs links GDAL, GDCM and OpenEXR -- unused by a
    JPEG/PNG decoder, demanded by the loader all the same. Measured: three
    libraries copied, six more still "not found".
    """
    src = _build_script()
    assert "BASE_SYSTEM=" in src, "the tail must be copied minus the base system"
    # Not an opencv-only filter any more.
    assert "s/^[[:space:]]*\\(libopencv[^ ]*\\) =>" not in src


def test_the_base_system_is_left_to_the_image():
    """libc/libstdc++/libgomp/Vulkan must come from the image: shipping a
    second C library next to the first is how a binary stops starting."""
    src = _build_script()
    base = src.split("BASE_SYSTEM=")[1].split("\n")[0]
    for lib in ("libc", "libstdc", "libgcc_s", "libgomp", "libpthread",
                "libvulkan", "ld-linux"):
        assert lib in base, f"{lib} should be excluded from the private copy"


def test_symlinks_are_dereferenced():
    """Most of these paths are symlinks; copying the link gives a 20-byte file
    and a mystery at load time -- which is what made the first measurement of
    this set read 20 MB when it was not."""
    src = _build_script()
    assert 'cp -Lf "$path"' in src


def test_the_image_build_carries_the_private_directory():
    src = _image_script()
    assert '${bin_path}-libs' in src
    assert "local-bin/$(basename \"$priv_libs\")" in src


def test_the_private_libraries_do_not_go_into_local_libs():
    """local-libs is global; this must stay private to the one binary."""
    src = _build_script()
    assert "local-libs" in src, "the reason should be written down"
    assert "cp" not in src.split("local-libs")[1].split("\n")[0]


def test_every_private_library_gets_its_own_runpath():
    """DT_RUNPATH is not inherited by a dependency's dependencies.

    With it set only on the binary, the binary found libopencv_imgcodecs and
    libopencv_imgcodecs then failed to find libjpeg, GDAL and GDCM -- six
    libraries still "not found" with all 98 of them sitting in the directory
    next to it.
    """
    src = _build_script()
    body = src.split("$LIBDIR")[0]  # ensure the loop is in the lib section
    assert 'patchelf --set-rpath \'$ORIGIN\' "$so"' in src
    assert "DT_RUNPATH" in src, "the reason belongs next to the loop"
