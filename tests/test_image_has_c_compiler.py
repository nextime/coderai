"""The runtime image must carry a C compiler, for Triton.

Not a build-time dependency — a RUNTIME one. Triton (under vLLM) JIT-compiles its
CUDA driver module on the first engine start:

    RuntimeError: Failed to find C compiler. Please specify via CC environment
    variable or set triton.knobs.build.impl.

which the engine manager sees only as "vLLM exited (code 1) before becoming
ready". Every vLLM engine then falls back to another one, and because the OCR
fallback chain works, the requests still return 200 — so surya was never used and
nothing looked broken. A remote host sending PaddleOCR requests was the only
visible symptom.

The RunPod vLLM capability profile had gcc from the start; the main image did not.
This keeps the two from drifting again.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
# BOTH can ship: build_oci_image.sh picks Dockerfile.oci-venv for a --venv build
# and Dockerfile.oci otherwise. Checking only one is how gcc was added to the
# file that was not being used, and the build came out without a compiler.
DOCKERFILES = [ROOT / "packaging" / "linux" / "Dockerfile.oci",
               ROOT / "packaging" / "linux" / "Dockerfile.oci-venv",
               # The LongCat pod image too: block-sparse attention is pure Triton, so a
               # pod without a compiler dies on the first denoising step. Its bases
               # (capability-base-light) carry no compiler, so it must add its own.
               ROOT / "packaging" / "runpod" / "Dockerfile.capability-video-longcat",
               # The shared capability base: it carries torch and, via core.txt,
               # bitsandbytes — which reaches Triton's autotuner on import. Every
               # profile image built FROM it inherits the gap otherwise.
               ROOT / "packaging" / "runpod" / "Dockerfile.capability-base"]
RUNPOD_VLLM_APT = ROOT / "packaging" / "runpod" / "profiles" / "vllm.apt"


def _runtime_stage(text: str) -> str:
    """Everything from the last FROM onwards — the stage that actually ships."""
    starts = [m.start() for m in re.finditer(r"^FROM ", text, re.M)]
    return text[starts[-1]:] if starts else text


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_the_shipped_stage_installs_a_c_compiler(dockerfile):
    stage = _runtime_stage(dockerfile.read_text(encoding="utf-8"))
    assert re.search(r"^\s+gcc\s*\\?\s*$", stage, re.M), \
        f"no gcc in {dockerfile.name}'s runtime stage — Triton cannot build its driver"


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_it_also_has_the_headers_the_compile_needs(dockerfile):
    stage = _runtime_stage(dockerfile.read_text(encoding="utf-8"))
    assert re.search(r"^\s+libc6-dev\s*\\?\s*$", stage, re.M), \
        f"gcc without libc6-dev cannot compile driver.c ({dockerfile.name})"


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_the_reason_is_recorded_next_to_it(dockerfile):
    """A later cleanup would otherwise drop a compiler from a runtime image as
    obviously unnecessary."""
    text = dockerfile.read_text(encoding="utf-8")
    assert "Triton" in text and "runtime" in text.lower()
    assert "Failed to find C compiler" in text


def test_the_runpod_vllm_profile_agrees():
    """Both places run vLLM; both need the compiler."""
    if not RUNPOD_VLLM_APT.is_file():
        return
    pkgs = [l.strip() for l in RUNPOD_VLLM_APT.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")]
    assert "gcc" in pkgs

def test_the_light_base_is_deliberately_left_without_one():
    """capability-base-light ships no torch on purpose, so it has no Triton to build
    for. Anything that adds its own torch there — the LongCat 3.10 venv — must bring
    its own compiler, which is why Dockerfile.capability-video-longcat installs one
    even though its parent does not.
    """
    light = (ROOT / "packaging" / "runpod" /
             "Dockerfile.capability-base-light").read_text(encoding="utf-8")
    assert "torch" not in light.split("FROM")[-1].lower() or "No torch here" in light
    longcat = (ROOT / "packaging" / "runpod" /
               "Dockerfile.capability-video-longcat").read_text(encoding="utf-8")
    assert re.search(r"^\s+gcc\s*\\?\s*$", longcat, re.M), \
        "the light base gives it nothing, so it must install its own"
