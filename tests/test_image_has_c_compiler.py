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

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "packaging" / "linux" / "Dockerfile.oci"
RUNPOD_VLLM_APT = ROOT / "packaging" / "runpod" / "profiles" / "vllm.apt"


def _runtime_stage(text: str) -> str:
    """Everything from the last FROM onwards — the stage that actually ships."""
    starts = [m.start() for m in re.finditer(r"^FROM ", text, re.M)]
    return text[starts[-1]:] if starts else text


def test_the_shipped_stage_installs_a_c_compiler():
    stage = _runtime_stage(DOCKERFILE.read_text(encoding="utf-8"))
    assert re.search(r"^\s+gcc\s*\\?\s*$", stage, re.M), \
        "no gcc in the runtime stage — Triton cannot build its driver module"


def test_it_also_has_the_headers_the_compile_needs():
    stage = _runtime_stage(DOCKERFILE.read_text(encoding="utf-8"))
    assert re.search(r"^\s+libc6-dev\s*\\?\s*$", stage, re.M), \
        "gcc without libc6-dev cannot compile driver.c"


def test_the_reason_is_recorded_next_to_it():
    """A later cleanup would otherwise drop a compiler from a runtime image as
    obviously unnecessary."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "Triton" in text and "runtime" in text.lower()
    assert "Failed to find C compiler" in text


def test_the_runpod_vllm_profile_agrees():
    """Both places run vLLM; both need the compiler."""
    if not RUNPOD_VLLM_APT.is_file():
        return
    pkgs = [l.strip() for l in RUNPOD_VLLM_APT.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")]
    assert "gcc" in pkgs
