"""The stamp/signature detector, reachable from a venv that is not ours.

`detect.py` imported ultralytics IN-PROCESS, which works wherever the main venv
carries torch. The `vllm-ocr` pod image is light-core on purpose —
Dockerfile.capability FAILS the build if torch reaches the main venv — so the
detector could not live there without a third torch (~2.5 GB compressed) for a
workload that does not even want the card: the GPU is serving the model.

So ultralytics is installed beside surya, sharing the CPU torch already in that
venv, and reached through a worker. Measured, in the orchestrator's surya venv:
nvidia 3196 MB + triton 896 MB + torch 1175 MB = ~5.3 GB of GPU stack, for a
venv whose only local torch workload is surya's small line detector. A CPU torch
is ~250 MB, and the detector then costs ~30 MB on top instead of a torch.

The rule these tests hold: a detector that cannot be reached must look like a
MISSING detector (the ordinary 503), never like a broken install.
"""

import io
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.ocr import detect as dt                           # noqa: E402
from codai.ocr.base import OcrError                          # noqa: E402


@pytest.fixture(autouse=True)
def _no_shared_worker(monkeypatch):
    monkeypatch.setattr(dt._DetectWorker, "_inst", None, raising=False)
    yield
    dt._DetectWorker._inst = None


def _img(w=40, h=40):
    Image = pytest.importorskip("PIL.Image")
    return Image.new("RGB", (w, h), "white")


# ------------------------------------------------- in-process stays first #
def test_in_process_ultralytics_is_preferred(monkeypatch, tmp_path):
    """Nothing changes on an install whose main venv has torch."""
    pt = tmp_path / "w.pt"
    pt.write_bytes(b"x")
    used = {}

    class _YOLO:
        def __init__(self, path):
            used["path"] = path

        def predict(self, arr, conf=0.0, verbose=False):
            return []
    mod = type(sys)("ultralytics")
    mod.YOLO = _YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    monkeypatch.setattr(dt._DetectWorker, "get",
                        classmethod(lambda cls: pytest.fail("worker used")))
    d = dt.YoloDetector(str(pt), 0.35)
    assert d.detect(_img()) == []
    assert used["path"] == str(pt)


def test_missing_weights_is_a_400_before_anything_is_loaded(tmp_path):
    """The weights are operator-supplied — ocr.detect_model_path defaults to ""
    and a released image carries no models — so this is the common misconfig."""
    d = dt.YoloDetector(str(tmp_path / "nope.pt"), 0.35)
    with pytest.raises(OcrError) as e:
        d.ensure_loaded()
    assert e.value.status == 400


def test_an_empty_weights_path_is_also_a_400():
    d = dt.YoloDetector("", 0.35)
    with pytest.raises(OcrError) as e:
        d.ensure_loaded()
    assert e.value.status == 400


# ------------------------------------------------------- the venv fallback #
def test_no_ultralytics_and_no_venv_is_the_ordinary_503(monkeypatch, tmp_path):
    pt = tmp_path / "w.pt"
    pt.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    monkeypatch.setattr(dt._DetectWorker, "get", classmethod(lambda cls: None))
    d = dt.YoloDetector(str(pt), 0.35)
    with pytest.raises(OcrError) as e:
        d.ensure_loaded()
    assert e.value.status == 503
    assert "ultralytics is not installed" in str(e.value)


def test_the_worker_is_used_when_ultralytics_is_elsewhere(monkeypatch, tmp_path):
    pt = tmp_path / "w.pt"
    pt.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    seen = {}

    class _Fake:
        def detect(self, image, model_path, conf):
            seen["path"] = model_path
            seen["conf"] = conf
            seen["size"] = image.size
            return [{"label": "stamp", "conf": 0.9, "bbox": [1, 2, 3, 4]}]
    monkeypatch.setattr(dt._DetectWorker, "get", classmethod(lambda cls: _Fake()))
    d = dt.YoloDetector(str(pt), 0.42)
    got = d.detect(_img(50, 60))
    assert got == [{"label": "stamp", "conf": 0.9, "bbox": [1, 2, 3, 4]}]
    assert seen == {"path": str(pt), "conf": 0.42, "size": (50, 60)}


def test_the_surya_venv_is_looked_at_first():
    """That is where the pod image puts ultralytics, sharing the CPU torch
    already there — the reason it costs ~30 MB and not another torch."""
    assert dt._DetectWorker._VENVS[0] == "/opt/coderai/venvs/surya"


def test_a_venv_that_will_not_start_does_not_raise(monkeypatch, tmp_path):
    """A broken venv must degrade to 'no detector', not to a 500."""
    vdir = tmp_path / "venvs" / "surya" / "bin"
    vdir.mkdir(parents=True)
    (vdir / "python").write_text("#!/bin/false\n")
    monkeypatch.setattr(dt._DetectWorker, "_VENVS",
                        (str(tmp_path / "venvs" / "surya"),))

    def _boom(self, python):
        raise OcrError("nope", status=503)
    monkeypatch.setattr(dt._DetectWorker, "__init__", _boom)
    assert dt._DetectWorker.get() is None


def test_the_worker_is_shared_across_detectors(monkeypatch):
    calls = []

    class _Fake:
        pass
    monkeypatch.setattr(dt._DetectWorker, "_VENVS", ())
    # With no venvs, get() returns None every time and never caches a None as
    # if it were a worker -- so a venv appearing later is still found.
    assert dt._DetectWorker.get() is None
    assert dt._DetectWorker._inst is None


# -------------------------------------------------------- the worker script #
def test_the_worker_does_not_import_the_ocr_package():
    """`import codai.ocr` costs 377 MB and 1.5 s via litellm; the worker exists
    to stay small."""
    import ast
    src = (ROOT / "codai/ocr/workers/detect_worker.py").read_text()
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert "codai" not in names, sorted(names)


def test_the_worker_is_run_isolated():
    src = (ROOT / "codai/ocr/detect.py").read_text()
    assert '"-I"' in src, "the worker must not pick up modules from the cwd"


def test_the_worker_keeps_serving_after_a_bad_page():
    """One undecodable page must not cost the detector its loaded weights."""
    src = (ROOT / "codai/ocr/workers/detect_worker.py").read_text()
    body = src.split('if cmd != "detect"')[1]
    assert body.count("continue") >= 3, (
        "a failure path exits instead of answering and carrying on")


# ------------------------------------------------- the image-size reasoning #
def test_the_pod_venv_pins_a_cpu_torch():
    """A CUDA torch resolving in there silently adds ~5.3 GB AND puts surya's
    line detector on the card vLLM is serving from."""
    txt = (ROOT / "packaging/runpod/profiles/vllm-ocr.venv-surya.txt").read_text()
    assert "download.pytorch.org/whl/cpu" in txt
    assert "+cpu" in txt
    assert "ultralytics" in txt


def test_the_pod_venv_check_enforces_cpu_torch():
    chk = (ROOT / "packaging/runpod/profiles/vllm-ocr.venv-surya.check").read_text()
    assert "torch.version.cuda is None" in chk
    assert "from ultralytics import YOLO" in chk


def test_the_pod_image_does_not_bake_detector_weights():
    """ocr.detect_model_path defaults to "" and a released image must contain no
    models, configs, keys or secrets."""
    for f in ("vllm-ocr.txt", "vllm-ocr.venv-surya.txt"):
        txt = (ROOT / "packaging/runpod/profiles" / f).read_text()
        assert ".pt" not in txt.replace("detect_model_path", ""), f
