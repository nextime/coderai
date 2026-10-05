"""LoRA / QLoRA training for LongCat-Video.

coderai does NOT implement this training loop, and the reasons are the design:

1. LongCat's venv is standalone — Python 3.10, torch 2.6, transformers 4.41 — so a
   trainer there cannot ``from codai.api import loras`` the way
   ``tools/lora_train_worker.py`` does. That overlay venv inherits the parent's
   site-packages; this one deliberately does not.
2. The upstream repo has **no way to create trainable LoRA layers**. Its DiT exposes
   ``load_lora()`` / ``enable_loras()`` / ``disable_all_loras()`` and nothing that
   initialises one, and the adapters it loads use its own key layout. Reimplementing that
   blind would produce adapters the pipeline cannot load — which would look like support
   while being worse than none.

SimpleTuner implements it properly (``model_family: "longcat_video"``, LoRA and quantised
LoRA), so coderai generates its config and drives it. These tests cover the wiring, the
constraints checked before a long run starts, and the refusals.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import longcat_worker as W
from codai.api.lora_archs import (ARCHS, EXTERNAL_TARGETS, TARGETS, detect_arch,
                                  external_trainer, load_classes)
from codai.config import LongcatConfig


def _trainer():
    spec = importlib.util.spec_from_file_location(
        "longcat_train", ROOT / "tools" / "longcat_train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _trainer()
LORAS = (ROOT / "codai/api/loras.py").read_text()


# ---------------------------------------------------------------- the registry
def test_longcat_is_a_target():
    assert "longcat" in TARGETS
    assert "longcat" in ARCHS


def test_it_is_detected_by_path_and_by_target():
    assert detect_arch("meituan-longcat/LongCat-Video") == "longcat"
    assert detect_arch("anything", "longcat") == "longcat"
    # `target: video` still resolves LongCat weights to LongCat, not to Wan.
    assert detect_arch("/w/LongCat-Video", "video") == "longcat"


def test_wan_is_unaffected():
    """`target: video` has always meant Wan; existing configs must keep working."""
    assert detect_arch("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "video") == "wan"
    assert external_trainer("wan") == ""


def test_it_is_marked_as_externally_trained():
    assert external_trainer("longcat") == "simpletuner"
    assert "longcat" in EXTERNAL_TARGETS


def test_load_classes_refuses_with_the_reason():
    """Its classes are in the repo's own package, not diffusers. A bare AttributeError
    here would read like a diffusers version problem."""
    with pytest.raises(RuntimeError) as e:
        load_classes("longcat")
    msg = str(e.value)
    assert "does not train in this process" in msg
    assert "simpletuner" in msg.lower()


def test_the_in_process_architectures_still_load_normally():
    """Guard against the external branch swallowing everything."""
    for arch in ("ltx2", "h3", "krea"):
        assert external_trainer(arch) == ""


# ---------------------------------------------------------------- the config it writes
def test_the_simpletuner_config_names_the_model_family():
    cfg = T.build_config({"output_dir": "/o", "dataset_config": "/d.json",
                          "steps": 800, "rank": 8, "request": {}})
    assert cfg["model_family"] == "longcat_video"
    assert cfg["model_type"] == "lora"
    assert cfg["model_flavour"] == "final"


def test_the_checkpoint_is_left_to_the_flavour():
    """SimpleTuner's quickstart says to leave pretrained_model_name_or_path unset."""
    cfg = T.build_config({"output_dir": "/o", "dataset_config": "/d.json", "request": {}})
    assert "pretrained_model_name_or_path" not in cfg


def test_qlora_is_the_base_precision_switch():
    cfg = T.build_config({"output_dir": "/o", "dataset_config": "/d.json",
                          "base_precision": "int8-quanto", "request": {}})
    assert cfg["base_model_precision"] == "int8-quanto"


def test_bf16_omits_the_precision_key():
    cfg = T.build_config({"output_dir": "/o", "dataset_config": "/d.json",
                          "base_precision": "", "request": {}})
    assert "base_model_precision" not in cfg


def test_the_memory_settings_are_what_a_13b_needs():
    cfg = T.build_config({"output_dir": "/o", "dataset_config": "/d.json", "request": {}})
    assert cfg["train_batch_size"] == 1
    assert cfg["gradient_checkpointing"] is True
    assert cfg["lora_rank"] <= 8          # the quickstart's guidance for this model


def test_the_default_precision_is_quantised():
    """A 13.6B transformer plus optimiser state does not fit a consumer card at bf16."""
    assert LongcatConfig().train_base_precision == "int8-quanto"
    assert LongcatConfig().train_lora_rank <= 8
    assert LongcatConfig().train_gradient_checkpointing is True


# ---------------------------------------------------------------- the constraints
@pytest.mark.parametrize("frames", [93, 89, 85, 5, 1])
def test_valid_frame_counts_pass(frames):
    """(num_frames - 1) must be divisible by 4 — the VAE's rule, the same 4n+1 coderai
    already applies to Wan."""
    assert T.frame_problems(frames, "480x832") == []


@pytest.mark.parametrize("frames", [90, 92, 0, 2])
def test_invalid_frame_counts_are_caught_before_the_run(frames):
    probs = T.frame_problems(frames, "480x832")
    assert probs and "4n+1" in probs[0]


def test_sides_must_clear_the_vae_stride():
    assert T.frame_problems(93, "481x832")
    assert T.frame_problems(93, "480x833")
    assert T.frame_problems(93, "720x1280") == []


def test_a_malformed_resolution_is_reported():
    assert T.frame_problems(93, "big")


# ---------------------------------------------------------------- the venv
def test_the_training_venv_is_separate_from_the_inference_one():
    """SimpleTuner pins its own torch; sharing would leave neither working."""
    assert W.resolve_train_venv() != W.resolve_venv_dir()


def test_it_can_be_pointed_anywhere(monkeypatch, tmp_path):
    assert W.resolve_train_venv({"longcat_train_venv": str(tmp_path)}) == tmp_path
    monkeypatch.setenv("CODERAI_LONGCAT_TRAIN_VENV", str(tmp_path / "t"))
    assert W.resolve_train_venv() == tmp_path / "t"


def test_an_absent_trainer_refuses_with_the_reason_and_the_command(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "resolve_train_venv", lambda cfg=None: tmp_path / "nope")
    monkeypatch.setattr(W, "_cfg_section", lambda: None)
    with pytest.raises(RuntimeError) as e:
        W.ensure_train_built({})
    msg = str(e.value)
    assert "simpletuner" in msg.lower()
    assert "-m venv" in msg and "requirements-longcat-train.txt" in msg
    # and it says WHY coderai does not just do it itself
    assert "no way to create trainable LoRA layers" in msg


def test_auto_build_is_off_by_default():
    assert LongcatConfig().train_auto_build is False


# ---------------------------------------------------------------- the dataset refusal
def test_a_video_lora_will_not_be_trained_from_stills():
    """The honest refusal: SimpleTuner wants a captioned VIDEO dataset, and silently
    turning a list of reference images into one would train a different thing."""
    fn = LORAS[LORAS.index("def _train_externally"):]
    fn = fn[:fn.index("\ndef ")]
    assert "dataset_config" in fn
    assert "will not invent one" in fn
    assert "50-100 clips" in fn


def test_a_missing_dataset_file_is_an_error():
    fn = LORAS[LORAS.index("def _train_externally"):]
    fn = fn[:fn.index("\ndef ")]
    assert "does not exist" in fn


def test_the_request_carries_the_dataset_fields():
    from codai.api.loras import LoraTrainRequest
    r = LoraTrainRequest(name="lc", base_model="meituan-longcat/LongCat-Video",
                         target="longcat", dataset_config="/d.json",
                         train_resolution="480x832")
    assert r.dataset_config == "/d.json" and r.train_resolution == "480x832"


# ---------------------------------------------------------------- the protocol
def test_progress_uses_the_existing_json_lines_protocol():
    """So the job records, /v1/loras/progress and the Tasks page need no special case."""
    assert "_drain_progress" in (ROOT / "codai/api/longcat_worker.py").read_text()
    fn = LORAS[LORAS.index("def _train_externally"):]
    fn = fn[:fn.index("\ndef ")]
    assert "_set_progress" in fn


def test_the_trainer_reports_the_adapter_it_wrote():
    src = (ROOT / "tools/longcat_train.py").read_text()
    assert "rglob" in src and "safetensors" in src
    assert "no .safetensors appeared" in src      # rather than claiming success


def test_a_nonzero_exit_is_a_failure_with_the_tail():
    src = (ROOT / "tools/longcat_train.py").read_text()
    assert "SimpleTuner exited" in src and "Last output" in src


def test_the_int8_loader_is_imported_from_the_right_module():
    """It is in longcat_video.modules.quantization, NOT the DiT module — importing it
    from the DiT module raised ImportError at the moment INT8 was asked for."""
    svc = (ROOT / "tools/longcat_service.py").read_text()
    assert "from longcat_video.modules.quantization import load_quantized_dit" in svc
