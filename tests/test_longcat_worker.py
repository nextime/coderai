"""Step 2 of the LongCat-Video integration: the worker, the service contract and dispatch.

LongCat runs in an isolated Python 3.10 venv behind a managed subprocess, because it pins
Python 3.10, torch 2.6+cu124, transformers 4.41 and numpy 1.26 against the main venv's
3.13 / 2.11+cu130 / 5.x / 2.4. These tests cover the parts that are decidable without the
venv, the ~83 GB checkpoint or a GPU: detection, identity, venv/source resolution, the
remote escape hatch, the segment arithmetic and the stage contracts.

What is NOT covered here, and is therefore not yet claimed to work: a real generation.
That needs the venv, the repo checkout and the weights.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import longcat_worker as W


def _common():
    spec = importlib.util.spec_from_file_location(
        "longcat_common", ROOT / "tools" / "longcat_common.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LC = _common()


# ------------------------------------------------------------ detection
def test_the_backend_pin_wins():
    """H3's documented `backend` switch is dead because the video dispatch path never
    unwraps _raw_cfg; LongCat's must actually work."""
    assert W.is_longcat_model("anything-at-all", {"backend": "longcat"})
    assert W.is_longcat_model("x", {"backend": "longcat-video"})


def test_an_alias_is_honoured():
    """An entry is routable by `alias or path or id`, and aliases are how one model
    carries several configs and how several models share an endpoint."""
    assert W.is_longcat_model("x", {"alias": "longcat", "path": "meituan/whatever"})
    assert W.is_longcat_model("", {"model_id": "longcat-720p"})


def test_the_checkpoint_name_is_recognised():
    assert W.is_longcat_model("meituan-longcat/LongCat-Video")
    assert W.is_longcat_model("LongCat_Video")          # underscores normalised


def test_another_backend_pin_declines():
    """A model pinned to a different engine must not be grabbed by a name match."""
    assert not W.is_longcat_model("longcat-ish", {"backend": "vllm"})
    assert not W.is_longcat_model("longcat-ish", {"backend": "runpod"})


def test_unrelated_models_are_left_alone():
    assert not W.is_longcat_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    assert not W.is_longcat_model("MiniMaxAI/MiniMax-H3")
    assert not W.is_longcat_model("")


# ------------------------------------------------------------ identity
def test_sibling_configs_are_separate_services():
    """A bf16 entry and an fp8 entry of the same weights are DIFFERENT residencies with
    different footprints; sharing one service (or one measured VRAM figure) would make
    the second inherit the first's."""
    a = W.service_key("/w/longcat", {"config_id": "aaa"})
    b = W.service_key("/w/longcat", {"config_id": "bbb"})
    assert a != b
    assert W.service_key("/w/longcat", {"variant": "fp8"}) != \
        W.service_key("/w/longcat", {"variant": "bf16"})


def test_the_same_config_is_one_service():
    cfg = {"config_id": "aaa"}
    assert W.service_key("/w/longcat", cfg) == W.service_key("/w/longcat", cfg)


def test_the_path_is_resolved_from_the_entry():
    assert W.resolve_model_path("x", {"model_path": "/weights/lc"}) == "/weights/lc"
    assert W.resolve_model_path("x", {"path": "/p"}) == "/p"
    assert W.resolve_model_path("named") == "named"


# ------------------------------------------------------------ venv + source
def test_the_venv_can_be_pointed_anywhere(monkeypatch, tmp_path):
    assert W.resolve_venv_dir({"longcat_venv": str(tmp_path)}) == tmp_path
    monkeypatch.setenv("CODERAI_LONGCAT_VENV", str(tmp_path / "env"))
    assert W.resolve_venv_dir() == tmp_path / "env"


def test_the_source_can_be_pointed_anywhere(monkeypatch, tmp_path):
    assert W.resolve_source_dir({"longcat_source": str(tmp_path)}) == tmp_path
    monkeypatch.setenv("CODERAI_LONGCAT_SRC", str(tmp_path / "src"))
    assert W.resolve_source_dir() == tmp_path / "src"


def test_an_unusable_venv_refuses_with_the_manual_command(monkeypatch, tmp_path):
    """auto_build is off by default (torch 2.6 plus its CUDA runtime is gigabytes), so
    the error has to tell the operator exactly what to run."""
    monkeypatch.setattr(W, "resolve_venv_dir", lambda cfg=None: tmp_path / "nope")
    monkeypatch.setattr(W, "_cfg_section", lambda: None)
    with pytest.raises(RuntimeError) as e:
        W.ensure_built({})
    msg = str(e.value)
    assert "-m venv" in msg and "requirements-longcat.txt" in msg
    assert "auto_build" in msg


def test_auto_build_without_a_310_interpreter_says_so(monkeypatch, tmp_path):
    """coderai cannot create this venv itself — its own interpreter is 3.13."""
    monkeypatch.setattr(W, "resolve_venv_dir", lambda cfg=None: tmp_path / "nope")
    monkeypatch.setattr(W, "_cfg_section",
                        lambda: type("S", (), {"auto_build": True, "python": ""})())
    monkeypatch.setattr(W, "_find_python310", lambda cfg=None: None)
    with pytest.raises(RuntimeError) as e:
        W.ensure_built({})
    assert "Python 3.10" in str(e.value)


def test_the_venv_probe_checks_versions_not_just_importability():
    """The image rewrites pyvenv.cfg; a venv re-pointed at the wrong interpreter would
    import the MAIN venv's torch 2.11 and pass a bare `import torch`."""
    src = (ROOT / "codai/api/longcat_worker.py").read_text()
    probe = src[src.index("def _venv_ok"):src.index("def _find_python310")]
    assert "(3,10)" in probe and "'2.6'" in probe and "'4.41'" in probe


def test_the_current_interpreter_is_not_310():
    """If this ever fails the isolation rationale needs revisiting, not the test."""
    assert sys.version_info[:2] != (3, 10)


# ------------------------------------------------------------ remote escape hatch
def test_a_configured_service_url_is_proxied_and_nothing_is_built(monkeypatch):
    """This is what makes the engine remotizable for free: ensure_service returns a URL
    either way, so no caller can tell a pod from a local subprocess."""
    built = []
    monkeypatch.setattr(W, "ensure_built", lambda cfg=None: built.append(1))
    monkeypatch.setattr(W, "_health_ok", lambda url: True)
    monkeypatch.setattr(W, "_cfg_section", lambda: None)
    url = W.ensure_service("/w/lc", {"service_url": "http://pod:8000/"})
    assert url == "http://pod:8000"
    assert built == []


def test_a_dead_remote_service_is_an_error_not_a_silent_local_start(monkeypatch):
    monkeypatch.setattr(W, "_health_ok", lambda url: False)
    monkeypatch.setattr(W, "_cfg_section", lambda: None)
    with pytest.raises(RuntimeError) as e:
        W.ensure_service("/w/lc", {"service_url": "http://pod:8000"})
    assert "health check" in str(e.value)


# ------------------------------------------------------------ shared contracts
def test_the_segment_arithmetic_matches_upstream():
    """93 frames per call of which 13 re-render the previous tail, and upstream's demo
    uses 11 segments for "a 1-minute video" — which only works out if a segment ADDS 80."""
    assert LC.frames_for(1) == 93
    assert LC.frames_for(2) == 93 + 80
    assert 59.0 <= LC.duration_seconds(11) <= 60.5
    assert LC.segments_for(LC.frames_for(7)) == 7


def test_a_short_request_is_one_segment():
    assert LC.segments_for(10) == 1
    assert LC.segments_for(93) == 1
    assert LC.segments_for(94) == 2


def test_the_presets_map_to_stages():
    assert LC.resolve_stages("draft") == ("base",)
    assert LC.resolve_stages("fast") == ("distill",)
    assert LC.resolve_stages("best") == ("base", "refinement")


def test_an_explicit_stage_overrides_the_preset():
    assert LC.resolve_stages("best", "distill") == ("distill",)


def test_unknown_presets_and_stages_are_rejected():
    with pytest.raises(ValueError):
        LC.resolve_stages("ultra")
    with pytest.raises(ValueError):
        LC.resolve_stages("fast", "polish")


def test_the_distilled_stage_is_the_cheap_one():
    assert LC.stage_params("distill")["num_inference_steps"] == 16
    assert LC.stage_params("base")["num_inference_steps"] == 50
    assert LC.stage_params("distill")["guidance_scale"] == 1.0


def test_caller_overrides_win_over_stage_defaults():
    p = LC.stage_params("base", steps=8, guidance=2.5)
    assert p == {"num_inference_steps": 8, "guidance_scale": 2.5}


def test_a_missing_checkpoint_is_reported_before_loading(tmp_path):
    """Otherwise it surfaces minutes in, as a from_pretrained traceback from inside the
    venv."""
    assert LC.checkpoint_problems("")
    assert LC.checkpoint_problems(str(tmp_path / "absent"))
    for sub in LC.CHECKPOINT_SUBDIRS:
        (tmp_path / sub).mkdir()
    assert LC.checkpoint_problems(str(tmp_path)) == []


def test_a_stage_whose_lora_is_missing_is_reported(tmp_path):
    """Stages 2 and 3 ARE their LoRAs; without them the stage silently is not itself."""
    for sub in LC.CHECKPOINT_SUBDIRS:
        (tmp_path / sub).mkdir()
    probs = LC.checkpoint_problems(str(tmp_path), ("distill",))
    assert probs and "cfg_step_lora" in probs[0]
    (tmp_path / "lora").mkdir()
    (tmp_path / "lora" / "cfg_step_lora.safetensors").write_text("x")
    assert LC.checkpoint_problems(str(tmp_path), ("distill",)) == []
    assert LC.checkpoint_problems(str(tmp_path), ("refinement",))     # the other one


def test_a_source_checkout_without_the_package_is_reported(tmp_path):
    """`longcat_video` is in the repo, not on PyPI — the venv alone is not enough."""
    assert LC.source_problems(str(tmp_path))
    pkg = tmp_path / "longcat_video"
    (pkg / "modules").mkdir(parents=True)
    (pkg / "pipeline_longcat_video.py").write_text("x")
    (pkg / "modules" / "longcat_video_dit.py").write_text("x")
    assert LC.source_problems(str(tmp_path)) == []


# ------------------------------------------------------------ the service CLI
def test_the_service_script_parses_its_arguments():
    """It runs under a different interpreter, so a signature mismatch between worker and
    service shows up only at launch — check the contract from here."""
    out = subprocess.run([sys.executable, str(ROOT / "tools/longcat_service.py"), "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    for flag in ("--model", "--source", "--port", "--dtype", "--offload",
                 "--attention", "--preload"):
        assert flag in out.stdout


def test_the_worker_launches_the_service_with_flags_it_accepts():
    src = (ROOT / "codai/api/longcat_worker.py").read_text()
    svc = (ROOT / "tools/longcat_service.py").read_text()
    for flag in ("--model", "--source", "--host", "--port", "--dtype"):
        assert f'"{flag}"' in src, f"worker never passes {flag}"
        assert f'"{flag}"' in svc, f"service does not declare {flag}"


# ------------------------------------------------------------ dispatch
def test_the_dispatch_branch_unwraps_raw_cfg():
    """build_runtime_kwargs output has a FIXED key list for video — no backend, no alias,
    no path — all of which survive only inside _raw_cfg. H3 inherited that and its
    backend switch is dead; this one must not."""
    src = (ROOT / "codai/api/video.py").read_text()
    branch = src[src.index("_lc_cfg = dict("):src.index("if _is_longcat:")]
    assert "_raw_cfg" in branch and "setdefault" in branch
    assert "is_longcat_model" in branch


def test_the_request_model_accepts_the_staging_fields():
    from codai.pydantic.videorequest import VideoGenerationRequest
    r = VideoGenerationRequest(model="m", prompt="p", quality="best", stage="base")
    assert (r.quality, r.stage) == ("best", "base")
    assert VideoGenerationRequest(model="m", prompt="p").quality is None


def test_unimplemented_modes_are_refused_not_downgraded():
    """Step 2 is t2v only. A silent downgrade would look like a quality bug."""
    src = (ROOT / "codai/api/video.py").read_text()
    fn = src[src.index("async def _generate_longcat"):]
    fn = fn[:fn.index("\nasync def ")]
    assert "is not implemented yet" in fn
    assert "status_code=400" in fn
