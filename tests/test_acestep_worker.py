"""ACE-Step: an isolated-venv worker that evicts like any other model.

ACE-Step pins spacy==3.8.4, which publishes no cp313 wheel, so it runs on its
own python 3.12 behind a local HTTP service. The things worth pinning here are
the ones that fail silently rather than loudly:

* a precision that upstream's two-way dtype switch would turn into float32,
* request defaults that belong to MusicGen leaking into an ACE-Step call,
* a handle that does not free its VRAM when the manager evicts it.
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

from codai.api import acestep_worker
from codai.api.audio_gen import _detect_audio_gen_type, _generate_audio
from codai.pydantic.audiogenrequest import AudioGenerationRequest

_SERVICE = Path(__file__).resolve().parents[1] / "tools" / "acestep_service.py"


def _load_service():
    """Import the service module the way nothing else can: it normally runs in
    the 3.12 venv, but its top level is stdlib-only, so it imports here."""
    spec = importlib.util.spec_from_file_location("_acestep_service", _SERVICE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", [
    "ACE-Step/ACE-Step-v1-3.5B",
    "/AI/models/acestep-v1",
    "ace-step-v1-3.5b",
])
def test_an_ace_step_model_routes_to_the_ace_step_branch(name):
    assert _detect_audio_gen_type(name) == "ace_step"


def test_other_audio_models_are_untouched():
    assert _detect_audio_gen_type("facebook/musicgen-small") == "musicgen"
    assert _detect_audio_gen_type("stabilityai/stable-audio-open-1.0") == "stable_audio"


# --------------------------------------------------------------------------- #
# precision
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("precision,expect", [
    (None, "bfloat16"),
    ("bf16", "bfloat16"),
    ("f16", "float16"),
    ("fp16", "float16"),
    ("f32", "float32"),
    ("fp32", "float32"),
    ("nonsense", "bfloat16"),
])
def test_precision_becomes_a_name_torch_actually_has(precision, expect):
    cfg = {} if precision is None else {"precision": precision}
    assert acestep_worker._dtype_name(cfg) == expect


def test_the_service_forces_the_dtype_past_the_pipelines_two_way_switch():
    """Upstream resolves `torch.bfloat16 if dtype == "bfloat16" else
    torch.float32`, so asking it for float16 gets float32 -- twice the VRAM,
    no error. ACE_PIPELINE_DTYPE is read after that switch, so the service
    must set it, or `precision: fp16` is a lie."""
    src = _SERVICE.read_text()
    assert 'os.environ["ACE_PIPELINE_DTYPE"] = a.dtype' in src


# --------------------------------------------------------------------------- #
# venv resolution
# --------------------------------------------------------------------------- #

def test_an_explicit_venv_wins(monkeypatch):
    monkeypatch.setenv("CODERAI_ACESTEP_VENV", "/somewhere/else")
    assert acestep_worker.resolve_venv_dir() == Path("/somewhere/else")


def test_without_a_baked_venv_it_falls_back_under_the_home_dir(monkeypatch):
    monkeypatch.delenv("CODERAI_ACESTEP_VENV", raising=False)
    monkeypatch.setattr(acestep_worker, "_BAKED_VENVS",
                        (Path("/nonexistent/a"), Path("/nonexistent/b")))
    assert acestep_worker.resolve_venv_dir() == Path(
        os.path.expanduser("~/.coderai/acestep_venv"))


def test_both_baked_layouts_are_accepted(monkeypatch, tmp_path):
    """The local image bakes /opt/coderai/<name>_venv, the pod images
    /opt/coderai/venvs/<name>. A worker that knows only one of them rebuilds a
    6 GB venv on a machine rented by the second."""
    monkeypatch.delenv("CODERAI_ACESTEP_VENV", raising=False)
    first, second = tmp_path / "a", tmp_path / "b"
    (second / "bin").mkdir(parents=True)
    monkeypatch.setattr(acestep_worker, "_BAKED_VENVS", (first, second))
    assert acestep_worker.resolve_venv_dir() == second


# --------------------------------------------------------------------------- #
# generation: the defaults are ACE-Step's, not MusicGen's
# --------------------------------------------------------------------------- #

class _FakePipe:
    def __init__(self):
        self.calls = []

    def generate(self, **kw):
        self.calls.append(kw)
        return b"RIFF....fake wav"


def _run(**fields):
    pipe = _FakePipe()
    req = AudioGenerationRequest(model="ACE-Step/ACE-Step-v1-3.5B",
                                 prompt="a slow piano ballad", **fields)
    out, ext = _generate_audio(pipe, "ACE-Step/ACE-Step-v1-3.5B", req)
    return pipe.calls[0], out, ext


def test_the_wav_from_the_worker_is_returned_as_it_arrives():
    """It is already a finished wav. Re-encoding it here would mean decoding
    audio this process has no reason to touch."""
    _kw, out, ext = _run()
    assert (out, ext) == (b"RIFF....fake wav", "wav")


def test_an_unsent_duration_means_ace_steps_default_not_musicgens():
    kw, _o, _e = _run()
    assert kw["duration"] == 60.0
    assert kw["guidance_scale"] == 15.0
    assert kw["steps"] == 60


def test_a_duration_the_client_really_asked_for_is_honoured_even_at_the_shared_default():
    """10.0 is the shared field default AND a legitimate request. Guessing
    'unset' from the value would silently stretch it to 60 seconds."""
    kw, _o, _e = _run(duration=10.0, cfg_coef=3.0)
    assert kw["duration"] == 10.0
    assert kw["guidance_scale"] == 3.0


def test_lyrics_and_steps_reach_the_worker():
    kw, _o, _e = _run(lyrics="[verse]\nthe tide goes out", steps=27, seed=7)
    assert kw["lyrics"].startswith("[verse]")
    assert kw["steps"] == 27
    assert kw["seed"] == 7


def test_lyrics_and_steps_are_declared_fields_not_just_extras():
    """They were readable only because the request model allows extras, so
    they were missing from the schema and from the generated docs."""
    assert {"lyrics", "steps"} <= set(AudioGenerationRequest.model_fields)


# --------------------------------------------------------------------------- #
# eviction
# --------------------------------------------------------------------------- #

def test_the_handle_frees_its_vram_the_way_the_manager_asks(monkeypatch):
    """The manager's unload dispatch looks for cleanup() first. For a worker
    the VRAM is in another process, so cleanup() must stop it -- nothing else
    frees it."""
    stopped = []
    monkeypatch.setattr(acestep_worker, "ensure_service",
                        lambda name, cfg=None: "http://127.0.0.1:1")
    monkeypatch.setattr(acestep_worker, "stop_service", stopped.append)
    handle = acestep_worker.AceStep("ACE-Step/ACE-Step-v1-3.5B", {})
    assert callable(getattr(handle, "cleanup", None))
    handle.cleanup()
    assert stopped == ["ACE-Step/ACE-Step-v1-3.5B"]


def test_the_weights_load_before_the_service_reports_healthy():
    """coderai measures a model's VRAM as the free-memory delta across the
    load, and that window closes when the worker goes healthy. Constructing
    ACEStepPipeline loads nothing -- it reads the checkpoints on first call --
    so the service has to load them itself, or the model measures ~0 GB and
    eviction treats a resident 3.5B model as free."""
    src = _SERVICE.read_text()
    load_at = min(src.index("load_checkpoint"), src.index("ENGINE._ensure()"))
    assert load_at < src.index("serve_forever()")


def test_quantized_without_the_int4_weights_says_what_is_missing():
    """`quantized=True` does not quantize: it loads ACE-Step's separate int4wo
    export. Against an ordinary checkpoint that is a FileNotFoundError several
    minutes into a load."""
    mod = _load_service()
    eng = mod.Engine(checkpoint_dir="/nonexistent/ckpt", quantized=True)
    with pytest.raises(RuntimeError) as err:
        eng._ensure()
    msg = str(err.value)
    assert "int4" in msg and "/nonexistent/ckpt" in msg
