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


# --------------------------------------------------------------------------- #
# the environment a foreign interpreter is spawned into
# --------------------------------------------------------------------------- #

def test_the_leaky_python_vars_are_stripped_for_a_foreign_interpreter():
    """Measured in the shipped container: the per-engine launchers export
    PYTHONHOME=/opt/coderai/python (the standalone 3.13) while PID 1 and an
    interactive `docker exec` do not. A 3.12 interpreter inheriting it aborts
    before running any code -- so the probe failed in milliseconds, and the
    caller reported a missing requirements file instead."""
    from codai.api.venv_env import clean_py_env
    env = clean_py_env({"PYTHONHOME": "/opt/coderai/python",
                        "PYTHONPATH": "/opt/coderai/app",
                        "VIRTUAL_ENV": "/opt/coderai/python",
                        "HF_HOME": "/cache/huggingface"})
    assert "PYTHONHOME" not in env and "PYTHONPATH" not in env
    assert "VIRTUAL_ENV" not in env
    # Everything else is left alone: the worker needs the HF cache vars.
    assert env["HF_HOME"] == "/cache/huggingface"


def test_every_spawn_of_the_venvs_python_cleans_the_environment():
    """One missed call site is a worker that dies only inside the container."""
    import inspect

    src = inspect.getsource(acestep_worker)
    spawns = [l for l in src.splitlines()
              if "subprocess.run(" in l or "subprocess.Popen(" in l]
    assert spawns, "no spawn sites found -- did the worker stop using subprocess?"
    # Each spawn either passes env= explicitly or is in a block that built one.
    assert src.count("clean_py_env()") >= 5, (
        "expected every spawn path (probe, venv create x2, install, service) "
        f"to clean the env; found {src.count('clean_py_env()')} uses")


def test_longcat_shares_the_same_helper():
    """It had this fix first, with a docstring saying coderai never sets those
    vars -- which the container disproved. One implementation now."""
    import inspect

    from codai.api import longcat_worker
    assert "clean_py_env" in inspect.getsource(longcat_worker._clean_py_env)


# --------------------------------------------------------------------------- #
# the baked image variant
# --------------------------------------------------------------------------- #

_OVERLAY = Path(__file__).resolve().parents[1] / "packaging" / "linux" / \
    "Dockerfile.acestep-overlay"


def test_the_overlay_bakes_the_venv_where_the_worker_looks_for_it():
    """The variant is only useful if the worker finds it without being told.
    A path that disagrees with _BAKED_VENVS means a 6.4 GB layer the worker
    ignores while it rebuilds the same venv on first use."""
    baked = str(acestep_worker._BAKED_VENVS[0])
    assert baked == "/opt/coderai/acestep_venv"
    assert f"{baked}/" in _OVERLAY.read_text()


def test_the_overlay_runs_the_venvs_python_with_pythonhome_cleared():
    """The runtime image sets PYTHONHOME to the standalone 3.13, so the
    overlay's own import check would die on the trap it is meant to catch --
    and `exit 1` on a venv that is perfectly fine."""
    src = _OVERLAY.read_text()
    assert "env -u PYTHONHOME" in src
    assert "import sys, acestep, torch, soundfile" in src


def test_no_comment_lines_inside_the_overlays_run_continuation():
    """Docker strips comment lines inside a continuation, splicing the shell
    lines around them together. The comment belongs above the RUN."""
    lines = _OVERLAY.read_text().splitlines()
    for prev, cur in zip(lines, lines[1:]):
        if prev.rstrip().endswith("\\"):
            assert not cur.strip().startswith("#"), f"comment spliced after: {prev}"
