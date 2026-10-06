"""The hand-off from coderai to the external LongCat trainer.

This function had never executed. Its first real run died on

    _lora_dir() missing 1 required positional argument: 'name'

one line in — after the dataset had been built and sent, so every earlier fix was
working and this was simply the next thing in the way. A unit test that calls it
with the worker stubbed would have caught that without a GPU, a checkpoint or a
venv, which is why it exists now.
"""
import json
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

loras = pytest.importorskip("codai.api.loras")
# The function imports the worker locally (`from codai.api import
# longcat_worker`), so the module object is what has to be patched.
longcat_worker = pytest.importorskip("codai.api.longcat_worker")


@pytest.fixture
def dataset(tmp_path):
    cfg = tmp_path / "data_backend.json"
    cfg.write_text(json.dumps([{"id": "x", "dataset_type": "video"}]))
    return cfg


def _req(**kw):
    kw.setdefault("name", "vfighter_khumalo__longcat")
    kw.setdefault("dataset_config", "")
    return types.SimpleNamespace(**kw)


def _run(monkeypatch, req, **extra):
    """Call the real function with only the worker replaced."""
    seen = {}

    def _fake_train(job, workdir, on_progress=None):
        seen["job"] = job
        seen["workdir"] = workdir
        if on_progress:
            on_progress(step=1, total=10, message="warming up")
        return {"path": job["output_dir"] + "/pytorch_lora_weights.safetensors"}

    monkeypatch.setattr(longcat_worker, "train_lora", _fake_train)
    monkeypatch.setattr(loras, "_set_progress", lambda **kw: None)
    out = loras._train_externally(
        extra.pop("arch", "longcat"), req, extra.pop("base_path", "/models/longcat"),
        extra.pop("images", []), extra.pop("instance_prompt", "a fighter"),
        extra.pop("steps", 800), extra.pop("rank", 8),
        extra.pop("resolution", "480x832"), extra.pop("lr", 1e-4),
        extra.pop("seed", 42))
    return out, seen


def test_it_reaches_the_worker_at_all(monkeypatch, dataset):
    """The regression: it used to raise before ever getting here."""
    out, seen = _run(monkeypatch, _req(dataset_config=str(dataset)))
    assert seen.get("job"), "the worker was never called"
    assert out["trainer"] == "simpletuner"
    assert out["arch"] == "longcat"


def test_the_output_directory_is_the_loras_directory_for_that_name(monkeypatch, dataset):
    """_lora_dir(name) IS the per-LoRA directory. Joining the name onto it again
    would have written <loras>/<name>/<name>."""
    name = "vfighter_khumalo__longcat"
    _, seen = _run(monkeypatch, _req(name=name, dataset_config=str(dataset)))
    out_dir = pathlib.Path(seen["job"]["output_dir"])
    assert out_dir == pathlib.Path(loras._lora_dir(name))
    assert out_dir.name == name
    assert out_dir.parent.name != name, "the name is doubled in the path"


def test_the_job_carries_what_the_trainer_needs(monkeypatch, dataset):
    _, seen = _run(monkeypatch, _req(dataset_config=str(dataset)),
                   steps=1200, rank=16, seed=7)
    job = seen["job"]
    assert job["dataset_config"] == str(dataset)
    assert job["steps"] == 1200 and job["rank"] == 16 and job["seed"] == 7
    assert job["num_frames"] == 93           # 4n+1, one LongCat segment
    assert job["arch"] == "longcat"
    assert job["base_path"] == "/models/longcat"


def test_a_missing_dataset_is_refused_before_any_work(monkeypatch):
    with pytest.raises(RuntimeError, match="dataset_config"):
        _run(monkeypatch, _req(dataset_config=""))


def test_a_dataset_path_that_does_not_exist_is_refused(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        _run(monkeypatch, _req(dataset_config=str(tmp_path / "nope.json")))


def test_progress_from_the_worker_is_accepted(monkeypatch, dataset):
    """The worker emits JSON lines as **kwargs; a signature mismatch here would
    only surface mid-training."""
    got = []
    monkeypatch.setattr(longcat_worker, "train_lora",
                        lambda job, workdir, on_progress=None: (
                            on_progress(step=3, total=9, message="step"),
                            {"path": job["output_dir"]})[1])
    monkeypatch.setattr(loras, "_set_progress", lambda **kw: got.append(kw))
    loras._train_externally("longcat", _req(dataset_config=str(dataset)),
                            "/models/longcat", [], "p", 10, 4, "480x832", 1e-4, 1)
    assert got and got[0]["step"] == 3 and got[0]["total"] == 9


# ── how SimpleTuner is actually invoked ─────────────────────────────────────

def test_the_config_is_handed_over_the_way_simpletuner_reads_it():
    """SimpleTuner takes no --config. Its loader picks a configuration BACKEND and
    asks that backend where to look; the JSON one reads CONFIG_PATH and otherwise
    defaults to ./config/config.json relative to cwd. Passing --config produced:

        ValueError: JSON configuration file not found. Paths tried: config/config.json
    """
    src = (ROOT / "tools" / "longcat_train.py").read_text(encoding="utf-8")
    assert '"--config", str(cfg_path)' not in src, "that argument is ignored"
    assert 'env["CONFIG_PATH"] = str(cfg_path)' in src
    assert 'env["SIMPLETUNER_CONFIG_BACKEND"] = "json"' in src
    assert "env=env" in src, "the environment must reach the child"


def test_the_config_is_also_written_where_the_default_lookup_finds_it():
    """Belt and braces: if a loader change ignored CONFIG_PATH, the default
    ./config/config.json is there too."""
    src = (ROOT / "tools" / "longcat_train.py").read_text(encoding="utf-8")
    assert 'fallback = workdir / "config"' in src
    assert '(fallback / "config.json").write_text' in src


# ── two frame rules, not one ────────────────────────────────────────────────
# Generation obeys the VAE's 4n+1 (93, 173, 333 are whole LongCat segments).
# TRAINING is stricter: SimpleTuner's longcat_video family rounds DOWN to 8n+1
# and says nothing, so a job asking for 93 trained at 89 and the config did not
# say so. Our validator was checking the generation rule.

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("lc_train", ROOT / "tools" / "longcat_train.py")
LT = _ilu.module_from_spec(_spec)
sys.modules["lc_train"] = LT
_spec.loader.exec_module(LT)


@pytest.mark.parametrize("asked,expected", [
    (93, 89),    # a whole LongCat segment — legal to GENERATE, not to train at
    (173, 169),  # two segments
    (333, 329),  # four
    (89, 89), (97, 97), (105, 105),   # already 8n+1
    (96, 89), (8, 1), (1, 1), (0, 1),
])
def test_the_training_frame_count_rounds_the_way_simpletuner_does(asked, expected):
    assert LT.training_frames(asked) == expected


def test_it_matches_simpletuners_own_formula():
    """Mirrored from longcat_video/model.py: ((n - 1) // 8) * 8 + 1, floored at 1."""
    for n in range(0, 400):
        theirs = n if n % 8 == 1 else max(((n - 1) // 8) * 8 + 1, 1)
        assert LT.training_frames(n) == max(theirs, 1), n


def test_a_rounded_count_always_validates():
    for n in (93, 173, 333, 96, 7):
        assert LT.frame_problems(LT.training_frames(n), "480x832") == []


def test_the_validator_now_states_the_training_rule():
    problems = LT.frame_problems(93, "480x832")
    assert problems and "8n+1" in problems[0]
    assert "4n+1" in problems[0], "it should say why 93 looked valid"


def test_resolution_is_still_checked():
    assert LT.frame_problems(89, "481x832")
    assert LT.frame_problems(89, "480x833")
    assert LT.frame_problems(89, "nonsense")
    assert LT.frame_problems(89, "480x832") == []


def test_a_clip_length_is_not_rejected_merely_for_being_4n_plus_1():
    """93 is what the dataset builder uses. Blocking it would stop a run that
    previously worked; it is adjusted and reported instead."""
    src = (ROOT / "tools" / "longcat_train.py").read_text(encoding="utf-8")
    assert "is not 8n+1, which training requires" in src
    assert 'job["num_frames"] = trains_at' in src
