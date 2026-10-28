"""LatentSync and MuseTalk are engines coderai installs, not engines it hopes to find.

The presenter has ranked both ahead of wav2lip for a while — wav2lip works at 96x96
and upscales, which is the soft mouth on a 1080p reel — and `engine: "auto"` has
preferred them whenever there was a base clip. But nothing ever installed them, so
every selection ended at "neither on PATH nor installed", and the generic call the
presenter did make would not have worked either: LatentSync requires its checkpoint
and UNet config, and MuseTalk takes a YAML task list and writes into a result
directory rather than taking three paths.

What is asserted here: the specs name real files, the invocations match each
upstream CLI, availability never lies and never downloads, and the presenter asks
this module rather than guessing from the directory.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party"


@pytest.fixture
def le():
    from codai.api import lipsync_engines
    return lipsync_engines


# ── the vendored source ──────────────────────────────────────────────────────

@pytest.mark.parametrize("engine,pkg", [("latentsync", "latentsync"),
                                        ("musetalk", "musetalk")])
def test_the_inference_source_is_in_the_repo(engine, pkg):
    """Vendored, not cloned: an image someone else pulls has to run without
    network access to a third party."""
    d = VENDOR / engine
    assert (d / "scripts" / "inference.py").is_file()
    assert (d / pkg).is_dir()


@pytest.mark.parametrize("engine", ["latentsync", "musetalk"])
def test_the_licence_and_the_pin_travel_with_it(engine):
    lic = VENDOR / f"{engine.upper()}-LICENSE"
    pin = VENDOR / f"{engine.upper()}-PINNED"
    assert lic.is_file() and lic.stat().st_size > 1000
    text = pin.read_text(encoding="utf-8")
    assert "github.com" in text
    # A commit, not a branch: a branch is not a pin.
    assert any(len(w) == 40 and all(c in "0123456789abcdef" for c in w)
               for w in text.split())


def test_no_training_scripts_came_along():
    """Only what inference reaches. Training pulls in another dependency set."""
    for engine in ("latentsync", "musetalk"):
        scripts = sorted(p.name for p in (VENDOR / engine / "scripts").iterdir())
        assert scripts == ["inference.py"], (engine, scripts)


def test_latentsync_keeps_the_config_its_entrypoint_reads():
    """scripts/inference.py does DDIMScheduler.from_pretrained("configs") and
    picks the whisper checkpoint from cross_attention_dim in the UNet config, so
    both files have to be present for a run to start at all."""
    cfg = VENDOR / "latentsync" / "configs"
    assert (cfg / "scheduler_config.json").is_file()
    assert (cfg / "unet" / "stage2_512.yaml").is_file()


# ── the specs ────────────────────────────────────────────────────────────────

def test_both_engines_are_managed(le):
    assert set(le.MANAGED) == {"latentsync", "musetalk"}


def test_an_unknown_engine_says_so(le):
    with pytest.raises(ValueError) as exc:
        le.spec("hallo")
    assert "latentsync" in str(exc.value) and "musetalk" in str(exc.value)


@pytest.mark.parametrize("engine", ["latentsync", "musetalk"])
def test_the_vendor_directory_a_spec_names_exists(le, engine):
    assert le.vendor_dir(engine).is_dir()


@pytest.mark.parametrize("engine", ["latentsync", "musetalk"])
def test_the_requirements_file_a_spec_names_exists(le, engine):
    """A pip dep belongs in a requirements file, not only in code."""
    assert (ROOT / le.spec(engine).requirements).is_file()


@pytest.mark.parametrize("engine", ["latentsync", "musetalk"])
def test_the_entrypoint_a_spec_names_is_vendored(le, engine):
    sp = le.spec(engine)
    assert (le.vendor_dir(engine) / (sp.entrypoint.replace(".", "/") + ".py")).is_file()


def test_no_weight_set_depends_on_a_google_drive_id(le):
    """Upstream MuseTalk pulls the BiSeNet face-parsing checkpoint with gdown.
    Every file coderai fetches comes from the Hub, so there is one provenance
    and one auth story."""
    for engine in le.MANAGED:
        for ws in le.spec(engine).weights:
            assert "/" in ws.repo_id and " " not in ws.repo_id
            assert ws.patterns


def test_every_ready_marker_is_either_vendored_or_downloaded(le):
    """A marker that nothing provides would make an install permanently
    incomplete, and the failure would look like a broken download."""
    for engine in le.MANAGED:
        sp = le.spec(engine)
        provided = {m for m in sp.ready_markers
                    if (le.vendor_dir(engine) / m).exists()}
        for ws in sp.weights:
            for pat in ws.patterns:
                rel = f"{ws.dest}/{pat}" if ws.dest else pat
                provided |= {m for m in sp.ready_markers if m == rel}
        assert set(sp.ready_markers) == provided, (engine,
                                                   set(sp.ready_markers) - provided)


def test_the_interpreter_is_not_the_servers(le):
    """Both pin stacks the server cannot host; sharing its venv is the bug this
    module exists to avoid."""
    import sys
    for engine in le.MANAGED:
        assert le.spec(engine).py_version != f"{sys.version_info[0]}.{sys.version_info[1]}"


# ── the invocations ──────────────────────────────────────────────────────────

def test_latentsync_is_given_the_checkpoint_and_config_it_demands(le, tmp_path):
    """Both are `required=True` upstream, so the three-flag call the presenter
    used to make could only ever have exited non-zero."""
    argv, _ = le.spec("latentsync").argv(tmp_path, "/in.mp4", "/a.wav", "/o.mp4", {})
    assert "--inference_ckpt_path" in argv
    assert "--unet_config_path" in argv
    assert argv[argv.index("--video_path") + 1] == "/in.mp4"
    assert argv[argv.index("--audio_path") + 1] == "/a.wav"
    assert argv[argv.index("--video_out_path") + 1] == "/o.mp4"


def test_latentsync_defaults_to_the_config_whose_whisper_is_shipped(le, tmp_path):
    """cross_attention_dim 768 makes the pipeline look for whisper/small.pt,
    which is not in the weights repo; stage2_512 is the 384 one."""
    argv, _ = le.spec("latentsync").argv(tmp_path, "/in.mp4", "/a.wav", "/o.mp4", {})
    assert argv[argv.index("--unet_config_path") + 1] == "configs/unet/stage2_512.yaml"


def test_latentsync_options_reach_the_command(le, tmp_path):
    argv, _ = le.spec("latentsync").argv(
        tmp_path, "/in.mp4", "/a.wav", "/o.mp4",
        {"steps": 30, "guidance_scale": 2.0, "seed": 7, "deepcache": False})
    assert argv[argv.index("--inference_steps") + 1] == "30"
    assert argv[argv.index("--guidance_scale") + 1] == "2.0"
    assert argv[argv.index("--seed") + 1] == "7"
    assert "--enable_deepcache" not in argv


def test_musetalk_gets_a_task_yaml_rather_than_paths(le, tmp_path):
    argv, _ = le.spec("musetalk").argv(tmp_path, "/in.mp4", "/a.wav", "/o.mp4", {})
    assert "--video_path" not in argv
    cfg = pathlib.Path(argv[argv.index("--inference_config") + 1])
    assert cfg.is_file()
    body = cfg.read_text(encoding="utf-8")
    assert "task_0:" in body and "/in.mp4" in body and "/a.wav" in body


def test_musetalk_v15_is_the_default_and_names_its_own_checkpoint(le, tmp_path):
    """The two versions keep differently-named weights in different directories,
    so the version and the path cannot be chosen independently."""
    argv, _ = le.spec("musetalk").argv(tmp_path, "/in.mp4", "/a.wav", "/o.mp4", {})
    assert argv[argv.index("--version") + 1] == "v15"
    assert argv[argv.index("--unet_model_path") + 1] == "models/musetalkV15/unet.pth"
    argv, _ = le.spec("musetalk").argv(tmp_path, "/in.mp4", "/a.wav", "/o.mp4",
                                       {"version": "v1"})
    assert argv[argv.index("--unet_model_path") + 1] == "models/musetalk/pytorch_model.bin"


def test_musetalk_refuses_a_version_it_has_no_weights_for(le, tmp_path):
    with pytest.raises(ValueError):
        le.spec("musetalk").argv(tmp_path, "/in.mp4", "/a.wav", "/o.mp4",
                                 {"version": "v2"})


def test_musetalk_moves_its_result_to_where_the_caller_asked(le, tmp_path):
    """It writes into --result_dir under a name of its own choosing, which is not
    the path the compositor is going to look for."""
    out = tmp_path / "wanted.mp4"
    argv, finish = le.spec("musetalk").argv(tmp_path, "/in.mp4", "/a.wav",
                                            str(out), {})
    result_dir = pathlib.Path(argv[argv.index("--result_dir") + 1])
    (result_dir / "task_0").mkdir(parents=True)
    (result_dir / "task_0" / "whatever.mp4").write_bytes(b"\x00mp4")
    assert finish() == str(out)
    assert out.read_bytes() == b"\x00mp4"


def test_a_run_that_produced_nothing_is_not_reported_as_success(le, tmp_path):
    argv, finish = le.spec("musetalk").argv(tmp_path, "/in.mp4", "/a.wav",
                                            str(tmp_path / "o.mp4"), {})
    with pytest.raises(FileNotFoundError):
        finish()


# ── availability ─────────────────────────────────────────────────────────────

def test_availability_is_false_and_silent_for_a_missing_install(le, monkeypatch,
                                                                tmp_path):
    """The question a compose job asks has to be cheap and must not download."""
    monkeypatch.setattr(le, "install_dir", lambda n: tmp_path / n)
    assert le.is_installed("latentsync") is False
    assert le.is_available("latentsync") is False


def test_a_partial_install_does_not_count_as_installed(le, monkeypatch, tmp_path):
    """The old presenter check was "an entrypoint and any .pt anywhere", which
    read an interrupted download as a working engine."""
    root = tmp_path / "latentsync"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "inference.py").write_text("x")
    (root / "checkpoints").mkdir()
    (root / "checkpoints" / "latentsync_unet.pt").write_bytes(b"partial")
    monkeypatch.setattr(le, "install_dir", lambda n: root)
    assert le.is_installed("latentsync") is False
    missing = le.missing_pieces("latentsync")
    assert "checkpoints/whisper/tiny.pt" in missing


def test_a_run_without_an_install_says_what_is_missing(le, monkeypatch, tmp_path):
    monkeypatch.setattr(le, "install_dir", lambda n: tmp_path / n)
    with pytest.raises(RuntimeError) as exc:
        le.run("latentsync", "/in.mp4", "/a.wav", "/o.mp4")
    msg = str(exc.value)
    assert "not installed" in msg
    assert "latentsync_unet.pt" in msg
    # And it must not have tried to fetch anything to find that out.
    assert not (tmp_path / "latentsync" / "checkpoints").exists()


def test_an_unknown_engine_is_simply_not_available(le):
    assert le.is_installed("hallo") is False


# ── the presenter asks this module ───────────────────────────────────────────

def test_the_presenter_defers_to_the_managed_check(monkeypatch):
    from codai.api import lipsync_engines
    from codai.compose import presenter
    seen = []

    def _fake(name):
        seen.append(name)
        return True

    monkeypatch.setattr(lipsync_engines, "is_installed", _fake)
    assert presenter._managed_engine_ready("latentsync") is True
    assert seen == ["latentsync"]


def test_an_unmanaged_engine_still_uses_the_directory_heuristic(monkeypatch,
                                                                tmp_path):
    """An operator's hand-rolled Hallo install stays selectable."""
    from codai.compose import presenter
    d = tmp_path / "hallo"
    d.mkdir()
    (d / "inference.py").write_text("x")
    (d / "w.pth").write_bytes(b"x")
    monkeypatch.setattr(presenter, "_managed_engine_dir", lambda n: d)
    assert presenter._managed_engine_ready("hallo") is True


def test_the_presenter_still_ranks_them_ahead_of_wav2lip():
    from codai.compose import presenter
    assert presenter._AUTO_WITH_BASE.index("latentsync") < \
        presenter._AUTO_WITH_BASE.index("wav2lip")
    assert presenter._AUTO_WITH_BASE.index("musetalk") < \
        presenter._AUTO_WITH_BASE.index("wav2lip")
