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


# ── the install environment ──────────────────────────────────────────────────

def test_a_package_that_cannot_build_isolated_is_named_one_at_a_time(le):
    """mmpose depends on chumpy, whose setup.py does `import pip` — which an
    isolated build environment has not got. Disabling isolation wholesale would
    let every other package build against a half-installed venv, so the
    exception is per package."""
    assert le.spec("musetalk").no_build_isolation == ("chumpy",)
    assert le.spec("latentsync").no_build_isolation == ()


def test_the_uv_install_seeds_build_tools_only_when_something_needs_them(le):
    """A uv venv ships no pip, so a package building against it finds none."""
    import inspect
    src = inspect.getsource(le._install_requirements)
    seed = src[src.index("if sp.no_build_isolation:"):]
    assert '"pip", "setuptools", "wheel"' in seed
    assert "--no-build-isolation-package" in seed


def test_large_wheels_get_a_timeout_that_fits_them(le):
    """uv's default aborts mid-unzip on a slow third-party index, and reports it
    as a corrupt wheel rather than a timeout."""
    import inspect
    src = inspect.getsource(le._install_requirements)
    assert "UV_HTTP_TIMEOUT" in src


def test_musetalk_keeps_the_runtime_pins_that_look_like_cruft():
    """Three pins in requirements-musetalk.txt invite a tidy-up and each one
    breaks the engine at import time if it goes.

    mmcv 2.1.0 (not 2.2.0) because mmdet asserts mmcv < 2.2.0 and openmmlab
    publishes a cp311 2.1.0 wheel on exactly one index; torch 2.1.2 because that
    index is cu121/torch2.1; setuptools < 81 because mmengine still imports
    pkg_resources, which 81 removed.
    """
    req = (ROOT / "requirements-musetalk.txt").read_text(encoding="utf-8")
    assert "mmcv==2.1.0" in req
    assert "mmdet==3.3.0" in req
    assert "torch==2.1.2+cu121" in req
    assert "setuptools<81" in req
    assert "cu121/torch2.1/index.html" in req


def test_the_probe_imports_the_submodule_that_actually_fails(le):
    """`import mmpose` succeeds without mmdet; `import mmpose.apis` is what
    reaches rtmo_head and dies. Probing the package would call a venv that
    cannot run the engine usable."""
    assert "mmpose.apis" in le.spec("musetalk").probe_imports
    assert "mmpose" not in le.spec("musetalk").probe_imports


def test_the_venv_probe_runs_where_the_engine_runs(le):
    """face_detection is vendored under musetalk/utils, not installed, so a
    bare-interpreter probe reports a complete install as broken."""
    import inspect
    src = inspect.getsource(le._pip_ok)
    assert "_engine_env" in src
    assert "cwd=" in src


def test_a_failed_run_keeps_its_evidence_but_not_for_ever(le, tmp_path):
    """The frames and the task file are what you read to find out why a run
    failed, so a failure leaves them; a day later they are only clutter in the
    install root."""
    import os
    import time
    parent = tmp_path / "coderai-runs"
    old = parent / "old"
    recent = parent / "recent"
    for d in (old, recent):
        d.mkdir(parents=True)
    os.utime(old, (time.time() - 48 * 3600, time.time() - 48 * 3600))
    assert le._sweep_run_dirs(parent) == 1
    assert recent.is_dir() and not old.exists()


def test_the_sweep_never_removes_the_run_being_set_up(le, tmp_path):
    import os
    import time
    parent = tmp_path / "coderai-runs"
    mine = parent / "mine"
    mine.mkdir(parents=True)
    os.utime(mine, (time.time() - 48 * 3600, time.time() - 48 * 3600))
    assert le._sweep_run_dirs(parent, keep=mine) == 0
    assert mine.is_dir()


def test_the_sweep_is_silent_when_there_is_nothing_to_sweep(le, tmp_path):
    assert le._sweep_run_dirs(tmp_path / "never-made") == 0


# ── management: an operator's surface, not a python -c ───────────────────────

def test_an_engine_reports_everything_the_ui_needs_without_the_network(le,
                                                                       monkeypatch,
                                                                       tmp_path):
    monkeypatch.setattr(le, "install_dir", lambda n: tmp_path / n)
    monkeypatch.setattr(le, "venv_dir", lambda n: tmp_path / (n + "_venv"))
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("status touched the network"))
    s = le.install_state("latentsync")
    assert s["name"] == "latentsync"
    assert s["installed"] is False and s["available"] is False
    assert s["state"] == "idle"
    assert s["python"] == "3.11"
    assert any("LatentSync" in w for w in s["weights"])
    assert "checkpoints/latentsync_unet.pt" in s["missing"]


def test_every_managed_engine_has_a_state(le):
    assert {s["name"] for s in le.all_states()} == set(le.MANAGED)


def test_an_install_runs_in_the_background_and_reports_as_it_goes(le, monkeypatch):
    import threading
    released = threading.Event()

    def _slow(name, log=None):
        log("working")
        released.wait(5)
        log("finished")
        return ("root", "py")

    monkeypatch.setattr(le, "ensure_installed", _slow)
    monkeypatch.setattr(le, "_installs", {}, raising=False)
    st = le.start_install("latentsync")
    assert st["state"] == "running"
    # Asking again joins the running job rather than starting a second.
    assert le.start_install("latentsync")["state"] == "running"
    released.set()
    for _ in range(100):
        if le.install_state("latentsync")["state"] == "done":
            break
        import time as _t
        _t.sleep(0.05)
    done = le.install_state("latentsync")
    assert done["state"] == "done"
    assert "working" in done["log"] and "finished" in done["log"]


def test_a_failed_install_keeps_the_reason_where_it_can_be_read(le, monkeypatch):
    def _boom(name, log=None):
        log("fetching")
        raise RuntimeError("no wheel for this host")

    monkeypatch.setattr(le, "ensure_installed", _boom)
    monkeypatch.setattr(le, "_installs", {}, raising=False)
    le.start_install("musetalk")
    for _ in range(100):
        if le.install_state("musetalk")["state"] in ("failed", "done"):
            break
        import time as _t
        _t.sleep(0.05)
    s = le.install_state("musetalk")
    assert s["state"] == "failed"
    assert "no wheel for this host" in s["error"]
    assert any("fetching" in line for line in s["log"])


def test_the_state_never_carries_a_thread_object(le, monkeypatch):
    """It is serialised to JSON for the admin page."""
    import json
    monkeypatch.setattr(le, "ensure_installed", lambda name, log=None: ("r", "p"))
    monkeypatch.setattr(le, "_installs", {}, raising=False)
    le.start_install("latentsync")
    json.dumps(le.install_state("latentsync"))      # must not raise


def test_removing_an_engine_takes_the_venv_and_the_weights(le, monkeypatch,
                                                            tmp_path):
    root, venv = tmp_path / "latentsync", tmp_path / "latentsync_venv"
    (root / "checkpoints").mkdir(parents=True)
    (root / "checkpoints" / "latentsync_unet.pt").write_bytes(b"w")
    venv.mkdir()
    monkeypatch.setattr(le, "install_dir", lambda n: root)
    monkeypatch.setattr(le, "venv_dir", lambda n: venv)
    monkeypatch.setattr(le, "_installs", {}, raising=False)
    le.uninstall("latentsync")
    assert not root.exists() and not venv.exists()


def test_rebuilding_a_venv_keeps_the_gigabytes(le, monkeypatch, tmp_path):
    """The weights are the slow part and a dependency change does not alter
    them, so a venv rebuild must not re-download 6 GB."""
    root, venv = tmp_path / "latentsync", tmp_path / "latentsync_venv"
    (root / "checkpoints").mkdir(parents=True)
    kept = root / "checkpoints" / "latentsync_unet.pt"
    kept.write_bytes(b"w")
    (root / "scripts").mkdir()
    (root / "scripts" / "inference.py").write_text("x")
    le._seed_stamp(root).write_text("stamp")
    venv.mkdir()
    monkeypatch.setattr(le, "install_dir", lambda n: root)
    monkeypatch.setattr(le, "venv_dir", lambda n: venv)
    monkeypatch.setattr(le, "_installs", {}, raising=False)
    le.uninstall("latentsync", keep_weights=True)
    assert kept.exists(), "the weights were deleted on a venv rebuild"
    assert not venv.exists()
    assert not (root / "scripts").exists(), "the code should be re-seeded"
    assert not le._seed_stamp(root).exists()


def test_an_engine_cannot_be_removed_while_it_installs(le, monkeypatch):
    monkeypatch.setattr(le, "_installs", {"latentsync": {"state": "running"}},
                        raising=False)
    with pytest.raises(RuntimeError) as e:
        le.uninstall("latentsync")
    assert "being installed" in str(e.value)


def test_the_admin_routes_exist_and_are_admin_only():
    import inspect
    from codai.admin import routes
    src = inspect.getsource(routes)
    for path in ("/admin/api/lipsync-engines",
                 "/admin/api/lipsync-engine-install",
                 "/admin/api/lipsync-engine-remove"):
        assert path in src, path
    # Each one behind require_admin, like every other model action.
    tail = src[src.index("/admin/api/lipsync-engines"):]
    assert tail.count("Depends(require_admin)") >= 3


def test_the_models_page_offers_the_engines():
    page = (ROOT / "codai" / "admin" / "templates" / "models.html").read_text(
        encoding="utf-8")
    assert 'id="tab-lipsync"' in page
    assert "loadLipsyncEngines" in page
    assert "lipsync-engine-install" in page
    assert "lipsync-engine-remove" in page


# ── VRAM: an engine is a model, and the ledger has to say so ─────────────────

def test_both_engines_are_carded_like_any_other_model():
    """Not a special case: a row in video_models, so the models page configures
    them and the manager accounts for them."""
    import json
    import pathlib
    p = pathlib.Path.home() / ".coderai" / "models.json"
    if not p.is_file():
        pytest.skip("no local models.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    by = {m.get("alias"): m for m in d.get("video_models", [])
          if isinstance(m, dict)}
    for name in ("latentsync", "musetalk"):
        c = by.get(name)
        assert c, f"{name} has no card"
        assert c["backend"] == "lipsync"
        assert "lip_sync" in c["capabilities"]
        assert float(c["used_vram_gb"]) > 0


def test_the_reservation_prefers_the_card_then_the_measurement(le, monkeypatch):
    monkeypatch.setattr(le, "card", lambda n: {"used_vram_gb": 12.5})
    assert le.reserved_vram_gb("latentsync") == 12.5
    monkeypatch.setattr(le, "card", lambda n: {})
    import sys
    import types
    mod = types.ModuleType("codai.models.manager")
    mod.multi_model_manager = types.SimpleNamespace(
        _measured_vram_gb={"video:latentsync": 21.0})
    monkeypatch.setitem(sys.modules, "codai.models.manager", mod)
    assert le.reserved_vram_gb("latentsync") == 21.0


def test_an_unmeasured_engine_falls_back_to_its_own_default(le, monkeypatch):
    import sys
    import types
    monkeypatch.setattr(le, "card", lambda n: {})
    mod = types.ModuleType("codai.models.manager")
    mod.multi_model_manager = types.SimpleNamespace(_measured_vram_gb={})
    monkeypatch.setitem(sys.modules, "codai.models.manager", mod)
    assert le.reserved_vram_gb("latentsync") == le.DEFAULT_ENGINE_VRAM_GB["latentsync"]
    assert le.reserved_vram_gb("musetalk") == le.DEFAULT_ENGINE_VRAM_GB["musetalk"]


def test_a_measurement_is_a_high_water_mark(le, monkeypatch):
    import sys
    import types
    m = types.SimpleNamespace(_measured_vram_gb={})
    mod = types.ModuleType("codai.models.manager")
    mod.multi_model_manager = m
    monkeypatch.setitem(sys.modules, "codai.models.manager", mod)
    le.record_measurement("latentsync", 17.0)
    le.record_measurement("latentsync", 9.0)       # a 2s clip is not a 30s one
    assert m._measured_vram_gb["video:latentsync"] == 17.0


def test_an_engine_disabled_on_the_models_page_is_not_offered(monkeypatch):
    from codai.api import lipsync_engines as le
    from codai.compose import presenter
    monkeypatch.setattr(le, "is_installed", lambda n: True)
    monkeypatch.setattr(le, "is_enabled", lambda n: False)
    assert presenter._managed_engine_ready("latentsync") is False
    monkeypatch.setattr(le, "is_enabled", lambda n: True)
    assert presenter._managed_engine_ready("latentsync") is True


def test_an_engine_with_no_card_is_not_treated_as_disabled(le, monkeypatch):
    monkeypatch.setattr(le, "card", lambda n: {})
    assert le.is_enabled("latentsync") is True


# ── an in-flight render is queued behind, never killed ───────────────────────

def test_cleanup_waits_for_the_render_rather_than_killing_it(le):
    """A half-finished lip-sync is worth nothing, and the request that triggered
    the eviction is not more important than the one already running."""
    import types
    killed = []
    waited = []
    proc = types.SimpleNamespace(
        poll=lambda: None,
        wait=lambda timeout=None: waited.append(timeout),
        terminate=lambda: killed.append("terminate"),
        kill=lambda: killed.append("kill"))
    h = le._EngineHandle("latentsync", proc)
    h.cleanup()
    assert killed == [], "an in-flight render was killed"
    assert waited and waited[0] >= 600


def test_cleanup_on_a_finished_render_does_nothing(le):
    import types
    calls = []
    proc = types.SimpleNamespace(poll=lambda: 0,
                                 wait=lambda timeout=None: calls.append(1),
                                 terminate=lambda: calls.append(2),
                                 kill=lambda: calls.append(3))
    le._EngineHandle("latentsync", proc).cleanup()
    assert calls == []


def test_a_render_marks_the_model_busy_so_eviction_queues_behind_it():
    """acquire_stt_backend pops the instance pool, which is where _is_key_busy
    looks — so without this an engine mid-render reads as idle and eviction
    takes the card out from under it."""
    from codai.models.manager import multi_model_manager as m
    key = "video:latentsync"
    assert m._is_key_busy(key) is False
    with m.external_busy(key):
        assert m._is_key_busy(key) is True
        with m.external_busy(key):            # nests
            assert m._is_key_busy(key) is True
        assert m._is_key_busy(key) is True
    assert m._is_key_busy(key) is False


def test_busy_is_cleared_even_when_the_render_raises():
    from codai.models.manager import multi_model_manager as m
    key = "video:musetalk"
    with pytest.raises(ValueError):
        with m.external_busy(key):
            raise ValueError("render failed")
    assert m._is_key_busy(key) is False


def test_the_run_drops_the_registration_when_it_is_over(le):
    """One-shot: when the subprocess exits its VRAM is gone, so the ledger must
    stop claiming it or the card looks permanently fuller than it is."""
    import inspect
    src = inspect.getsource(le._run_managed)
    assert "unload_model" in src
    assert "finally:" in src
