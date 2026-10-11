"""The stale-pod reaper must not kill a pod that was just created.

coderai runs a front plus an engine per GPU, each with its own pool registry and
its own reaper, so a pod one process created is an orphan in another's view until
the shared registry catches up. That race is already documented in the source,
from a live incident: "the nvidia engine created a pod and the radeon engine
terminated it four seconds later, then the reverse — a mutual kill loop that
rented and destroyed pods until stopped."

The defence against it is an age guard — "never reap a pod younger than this,
whatever the registry says". It was written as:

    age = _pod_age_seconds(p)
    if age is not None and age < REAP_GRACE_SECONDS:
        continue

and `list_pods()` returned {id, name, status, cost_per_hr, image} — no timestamp
of any kind. So `age` was None for every pod, the condition was never true, and
the backstop had never protected anything. It recurred on the Digesta
orchestrator, which has two config dirs and therefore two registries: pod created
19:24:56, reaped by a sibling 19:24:57, after which OCR waited out its 900s boot
timeout on a pod that no longer existed and the pool rented another.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                          # noqa: E402


def test_an_undatable_pod_is_left_alone():
    """The only state the guard was ever actually in. Not reaping costs one more
    cycle; reaping a live pod costs the work on it and bills in a loop."""
    assert rw._pod_age_seconds({"id": "p1", "name": "coderai-x-ocr"}) is None


def test_uptime_is_read_when_runpod_reports_it():
    assert rw._pod_age_seconds({"uptime_s": 12.0}) == 12.0


def test_the_grace_period_is_longer_than_any_boot():
    """A pod still pulling a 12 GB image must not look like a leftover: the
    image pull alone ran past 128s in the incident."""
    assert rw.REAP_GRACE_SECONDS >= 900


def test_the_listing_now_carries_an_age():
    """Without a timestamp in the query the guard can only ever abstain."""
    src = (ROOT / "codai/api/runpod_client.py").read_text()
    blk = src.split("def list_pods")[1].split("def ")[0]
    assert "uptimeInSeconds" in blk
    assert '"uptime_s"' in blk


def test_an_unknown_age_does_not_fall_through_to_termination():
    """The shape of the bug: `age is not None and age < GRACE` skipped the guard
    for exactly the pods it existed to protect."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    blk = src.split("def reap_orphans")[1]
    assert "if age is None:" in blk
    assert "age is not None and age < REAP_GRACE_SECONDS" not in blk


# ------------------------------------------- one registry, whoever is asking #
# The registry is the cross-process shield: a pod one process created must not
# look like an orphan to another's reaper. It is only a shield if every process
# resolves the SAME file.


def test_the_registry_falls_back_to_the_xdg_path_not_the_legacy_one(monkeypatch,
                                                                    tmp_path):
    """The container's launcher exports XDG_CONFIG_HOME and puts the real config
    dir at $CODERAI_CONFIG_DIR/coderai. A process without the config manager used
    to fall back to ~/.coderai — one directory up from where the config manager
    writes — so the two read different registries."""
    import sys as _sys
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # No config manager in this process, which is the case that diverged.
    monkeypatch.setitem(_sys.modules, "codai.admin.routes",
                        type(_sys)("codai.admin.routes"))
    rw._REGISTRY_PATH_LOGGED = True
    got = rw._pod_registry_path()
    assert got == str(tmp_path / "coderai" / "runpod-pods.json")


def test_the_config_manager_still_wins_when_it_is_there(monkeypatch, tmp_path):
    import sys as _sys
    import types
    fake = types.ModuleType("codai.admin.routes")
    fake.config_manager = types.SimpleNamespace(config_dir=str(tmp_path / "live"))
    monkeypatch.setitem(_sys.modules, "codai.admin.routes", fake)
    rw._REGISTRY_PATH_LOGGED = True
    assert rw._pod_registry_path() == str(tmp_path / "live" / "runpod-pods.json")


def test_the_resolved_path_is_logged_so_a_split_is_visible(capsys, monkeypatch,
                                                           tmp_path):
    """A registry that silently splits is invisible until it costs a GPU."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    rw._REGISTRY_PATH_LOGGED = False
    rw._pod_registry_path()
    assert "pod registry:" in capsys.readouterr().out
