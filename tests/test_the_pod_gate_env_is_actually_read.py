"""The admission gate a pod is sent has to be read BY the pod.

`pod_plan` has emitted `CODERAI_SERVER_MAX_PARALLEL_REQUESTS` and
`..._QUEUE_MAX_SIZE` since pods shipped, with a comment saying that without
them "the pod runs the shared-workstation defaults and its admission gate, not
its GPU, is the ceiling". They were read in `codai/main.py`.

A pod never runs `codai/main.py`. Its boot.sh execs
`python -m uvicorn codai.api.app:app`, so both variables were set on every
rented pod and then ignored: the defaults of 2 in flight / 6 queued stood, and
the gate the comment describes was never applied to the machine it was written
for.

So the read moved into `ConfigManager`, which runs under BOTH entry points.
These tests pin that, and pin the boot path that makes it necessary -- if a pod
ever does start going through main.py, the second test says so rather than
leaving two readers to disagree.
"""
import os
import pathlib
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(env):
    from codai.config import ConfigManager
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        cm = ConfigManager(tempfile.mkdtemp())
        cm.load()
        return cm
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- 1. the gate reaches a pod ------------------------------------------

def test_the_parallel_gate_env_is_applied_by_the_config_manager():
    cm = _load({"CODERAI_SERVER_MAX_PARALLEL_REQUESTS": "192",
                "CODERAI_SERVER_QUEUE_MAX_SIZE": "512"})
    assert cm.config.server.max_parallel_requests == 192
    assert cm.config.server.queue_max_size == 512


def test_the_admission_policy_env_is_applied_by_the_config_manager():
    cm = _load({"CODERAI_ADMISSION_CONTROL": "on",
                "CODERAI_ADMISSION_KV_TARGET_PCT": "0.7",
                "CODERAI_ADMISSION_MIN_PARALLEL": "16",
                "CODERAI_ADMISSION_EXPECTED_TOKENS": "14336"})
    s = cm.config.server
    assert s.admission_control == "on"
    assert s.admission_kv_target_pct == pytest.approx(0.7)
    assert s.admission_min_parallel == 16
    assert s.admission_expected_tokens == 14336


def test_a_bad_value_leaves_the_setting_alone():
    cm = _load({"CODERAI_SERVER_MAX_PARALLEL_REQUESTS": "lots"})
    assert cm.config.server.max_parallel_requests == 2


def test_an_absent_variable_changes_nothing():
    cm = _load({})
    assert cm.config.server.admission_control == "auto"


# --- 2. why it cannot live in main.py -----------------------------------

def test_a_pod_boots_uvicorn_on_the_app_and_not_main():
    boot = (ROOT / "packaging/runpod/boot.sh").read_text()
    assert "uvicorn codai.api.app:app" in boot
    assert "codai/main.py" not in boot and "codai.main" not in boot, (
        "a pod now runs main.py, so the env read there applies to pods too — "
        "reconcile the two readers rather than leaving them to disagree")


def test_the_controller_starts_in_the_lifespan_not_only_in_main():
    app = (ROOT / "codai/api/app.py").read_text()
    assert "admission.start(" in app, (
        "started only in main.py means never started on a pod")


# --- 3. the policy travels to a pod -------------------------------------

def test_only_an_explicitly_set_policy_travels():
    """An unset knob must arrive ABSENT, not as today's default restated: a
    pod pinned to a default it was never configured with would not follow a
    later change to that default."""
    from codai.api import runpod_worker

    class _Srv:
        admission_control = "on"
        admission_min_parallel = 16
        admission_kv_target_pct = 0.7
        admission_kv_ceiling_pct = 0.95
        admission_interval_s = 5.0
        admission_backoff = 0.75
        admission_recover_after = 3
        admission_expected_tokens = 0

    class _Cfg:
        server = _Srv()

    class _CM:
        config = _Cfg()
        explicit_server_keys = {"admission_control", "admission_min_parallel"}

    import codai.admin.routes as ar
    old = getattr(ar, "config_manager", None)
    ar.config_manager = _CM()
    try:
        env = runpod_worker._pod_admission_env(None)
    finally:
        ar.config_manager = old
    assert env == {"CODERAI_ADMISSION_CONTROL": "on",
                   "CODERAI_ADMISSION_MIN_PARALLEL": "16"}


def test_no_config_manager_sends_nothing_rather_than_failing():
    from codai.api import runpod_worker
    import codai.admin.routes as ar
    old = getattr(ar, "config_manager", None)
    ar.config_manager = None
    try:
        assert runpod_worker._pod_admission_env(None) == {}
    finally:
        ar.config_manager = old


def test_every_forwarded_name_is_one_the_config_manager_reads():
    """The two lists are in different files; a name in one and not the other
    is a setting that travels and is then ignored, which is the whole bug this
    file exists for."""
    from codai.api.runpod_worker import _POD_ADMISSION_ENV
    cfg_src = (ROOT / "codai/config.py").read_text()
    for _attr, name, _cast in _POD_ADMISSION_ENV:
        assert f'"{name}"' in cfg_src, f"{name} is sent to pods but never read"


# --- 4. the measurement the controller reads ----------------------------

def test_the_boot_log_concurrency_line_is_captured():
    from codai.api.vllm_worker import _MAX_CONCURRENCY
    line = ("INFO 11-06 08:00:00 [kv_cache_utils.py:1] Maximum concurrency "
            "for 32,768 tokens per request: 12.33x")
    m = _MAX_CONCURRENCY.search(line)
    assert m and m.group(1) == "32,768" and m.group(2) == "12.33"


def test_kv_capacity_is_empty_when_no_engine_has_booted():
    """{} and not a zero: a caller must be able to tell "not measured" from
    "measured as nothing", because the second would veto every admission."""
    from codai.api import vllm_worker
    got = vllm_worker.kv_capacity()
    assert got == {}


def test_engine_state_reports_the_controller():
    app = (ROOT / "codai/api/app.py").read_text()
    i = app.find('"/internal/engine-state"')
    assert i > 0, "the endpoint the orchestrator polls is gone"
    body = app[i:app.find("\n@app.", i + 10)]
    assert '"admission": admission' in body, (
        "the controller's state must travel on the endpoint the orchestrator "
        "already polls, or the limit is only visible in the pod's own log")
