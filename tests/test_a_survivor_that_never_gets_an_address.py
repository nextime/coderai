"""A pod nothing can reach must not bill unowned.

`register_pod` writes a pod's url only once it is ready, so a restart landing
inside a boot leaves a registry entry with no address. `_recover_pod_url` asks
RunPod for one, which works the moment a port is published -- observed twice on
2026-11-09, both pods adopted ~10 minutes after a restart:

    [runpod] pod mdhsihuuvzhj8z was mid-boot when its owner stopped;
             recovered its address from RunPod: https://195.26.233.95:25768

But when the port never appears -- the failure that burned nine pods earlier
that night, hosts that publish only RunPod's own agent port -- the old code
printed "not adopting" on every pass and left the pod running. Nothing owned
it, so nothing would tear it down until the reaper's 20-minute grace, and the
pool rented a replacement beside it in the meantime: double billing for a pod
that could never serve.

The pool's own boot window now decides, exactly as it would for a pod this
process had rented itself.
"""
import time
import types

import pytest

from codai.api import runpod_worker as rw


def _registry(monkeypatch, entries):
    monkeypatch.setattr(rw, "_registry_update", lambda _fn: dict(entries))
    monkeypatch.setattr(rw, "_is_this_process", lambda _info: False)


def _entry(created, **over):
    e = {"pool": "pool:test", "proc": "someone-else", "created": created,
         "at": created, "port": 8000, "hourly_usd": 0.59}
    e.update(over)
    return e


@pytest.fixture
def killed(monkeypatch):
    """Record what the code would terminate, without touching RunPod."""
    seen = []
    monkeypatch.setattr(rw, "_terminate_unreachable_survivor",
                        lambda pid: seen.append(str(pid)) or True)
    monkeypatch.setattr(rw, "_recover_pod_url", lambda pid, info: "")
    return seen


def test_a_survivor_past_its_boot_window_is_torn_down(killed, monkeypatch, capsys):
    _registry(monkeypatch, {"deadpod": _entry(time.time() - 4000)})
    got = rw.find_shared_pod("pool:test", boot_window_s=3030.0)
    assert got == (None, "", "", 0.0)
    assert killed == ["deadpod"]
    out = capsys.readouterr().out
    assert "never published an address" in out
    assert "bill unowned" in out


def test_a_survivor_still_inside_its_window_is_left_alone(killed, monkeypatch, capsys):
    """Both pods that behaved this way on 2026-11-09 were worth waiting for."""
    _registry(monkeypatch, {"bootingpod": _entry(time.time() - 300)})
    got = rw.find_shared_pod("pool:test", boot_window_s=3030.0)
    assert got == (None, "", "", 0.0)
    assert killed == [], "a pod 300s into a 3030s boot must not be killed"
    assert "still booting when its owner stopped" in capsys.readouterr().out


def test_an_unknown_boot_window_never_kills(killed, monkeypatch):
    """Fail closed: without a window there is no deadline to be past.

    A caller that does not pass one (anything but the pool) must not acquire
    the power to terminate pods as a side effect.
    """
    _registry(monkeypatch, {"oldpod": _entry(time.time() - 99999)})
    rw.find_shared_pod("pool:test")
    assert killed == []


def test_an_entry_with_no_timestamp_never_kills(killed, monkeypatch):
    """An unknown age is not an expired one -- the same trap the reaper hit
    when `age is None` skipped its grace check entirely and a one-second-old
    pod was reaped as a leftover."""
    _registry(monkeypatch, {"nodate": _entry(0)})
    rw.find_shared_pod("pool:test", boot_window_s=600.0)
    assert killed == []


def test_a_recoverable_address_is_used_rather_than_killed(monkeypatch):
    """If RunPod has published a port, the pod is adopted, not torn down."""
    seen = []
    monkeypatch.setattr(rw, "_terminate_unreachable_survivor",
                        lambda pid: seen.append(pid) or True)
    monkeypatch.setattr(rw, "_recover_pod_url",
                        lambda pid, info: "https://1.2.3.4:25768")
    monkeypatch.setattr(rw, "_pod_health_ok",
                        lambda url, path="", api_key="", timeout=0: True)
    _registry(monkeypatch, {"latepod": _entry(time.time() - 99999)})
    pid, url, _key, rate = rw.find_shared_pod("pool:test", boot_window_s=600.0)
    assert pid == "latepod" and url == "https://1.2.3.4:25768"
    assert rate == pytest.approx(0.59)
    assert seen == [], "an addressable pod must never be terminated here"


def test_the_pool_passes_its_own_window(monkeypatch):
    """The window must come from the pool, not a constant.

    Passed positionally once, it landed in `api_key` and silently shifted every
    argument -- so this pins that it arrives as the keyword.
    """
    captured = {}

    def _fake(pool_key, health_path="/v1/models", api_key="", image="",
              gpus=None, exclude=None, boot_window_s=0.0):
        captured["window"] = boot_window_s
        return None, "", "", 0.0

    monkeypatch.setattr(rw, "find_shared_pod", _fake)
    monkeypatch.setattr(rw, "_share",
                        types.SimpleNamespace(is_follower=lambda: False))
    monkeypatch.setattr(rw, "pod_plan",
                        lambda *a, **k: {"image": "img:latest"})
    monkeypatch.setattr(rw, "narrow_gpu_choice",
                        lambda *a, **k: {"gpu_types": [], "allow_other_gpus": False})

    stub = types.SimpleNamespace(
        mcfg=types.SimpleNamespace(boot_timeout_s=1200, load_timeout_s=900,
                                   warmup_timeout_s=900, warmup_on_boot=True,
                                   gpu_types=[], allow_other_gpus=False),
        model_key="pool:test", served="m", entry={}, seed_entries=None,
        api_key="", health_path="/v1/models", pods=[],
        _cv=_NullCV(),
        client_gpu_narrow=lambda: {},
        boot_window_s=lambda: rw.RunpodPodPool.boot_window_s(stub),
        PORT_TIMEOUT_S=rw.RunpodPodPool.PORT_TIMEOUT_S,
        HEALTH_TIMEOUT_S=rw.RunpodPodPool.HEALTH_TIMEOUT_S,
        BOOT_DEADLINE_SLACK_S=rw.RunpodPodPool.BOOT_DEADLINE_SLACK_S)
    monkeypatch.setattr(rw, "_model_path_for", lambda _k: "")

    assert rw.RunpodPodPool._adopt_shared_pod(stub) is False
    assert captured["window"] == pytest.approx(1200 + 900 + 900 + 30)


class _NullCV:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
