"""A restart should pick up the fleet it already paid for.

Adoption existed but ran only inside `acquire()`, when a request needed a pod.
Nothing reconciled the registry at boot, so after every restart the pods this
deployment was already paying for sat unowned until a request happened to want
that pool or the reaper's 20-minute grace expired. Observed 2026-10-16: a
forced upgrade at 00:01 orphaned three pods; at 00:04 the surya pool rented a
NEW pod while a healthy surya pod from before the restart was still running and
still in the registry; the three were reaped at 00:21 having billed $1.11
between them, "not tracked by any pool and its owner stopped reporting".

And a pod that was still BOOTING when its owner died could never be adopted at
all: register_pod writes the url only once the pod is ready, and
find_shared_pod skipped a url-less entry in silence. These images take 6-14
minutes to boot, so a restart has a good chance of landing inside one.
"""

import types

import pytest

from codai.api import runpod_worker as rw


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """A registry backed by a dict, not a file."""
    store = {}

    def _update(fn):
        fn(store)
        return dict(store)

    monkeypatch.setattr(rw, "_registry_update", _update)
    return store


def _entry(pool="m", url="https://1.2.3.4:1", image="img", gpu="A40", key="k"):
    return {"pool": pool, "url": url, "image": image, "gpu": gpu, "key": key,
            "pid": 1, "proc": "1-deadbeef", "hourly_usd": 0.5}


# ----------------------------------------------- a mid-boot pod is adoptable #
def test_a_pod_with_no_url_is_recovered_from_runpod(registry, monkeypatch):
    """The case that could never be adopted: killed mid-boot, url never written."""
    registry["p1"] = _entry(url="")
    monkeypatch.setattr(rw, "_recover_pod_url",
                        lambda pid, info: "https://5.6.7.8:9")
    monkeypatch.setattr(rw, "_pod_health_ok", lambda url, **kw: True)
    pid, url, key, rate = rw.find_shared_pod("m", image="img", gpus=["A40"])
    assert pid == "p1"
    assert url == "https://5.6.7.8:9"


def test_a_pod_that_runpod_cannot_place_is_refused_out_loud(registry, monkeypatch, capsys):
    registry["p1"] = _entry(url="")
    monkeypatch.setattr(rw, "_recover_pod_url", lambda pid, info: "")
    pid, url, _k, _r = rw.find_shared_pod("m", image="img", gpus=["A40"])
    assert pid is None
    out = capsys.readouterr().out
    assert "no address recorded" in out
    assert "still booting" in out


# --------------------------------------------- every refusal says why it refused #
def test_a_pod_that_does_not_answer_says_so(registry, monkeypatch, capsys):
    """Was silent. Renting beside a pod that did not answer looked identical in
    the log to renting because there was nothing to adopt."""
    registry["p1"] = _entry()
    monkeypatch.setattr(rw, "_pod_health_ok", lambda url, **kw: False)
    pid, _u, _k, _r = rw.find_shared_pod("m", image="img", gpus=["A40"])
    assert pid is None
    out = capsys.readouterr().out
    assert "did not answer" in out
    assert "p1" in out


def test_the_image_and_card_refusals_still_speak(registry, monkeypatch, capsys):
    registry["p1"] = _entry(image="other")
    registry["p2"] = _entry(gpu="L40S")
    monkeypatch.setattr(rw, "_pod_health_ok", lambda url, **kw: True)
    rw.find_shared_pod("m", image="img", gpus=["A40"])
    out = capsys.readouterr().out
    assert "not adopting pod p1" in out
    assert "not adopting pod p2" in out


def test_a_healthy_matching_pod_is_still_adopted(registry, monkeypatch):
    """The fix must not make adoption pickier than it was."""
    registry["p1"] = _entry()
    monkeypatch.setattr(rw, "_pod_health_ok", lambda url, **kw: True)
    pid, url, key, rate = rw.find_shared_pod("m", image="img", gpus=["A40"])
    assert (pid, url, key, rate) == ("p1", "https://1.2.3.4:1", "k", 0.5)


def test_our_own_pods_are_left_to_their_pool(registry, monkeypatch):
    registry["p1"] = dict(_entry(), proc=rw._PROC_ID)
    monkeypatch.setattr(rw, "_pod_health_ok", lambda url, **kw: True)
    assert rw.find_shared_pod("m", image="img", gpus=["A40"])[0] is None


# ------------------------------------------------------ reclaim at startup #
def _acct(enabled=True):
    return types.SimpleNamespace(enabled=enabled, api_key="k")


def test_startup_reclaims_a_pod_that_outlived_its_process(registry, monkeypatch):
    registry["p1"] = _entry(pool="mymodel")
    monkeypatch.setattr(rw, "_model_entry",
                        lambda k: {"id": "mymodel", "runpod": {"mode": "pods"}})
    adopted = []

    class _Pool:
        def _adopt_shared_pod(self):
            adopted.append(True)
            return True

    monkeypatch.setattr(rw, "get_pod_pool", lambda *a, **k: _Pool())
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "get_active_runpod_config", lambda: _acct())
    assert rw.reclaim_registered_pods() == 1
    assert adopted


def test_reclaim_runs_before_the_reaper_can_kill_them():
    """Order matters: a pod we are already paying for should rejoin its pool,
    not be terminated and rented again a minute later."""
    import pathlib
    src = pathlib.Path(rw.__file__).read_text()
    body = src.split("def _scaler_loop")[1].split("while True:")[0]
    assert body.index("reclaim_registered_pods") < body.index("reap_orphans")


def test_reclaim_leaves_a_shared_pool_alone(registry, monkeypatch, capsys):
    """A `pool:` key has no single model entry to rebuild its shape from."""
    registry["p1"] = _entry(pool="pool:shared")
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "get_active_runpod_config", lambda: _acct())
    assert rw.reclaim_registered_pods() == 0
    assert "leaving shared pool" in capsys.readouterr().out


def test_reclaim_says_when_a_model_is_gone(registry, monkeypatch, capsys):
    registry["p1"] = _entry(pool="retired-model")
    monkeypatch.setattr(rw, "_model_entry", lambda k: {})
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "get_active_runpod_config", lambda: _acct())
    assert rw.reclaim_registered_pods() == 0
    assert "no longer configures it" in capsys.readouterr().out


def test_reclaim_skips_our_own_entries(registry, monkeypatch):
    registry["p1"] = dict(_entry(pool="mymodel"), proc=rw._PROC_ID)
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "get_active_runpod_config", lambda: _acct())
    assert rw.reclaim_registered_pods() == 0


def test_reclaim_is_a_no_op_when_runpod_is_off(registry, monkeypatch):
    registry["p1"] = _entry(pool="mymodel")
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "get_active_runpod_config", lambda: _acct(enabled=False))
    assert rw.reclaim_registered_pods() == 0


def test_reclaim_survives_one_bad_pool(registry, monkeypatch):
    """One model's failure must not stop the others being reclaimed."""
    registry["p1"] = _entry(pool="bad")
    registry["p2"] = _entry(pool="good")
    monkeypatch.setattr(rw, "_model_entry", lambda k: {"id": k, "runpod": {}})

    def _pool(key, *a, **k):
        if key == "bad":
            raise RuntimeError("no config")
        return types.SimpleNamespace(_adopt_shared_pod=lambda: True)

    monkeypatch.setattr(rw, "get_pod_pool", _pool)
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "get_active_runpod_config", lambda: _acct())
    assert rw.reclaim_registered_pods() == 1


# ----------------------------------------------------- the whole fleet, not one #
def test_reclaim_takes_back_every_pod_of_a_pool(registry, monkeypatch):
    """A fleet of sixteen reclaimed ONE pod and left fifteen billing in the
    first version of this — the bug with extra steps."""
    for i in range(16):
        registry[f"p{i}"] = _entry(pool="mymodel")
    monkeypatch.setattr(rw, "_model_entry", lambda k: {"id": k, "runpod": {}})
    taken = {"n": 0}

    class _Pool:
        def _adopt_shared_pod(self):
            taken["n"] += 1
            return taken["n"] <= 16

    monkeypatch.setattr(rw, "get_pod_pool", lambda *a, **k: _Pool())
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "get_active_runpod_config", lambda: _acct())
    assert rw.reclaim_registered_pods() == 16


def test_a_pool_that_keeps_saying_yes_cannot_spin(registry, monkeypatch):
    """Adoption removes nothing by itself, so the loop is bounded by what the
    registry holds rather than by the answer."""
    registry["p1"] = _entry(pool="mymodel")
    monkeypatch.setattr(rw, "_model_entry", lambda k: {"id": k, "runpod": {}})
    monkeypatch.setattr(rw, "get_pod_pool", lambda *a, **k: types.SimpleNamespace(
        _adopt_shared_pod=lambda: True))          # always yes
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "get_active_runpod_config", lambda: _acct())
    assert rw.reclaim_registered_pods() == 1      # one entry -> one attempt


# ------------------------------------------- the knob is reachable from the UI #
def test_parallel_boots_is_editable_in_the_admin_form():
    """It decides how many GPUs start billing together, which is exactly the
    kind of thing an operator should not have to edit models.json for."""
    import pathlib
    from codai import admin
    html = (pathlib.Path(admin.__file__).parent / "templates" / "models.html").read_text()
    assert 'id="cfg-rp-parallel-boots"' in html
    assert "rp.scale_up_max_parallel_boots" in html
    assert "scale_up_max_parallel_boots: intOrU('cfg-rp-parallel-boots')" in html


def test_parallel_boots_survives_the_save():
    """A field the form sends and the save path drops is worse than no field."""
    import pathlib
    from codai.admin import routes
    src = pathlib.Path(routes.__file__).read_text()
    assert '"scale_up_max_parallel_boots"' in src
