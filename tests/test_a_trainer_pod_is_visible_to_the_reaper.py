"""A trainer pod belonged to nothing that could clean it up.

It is named `coderai-train-job-<id>-s0`, so `reap_orphans`'s
`coderai-<deployment_id>-` prefix never matched it, and it lived only in the
fleet's own record file -- not the registry that startup reclaim reads. While
the job ran, nothing noticed. When the process died (a crash, or one of
several forced upgrades in a day) the pod billed on with no owner and nothing
to collect it.

The same absence had a second cost: the admin pod table calls anything outside
a POOL "unmanaged", and a trainer pod never joins a pool. Three healthy pods
of three live training jobs were being displayed as abandoned -- and acting on
that label would have destroyed the run.

One registry entry fixes both ends. While the owner heartbeats,
`_known_pod_ids` counts the pod and the page stops lying about it; once the
owner is gone the entry stops vouching after the reap grace and the pod is
collected like any other leftover.
"""

import inspect

import pytest

from codai.api import text_lora_fleet as flt


class Client:
    def __init__(self):
        self.terminated = []

    def create_pod(self, **kw):
        return "pod-1"

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


@pytest.fixture(autouse=True)
def _never_touch_the_real_registry(monkeypatch, tmp_path):
    """Point the registry at a temp dir for EVERY test in this file.

    Without this, a test that rents a pod writes into the developer's own
    ~/.config/coderai/runpod-pods.json -- which is exactly what happened: five
    fake trainer pods landed there and broke an unrelated test
    (test_a_truly_empty_fleet_is_still_empty) in a way that survived the run,
    because the file persists. A test that writes outside its tmp_path is a
    test that fails somebody else's.
    """
    monkeypatch.setenv("CODERAI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import codai.api.runpod_worker as rw
    monkeypatch.setattr(rw, "_pod_registry_path",
                        lambda: str(tmp_path / "runpod-pods.json"))


@pytest.fixture
def registry(monkeypatch):
    """Capture what the fleet writes to the deployment registry."""
    seen = {"registered": [], "forgotten": []}
    import codai.api.runpod_worker as rw
    monkeypatch.setattr(rw, "register_pod",
                        lambda pod_id, **kw: seen["registered"].append((pod_id, kw)))
    monkeypatch.setattr(rw, "unregister_pod",
                        lambda pod_id: seen["forgotten"].append(pod_id))
    return seen


def _fleet(client=None, tmp=None, monkeypatch=None):
    spec = flt.TrainerPodSpec(image="img:1", gpu_types=["A40"],
                              hourly_usd=0.59)
    f = flt.TrainerFleet(client or Client(), "job-abc", spec)
    if monkeypatch is not None:
        monkeypatch.setattr(f, "_flush", lambda: None)
    return f


# ----------------------------------------------------------- on renting ----

def test_a_rented_pod_is_registered(registry, monkeypatch):
    f = _fleet(monkeypatch=monkeypatch)
    pod = f._rent_one(0)
    assert pod is not None
    assert registry["registered"], "the pod was not recorded as the deployment's"
    pod_id, kw = registry["registered"][0]
    assert pod_id == "pod-1"


def test_the_entry_names_the_owning_job(registry, monkeypatch):
    """So a stale entry says which run died, and the page can say who owns it."""
    f = _fleet(monkeypatch=monkeypatch)
    f._rent_one(0)
    _pid, kw = registry["registered"][0]
    assert kw["pool_key"] == "text-lora:job-abc"


def test_the_entry_carries_the_bearer_and_the_price(registry, monkeypatch):
    """The bearer so an adopted pod is not answered 401 for the rest of its
    paid life; the price so a stale pod's cost can be stated."""
    f = _fleet(monkeypatch=monkeypatch)
    pod = f._rent_one(0)
    _pid, kw = registry["registered"][0]
    assert kw["api_key"] == pod.secret and kw["api_key"]
    assert kw["hourly_usd"] == 0.59
    assert kw["image"] == "img:1"


def test_a_registry_failure_does_not_fail_the_rental(monkeypatch, capsys):
    """Bookkeeping must not cost a rented card -- but it must be said, because
    the consequence is a pod that leaks."""
    import codai.api.runpod_worker as rw
    monkeypatch.setattr(rw, "register_pod",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    f = _fleet(monkeypatch=monkeypatch)
    assert f._rent_one(0) is not None
    out = capsys.readouterr().out
    assert "could not register pod" in out
    assert "not be reclaimed or reaped" in out


# --------------------------------------------------------- on releasing ----

def test_a_released_pod_is_forgotten(registry, monkeypatch):
    f = _fleet(monkeypatch=monkeypatch)
    f._rent_one(0)
    assert f.release(0, "done") is True
    assert registry["forgotten"] == ["pod-1"]


def test_a_failed_terminate_keeps_the_registry_entry(registry, monkeypatch):
    """The pod may still be running. Dropping the entry would hide it from the
    one sweep meant to find it -- the exact leak this closes."""
    class Stubborn(Client):
        def terminate_pod(self, pod_id):
            raise RuntimeError("RunPod says no")

    f = _fleet(Stubborn(), monkeypatch=monkeypatch)
    f._rent_one(0)
    assert f.release(0, "done") is False
    assert registry["forgotten"] == [], "a possibly-live pod was forgotten"


def test_the_unregister_happens_after_the_terminate():
    src = inspect.getsource(flt.TrainerFleet.release)
    assert src.index("terminate_pod") < src.index("_unregister")
    assert "failed terminate" in src and "still running" in src


# -------------------------------------------- how the page labels it -------
#
# Registering was not enough on its own: the registry row said
# `state: "unowned"` with the note "held by no pool here … and still billing",
# which is true of a fleet pod and tells the reader the opposite of the truth.

from codai.api import runpod_worker as rw


def test_the_prefix_is_defined_once():
    """Writer and readers cannot drift: a prefix typed twice is a pod labelled
    unowned while a job is using it."""
    from codai.api.text_lora_fleet import _fleet_pool_prefix
    assert _fleet_pool_prefix() == rw.FLEET_POOL_PREFIX


def test_a_live_fleet_pod_reads_like_a_pool_pod():
    """The point of the consolidation: state and health come from the SAME
    field a pool uses -- an address in the registry -- so "booting" and
    "ready" mean the same thing for a trainer as for chandra. It previously
    reported state "training" with healthy=False in both cases, which the page
    drew as DOWN for a pod doing exactly what it should."""
    src = inspect.getsource(rw.pods_status)
    i = src.index("fleet_job and (mine or fresh)")
    block = src[i:i + 2000]      # the comment above the assignment is long
    assert 'state = "ready" if has_url else "provisioning"' in block
    assert "healthy = has_url" in block
    assert "one row\n            # renderer serves both" in block


def test_a_live_fleet_pod_is_not_called_still_billing():
    """That phrase is what made three working pods look like a leak."""
    src = inspect.getsource(rw.pods_status)
    i = src.index("fleet_job and (mine or fresh)")
    assert "still billing" not in src[i:i + 500]


def test_a_dead_jobs_pod_names_the_job_that_died():
    src = inspect.getsource(rw.pods_status)
    i = src.index("Not ours and past the grace")
    block = src[i:i + 400]
    assert "whose process is gone" in block
    assert "still billing" in block


def test_a_non_fleet_registry_pod_keeps_the_old_wording():
    """A serving pool's orphan is a different situation and still reads that
    way."""
    src = inspect.getsource(rw.pods_status)
    assert "in the registry but held by no pool here" in src


# ------------------------- and it has to be VISIBLE, not merely correct -----
#
# Registering fixed the label and broke the visibility. A sibling's POOL pod is
# skipped in the registry rows because that sibling's pool reports it above; a
# FLEET pod is reported by nothing -- it joins no pool, and being registered
# took it out of the account-only rows too. Two live training pods appeared
# NOWHERE on the RunPod page, which is worse than the wrong label it replaced:
# a pod you cannot see is one you cannot stop.

def test_a_fleet_pod_is_not_skipped_as_a_siblings_business():
    src = inspect.getsource(rw.pods_status)
    assert "if not mine and fresh and not fleet_job:" in src


def test_the_skip_explains_why_a_fleet_pod_is_different():
    src = inspect.getsource(rw.pods_status)
    i = src.index("a live sibling is managing it")
    block = src[max(0, i - 1400):i]
    assert "reported by nothing" in block
    assert "NOWHERE on the page" in block


def test_a_live_fleet_pod_in_another_process_is_still_training():
    """`mine` is false for a pod rented in an engine while the page is served
    by the front -- which is the normal arrangement, not an edge case."""
    src = inspect.getsource(rw.pods_status)
    assert "if fleet_job and (mine or fresh):" in src


def test_the_note_says_when_another_process_owns_it():
    src = inspect.getsource(rw.pods_status)
    assert "(another engine process)" in src


def test_the_row_carries_a_console_link():
    """A row about a pod costing money must be actionable; this one passed ""
    for the one place you could go and look."""
    src = inspect.getsource(rw.pods_status)
    assert "pod_console_url(str(pod_id))" in src
    assert '"console_url": console' in src


def test_the_price_is_the_cards_not_the_lists_estimate():
    """A pod reported $0.33/hr while RunPod charged $0.59, because the spec
    carried one estimate for the whole candidate list."""
    import dataclasses
    from codai.api import text_lora_fleet as f
    assert any(x.name == "gpu_prices" for x in dataclasses.fields(f.TrainerPodSpec))
    src = inspect.getsource(f.TrainerFleet._rent_one)
    assert "placeholder.hourly_usd = won_price" in src


def test_the_address_is_reported_when_it_appears():
    """Without this the registry never learns the pod answered, and the row
    stays "booting" for the whole run."""
    from codai.api import text_lora_fleet as f
    src = inspect.getsource(f.TrainerFleet.address)
    assert "self._register(pod)" in src


def test_a_stale_fleet_entry_is_still_reported_as_a_leak():
    """The whole point of registering: once the owner stops heartbeating the
    pod must read as abandoned, naming the job that died."""
    src = inspect.getsource(rw.pods_status)
    i = src.index("Not ours and past the grace")
    assert "whose process is gone" in src[i:i + 400]


def test_the_state_is_one_the_page_already_renders():
    """I invented "booting". The page's renderer handles `unowned`,
    `provisioning` and `quarantined` and then falls through to
    `healthy ? 'healthy' : 'down'`, so an unknown state is drawn in red as
    DOWN -- which is what an operator saw for two pods coming up normally.

    Asserted against the template so the two cannot drift: a state the page
    cannot draw is a state that lies.
    """
    from pathlib import Path
    tmpl = (Path(__file__).resolve().parents[1]
            / "codai/admin/templates/runpod.html").read_text()
    src = inspect.getsource(rw.pods_status)
    i = src.index("fleet_job and (mine or fresh)")
    emitted = src[i:i + 2000]
    for state in ("ready", "provisioning"):
        assert f'"{state}"' in emitted
        assert state in tmpl, f"the page has no case for {state!r}"
    # The ASSIGNMENT, not the prose: the comment above it explains the word
    # "booting" precisely because that one was wrong.
    assert 'else "booting"' not in emitted, "an invented state is drawn as down"
