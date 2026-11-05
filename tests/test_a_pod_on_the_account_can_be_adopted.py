"""A pod the account is billing can be taken OVER, not only taken down.

The page already shows `unmanaged` rows: pods on the same RunPod account that
no pool, boot, registry entry or parked set here knows about -- another host
renting on the same account, or a pod left behind by a process whose registry
entry was lost. The only action offered on one was to destroy it.

For a healthy pod that is already rented, already billing and running the
model a pool here wants, destroying it is the expensive answer: it throws away
a card and a 6-14 minute boot that have both been paid for, and under a warm
floor the pool then rents a replacement and pays that boot again.

What made this hard was the BEARER. Everything else an adoption needs is on
the account listing -- address, card, image, rate -- but a pod locked to a
token answers 401 to everything without it, and `get_pod` does not report how
a pod was started. It turns out to be recoverable from the launch itself, and
in TWO places, because `pod_plan` writes it to two: a coderai pod reads it
from its environment, while vLLM and llama.cpp take it as `--api-key` and it
lands in the pod's args. The args case is the common one, not the exception.

Adoption is then driven through the REGISTRY rather than by inserting a handle
into a pool: the entry is written with no live owner, and the pool's own
`_adopt_survivors` sweep takes it via `find_shared_pod`. So the image check,
the card check, the health probe and the per-pod bearer are the ones already
in use everywhere else, and an adoption that should not happen is refused by
the code that refuses it everywhere else.

THE HAZARD, pinned here because it is the reason this is not simply automatic:
a pod named for another deployment may still be MANAGED by that deployment on
a host this one cannot see. Two controllers on one pod each dispatch to it,
each miscount its in-flight work, and either may terminate it under the other.
Nothing RunPod reports says who is managing a pod, so it cannot be detected --
only refused by default, and overridden by an operator asserting what the API
cannot tell us.
"""
import inspect
import threading

import pytest

from codai.api import runpod_client as rc
from codai.api import runpod_worker as rw


# ── the bearer, out of how the pod was started ───────────────────────────────

def test_a_coderai_pods_token_comes_from_its_environment():
    assert rw.pod_bearer_from_launch(
        {"env": {"CODERAI_API_TOKEN": "tok-a"}, "args": ""}) == "tok-a"


def test_a_custom_images_token_is_found_under_the_vllm_name():
    assert rw.pod_bearer_from_launch({"env": {"VLLM_API_KEY": "tok-b"}}) == "tok-b"


def test_a_vllm_pods_token_comes_from_its_ARGS():
    """The common case: vLLM and llama.cpp take the bearer as a flag, so it is
    not in the environment at all."""
    assert rw.pod_bearer_from_launch(
        {"env": {}, "args": "--served-model-name m --api-key tok-c --port 8000"}
    ) == "tok-c"


def test_the_equals_form_is_found_too():
    assert rw.pod_bearer_from_launch({"args": "--api-key=tok-d"}) == "tok-d"


def test_a_token_nested_in_a_staged_bash_command_is_found():
    """A staged pod's command is a `bash -c` string with the whole server
    command inside ONE argument, so splitting on spaces finds nothing."""
    launch = {"cmd": ["bash", "-c",
                      "download && python3 -m vllm.entrypoints.openai.api_server "
                      "--api-key tok-e --port 8000"]}
    assert rw.pod_bearer_from_launch(launch) == "tok-e"


def test_the_environment_wins_over_the_args():
    """A coderai pod gets both; the environment is what its auth middleware
    reads."""
    assert rw.pod_bearer_from_launch(
        {"env": {"CODERAI_API_TOKEN": "env-tok"}, "args": "--api-key arg-tok"}
    ) == "env-tok"


def test_no_token_is_not_an_error():
    """A pool configured without one launches its pods open."""
    assert rw.pod_bearer_from_launch({"env": {}, "args": "--port 8000"}) == ""
    assert rw.pod_bearer_from_launch({}) == ""
    assert rw.pod_bearer_from_launch(None) == ""


def test_a_quoted_token_does_not_swallow_the_rest_of_the_line():
    assert rw.pod_bearer_from_launch(
        {"args": "--api-key 'tok-f' --port 8000"}) == "tok-f"


# ── the three shapes RunPod reports an environment in ────────────────────────

def test_env_comes_back_as_a_map_from_rest_v2():
    assert rc._env_map({"A": "1", "B": 2}) == {"A": "1", "B": "2"}


def test_env_comes_back_as_key_equals_value_strings_from_graphql():
    assert rc._env_map(["A=1", "B=2"]) == {"A": "1", "B": "2"}


def test_env_comes_back_as_key_value_objects():
    assert rc._env_map([{"key": "A", "value": "1"}]) == {"A": "1"}


def test_a_value_containing_an_equals_sign_survives():
    """Base64 and padded tokens end in '='; splitting on every '=' would cut
    the token in half and the pod would 401."""
    assert rc._env_map(["TOK=abc=="]) == {"TOK": "abc=="}


def test_a_missing_environment_is_empty_not_a_crash():
    assert rc._env_map(None) == {} and rc._env_map("") == {}


# ── the token must not travel on the paths that get rendered or logged ───────

def test_the_launch_lookup_is_separate_from_the_row_lookups():
    """`get_pod`/`list_pods` rows are rendered on the admin page and logged. A
    token that reaches either is a token leaked, so the launch fields travel on
    a path of their own."""
    row = inspect.getsource(rc._v2_pod_row)
    assert '"env"' not in row and "'env'" not in row, (
        "the row mapper must not carry a pod's environment")
    assert "args" not in row.replace("with_ports", ""), (
        "the row mapper must not carry a pod's args either")
    assert "env" in inspect.getsource(rc.RunpodClient.get_pod_launch)


def test_the_unmanaged_rows_carry_no_credential_fields(monkeypatch):
    monkeypatch.setattr(rw, "account_listing", lambda force=False: [
        {"id": "p1", "status": "RUNNING", "name": "someones-pod",
         "image": "ghcr.io/x/y:latest", "cost_per_hr": 0.59, "uptime_s": 60,
         "env": {"CODERAI_API_TOKEN": "tok-secret"},
         "args": "--api-key tok-secret"},
    ])
    rows = rw.account_only_pod_rows(set())
    assert len(rows) == 1
    blob = repr(rows[0])
    assert "tok-secret" not in blob, "a bearer must never reach the page"
    assert "env" not in rows[0] and "args" not in rows[0]


# ── who rented it, read off the name ────────────────────────────────────────

class _Acct:
    enabled = True
    api_key = "k"
    deployment_id = "digesta-ocr"


def test_our_own_pod_is_recognised_by_its_name():
    ours, claimed = rw._pod_name_owner(
        "coderai-digesta-ocr-digesta-qwen38-82fc2c", _Acct())
    assert ours is True
    assert claimed == "digesta-qwen38", "the uuid suffix is not part of the key"


def test_another_deployments_pod_is_not_ours():
    ours, claimed = rw._pod_name_owner(
        "coderai-someone-else-digesta-qwen38-82fc2c", _Acct())
    assert ours is False and claimed == ""


def test_a_pod_nobody_here_named_is_not_ours():
    assert rw._pod_name_owner("my-hand-started-pod", _Acct()) == (False, "")
    assert rw._pod_name_owner("", _Acct()) == (False, "")


def test_a_hand_renamed_pod_keeps_its_whole_tail():
    """Only a real uuid suffix is stripped; a pod renamed by hand loses
    nothing, or the claimed key would be wrong."""
    ours, claimed = rw._pod_name_owner("coderai-digesta-ocr-qwen38-spare", _Acct())
    assert ours is True and claimed == "qwen38-spare"


# ── the adoption itself ─────────────────────────────────────────────────────

def _pool(key="digesta-qwen38", port=8000):
    p = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    p._cv = threading.Condition()
    p.model_key = key
    p.pods = []
    p.mcfg = type("M", (), {"port": port})()
    p._adopted = []
    p._adopt_shared_pod = lambda: (p._adopted.append(1) or True)
    return p


@pytest.fixture
def live(monkeypatch):
    """One RUNNING pod on the account, rented by THIS deployment, and one pool
    that would take it."""
    pod = {"id": "lc4uzl4h8ujhw5", "status": "RUNNING",
           "name": "coderai-digesta-ocr-digesta-qwen38-82fc2c",
           "image": "ghcr.io/nextime/coderai-vllm:latest",
           "gpu": "RTX A6000", "gpu_count": 1, "vram_gb": 48.0,
           "cost_per_hr": 0.82,
           "ports": [{"ip": "1.2.3.4", "isIpPublic": True, "privatePort": 8000,
                      "publicPort": 40000, "type": "tcp"}]}
    launch = {"env": {}, "args": "--api-key pod-own-token", "cmd": []}

    class _Client:
        def __init__(self, acct): pass
        def get_pod(self, pid): return dict(pod)
        def get_pod_launch(self, pid): return dict(launch)

    monkeypatch.setattr(rc, "RunpodClient", _Client)
    monkeypatch.setattr("codai.models.manager.get_active_runpod_config",
                        lambda: _Acct())
    monkeypatch.setattr(rw, "_pod_health_ok", lambda *a, **k: True)
    monkeypatch.setattr(rw, "forget_account_listing", lambda: None)
    written = {}

    def _register(pid, pool_key="", url="", api_key="", **kw):
        written.update({"pod_id": pid, "pool": pool_key, "url": url,
                        "key": api_key, **kw})
        return None
    monkeypatch.setattr(rw, "register_pod", _register)
    disowned = {}

    def _reg_update(fn):
        d = {"lc4uzl4h8ujhw5": dict(written) or {"pool": "x"}}
        fn(d)
        disowned.update(d.get("lc4uzl4h8ujhw5") or {})
        return d
    monkeypatch.setattr(rw, "_registry_update", _reg_update)
    pool = _pool()
    monkeypatch.setattr(rw, "_pools", {"digesta-qwen38": pool})
    return {"pod": pod, "launch": launch, "written": written,
            "disowned": disowned, "pool": pool}


def test_our_own_survivor_is_adopted_with_no_override(live):
    out = rw.adopt_account_pod("lc4uzl4h8ujhw5")
    assert out["pool"] == "digesta-qwen38"
    assert out["offered"] is True
    assert out["adopted"] is True
    assert live["pool"]._adopted, "the pool is nudged, not left to the sweep"


def test_the_pods_own_bearer_is_what_gets_recorded(live):
    rw.adopt_account_pod("lc4uzl4h8ujhw5")
    assert live["written"]["key"] == "pod-own-token", (
        "a pod answers only to the token it booted with")


def test_the_card_and_image_are_recorded_so_the_pool_can_check_them(live):
    rw.adopt_account_pod("lc4uzl4h8ujhw5")
    w = live["written"]
    assert w["image"] == "ghcr.io/nextime/coderai-vllm:latest"
    assert w["gpu"] == "RTX A6000"
    assert w["hourly_usd"] == pytest.approx(0.82)
    assert w["vram_gb"] == pytest.approx(48.0)


def test_the_entry_is_written_with_no_live_owner(live):
    """`find_shared_pod` SKIPS this process's own entries, because a pool
    handles its own pods. An entry written as ours would be invisible to the
    sweep that is supposed to pick it up."""
    rw.adopt_account_pod("lc4uzl4h8ujhw5")
    assert live["disowned"].get("proc") == "offered-for-adoption"
    assert live["disowned"].get("pid") == 0
    assert not rw._is_this_process(live["disowned"])


def test_a_foreign_pod_is_refused_until_it_is_forced(live):
    live["pod"]["name"] = "coderai-another-host-qwen38-82fc2c"
    with pytest.raises(rw.PodNotOursError) as exc:
        rw.adopt_account_pod("lc4uzl4h8ujhw5")
    msg = str(exc.value)
    assert "not rented by this deployment" in msg
    assert "terminate it under the other" in msg, (
        "the refusal must say what the hazard IS, not just that there is one")
    assert not live["pool"]._adopted


def test_a_foreign_pod_is_adopted_when_the_operator_forces_it(live):
    live["pod"]["name"] = "coderai-another-host-qwen38-82fc2c"
    out = rw.adopt_account_pod("lc4uzl4h8ujhw5",
                              pool_key="digesta-qwen38", force=True)
    assert out["adopted"] is True


def test_a_pod_that_is_not_running_is_refused(live):
    live["pod"]["status"] = "EXITED"
    with pytest.raises(LookupError) as exc:
        rw.adopt_account_pod("lc4uzl4h8ujhw5")
    assert "nothing to adopt" in str(exc.value)


def test_a_pod_with_no_port_yet_is_refused(live):
    live["pod"]["ports"] = []
    with pytest.raises(LookupError) as exc:
        rw.adopt_account_pod("lc4uzl4h8ujhw5")
    assert "still" in str(exc.value) and "booting" in str(exc.value)


def test_a_pod_no_pool_here_would_take_is_refused_and_says_what_is_running(live, monkeypatch):
    monkeypatch.setattr(rw, "_pools", {"something-else": _pool("something-else")})
    with pytest.raises(LookupError) as exc:
        rw.adopt_account_pod("lc4uzl4h8ujhw5")
    msg = str(exc.value)
    assert "digesta-qwen38" in msg, "it must say what the pod's name claimed"
    assert "something-else" in msg, "and what pools are actually running"


def test_an_unauthenticatable_pod_is_refused_rather_than_served(live, monkeypatch):
    """No token recoverable AND it will not answer without one: serving it
    would 401 every request, which is worse than not adopting it."""
    live["launch"]["args"] = "--port 8000"
    monkeypatch.setattr(rw, "_pod_health_ok", lambda *a, **k: False)
    with pytest.raises(PermissionError) as exc:
        rw.adopt_account_pod("lc4uzl4h8ujhw5")
    assert "401" in str(exc.value)
    assert not isinstance(exc.value, rw.PodNotOursError), (
        "forcing cannot make an unauthenticatable pod answer, so this refusal "
        "must not be the forceable one")


def test_an_open_pod_with_no_token_is_adopted(live, monkeypatch):
    """A pool configured without a bearer launches its pods open; such a pod
    answers, so it is adoptable with no token at all."""
    live["launch"]["args"] = "--port 8000"
    out = rw.adopt_account_pod("lc4uzl4h8ujhw5")
    assert out["adopted"] is True
    assert live["written"]["key"] == ""


def test_a_pod_already_managed_here_is_reported_not_re_registered(live):
    handle = type("H", (), {"pod_id": "lc4uzl4h8ujhw5"})()
    live["pool"].pods.append(handle)
    out = rw.adopt_account_pod("lc4uzl4h8ujhw5")
    assert out.get("already") is True
    assert not live["written"], "a live owner's entry must not be overwritten"


def test_adoption_uses_the_pools_own_port(live):
    """The pod's PUBLIC port is random per pod; the private one is what its
    server listens on and what the pool would have rented it on."""
    rw.adopt_account_pod("lc4uzl4h8ujhw5")
    assert "1.2.3.4:40000" in live["written"]["url"]


def test_no_pod_id_is_a_lookup_error():
    with pytest.raises(LookupError):
        rw.adopt_account_pod("")


def test_runpod_disabled_is_reported_as_such(monkeypatch):
    monkeypatch.setattr("codai.models.manager.get_active_runpod_config",
                        lambda: type("A", (), {"enabled": False, "api_key": ""})())
    with pytest.raises(RuntimeError) as exc:
        rw.adopt_account_pod("p1")
    assert "not enabled" in str(exc.value)


# ── the endpoint and the page ───────────────────────────────────────────────

def test_the_endpoint_maps_each_refusal_to_its_own_status():
    from codai.admin import routes
    src = inspect.getsource(routes.api_runpod_pod_adopt)
    assert "PodNotOursError" in src and "status_code=409" in src
    assert '"needs_force": True' in src and '"needs_force": False' in src, (
        "the page must be able to tell the forceable refusal from the other "
        "one without matching on wording")
    assert "LookupError" in src and "status_code=404" in src
    assert "RuntimeError" in src and "status_code=400" in src
    assert "adopt_account_pod" in src
    assert "run_on_control_pool" in src, (
        "the adoption talks to RunPod and probes the pod; it must not run on "
        "the event loop")


def test_the_endpoint_is_admin_scoped():
    from codai.admin import routes
    sig = inspect.signature(routes.api_runpod_pod_adopt)
    assert "username" in sig.parameters


def test_the_page_offers_adopt_only_on_an_unmanaged_row():
    page = (__import__("pathlib").Path(rw.__file__).resolve().parents[2]
            / "codai" / "admin" / "templates" / "runpod.html").read_text()
    i = page.index("function _killCell")
    cell = page[i:i + 1800]
    assert "rpAdopt(" in cell
    assert "p.state === 'unmanaged'" in cell
    assert "function rpAdopt" in page
    assert "pod-adopt" in page


def test_the_page_re_offers_the_forced_adoption_with_the_servers_own_words():
    """The 409 explains a hazard only the operator can rule out. Summarising it
    into "are you sure?" would hide the thing they have to decide."""
    page = (__import__("pathlib").Path(rw.__file__).resolve().parents[2]
            / "codai" / "admin" / "templates" / "runpod.html").read_text()
    i = page.index("async function rpAdopt")
    body = page[i:i + 2000]
    assert "409" in body
    assert "d.detail.needs_force" in body, (
        "only the forceable refusal may be re-offered")
    assert "confirm(say(d)" in body, "show the server's sentence verbatim"
    assert "force: true" in body or "post(true)" in body
