"""A LoRA and the base it adapts share one path — so a path is not an identity.

A shared pool is one pod serving several names over one copy of the weights: a
base model and its adapters. Live, that pod answered

    404 Client Error: Not Found for url: https://…:22065/v1/chat/completions

for every request naming the adapter, and kept answering it: the pool keeps the
pod warm, so nothing ever replaced it. Asked directly, the pod listed only the
base and reported `{"loras": []}`.

Which of the two names worked depended on which model happened to rent the pod,
so it looked intermittent. Five places decided "same model?" by comparing
`path`, and an adapter's path IS its base's path:

* the orchestrator seeded the OTHER pool members with no served name, so they
  arrived with no alias and were registered under their path;
* the pod's boot seeder skipped the second entry as already known;
* `register_model_runtime` did the same, which made teaching a running pod a
  no-op;
* the engine-side seed filter in main.py did it a third time;
* and vLLM's service key included the served name, so had the adapter got
  through it would have started a SECOND copy of a 27B on the same card.

Identity is (weights, name).
"""

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                      # noqa: E402
from codai.api import vllm_worker as vw                        # noqa: E402
from codai.config import _same_model, seed_entry_name          # noqa: E402


BASE_REPO = "vendor/Some-27B-AWQ-INT4"

BASE = {"id": "base-awq", "alias": "base", "path": BASE_REPO,
        "runpod": {"pool": "shared", "served_model": "base-awq",
                   "hf_repo": BASE_REPO, "engine": "coderai-vllm"}}
ADAPTER = {"id": "tuned-alpha", "enabled": True,
           "loras": [{"name": "tuned-alpha", "path": "/local/loras/tuned-alpha",
                      "weight": 1.0}],
           "runpod": {"pool": "shared", "served_model": "tuned-alpha",
                      "hf_repo": BASE_REPO, "engine": "coderai-vllm"}}


class _CM:
    def __init__(self, data):
        self.models_data = data


@pytest.fixture
def catalogue(monkeypatch):
    import codai.admin.routes as routes
    data = {"text_models": [dict(BASE), dict(ADAPTER)]}
    monkeypatch.setattr(routes, "config_manager", _CM(data), raising=False)
    return data


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
def test_the_same_path_under_a_different_name_is_a_different_model():
    assert not _same_model({"path": BASE_REPO, "alias": "base"},
                           {"path": BASE_REPO, "alias": "tuned-alpha"})


def test_the_same_path_under_the_same_name_is_the_same_model():
    assert _same_model({"path": BASE_REPO, "alias": "base"},
                       {"path": BASE_REPO, "id": "base"})


def test_an_unnamed_entry_still_matches_on_the_path_alone():
    """Nothing to tell them apart by, so the old behaviour has to stand."""
    assert _same_model({"path": BASE_REPO}, {"path": BASE_REPO, "alias": "x"})


def test_a_different_path_is_never_the_same_model():
    assert not _same_model({"path": "other/repo", "alias": "base"},
                           {"path": BASE_REPO, "alias": "base"})


# --------------------------------------------------------------------------- #
# what the orchestrator sends
# --------------------------------------------------------------------------- #
def test_a_pool_sibling_is_seeded_under_the_name_it_will_be_asked_for():
    assert rw.seed_served_name(ADAPTER) == "tuned-alpha"


def test_a_sibling_without_a_served_model_falls_back_to_its_id():
    assert rw.seed_served_name({"id": "plain", "runpod": {}}) == "plain"


def test_the_sibling_seed_carries_an_alias(catalogue):
    """Without one the pod registers it under its path, which is the base's."""
    extras = rw.shared_pool_seed_entries("shared", "base-awq")
    assert [e.get("id") for e in extras] == ["tuned-alpha"]
    seed = json.loads(rw.seed_model_env(extras[0], rw.seed_served_name(extras[0]), ""))[0]
    assert seed["alias"] == "tuned-alpha"
    assert seed["path"] == BASE_REPO


def test_the_old_call_produced_no_alias_at_all(catalogue):
    """The regression itself, pinned: this is what used to be sent."""
    extras = rw.shared_pool_seed_entries("shared", "base-awq")
    seed = json.loads(rw.seed_model_env(extras[0], "", ""))[0]
    assert "alias" not in seed


# --------------------------------------------------------------------------- #
# what the pod does with it
# --------------------------------------------------------------------------- #
def test_the_boot_seeder_keeps_both_the_base_and_its_adapter(monkeypatch, tmp_path):
    from codai.config import ConfigManager

    cm = ConfigManager.__new__(ConfigManager)
    cm.models_data = {"text_models": [{"path": BASE_REPO, "alias": "base-awq"}]}
    monkeypatch.setenv("CODERAI_SEED_MODELS", json.dumps([
        {"path": BASE_REPO, "alias": "base-awq", "model_type": "text_models"},
        {"path": BASE_REPO, "alias": "tuned-alpha", "model_type": "text_models",
         "loras": [{"path": "sha256:abc", "name": "tuned-alpha", "weight": 1.0}]},
    ]))
    cm._seed_models_from_env()
    names = [seed_entry_name(e) for e in cm.models_data["text_models"]]
    assert names == ["base-awq", "tuned-alpha"]


def test_registering_an_adapter_does_not_collapse_into_the_base(monkeypatch):
    """Teaching a running pod used to return 'already known' and change nothing."""
    import codai.admin.routes as routes
    from codai.api import models_transfer as mt

    data = {"text_models": [{"path": BASE_REPO, "alias": "base-awq"}]}
    monkeypatch.setattr(routes, "config_manager", _CM(data), raising=False)

    registered = {}

    class _MM:
        model_aliases = {}
        _assigned_model_keys = set()

        def set_default_model(self, name, config=None):
            registered[name] = config or {}

        def set_assigned_models(self, keys):
            self._assigned_model_keys = set(keys)

    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "multi_model_manager", _MM(), raising=False)

    state = mt.register_model_runtime(
        {"path": BASE_REPO, "alias": "tuned-alpha",
         "loras": [{"path": "sha256:abc", "name": "tuned-alpha", "weight": 1.0}]},
        "text_models")
    assert state == "added"
    # Keyed by the NAME, because the path belongs to the base.
    assert "tuned-alpha" in registered
    assert registered["tuned-alpha"]["path"] == BASE_REPO
    assert len(data["text_models"]) == 2


def test_re_registering_the_same_model_is_still_idempotent(monkeypatch):
    import codai.admin.routes as routes
    from codai.api import models_transfer as mt

    data = {"text_models": [{"path": BASE_REPO, "alias": "base-awq"}]}
    monkeypatch.setattr(routes, "config_manager", _CM(data), raising=False)
    import codai.models.manager as mgr

    class _MM:
        model_aliases = {}
        _assigned_model_keys = set()
        def set_default_model(self, name, config=None): pass
        def set_assigned_models(self, keys): pass

    monkeypatch.setattr(mgr, "multi_model_manager", _MM(), raising=False)
    assert mt.register_model_runtime(
        {"path": BASE_REPO, "alias": "base-awq"}, "text_models") == "known"
    assert len(data["text_models"]) == 1


# --------------------------------------------------------------------------- #
# one engine, not two
# --------------------------------------------------------------------------- #
class _Cfg:
    model_path = BASE_REPO
    model_id = "base-awq"
    ctx = 0
    gpu_memory_utilization = 0.0
    tensor_parallel_size = 0
    pipeline_parallel_size = 0
    distributed_executor_backend = ""
    max_num_seqs = 0
    dtype = ""
    quantization = ""
    extra_args = ""


def test_one_service_key_per_set_of_weights():
    """Keyed by name too, an adapter started a second copy of the base."""
    a = vw.service_key_for(_Cfg(), BASE_REPO, "base-awq")
    b = vw.service_key_for(_Cfg(), BASE_REPO, "tuned-alpha")
    assert a == b


def test_the_engine_answers_to_the_base_name_not_the_adapters(catalogue):
    """--served-model-name must not repeat a --lora-modules name."""
    names = vw._served_names(dict(BASE), "base-awq", BASE_REPO)
    assert "tuned-alpha" not in names
    assert "base" in names or "base-awq" in names


def test_an_adapter_request_still_launches_an_engine_serving_the_base(catalogue):
    """Started by the adapter, the engine must still answer for the base."""
    names = vw._served_names(dict(BASE), "tuned-alpha", BASE_REPO)
    assert "tuned-alpha" not in names
    assert names and names != [BASE_REPO]


def test_a_model_with_no_adapters_is_named_exactly_as_before(monkeypatch):
    import codai.admin.routes as routes
    monkeypatch.setattr(routes, "config_manager", _CM({"text_models": []}),
                        raising=False)
    assert vw._served_names({}, "plain-model", "/weights") == ["plain-model"]


# --------------------------------------------------------------------------- #
# an engine that does not serve the name is replaced, not reused
# --------------------------------------------------------------------------- #
class _Proc:
    def poll(self):
        return None


def _running(monkeypatch, key, url="http://127.0.0.1:9", stopped=None):
    if stopped is None:
        stopped = []
    monkeypatch.setattr(vw, "_services", {key: {"proc": _Proc(), "url": url}},
                        raising=False)
    monkeypatch.setattr(vw, "_health_ok", lambda u: True)
    monkeypatch.setattr(vw, "_stop_proc",
                        lambda svc, k="": stopped.append(k or svc["url"]))
    return stopped


def test_a_running_engine_that_serves_the_name_is_reused(monkeypatch):
    key = vw.service_key_for(_Cfg(), BASE_REPO, "base-awq")
    stopped = []
    _running(monkeypatch, key, stopped=stopped)
    monkeypatch.setattr(vw, "_serves_name", lambda url, name: True)
    assert vw._ensure_service_once(_Cfg(), BASE_REPO, "base-awq") == "http://127.0.0.1:9"
    assert stopped == []


def test_an_engine_taught_a_model_after_it_started_is_relaunched(monkeypatch):
    """vLLM registers adapters at LAUNCH. A pod is taught after it boots, so the
    running engine cannot serve what it was just told about."""
    key = vw.service_key_for(_Cfg(), BASE_REPO, "base-awq")
    stopped = []
    _running(monkeypatch, key, stopped=stopped)
    monkeypatch.setattr(vw, "_serves_name", lambda url, name: False)
    monkeypatch.setattr(vw, "ensure_built",
                        lambda cfg: (_ for _ in ()).throw(RuntimeError("relaunched")))
    with pytest.raises(RuntimeError, match="relaunched"):
        vw._ensure_service_once(_Cfg(), BASE_REPO, "tuned-alpha")
    assert stopped == [key]


def test_an_unanswered_probe_never_costs_a_relaunch(monkeypatch):
    """Restarting a loaded 27B because one probe timed out is worse than the doubt."""
    key = vw.service_key_for(_Cfg(), BASE_REPO, "base-awq")
    stopped = []
    _running(monkeypatch, key, stopped=stopped)
    monkeypatch.setattr(vw, "_serves_name", lambda url, name: None)
    assert vw._ensure_service_once(_Cfg(), BASE_REPO, "tuned-alpha") == "http://127.0.0.1:9"
    assert stopped == []


def test_the_engine_answers_the_same_set_whichever_name_started_it(catalogue):
    """Order-dependence is the bug; the launch must not inherit it."""
    by_base = vw._served_names(dict(BASE), "base-awq", BASE_REPO)
    by_adapter = vw._served_names(dict(BASE), "tuned-alpha", BASE_REPO)
    assert set(by_base) == set(by_adapter)
    assert BASE_REPO in by_base


# --------------------------------------------------------------------------- #
# the adapter has to be FOUND, and then it has to be THERE
# --------------------------------------------------------------------------- #
SEEDED_BASE = {"path": BASE_REPO, "alias": "base-awq"}
SEEDED_ADAPTER = {"path": BASE_REPO, "alias": "tuned-alpha",
                  "loras": [{"path": "sha256:abc", "name": "tuned-alpha",
                             "weight": 1.0}]}


def test_a_seeded_sibling_is_not_mistaken_for_the_model_itself(monkeypatch):
    """A seeded entry has `alias` and no `id`. Comparing ids compared "" with
    "", matched every sibling, and excluded all of them — so the pod launched
    with no --enable-lora and answered under the adapter's name with BASE
    weights. Measured on a live pod: `--served-model-name <base>` and nothing
    else on the command line."""
    import codai.admin.routes as routes
    monkeypatch.setattr(routes, "config_manager",
                        _CM({"text_models": [SEEDED_BASE, SEEDED_ADAPTER]}),
                        raising=False)
    specs = vw._sibling_lora_specs(SEEDED_BASE)
    assert [s["name"] for s in specs] == ["tuned-alpha"]


def test_a_model_does_not_collect_its_own_adapter_twice(monkeypatch):
    import codai.admin.routes as routes
    monkeypatch.setattr(routes, "config_manager",
                        _CM({"text_models": [SEEDED_BASE, SEEDED_ADAPTER]}),
                        raising=False)
    specs = vw._sibling_lora_specs(SEEDED_ADAPTER)
    assert [s["name"] for s in specs] == ["tuned-alpha"]


def test_an_adapter_that_cannot_be_resolved_is_not_served_under_its_name(monkeypatch):
    """Refusing the NAME is the point. Serving the base under an adapter's name
    is a fluent, confident, wrong answer that nothing downstream can detect."""
    import codai.admin.routes as routes
    monkeypatch.setattr(routes, "config_manager",
                        _CM({"text_models": [SEEDED_BASE, SEEDED_ADAPTER]}),
                        raising=False)
    monkeypatch.setattr("codai.models.text_loras.local_path", lambda src: None)
    assert vw._lora_args(SEEDED_BASE) == []
    assert "tuned-alpha" not in vw._served_names(SEEDED_BASE, "base-awq", BASE_REPO)


def test_the_adapter_bytes_go_to_the_pod_before_anything_loads(monkeypatch):
    """vLLM registers adapters when it STARTS; bytes that arrive later cost a
    full reload of the base model to take effect."""
    from codai.backends import runpod as rb
    import codai.models.manager as mgr
    import codai.api.remote_gateway as gw

    sent = []
    monkeypatch.setattr(mgr, "_model_entry_for",
                        lambda n: {"id": n, "loras": [{"path": "/local/a", "name": n}]},
                        raising=False)
    monkeypatch.setattr(gw, "ensure_text_loras",
                        lambda base, key, entry: sent.append(base))

    b = rb.RunpodBackend.__new__(rb.RunpodBackend)
    b._model_id = "tuned-alpha"
    b._served_name = "tuned-alpha"
    b._pool = type("P", (), {"api_key": "k"})()
    rb.RunpodBackend._ADAPTERS_SENT.clear()
    b._ensure_adapters("https://pod")
    b._ensure_adapters("https://pod")           # once per pod, not per request
    assert sent == ["https://pod"]


def test_a_model_with_no_local_adapters_sends_nothing(monkeypatch):
    from codai.backends import runpod as rb
    import codai.models.manager as mgr
    import codai.api.remote_gateway as gw

    monkeypatch.setattr(mgr, "_model_entry_for", lambda n: {"id": n}, raising=False)
    monkeypatch.setattr(gw, "ensure_text_loras",
                        lambda *a, **k: pytest.fail("nothing to send"))
    b = rb.RunpodBackend.__new__(rb.RunpodBackend)
    b._model_id = "plain"
    b._served_name = "plain"
    b._pool = type("P", (), {"api_key": "k"})()
    rb.RunpodBackend._ADAPTERS_SENT.clear()
    b._ensure_adapters("https://pod")


# --------------------------------------------------------------------------- #
# a pool keeps the LoRA door open, and an adapter walks through it
# --------------------------------------------------------------------------- #
def test_a_pool_engine_enables_lora_before_it_has_any_adapter(monkeypatch):
    """--enable-lora cannot be turned on later, so an engine without it has to
    RELOAD the base model the first time an adapter arrives."""
    import codai.admin.routes as routes
    monkeypatch.setattr(routes, "config_manager", _CM({"text_models": [SEEDED_BASE]}),
                        raising=False)
    monkeypatch.setenv("CODERAI_VLLM_ENABLE_LORA", "1")
    args = vw._lora_args(SEEDED_BASE)
    assert "--enable-lora" in args
    assert "--lora-modules" not in args
    assert "--max-lora-rank" in args and "--max-loras" in args


def test_a_model_that_serves_no_pool_pays_nothing(monkeypatch):
    import codai.admin.routes as routes
    monkeypatch.delenv("CODERAI_VLLM_ENABLE_LORA", raising=False)
    monkeypatch.setattr(routes, "config_manager", _CM({"text_models": [SEEDED_BASE]}),
                        raising=False)
    assert vw._lora_args(SEEDED_BASE) == []


def test_the_models_own_pool_also_opens_the_door(monkeypatch):
    """An install driving vLLM directly has no pod env to read."""
    import codai.admin.routes as routes
    monkeypatch.delenv("CODERAI_VLLM_ENABLE_LORA", raising=False)
    entry = dict(BASE)
    monkeypatch.setattr(routes, "config_manager", _CM({"text_models": [entry]}),
                        raising=False)
    assert "--enable-lora" in vw._lora_args(entry)


def test_the_pod_is_told_it_serves_a_pool():
    import json
    mcfg = rw.parse_model_runpod({"pool": "shared", "engine": "coderai-vllm",
                                  "hf_repo": BASE_REPO})
    plan = rw._plan("coderai", "img", "", mcfg, "key", dict(BASE), "base-awq")
    assert plan["env"]["CODERAI_VLLM_ENABLE_LORA"] == "1"


def test_a_model_with_no_pool_is_not():
    mcfg = rw.parse_model_runpod({"engine": "coderai-vllm", "hf_repo": BASE_REPO})
    plan = rw._plan("coderai", "img", "", mcfg, "key", dict(BASE), "base-awq")
    assert "CODERAI_VLLM_ENABLE_LORA" not in plan["env"]


def test_a_new_adapter_is_handed_to_the_running_engine_not_a_restart(monkeypatch):
    """The whole point: one A40, the base loaded once, adapters added live."""
    import codai.admin.routes as routes
    monkeypatch.setattr(routes, "config_manager",
                        _CM({"text_models": [SEEDED_BASE, SEEDED_ADAPTER]}),
                        raising=False)
    monkeypatch.setattr("codai.models.text_loras.local_path",
                        lambda src: "/blobs/tuned-alpha")
    posted = {}

    def _urlopen(req, timeout=0):
        posted["url"] = req.full_url
        posted["body"] = req.data
        return type("R", (), {"read": lambda self: b"{}"})()

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    assert vw._load_lora_runtime("http://e", SEEDED_BASE, "tuned-alpha") is True
    assert posted["url"].endswith("/v1/load_lora_adapter")
    assert b'"lora_name": "tuned-alpha"' in posted["body"]


def test_an_engine_that_refuses_at_runtime_still_gets_restarted(monkeypatch):
    """A pod rented an hour ago may run an older image than we would launch."""
    import urllib.error
    import codai.admin.routes as routes
    monkeypatch.setattr(routes, "config_manager",
                        _CM({"text_models": [SEEDED_BASE, SEEDED_ADAPTER]}),
                        raising=False)
    monkeypatch.setattr("codai.models.text_loras.local_path", lambda src: "/blobs/a")

    def _boom(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert vw._load_lora_runtime("http://e", SEEDED_BASE, "tuned-alpha") is False


def test_an_unknown_name_is_not_sent_anywhere(monkeypatch):
    import codai.admin.routes as routes
    monkeypatch.setattr(routes, "config_manager",
                        _CM({"text_models": [SEEDED_BASE, SEEDED_ADAPTER]}),
                        raising=False)
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: pytest.fail("nothing to register"))
    assert vw._load_lora_runtime("http://e", SEEDED_BASE, "not-an-adapter") is False
