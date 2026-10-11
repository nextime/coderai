"""Surya attached to the model layer instead of to a GPU this box may not have.

On 2026-10-11 OCR on the Digesta install (`ru002848`) answered HTTP 503: "OCR engine
'surya' failed to load: vLLM isolated venv not found at /cache/vllm_venv". The host has
no GPU at all — renting A40 pods on RunPod is that install's entire job — so the venv it
was asked to build could never have run, and the Surya-2 checkpoint it needed was already
a configured model (`backend: runpod`, `engine: coderai-vllm`) that the OCR path had no
way to ask for.

The OCR config was asking the wrong question. `surya_serve` enumerated server PROCESSES
(a local torch venv, our own vLLM, a URL typed in by hand) when the answer the operator
already wrote down was a MODEL. `model` mode asks the question the rest of the server
answers for everything else: where is this model served, and with what key.

What these tests hold:

* the resolver delegates — it does not invent a second path to a GPU, and a model nothing
  serves over HTTP is a clear 503, not a silent loopback;
* an endpoint is a LEASE: a reaped pod costs one re-resolve, not every page after it;
* a pod's bearer token reaches Surya, through the one setting Surya's vLLM client reads;
* the served modes claim no local VRAM, because they hold none.
"""

import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SURYA_SRC = (ROOT / "codai/ocr/surya.py").read_text()
SERVED_SRC = (ROOT / "codai/ocr/served_model.py").read_text()
CONFIG_SRC = (ROOT / "codai/config.py").read_text()
SETTINGS_HTML = (ROOT / "codai/admin/templates/settings.html").read_text()
ROUTES_SRC = (ROOT / "codai/admin/routes.py").read_text()

from codai.ocr.base import OcrError                       # noqa: E402
from codai.ocr import served_model                        # noqa: E402
from codai.ocr.surya import SuryaEngine                   # noqa: E402


class Cfg:
    """Just the OCR fields these paths read."""

    def __init__(self, **kw):
        self.surya_accept_license = True
        self.surya_serve = "model"
        self.surya_model = "datalab-to/surya-ocr-2"
        self.surya_model_id = ""
        self.surya_server_url = ""
        self.surya_venv = ""
        self.surya_langs = "it"
        self.surya_auto_build = False
        self.vlm_gpu_memory_utilization = 0.0
        self.__dict__.update(kw)


class FakeBackend:
    """A backend that knows where its model is being served."""

    def __init__(self, url="https://pod-8000.proxy.runpod.net/v1", key="podtoken",
                 served="datalab-to/surya-ocr-2"):
        self.url, self.key, self._served_name = url, key, served
        self.calls = 0

    def openai_endpoint(self):
        self.calls += 1
        return self.url, self.key


def _models_layer(monkeypatch, runpod=False, engine=None):
    """Steer the model layer. Patched as ATTRIBUTES of the real module: the resolver
    does `from codai.models import manager`, which reads the package attribute and
    never looks at sys.modules — a fake installed there is simply not seen (and a test
    that thinks it is passes for the wrong reason)."""
    from codai.models import manager as real
    monkeypatch.setattr(real, "runpod_should_handle", lambda m: runpod)
    monkeypatch.setattr(real, "resolve_engine_backend", lambda m: engine)
    monkeypatch.setattr(real, "get_active_runpod_config", lambda: None, raising=False)
    return real


@pytest.fixture(autouse=True)
def _clean_cache():
    served_model.forget()
    yield
    served_model.forget()


# ------------------------------------------------------------------ the resolver
def test_the_endpoint_comes_from_whichever_backend_owns_the_model(monkeypatch):
    """The resolver's whole job is delegation: it must not know what a pod is."""
    backend = FakeBackend()
    monkeypatch.setattr(served_model, "_entry_backend", lambda mid: backend)
    base, key, served = served_model.endpoint_for("datalab-to/surya-ocr-2")
    assert base == "https://pod-8000.proxy.runpod.net/v1"
    assert key == "podtoken"
    assert served == "datalab-to/surya-ocr-2"


def test_the_backend_is_resolved_once_not_per_page(monkeypatch):
    """Rebuilding a pod pool per page would re-read config, re-register the pool and
    log a rental decision for every page of a 400-page file."""
    built = []

    def _build(mid):
        built.append(mid)
        return FakeBackend()

    monkeypatch.setattr(served_model, "_entry_backend", _build)
    for _ in range(5):
        served_model.endpoint_for("m")
    assert built == ["m"]


def test_a_model_nobody_serves_over_http_says_so(monkeypatch):
    """The alternative is a loopback to our own /v1, which would hide the
    misconfiguration behind an extra hop and drop Surya's guided decoding."""
    _models_layer(monkeypatch)
    with pytest.raises(OcrError) as caught:
        served_model.endpoint_for("local-only-model")
    assert caught.value.status == 503
    assert "not served over HTTP" in str(caught.value)


def test_an_empty_model_id_is_a_configuration_error_not_a_lookup(monkeypatch):
    with pytest.raises(OcrError) as caught:
        served_model.endpoint_for("   ")
    assert caught.value.status == 400


def test_forget_lets_a_config_change_take_effect(monkeypatch):
    built = []
    monkeypatch.setattr(served_model, "_entry_backend",
                        lambda mid: built.append(mid) or FakeBackend())
    served_model.endpoint_for("m")
    served_model.forget("m")
    served_model.endpoint_for("m")
    assert built == ["m", "m"]


def test_the_health_base_drops_the_openai_suffix():
    """Surya probes /health on the base WITHOUT /v1, and refuses to attach when it
    does not answer 200 — so getting this wrong fails every served mode."""
    assert served_model.health_url("http://h:8000/v1") == "http://h:8000"
    assert served_model.health_url("http://h:8000/v1/") == "http://h:8000"
    assert served_model.health_url("http://h:8000") == "http://h:8000"


def test_the_resolver_delegates_rather_than_reimplementing_a_backend():
    """A copy of the pod-rental logic here would drift from the real one."""
    for forbidden in ("runpod_client", "graphql", "_provision", "gpu_types"):
        assert forbidden not in SERVED_SRC


# ------------------------------------------------------------------- the engine
def test_model_mode_hands_surya_the_url_the_name_and_the_key(monkeypatch):
    """Three values, because Surya refuses a server whose /v1/models disagrees with
    the checkpoint it expects, and a pod URL is useless without its token."""
    eng = SuryaEngine(Cfg(surya_model_id="datalab-to/surya-ocr-2"))
    eng._server_url = "https://pod-8000.proxy.runpod.net/v1"
    eng._server_key = "podtoken"
    eng._served_name = "datalab-to/surya-ocr-2"
    env = eng._worker_env()
    assert env["SURYA_INFERENCE_URL"] == "https://pod-8000.proxy.runpod.net/v1"
    assert env["SURYA_MODEL_CHECKPOINT"] == "datalab-to/surya-ocr-2"
    assert env["VLLM_API_KEY"] == "podtoken"


def test_a_locked_endpoint_is_attached_to_as_vllm():
    """Surya's llama.cpp client hardcodes api_key="EMPTY"; only its vLLM client reads
    VLLM_API_KEY. Attaching to a bearer-locked pod as "llamacpp" would 401 on every
    page, so the backend name is not simply the mode name."""
    eng = SuryaEngine(Cfg(surya_model_id="m"))
    eng._server_url, eng._server_key = "http://pod/v1", "tok"
    assert eng._worker_env()["SURYA_INFERENCE_BACKEND"] == "vllm"


def test_the_external_server_mode_still_attaches_as_llamacpp():
    eng = SuryaEngine(Cfg(surya_serve="llamacpp", surya_server_url="http://h/v1"))
    eng._server_url = "http://h/v1"
    assert eng._worker_env()["SURYA_INFERENCE_BACKEND"] == "llamacpp"


def test_server_is_accepted_as_a_synonym_of_llamacpp():
    """olmOCR calls this mode "server"; one subsystem, one vocabulary."""
    assert SuryaEngine(Cfg(surya_serve="server"))._serve_mode() == "llamacpp"


def test_model_mode_needs_a_model_id_and_says_which_setting(monkeypatch):
    eng = SuryaEngine(Cfg(surya_model_id="", surya_model=""))
    with pytest.raises(OcrError) as caught:
        eng.load()
    assert caught.value.status == 400
    assert "surya_model_id" in str(caught.value)


def test_the_checkpoint_setting_is_the_fallback_model_id(monkeypatch):
    """surya_model already names the Surya2 checkpoint; an install that set only that
    one should not have to repeat it."""
    seen = {}

    def _endpoint(mid):
        seen["id"] = mid
        return "http://pod/v1", "k", mid

    monkeypatch.setattr(served_model, "endpoint_for", _endpoint)
    eng = SuryaEngine(Cfg(surya_model_id="", surya_model="datalab-to/surya-ocr-2"))
    monkeypatch.setattr(SuryaEngine.__mro__[1], "load", lambda self: None)
    eng.load()
    assert seen["id"] == "datalab-to/surya-ocr-2"


def test_the_served_modes_hold_no_local_vram():
    """A 24-instance pool once asked to free 24 GB for workers that hold nothing. The
    pod's VRAM is the pod's, and it is accounted where it is spent."""
    for mode in ("model", "vllm", "llamacpp", "server"):
        assert SuryaEngine(Cfg(surya_serve=mode)).vram_gb() < 0.1
    assert SuryaEngine(Cfg(surya_serve="local")).vram_gb() >= 1.0


def test_model_mode_reserves_no_vram_on_this_box():
    """vllm mode must clear room on the local card first; model mode has no card to
    clear, and reserving against one would block OCR on a host with no GPU."""
    assert SuryaEngine(Cfg(surya_serve="model")).prelaunch_vram_gb() == 0.0


# -------------------------------------------------------------------- the lease
def _engine_with_fake_worker(monkeypatch, alive, fail_times=1):
    eng = SuryaEngine(Cfg(surya_model_id="m"))
    eng._server_url = "http://pod/v1"
    state = {"fails": fail_times, "loads": 0, "pages": 0}

    def _recognize(self, image):
        state["pages"] += 1
        if state["fails"] > 0:
            state["fails"] -= 1
            raise OcrError("surya: worker closed the pipe", status=500)
        return "page"

    monkeypatch.setattr(SuryaEngine.__mro__[1], "recognize_image", _recognize)
    monkeypatch.setattr(SuryaEngine, "load", lambda self: state.__setitem__("loads", state["loads"] + 1))
    monkeypatch.setattr(SuryaEngine, "cleanup", lambda self: None)
    monkeypatch.setattr(SuryaEngine, "_endpoint_alive", lambda self: alive)
    return eng, state


def test_a_reaped_pod_costs_one_re_resolve_not_the_document(monkeypatch):
    """min_pods=0 means the pod goes away when OCR goes quiet. Without this the
    worker stays pinned to a dead URL and every later page fails."""
    eng, state = _engine_with_fake_worker(monkeypatch, alive=False)
    assert eng.recognize_image(object()) == "page"
    assert state["loads"] == 1


def test_a_real_error_is_reported_not_retried_against_a_healthy_server(monkeypatch):
    """Restarting the worker on every failure would double the cost of a page that
    was always going to fail, and hide the reason."""
    eng, state = _engine_with_fake_worker(monkeypatch, alive=True)
    with pytest.raises(OcrError):
        eng.recognize_image(object())
    assert state["loads"] == 0


def test_the_retry_happens_once(monkeypatch):
    """A second failure is an outage, not an expired lease."""
    eng, state = _engine_with_fake_worker(monkeypatch, alive=False, fail_times=2)
    with pytest.raises(OcrError):
        eng.recognize_image(object())
    assert state["loads"] == 1


def test_only_model_mode_re_resolves(monkeypatch):
    """A URL the operator typed is theirs; coderai does not go looking for another."""
    eng, state = _engine_with_fake_worker(monkeypatch, alive=False)
    eng.cfg.surya_serve = "llamacpp"
    with pytest.raises(OcrError):
        eng.recognize_image(object())
    assert state["loads"] == 0


# ------------------------------------------------------------ what it is wired to
def test_the_mode_is_configurable_from_the_admin_ui():
    """A serving decision that only exists in a hand-edited config.json is not
    configured — the standing rule is that a model is orchestrated from the UI too."""
    assert 's-ocr-surya-serve' in SETTINGS_HTML
    assert 's-ocr-surya-model-id' in SETTINGS_HTML
    assert '<option value="model">' in SETTINGS_HTML
    assert "surya_model_id: document.getElementById('s-ocr-surya-model-id')" in SETTINGS_HTML
    assert "ocr.surya_model_id" in SETTINGS_HTML


def test_the_admin_api_round_trips_the_new_field():
    """Saved by the GUI, and not dropped by the whitelist on the way back in."""
    assert '"surya_model_id": c.ocr.surya_model_id,' in ROUTES_SRC
    assert 'if "surya_model_id" in d:' in ROUTES_SRC
    assert '"surya_model_id": self.config.ocr.surya_model_id,' in CONFIG_SRC


def test_a_pod_can_be_told_the_mode_through_the_environment():
    """A rented pod gets its OCR settings from the renting install, never its own
    config file."""
    assert '("CODERAI_OCR_SURYA_MODEL_ID", "surya_model_id")' in CONFIG_SRC


def test_the_incident_is_recorded_where_the_mode_is_defined():
    """-GPU-less is the case that found this; the next reader deserves the why."""
    assert "no GPU" in SERVED_SRC
    assert "NO_VLLM_VENV" in SERVED_SRC


# ------------------------------------------------------ images through a proxy
def test_a_proxy_backend_forwards_images_instead_of_flattening_them():
    """The front turns multipart content into the literal text "[image_url content]"
    unless the backend claims vision, and only the llama.cpp backend ever did — so
    every page sent to a VLM served by RunPod or by vLLM arrived as that placeholder
    and the model answered, fluently, about nothing.

    A proxy is not the right place to make that judgement: forwarded, a text-only
    model's server answers 400 and a VLM's server reads the page. This is the path a
    client takes when it drives olmOCR itself."""
    from codai.backends.runpod import RunpodBackend
    from codai.backends.remote_openai import RemoteOpenAIBackend
    from codai.backends.vllm import VllmBackend
    for cls in (RunpodBackend, RemoteOpenAIBackend, VllmBackend):
        assert getattr(cls, "supports_vision", False) is True, cls.__name__


def test_the_front_asks_the_backend_and_nothing_else():
    """If this gate moves, the flag above stops having any effect."""
    text = (ROOT / "codai/api/text.py").read_text()
    assert "'supports_vision', False" in text


# ------------------------------------------- structured output through a proxy
def test_a_proxy_forwards_the_schema_it_was_given():
    """It used to keep only {"type":"json_object"} and drop everything else, so a
    json_schema — the strict form current clients send, Surya's guided decoding among
    them — never reached the server, which then answered free text the caller could
    not parse. The streamed path dropped it outright, json_object included."""
    for mod in ("codai/backends/runpod.py", "codai/backends/vllm.py"):
        src = (ROOT / mod).read_text()
        assert 'response_format.get("type") == "json_object"' not in src, mod
        assert src.count('payload["response_format"] = response_format') == 2, mod


# ------------------------------------------------------------- the three places
def _cluster(monkeypatch, nodes, listing, enabled=True, token="clustertok",
             pin=None, model="m"):
    """Stand in for the cluster config, the model entry and the nodes' catalogues.

    ``pin`` is the model entry's ``engine`` — what makes a node eligible at all.
    """
    import types as _t

    cfg = _t.SimpleNamespace(cluster=_t.SimpleNamespace(enabled=enabled, token=token))
    entries = [{"id": model, **({"engine": pin} if pin else {})}]
    monkeypatch.setitem(sys.modules, "codai.admin.routes",
                        _t.SimpleNamespace(config_manager=_t.SimpleNamespace(
                            config=cfg, models_data={"text_models": entries})))

    class _Reply:
        def __init__(self, ids):
            self.status_code = 200 if ids is not None else 500
            self._ids = ids or []

        def json(self):
            return {"data": [{"id": i} for i in self._ids]}

    class _Client:
        def __init__(self, spec):
            self.spec = spec

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            for node in nodes:
                if node.url.rstrip("/") in url:
                    return _Reply(listing.get(node.name))
            return _Reply(None)

    monkeypatch.setitem(sys.modules, "codai.cluster.nodes",
                        _t.SimpleNamespace(enabled_nodes=lambda c: nodes,
                                           sync_client=_Client))
    return cfg


class _Node:
    def __init__(self, name, url, api_key=""):
        self.name, self.url, self.api_key = name, url, api_key

    def headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


def test_a_model_held_by_a_cluster_node_is_found(monkeypatch):
    """A node is a whole coderai registered as a remote ENGINE, so no backend in the
    model layer knows it exists. Without asking the nodes, a node-hosted VLM answered
    "not served over HTTP" — on an install that was serving it perfectly well.

    Which node serves a given page is now decided per request by load
    (codai.ocr.dispatch), so this asserts the discovery: the node that holds the
    model is a candidate, carrying the shared cluster token."""
    from codai.models import placement as dispatch
    from codai.models import placement as dispatch
    dispatch.forget()
    _models_layer(monkeypatch)
    nodes = [_Node("alpha", "https://a.lan:8443"), _Node("beta", "https://b.lan:8443")]
    _cluster(monkeypatch, nodes, {"alpha": ["qwen"], "beta": ["datalab-to/surya-ocr-2"]})
    _cluster(monkeypatch, nodes, {"alpha": ["qwen"], "beta": ["datalab-to/surya-ocr-2"]},
             pin="beta", model="datalab-to/surya-ocr-2")
    cands = dispatch._node_candidates("datalab-to/surya-ocr-2")
    assert [c.base for c in cands] == ["https://b.lan:8443/v1"]
    assert cands[0].api_key == "clustertok"   # shared token fills a blank node key


def test_every_node_that_holds_it_is_a_candidate_not_just_the_first(monkeypatch):
    """The first node that answered used to win for ever, however busy it got."""
    from codai.models import placement as dispatch
    from codai.models import placement as dispatch
    dispatch.forget()
    _models_layer(monkeypatch)
    nodes = [_Node("alpha", "https://a.lan:8443"), _Node("beta", "https://b.lan:8443")]
    _cluster(monkeypatch, nodes, {"alpha": ["m"], "beta": ["m"]}, pin="alpha, beta")
    assert len(dispatch._node_candidates("m")) == 2


def test_a_node_that_does_not_answer_is_skipped_not_fatal(monkeypatch):
    from codai.models import placement as dispatch
    from codai.models import placement as dispatch
    dispatch.forget()
    _models_layer(monkeypatch)
    nodes = [_Node("down", "https://d.lan:8443"), _Node("up", "https://u.lan:8443")]
    _cluster(monkeypatch, nodes, {"down": None, "up": ["m"]}, pin="down, up")
    assert [c.base for c in dispatch._node_candidates("m")] == ["https://u.lan:8443/v1"]


def test_the_nodes_are_not_asked_when_clustering_is_off(monkeypatch):
    """Probing every configured node on an install that does not cluster would add a
    timeout to the first page of every document."""
    from codai.models import placement as dispatch
    from codai.models import placement as dispatch
    dispatch.forget()
    _models_layer(monkeypatch)
    _cluster(monkeypatch, [_Node("alpha", "https://a.lan:8443")],
             {"alpha": ["m"]}, enabled=False, pin="alpha")
    assert dispatch._node_candidates("m") == []


def test_a_node_is_never_used_unless_the_model_is_pinned_to_it(monkeypatch):
    """Asking every node what it holds, and using whoever answers, is
    configuration by probe: it puts work on a machine nobody chose and moves it
    the moment a different node replies first. A node serves a model because the
    operator pinned the model to it — the same `engine` pin the front router
    already treats as a hard constraint."""
    from codai.models import placement as dispatch
    dispatch.forget()
    _models_layer(monkeypatch)
    _cluster(monkeypatch, [_Node("alpha", "https://a.lan:8443")], {"alpha": ["m"]})
    assert dispatch._node_candidates("m") == []


def test_a_pin_to_one_node_does_not_enlist_the_others(monkeypatch):
    from codai.models import placement as dispatch
    dispatch.forget()
    _models_layer(monkeypatch)
    nodes = [_Node("alpha", "https://a.lan:8443"), _Node("beta", "https://b.lan:8443")]
    _cluster(monkeypatch, nodes, {"alpha": ["m"], "beta": ["m"]}, pin="beta")
    assert [c.base for c in dispatch._node_candidates("m")] == ["https://b.lan:8443/v1"]


def test_a_pinned_node_that_does_not_hold_the_model_is_not_invented(monkeypatch):
    """Better an honest "nothing serves this" than pages sent to a node that
    cannot read them."""
    from codai.models import placement as dispatch
    dispatch.forget()
    _models_layer(monkeypatch)
    _cluster(monkeypatch, [_Node("alpha", "https://a.lan:8443")], {"alpha": ["other"]},
             pin="alpha")
    assert dispatch._node_candidates("m") == []


def test_runpod_wins_over_a_node_that_also_lists_it(monkeypatch):
    """An explicit backend pin on the entry is the operator's decision and is
    authoritative; node discovery is the fallback, not an override.

    Load-aware picking must not quietly repeal that: a pool that is the model's
    configured home is marked primary and is taken first, even though a node is
    the cheaper hop and would otherwise win on load."""
    from codai.models import placement as dispatch
    dispatch.forget()
    pool_taken = []

    node = dispatch.Candidate(key="node:alpha", kind="node",
                              base="https://a.lan:8443/v1", priority=1)

    def _acquire():
        pool_taken.append(1)
        return "https://pod/v1", "podtok", lambda: None
    pod = dispatch.Candidate(key="pod:m", kind="pod", base="", priority=2,
                             acquire=_acquire, primary=True)
    monkeypatch.setattr(dispatch, "candidates", lambda m: [node, pod])
    monkeypatch.setattr(dispatch, "_node_reported_load", lambda c: 0)
    lease = dispatch.choose("m")
    assert (lease.base, lease.api_key) == ("https://pod/v1", "podtok")
    assert pool_taken == [1]
    lease.release()
