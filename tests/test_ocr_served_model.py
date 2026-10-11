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
    mod = types.SimpleNamespace(
        runpod_should_handle=lambda m: False,
        resolve_engine_backend=lambda m: None,
        get_active_runpod_config=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "codai.models.manager", mod)
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
