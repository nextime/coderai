"""`/v1/models` must publish the context window it is serving with.

Anything that sends a growing conversation -- a continued generation, a resumed
OCR page, a long tool loop -- has to know when the window runs out, and the only
alternative is finding out from a 400 after the work is paid for.

coderai's OCR page resumption was the first casualty. It probed `/v1/models` for
`max_model_len`, coderai did not publish it, and the probe degraded in SILENCE:
the resumption budget ran with no ceiling and the log printed `?` with nothing to
say why. Two things are asserted here as a result -- that the field is published,
and that a server which answers WITHOUT it says so out loud.

The name is vLLM's, so a client written against a vLLM server reads it unchanged.
"""
import types

import pytest


# --- the field exists and means what it says ---------------------------------


def test_model_info_publishes_the_context_window():
    from codai.pydantic.textrequest import ModelInfo
    m = ModelInfo(id="x", max_model_len=32768)
    assert m.max_model_len == 32768
    assert "max_model_len" in m.model_dump()


def test_an_unknown_window_is_none_not_zero():
    """None means "not established here". 0 would read as a window of zero, and
    a client must not be able to confuse unknown with unlimited either."""
    from codai.pydantic.textrequest import ModelInfo
    assert ModelInfo(id="x").max_model_len is None


# --- where the number comes from, and in which order -------------------------


class _Mgr:
    """Just enough of a manager for the resolver: a loaded backend, and the
    resolved-path attribute the real one carries."""

    def __init__(self, backend=None, current_model=None):
        self.backend = backend
        self.current_model = current_model


@pytest.fixture
def resolve():
    from codai.models.manager import context_window_for
    return context_window_for


def test_the_listing_actually_calls_the_resolver():
    """The resolver being correct is worth nothing if the listing does not use
    it — which is how `max_model_len` came to be probed for and never
    published."""
    import inspect
    from codai.models.manager import MultiModelManager
    src = inspect.getsource(MultiModelManager.list_models)
    assert "max_model_len=context_window_for(self, model_id, meta)" in src


def test_a_live_backend_beats_the_configuration(resolve):
    """The two disagree in practice and only the served one is true: measured
    2026-10-13, `ctx: 18432` was configured and `--max-model-len 32768` served.
    A client trusting the config would plan around a window that does not
    exist."""
    backend = types.SimpleNamespace(_served_name="m", _model_id="m",
                                    context_length=lambda: 32768)
    got = resolve(_Mgr(backend=backend, current_model="/resolved/path"), "m",
                  {"runpod": {"ctx": 18432}})
    assert got == 32768


def test_the_backend_is_matched_on_what_it_says_it_serves(resolve):
    """`current_model` holds a RESOLVED PATH, not the id, so matching on it
    alone misses almost every time."""
    backend = types.SimpleNamespace(_served_name="m", _model_id="m",
                                    context_length=lambda: 8192)
    got = resolve(_Mgr(backend=backend, current_model="/opt/models/m.gguf"),
                  "m", {})
    assert got == 8192


def test_another_models_backend_is_not_asked(resolve):
    backend = types.SimpleNamespace(_served_name="other", _model_id="other",
                                    context_length=lambda: 999999)
    got = resolve(_Mgr(backend=backend), "m", {"n_ctx": 4096})
    assert got == 4096


def test_a_backend_that_raises_falls_back_to_configuration(resolve):
    def boom():
        raise RuntimeError("service not started")
    backend = types.SimpleNamespace(_served_name="m", _model_id="m",
                                    context_length=boom)
    assert resolve(_Mgr(backend=backend), "m", {"ctx": 16384}) == 16384


@pytest.mark.parametrize("meta,expected", [
    ({"max_model_len": 32768}, 32768),
    ({"n_ctx": 8192}, 8192),
    ({"ctx": 4096}, 4096),
    ({"context_length": 2048}, 2048),
    ({"vllm": {"max_model_len": 65536}}, 65536),
    ({"runpod": {"ctx": 18432}}, 18432),
    ({"llamacpp": {"n_ctx": 512}}, 512),
])
def test_every_spelling_a_backend_uses_is_read(resolve, meta, expected):
    """Several backends own this number and each spells it its own way; a
    resolver that knew one spelling would publish None for the others."""
    assert resolve(_Mgr(), "m", meta) == expected


@pytest.mark.parametrize("meta", [
    {}, {"ctx": 0}, {"ctx": None}, {"ctx": "wide"}, {"ctx": []},
    {"vllm": "not-a-dict"},
])
def test_nothing_usable_resolves_to_none(resolve, meta):
    """A stray string or list must not become a window, and must not raise."""
    assert resolve(_Mgr(), "m", meta) is None


def test_a_list_wrapped_number_is_unwrapped(resolve):
    """An old global default wrapped in a list is a real shape in these configs."""
    assert resolve(_Mgr(), "m", {"ctx": [16384]}) == 16384


# --- the OCR worker reads it, and complains when it is absent ----------------


def _worker_module():
    import importlib
    return importlib.import_module("codai.ocr.workers.ocr_worker")


class _Models:
    def __init__(self, payload):
        self._payload = payload

    def list(self):
        return types.SimpleNamespace(data=self._payload)


def _client(payload, base="http://stub/ctx-test"):
    c = types.SimpleNamespace(models=_Models(payload))
    c.base_url = base
    return c


def test_the_worker_reads_the_window_from_an_extra_field(capsys):
    """`max_model_len` is not in the OpenAI model schema, so the client parks it
    as an EXTRA field. Reading only attributes found nothing -- and said
    nothing."""
    mod = _worker_module()
    mod._CONTEXT_CACHE.clear()
    m = types.SimpleNamespace(id="m", model_extra={"max_model_len": 32768})
    assert mod._context_limit(_client([m], "http://stub/extra"), "m") == 32768
    assert "32768-token context window" in capsys.readouterr().err


def test_a_server_that_names_no_window_says_so(capsys):
    """The failure that actually happened. It must not pass in silence: the log
    showed '?' and no reason, which is a degradation indistinguishable from
    working."""
    mod = _worker_module()
    mod._CONTEXT_CACHE.clear()
    m = types.SimpleNamespace(id="m", model_extra={})
    assert mod._context_limit(_client([m], "http://stub/silent"), "m") == 0
    err = capsys.readouterr().err
    assert "does not report a context window" in err
    assert "surya_context_tokens" in err, "it does not say what to do about it"


def test_the_window_is_asked_for_once_per_server(capsys):
    mod = _worker_module()
    mod._CONTEXT_CACHE.clear()
    calls = []

    class _Counting(_Models):
        def list(self):
            calls.append(1)
            return super().list()

    c = types.SimpleNamespace(
        models=_Counting([types.SimpleNamespace(id="m",
                                                model_extra={"max_model_len": 4096})]))
    c.base_url = "http://stub/once"
    assert mod._context_limit(c, "m") == 4096
    assert mod._context_limit(c, "m") == 4096
    assert len(calls) == 1, "it re-probes a model that cannot change shape"


def test_the_backend_on_a_loaded_sub_manager_is_found(resolve):
    """`list_models` is a method of the MULTI manager, which has no `.backend`
    of its own — the per-model manager does. Handling only the single-manager
    shape meant the live path could never fire at all."""
    inner = _Mgr(backend=types.SimpleNamespace(
        _served_name="m", _model_id="m", context_length=lambda: 32768))
    outer = _Mgr()
    outer.models = {"m": inner}
    assert resolve(outer, "m", {"runpod": {"ctx": 18432}}) == 32768


def test_a_sub_manager_under_an_unrelated_key_is_still_found(resolve):
    """The dict is keyed by the manager's own scheme (a path, "whisper:x"), so a
    direct lookup is not guaranteed even when the model IS loaded."""
    inner = _Mgr(backend=types.SimpleNamespace(
        _served_name="m", _model_id="m", context_length=lambda: 8192))
    outer = _Mgr()
    outer.models = {"/opt/models/some-path.gguf": inner}
    assert resolve(outer, "m", {}) == 8192


def test_a_loaded_sub_manager_for_another_model_is_not_used(resolve):
    inner = _Mgr(backend=types.SimpleNamespace(
        _served_name="other", _model_id="other", context_length=lambda: 999999))
    outer = _Mgr()
    outer.models = {"other": inner}
    assert resolve(outer, "m", {"ctx": 4096}) == 4096
