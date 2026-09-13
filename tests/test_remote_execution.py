"""Remote execution: workers, mux engines, remote text models, the gateway.

Everything here runs against local stub servers — no GPU, no weights, no network.
"""

import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _QuietServer(ThreadingHTTPServer):
    """A cancelled stream closes the socket mid-write; that is not a test failure."""

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            super().handle_error(request, client_address)


def _serve(handler_cls):
    """Start a stub HTTP server on a free port; return (base_url, shutdown)."""
    srv = _QuietServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", srv.shutdown


# --------------------------------------------------------------------------- #
# workers: an unreachable service_url must fail loudly, not hang or build a venv
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod,attr", [
    ("codai.api.vllm_worker", "ensure_service"),
    ("codai.api.ds4_worker", "ensure_service"),
    ("codai.api.kt_worker", "ensure_service"),
])
def test_worker_service_url_health_check_is_enforced(monkeypatch, mod, attr):
    import importlib
    m = importlib.import_module(mod)

    class Cfg:
        service_url = "http://127.0.0.1:1"     # nothing listens there

    with pytest.raises(RuntimeError, match="not answering its health check"):
        getattr(m, attr)(Cfg())


# --------------------------------------------------------------------------- #
# colibri / k3: the mux service and its client
# --------------------------------------------------------------------------- #
class _StubEngine:
    family = "glm"

    def __init__(self):
        self.paused = 0

    def is_alive(self):
        return True

    def log_tail(self):
        return ""

    def pause(self):
        self.paused += 1

    def resume(self):
        self.paused -= 1

    def run(self, prompt, max_tokens, temperature, top_p, on_text, cancelled=None):
        for word in ("Hello", " from", " the", " remote", " engine"):
            if cancelled and cancelled():
                break
            on_text(word)
            time.sleep(0.005)
        return {"completion_tokens": 5, "prompt_tokens": len(prompt) // 4,
                "tokens_per_second": 42.0, "cache_hit_percent": 0.0,
                "rss_gb": 1.0, "length_limited": False}


@pytest.fixture()
def mux_service():
    import tools.mux_service as svc
    svc.ENGINE = _StubEngine()
    url, shutdown = _serve(svc.Handler)
    yield url, svc
    shutdown()


def test_remote_mux_engine_streams_and_reports_stats(mux_service):
    url, svc = mux_service
    from codai.api.mux_remote import remote_engine_for

    eng = remote_engine_for("colibri", url)
    assert eng.is_alive() and eng.family == "glm"

    chunks = []
    stats = eng.run("<prompt>", 128, 0.7, 1.0, on_text=chunks.append)
    assert "".join(chunks) == "Hello from the remote engine"
    assert stats["completion_tokens"] == 5 and stats["tokens_per_second"] == 42.0


def test_remote_mux_engine_cancels_mid_stream(mux_service):
    url, _ = mux_service
    from codai.api.mux_remote import remote_engine_for

    eng = remote_engine_for("colibri", url)
    chunks = []
    eng.run("<p>", 128, 0.7, 1.0, on_text=chunks.append,
            cancelled=lambda: len(chunks) >= 1)
    assert len(chunks) == 1


def test_remote_mux_engine_propagates_engine_errors(mux_service):
    url, svc = mux_service
    from codai.api.mux_remote import remote_engine_for
    eng = remote_engine_for("colibri", url)

    class Boom(_StubEngine):
        def run(self, *a, **k):
            raise RuntimeError("engine exploded")

    svc.ENGINE = Boom()
    with pytest.raises(RuntimeError, match="engine exploded"):
        eng.run("x", 8, 0.0, 1.0, on_text=lambda t: None)


def test_remote_mux_engine_rejects_a_dead_service():
    from codai.api.mux_remote import remote_engine_for
    with pytest.raises(RuntimeError, match="not answering its health check"):
        remote_engine_for("k3", "http://127.0.0.1:1")


# --------------------------------------------------------------------------- #
# text models: RemoteOpenAIBackend
# --------------------------------------------------------------------------- #
class _OpenAIStub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: dict = {}

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._send(200, {"data": [{"id": "remote-llm"}]})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n))
        type(self).seen = {"model": req.get("model"), "path": self.path,
                           "auth": self.headers.get("Authorization")}
        if not req.get("stream"):
            return self._send(200, {"choices": [{"message": {"content": "remote says hi"}}],
                                    "usage": {"prompt_tokens": 3, "completion_tokens": 4,
                                              "total_tokens": 7}})
        body = b"".join(
            b"data: " + json.dumps({"choices": [{"delta": {"content": w}}]}).encode() + b"\n\n"
            for w in ("str", "eam", "ed")) + b"data: [DONE]\n\n"
        self._send(200, body, "text/event-stream")


@pytest.fixture()
def remote_llm(monkeypatch):
    url, shutdown = _serve(_OpenAIStub)
    entry = {"path": "/AI/guffcache/fake-Q4_0.gguf", "alias": "remote-gguf",
             "service_url": url, "served_model": "remote-llm",
             "api_key": "sekrit", "n_ctx": 8192}
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "_model_entry_for",
                        lambda name: entry if name == "remote-gguf" else None)
    yield entry
    shutdown()


def test_gguf_model_with_service_url_is_served_remotely(remote_llm):
    import codai.models.manager as mgr
    from codai.backends.remote_openai import RemoteOpenAIBackend, remote_should_handle

    assert remote_should_handle("remote-gguf")
    assert not remote_should_handle("some-other-model")
    # No local weights: never downloaded, cached or charged VRAM.
    assert mgr._model_is_remote("remote-gguf")

    backend = RemoteOpenAIBackend()
    backend.load_model("remote-gguf")
    assert backend._url == remote_llm["service_url"] + "/v1"
    assert backend.get_context_size() == 8192

    assert backend.generate_chat([{"role": "user", "content": "hi"}]) == "remote says hi"
    assert backend.get_last_usage()["total_tokens"] == 7
    assert _OpenAIStub.seen["model"] == "remote-llm"
    assert _OpenAIStub.seen["auth"] == "Bearer sekrit"


def test_remote_backend_streams(remote_llm):
    from codai.backends.remote_openai import RemoteOpenAIBackend
    backend = RemoteOpenAIBackend()
    backend.load_model("remote-gguf")

    async def collect():
        return "".join([c async for c in backend.generate_chat_stream(
            [{"role": "user", "content": "hi"}])])

    assert asyncio.run(collect()) == "streamed"


def test_service_url_already_ending_in_v1_is_not_doubled(remote_llm):
    from codai.backends.remote_openai import RemoteOpenAIBackend
    remote_llm["service_url"] = remote_llm["service_url"] + "/v1/"
    backend = RemoteOpenAIBackend()
    backend.load_model("remote-gguf")
    assert backend._url.endswith("/v1") and not backend._url.endswith("/v1/v1")


# --------------------------------------------------------------------------- #
# RunPod: GGUF goes to llama.cpp, an HF repo goes to vLLM
# --------------------------------------------------------------------------- #
def test_runpod_pod_engine_selection():
    from codai.api.runpod_worker import (parse_model_runpod, resolve_pod_engine,
                                         _llamacpp_docker_args)

    hf = parse_model_runpod({"served_model": "Qwen/Qwen3.5-9B"})
    assert resolve_pod_engine(hf, "qwen", "Qwen/Qwen3.5-9B") == "vllm"

    gguf = parse_model_runpod({"hf_gguf": "bartowski/gemma-4-31B-GGUF:Q4_0", "ctx": 32768})
    assert resolve_pod_engine(gguf, "gemma", "/AI/g/gemma-Q4_0.gguf") == "llamacpp"
    args = _llamacpp_docker_args(gguf, "gemma")
    assert "-hf bartowski/gemma-4-31B-GGUF:Q4_0" in args and "-c 32768" in args

    # An explicit pin overrides the sniffing.
    assert resolve_pod_engine(parse_model_runpod({"engine": "vllm"}), "x", "/a/b.gguf") == "vllm"

    # A GGUF with nowhere to fetch it from is an error, not a broken pod.
    with pytest.raises(RuntimeError, match="hf_gguf"):
        _llamacpp_docker_args(parse_model_runpod({}), "foo.gguf")


# --------------------------------------------------------------------------- #
# the capability gateway
# --------------------------------------------------------------------------- #
def test_capability_paths_map_to_capabilities():
    from codai.api.remote_gateway import capability_for
    cases = {"/v1/images/generations": "images", "/v1/video/generations": "video",
             "/v1/images/to3d": "spatial", "/v1/3d/generate": "spatial",
             "/v1/audio/speech": "tts", "/v1/audio/transcriptions": "stt",
             "/v1/audio/clone": "voice", "/v1/audio/speaker-verify": "speaker",
             "/v1/embeddings": "embeddings", "/v1/ocr/batch": "ocr",
             # chat is served by RemoteOpenAIBackend, never by the gateway
             "/v1/chat/completions": ""}
    assert {p: capability_for(p) for p in cases} == cases


class _EchoRemote(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    hits: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        type(self).hits.append({"path": self.path, "len": len(body),
                                "ctype": self.headers.get("Content-Type", ""),
                                "body": body})
        out = json.dumps({"served_by": "remote"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture()
def gateway_app():
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from codai.api.remote_gateway import RemoteGatewayMiddleware
    import codai.api.remote_gateway as gw

    app = FastAPI()

    @app.post("/v1/images/generations")
    async def images(req: Request):
        return {"served_by": "local", "prompt": (await req.json()).get("prompt")}

    @app.post("/v1/audio/transcriptions")
    async def stt(req: Request):
        form = await req.form()
        return {"served_by": "local", "file_len": len(await form["file"].read())}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        await req.json()
        return {"served_by": "local"}

    app.add_middleware(RemoteGatewayMiddleware)
    _EchoRemote.hits = []
    url, shutdown = _serve(_EchoRemote)
    gw.any_remote_configured = lambda: True
    gw.capability_endpoints = lambda: {}
    gw._model_remote = lambda m: ""
    yield TestClient(app), gw, url
    shutdown()


def test_gateway_is_transparent_when_nothing_is_remote(gateway_app):
    client, gw, _ = gateway_app
    gw.any_remote_configured = lambda: False

    r = client.post("/v1/images/generations", json={"model": "sdxl", "prompt": "cat"})
    assert r.json() == {"served_by": "local", "prompt": "cat"}
    # A buffered-then-replayed multipart body must still parse downstream.
    r = client.post("/v1/audio/transcriptions", data={"model": "whisper"},
                    files={"file": ("a.wav", b"RIFF" * 200)})
    assert r.json() == {"served_by": "local", "file_len": 800}


def test_gateway_forwards_by_capability_and_leaves_the_rest_local(gateway_app):
    client, gw, url = gateway_app
    gw.capability_endpoints = lambda: {"images": url}

    assert client.post("/v1/images/generations",
                       json={"model": "sdxl", "prompt": "cat"}).json()["served_by"] == "remote"
    assert json.loads(_EchoRemote.hits[-1]["body"])["prompt"] == "cat"
    assert _EchoRemote.hits[-1]["path"] == "/v1/images/generations"

    # stt isn't configured, so it is served locally with its body intact.
    r = client.post("/v1/audio/transcriptions", data={"model": "whisper"},
                    files={"file": ("a.wav", b"RIFF" * 200)})
    assert r.json() == {"served_by": "local", "file_len": 800}


def test_gateway_forwards_multipart_verbatim(gateway_app):
    client, gw, url = gateway_app
    gw.capability_endpoints = lambda: {"stt": url}

    r = client.post("/v1/audio/transcriptions", data={"model": "whisper"},
                    files={"file": ("a.wav", b"RIFF" * 200)})
    assert r.json()["served_by"] == "remote"
    assert "multipart/form-data" in _EchoRemote.hits[-1]["ctype"]
    assert _EchoRemote.hits[-1]["len"] > 800


def test_per_model_service_url_beats_the_capability_map(gateway_app):
    client, gw, url = gateway_app
    gw._model_remote = lambda m: url if m == "sdxl-remote" else ""

    assert client.post("/v1/images/generations",
                       json={"model": "sdxl-remote"}).json()["served_by"] == "remote"
    assert client.post("/v1/images/generations",
                       json={"model": "sdxl"}).json()["served_by"] == "local"


def test_chat_is_never_gatewayed(gateway_app):
    client, gw, url = gateway_app
    gw.capability_endpoints = lambda: {"images": url}
    gw._model_remote = lambda m: url

    assert client.post("/v1/chat/completions",
                       json={"model": "anything"}).json()["served_by"] == "local"


def test_unreachable_remote_is_a_clean_502(gateway_app):
    client, gw, _ = gateway_app
    gw.capability_endpoints = lambda: {"images": "http://127.0.0.1:1"}

    r = client.post("/v1/images/generations", json={"model": "sdxl"})
    assert r.status_code == 502
    assert "unreachable" in r.json()["error"]["message"]
