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
    # Restore on teardown: these tests replace module-level functions, and a
    # leftover "every model is remote" lambda silently changes what LATER tests
    # are measuring.
    saved = {name: getattr(gw, name)
             for name in ("any_remote_configured", "capability_endpoints",
                          "capability_pods", "_model_remote")}
    gw.any_remote_configured = lambda: True
    gw.capability_endpoints = lambda: {}
    gw._model_remote = lambda m: ""
    try:
        yield TestClient(app), gw, url
    finally:
        for name, fn in saved.items():
            setattr(gw, name, fn)
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


def test_capability_served_by_a_managed_pod(gateway_app, monkeypatch):
    """A capability can point at a RunPod pool instead of a fixed URL: the
    gateway borrows a pod for the request and gives it back afterwards."""
    client, gw, url = gateway_app
    released = []

    class _Handle:
        pod_id = "pod-1"

    class _Pool:
        def acquire(self):
            return _Handle(), url

        def release(self, handle):
            released.append(handle.pod_id)

    import codai.api.runpod_worker as rw
    monkeypatch.setattr(rw, "get_capability_pool", lambda cap, block: _Pool())
    gw.capability_pods = lambda: {"images": {"max_pods": 2}}
    gw.capability_endpoints = lambda: {}

    r = client.post("/v1/images/generations", json={"model": "sdxl", "prompt": "cat"})
    assert r.json()["served_by"] == "remote"
    # The pod is handed back even on the happy path, so the idle reaper can see it.
    assert released == ["pod-1"]


def test_runpod_shorthand_endpoint_uses_the_pool(gateway_app, monkeypatch):
    client, gw, url = gateway_app
    import codai.api.runpod_worker as rw
    asked = []

    class _Pool:
        def acquire(self):
            return object(), url

        def release(self, handle):
            pass

    monkeypatch.setattr(rw, "get_capability_pool",
                        lambda cap, block: asked.append(cap) or _Pool())
    gw.capability_endpoints = lambda: {"video": "runpod"}
    gw.capability_pods = lambda: {}

    assert client.post("/v1/video/generations",
                       json={"model": "wan"}).json()["served_by"] == "remote"
    assert asked == ["video"]


def test_pod_that_cannot_be_provisioned_is_a_502_not_a_local_run(gateway_app, monkeypatch):
    """Falling back to local would quietly load a 30 GB model on the wrong box."""
    client, gw, _ = gateway_app
    import codai.api.runpod_worker as rw

    def _boom(cap, block):
        raise RuntimeError("RunPod is not enabled")

    monkeypatch.setattr(rw, "get_capability_pool", _boom)
    gw.capability_pods = lambda: {"images": {}}
    gw.capability_endpoints = lambda: {}

    r = client.post("/v1/images/generations", json={"model": "sdxl"})
    assert r.status_code == 502 and "RunPod is not enabled" in r.json()["error"]["message"]


def test_pod_plan_per_engine():
    from codai.api.runpod_worker import (parse_model_runpod, pod_plan,
                                         LLAMACPP_POD_IMAGE, DEFAULT_POD_IMAGE)

    vllm = pod_plan(parse_model_runpod({"served_model": "Qwen/Qwen3.5-9B"}),
                    "Qwen/Qwen3.5-9B")
    assert vllm["image"] == DEFAULT_POD_IMAGE and vllm["health_path"] == "/v1/models"

    gguf = pod_plan(parse_model_runpod({"hf_gguf": "user/repo:Q4_K_M"}), "m",
                    model_path="/AI/m.gguf")
    assert gguf["image"] == LLAMACPP_POD_IMAGE and "-hf user/repo:Q4_K_M" in gguf["args"]

    # A coderai pod serves whole capabilities; it is ready as soon as /healthz answers.
    cod = pod_plan(parse_model_runpod({"engine": "coderai", "image": "reg/coderai:base"}),
                   "images")
    assert cod["image"] == "reg/coderai:base" and cod["health_path"] == "/healthz"

    # …but it needs an image: we cannot invent a registry the pod can pull from.
    with pytest.raises(RuntimeError, match="set `image`"):
        pod_plan(parse_model_runpod({"engine": "coderai"}), "images")

    # `custom` takes verbatim args and an explicit probe.
    cus = pod_plan(parse_model_runpod({"engine": "custom", "image": "me/thing:1",
                                       "docker_args": "--serve", "health_path": "/ping"}),
                   "x")
    assert cus["args"] == "--serve" and cus["health_path"] == "/ping"


def test_every_v1_endpoint_is_either_mapped_or_deliberately_skipped():
    """A new /v1 endpoint that nobody mapped would silently stay local."""
    import re
    from codai.api.remote_gateway import capability_for, _SKIP_PATHS

    root = Path(__file__).resolve().parents[1] / "codai" / "api"
    pat = re.compile(r'@router\.(?:post|get|put|delete)\("(/v1[^"]*)"')
    paths = {m for f in root.glob("*.py") for m in pat.findall(f.read_text())}
    assert paths, "no /v1 routes found — did the decorator style change?"

    from codai.api.remote_gateway import _ORCHESTRATION_PREFIXES

    unmapped = sorted(p for p in paths
                      if not capability_for(p) and p not in _SKIP_PATHS
                      and not p.startswith(_ORCHESTRATION_PREFIXES))
    assert unmapped == [], f"unmapped /v1 endpoints: {unmapped}"


def test_generated_file_urls_are_rewritten_and_followed_back(gateway_app, monkeypatch):
    """`response_format: url` returns a URL built from the REMOTE's base. Left
    alone it points at a host the client may not reach, and fetching it here is a
    404 — the file is on the pod."""
    client, gw, url = gateway_app
    from fastapi import Request

    app = client.app

    @app.post("/v1/images/edits")
    async def edits(req: Request):          # pragma: no cover - remote in this test
        return {"served_by": "local"}

    @app.get("/v1/files/{name}")
    async def files(name: str):             # pragma: no cover - remote in this test
        return {"served_by": "local", "name": name}

    class _Files(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, obj):
            raw = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            # exactly what a remote coderai answers with response_format=url
            self._send({"data": [{"url": "http://pod-xyz:8000/v1/files/out-1.png"}]})

        def do_GET(self):
            self._send({"served_by": "remote-files", "path": self.path})

    remote, shutdown = _serve(_Files)
    try:
        gw.capability_endpoints = lambda: {"images": remote}
        r = client.post("/v1/images/edits", json={"model": "sdxl"})
        got = r.json()["data"][0]["url"]
        # rewritten to this instance...
        assert got.endswith("/v1/files/out-1.png") and "pod-xyz" not in got
        # ...and we remember which remote actually holds it
        assert gw.file_origin("out-1.png") == remote

        # so fetching it follows back to that remote, not to the local route
        gw.capability_endpoints = lambda: {}
        assert client.get("/v1/files/out-1.png").json()["served_by"] == "remote-files"
        # a file we never saw is served locally
        assert client.get("/v1/files/unknown.png").json()["served_by"] == "local"
    finally:
        shutdown()


def test_capabilities_can_share_one_pod_pool(monkeypatch):
    """A pod is a whole GPU. Three capabilities naming one pool must rent one
    card between them, not three."""
    import codai.api.runpod_worker as rw

    class _Acct:
        enabled = True

    monkeypatch.setattr("codai.models.manager.get_active_runpod_config", lambda: _Acct())
    monkeypatch.setattr(rw, "_pools", {})
    monkeypatch.setattr(rw, "_ensure_scaler", lambda: None)

    settings = {"pool": "media", "image": "reg/coderai:base", "max_pods": 3,
                "max_hourly_usd": 0.6}
    images = rw.get_capability_pool("images", settings)
    video = rw.get_capability_pool("video", {"pool": "media"})
    tts = rw.get_capability_pool("tts", {"pool": "media"})

    assert images is video is tts
    assert images.model_key == "capability:media"
    # A bare reference must not reset the pool it points at to defaults.
    assert images.mcfg.max_pods == 3 and images.mcfg.image == "reg/coderai:base"

    # Without `pool`, every capability still gets its own.
    ocr = rw.get_capability_pool("ocr", {"image": "reg/coderai:base"})
    assert ocr is not images and ocr.model_key == "capability:ocr"


def test_unreachable_remote_is_a_clean_502(gateway_app):
    client, gw, _ = gateway_app
    gw.capability_endpoints = lambda: {"images": "http://127.0.0.1:1"}

    r = client.post("/v1/images/generations", json={"model": "sdxl"})
    assert r.status_code == 502
    assert "unreachable" in r.json()["error"]["message"]


def test_capability_pods_default_to_the_published_images():
    """Turning a capability remote should not require knowing an image name."""
    from codai.api.runpod_worker import default_capability_image

    assert default_capability_image("images") == "ghcr.io/nextime/coderai-images:latest"
    assert default_capability_image("video") == "ghcr.io/nextime/coderai-video:latest"
    # Capabilities that ride another profile's dependency set.
    assert default_capability_image("rerank").endswith("coderai-embeddings:latest")
    assert default_capability_image("speaker").endswith("coderai-stt:latest")
    assert default_capability_image("stems").endswith("coderai-audio:latest")
    # No published image: say so rather than default to a tag that 404s minutes
    # into a pod boot.
    assert default_capability_image("pipelines") == ""
    assert default_capability_image("loras") == ""


def test_the_default_image_repo_is_overridable(monkeypatch):
    """A fork or a private registry must be able to take over the whole set."""
    import importlib
    monkeypatch.setenv("CODERAI_CAPABILITY_IMAGE_REPO", "registry.example.com/me/cai")
    monkeypatch.setenv("CODERAI_CAPABILITY_IMAGE_TAG", "0.2.2")
    import codai.api.runpod_worker as rw
    importlib.reload(rw)
    try:
        assert rw.default_capability_image("tts") == "registry.example.com/me/cai-tts:0.2.2"
    finally:
        monkeypatch.undo()
        importlib.reload(rw)


def test_a_capability_pool_picks_up_the_default_image(monkeypatch):
    import codai.api.runpod_worker as rw

    class _Acct:
        enabled = True

    monkeypatch.setattr("codai.models.manager.get_active_runpod_config", lambda: _Acct())
    monkeypatch.setattr(rw, "_pools", {})
    monkeypatch.setattr(rw, "_ensure_scaler", lambda: None)

    pool = rw.get_capability_pool("video", {})          # nothing configured at all
    assert pool.mcfg.image == "ghcr.io/nextime/coderai-video:latest"
    assert pool.mcfg.engine == "coderai"

    # An explicit image still wins.
    monkeypatch.setattr(rw, "_pools", {})
    own = rw.get_capability_pool("video", {"image": "me/mine:1"})
    assert own.mcfg.image == "me/mine:1"


# --------------------------------------------------------------------------- #
# per-model placement: two models of the SAME kind, placed differently
# --------------------------------------------------------------------------- #
@pytest.fixture()
def video_models(monkeypatch):
    entries = {
        "wan-local": {"path": "Wan-AI/Wan2.2-I2V-A14B", "model_type": "video_models",
                      "placement": "local"},
        "wan-remote": {"path": "Wan-AI/Wan2.2-TI2V-5B", "model_type": "video_models",
                       "backend": "runpod",
                       "runpod": {"mode": "pods", "max_hourly_usd": 0.9}},
        "wan-plain": {"path": "some/other-video", "model_type": "video_models"},
    }
    import codai.models.manager as mgr
    import codai.api.runpod_worker as rw
    monkeypatch.setattr(mgr, "_model_entry_for", lambda n: entries.get(n))
    monkeypatch.setattr(mgr, "get_active_runpod_config",
                        lambda: type("A", (), {"enabled": True})())
    monkeypatch.setattr(rw, "_pools", {})
    monkeypatch.setattr(rw, "_ensure_scaler", lambda: None)
    return entries


def _body(model):
    return json.dumps({"model": model, "prompt": "x"}).encode()


def test_two_video_models_can_be_placed_differently(video_models, monkeypatch):
    import codai.api.remote_gateway as gw
    from codai.api.remote_gateway import resolve_target
    import codai.api.runpod_worker as rw

    path = "/v1/video/generations"
    # The whole video capability is remote…
    monkeypatch.setattr(gw, "capability_endpoints", lambda: {"video": "http://pod:8000"})

    # …but a model pinned local stays local.
    assert resolve_target(path, "POST", "", _body("wan-local"), "application/json") is None

    # A model pinned to RunPod gets its OWN pod, with its own budget.
    t = resolve_target(path, "POST", "", _body("wan-remote"), "application/json")
    assert t.pool is not None and t.pool.mcfg.max_hourly_usd == 0.9

    # And a model with nothing set follows the capability.
    t2 = resolve_target(path, "POST", "", _body("wan-plain"), "application/json")
    assert t2.url == "http://pod:8000" and t2.pool is None


def test_a_video_models_pod_uses_the_video_image_and_is_told_about_the_model(video_models):
    from codai.api.runpod_worker import pod_plan, parse_model_runpod

    entry = video_models["wan-remote"]
    plan = pod_plan(parse_model_runpod(entry["runpod"]), "wan-remote", "wan-remote",
                    entry=entry, api_key="tok")
    # vLLM cannot serve a diffusion model: the model's KIND picks the image.
    assert plan["image"] == "ghcr.io/nextime/coderai-video:latest"
    assert plan["health_path"] == "/healthz"
    # A fresh pod has an empty catalogue and must be told what it is serving.
    seeded = json.loads(plan["env"]["CODERAI_SEED_MODELS"])
    assert seeded[0]["path"] == "Wan-AI/Wan2.2-TI2V-5B"
    assert seeded[0]["model_type"] == "video_models"
    assert plan["env"]["CODERAI_API_TOKEN"] == "tok"


def test_a_local_only_path_is_not_seeded():
    """A /AI/... path means nothing on a rented machine; don't pretend it does."""
    from codai.api.runpod_worker import seed_model_env
    assert seed_model_env({"path": "/AI/models/local-only", "model_type": "video_models"}) == ""
    assert seed_model_env({"path": "org/repo", "model_type": "video_models"}) != ""


# --------------------------------------------------------------------------- #
# orchestration stays local
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [
    "/v1/pipelines/story", "/v1/pipelines/image-to-video",
    "/v1/characters/generate", "/v1/environments/generate"])
def test_orchestration_endpoints_are_never_forwarded(path, monkeypatch):
    """These are a sequence of calls to other endpoints. The chain stays here and
    each step is placed by its own model; forwarding the chain would place every
    step by the pod's catalogue instead."""
    import codai.api.remote_gateway as gw
    from codai.api.remote_gateway import resolve_target, capability_for

    monkeypatch.setattr(gw, "capability_endpoints",
                        lambda: {"pipelines": "http://pod:8000",
                                 "characters": "http://pod:8000",
                                 "environments": "http://pod:8000"})
    monkeypatch.setattr(gw, "_model_remote", lambda m: "http://pod:8000")
    assert capability_for(path) == ""
    assert resolve_target(path, "POST", "", b'{"model":"x"}', "application/json") is None


# --------------------------------------------------------------------------- #
# LoRAs follow the request to the remote
# --------------------------------------------------------------------------- #
def test_loras_are_uploaded_to_the_remote_once(tmp_path, monkeypatch):
    import hashlib
    from codai.api import remote_gateway as gw

    weights = tmp_path / "style.safetensors"
    weights.write_bytes(b"fake-lora-weights" * 100)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()

    monkeypatch.setattr("codai.api.loras.resolve_lora_ref",
                        lambda spec: str(weights) if spec.get("model") else None)

    calls = {"checked": 0, "uploaded": 0}
    have = {"blob": False}

    class _Resp:
        def __init__(self, code): self.status_code = code
        def raise_for_status(self): pass

    def fake_get(url, **kw):
        calls["checked"] += 1
        return _Resp(200 if have["blob"] else 404)

    def fake_post(url, **kw):
        calls["uploaded"] += 1
        have["blob"] = True
        return _Resp(200)

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", fake_post)

    body = json.dumps({"model": "sdxl", "prompt": "x",
                       "loras": [{"model": str(weights), "weight": 0.8}]}).encode()
    out = gw.sync_loras("http://pod:8000", "tok", body)
    spec = json.loads(out)["loras"][0]
    # The local path is replaced by the content hash the remote now holds.
    assert spec["id"] == f"sha256:{digest}" and spec["weight"] == 0.8
    assert "model" not in spec
    assert calls["uploaded"] == 1

    # Second request: the remote already has it, so nothing is uploaded again.
    gw.sync_loras("http://pod:8000", "tok", out)
    assert calls["uploaded"] == 1


def test_an_unresolvable_lora_is_left_for_the_remote(monkeypatch):
    """An HF repo id is not a local file — the remote fetches it itself."""
    from codai.api import remote_gateway as gw
    monkeypatch.setattr("codai.api.loras.resolve_lora_ref", lambda spec: None)
    body = json.dumps({"loras": [{"model": "org/some-lora", "weight": 1.0}]}).encode()
    assert gw.sync_loras("http://pod:8000", "", body) is body


# --------------------------------------------------------------------------- #
# where a pod gets the weights
# --------------------------------------------------------------------------- #
def test_hf_repo_id_is_recovered_from_a_local_cache_path():
    """Most local models came from HuggingFace: the cache path still says which
    repo, so a pod can fetch them without anyone typing an id."""
    from codai.api.runpod_worker import resolve_model_source, parse_model_runpod

    cached = {"path": "/AI/huggingface/hub/models--Qwen--Qwen3.5-9B/"
                      "snapshots/abc123/model.safetensors"}
    assert resolve_model_source(cached, parse_model_runpod({})) == ("hf", "Qwen/Qwen3.5-9B")
    # An entry whose path IS a repo id needs nothing.
    assert resolve_model_source({"path": "Qwen/Qwen3.5-9B"},
                                parse_model_runpod({})) == ("hf", "Qwen/Qwen3.5-9B")
    # A bare local file with nothing else: we cannot invent a source.
    assert resolve_model_source({"path": "/AI/guffcache/merged.gguf"},
                                parse_model_runpod({})) == ("", "")


def test_explicit_url_and_upload_sources():
    from codai.api.runpod_worker import resolve_model_source, parse_model_runpod

    entry = {"path": "/AI/guffcache/merged-Q4.gguf"}
    url = parse_model_runpod({"model_url": "https://host/merged-Q4.gguf"})
    assert resolve_model_source(entry, url) == ("url", "https://host/merged-Q4.gguf")
    up = parse_model_runpod({"source": "upload"})
    assert resolve_model_source(entry, up) == ("upload", "/AI/guffcache/merged-Q4.gguf")


def test_each_pod_server_gets_a_source_it_can_actually_use():
    from codai.api.runpod_worker import (_llamacpp_docker_args, _vllm_docker_args,
                                         parse_model_runpod)

    gguf = {"path": "/AI/guffcache/merged-Q4.gguf"}
    # llama.cpp downloads a URL itself (-mu) — the usual way to serve a one-off GGUF.
    args = _llamacpp_docker_args(parse_model_runpod({"model_url": "https://h/m.gguf"}),
                                 "m", gguf)
    assert "-mu https://h/m.gguf" in args

    # vLLM cannot download a URL itself, but the POD can before vLLM starts, so
    # the args name the staged path rather than refusing the URL.
    staged = _vllm_docker_args(parse_model_runpod({"model_url": "https://h/m.bin"}),
                               "/AI/local/m", {"path": "/AI/local/m"})
    assert "--model /runpod-volume/staged/m.bin" in staged
    # Upload cannot work for a server that needs the file before it starts.
    with pytest.raises(RuntimeError, match="cannot work here"):
        _llamacpp_docker_args(parse_model_runpod({"source": "upload"}), "m", gguf)
    # Nothing at all: a clear instruction, not a mystery.
    with pytest.raises(RuntimeError, match="where to get the weights"):
        _llamacpp_docker_args(parse_model_runpod({}), "m", gguf)


def test_a_coderai_pod_is_seeded_with_something_it_can_fetch():
    from codai.api.runpod_worker import pod_plan, parse_model_runpod

    entry = {"path": "/AI/models/my-merge", "model_type": "video_models"}
    plan = pod_plan(parse_model_runpod({"model_url": "https://host/my-merge.safetensors"}),
                    "my-merge", "my-merge", entry=entry)
    seeded = json.loads(plan["env"]["CODERAI_SEED_MODELS"])[0]
    # The local path is replaced by something the pod can actually resolve.
    assert seeded["path"] == "https://host/my-merge.safetensors"


def test_keep_warm_is_off_by_default_and_means_one_pod():
    from codai.api.runpod_worker import parse_model_runpod

    assert parse_model_runpod({}).keep_warm is False
    assert parse_model_runpod({}).min_pods == 0          # scale to zero: costs nothing idle
    warm = parse_model_runpod({"keep_warm": True})
    assert warm.keep_warm is True and warm.min_pods == 1
    # An explicit larger min_pods is not reduced by it.
    assert parse_model_runpod({"keep_warm": True, "min_pods": 3}).min_pods == 3


# --------------------------------------------------------------------------- #
# LoRA / QLoRA on TEXT models
# --------------------------------------------------------------------------- #
def test_text_lora_config_shapes_all_resolve():
    """The config grew several shapes over time; all of them must work."""
    from codai.models.text_loras import configured_specs

    one = configured_specs({"lora_path": "/AI/loras/style.safetensors", "lora_scale": 0.7})
    assert one == [{"source": "/AI/loras/style.safetensors", "weight": 0.7,
                    "name": "style"}]

    many = configured_specs({"loras": [
        {"path": "org/adapter-a", "weight": 0.5, "name": "a"},
        {"model": "/AI/loras/b.safetensors"}]})
    assert [s["name"] for s in many] == ["a", "b"]
    assert many[0]["weight"] == 0.5 and many[1]["weight"] == 1.0

    # The same adapter named twice is applied once, not twice.
    dup = configured_specs({"lora_path": ["org/x", "org/x"]})
    assert len(dup) == 1
    assert configured_specs({}) == []


def test_llamacpp_needs_a_gguf_adapter(tmp_path):
    """llama.cpp cannot load a PEFT safetensors: report it instead of failing
    deep inside the model load."""
    from codai.models.text_loras import gguf_adapter

    peft = tmp_path / "adapter"
    peft.mkdir()
    (peft / "adapter_model.safetensors").write_bytes(b"x")
    assert gguf_adapter({"source": str(peft)}) == ""

    conv = tmp_path / "adapter.gguf"
    conv.write_bytes(b"x")
    assert gguf_adapter({"source": str(conv)}) == str(conv)


def test_vllm_serves_configured_adapters():
    from codai.api.vllm_worker import _lora_args

    args = _lora_args({"lora_path": "/AI/loras/style.safetensors"})
    assert "--enable-lora" in args
    assert "style=/AI/loras/style.safetensors" in args
    # A rank ceiling that refuses real adapters is a bad default to inherit.
    assert "--max-lora-rank" in args
    assert _lora_args({}) == []


def test_a_pod_only_gets_adapters_it_can_resolve(capsys):
    """A pod cannot read this disk. An HF repo id travels; a local path does not,
    and must be reported rather than baked into a doomed launch command."""
    from codai.api.runpod_worker import _vllm_docker_args, parse_model_runpod

    remote_ok = _vllm_docker_args(parse_model_runpod({}), "Qwen/Qwen3.5-9B",
                                  {"path": "Qwen/Qwen3.5-9B", "lora_path": "org/my-lora"})
    assert "--enable-lora" in remote_ok and "my-lora=org/my-lora" in remote_ok

    local_only = _vllm_docker_args(parse_model_runpod({}), "Qwen/Qwen3.5-9B",
                                   {"path": "Qwen/Qwen3.5-9B",
                                    "lora_path": "/AI/loras/local.safetensors"})
    assert "--enable-lora" not in local_only
    assert "not sent to the pod" in capsys.readouterr().out


def test_a_coderai_pod_is_told_about_the_models_adapters():
    """A coderai pod applies LoRAs itself, so its seed must carry them — along
    with load_in_4bit, which is what makes a QLoRA load against its own base."""
    from codai.api.runpod_worker import seed_model_env

    seeded = json.loads(seed_model_env(
        {"path": "org/model", "model_type": "text_models",
         "lora_path": "org/adapter", "lora_scale": 0.6, "load_in_4bit": True}))[0]
    # Normalised to one shape the pod can act on, weight preserved.
    assert seeded["loras"] == [{"path": "org/adapter", "weight": 0.6,
                                "name": "adapter"}]
    # load_in_4bit is what makes a QLoRA adapter load against its own base.
    assert seeded["load_in_4bit"] is True


# --------------------------------------------------------------------------- #
# a LOCAL text adapter on RunPod
# --------------------------------------------------------------------------- #
def test_a_local_text_adapter_routes_to_a_coderai_pod():
    """vLLM and llama.cpp resolve adapters themselves at launch and have no
    endpoint to receive one, so an adapter that exists only here forces the pod
    that CAN be sent it."""
    from codai.api.runpod_worker import resolve_pod_engine, pod_plan, parse_model_runpod

    plain = {"path": "Qwen/Qwen3.5-9B", "model_type": "text_models"}
    assert resolve_pod_engine(parse_model_runpod({}), "q", plain["path"], plain) == "vllm"

    local = dict(plain, lora_path="/AI/loras/mine.safetensors")
    assert resolve_pod_engine(parse_model_runpod({}), "q", plain["path"], local) == "coderai"
    assert pod_plan(parse_model_runpod({}), "q", "q",
                    entry=local)["image"].endswith("coderai-text:latest")

    # A published adapter needs none of that: vLLM can fetch it itself.
    published = dict(plain, lora_path="org/published-lora")
    assert resolve_pod_engine(parse_model_runpod({}), "q", plain["path"],
                              published) == "vllm"


def test_the_pod_is_told_the_adapters_content_id(tmp_path, monkeypatch):
    """The pod is created before the adapter can be sent, so it is told the hash
    the file WILL have — computed here — and sent the bytes afterwards."""
    import hashlib
    from codai.api.runpod_worker import seed_model_env

    adapter = tmp_path / "mine.safetensors"
    adapter.write_bytes(b"adapter-bytes" * 50)
    digest = hashlib.sha256(adapter.read_bytes()).hexdigest()

    seeded = json.loads(seed_model_env(
        {"path": "Qwen/Qwen3.5-9B", "model_type": "text_models",
         "lora_path": str(adapter), "lora_scale": 0.7}))[0]
    assert seeded["loras"] == [{"path": f"sha256:{digest}", "weight": 0.7,
                                "name": "mine"}]
    # The local path must not survive: it means nothing on the pod.
    assert "lora_path" not in seeded


def test_local_adapters_are_sent_to_the_pod_once(tmp_path, monkeypatch):
    from codai.api import remote_gateway as gw

    adapter = tmp_path / "mine.safetensors"
    adapter.write_bytes(b"weights" * 100)

    calls = {"get": 0, "post": 0}
    have = {"blob": False}

    class _R:
        def __init__(self, code): self.status_code = code
        def raise_for_status(self): pass

    monkeypatch.setattr("requests.get",
                        lambda url, **kw: (calls.__setitem__("get", calls["get"] + 1),
                                           _R(200 if have["blob"] else 404))[1])

    def _post(url, **kw):
        calls["post"] += 1
        have["blob"] = True
        return _R(200)

    monkeypatch.setattr("requests.post", _post)

    entry = {"path": "Qwen/Qwen3.5-9B", "model_type": "text_models",
             "lora_path": str(adapter), "lora_scale": 0.5}
    out = gw.ensure_text_loras("http://pod:8000", "tok", entry)
    assert calls["post"] == 1
    # The config handed to the remote names the adapter by content, not by path.
    assert out["loras"][0]["path"].startswith("sha256:")
    assert out["loras"][0]["weight"] == 0.5
    assert "lora_path" not in out

    gw.ensure_text_loras("http://pod:8000", "tok", entry)
    assert calls["post"] == 1          # already there — nothing re-sent


def test_a_published_adapter_is_not_uploaded(monkeypatch):
    from codai.api import remote_gateway as gw

    def _boom(*a, **k):
        raise AssertionError("an HF repo id must never be uploaded")

    monkeypatch.setattr("requests.post", _boom)
    out = gw.ensure_text_loras("http://pod:8000", "tok",
                               {"path": "org/m", "lora_path": "org/adapter"})
    assert out["loras"][0]["path"] == "org/adapter"


def test_a_content_addressed_adapter_resolves_through_the_blob_store(monkeypatch):
    """On the pod the adapter arrives in the blob store and the config names it
    by hash — which has to resolve back to a file."""
    from codai.models import text_loras

    monkeypatch.setattr("codai.api.loras.resolve_lora_ref",
                        lambda spec: "/blobs/abc" if spec.get("id") else None)
    monkeypatch.setattr(text_loras.os.path, "exists", lambda p: p == "/blobs/abc")
    assert text_loras.local_path("sha256:" + "a" * 64) == "/blobs/abc"
    # And it is not mistaken for something a remote could fetch by name.
    assert text_loras.is_portable("sha256:" + "a" * 64) is False
    assert text_loras.is_portable("org/adapter") is True


# --------------------------------------------------------------------------- #
# staging: the pod downloads before its server starts
# --------------------------------------------------------------------------- #
def test_a_vllm_pod_can_download_a_model_before_starting():
    """The third way onto a vLLM pod: not on HuggingFace, not uploadable (no
    receiver) — but downloadable from any host the pod can reach, BEFORE vLLM
    starts. That needs the image's ENTRYPOINT overridden, since vLLM is it."""
    from codai.api.runpod_worker import pod_plan, parse_model_runpod, STAGE_DIR

    entry = {"path": "/AI/models/merged", "model_type": "text_models"}
    plan = pod_plan(parse_model_runpod(
        {"engine": "vllm", "model_url": "https://host/merged.tar",
         "model_url_is_tar": True}), "m", "m", entry=entry)

    assert plan["entrypoint"] == ["/bin/sh", "-c"]
    script = plan["start_cmd"][0]
    # Downloads the tar, extracts to a directory of the same name without the
    # suffix, and serves THAT — one definition of the path, not two.
    assert f"{STAGE_DIR}/merged.tar" in script          # the archive
    assert f"-C '{STAGE_DIR}/merged'" in script          # extracted here
    assert script.rstrip().endswith(f"--served-model-name {STAGE_DIR}/merged")
    assert script.count("--model ") == 1                 # never two --model flags


def test_llamacpp_downloads_its_own_model_but_stages_an_adapter():
    """llama-server fetches a model URL itself (-mu), so only the adapter needs
    staging — and it takes one, with its scale."""
    from codai.api.runpod_worker import pod_plan, parse_model_runpod, STAGE_DIR

    entry = {"path": "Qwen/Qwen3.5-9B", "model_type": "text_models",
             "loras": [{"path": "https://host/a.safetensors", "weight": 0.7,
                        "name": "mine"}]}
    plan = pod_plan(parse_model_runpod({"engine": "llamacpp", "hf_gguf": "u/r:Q4"}),
                    "q", "q", entry=entry)
    script = plan["start_cmd"][0]
    assert f"--lora-scaled {STAGE_DIR}/loras/a.safetensors 0.7" in script
    assert "-hf u/r:Q4" in script          # the model still comes from HF


def test_upload_to_a_vllm_pod_says_what_to_do_instead():
    from codai.api.runpod_worker import _vllm_docker_args, parse_model_runpod

    with pytest.raises(RuntimeError, match="model_url"):
        _vllm_docker_args(parse_model_runpod({"source": "upload"}),
                          "/AI/local/m", {"path": "/AI/local/m"})


def test_staging_quotes_what_it_downloads():
    """The URL is configuration, but it still ends up in a shell command."""
    from codai.api.runpod_worker import stage_script

    script = stage_script(
        [{"url": "https://host/a'; rm -rf /; echo '", "dest": "/tmp/x"}], "server")
    # The quote is escaped, so the injected text stays one argument.
    assert "rm -rf /" in script and "'\\''" in script
    assert script.rstrip().endswith("exec server")


def test_no_staging_when_nothing_needs_downloading():
    """An HF repo id needs no entrypoint override — leave the image alone."""
    from codai.api.runpod_worker import pod_plan, parse_model_runpod

    plan = pod_plan(parse_model_runpod({}), "Qwen/Qwen3.5-9B", "q",
                    entry={"path": "Qwen/Qwen3.5-9B", "model_type": "text_models"})
    assert "entrypoint" not in plan and "start_cmd" not in plan


# --------------------------------------------------------------------------- #
# test run
# --------------------------------------------------------------------------- #
@pytest.fixture()
def test_run(monkeypatch):
    """The test-run endpoint with a fake dispatcher, so no model is loaded."""
    import codai.models.manager as mgr
    import codai.broker.asgi_bridge as bridge

    entries = {
        "my-llm": {"path": "Qwen/Qwen3.5-9B", "model_type": "text_models"},
        "my-video": {"path": "org/wan", "model_type": "video_models",
                     "backend": "runpod", "runpod": {"mode": "pods"}},
        "my-embed": {"path": "org/e5", "model_type": "embedding_models",
                     "service_url": "http://box:8000"},
        "pinned": {"path": "org/x", "model_type": "text_models", "placement": "local"},
    }
    monkeypatch.setattr(mgr, "_model_entry_for", lambda n: entries.get(n))
    sent = {}

    async def fake_exec(request, *, method, path, headers=None, body=b""):
        sent.update({"path": path, "headers": headers or {}, "body": json.loads(body)})
        if path == "/v1/chat/completions":
            return {"status_code": 200, "body": json.dumps(
                {"choices": [{"message": {"content": "OK"}}]}).encode()}
        if path == "/v1/embeddings":
            return {"status_code": 200, "body": json.dumps(
                {"data": [{"embedding": [0.0] * 768}]}).encode()}
        return {"status_code": 503, "body": json.dumps({"detail": "no engine"}).encode()}

    monkeypatch.setattr(bridge, "execute_api_request", fake_exec)
    return sent


def _run_test(model, where="auto"):
    from codai.api.model_test import test_model, ModelTestRequest

    class _Req:
        headers = {}

    return asyncio.run(test_model(ModelTestRequest(model=model, where=where), _Req()))


def test_a_test_run_makes_a_real_request_for_the_models_kind(test_run):
    llm = _run_test("my-llm")
    assert llm["ok"] and llm["ran"] == "/v1/chat/completions" and llm["sample"] == "OK"

    emb = _run_test("my-embed")
    assert emb["ran"] == "/v1/embeddings" and emb["sample"] == "768 dimensions"
    # It reports WHERE it ran — the point of the test.
    assert emb["where"] == "remote" and emb["target"] == "http://box:8000"


def test_a_test_run_can_force_where_it_runs(test_run):
    """You cannot verify a pod by sending it a request the config serves locally."""
    from codai.api.remote_gateway import PLACEMENT_HEADER

    _run_test("my-llm", where="local")
    assert test_run["headers"][PLACEMENT_HEADER] == "local"
    # Forcing remote only dispatches for a model that really is configured
    # remote; my-embed has a service_url.
    _run_test("my-embed", where="runpod")
    assert test_run["headers"][PLACEMENT_HEADER] == "remote"


def test_kinds_without_honest_input_report_reachability_and_say_so(test_run):
    """A face swap needs real faces: a drawn one does not survive face
    detection, and borrowing a real person's photo is not ours to do. Don't run
    one, and don't pretend the check proved more than it did.

    Video USED to be here on cost grounds. It is probed for real now — the
    smallest clip a model can make — because "expensive" was an argument for
    asking for less, not for proving nothing."""
    import codai.models.manager as mgr
    entry = {"path": "org/swap", "model_type": "image_models"}
    saved = mgr._model_entry_for
    mgr._model_entry_for = lambda n: entry
    try:
        import codai.api.model_test as mt
        saved_describe = mt._describe
        mt._describe = lambda m: (entry, "faceswap")
        try:
            out = _run_test("my-swap")
        finally:
            mt._describe = saved_describe
    finally:
        mgr._model_entry_for = saved
    assert out["ran"] == "reachability"
    assert "no generation was run" in out["note"]
    assert "real images" in out["note"]


def test_a_failing_probe_reports_the_real_error(test_run):
    out = _run_test("pinned")          # the fake dispatcher 503s for this path? no:
    assert out["where"] == "local"     # pinned local is honoured
    # A kind whose probe fails surfaces the server's own message, not a generic one.
    import codai.models.manager as mgr
    entry = {"path": "org/tts", "model_type": "tts_models"}
    saved = mgr._model_entry_for
    mgr._model_entry_for = lambda n: entry
    try:
        bad = _run_test("my-tts")
    finally:
        mgr._model_entry_for = saved
    assert bad["ok"] is False and bad["error"] == "no engine" and bad["status"] == 503


def test_the_placement_override_is_honoured_by_the_router():
    """'local' must win over any configuration, and 'remote' must refuse to fall
    back — otherwise a forced test could quietly pass in the wrong place."""
    from codai.api.remote_gateway import resolve_target

    body = json.dumps({"model": "anything"}).encode()
    assert resolve_target("/v1/images/generations", "POST", "", body,
                          "application/json", force="local") is None
    with pytest.raises(RuntimeError, match="nothing configures"):
        resolve_target("/v1/images/generations", "POST", "", body,
                       "application/json", force="remote")


def test_forcing_runpod_fails_when_nothing_is_configured_remote(test_run):
    """The bug this exists to prevent: /v1/chat/completions is outside the
    gateway, so a forced header is ignored there — a text model with no remote
    config would have been served LOCALLY while the result said "remote"."""
    out = _run_test("my-llm", where="runpod")
    assert out["ok"] is False
    assert "nothing configures" in out["error"]
    assert out["ran"] == ""          # nothing was dispatched at all

    # A model that IS configured remote still runs.
    out2 = _run_test("my-video", where="runpod")
    assert out2["where"] == "runpod" and "nothing configures" not in str(out2)


def test_a_200_with_no_output_is_not_a_pass(test_run, monkeypatch):
    """A reasoning model can answer with nothing when the token budget is tight.
    HTTP 200 plus zero content is a failure, not a pass."""
    import codai.broker.asgi_bridge as bridge

    async def empty_reply(request, *, method, path, headers=None, body=b""):
        return {"status_code": 200, "body": json.dumps(
            {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
             "usage": {"completion_tokens": 0}}).encode()}

    monkeypatch.setattr(bridge, "execute_api_request", empty_reply)
    out = _run_test("my-llm")
    assert out["status"] == 200
    assert out["ok"] is False
    assert "no text" in out["error"] and "completion_tokens=0" in out["error"]


def test_the_text_probe_leaves_room_for_a_reasoning_model():
    from codai.api.model_test import _probe_for

    _, body = _probe_for("text", "m")
    assert body["max_tokens"] >= 64


def test_a_capability_pod_is_told_what_it_can_serve(monkeypatch):
    """Observed live: a healthy embeddings pod answered "Model 'bge-m3' is not
    available. Use one of: " with nothing after the colon. A pod starts with an
    empty catalogue; a capability pod serves many models, so it is seeded with
    this deployment's models of that capability."""
    import codai.admin.routes as ar
    import codai.api.runpod_worker as rw

    class _CM:
        models_data = {
            "embedding_models": [
                {"path": "BAAI/bge-m3", "model_type": "embedding_models"},
                {"path": "/AI/local/only", "model_type": "embedding_models"},
            ],
            "image_models": [{"path": "org/sdxl", "model_type": "image_models"}],
        }

    monkeypatch.setattr(ar, "config_manager", _CM())
    seeds = rw.capability_seed_entries("embeddings")
    # Only what the pod can fetch: a local path would register a model it could
    # never load, turning "not available" into a confusing load failure.
    assert [s["path"] for s in seeds] == ["BAAI/bge-m3"]

    plan = rw.pod_plan(rw.parse_model_runpod({"engine": "coderai", "image": "img"}),
                       "embeddings", "capability:embeddings", api_key="tok",
                       seed_entries=seeds)
    registered = json.loads(plan["env"]["CODERAI_SEED_MODELS"])
    assert [r["path"] for r in registered] == ["BAAI/bge-m3"]
    # The token must survive the same path — both travel in the pod's env.
    assert plan["env"]["CODERAI_API_TOKEN"] == "tok"


# --------------------------------------------------------------------------- #
# a pod as an extension of this system: it learns models at runtime
# --------------------------------------------------------------------------- #
class _LearningPod(BaseHTTPRequestHandler):
    """A pod with an empty catalogue that can be taught."""
    protocol_version = "HTTP/1.1"
    known: set = set()
    seen: list = []

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        type(self).seen.append(self.path)
        if self.path == "/v1/models/register":
            for e in (body if isinstance(body, list) else [body]):
                type(self).known.add(e["path"])
            return self._send(200, {"added": sorted(type(self).known)})
        if body.get("model") not in type(self).known:
            return self._send(404, {"detail": f"Model '{body.get('model')}' is not "
                                              "available. Use one of: "})
        return self._send(200, {"data": [{"embedding": [0.1] * 8}]})


@pytest.fixture()
def learning_pod(monkeypatch):
    _LearningPod.known = set()
    _LearningPod.seen = []
    url, shutdown = _serve(_LearningPod)
    import codai.models.manager as mgr
    from codai.api import remote_gateway as gw
    monkeypatch.setattr(mgr, "_model_entry_for", lambda n: (
        {"path": "BAAI/bge-m3", "model_type": "embedding_models"}
        if n == "BAAI/bge-m3" else None))
    monkeypatch.setattr(gw, "_REGISTERED", set())
    yield url, gw
    shutdown()


def test_an_unknown_model_answer_is_recognised():
    from codai.api.remote_gateway import _is_unknown_model

    assert _is_unknown_model(404, b'{"detail": "Model \'x\' is not available. Use one of: "}')
    assert _is_unknown_model(400, b'{"error": "unknown model"}')
    # A real failure must not be mistaken for one, or we would retry forever.
    assert not _is_unknown_model(500, b'{"detail": "out of memory"}')
    assert not _is_unknown_model(200, b'{"data": []}')


def test_a_pod_is_taught_a_model_it_does_not_know(learning_pod):
    url, gw = learning_pod
    assert _LearningPod.known == set()            # empty catalogue at boot

    assert gw.teach_model(url, "", "BAAI/bge-m3") is True
    assert "BAAI/bge-m3" in _LearningPod.known

    # Teaching the same pod twice is wasted work: it keeps what it learns.
    _LearningPod.seen.clear()
    assert gw.teach_model(url, "", "BAAI/bge-m3") is False
    assert _LearningPod.seen == []


def test_a_model_the_pod_could_never_fetch_is_refused_with_a_reason(learning_pod,
                                                                    monkeypatch):
    url, gw = learning_pod
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "_model_entry_for",
                        lambda n: {"path": "/AI/local/only",
                                   "model_type": "embedding_models"})
    # A local path means nothing on a pod: say so rather than register something
    # that would fail later at load time.
    assert gw.teach_model(url, "", "local-only") is False
    assert _LearningPod.known == set()


def test_registering_a_model_makes_it_usable_not_just_listed(monkeypatch):
    """Writing models_data alone is not enough: request validation asks the
    MODEL MANAGER for its allowed identifiers. A live pod proved it — seeded
    with 5 models and still answering "not available. Use one of: " with an
    empty list."""
    import codai.admin.routes as ar
    from codai.api import models_transfer as mt

    registered = {}

    class _MM:
        model_aliases = {}
        _assigned_model_keys = set()

        def set_embedding_model(self, path, config=None):
            registered["path"] = path
            registered["config"] = config

        def set_assigned_models(self, keys):
            registered["assigned"] = set(keys)

    class _CM:
        models_data = {}

    monkeypatch.setattr(ar, "config_manager", _CM())
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "multi_model_manager", _MM())

    state = mt.register_model_runtime(
        {"path": "BAAI/bge-m3", "model_type": "embedding_models", "alias": "bge-m3"},
        "embedding_models")
    assert state == "added"
    # In the catalogue…
    assert _CM.models_data["embedding_models"][0]["path"] == "BAAI/bge-m3"
    # …AND known to the manager, which is what validation consults.
    assert registered["path"] == "BAAI/bge-m3"
    # …AND reachable by the alias a request will actually use.
    assert _MM.model_aliases["bge-m3"] == "BAAI/bge-m3"
    # …AND not filtered out by an engine assignment it was never part of.
    assert {"BAAI/bge-m3", "bge-m3"} <= registered["assigned"]


def test_a_test_run_reports_what_the_pods_are_serving(monkeypatch, test_run):
    """Three failures in a row were diagnosed by finding the pod URL in engine
    logs and asking it by hand. The answer should carry that itself."""
    import codai.api.runpod_worker as rw
    from codai.api import model_test as mt

    monkeypatch.setattr(rw, "pods_status", lambda: [
        {"pod_id": "pod-1", "model": "capability:embeddings", "gpu": "RTX 2000 Ada",
         "state": "ready", "uptime_s": 140, "live_cost_usd": 0.009}])

    class _Pool:
        api_key = "tok"
        _cv = __import__("threading").Condition()
        pods = [rw.PodHandle(pod_id="pod-1", url="http://pod-1:8000",
                             hourly_usd=0.24, started_at=0.0)]

    monkeypatch.setattr(rw, "_pools", {"capability:embeddings": _Pool()})
    monkeypatch.setattr(mt, "_remote_catalogue",
                        lambda url, key: ["BAAI/bge-m3", "bge-m3"])

    out = _run_test("my-llm")
    pods = out["pods"]
    assert pods[0]["pod"] == "pod-1" and pods[0]["gpu"] == "RTX 2000 Ada"
    assert pods[0]["url"] == "http://pod-1:8000"
    # The catalogue is the fact that mattered: an empty list is the whole story.
    assert pods[0]["serves"] == ["BAAI/bge-m3", "bge-m3"]
    assert pods[0]["usd_so_far"] == 0.009


# --- synthesised probes for upload-shaped capabilities -------------------- #

def test_stt_and_ocr_probes_are_real_files_not_reachability_checks():
    """These two were reachability-only, which reported ok:true for a pod that
    had never served a request. espeak and PIL can produce honest input."""
    from codai.api.model_test import _upload_probe, _REACHABILITY_ONLY

    assert "stt" not in _REACHABILITY_ONLY and "ocr" not in _REACHABILITY_ONLY

    path, ct, body = _upload_probe("stt", "whisper0")
    assert path == "/v1/audio/transcriptions"
    assert ct.startswith("multipart/form-data; boundary=")
    assert b"RIFF" in body and b'name="model"' in body

    path, ct, body = _upload_probe("ocr", "surya")
    assert path == "/v1/ocr"
    assert b"\x89PNG" in body and b'name="engine"' in body


def test_a_transcription_that_does_not_match_what_was_spoken_fails():
    """A model returning fluent nonsense passes any 'is it non-empty?' check.
    The probe speaks known words, so the result is checked against them."""
    import json
    from codai.api.model_test import _empty_result, _SPOKEN, _PRINTED

    ok = json.dumps({"text": _SPOKEN.upper() + "."}).encode()
    assert _empty_result(ok, "stt") == ""

    wrong = json.dumps({"text": "entirely unrelated output"}).encode()
    assert "does not match what was sent" in _empty_result(wrong, "stt")

    silent = json.dumps({"text": "   "}).encode()
    assert "returned no text" in _empty_result(silent, "stt")

    # OCR responses are not one shape: flat, paged, or blocked all count.
    paged = json.dumps({"pages": [{"blocks": [{"text": _PRINTED}]}]}).encode()
    assert _empty_result(paged, "ocr") == ""


def test_an_upload_probe_falls_back_rather_than_failing_the_model(monkeypatch):
    """No espeak on this box is our gap, not the model's: fall back to the
    reachability check instead of reporting the model broken."""
    import codai.api.model_test as mt

    def _boom():
        raise FileNotFoundError("espeak")

    monkeypatch.setattr(mt, "_spoken_wav", _boom)
    assert mt._upload_probe("stt", "whisper0") == (None, "", b"")


# --- the test harness must not move a whole capability --------------------- #

def test_the_state_endpoint_reports_the_serving_process_view(monkeypatch):
    """Writing models.json does not change routing: the engine takes the reload
    only when idle. A whole test pass reported false 'not configured' failures
    because it trusted the file instead of asking the process."""
    import asyncio
    import codai.api.model_test as mt

    entry = {"path": "org/m", "model_type": "image_models", "backend": "runpod",
             "runpod": {"mode": "pods"}}
    monkeypatch.setattr(mt, "_describe", lambda m: (entry, "images"))
    monkeypatch.setattr("codai.api.remote_gateway.model_placement",
                        lambda m: ("pod", (entry, entry["runpod"])))

    state = asyncio.run(mt.test_state(model="org/m"))
    assert state["placement"] == "pod"
    assert state["backend"] == "runpod" and state["has_runpod_block"] is True
    assert "pid" in state          # which process answered, not which file exists


def test_the_harness_pins_one_model_and_never_a_capability(tmp_path, monkeypatch):
    """remotes.endpoints[capability] moves EVERY request of that kind — that is
    how production embeddings ended up on a rented pod. The harness must edit
    the single model entry instead."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "runpod_model_test", "tools/runpod_model_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    models = tmp_path / "models.json"
    models.write_text(json.dumps({
        "image_models": [{"path": "org/a", "alias": "a"},
                         {"path": "org/b", "alias": "b"}],
        "text_models": [{"path": "org/t"}],
    }))
    monkeypatch.setattr(mod, "MODELS", str(models))

    assert mod._pin_to_runpod("a", "coderai", 24, 1.0) is True
    after = json.loads(models.read_text())

    pinned = after["image_models"][0]
    assert pinned["backend"] == "runpod" and pinned["runpod"]["engine"] == "coderai"
    # Its sibling of the SAME capability is untouched: that is the whole point.
    assert after["image_models"][1] == {"path": "org/b", "alias": "b"}
    assert after["text_models"] == [{"path": "org/t"}]
    # And nothing anywhere set a capability-wide endpoint.
    assert "remotes" not in after


def test_the_harness_refuses_to_test_a_config_the_engine_has_not_taken(monkeypatch):
    """Testing against a stale config is worse than not testing: it once
    reported a remote pass for a request that ran somewhere else."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "runpod_model_test", "tools/runpod_model_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(mod, "CONFIG_TIMEOUT_S", 0.1)
    monkeypatch.setattr(mod, "_curl",
                        lambda *a, **k: {"placement": "(none)", "backend": "",
                                         "has_runpod_block": False})
    problem = mod._wait_for_engine("org/a")
    assert "never picked up the config" in problem


def test_voice_audio_and_video_are_probed_for_real(monkeypatch):
    """Three of these were listed as un-probeable. Only one of the reasons was
    true: voice cloning needs a reference sample AND its transcript, which is
    exactly what espeak produces, and audio/video only needed asking for less."""
    from codai.api.model_test import _probe_for, _REACHABILITY_ONLY, _SPOKEN

    assert "voice" not in _REACHABILITY_ONLY
    assert "audio_gen" not in _REACHABILITY_ONLY
    assert "video" not in _REACHABILITY_ONLY

    path, body = _probe_for("voice", "m")
    assert path == "/v1/audio/clone"
    assert body["ref_text"] == _SPOKEN and body["ref_audio"], "no reference sample"

    path, body = _probe_for("audio_gen", "m")
    assert path == "/v1/audio/generate" and body["duration"] == 2.0

    path, body = _probe_for("video", "m")
    assert path == "/v1/video/generations"
    assert body["num_frames"] == 9 and body["num_inference_steps"] == 4

    # A face swap genuinely cannot be synthesised, and says why.
    assert "real images" in _REACHABILITY_ONLY["faceswap"]


def test_a_200_carrying_no_media_is_not_a_pass():
    """There is no ground truth for generated audio or video, so the check is
    'did anything come back' — but a 200 with an empty body must still fail."""
    from codai.api.model_test import _empty_result

    assert _empty_result(json.dumps({"b64_wav": "QUJD"}).encode(), "voice") == ""
    assert _empty_result(json.dumps({"url": "/v1/files/x.mp4"}).encode(), "video") == ""
    for cap in ("voice", "audio_gen", "video"):
        assert "no media" in _empty_result(json.dumps({"ok": True}).encode(), cap)


def test_the_gateway_finds_the_ocr_engine_field():
    """/v1/ocr names its engine in a field called `engine`, not `model`. The
    gateway could not see it, so a per-model OCR placement resolved correctly
    and was then refused: 'nothing configures ocr to run remotely'."""
    from codai.api.remote_gateway import _model_from_body, _MODEL_FIELD

    assert _MODEL_FIELD["/v1/ocr"] == "engine"

    body = (b'--B\r\nContent-Disposition: form-data; name="engine"\r\n\r\nsurya\r\n'
            b'--B\r\nContent-Disposition: form-data; name="file"; '
            b'filename="a.png"\r\n\r\n\x89PNG\r\n--B--\r\n')
    ct = "multipart/form-data; boundary=B"
    assert _model_from_body(body, ct, "engine") == "surya"
    # and the default field is still `model`, absent here
    assert _model_from_body(body, ct) == ""

    # JSON bodies honour the field too.
    assert _model_from_body(b'{"engine":"doctr"}', "application/json", "engine") == "doctr"


def test_audiocraft_runs_in_its_own_venv_and_looks_like_musicgen(monkeypatch):
    """audiocraft pins torch==2.1.0, which has no build for the CUDA this
    server runs — installing it beside the server leaves the GPU unusable. It
    gets its own venv, like parler-tts and the OCR engines, and the generation
    path must not be able to tell the difference."""
    from codai.api import audiocraft_worker as aw

    # The surface the audio path actually calls.
    for name in ("sample_rate", "set_generation_params", "generate",
                 "generate_with_chroma"):
        assert hasattr(aw.AudiocraftModel, name), name

    # available() is honest when there is no venv to talk to.
    monkeypatch.setattr(aw, "_VENV", Path("/nonexistent/venv"))
    assert aw.available() is False


def test_musicgen_falls_back_in_order(monkeypatch):
    """In-process audiocraft, then the isolated venv, then transformers. The
    last one has no melody conditioning, so it must be last, not first."""
    import codai.api.audio_gen as ag
    from codai.api import audiocraft_worker as aw

    calls = []
    monkeypatch.setattr(aw, "available", lambda: True)
    monkeypatch.setattr(aw, "AudiocraftModel",
                        lambda name: calls.append(("venv", name)) or "VENV")

    # No in-process audiocraft: the venv wins over transformers.
    import builtins
    real_import = builtins.__import__

    def _no_audiocraft(name, *a, **k):
        if name == "audiocraft.models" or name == "audiocraft":
            raise ImportError("no audiocraft here")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_audiocraft)
    got = ag._load_musicgen("facebook/musicgen-small", "cpu")
    assert got == "VENV" and calls == [("venv", "facebook/musicgen-small")]


def test_audio_backend_is_selectable_and_travels_to_a_pod(monkeypatch):
    """audiocraft and transformers are not equivalent — melody conditioning and
    AudioGen exist only in the first — so the choice must be the operator's, and
    must not be decided by whatever happens to be installed on a pod."""
    from codai.api.audio_gen import _audio_backend
    from codai.api.runpod_worker import pod_plan, parse_model_runpod

    assert _audio_backend(None) == "auto"
    assert _audio_backend({"audio_backend": "transformers"}) == "transformers"
    assert _audio_backend({"audio_backend": "nonsense"}) == "auto"

    monkeypatch.setenv("CODERAI_AUDIO_BACKEND", "audiocraft")
    assert _audio_backend(None) == "audiocraft"
    # a model's own setting beats the environment it lands in
    assert _audio_backend({"audio_backend": "transformers"}) == "transformers"

    mcfg = parse_model_runpod({"engine": "coderai", "image": "x:1"})
    entry = {"path": "facebook/musicgen-small", "model_type": "audio_gen_models",
             "audio_backend": "transformers"}
    assert pod_plan(mcfg, "audio_gen", entry=entry)["env"]["CODERAI_AUDIO_BACKEND"] \
        == "transformers"

    # unset means the pod decides for itself, rather than being told wrongly
    entry.pop("audio_backend")
    assert "CODERAI_AUDIO_BACKEND" not in pod_plan(mcfg, "audio_gen", entry=entry)["env"]


def test_asking_for_audiocraft_when_absent_is_an_error_not_a_substitution():
    """Silently serving transformers when audiocraft was asked for would hand
    back something that cannot do what the caller requested."""
    import builtins
    import codai.api.audio_gen as ag

    real = builtins.__import__

    def _no_audiocraft(name, *a, **k):
        if name.startswith("audiocraft"):
            raise ImportError("not installed")
        return real(name, *a, **k)

    builtins.__import__ = _no_audiocraft
    try:
        try:
            ag._load_musicgen("facebook/musicgen-small", "cpu",
                              {"audio_backend": "audiocraft"})
        except RuntimeError as exc:
            assert "audiocraft is not installed" in str(exc)
        else:
            raise AssertionError("expected a RuntimeError")
    finally:
        builtins.__import__ = real
