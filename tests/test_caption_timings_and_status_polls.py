"""Two bugs seen in production on a long render.

1. Every scene reported ``word timestamps failed ('str' object has no attribute
   'filename'); estimated instead``. The compose renderer asks for word timings
   on a narration clip it wrote to disk, so it passes a NAME where the HTTP
   endpoint passes an UploadFile, and the transcription core read ``.filename``
   off it unconditionally. The caller caught the AttributeError and fell back to
   estimated timings, so captions drifted from the voice on every render and
   nothing ever failed loudly.

2. Polling a composition's status spent the generation budget: ``GET
   /v1/video/compose/{id}`` sits under the ``/v1/video/`` prefix, which allows
   10 requests a minute per IP, so a client that polls a long render from the
   same address as its renders got 429 on the renders.
"""

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
#  1. the caption word timings
# ===========================================================================

def _tr():
    import codai.api.transcriptions as tr
    return tr


class _Upload:
    """What FastAPI hands the endpoint."""
    def __init__(self, filename):
        self.filename = filename


@pytest.mark.parametrize("given, want", [
    ("/tmp/coderai-compose-x/scene0.wav", "scene0.wav"),   # compose: a path
    ("scene0.wav", "scene0.wav"),
    (b"/tmp/a/scene1.mp3", "scene1.mp3"),
    (_Upload("narration.m4a"), "narration.m4a"),            # the endpoint
    (_Upload(None), ""),                                    # a nameless upload
    (None, ""),
])
def test_upload_name_accepts_a_path_an_upload_or_nothing(given, want):
    assert _tr()._upload_name(given) == want


def test_a_plain_name_still_yields_the_extension_the_backends_need():
    """Only the suffix is read from it, which is why a name is enough."""
    import os
    tr = _tr()
    raw = os.path.splitext(_tr()._upload_name("/var/tmp/clip.flac"))[1].lower()
    assert raw == ".flac" and raw in tr._SAFE_EXTENSIONS


def test_the_transcription_core_no_longer_reads_filename_off_the_argument():
    """The regression was one attribute access; keep it from coming back.

    Read as source rather than behaviour because reaching that line means
    resolving and loading an audio model.
    """
    tr = _tr()
    for fn in (tr._run_transcription, tr._run_transcription_inner):
        src = inspect.getsource(fn)
        assert "file.filename" not in src, (
            f"{fn.__name__} reads .filename off an argument that is a plain "
            f"name when compose calls it")


def test_stt_words_hands_transcription_something_it_accepts(monkeypatch, tmp_path):
    """The compose path, end to end over a stubbed transcription.

    The stub does what the real core does with the argument -- nothing more
    than ask it for a name -- and the assertion is that no exception reaches
    the renderer, because an exception there is a silent downgrade to estimated
    timings rather than a failure.
    """
    import asyncio
    import codai.api.compose as comp
    tr = _tr()

    audio = tmp_path / "scene0.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    seen = {}

    async def _fake(file_content, model, language, prompt, response_format,
                    temperature, file, target_language=None, diarize=False,
                    num_speakers=None, word_timestamps=False):
        seen["name"] = tr._upload_name(file)       # what the core does with it
        seen["content"] = file_content
        seen["word_timestamps"] = word_timestamps
        return {"text": "ciao", "words": [{"word": "ciao", "start": 0.0, "end": 0.4}]}

    monkeypatch.setattr(tr, "_run_transcription", _fake)
    # A loop RUNNING in another thread, because that is the real shape: the
    # render runs in a worker thread and hands its coroutines to the API's loop
    # with run_coroutine_threadsafe, which never completes against an idle one.
    import threading
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        words = comp._stt_words(loop, str(audio), "whisper-1", "it")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert words == [{"word": "ciao", "start": 0.0, "end": 0.4}]
    assert seen["name"] == "scene0.wav"
    assert seen["content"] == b"RIFF....WAVEfmt "
    assert seen["word_timestamps"] is True


# ===========================================================================
#  2. the status polls
# ===========================================================================

def _rl():
    import codai.api.ratelimit as rl
    return rl


@pytest.mark.parametrize("method, path, poll", [
    ("GET",  "/v1/video/compose/abc123", True),
    ("get",  "/v1/video/compose/abc123", True),
    ("GET",  "/v1/video/talking-head/abc123", True),
    # the sibling POST is an action, not a poll
    ("POST", "/v1/video/compose/abc123/cancel", False),
    # creating a render, and listing jobs, stay limited
    ("POST", "/v1/video/compose", False),
    ("GET",  "/v1/video/compose", False),
    ("GET",  "/v1/video/generations", False),
    ("GET",  "/v1/chat/completions", False),
])
def test_only_a_read_only_job_status_counts_as_a_poll(method, path, poll):
    assert _rl()._is_status_poll(method, path) is poll


def test_a_status_poll_is_not_waived_from_authentication():
    """It names the job's output, so it stays behind the Bearer check."""
    rl = _rl()
    assert "/v1/video/compose/" not in rl._PROGRESS_PATHS
    assert not any(p.startswith("/v1/video/compose")
                   for p in rl.BearerAuthMiddleware._EXEMPT_PATHS)


def test_polling_a_render_does_not_spend_the_render_budget():
    """The bug, reproduced: the polls used to exhaust /v1/video/ at 10/minute."""
    starlette = pytest.importorskip("starlette.applications")
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient
    rl = _rl()

    async def ok(request):
        return PlainTextResponse("ok")

    app = starlette.Starlette(routes=[
        Route("/v1/video/compose/{job_id}", ok),
        Route("/v1/video/compose", ok, methods=["GET", "POST"]),
    ])
    app.add_middleware(rl.RateLimitMiddleware, limits={"/v1/video/": (10, 60)})
    client = TestClient(app)

    # Three times the whole budget, as a 4-minute render polled every 8 s would.
    for i in range(30):
        r = client.get("/v1/video/compose/job-1")
        assert r.status_code == 200, f"poll {i + 1} was refused: {r.text}"

    # And the budget is still there for the work.
    assert client.post("/v1/video/compose").status_code == 200


def test_a_rate_limited_refusal_tells_the_client_when_to_come_back():
    """Retry-After, not just X-RateLimit-Reset: it is what clients obey."""
    starlette = pytest.importorskip("starlette.applications")
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient
    rl = _rl()

    async def ok(request):
        return PlainTextResponse("ok")

    app = starlette.Starlette(routes=[Route("/v1/video/compose", ok,
                                            methods=["GET", "POST"])])
    app.add_middleware(rl.RateLimitMiddleware, limits={"/v1/video/": (2, 60)})
    client = TestClient(app)

    for _ in range(2):
        assert client.post("/v1/video/compose").status_code == 200
    r = client.post("/v1/video/compose")
    assert r.status_code == 429
    assert r.headers.get("Retry-After") == "60"
    assert int(r.headers["X-RateLimit-Reset"]) > 0
    assert r.headers["X-RateLimit-Remaining"] == "0"


# ===========================================================================
#  1b. the other half of the same bug: a model that cannot time words
# ===========================================================================
#  Fixing the AttributeError alone would have changed nothing visible: the
#  model compose picked by default was a whisper-server one, whose runner
#  returns text with no word array, so the timings were still estimated and
#  still silently.

class _Model:
    def __init__(self, mid, mtype="audio"):
        self.id, self.type = mid, mtype


def _audio_catalogue(monkeypatch, entries):
    """``entries``: {model id: backend}. Patched at the manager boundary."""
    import codai.api.compose as comp
    from codai.models import manager as mgr

    monkeypatch.setattr(mgr.multi_model_manager, "list_models",
                        lambda *a, **k: [_Model(i) for i in entries], raising=False)
    monkeypatch.setattr(mgr.multi_model_manager, "_config_for_model",
                        lambda name: {"_raw_cfg": {"backend": entries.get(name, "")}},
                        raising=False)
    return comp


@pytest.mark.parametrize("backend, able", [
    ("wav2vec2", True),
    ("whisper_hf", True),
    ("crisperwhisper", True),
    ("vosk", True),
    ("", True),                 # no backend named = faster-whisper
    ("faster-whisper", True),
    ("whisper-server", False),  # whisper.cpp runner: text only
    ("nemo", False),            # the canary worker ignores the flag
    ("something-new", False),   # unknown: preferred against, never refused
])
def test_which_backends_can_time_words(backend, able):
    import codai.api.stt_backends as stt
    assert stt.reports_word_timings(backend) is able


def test_the_default_pick_is_a_model_that_can_time_words(monkeypatch):
    """The bug: "whisper" first, by name, picked the one backend with no words."""
    comp = _audio_catalogue(monkeypatch, {
        "whisper": "whisper-server",     # what the old order chose
        "canary": "nemo",
        "whisper3-it": "crisperwhisper",
        "wav2vec2-it": "wav2vec2",
    })
    from codai.api.compose import CaptionSpec
    assert comp._pick_stt_model(CaptionSpec(enabled=True)) == "whisper3-it"


def test_an_explicit_stt_model_is_still_obeyed(monkeypatch):
    """The operator may know better; the pick is a default, not a policy."""
    comp = _audio_catalogue(monkeypatch, {"whisper": "whisper-server",
                                          "wav2vec2-it": "wav2vec2"})
    from codai.api.compose import CaptionSpec
    spec = CaptionSpec(enabled=True, stt_model="whisper")
    assert comp._pick_stt_model(spec) == "whisper"


def test_a_catalogue_that_cannot_time_words_still_yields_a_model(monkeypatch):
    """Estimated timings beat refusing to render; the renderer says which."""
    comp = _audio_catalogue(monkeypatch, {"whisper": "whisper-server",
                                          "canary": "nemo"})
    from codai.api.compose import CaptionSpec
    assert comp._pick_stt_model(CaptionSpec(enabled=True)) in ("whisper", "canary")


def test_no_audio_model_at_all_picks_nothing(monkeypatch):
    comp = _audio_catalogue(monkeypatch, {})
    from codai.api.compose import CaptionSpec
    assert comp._pick_stt_model(CaptionSpec(enabled=True)) is None


def test_the_renderer_warns_when_a_model_returns_no_timings():
    """The downgrade has to be visible; it was the silence that cost the renders."""
    import inspect
    import codai.api.compose as comp
    src = inspect.getsource(comp._run_job)
    assert "returned no word timings" in src
    assert "_no_words_warned" in src
