"""OCR has to scale like everything else, without losing what made it accurate.

An OCR pipeline is handed ONE server address and drives it itself, because that is
what keeps its page images, its strict json_schema and its logprobs intact — a hop
that parses and re-emits the request lost all three, twice. But one address cannot
spread load:

  * a RunPod pool grows when its least-loaded pod reaches
    `scale_up_inflight_per_pod` IN-FLIGHT requests, counted in
    `RunpodBackend._acquire_endpoint`. `openai_endpoint()` releases the slot at
    once (by design — an attachment must not pin a scaling slot), so the pool saw
    zero load however hard OCR ran, and the engine had only one pod's URL anyway;
  * several cluster nodes serving the model resolved to whichever answered first,
    for ever, however busy it got;
  * a local model configured to burst to RunPod never did: that trigger lives on
    the proxy path an attached engine bypasses.

So the engine now gets a lease on an address of ours that forwards each request
verbatim to the least busy server. These tests pin both halves: that the choosing
is load-aware and makes the pool grow, and that the forwarding still decides
nothing about the request.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.models import placement as dispatch
from codai.ocr import gateway                            # noqa: E402
from codai.ocr.base import OcrError                                # noqa: E402


def _reset():
    dispatch.forget()
    gateway.revoke()
    # The upstream client is pooled and cached per event loop; a cached one would
    # carry a previous test's stub into the next test.
    gateway._clients.clear()


def _cand(key, kind="node", base="http://n/v1", priority=1, node_url="",
          acquire=None):
    return dispatch.Candidate(key=key, kind=kind, base=base, priority=priority,
                              node_url=node_url, acquire=acquire)


def _fixed_candidates(monkeypatch, cands):
    monkeypatch.setattr(dispatch, "candidates", lambda m: list(cands))


def _no_node_load(monkeypatch):
    """Node-reported load is a network call; default it to quiet in unit tests."""
    monkeypatch.setattr(dispatch, "_node_reported_load", lambda c: 0)


# ------------------------------------------------------------ least busy first #
def test_the_least_busy_node_gets_the_page(monkeypatch):
    _reset()
    _no_node_load(monkeypatch)
    a, b = _cand("node:a", base="http://a/v1"), _cand("node:b", base="http://b/v1")
    _fixed_candidates(monkeypatch, [a, b])
    # Put two requests on A; B is idle.
    held = [dispatch.choose("m"), dispatch.choose("m")]
    assert {h.key for h in held} == {"node:a", "node:b"}       # one each, then
    third = dispatch.choose("m")
    assert dispatch.inflight(third.key) >= 1
    for h in held + [third]:
        h.release()
    assert dispatch.inflight() == 0


def test_releasing_frees_the_slot(monkeypatch):
    _reset()
    _no_node_load(monkeypatch)
    only = _cand("node:a")
    _fixed_candidates(monkeypatch, [only])
    lease = dispatch.choose("m")
    assert dispatch.inflight("node:a") == 1
    lease.release()
    assert dispatch.inflight("node:a") == 0
    lease.release()                                            # idempotent
    assert dispatch.inflight("node:a") == 0


def test_a_node_busy_with_someone_elses_work_is_avoided(monkeypatch):
    """A node serves more than our OCR pages. Counting only what WE sent it would
    route onto a node already saturated by its own traffic."""
    _reset()
    quiet = _cand("node:quiet", base="http://quiet/v1", node_url="http://quiet")
    busy = _cand("node:busy", base="http://busy/v1", node_url="http://busy")
    _fixed_candidates(monkeypatch, [busy, quiet])
    monkeypatch.setattr(dispatch, "_node_reported_load",
                        lambda c: 9 if c.key == "node:busy" else 0)
    lease = dispatch.choose("m")
    assert lease.base == "http://quiet/v1"
    lease.release()


def test_a_local_server_is_preferred_while_it_is_not_busy(monkeypatch):
    """It is the cheapest hop and already paid for."""
    _reset()
    _no_node_load(monkeypatch)
    local = _cand("local:m", kind="local", base="http://127.0.0.1:8000/v1", priority=0)
    node = _cand("node:a")
    _fixed_candidates(monkeypatch, [node, local])
    lease = dispatch.choose("m", spill_after=4)
    assert lease.kind == "local"
    lease.release()


# ----------------------------------------------------------------- pods & growth #
class _FakePool:
    """Stands in for PodPool: records acquire/release, which is what growth rides on."""

    def __init__(self):
        self.acquired = 0
        self.released = 0
        self.api_key = "podkey"

    def acquire(self):
        self.acquired += 1
        return (f"pod{self.acquired}", "http://pod.runpod/")

    def release(self, pod):
        self.released += 1


def _pod_cand(pool, key="pod:m", priority=2):
    def _acquire():
        pod, url = pool.acquire()
        return url.rstrip("/") + "/v1", pool.api_key, lambda: pool.release(pod)
    return dispatch.Candidate(key=key, kind="pod", base="", priority=priority,
                              acquire=_acquire)


def test_a_runpod_only_model_goes_through_the_pool_every_request(monkeypatch):
    """This is what makes the pool scale: it counts in-flight in acquire(), so the
    request has to pass through it rather than reuse a remembered URL."""
    _reset()
    pool = _FakePool()
    _fixed_candidates(monkeypatch, [_pod_cand(pool)])
    for _ in range(3):
        dispatch.choose("m").release()
    assert pool.acquired == 3
    assert pool.released == 3


def test_the_pod_slot_is_returned_even_when_the_caller_forgets_nothing(monkeypatch):
    _reset()
    pool = _FakePool()
    _fixed_candidates(monkeypatch, [_pod_cand(pool)])
    with dispatch.choose("m"):
        assert pool.released == 0
    assert pool.released == 1


def test_a_busy_local_server_spills_to_a_pod(monkeypatch):
    """local + spillover: the local server keeps the first requests, and the
    overflow rents instead of queueing behind them."""
    _reset()
    _no_node_load(monkeypatch)
    local = _cand("local:m", kind="local", base="http://127.0.0.1:8000/v1", priority=0)
    pool = _FakePool()
    _fixed_candidates(monkeypatch, [local, _pod_cand(pool)])
    held = [dispatch.choose("m", spill_after=2) for _ in range(2)]
    assert [h.kind for h in held] == ["local", "local"]
    overflow = dispatch.choose("m", spill_after=2)
    assert overflow.kind == "pod"
    assert pool.acquired == 1
    for h in held + [overflow]:
        h.release()


def test_zero_threshold_never_spills_on_load_alone(monkeypatch):
    """0 means "do not rent because of load" — an install that wants its own GPU
    used and nothing billed must be able to say so."""
    _reset()
    _no_node_load(monkeypatch)
    local = _cand("local:m", kind="local", base="http://l/v1", priority=0)
    pool = _FakePool()
    _fixed_candidates(monkeypatch, [local, _pod_cand(pool)])
    held = [dispatch.choose("m", spill_after=0) for _ in range(5)]
    assert {h.kind for h in held} == {"local"}
    assert pool.acquired == 0
    for h in held:
        h.release()


def test_a_pool_is_still_used_when_it_is_all_there_is(monkeypatch):
    _reset()
    pool = _FakePool()
    _fixed_candidates(monkeypatch, [_pod_cand(pool)])
    lease = dispatch.choose("m", spill_after=0)
    assert lease.kind == "pod"
    lease.release()


def test_an_ordinary_pod_pool_is_tried_before_a_spillover_one(monkeypatch):
    """Spillover is the burst of last resort: it exists because the first choice
    was busy."""
    _reset()
    _no_node_load(monkeypatch)
    main, spill = _FakePool(), _FakePool()
    _fixed_candidates(monkeypatch, [_pod_cand(spill, key="spill:m", priority=3),
                                    _pod_cand(main, key="pod:m", priority=2)])
    dispatch.choose("m").release()
    assert (main.acquired, spill.acquired) == (1, 0)


def test_a_pool_that_cannot_rent_falls_through_to_the_next(monkeypatch):
    _reset()
    _no_node_load(monkeypatch)

    def _broken():
        raise RuntimeError("no capacity in region")
    broken = dispatch.Candidate(key="pod:m", kind="pod", base="", priority=2,
                                acquire=_broken)
    good = _FakePool()
    _fixed_candidates(monkeypatch, [broken, _pod_cand(good, key="spill:m", priority=3)])
    lease = dispatch.choose("m")
    assert good.acquired == 1
    lease.release()


def test_a_busy_fixed_server_still_serves_when_nothing_can_be_rented(monkeypatch):
    """Queued on a busy GPU beats failing the page."""
    _reset()
    _no_node_load(monkeypatch)

    def _broken():
        raise RuntimeError("budget exhausted")
    local = _cand("local:m", kind="local", base="http://l/v1", priority=0)
    _fixed_candidates(monkeypatch, [
        local, dispatch.Candidate(key="pod:m", kind="pod", base="", priority=2,
                                  acquire=_broken)])
    held = [dispatch.choose("m", spill_after=1) for _ in range(2)]
    assert [h.kind for h in held] == ["local", "local"]
    for h in held:
        h.release()


def test_nothing_anywhere_is_a_clear_503(monkeypatch):
    _reset()
    _fixed_candidates(monkeypatch, [])
    try:
        dispatch.choose("datalab-to/surya-ocr-2")
    except OcrError as exc:
        assert exc.status == 503
        assert "not served over HTTP" in str(exc)
    else:
        raise AssertionError("expected an OcrError")


# ------------------------------------------------------------------- the lease #
def test_the_engine_url_is_what_surya_can_attach_to(monkeypatch):
    """Surya derives the health URL by stripping a trailing /v1, so the address it
    is given must end in /v1 and leave a working root behind."""
    _reset()
    monkeypatch.setenv("CODERAI_SELF_BASE", "http://127.0.0.1:8777")
    token = gateway.issue("m", "m")
    url = gateway.engine_url(token)
    assert url == f"http://127.0.0.1:8777/v1/ocr/attach/{token}/v1"
    assert url.endswith("/v1")
    assert url[: -len("/v1")] == f"http://127.0.0.1:8777/v1/ocr/attach/{token}"


def test_a_lease_is_a_credential_and_nothing_else():
    _reset()
    token = gateway.issue("m", "m")
    assert gateway.token_is_valid(token)
    assert not gateway.token_is_valid("guessed")
    assert not gateway.token_is_valid("")
    gateway.revoke("m")
    assert not gateway.token_is_valid(token)


def test_two_engines_get_two_leases():
    _reset()
    a = gateway.issue("surya", "surya")
    b = gateway.issue("olmocr", "olmocr")
    assert a != b
    assert gateway.lease(a)["model_id"] == "surya"
    assert gateway.lease(b)["model_id"] == "olmocr"


def test_revoking_one_model_leaves_the_other():
    _reset()
    a = gateway.issue("surya", "surya")
    b = gateway.issue("olmocr", "olmocr")
    gateway.revoke("surya")
    assert not gateway.token_is_valid(a)
    assert gateway.token_is_valid(b)


# ------------------------------------------------------- the resolver uses it #
def test_model_mode_hands_the_engine_the_gateway(monkeypatch):
    _reset()
    monkeypatch.setenv("CODERAI_SELF_BASE", "http://127.0.0.1:8777")
    from codai.ocr import served_model
    monkeypatch.setattr(served_model, "_fanout_enabled", lambda: True)
    _fixed_candidates(monkeypatch, [_cand("node:a")])
    base, key, served = served_model.endpoint_for("datalab-to/surya-ocr-2")
    assert "/v1/ocr/attach/" in base and base.endswith("/v1")
    assert gateway.token_is_valid(key)
    assert served == "datalab-to/surya-ocr-2"


def test_the_served_name_is_what_the_engine_expects(monkeypatch):
    """probe_model_id compares data[0].id with SURYA_MODEL_CHECKPOINT and refuses
    on a mismatch, so the gateway answers with the name the engine was told —
    which is also what makes several upstreams interchangeable to it."""
    _reset()
    token = gateway.issue("datalab-to/surya-ocr-2", "datalab-to/surya-ocr-2")
    assert gateway.lease(token)["served_name"] == "datalab-to/surya-ocr-2"


def test_fanout_off_goes_back_to_one_direct_address(monkeypatch):
    """The direct attachment stays available: it is one hop fewer, and an install
    with a single server loses nothing by it."""
    _reset()
    from codai.ocr import served_model
    monkeypatch.setattr(served_model, "_fanout_enabled", lambda: False)

    class _B:
        def openai_endpoint(self):
            return "http://pod/v1", "k"
        served_name = "m"
    monkeypatch.setattr(served_model, "_entry_backend", lambda m: _B())
    base, key, served = served_model.endpoint_for("m")
    assert base == "http://pod/v1" and key == "k"


def test_a_blank_model_id_is_still_a_400(monkeypatch):
    _reset()
    from codai.ocr import served_model
    try:
        served_model.endpoint_for("   ")
    except OcrError as exc:
        assert exc.status == 400
    else:
        raise AssertionError("expected an OcrError")


# --------------------------------------------------------------- health probe #
def test_health_is_200_before_any_pod_exists(monkeypatch):
    """A scale-to-zero pool is HEALTHY: it can rent on demand. Surya refuses to
    attach to a URL whose /health is not 200, so answering only once a pod is up
    would mean renting an A40 just to pass a probe."""
    _reset()
    import asyncio
    pool = _FakePool()
    _fixed_candidates(monkeypatch, [_pod_cand(pool)])
    token = gateway.issue("m", "m")
    reply = asyncio.run(gateway.health(token))
    assert reply.status_code == 200
    assert pool.acquired == 0          # nothing was rented to answer a probe


def test_health_refuses_when_nothing_serves_the_model(monkeypatch):
    _reset()
    import asyncio
    _fixed_candidates(monkeypatch, [])
    token = gateway.issue("m", "m")
    reply = asyncio.run(gateway.health(token))
    assert reply.status_code == 503


def test_an_unknown_lease_is_a_404():
    _reset()
    import asyncio
    reply = asyncio.run(gateway.health("nope"))
    assert reply.status_code == 404


# ------------------------------------------- the hop must decide NOTHING at all #
#: A real surya request: an inline page image, a strict json_schema, logprobs on.
#: Every one of these was dropped by a proxy that parsed and re-emitted it, which
#: is why this hop copies bytes instead.
SURYA_BODY = (b'{"model":"datalab-to/surya-ocr-2","messages":[{"role":"user",'
              b'"content":[{"type":"image_url","image_url":{"url":"data:image/png;'
              b'base64,iVBORw0KGgo="}},{"type":"text","text":"OCR this page"}]}],'
              b'"logprobs":true,"response_format":{"type":"json_schema",'
              b'"json_schema":{"name":"structured_output","strict":true,'
              b'"schema":{"type":"object"}}},"max_tokens":4096}')

#: And the answer, with the logprobs surya turns into each line's confidence.
UPSTREAM_REPLY = (b'{"choices":[{"index":0,"message":{"role":"assistant",'
                  b'"content":"text"},"finish_reason":"stop","logprobs":'
                  b'{"content":[{"token":"A","logprob":-0.25}]}}],'
                  b'"usage":{"prompt_tokens":1,"completion_tokens":1}}')


class _CapturingClient:
    """Captures what the gateway sends upstream and answers verbatim."""

    sent = {}

    def __init__(self, *a, **kw):
        self.timeout = kw.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, content=None, headers=None):
        type(self).sent = {"url": url, "content": content, "headers": headers or {}}

        class _R:
            status_code = 200
            headers = {"content-type": "application/json"}
            content = UPSTREAM_REPLY
        return _R()


class _Req:
    """The minimum of a Starlette request the route touches."""

    def __init__(self, body, content_type="application/json"):
        self._body = body
        self.headers = {"content-type": content_type}

    async def body(self):
        return self._body


def _forward(monkeypatch, body=SURYA_BODY, **kw):
    import asyncio
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _CapturingClient)
    _no_node_load(monkeypatch)
    _fixed_candidates(monkeypatch, kw.pop("cands", [_cand("node:a", base="http://a/v1")]))
    token = gateway.issue("datalab-to/surya-ocr-2", "datalab-to/surya-ocr-2")
    reply = asyncio.run(gateway.chat_completions(token, _Req(body)))
    return reply, _CapturingClient.sent


def test_the_request_body_arrives_upstream_unchanged(monkeypatch):
    """Byte-identical: the image stays an image, the schema stays strict, and
    logprobs stays asked for. This is the whole licence for adding a hop."""
    _reset()
    _, sent = _forward(monkeypatch)
    assert sent["content"] == SURYA_BODY


def test_the_reply_comes_back_unchanged(monkeypatch):
    """Including the logprobs. Re-serialising the response is how they were lost."""
    _reset()
    reply, _ = _forward(monkeypatch)
    assert reply.body == UPSTREAM_REPLY
    assert b'"logprobs"' in reply.body
    assert reply.status_code == 200


def test_it_goes_to_the_chosen_upstream(monkeypatch):
    _reset()
    _, sent = _forward(monkeypatch, cands=[_cand("node:b", base="http://chosen/v1")])
    assert sent["url"] == "http://chosen/v1/chat/completions"


def test_a_locked_upstream_is_given_its_bearer(monkeypatch):
    _reset()
    cand = _cand("node:a", base="http://a/v1")
    cand.api_key = "nodetoken"
    _, sent = _forward(monkeypatch, cands=[cand])
    assert sent["headers"]["Authorization"] == "Bearer nodetoken"


def test_an_unlocked_upstream_gets_no_authorization_header(monkeypatch):
    """A local vLLM with no key must not receive an empty bearer, which some
    servers reject outright."""
    _reset()
    _, sent = _forward(monkeypatch)
    assert "Authorization" not in sent["headers"]


def test_the_slot_is_released_after_forwarding(monkeypatch):
    """Otherwise the first page would make its upstream look permanently busy and
    every later page would route elsewhere — or rent."""
    _reset()
    _forward(monkeypatch)
    assert dispatch.inflight() == 0


def test_the_pod_slot_is_released_even_if_the_upstream_errors(monkeypatch):
    _reset()
    import asyncio
    import httpx

    class _Failing(_CapturingClient):
        async def post(self, url, content=None, headers=None):
            raise RuntimeError("connection reset")
    monkeypatch.setattr(httpx, "AsyncClient", _Failing)
    pool = _FakePool()
    _fixed_candidates(monkeypatch, [_pod_cand(pool)])
    token = gateway.issue("m", "m")
    reply = asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    assert reply.status_code == 502
    assert pool.released == 1
    assert dispatch.inflight() == 0


def test_an_unknown_lease_cannot_forward_anything(monkeypatch):
    _reset()
    import asyncio
    reply = asyncio.run(gateway.chat_completions("forged", _Req(SURYA_BODY)))
    assert reply.status_code == 404


def test_no_server_available_is_reported_as_the_resolver_said(monkeypatch):
    _reset()
    import asyncio
    _fixed_candidates(monkeypatch, [])
    token = gateway.issue("m", "m")
    reply = asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    assert reply.status_code == 503


def test_models_answers_the_name_the_engine_checks(monkeypatch):
    _reset()
    import asyncio
    token = gateway.issue("datalab-to/surya-ocr-2", "datalab-to/surya-ocr-2")
    reply = asyncio.run(gateway.models(token))
    import json
    data = json.loads(reply.body)
    assert data["data"][0]["id"] == "datalab-to/surya-ocr-2"


# ------------------------------------------------- the engine must get THROUGH #
# Two gates stand in front of this route on a real install, and the engine can
# satisfy neither the usual way: it has no API token, and surya's probes send no
# headers at all (`probe_health` / `probe_model_id` are bare httpx GETs). If
# either gate refuses, surya does not degrade — it refuses to attach, and every
# page fails with "SURYA_INFERENCE_URL is not reachable at /health".


class _Probe:
    """A request shaped like surya's probe: no Authorization header at all."""

    def __init__(self, path, headers=None):
        import types
        self.url = types.SimpleNamespace(path=path)
        self.headers = headers or {}
        self.scope = {"server": ("127.0.0.1", 8777), "type": "http"}


def _through_bearer_gate(path, headers=None):
    """True when BearerAuthMiddleware lets the request reach the route."""
    import asyncio
    from codai.api.ratelimit import BearerAuthMiddleware
    mw = BearerAuthMiddleware(lambda scope, receive, send: None)
    passed = []

    async def _call_next(_req):
        passed.append(True)
        return "reached-the-route"

    reply = asyncio.run(mw.dispatch(_Probe(path, headers), _call_next))
    return bool(passed) and reply == "reached-the-route"


def test_a_probe_with_no_headers_at_all_gets_through(monkeypatch):
    """surya checks /health with a bare httpx GET and refuses to attach if it is
    not 200. The lease is therefore read from the URL, not from a header."""
    _reset()
    monkeypatch.setenv("CODERAI_API_TOKEN", "some-api-token")
    token = gateway.issue("m", "m")
    assert _through_bearer_gate(f"/v1/ocr/attach/{token}/health")
    assert _through_bearer_gate(f"/v1/ocr/attach/{token}/v1/models")
    assert _through_bearer_gate(f"/v1/ocr/attach/{token}/v1/chat/completions",
                                {"authorization": f"Bearer {token}"})


def test_a_forged_lease_is_not_a_way_in(monkeypatch):
    """Otherwise the exemption would be a hole in the API gate rather than a
    credential the engine happens to hold."""
    _reset()
    monkeypatch.setenv("CODERAI_API_TOKEN", "some-api-token")
    gateway.issue("m", "m")
    assert not _through_bearer_gate("/v1/ocr/attach/forged/health")


def test_a_revoked_lease_stops_working(monkeypatch):
    _reset()
    monkeypatch.setenv("CODERAI_API_TOKEN", "some-api-token")
    token = gateway.issue("m", "m")
    assert _through_bearer_gate(f"/v1/ocr/attach/{token}/health")
    gateway.revoke("m")
    assert not _through_bearer_gate(f"/v1/ocr/attach/{token}/health")


def test_the_lease_does_not_unlock_the_rest_of_the_api(monkeypatch):
    """A lease authorises its own gateway path and nothing else."""
    _reset()
    monkeypatch.setenv("CODERAI_API_TOKEN", "some-api-token")
    token = gateway.issue("m", "m")
    assert not _through_bearer_gate("/v1/chat/completions",
                                    {"authorization": f"Bearer {token}"})
    assert not _through_bearer_gate("/v1/models",
                                    {"authorization": f"Bearer {token}"})


def test_the_engine_reaches_an_engine_only_process(monkeypatch):
    """With CODERAI_INTERNAL_TOKEN set, the engine app rejects every request that
    does not carry the front's internal header — and the OCR worker cannot carry
    it. The gateway path is exempted, which is safe because the lease is still
    required behind it."""
    import asyncio
    from codai.api.app import _InternalAuthMiddleware
    monkeypatch.setenv("CODERAI_INTERNAL_TOKEN", "internal-secret")
    reached = []

    async def _inner(scope, receive, send):
        reached.append(scope["path"])

    mw = _InternalAuthMiddleware(_inner)
    token = gateway.issue("m", "m")

    sent = []

    async def _send(msg):
        sent.append(msg)

    # No internal header, as the OCR worker sends it.
    asyncio.run(mw({"type": "http", "path": f"/v1/ocr/attach/{token}/health",
                    "headers": []}, None, _send))
    assert reached == [f"/v1/ocr/attach/{token}/health"]

    # Everything else is still refused without the header.
    reached.clear()
    asyncio.run(mw({"type": "http", "path": "/v1/chat/completions",
                    "headers": []}, None, _send))
    assert reached == []
    assert any(m.get("status") == 403 for m in sent if m.get("type")
               == "http.response.start")


# ------------------------------------------------- seamless from the client in #
# A client asks for OCR with a model it has heard of — the id it sees in
# /v1/models — and should not have to know that we read pages with it by driving
# a pipeline called "surya", nor where that model is currently running.


def _mgr(**ocr):
    import types
    from codai.ocr.manager import OcrManager
    mgr = OcrManager.__new__(OcrManager)
    fields = dict(default_engine="paddle", surya_model_id="", surya_model="",
                  olmocr_model_id="", olmocr_model="", surya_enabled=True,
                  surya_accept_license=True, olmocr_enabled=True,
                  olmocr_accept_license=True, enabled=True,
                  doctr_enabled=True, paddle_enabled=True)
    fields.update(ocr)
    cfg = types.SimpleNamespace(**fields)
    mgr._require_cfg = lambda: cfg
    return mgr


def test_a_client_can_name_the_model_instead_of_the_engine():
    mgr = _mgr(surya_model_id="datalab-to/surya-ocr-2")
    assert mgr.resolve_engine("datalab-to/surya-ocr-2") == "surya"


def test_the_olmocr_checkpoint_names_olmocr():
    mgr = _mgr(olmocr_model_id="allenai/olmOCR-2-7B-1025-FP8")
    assert mgr.resolve_engine("allenai/olmOCR-2-7B-1025-FP8") == "olmocr"


def test_naming_the_engine_still_works():
    """The existing clients say "surya", and must keep working unchanged."""
    mgr = _mgr(surya_model_id="datalab-to/surya-ocr-2")
    assert mgr.resolve_engine("surya") == "surya"


def test_the_checkpoint_field_counts_too_not_only_the_model_id():
    """An install serving surya locally sets surya_model, not surya_model_id."""
    mgr = _mgr(surya_model="datalab-to/surya-ocr-2")
    assert mgr.resolve_engine("datalab-to/surya-ocr-2") == "surya"


def test_a_model_we_do_not_read_pages_with_is_still_a_clear_400():
    from codai.ocr.base import OcrError
    mgr = _mgr(surya_model_id="datalab-to/surya-ocr-2")
    try:
        mgr.resolve_engine("meta-llama/Llama-3.3-70B")
    except OcrError as exc:
        assert exc.status == 400
        assert "engines:" in str(exc)
    else:
        raise AssertionError("expected an OcrError")


# ------------------------------------- how busy is too busy is the MODEL's own #
def test_the_spill_threshold_comes_from_the_model(monkeypatch):
    """Not from a second OCR-shaped copy of a number the model already carries."""
    from codai.models import placement
    monkeypatch.setattr(placement, "model_entry",
                        lambda m: {"id": m, "runpod": {"scale_up_inflight_per_pod": 9}})
    assert placement.spill_threshold("m") == 9


def test_a_pool_with_no_threshold_set_uses_the_documented_default(monkeypatch):
    from codai.models import placement
    monkeypatch.setattr(placement, "model_entry", lambda m: {"id": m, "runpod": {}})
    assert placement.spill_threshold("m") == 4


def test_a_model_with_nothing_to_rent_never_spills_on_load(monkeypatch):
    """No RunPod block means no pod to burst to, so load alone cannot move a page."""
    from codai.models import placement
    monkeypatch.setattr(placement, "model_entry", lambda m: {"id": m})
    assert placement.spill_threshold("m") == 0


# ------------------------------------------- no engine quietly moves to the CPU #
# docTR used to call .cuda() if torch happened to see a card and otherwise just
# stay on the CPU, so a GPU-less install read pages many times slower with
# nothing in the response to say why. Nothing else in coderai moves GPU work to
# the CPU on its own, and paddle refuses outright in the same situation.


def _doctr_load(use_gpu, cuda_available, monkeypatch):
    """Run the REAL DoctrEngine.load(), with docTR and torch stubbed.

    Driving the real method matters: a helper that re-implemented the placement
    rules would keep passing after the rules changed in doctr.py.
    """
    import types
    from codai.ocr.doctr import DoctrEngine

    placed = []
    model = types.SimpleNamespace(cpu=lambda: placed.append("cpu") or model,
                                  cuda=lambda: placed.append("gpu") or model)
    doctr_models = types.SimpleNamespace(ocr_predictor=lambda **kw: model)
    monkeypatch.setitem(sys.modules, "doctr",
                        types.SimpleNamespace(models=doctr_models))
    monkeypatch.setitem(sys.modules, "doctr.models", doctr_models)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available,
                                   empty_cache=lambda: None)))

    eng = DoctrEngine.__new__(DoctrEngine)
    eng.cfg = types.SimpleNamespace(doctr_use_gpu=use_gpu, doctr_det_arch="",
                                    doctr_reco_arch="")
    eng._model = None
    eng.load()
    return placed


def test_doctr_refuses_rather_than_running_on_the_cpu_unasked(monkeypatch):
    """A 503 that names the setting beats a page that takes minutes for no
    visible reason."""
    from codai.ocr.base import OcrError
    try:
        _doctr_load(use_gpu=True, cuda_available=False, monkeypatch=monkeypatch)
    except OcrError as exc:
        assert exc.status == 503
        assert "doctr_use_gpu" in str(exc)
    else:
        raise AssertionError("expected docTR to refuse")


def test_doctr_on_cpu_is_still_fully_supported_when_asked_for(monkeypatch):
    """Refusing the silent fallback must not remove the deliberate choice."""
    assert _doctr_load(use_gpu=False, cuda_available=False,
                       monkeypatch=monkeypatch) == ["cpu"]


def test_doctr_uses_the_gpu_when_it_has_one(monkeypatch):
    assert _doctr_load(use_gpu=True, cuda_available=True,
                       monkeypatch=monkeypatch) == ["gpu"]


def test_doctr_does_not_touch_the_cpu_path_when_it_refuses(monkeypatch):
    """It must refuse BEFORE placing the model anywhere."""
    from codai.ocr.base import OcrError
    try:
        _doctr_load(use_gpu=True, cuda_available=False, monkeypatch=monkeypatch)
    except OcrError:
        pass


def test_the_refusal_message_names_the_way_out():
    import inspect
    from codai.ocr.doctr import DoctrEngine
    src = inspect.getsource(DoctrEngine.load)
    assert "ocr.doctr_use_gpu = false" in src
    assert "NOT silently run on the CPU" in src


# -------------------------------------------- a request that names no engine #
def test_no_engine_in_the_request_uses_the_configured_default():
    """`ocr.default_engine` is the whole answer: a client that just posts a file
    gets whatever the install decided OCR means."""
    mgr = _mgr(default_engine="surya", surya_model_id="datalab-to/surya-ocr-2")
    assert mgr.resolve_engine(None) == "surya"
    assert mgr.resolve_engine("") == "surya"


def test_the_default_may_itself_be_a_model_id():
    """The same two spellings work for the default as for a request, so an
    operator can write down the checkpoint they think in rather than the name of
    the pipeline that drives it."""
    mgr = _mgr(default_engine="allenai/olmOCR-2-7B-1025-FP8",
               olmocr_model_id="allenai/olmOCR-2-7B-1025-FP8")
    assert mgr.resolve_engine(None) == "olmocr"


def test_an_explicit_engine_still_overrides_the_default():
    mgr = _mgr(default_engine="surya", surya_model_id="datalab-to/surya-ocr-2")
    assert mgr.resolve_engine("doctr") == "doctr"


# ------------------------------------------------ what can I ask this install #
# Naming an engine the install had not enabled was a 400, and there was nothing
# to ask beforehand — the client had to know, or guess.


def test_the_catalogue_lists_every_engine_with_whether_it_is_enabled():
    mgr = _mgr(default_engine="surya", surya_model_id="datalab-to/surya-ocr-2",
               paddle_enabled=False, doctr_enabled=False)
    cat = mgr.engine_catalogue()
    rows = {r["id"]: r for r in cat["data"]}
    # Against the REGISTRY, not a hand-written list: the point of the catalogue is that
    # a client never has to know which engines exist, and a frozen list here would mean
    # adding one silently left it out of the only place that answers that question.
    from codai.ocr.manager import _ENGINE_FACTORIES
    assert set(rows) == set(_ENGINE_FACTORIES)
    assert rows["surya"]["enabled"] is True
    assert rows["paddle"]["enabled"] is False


def test_a_disabled_engine_says_why():
    """So the answer is actionable rather than just negative."""
    mgr = _mgr(paddle_enabled=False)
    rows = {r["id"]: r for r in mgr.engine_catalogue()["data"]}
    assert "ocr.paddle_enabled" in rows["paddle"]["reason"]


def test_an_unaccepted_licence_is_not_reported_as_merely_disabled():
    """It is one checkbox away, and saying "not enabled" would send the operator
    to the wrong setting."""
    mgr = _mgr(surya_enabled=True, surya_accept_license=False)
    rows = {r["id"]: r for r in mgr.engine_catalogue()["data"]}
    assert "licence" in rows["surya"]["reason"]


def test_the_catalogue_publishes_the_model_names_that_also_work():
    """The aliases are the whole point of accepting a model id in the request."""
    mgr = _mgr(surya_model_id="datalab-to/surya-ocr-2",
               olmocr_model_id="allenai/olmOCR-2-7B-1025-FP8")
    rows = {r["id"]: r for r in mgr.engine_catalogue()["data"]}
    assert "datalab-to/surya-ocr-2" in rows["surya"]["aliases"]
    assert rows["olmocr"]["model"] == "allenai/olmOCR-2-7B-1025-FP8"
    # And every advertised alias really does resolve.
    for row in rows.values():
        for alias in row.get("aliases", []):
            assert mgr.resolve_engine(alias) == row["id"]


def test_the_catalogue_says_which_engines_can_return_boxes():
    """olmOCR is a document VLM: it transcribes a page and has no notion of where
    a word was, so a client needing coordinates must be able to see that."""
    mgr = _mgr()
    rows = {r["id"]: r for r in mgr.engine_catalogue()["data"]}
    assert rows["olmocr"]["returns"]["boxes"] is False
    assert rows["surya"]["returns"]["boxes"] is True
    assert rows["paddle"]["returns"]["regions"] is True


def test_the_catalogue_distinguishes_a_pipeline_from_an_engine():
    """Whether "where does it run" is even a question depends on this."""
    mgr = _mgr()
    rows = {r["id"]: r for r in mgr.engine_catalogue()["data"]}
    assert rows["surya"]["kind"] == "pipeline"
    assert rows["paddle"]["kind"] == "engine"


def test_the_catalogue_reports_what_a_request_with_no_engine_will_use():
    mgr = _mgr(default_engine="surya", surya_model_id="datalab-to/surya-ocr-2")
    cat = mgr.engine_catalogue()
    assert cat["default"] == "surya"
    assert cat["default_engine"] == mgr.resolve_engine(None)


def test_a_default_written_as_a_model_id_is_reported_resolved():
    """`default` is what the operator wrote; `default_engine` is what it means."""
    mgr = _mgr(default_engine="allenai/olmOCR-2-7B-1025-FP8",
               olmocr_model_id="allenai/olmOCR-2-7B-1025-FP8")
    cat = mgr.engine_catalogue()
    assert cat["default"] == "allenai/olmOCR-2-7B-1025-FP8"
    assert cat["default_engine"] == "olmocr"


def test_the_catalogue_does_not_leak_where_the_model_runs():
    """Placement is configuration and can change between two requests; a client
    that learned it would start depending on it."""
    mgr = _mgr(surya_model_id="datalab-to/surya-ocr-2")
    blob = repr(mgr.engine_catalogue())
    for leak in ("runpod", "pod", "node", "127.0.0.1", "http"):
        assert leak not in blob.lower()


def test_the_public_engines_endpoint_is_not_covered_by_the_lease_exemption(monkeypatch):
    """`/v1/ocr/engines` is a normal, authenticated API route. The lease exemption
    covers `/v1/ocr/attach/` only — which is why the gateway does NOT live under
    `/v1/ocr/engine*`, one keystroke from it."""
    _reset()
    monkeypatch.setenv("CODERAI_API_TOKEN", "some-api-token")
    gateway.issue("m", "m")
    assert not _through_bearer_gate("/v1/ocr/engines")
    assert gateway.PREFIX == "/v1/ocr/attach"


def test_the_internal_gate_exempts_only_the_attach_prefix(monkeypatch):
    import asyncio
    from codai.api.app import _InternalAuthMiddleware
    monkeypatch.setenv("CODERAI_INTERNAL_TOKEN", "internal-secret")
    reached = []

    async def _inner(scope, receive, send):
        reached.append(scope["path"])

    mw = _InternalAuthMiddleware(_inner)

    async def _send(msg):
        pass

    for path in ("/v1/ocr/engines", "/v1/ocr", "/v1/ocr/engine-stats"):
        reached.clear()
        asyncio.run(mw({"type": "http", "path": path, "headers": []}, None, _send))
        assert reached == [], f"{path} must still require the internal token"


# --------------------------------------------- one handshake per pod, not per page #
# The gateway built a new httpx.AsyncClient for every request, so each page paid a
# fresh TCP and TLS handshake to the pod. With surya_instances: 24 that is 24
# streams reconnecting per page against a round trip that already dominates.


def test_the_upstream_client_is_reused_within_one_event_loop(monkeypatch):
    """The point of the fix: a second page on the same loop reuses the connection
    pool instead of building another one."""
    _reset()
    import asyncio
    import httpx
    built = []

    class _Counting(_CapturingClient):
        def __init__(self, *a, **kw):
            built.append(kw)
            super().__init__(*a, **kw)

        @property
        def is_closed(self):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", _Counting)
    _no_node_load(monkeypatch)
    _fixed_candidates(monkeypatch, [_cand("node:a", base="http://a/v1")])
    token = gateway.issue("m", "m")

    async def _two_pages():
        await gateway.chat_completions(token, _Req(SURYA_BODY))
        await gateway.chat_completions(token, _Req(SURYA_BODY))

    asyncio.run(_two_pages())
    assert len(built) == 1, "a second page must not build a second client"


def test_the_client_is_created_with_keep_alive_limits(monkeypatch):
    """A pool with no keep-alive would reconnect anyway."""
    _reset()
    import asyncio
    import httpx
    built = []

    class _Counting(_CapturingClient):
        def __init__(self, *a, **kw):
            built.append(kw)
            super().__init__(*a, **kw)

        @property
        def is_closed(self):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", _Counting)
    _no_node_load(monkeypatch)
    _fixed_candidates(monkeypatch, [_cand("node:a", base="http://a/v1")])
    token = gateway.issue("m", "m")
    asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    assert built and built[0].get("limits") is not None
    assert built[0].get("timeout") == gateway.UPSTREAM_TIMEOUT_S


def test_a_client_is_not_shared_between_different_loops(monkeypatch):
    """An AsyncClient binds to the loop that made it; using one from another loop
    is an error waiting for load to find it."""
    _reset()
    import asyncio
    import httpx
    built = []

    class _Counting(_CapturingClient):
        def __init__(self, *a, **kw):
            built.append(kw)
            super().__init__(*a, **kw)

        @property
        def is_closed(self):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", _Counting)
    _no_node_load(monkeypatch)
    _fixed_candidates(monkeypatch, [_cand("node:a", base="http://a/v1")])
    token = gateway.issue("m", "m")
    asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    assert len(built) == 2, "each loop needs its own client"


def test_a_dead_loop_does_not_keep_its_client_alive(monkeypatch):
    """Keyed weakly by the loop object, so a finished loop's entry disappears
    rather than accumulating — and cannot be handed to a new loop that happens to
    reuse the address."""
    _reset()
    import asyncio
    import gc
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _CapturingClient)
    _no_node_load(monkeypatch)
    _fixed_candidates(monkeypatch, [_cand("node:a", base="http://a/v1")])
    token = gateway.issue("m", "m")
    asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    gc.collect()
    assert len(gateway._clients) == 0


# ---------------------------------------- local means a GPU here, or not at all #
# Surya's `local` mode runs detection and recognition in its own venv, on this
# machine. Its own settings pick the device and settle for the CPU whenever torch
# sees no CUDA (surya/settings.py TORCH_DEVICE_MODEL: cuda -> mps -> cpu), where a
# page takes minutes and the result looks perfectly normal. paddle and docTR
# refuse in that situation; surya was the one engine that did not.


def _surya(serve="local", use_gpu=True):
    import types
    from codai.ocr.surya import SuryaEngine
    eng = SuryaEngine.__new__(SuryaEngine)
    eng.cfg = types.SimpleNamespace(surya_serve=serve, surya_use_gpu=use_gpu,
                                    surya_model_id="", surya_model="",
                                    surya_server_url="", surya_accept_license=True,
                                    surya_venv="", surya_auto_build=False)
    eng._server_url = ""
    eng._server_key = ""
    eng._served_name = ""
    return eng


def test_local_mode_refuses_a_machine_with_no_gpu(monkeypatch):
    from codai.ocr.base import OcrError
    import codai.ocr.surya as mod
    monkeypatch.setattr(mod, "_cuda_present", lambda: False)
    monkeypatch.setattr(mod.SuryaEngine.__bases__[0], "load", lambda self: None,
                        raising=False)
    try:
        _surya("local").load()
    except OcrError as exc:
        assert exc.status == 503
        assert "surya_use_gpu" in str(exc)
        assert "surya_serve" in str(exc)        # and it names the way out
    else:
        raise AssertionError("expected local mode to refuse")


def test_local_mode_loads_where_there_is_a_gpu(monkeypatch):
    import codai.ocr.surya as mod
    monkeypatch.setattr(mod, "_cuda_present", lambda: True)
    monkeypatch.setattr(mod.SuryaEngine.__bases__[0], "load", lambda self: None,
                        raising=False)
    _surya("local").load()                      # no raise


def test_the_cpu_can_still_be_chosen_deliberately(monkeypatch):
    """Refusing the silent fallback must not remove the deliberate option."""
    import codai.ocr.surya as mod
    monkeypatch.setattr(mod, "_cuda_present", lambda: False)
    monkeypatch.setattr(mod.SuryaEngine.__bases__[0], "load", lambda self: None,
                        raising=False)
    _surya("local", use_gpu=False).load()       # no raise


def test_the_worker_is_told_the_device_rather_than_guessing():
    assert _surya("local").\
        _local_env() == {"TORCH_DEVICE": "cuda"}
    assert _surya("local", use_gpu=False)._local_env() == {"TORCH_DEVICE": "cpu"}


def test_no_gpu_detected_means_no_gpu(monkeypatch):
    """Fail closed. The mode exists to use a GPU, and guessing one is there ends
    in a CPU run that looks healthy and never finishes."""
    import builtins
    import codai.ocr.surya as mod
    real_import = builtins.__import__

    def _no_torch(name, *a, **kw):
        if name == "torch":
            raise ImportError("no torch in this venv")
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", _no_torch)
    monkeypatch.setattr(mod.shutil if hasattr(mod, "shutil") else mod,
                        "__name__", "x", raising=False)
    import shutil
    monkeypatch.setattr(shutil, "which", lambda n: None)
    assert mod._cuda_present() is False


def test_local_is_no_longer_what_an_unconfigured_install_gets():
    """It has to be asked for: the default used to be the one mode that needs a
    GPU on the very box whose job is to rent one."""
    from codai.config import OcrConfig
    assert OcrConfig().surya_serve == "model"
    assert _surya(serve="")._serve_mode() == "model"


# ------------------------------------------- a pod serves OUR certificate, on an IP #
# The first real document found this: the pod came up at https://194.68.245.79:22198
# and every page came back "CERTIFICATE_VERIFY_FAILED: unable to get local issuer
# certificate". A direct-TCP pod serves the certificate coderai's own CA signed,
# on a bare IP that certificate can never name — which is why pod_http pins to
# that CA and switches hostname checking off. This hop is async, so it could not
# reuse that module and had to be told the same thing.


def test_a_pod_url_gets_the_pinned_trust_policy(monkeypatch):
    _reset()
    import asyncio
    import httpx
    built = []

    class _Counting(_CapturingClient):
        def __init__(self, *a, **kw):
            built.append(kw)
            super().__init__(*a, **kw)

        @property
        def is_closed(self):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", _Counting)
    monkeypatch.setattr(gateway, "_pinned_ssl_context", lambda: "PINNED-CTX")
    _no_node_load(monkeypatch)
    _fixed_candidates(monkeypatch, [_cand("pod:m", kind="pod",
                                          base="https://194.68.245.79:22198/v1")])
    token = gateway.issue("m", "m")
    asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    assert built and built[0]["verify"] == "PINNED-CTX"


def test_an_ordinary_https_upstream_keeps_normal_verification(monkeypatch):
    """A cluster node has a certificate a public CA signed; the pinned context
    trusts ONLY our CA, so using it everywhere would reject the node."""
    _reset()
    import asyncio
    import httpx
    built = []

    class _Counting(_CapturingClient):
        def __init__(self, *a, **kw):
            built.append(kw)
            super().__init__(*a, **kw)

        @property
        def is_closed(self):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", _Counting)
    _no_node_load(monkeypatch)
    _fixed_candidates(monkeypatch, [_cand("node:a", base="https://node.lan:8443/v1")])
    token = gateway.issue("m", "m")
    asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    assert built and built[0]["verify"] is True


def test_the_two_policies_do_not_share_a_client(monkeypatch):
    """Otherwise whichever upstream came first would decide the trust for both."""
    _reset()
    import asyncio
    import httpx
    built = []

    class _Counting(_CapturingClient):
        def __init__(self, *a, **kw):
            built.append(kw)
            super().__init__(*a, **kw)

        @property
        def is_closed(self):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", _Counting)
    monkeypatch.setattr(gateway, "_pinned_ssl_context", lambda: "PINNED-CTX")

    async def _both():
        _fixed_candidates(monkeypatch, [_cand("node:a", base="https://node.lan/v1")])
        await gateway.chat_completions(tok, _Req(SURYA_BODY))
        _fixed_candidates(monkeypatch, [_cand("pod:m", kind="pod",
                                              base="https://10.0.0.5:8000/v1")])
        await gateway.chat_completions(tok, _Req(SURYA_BODY))

    _no_node_load(monkeypatch)
    tok = gateway.issue("m", "m")
    asyncio.run(_both())
    assert [b["verify"] for b in built] == [True, "PINNED-CTX"]


def test_a_plain_http_upstream_is_not_pinned(monkeypatch):
    """A local vLLM on loopback has no certificate at all."""
    from codai.api.pod_http import is_pinned_url
    assert is_pinned_url("http://127.0.0.1:8000/v1") is False
    assert is_pinned_url("https://194.68.245.79:22198/v1") is True
    assert is_pinned_url("https://node.lan:8443/v1") is False


# --------------------------------------- a pod can vanish under a document #
# RunPod reclaimed an A40 mid-document today. The pool went on reporting it
# healthy, so every page chose the same corpse: the in-flight count climbed and
# nothing drained, which a client sees as "requests neither complete nor time
# out". RunpodBackend has always reported this (_unreachable); a caller holding a
# lease has to do the same.


def _dead_then_alive(monkeypatch, failures=1):
    """An upstream that refuses the first N connections, then answers."""
    import httpx
    state = {"n": 0}

    class _Flaky(_CapturingClient):
        @property
        def is_closed(self):
            return False

        async def post(self, url, content=None, headers=None):
            state["n"] += 1
            if state["n"] <= failures:
                raise httpx.ConnectError("connection refused")
            return await super().post(url, content=content, headers=headers)

    monkeypatch.setattr(httpx, "AsyncClient", _Flaky)
    return state


def _pod_cands(monkeypatch, discarded):
    """Two successive pod leases, each able to report itself dead."""
    def _acquire():
        return ("https://10.0.0.9:8000/v1", "k", lambda: None,
                lambda: discarded.append(1))
    cand = dispatch.Candidate(key="pod:m", kind="pod", base="", priority=2,
                              acquire=_acquire, primary=True)
    monkeypatch.setattr(dispatch, "candidates", lambda m: [cand])
    monkeypatch.setattr(dispatch, "_node_reported_load", lambda c: 0)


def test_a_vanished_pod_is_reported_and_the_page_retried(monkeypatch):
    _reset()
    import asyncio
    discarded = []
    _dead_then_alive(monkeypatch, failures=1)
    _pod_cands(monkeypatch, discarded)
    token = gateway.issue("m", "m")
    reply = asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    assert discarded == [1]                  # the pool was told
    assert reply.status_code == 200          # and the page still got read


def test_it_gives_up_after_one_retry_rather_than_looping(monkeypatch):
    """A real outage must not become a retry loop against every pod in turn."""
    _reset()
    import asyncio
    discarded = []
    _dead_then_alive(monkeypatch, failures=5)
    _pod_cands(monkeypatch, discarded)
    token = gateway.issue("m", "m")
    reply = asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    assert reply.status_code == 503
    assert len(discarded) == 2               # both attempts reported, then stop


def test_the_inflight_count_does_not_outlive_a_dropped_pod(monkeypatch):
    """The symptom the client actually saw: a count that only grew."""
    _reset()
    import asyncio
    discarded = []
    _dead_then_alive(monkeypatch, failures=5)
    _pod_cands(monkeypatch, discarded)
    token = gateway.issue("m", "m")
    asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    assert dispatch.inflight() == 0


def test_a_slow_server_is_not_mistaken_for_a_dead_one(monkeypatch):
    """A read timeout is a model taking its time over a dense page. Discarding a
    healthy pod for that would be an outage of its own."""
    _reset()
    import asyncio
    import httpx
    discarded = []

    class _Slow(_CapturingClient):
        @property
        def is_closed(self):
            return False

        async def post(self, url, content=None, headers=None):
            raise httpx.ReadTimeout("too slow")

    monkeypatch.setattr(httpx, "AsyncClient", _Slow)
    _pod_cands(monkeypatch, discarded)
    token = gateway.issue("m", "m")
    reply = asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    assert reply.status_code == 502          # reported, not retried
    assert discarded == []                   # and the pod kept
    assert dispatch.inflight() == 0


def test_the_candidate_set_is_rebuilt_after_a_death():
    """It named a pod that is gone; keeping it cached would hand it out again."""
    _reset()
    dispatch._candidates["m"] = (9e9, [_cand("node:a")])
    lease = dispatch.Lease("http://a/v1", "", "node:a", "node")
    lease.dead()
    assert "m" not in dispatch._candidates


# ------------------------------- a closed keep-alive socket is not a dead pod #
# On 2026-10-12 the OCR pool burned through five A40s between 19:14 and 20:11.
# Not one of them was broken: each was killed mid-document while its own log
# still read `"POST /v1/chat/completions HTTP/1.1" 200 OK` at 4664 tokens/s.
# What killed them was `httpx.RemoteProtocolError("Server disconnected without
# sending a response")` — the error httpx raises when a POOLED connection was
# closed by the far end between two requests. uvicorn closes an idle connection
# after 5s and nothing in coderai raises that; this module held them for 60. So
# the first page after any quiet gap raised it, the gateway read it as "nothing
# is at that address", and the pool TERMINATED a rented GPU. Each pod then cost
# a 6-minute image pull to replace, which is what the operator saw as OCR
# flapping.
def _stale_socket(monkeypatch, failures=1, alive=True, fresh_ok=True):
    """An upstream whose pooled connection is closed, but which is perfectly fine.

    ``alive`` is whether its health check answers — the difference between a
    closed socket and a pod that really has gone. ``fresh_ok`` is whether the
    resend on a brand-new connection succeeds, which is how a stale socket is
    told apart from a server that is dropping every connection it is given.
    """
    import httpx
    state = {"posts": 0, "fresh_posts": 0, "probes": 0}

    class _Stale(_CapturingClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            # A one-shot client is the one built with keep-alive disabled.
            limits = kw.get("limits")
            self.pooled = not (limits is not None
                               and getattr(limits, "max_keepalive_connections", 1) == 0)

        @property
        def is_closed(self):
            return False

        async def aclose(self):
            return None

        async def get(self, url, headers=None, timeout=None):
            state["probes"] += 1
            if not alive:
                raise httpx.ConnectError("connection refused")

            class _R:
                status_code = 200
            return _R()

        async def post(self, url, content=None, headers=None):
            if self.pooled:
                state["posts"] += 1
                if state["posts"] <= failures:
                    raise httpx.RemoteProtocolError(
                        "Server disconnected without sending a response.")
            else:
                state["fresh_posts"] += 1
                if not fresh_ok:
                    raise httpx.RemoteProtocolError(
                        "Server disconnected without sending a response.")
            return await super().post(url, content=content, headers=headers)

    monkeypatch.setattr(httpx, "AsyncClient", _Stale)
    return state


def test_a_closed_pooled_connection_is_not_a_dead_address():
    """The classification itself: this is the line that cost five GPUs."""
    import httpx
    exc = httpx.RemoteProtocolError("Server disconnected without sending a response.")
    assert gateway._is_stale_socket(exc)
    assert not gateway._is_dead_upstream(exc)
    # What a genuinely empty address looks like, by contrast.
    assert gateway._is_dead_upstream(httpx.ConnectError("connection refused"))
    assert not gateway._is_stale_socket(httpx.ConnectError("connection refused"))


def test_a_stale_socket_resends_the_page_and_keeps_the_pod(monkeypatch):
    """The page is read, on a connection of its own, and nothing is terminated."""
    _reset()
    import asyncio
    discarded = []
    state = _stale_socket(monkeypatch, failures=1)
    _pod_cands(monkeypatch, discarded)
    token = gateway.issue("m", "m")
    reply = asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    assert reply.status_code == 200          # the page was read
    assert reply.body == UPSTREAM_REPLY      # and verbatim, as always
    assert state["fresh_posts"] == 1         # resent off the pool
    assert discarded == []                   # the $0.59/hr A40 is still rented
    assert dispatch.inflight() == 0


def test_the_resend_goes_to_the_same_pod_not_a_new_one(monkeypatch):
    """Re-choosing would send the page to a different pod, or rent one. The
    upstream we have is fine; only its socket was not."""
    _reset()
    import asyncio
    seen = []
    _stale_socket(monkeypatch, failures=1)

    def _acquire():
        seen.append(1)
        return ("https://10.0.0.9:8000/v1", "k", lambda: None, lambda: None)
    cand = dispatch.Candidate(key="pod:m", kind="pod", base="", priority=2,
                              acquire=_acquire, primary=True)
    monkeypatch.setattr(dispatch, "candidates", lambda m: [cand])
    monkeypatch.setattr(dispatch, "_node_reported_load", lambda c: 0)
    token = gateway.issue("m", "m")
    reply = asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    assert reply.status_code == 200
    assert len(seen) == 1                    # one lease, not two


def test_a_pod_that_answers_its_health_check_is_never_terminated(monkeypatch):
    """Even when the resend fails too: dropping a pod is irreversible, so the
    address is asked directly before anything is billed away."""
    _reset()
    import asyncio
    discarded = []
    state = _stale_socket(monkeypatch, failures=99, alive=True, fresh_ok=False)
    _pod_cands(monkeypatch, discarded)
    token = gateway.issue("m", "m")
    reply = asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    assert reply.status_code == 502          # this page failed, honestly reported
    assert discarded == []                   # but the pod kept
    assert state["probes"] >= 1              # because it was asked
    assert dispatch.inflight() == 0


def test_a_pod_that_answers_nothing_at_all_is_still_dropped(monkeypatch):
    """The original bug must stay fixed: a pod RunPod reclaimed mid-document
    still has to leave the pool, or every following page picks the corpse."""
    _reset()
    import asyncio
    discarded = []
    _stale_socket(monkeypatch, failures=99, alive=False, fresh_ok=False)
    _pod_cands(monkeypatch, discarded)
    token = gateway.issue("m", "m")
    reply = asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    assert reply.status_code == 503          # both attempts, then give up
    assert len(discarded) == 2               # and the pool was told, both times


def test_the_health_probe_counts_a_401_as_alive(monkeypatch):
    """A 401 is a server, authenticating. Only silence means nothing is there."""
    import asyncio
    import httpx

    class _Guarded(_CapturingClient):
        async def aclose(self):
            return None

        async def get(self, url, headers=None, timeout=None):
            class _R:
                status_code = 401
            return _R()

    monkeypatch.setattr(httpx, "AsyncClient", _Guarded)
    assert asyncio.run(gateway._still_alive("https://10.0.0.9:8000/v1", "k"))


def test_the_health_probe_asks_the_root_not_the_v1_path(monkeypatch):
    """/healthz lives at the root; asking /v1/healthz would 404 and read as dead."""
    import asyncio
    import httpx
    asked = []

    class _Noting(_CapturingClient):
        async def aclose(self):
            return None

        async def get(self, url, headers=None, timeout=None):
            asked.append(url)
            raise httpx.ConnectError("nothing here")

    monkeypatch.setattr(httpx, "AsyncClient", _Noting)
    assert not asyncio.run(gateway._still_alive("https://10.0.0.9:8000/v1"))
    assert asked == ["https://10.0.0.9:8000/healthz",
                     "https://10.0.0.9:8000/health"]


def test_idle_connections_are_dropped_before_the_server_drops_them():
    """uvicorn's default keep-alive is 5s. Holding a pooled connection longer
    than the server does guarantees the stale-socket race on every quiet gap,
    which is how this outage started."""
    assert gateway._KEEPALIVE_EXPIRY_S < 5.0
    # Still long enough to reuse a connection within a burst of pages, which is
    # the whole reason the pool exists.
    assert gateway._KEEPALIVE_EXPIRY_S >= 1.0


def test_a_read_timeout_still_never_probes_or_drops(monkeypatch):
    """Unchanged: a dense page is slow, not dead — and must not even cost a probe."""
    _reset()
    import asyncio
    import httpx
    discarded = []
    probes = []

    class _Slow(_CapturingClient):
        @property
        def is_closed(self):
            return False

        async def get(self, url, headers=None, timeout=None):
            probes.append(url)
            raise httpx.ConnectError("x")

        async def post(self, url, content=None, headers=None):
            raise httpx.ReadTimeout("too slow")

    monkeypatch.setattr(httpx, "AsyncClient", _Slow)
    _pod_cands(monkeypatch, discarded)
    token = gateway.issue("m", "m")
    reply = asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    assert reply.status_code == 502
    assert discarded == []
    assert probes == []
