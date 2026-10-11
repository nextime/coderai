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
    assert url == f"http://127.0.0.1:8777/v1/ocr/engine/{token}/v1"
    assert url.endswith("/v1")
    assert url[: -len("/v1")] == f"http://127.0.0.1:8777/v1/ocr/engine/{token}"


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
    assert "/v1/ocr/engine/" in base and base.endswith("/v1")
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
    assert _through_bearer_gate(f"/v1/ocr/engine/{token}/health")
    assert _through_bearer_gate(f"/v1/ocr/engine/{token}/v1/models")
    assert _through_bearer_gate(f"/v1/ocr/engine/{token}/v1/chat/completions",
                                {"authorization": f"Bearer {token}"})


def test_a_forged_lease_is_not_a_way_in(monkeypatch):
    """Otherwise the exemption would be a hole in the API gate rather than a
    credential the engine happens to hold."""
    _reset()
    monkeypatch.setenv("CODERAI_API_TOKEN", "some-api-token")
    gateway.issue("m", "m")
    assert not _through_bearer_gate("/v1/ocr/engine/forged/health")


def test_a_revoked_lease_stops_working(monkeypatch):
    _reset()
    monkeypatch.setenv("CODERAI_API_TOKEN", "some-api-token")
    token = gateway.issue("m", "m")
    assert _through_bearer_gate(f"/v1/ocr/engine/{token}/health")
    gateway.revoke("m")
    assert not _through_bearer_gate(f"/v1/ocr/engine/{token}/health")


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
    asyncio.run(mw({"type": "http", "path": f"/v1/ocr/engine/{token}/health",
                    "headers": []}, None, _send))
    assert reached == [f"/v1/ocr/engine/{token}/health"]

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
    assert set(rows) == {"paddle", "doctr", "surya", "olmocr"}
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
