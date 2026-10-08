"""The front must enforce the bearer on /v1, and must not fall open.

Reproduced on the published 0.2.83 image before this: GET /v1/models returned
200 with no Authorization header at all. The front answers some /v1 paths from
its own registry and proxies the rest, so whether a request needed a token
depended on which component served it.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = (ROOT / "codai/frontproxy/app.py").read_text()


def test_the_front_gates_v1():
    assert '@app.middleware("http")' in SRC
    assert "async def _api_bearer_gate" in SRC
    blk = SRC.split("async def _api_bearer_gate")[1].split("def ")[0]
    assert 'path.startswith("/v1/")' in blk
    assert "status_code=401" in blk
    assert "_api_request_authorised(request)" in blk


def test_probes_and_progress_are_exempt():
    """A probe behind auth reads as "backend down"; surya probes /health."""
    from codai.frontproxy.app import _API_AUTH_EXEMPT
    assert "/v1/health" in _API_AUTH_EXEMPT
    for p in ("/v1/images/progress", "/v1/video/progress",
              "/v1/audio/progress", "/v1/loras/progress"):
        assert p in _API_AUTH_EXEMPT, p
    # /healthz and /health are not under /v1 at all, so the gate never sees them.
    assert not any(x in _API_AUTH_EXEMPT for x in ("/healthz", "/health"))


def test_it_accepts_the_same_credentials_the_engine_does():
    blk = SRC.split("def _api_request_authorised")[1].split("app = FastAPI")[0]
    assert "CODERAI_API_TOKEN" in blk          # a locked capability image
    assert "cluster" in blk and "token" in blk # a head forwarding to this node
    assert "verify_token" in blk               # auth.json API tokens
    assert "validate_session" in blk           # a logged-in browser
    assert "x-coderai-broker-authed" in blk    # aisbf relay
    assert 'request.scope.get("server") == ("internal", 80)' in blk


def test_it_does_not_fall_open_without_credentials_configured():
    """The engine's middleware returns call_next when there is no user database
    and no env token. That fallback is exactly how an exposed orchestrator
    served /v1 to anyone, so the front must refuse instead."""
    blk = SRC.split("def _api_request_authorised")[1].split("app = FastAPI")[0]
    tail = blk.split("if not config_dir:")[1]
    assert "return False" in tail.split("try:")[0], \
        "no config dir must refuse, not allow"
    assert blk.rstrip().endswith("return False"), \
        "the helper must end by refusing, not by allowing"


def test_it_uses_constant_time_comparison():
    blk = SRC.split("def _api_request_authorised")[1].split("app = FastAPI")[0]
    assert "compare_digest" in blk
    assert "== _env" not in blk and "== shared" not in blk
