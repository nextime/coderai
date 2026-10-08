"""Which address forward-auth actually compares against trusted_proxies.

Reported from the Digesta prod host on 2026-10-08: with forward-auth enabled, a
request carrying the right secret and user headers minted a session when it came
from inside the container (`127.0.0.1:8776`, the internal nginx) but 302'd to the
login form when it came from the host through the published port (`:8777`) --
the exact path the real nginx uses. The operator had 10.89.0.178 in
trusted_proxies and the front's own access log showed that very address as the
client, so the rejection looked impossible.

Two things were true at once:

1. The peer a trusted-header check sees is NOT the socket peer. uvicorn installs
   ProxyHeadersMiddleware by default with trusted_hosts="127.0.0.1", so for a
   connection arriving over loopback from the internal nginx it REWRITES
   scope["client"] from X-Forwarded-For. On a container network that is the
   address the proxy was SNATed to, not 127.0.0.1. The first class of tests here
   pins that resolution down, because everything else follows from it.

2. Their config.json was at a path the app does not read (it resolves config from
   the HOME-style `$CODERAI_CONFIG_DIR/coderai`), which is also why `enabled:
   true` alone did not switch the feature on and the env var was needed. So the
   list actually in effect was the DEFAULT ["127.0.0.1", "::1"] -- loopback in,
   container address out. One cause, both symptoms.

Hence the env override: the feature could already be turned on, and its secret
supplied, from a unit file, but the peer list could only come from the file. You
could enable it by env and have it reject every request from your own proxy.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
UI = (ROOT / "codai/frontproxy/ui_pages.py").read_text()


def _peer_seen_by_the_app(xff: str, socket_peer: str = "127.0.0.1",
                          trusted_hosts: str = "127.0.0.1") -> str:
    """What request.client.host is, behind uvicorn's proxy-headers middleware."""
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    async def who(request):
        return PlainTextResponse(request.client.host if request.client else "")

    app = ProxyHeadersMiddleware(Starlette(routes=[Route("/", who)]),
                                 trusted_hosts=trusted_hosts)
    client = TestClient(app, client=(socket_peer, 40000))
    return client.get("/", headers={"x-forwarded-for": xff} if xff else {}).text


# ------------------------------------------------- what the peer actually resolves to

def test_the_peer_is_the_forwarded_address_not_the_socket():
    """The TCP connection is from loopback, but the peer reported is the proxy's."""
    assert _peer_seen_by_the_app("10.89.0.178") == "10.89.0.178"


def test_a_loopback_hop_still_looks_like_loopback():
    """Which is why the in-container test minted and looked like proof."""
    assert _peer_seen_by_the_app("127.0.0.1") == "127.0.0.1"


def test_an_outer_proxy_adds_a_hop_and_the_peer_is_still_the_inner_one():
    """host nginx -> published port -> internal nginx: $proxy_add_x_forwarded_for
    appends, and uvicorn walks from the right to the first untrusted entry."""
    assert _peer_seen_by_the_app("203.0.113.5, 10.89.0.178") == "10.89.0.178"


def test_a_client_cannot_forge_loopback_by_sending_the_header():
    """Our nginx APPENDS the real address, so a spoofed entry is never the one
    picked -- the peer stays the proxy's own address, not the claimed 127.0.0.1."""
    assert _peer_seen_by_the_app("127.0.0.1, 10.89.0.178") == "10.89.0.178"


def test_the_default_trusted_proxies_reject_a_container_address():
    """The failure in one line: the list in effect was the default."""
    from codai.config import AdminForwardAuthConfig
    default = AdminForwardAuthConfig().trusted_proxies
    assert default == ["127.0.0.1", "::1"]
    assert _peer_seen_by_the_app("127.0.0.1") in default
    assert _peer_seen_by_the_app("10.89.0.178") not in default


# --------------------------------------------------------------- the env override

def test_the_trusted_proxy_list_has_an_env_override():
    assert "CODERAI_ADMIN_FORWARD_AUTH_TRUSTED_PROXIES" in UI


def test_the_env_list_is_read_where_the_trust_decision_is_made():
    blk = UI.split("def _fa_trusted(")[1].split("def _fa_identity")[0]
    assert "_fa_peer_allowlist(cfg)" in blk, "the override must not be bypassed"


def test_the_env_list_accepts_commas_or_spaces():
    blk = UI.split("def _fa_peer_allowlist")[1].split("def _fa_reject")[0]
    assert 'replace(",", " ").split()' in blk


def test_an_empty_env_value_does_not_wipe_the_configured_list():
    """An unset or blank variable must fall through to config.json, not trust
    nothing -- that would be a second way to break a working deployment."""
    blk = UI.split("def _fa_peer_allowlist")[1].split("def _fa_reject")[0]
    assert "if raw.strip():" in blk
    assert "cfg.trusted_proxies" in blk.split("if raw.strip():")[1]


def test_the_override_is_documented_with_the_reason():
    blk = UI.split("def _fa_peer_allowlist")[1].split("def _fa_reject")[0]
    assert "read-only" in blk or "does not read" in blk


# ------------------------------------------------------------------- diagnosability

def test_a_refusal_says_why():
    """A refusal is a 302 to the login form, which is indistinguishable from "no
    session yet". Without a reason in the log this took a day to find."""
    blk = UI.split("def _fa_reject")[1].split("def _fa_trusted(")[0]
    assert "forward-auth refused" in blk


def test_the_refusal_names_the_peer_and_the_list_it_was_checked_against():
    blk = UI.split("def _fa_trusted(")[1].split("def _fa_identity")[0]
    assert "trusted_proxies {trusted}" in blk, "print the list, not just the peer"
    assert "_fa_reject(peer," in blk


def test_a_missing_secret_and_a_wrong_secret_are_told_apart():
    """"no header" and "does not match" are different operator mistakes."""
    blk = UI.split("def _fa_trusted(")[1].split("def _fa_identity")[0]
    assert "no {cfg.shared_secret_header} header" in blk
    assert "does not match" in blk


def test_the_refusal_log_never_prints_the_secret():
    blk = UI.split("def _fa_trusted(")[1].split("def _fa_identity")[0]
    for bad in ("{secret}", "{sent}"):
        assert bad not in blk, f"{bad} would put a credential in the log"


def test_the_refusal_log_is_rate_limited_and_bounded():
    """A misconfigured proxy retries forever; it must not fill the disk."""
    blk = UI.split("def _fa_reject")[1].split("def _fa_trusted(")[0]
    assert "_FA_REJECT_WINDOW" in blk
    assert "_FA_REJECTS.clear()" in blk, "the key set must be bounded"
    assert "return False" in blk, "_fa_reject stands in for a refusal"


def test_rejections_are_keyed_by_peer_and_reason():
    blk = UI.split("def _fa_reject")[1].split("def _fa_trusted(")[0]
    assert "key = (peer, reason)" in blk


# ------------------------------------------------------------- the docstring is true

def test_the_docstring_no_longer_claims_the_immediate_peer():
    """It says "immediate peer" nowhere, because that is not what is checked --
    believing it is what made the report look impossible."""
    blk = UI.split("def _fa_trusted(")[1].split("def _fa_identity")[0]
    assert "immediate peer" not in blk
    assert "ProxyHeadersMiddleware" in blk, "say where the value comes from"
