"""httpx's default connection pool was the throughput ceiling on the whole
install, and nothing said so.

The front forwards to an engine over `self._long`, built with `pool=None` --
wait forever for a free connection -- and httpx defaults to
`max_connections=100`. Those two defaults together cap every forwarded request
at 100 in flight and make the 101st wait indefinitely.

Measured on a live orchestrator mid-campaign, with the client offering 512
pages at once:

    established to the front   514
    forwarded to the engine    100   <- the cap
    pages/s                    0.42  (the fleet was sized for ~3.3)
    mean page latency          238 s (service time measured at ~19 s)
    cost per page              12x the measured baseline

The pods were rented, healthy and unreachable. Worse, /v1/runpod/status is not
under /admin/api/ so it took the same unbounded path and never answered at all
(http=000 at 30 s, three times, while /healthz answered in 2.1 ms) -- and a
client sizing its in-flight window from `summary.pods_ready`, which is what the
throughput guide tells it to do, was flying blind as a direct result.

Two things are pinned here:

  1. the forwarding pool is sized from the install's configured page ceiling,
     never from httpx's default, and waiting for a connection is BOUNDED
  2. control-plane paths use a separate pool, so "what is the fleet doing" and
     "release this pod" still answer when the forwarding pool is full
"""
import httpx
import pytest

from codai.frontproxy.app import _forward_ceiling, _FWD_MIN, _FWD_MAX


class _Ocr:
    def __init__(self, n):
        self.max_concurrency = n


class _Cfg:
    def __init__(self, n=None):
        if n is not None:
            self.ocr = _Ocr(n)


# --- 1. the pool is sized, and never from the default --------------------

def test_the_ceiling_is_never_httpx_default():
    """100 is the number that caused this. It must not be reachable."""
    for n in (0, 1, 64, 100, 512, 8192, 10**6):
        assert _forward_ceiling(_Cfg(n)) >= _FWD_MIN > 100


def test_an_unreadable_config_falls_back_to_the_floor_not_the_default():
    """The config read is best-effort; silently reinstating the limit this
    exists to remove would be the worst possible fallback."""
    assert _forward_ceiling(_Cfg()) == _FWD_MIN


def test_the_ceiling_follows_the_configured_page_concurrency():
    assert _forward_ceiling(_Cfg(512)) > 512
    assert _forward_ceiling(_Cfg(4096)) > 4096


def test_the_ceiling_is_bounded():
    """A mistyped config must not ask for a file descriptor per page of a
    million-page corpus."""
    assert _forward_ceiling(_Cfg(10**9)) == _FWD_MAX


def test_headroom_is_left_for_everything_else_the_front_proxies():
    """A fleet saturating the page ceiling must not make the admin pages
    unreachable as a side effect."""
    for n in (512, 1024, 4096):
        assert _forward_ceiling(_Cfg(n)) - n >= 64


# --- the client is actually built with it --------------------------------

def test_the_long_client_declares_limits_and_a_bounded_pool_wait():
    src = open("codai/frontproxy/app.py").read()
    blk = src[src.index("self._long = httpx.AsyncClient("):]
    blk = blk[:blk.index("self._ctl")]
    assert "limits=httpx.Limits(max_connections=_fwd" in blk
    assert "pool=30.0" in blk, "waiting for a connection must be bounded"
    assert "pool=None" not in blk, "the unbounded wait is what reported as a hang"


def test_the_read_timeout_stays_unbounded():
    """A page may legitimately take minutes to be READ. Only the QUEUE for a
    connection is bounded -- confusing the two would truncate real work."""
    src = open("codai/frontproxy/app.py").read()
    blk = src[src.index("self._long = httpx.AsyncClient("):]
    blk = blk[:blk.index("self._ctl")]
    assert "read=None" in blk


# --- 2. the control plane has its own pool ------------------------------

def test_a_control_client_exists_with_its_own_pool():
    src = open("codai/frontproxy/app.py").read()
    blk = src[src.index("self._ctl = httpx.AsyncClient("):]
    blk = blk[:blk.index("headers=_auth)") + 14]
    assert "max_connections=16" in blk
    assert "pool=10.0" in blk, "even the control pool must not wait forever"


def test_the_control_paths_are_the_ones_needed_when_saturated():
    from codai.frontproxy.app import FrontProxy
    paths = set(FrontProxy._CONTROL_PATHS)
    # Reading the fleet, and the two calls that make it STOP costing money. A
    # pod that cannot be released because the release cannot get a connection
    # keeps billing, which is the most expensive way for this to fail.
    for p in ("/v1/runpod/status", "/v1/runpod/scale", "/v1/runpod/release"):
        assert p in paths, p


@pytest.mark.parametrize("path,control", [
    ("/v1/runpod/status", True),
    ("/v1/runpod/scale", True),
    ("/v1/runpod/release", True),
    ("/v1/ocr/batch", False),
    ("/v1/chat/completions", False),
    ("/v1/runpod/statusfoo", False),      # prefix must not match loosely
    ("/admin/api/models", False),
])
def test_control_path_classification(path, control):
    from codai.frontproxy.app import FrontProxy
    assert FrontProxy._is_control_path(FrontProxy, path) is control, path


def test_the_generic_proxy_sites_pass_the_path():
    """A classifier nothing consults is decoration. Every call site that
    forwards an arbitrary path must hand it over."""
    src = open("codai/frontproxy/app.py").read()
    assert src.count("self._lc(engine, ") >= 5
    # and none of the arbitrary-path sites may still call it path-blind
    for frag in ("self._lc(engine).build_request(\n            method, url,",
                 "self._lc(engine).request(method, engine.url + path"):
        assert frag not in src, frag


def test_a_remote_node_keeps_its_own_client():
    """The control pool is a localhost optimisation; a node has its own
    clients and its own token, and must not be reached through ours."""
    src = open("codai/frontproxy/app.py").read()
    blk = src[src.index("def _lc(self, engine"):]
    blk = blk[:blk.index("@staticmethod")]
    assert 'not getattr(engine, "remote", False)' in blk
