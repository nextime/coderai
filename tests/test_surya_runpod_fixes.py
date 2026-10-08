"""The three Surya-on-RunPod blockers reported from the Digesta host.

1. surya-ocr's vLLM client probes {service_url}/health; coderai served only
   /healthz, so /v1/ocr never became ready.
2. A rented pod ran the shared-workstation admission defaults (2 in flight,
   6 queued), so its own gate — not the GPU — capped throughput and everything
   above it got 429. The pod carries no config file, so env is the channel.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------- bug 1
def test_health_and_v1_health_alias_healthz():
    src = (ROOT / "codai/api/app.py").read_text()
    assert '@app.get("/health", include_in_schema=False)' in src
    assert '@app.get("/v1/health", include_in_schema=False)' in src
    body = src.split('async def health_alias')[1][:200]
    assert "return await healthz()" in body, "the alias must reuse the real probe"


def test_v1_health_needs_no_bearer_token():
    """A readiness probe that 401s is read as 'backend down'."""
    from codai.api.ratelimit import _PROGRESS_PATHS
    assert "/v1/health" in _PROGRESS_PATHS


def test_the_alias_returns_what_healthz_returns():
    import asyncio
    from codai.api.app import healthz, health_alias
    a = asyncio.run(healthz())
    b = asyncio.run(health_alias())
    assert a == b and a.get("ok") is True


# ----------------------------------------------------------------- bug 2
def test_the_pod_server_honours_concurrency_env():
    src = (ROOT / "codai/main.py").read_text()
    for name in ("CODERAI_SERVER_MAX_PARALLEL_REQUESTS", "CODERAI_SERVER_QUEUE_MAX_SIZE"):
        assert name in src, name
    # The front's existing knob must keep working.
    assert "CODERAI_MAX_PARALLEL" in src
    # Both the queue manager and the config object get updated, or something
    # downstream reads the stale value.
    blk = src.split("Apply queue scheduler settings")[1][:2000]
    assert "queue_manager.max_parallel_requests = int(_mp)" in blk
    assert "config.server.max_parallel_requests = int(_mp)" in blk
    assert "queue_manager.max_size = int(_qs)" in blk
    assert "config.server.queue_max_size = int(_qs)" in blk


def test_per_model_pod_concurrency_parses():
    from codai.api.runpod_worker import parse_model_runpod
    cfg = parse_model_runpod({"pod_max_parallel_requests": 48, "pod_queue_max_size": 256})
    assert cfg.pod_max_parallel_requests == 48
    assert cfg.pod_queue_max_size == 256
    # Absent means "leave the pod's own defaults alone", not zero concurrency.
    d = parse_model_runpod({})
    assert d.pod_max_parallel_requests == 0 and d.pod_queue_max_size == 0
    # Junk must not crash the provisioning path.
    j = parse_model_runpod({"pod_max_parallel_requests": "lots"})
    assert j.pod_max_parallel_requests == 0


def test_the_settings_reach_the_pod_as_env():
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    blk = src.split("def _plan(")[1].split("seeds = []")[0]
    assert 'env["CODERAI_SERVER_MAX_PARALLEL_REQUESTS"]' in blk
    assert 'env["CODERAI_SERVER_QUEUE_MAX_SIZE"]' in blk
    # Only when set: an unset value must not send "0" and wedge the pod shut.
    assert 'getattr(mcfg, "pod_max_parallel_requests", 0)' in blk


def test_zero_sends_nothing():
    """Sending CODERAI_SERVER_MAX_PARALLEL_REQUESTS=0 would admit nothing."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    blk = src.split("def _plan(")[1].split("seeds = []")[0]
    i = blk.index("pod_max_parallel_requests")
    assert "if getattr" in blk[max(0, i - 40):i], "must be guarded by a truthiness check"


# --------------------------------------------------- orchestration surfaces
def test_the_new_keys_survive_the_whitelist():
    src = (ROOT / "codai/admin/routes.py").read_text()
    blk = src.split("# Per-model runpod block")[1].split("# Per-model `host` block")[0]
    for k in ("pod_max_parallel_requests", "pod_queue_max_size"):
        assert f'"{k}"' in blk, k


@pytest.mark.parametrize("field", ["cfg-rp-pod-parallel", "cfg-rp-pod-queue"])
def test_configurable_from_the_model_page(field):
    html = (ROOT / "codai/admin/templates/models.html").read_text()
    assert html.count(field) >= 3, f"{field}: {html.count(field)} occurrences"


# ----------------------------------------------- packaging: the surya venv
def test_the_resolver_finds_a_venv_where_the_image_builder_bakes_it(tmp_path, monkeypatch):
    """<profile>.venv-<alias>.txt bakes /opt/coderai/venvs/<alias>, which is not
    /opt/coderai/<engine>_venv. Missing that made the baked venv invisible and
    the engine rebuilt it at first use."""
    from codai.ocr import subprocess_engine as se
    root = tmp_path / "opt" / "coderai"
    (root / "venvs" / "surya").mkdir(parents=True)
    monkeypatch.setattr(se.os.path, "isdir",
                        lambda p: str(p).startswith(str(root)) and se.os.path.exists(p))
    # Patch the literal prefix the resolver builds.
    src_fn = se._resolve_venv_dir
    monkeypatch.setattr(se, "_resolve_venv_dir", src_fn)
    got = None
    import os as _os
    real_isdir = _os.path.isdir
    def fake_isdir(p):
        p = str(p)
        if p.startswith("/opt/coderai/"):
            return real_isdir(str(root) + p[len("/opt/coderai"):])
        return real_isdir(p)
    monkeypatch.setattr(se.os.path, "isdir", fake_isdir)
    got = se._resolve_venv_dir("", "surya_venv", ("surya",))
    assert got == "/opt/coderai/venvs/surya"


def test_an_explicit_config_path_still_wins():
    from codai.ocr.subprocess_engine import _resolve_venv_dir
    assert _resolve_venv_dir("/somewhere/mine", "surya_venv", ("surya",)) == "/somewhere/mine"


def test_both_ocr_engines_declare_their_packaging_names():
    assert 'self.cfg.surya_venv, "surya_venv", ("surya",)' in \
        (ROOT / "codai/ocr/surya.py").read_text()
    assert 'self.cfg.paddle_venv, "paddle_venv", ("paddleocr", "paddle")' in \
        (ROOT / "codai/ocr/paddle.py").read_text()


def test_surya_is_not_in_the_ocr_main_venv():
    """surya caps pillow<11; coderai needs >=12. One of them loses."""
    main = (ROOT / "packaging/runpod/profiles/ocr.txt").read_text()
    body = [l for l in main.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert not any("surya" in l for l in body), body


def test_the_ocr_image_bakes_a_surya_venv():
    req = ROOT / "packaging/runpod/profiles/ocr.venv-surya.txt"
    chk = ROOT / "packaging/runpod/profiles/ocr.venv-surya.check"
    assert req.exists() and chk.exists()
    body = req.read_text()
    assert "surya-ocr==0.22.1" in body
    assert "pillow<11" in body, "without the cap the resolver may pull pillow 12 in"
    # The check must prove the pillow pin held, not just that surya imports.
    assert "pillow" in chk.read_text().lower()


# ------------------------------------- bug 3: one image = orchestrator + GUI
def test_the_light_app_serves_the_admin_gui_pages():
    """admin_router is the admin API (POST /login, /admin/api/*). The GUI pages
    were only on the front app, so a bare `uvicorn codai.api.app:app`
    orchestrator answered 404 for /admin and 405 for GET /login."""
    import os
    os.environ.setdefault("CODERAI_SKIP_HEAVY", "1")
    from fastapi.testclient import TestClient
    from codai.api.app import app
    c = TestClient(app, raise_server_exceptions=False)

    r = c.get("/admin", follow_redirects=False)
    assert r.status_code in (200, 302, 303), f"/admin -> {r.status_code}"
    r = c.get("/login", follow_redirects=False)
    assert r.status_code == 200, f"GET /login -> {r.status_code} (405 = API only)"
    for page in ("/admin/models", "/admin/runpod", "/admin/settings"):
        assert c.get(page, follow_redirects=False).status_code in (200, 302, 303), page


def test_the_gui_redirect_carries_the_forwarded_prefix():
    """Served under /coderai/, the login redirect must point inside the prefix or
    the browser leaves the deployment."""
    import os
    os.environ.setdefault("CODERAI_SKIP_HEAVY", "1")
    from fastapi.testclient import TestClient
    from codai.api.app import app
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/admin", headers={"X-Forwarded-Prefix": "/coderai"},
              follow_redirects=False)
    assert r.status_code in (302, 303)
    assert r.headers.get("location", "").startswith("/coderai/"), r.headers.get("location")


def test_the_api_survives_a_gui_registration_failure():
    """A missing templates or config dir must not take the orchestrator down."""
    src = (ROOT / "codai/api/app.py").read_text()
    blk = src.split("def _register_local_ui_pages")[1].split("_register_local_ui_pages()")[0]
    assert "except Exception" in blk
    assert "the API is unaffected" in blk


def test_the_forwarded_prefix_middleware_is_installed():
    src = (ROOT / "codai/api/app.py").read_text()
    assert "class _ForwardedPrefixMiddleware" in src
    assert "app.add_middleware(_ForwardedPrefixMiddleware)" in src
    assert "x-forwarded-prefix" in src


# ----------------------- exposed behind a proxy: tokens must be enforced
def test_api_tokens_are_enforced_under_a_bare_uvicorn_start():
    """BearerAuthMiddleware falls OPEN when admin.routes.session_manager is None
    and no CODERAI_API_TOKEN is set. main.py initialises it; `uvicorn
    codai.api.app:app` — the GPU-less orchestrator shape — did not, so every
    /v1/* call was served without credentials once published through a proxy."""
    import os, base64
    os.environ.setdefault("CODERAI_SKIP_HEAVY", "1")
    from fastapi.testclient import TestClient
    from codai.api.app import app
    from codai.admin import routes as ar

    assert ar.session_manager is not None, "importing the app must initialise it"
    c = TestClient(app, raise_server_exceptions=False)
    for hdr in ({}, {"Authorization": "Bearer sk-definitely-not-a-token"},
                {"Authorization": "Basic " + base64.b64encode(b"g:p").decode()}):
        assert c.get("/v1/models", headers=hdr).status_code == 401, hdr


def test_readiness_probes_stay_open_when_tokens_are_enforced():
    """A probe behind auth is not a probe: surya and every orchestrator health
    check would read 401 as 'backend down'."""
    import os
    os.environ.setdefault("CODERAI_SKIP_HEAVY", "1")
    from fastapi.testclient import TestClient
    from codai.api.app import app
    c = TestClient(app, raise_server_exceptions=False)
    for path in ("/healthz", "/health", "/v1/health"):
        assert c.get(path).status_code == 200, path


def test_the_gui_emits_prefixed_urls_behind_a_sub_path_proxy():
    """Served at https://host/coderai/, every URL the page emits must carry the
    prefix or the browser walks out of the deployment on the first click."""
    import os, re
    os.environ.setdefault("CODERAI_SKIP_HEAVY", "1")
    from fastapi.testclient import TestClient
    from codai.api.app import app
    c = TestClient(app, raise_server_exceptions=False)
    html = c.get("/login", headers={"X-Forwarded-Prefix": "/coderai"}).text
    assert 'action="/coderai/login"' in html, "the login form would post outside the prefix"
    assert "/coderai/static/admin/" in html, "assets would 404 through the proxy"
    assert re.search(r"ROOT_PATH\s*=\s*['\"]/coderai", html), "JS fetches would miss the prefix"


def test_a_fresh_install_gets_one_admin_and_is_not_locked_out():
    """Enforcing tokens would otherwise brick a brand-new config dir: no token
    means every /v1 call is refused and no user means nobody can log in to make
    one. Nothing else auto-creates that first user."""
    import json, pathlib, tempfile
    from codai.admin.auth import SessionManager
    from codai.api.app import _bootstrap_first_admin

    d = tempfile.mkdtemp()
    sm = SessionManager(pathlib.Path(d))
    _bootstrap_first_admin(sm, d)
    users = json.load(open(pathlib.Path(d) / "auth.json"))["users"]
    assert [u["username"] for u in users] == ["admin"]
    assert users[0]["role"] == "admin"
    # A password printed to a container log is not a password.
    assert users[0]["must_change_password"] is True


def test_the_bootstrap_never_touches_an_existing_user():
    import json, pathlib, tempfile
    from codai.admin.auth import SessionManager
    from codai.api.app import _bootstrap_first_admin

    d = tempfile.mkdtemp()
    sm = SessionManager(pathlib.Path(d))
    _bootstrap_first_admin(sm, d)
    before = json.load(open(pathlib.Path(d) / "auth.json"))["users"]
    _bootstrap_first_admin(sm, d)
    after = json.load(open(pathlib.Path(d) / "auth.json"))["users"]
    assert len(after) == 1
    assert before[0]["password_hash"] == after[0]["password_hash"]


def test_coderai_locks_every_pod_it_rents():
    """The operator never sets a pod's token: the pool generates one unless the
    model explicitly opts out, so a pod is never open on a public proxy URL."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    assert 'self.api_key = "cra-" + secrets.token_urlsafe(32)' in src
    assert 'allow_open_pod' in src
    i = src.index('self.api_key = "cra-"')
    guard = src[max(0, i - 200):i]
    assert "allow_open_pod" in guard, "generation must be the default, not opt-in"
