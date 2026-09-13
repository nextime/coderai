"""The admin UI must actually reach every setting — no JSON editing.

Two failure modes this catches, both invisible until someone tries to use the
page: a getElementById() for an id that no longer exists (silently reads/writes
nothing), and a form field the save path never collects (edits vanish on save).
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TPL = ROOT / "codai" / "admin" / "templates"


def _ids_defined(html: str) -> set:
    """Ids in the markup, plus ones the page assigns to elements it creates."""
    static = set(re.findall(r'\bid="([A-Za-z0-9_\-]+)"', html))
    dynamic = set(re.findall(r"\.id\s*=\s*'([A-Za-z0-9_\-]+)'", html))
    return static | dynamic


def _ids_referenced(html: str) -> set:
    """Only literal getElementById('x') calls — dynamic ids are built per row."""
    return set(re.findall(r"getElementById\(\s*'([A-Za-z0-9_\-]+)'\s*\)", html))


@pytest.mark.parametrize("page", ["settings.html", "models.html"])
def test_every_referenced_element_exists(page):
    html = (TPL / page).read_text()
    missing = sorted(_ids_referenced(html) - _ids_defined(html))
    assert missing == [], f"{page} reads ids that no element defines: {missing}"


def test_model_page_exposes_every_placement_setting():
    html = (TPL / "models.html").read_text()
    ids = _ids_defined(html)
    for want in (
            # serve this model from elsewhere
            "cfg-remote-url", "cfg-remote-served", "cfg-remote-key",
            # burst to RunPod
            "cfg-spill-enabled", "cfg-spill-busy", "cfg-spill-queue",
            "cfg-spill-nogpu", "cfg-spill-mode", "cfg-spill-endpoint",
            "cfg-spill-engine", "cfg-spill-hf-gguf", "cfg-spill-image",
            # pod shape, auth and scaling
            "cfg-rp-engine", "cfg-rp-hf-gguf", "cfg-rp-api-key", "cfg-rp-open",
            "cfg-rp-registry-auth", "cfg-rp-health-path", "cfg-rp-docker-args",
            "cfg-rp-scale-up", "cfg-rp-max-inflight", "cfg-rp-sticky"):
        assert want in ids, f"models.html has no field for {want}"


def test_model_page_saves_what_it_shows():
    """Every new field must be read back by the collect functions."""
    html = (TPL / "models.html").read_text()
    for want in ("cfg-remote-url", "cfg-spill-enabled", "cfg-spill-busy",
                 "cfg-rp-engine", "cfg-rp-api-key", "cfg-rp-open",
                 "cfg-rp-scale-up", "cfg-rp-max-inflight", "cfg-rp-sticky",
                 "cfg-rp-registry-auth", "cfg-rp-hf-gguf"):
        assert html.count(f"'{want}'") >= 2, \
            f"{want} is shown but never collected (or never populated)"


def test_settings_page_exposes_engine_service_urls_and_remotes():
    html = (TPL / "settings.html").read_text()
    ids = _ids_defined(html)
    for engine in ("ds4", "colibri", "k3", "kt", "vllm"):
        assert f"s-{engine}-service-url" in ids, f"no service URL field for {engine}"
        assert html.count(f"'s-{engine}-service-url'") >= 2, \
            f"{engine} service URL is shown but not loaded+saved"
    for want in ("s-runpod-registry-auth", "s-remotes-enabled", "s-remotes-key",
                 "s-remotes-max-body", "s-remotes-rows"):
        assert want in ids, f"settings.html has no field for {want}"
    assert "_collectRemotes()" in html and "_populateRemotes(" in html


def test_every_gateway_capability_has_a_row_in_the_ui():
    """A capability the gateway can route but the UI cannot set would be
    configurable only by editing JSON — exactly what this must avoid."""
    from codai.api.remote_gateway import _CAPABILITY_PREFIXES

    html = (TPL / "settings.html").read_text()
    listed = set(re.findall(r"\['([a-z_]+)','[^']+','[^']*'\]", html))
    routable = {cap for _, cap in _CAPABILITY_PREFIXES}
    missing = sorted(routable - listed)
    assert missing == [], f"capabilities with no UI row: {missing}"


def test_admin_accepts_every_pod_field_the_ui_sends():
    """The form can only be as good as the keys the backend keeps."""
    routes = (ROOT / "codai" / "admin" / "routes.py").read_text()
    for key in ("engine", "hf_gguf", "health_path", "docker_args",
                "registry_auth_id", "api_key", "allow_open_pod",
                "sticky_sessions", "max_inflight_per_pod", "on_busy",
                "service_url", "served_model"):
        assert f'"{key}"' in routes, f"admin API drops the {key} field"
