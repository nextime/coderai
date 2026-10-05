"""A rented pod must be able to serve the OCR engines this install is configured with.

Surya-2 and olmOCR-2 are VLMs: locally they ride coderai's own vLLM engine, which a pod
has none of, and a pod image will not get one (that venv is ~10.5 GB against a ~12 GB
ceiling, and it would be a second torch beside the one docTR and Surya already need).
A pod asked for either therefore used to be downgraded to docTR UNCONDITIONALLY — even
when the operator had pointed the engine at a reachable endpoint, which is all the served
modes need (PIL + requests, both already in the image).

So: keep the configured engine when a server is configured, seed it, and pin the fallback
chain to engines the image can actually serve.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codai.api import runpod_worker as rw
from codai.config import OcrConfig


@pytest.fixture
def ocr_cfg(monkeypatch):
    """Point runpod_worker's config accessor at an OcrConfig we control."""
    holder = {}

    def _set(**kw):
        holder["cfg"] = OcrConfig(**kw)
        return holder["cfg"]

    monkeypatch.setattr(rw, "_ocr_cfg", lambda: holder.get("cfg"))
    monkeypatch.setattr(rw, "_surya_accepted",
                        lambda: bool(getattr(holder.get("cfg"), "surya_accept_license", False)))
    return _set


# ------------------------------------------------------ the server-URL decision
def test_olmocr_in_server_mode_is_servable_on_a_pod(ocr_cfg):
    ocr_cfg(olmocr_serve="server", olmocr_server_url="https://api.example/v1")
    assert rw._ocr_external_server("olmocr") == "https://api.example/v1"


def test_olmocr_in_vllm_mode_is_not(ocr_cfg):
    """vllm mode means coderai's own engine — the thing a pod does not have."""
    ocr_cfg(olmocr_serve="vllm", olmocr_server_url="https://api.example/v1")
    assert rw._ocr_external_server("olmocr") == ""


def test_olmocr_in_model_mode_is_not(ocr_cfg):
    ocr_cfg(olmocr_serve="model")
    assert rw._ocr_external_server("olmocr") == ""


def test_server_mode_without_a_url_is_not(ocr_cfg):
    ocr_cfg(olmocr_serve="server", olmocr_server_url="")
    assert rw._ocr_external_server("olmocr") == ""


def test_surya_against_an_external_server_is_servable(ocr_cfg):
    ocr_cfg(surya_serve="llamacpp", surya_server_url="http://box:8080/v1")
    assert rw._ocr_external_server("surya") == "http://box:8080/v1"


def test_surya_on_the_local_vllm_is_not(ocr_cfg):
    ocr_cfg(surya_serve="vllm")
    assert rw._ocr_external_server("surya") == ""


def test_an_engine_that_needs_no_server_has_none(ocr_cfg):
    ocr_cfg()
    assert rw._ocr_external_server("paddle") == ""
    assert rw._ocr_external_server("doctr") == ""


def test_no_config_at_all_is_tolerated(monkeypatch):
    monkeypatch.setattr(rw, "_ocr_cfg", lambda: None)
    assert rw._ocr_external_server("olmocr") == ""


# ------------------------------------------------------ what the pod is told
class _Mcfg:
    """A RunpodModelConfig stand-in: _plan reads a handful of fields off it and we
    care about none of them here, so anything unset reads as None."""

    def __getattr__(self, _name):
        return None


def _seed(entry_path: str, alias: str = "", served: str = None) -> dict:
    """Run the OCR branch of the pod seeder and return the env it produced."""
    entry = {"path": entry_path}
    if alias:
        entry["alias"] = alias
    return rw._plan(engine="", image="img", args={}, mcfg=_Mcfg(),
                    api_key="k", entry=entry,
                    served=(entry_path if served is None else served),
                    capability="ocr").get("env", {})


def test_a_configured_olmocr_pod_keeps_olmocr_and_is_told_where_to_look(ocr_cfg):
    ocr_cfg(olmocr_serve="server", olmocr_server_url="https://api.example/v1",
            olmocr_model_id="olmocr-2")
    env = _seed("olmocr")
    assert env["CODERAI_OCR_DEFAULT_ENGINE"] == "olmocr"
    assert env["CODERAI_OCR_OLMOCR_ENABLED"] == "1"
    assert env["CODERAI_OCR_OLMOCR_SERVE"] == "server"
    assert env["CODERAI_OCR_OLMOCR_SERVER_URL"] == "https://api.example/v1"
    assert env["CODERAI_OCR_OLMOCR_MODEL_ID"] == "olmocr-2"


def test_an_unconfigured_olmocr_pod_still_falls_back_to_doctr(ocr_cfg):
    ocr_cfg(olmocr_serve="vllm")          # the local-only mode
    env = _seed("olmocr")
    assert env["CODERAI_OCR_DEFAULT_ENGINE"] == "doctr"
    assert "CODERAI_OCR_OLMOCR_SERVER_URL" not in env


def test_a_configured_surya_pod_keeps_surya(ocr_cfg):
    ocr_cfg(surya_serve="llamacpp", surya_server_url="http://box:8080/v1",
            surya_accept_license=True)
    env = _seed("surya")
    assert env["CODERAI_OCR_DEFAULT_ENGINE"] == "surya"
    assert env["CODERAI_OCR_SURYA_SERVE"] == "llamacpp"
    assert env["CODERAI_OCR_SURYA_SERVER_URL"] == "http://box:8080/v1"
    assert env["CODERAI_OCR_SURYA_ACCEPT_LICENSE"] == "1"


def test_the_licence_decision_is_still_made_by_the_renting_install(ocr_cfg):
    ocr_cfg(surya_serve="vllm")            # no server -> downgraded
    env = _seed("surya")
    assert env["CODERAI_OCR_DEFAULT_ENGINE"] == "doctr"
    # Asking for surya by name IS the licence decision, downgrade or not.
    assert env["CODERAI_OCR_SURYA_ACCEPT_LICENSE"] == "1"


# ------------------------------------------------------ the fallback chain
def test_the_fallback_chain_only_names_engines_the_image_has(ocr_cfg):
    """"auto" would hand a failed page to an engine with no server on a pod."""
    ocr_cfg(olmocr_serve="server", olmocr_server_url="https://api.example/v1")
    chain = _seed("olmocr")["CODERAI_OCR_FALLBACK_ENGINES"].split(",")
    assert chain[0] == "olmocr"            # the engine asked for leads
    assert "doctr" in chain                # and docTR is in the image
    assert "surya" not in chain and "paddle" not in chain


def test_a_paddle_pod_falls_back_to_paddle_then_doctr(ocr_cfg):
    ocr_cfg()
    env = _seed("paddle")
    assert env["CODERAI_OCR_DEFAULT_ENGINE"] == "paddle"
    assert env["CODERAI_OCR_PADDLE_VENV"] == "/opt/coderai/venvs/paddleocr"
    assert env["CODERAI_OCR_FALLBACK_ENGINES"].split(",") == ["paddle", "doctr"]


def test_the_seeded_names_are_the_ones_config_reads(ocr_cfg, monkeypatch):
    """The seeder and the reader must agree, or the pod silently ignores all of it."""
    ocr_cfg(olmocr_serve="server", olmocr_server_url="https://api.example/v1",
            surya_serve="llamacpp", surya_server_url="http://box:8080/v1")
    seeded = set(_seed("olmocr")) | set(_seed("surya"))
    src = Path("codai/config.py").read_text()
    for var in seeded:
        if var.startswith("CODERAI_OCR_") and not var.endswith("_ENABLED"):
            assert var in src, f"{var} is seeded to pods but config.py never reads it"


# ------------------------------------------------------ aliases
def test_an_aliased_entry_is_read_by_its_alias(ocr_cfg):
    """An entry is routable by `alias or path or id` throughout the manager, and
    aliases are how one model carries several configs and how several models sit
    behind one endpoint. An aliased OCR entry must not be misread and downgraded."""
    ocr_cfg(olmocr_serve="server", olmocr_server_url="https://api.example/v1")
    env = _seed("allenai/olmOCR-2-7B-1025-FP8", alias="olmocr", served="olmocr")
    assert env["CODERAI_OCR_DEFAULT_ENGINE"] == "olmocr"
    assert env["CODERAI_OCR_OLMOCR_SERVER_URL"] == "https://api.example/v1"


def test_the_served_name_also_identifies_the_engine(ocr_cfg):
    ocr_cfg()
    env = _seed("some/doctr-checkpoint", served="doctr")
    assert env["CODERAI_OCR_DEFAULT_ENGINE"] == "doctr"


def test_an_alias_that_names_no_engine_falls_back_to_the_path(ocr_cfg):
    ocr_cfg()
    env = _seed("paddle", alias="italian-documents")
    assert env["CODERAI_OCR_DEFAULT_ENGINE"] == "paddle"
    assert env["CODERAI_OCR_FALLBACK_ENGINES"].split(",") == ["paddle", "doctr"]


def test_an_aliased_surya_still_carries_the_licence(ocr_cfg):
    ocr_cfg(surya_serve="llamacpp", surya_server_url="http://box:8080/v1")
    env = _seed("datalab-to/surya-ocr-2", alias="surya", served="surya")
    assert env["CODERAI_OCR_DEFAULT_ENGINE"] == "surya"
    assert env["CODERAI_OCR_SURYA_ACCEPT_LICENSE"] == "1"
