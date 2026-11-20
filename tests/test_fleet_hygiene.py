"""Four small things that each cost a wrong diagnosis or a wasted boot.

None of them broke a request. They made the system harder to read while it was
breaking, which on a fleet of rented machines is its own kind of cost.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROFILES = ROOT / "packaging" / "runpod" / "profiles"


# --------------------------------------------------------------------- deps
@pytest.mark.parametrize("profile", [
    "audio", "ocr", "stt", "tts", "voice", "embeddings",
    "tts-xtts.venv-TTS", "stt-crisper.venv-transformers",
])
def test_a_profile_with_transformers_also_has_accelerate(profile):
    """transformers refuses a `device_map` load without accelerate, so the pod
    boots healthy, answers /v1/models, and fails EVERY request with something
    that reads as a model problem. Measured on a real A40 for embeddings: 303s
    of boot paid before anyone could find out."""
    text = (PROFILES / f"{profile}.txt").read_text().lower()
    if "transformers" not in text:
        pytest.skip(f"{profile} does not carry transformers")
    assert "accelerate" in text, f"{profile} has transformers and no accelerate"


def test_every_profile_is_covered_by_that_rule():
    """So a new profile cannot quietly reintroduce the gap."""
    missing = []
    for f in sorted(PROFILES.glob("*.txt")):
        t = f.read_text().lower()
        if any(l.startswith("transformers") for l in t.splitlines()) \
                and "accelerate" not in t:
            missing.append(f.name)
    assert not missing, f"transformers without accelerate: {missing}"


# --------------------------------------------------------------- pod names
def test_a_pool_pod_name_has_no_colon_in_it():
    """`pool:` is legal on RunPod and makes a console listing hard to read.
    Ownership is the `coderai-<deployment>-` prefix, never the rest."""
    import re
    src = (ROOT / "codai" / "api" / "runpod_worker.py").read_text()
    assert 'if _key.startswith("pool:")' in src
    # and the name is still prefixed, which is what the reaper matches
    assert re.search(r'name = \(f"coderai-\{_deploy_tag', src)


# ------------------------------------------------------------- the reaper
def test_renaming_the_deployment_does_not_orphan_running_pods():
    """Ownership was the NAME alone, so renaming `deployment_id` instantly
    orphaned every pod already running under the old prefix: still billing,
    and no longer anything's to reap. The registry is written when we create a
    pod, so it is the better authority; the prefix stays as the fallback for a
    pod this process never recorded."""
    src = (ROOT / "codai" / "api" / "runpod_worker.py").read_text()
    assert "name.startswith(prefix) or str(pid) in _ours" in src


# ---------------------------------------------------------------- the log
def test_the_seed_plan_is_announced_once_not_every_fifteen_seconds():
    """A plan is recomputed on every scaler pass while a pod boots, and the
    repeated line buries what matters in a log somebody is reading precisely
    because something went wrong."""
    from codai.api import runpod_worker as rw
    assert isinstance(rw._PLAN_SAID, dict)
    src = (ROOT / "codai" / "api" / "runpod_worker.py").read_text()
    assert "_PLAN_SAID.get(_sig)" in src


# -------------------------------------------------------------- warm-up
def test_an_embeddings_pod_is_warmed_on_the_embeddings_endpoint():
    """Asking an embeddings pod for a chat completion is a 404 BY DESIGN, and
    it was printed on every boot of a perfectly healthy pod. It reads as a
    broken pod; it cost a wrong diagnosis."""
    src = (ROOT / "codai" / "api" / "runpod_worker.py").read_text()
    i = src.index("def _warm_up") if "def _warm_up" in src else 0
    window = src[i:i + 4000] if i else src
    assert 'cap == "embeddings"' in window
    assert '"/v1/embeddings", {"model": name, "input": "ok"}' in window


# --------------------------------------------------------- the OCR lease
def test_an_engine_attachment_is_never_forwarded():
    """`/v1/ocr/attach/<lease>` is an engine attaching to THIS install's
    gateway. The lease was minted here and lives in this process's memory, so
    forwarding the request hands a pod a token it never issued:

        404 {"error": {"message": "unknown OCR engine lease"}}

    surya's client swallows that and returns an empty generation, so every
    page comes back unread with the GPU never asked to read one. Measured: the
    pod's vLLM logged a single 6-token warm-up for a whole document, and $88 of
    surya time in one day produced nothing.

    The gateway behind this prefix does its own forwarding, with its own pool
    and its own lease. That is the hop, and it has to start here."""
    from codai.api.remote_gateway import resolve_target
    for path in ("/v1/ocr/attach/TOK/v1/chat/completions",
                 "/v1/ocr/attach/TOK/v1/models",
                 "/v1/ocr/attach/TOK/health"):
        assert resolve_target(path, "POST", "", b"{}", "application/json") is None, path


def test_an_engine_attachment_is_not_forwarded_even_when_forced():
    """`force=remote` is a test lever; it must not break the lease."""
    from codai.api.remote_gateway import resolve_target
    assert resolve_target("/v1/ocr/attach/TOK/v1/chat/completions", "POST", "",
                          b"{}", "application/json", force="remote") is None


def test_real_ocr_work_is_still_placeable():
    """Only the ATTACHMENT is pinned. /v1/ocr itself must stay routable, or
    pinning the lease would strand the whole subsystem on the orchestrator."""
    from codai.api.remote_gateway import capability_for, _LOCAL_PREFIXES
    assert capability_for("/v1/ocr") == "ocr"
    assert not "/v1/ocr".startswith(_LOCAL_PREFIXES)
    assert not "/v1/ocr/batch".startswith(_LOCAL_PREFIXES)
