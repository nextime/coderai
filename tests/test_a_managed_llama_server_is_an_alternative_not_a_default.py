# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""A managed ``llama-server`` serves the SAME GGUF files the in-process
llama-cpp-python path serves. That makes one property load-bearing above all the
others: enabling this backend must not move the local GGUF catalogue onto it.
Every other engine here claims models by name or architecture; this one must
claim nothing it was not explicitly pinned to.

The second group covers the architecture gate. A llama.cpp tree tuned for one
CUDA architecture, built for another, does not fail cleanly -- it can compile,
load and answer. So the gate fails closed by default, and these tests pin that
default rather than the convenience of ignoring it.
"""

import json

import pytest

from codai.api import llamasrv_worker as worker
from codai.config import LlamaSrvConfig


# ---------------------------------------------------------------------------
# routing: pin-only
# ---------------------------------------------------------------------------

def _catalogue(monkeypatch, entries, cfg):
    """Point the real resolver at a fake catalogue and an enabled llamasrv.

    Patched as attributes of the real module: the resolver reads the package
    attribute, so a fake in sys.modules is simply not seen -- and a test that
    thinks it is passes for the wrong reason.
    """
    from codai.models import manager as real
    monkeypatch.setattr(real, "get_active_llamasrv_config", lambda: cfg)
    monkeypatch.setattr(real, "_model_entry_for",
                        lambda name: next((e for e in entries
                                           if e.get("id") == name), None))


def test_an_enabled_backend_does_not_claim_a_gguf_it_was_not_pinned_to(monkeypatch):
    """The whole point. llama-server serves the same files as the in-process
    path, so enabling it must change where NOTHING goes until a model says so."""
    from codai.models.manager import resolve_engine_backend
    entries = [{"id": "qwen3-8b.gguf", "path": "/models/qwen3-8b.gguf"},
               {"id": "a-llama-model.gguf", "path": "/models/a-llama-model.gguf"},
               {"id": "llamasrv", "path": "/models/llamasrv.gguf"}]
    _catalogue(monkeypatch, entries, LlamaSrvConfig(enabled=True))

    for entry in entries:
        assert resolve_engine_backend(entry["id"]) is None, (
            f"{entry['id']} was claimed by an engine with no pin on it")


def test_an_explicit_pin_routes_there(monkeypatch):
    from codai.models.manager import llamasrv_should_handle, resolve_engine_backend
    entries = [{"id": "drafted.gguf", "path": "/m/drafted.gguf",
                "backend": "llamasrv"}]
    _catalogue(monkeypatch, entries, LlamaSrvConfig(enabled=True))

    assert resolve_engine_backend("drafted.gguf") == "llamasrv"
    assert llamasrv_should_handle("drafted.gguf") is True


def test_a_pin_to_a_disabled_backend_routes_nowhere_rather_than_to_the_bindings(
        monkeypatch):
    """A pin to a disabled engine must NOT silently fall back to the in-process
    path: the model was pinned here for a reason (drafting, slots) and serving it
    without them would look like the feature working badly rather than off."""
    from codai.models.manager import resolve_engine_backend
    entries = [{"id": "drafted.gguf", "path": "/m/drafted.gguf",
                "backend": "llamasrv"}]
    _catalogue(monkeypatch, entries, LlamaSrvConfig(enabled=False))

    assert resolve_engine_backend("drafted.gguf") is None


def test_the_model_id_alias_routes_there(monkeypatch):
    from codai.models.manager import resolve_engine_backend
    entries = [{"id": "served-by-the-server", "path": "/m/x.gguf"}]
    _catalogue(monkeypatch, entries,
               LlamaSrvConfig(enabled=True, model_id="served-by-the-server"))

    assert resolve_engine_backend("served-by-the-server") == "llamasrv"


def test_the_front_resolver_agrees_with_the_manager():
    """The front routes a request without importing the model layer, so it has
    its own copy of the precedence rules. The two disagreeing means a model is
    assigned to one engine and proxied to another."""
    from codai.frontproxy.router import required_capability

    assert required_capability("x.gguf", backend="llamasrv") == "llamasrv"
    # Pin-only on this side too: enabled, with no pin and no alias, claims nothing.
    assert required_capability("some-model.gguf", path="/v1/chat/completions",
                               llamasrv_enabled=True) != "llamasrv"
    assert required_capability("the-alias", llamasrv_enabled=True,
                               llamasrv_model_id="the-alias") == "llamasrv"


def test_a_pinned_model_is_assigned_to_the_llamasrv_engine(tmp_path):
    """End to end through the front's assignment: a pinned entry must land on the
    node that owns the subprocess, not on the nvidia engine that holds GGUFs."""
    from codai.frontproxy.assignment import compute_assignment

    class E:
        def __init__(self, name, caps):
            self.name, self._caps = name, caps
            self.backend = "nvidia"
        def can_serve(self, cap):
            return cap is None or cap in self._caps

    models = tmp_path / "models.json"
    models.write_text(json.dumps({"text_models": [
        {"id": "plain.gguf", "path": "/m/plain.gguf"},
        {"id": "drafted.gguf", "path": "/m/drafted.gguf", "backend": "llamasrv"},
    ]}))
    engines = [E("nvidia", {"gguf", "transformers"}), E("llamasrv", {"llamasrv"})]

    got = compute_assignment(engines, str(models), llamasrv_cfg=LlamaSrvConfig(enabled=True))
    assert "/m/drafted.gguf" in got["llamasrv"] or "drafted.gguf" in got["llamasrv"]
    assert not any("plain" in m for m in got["llamasrv"])


# ---------------------------------------------------------------------------
# the architecture gate
# ---------------------------------------------------------------------------

def _caps(monkeypatch, caps):
    from codai.backends import gpu_probe
    monkeypatch.setattr(gpu_probe, "compute_capabilities", lambda: list(caps))


def test_a_blank_architecture_builds_for_the_cards_present(monkeypatch):
    _caps(monkeypatch, ["86", "86", "89"])
    arch, found = worker.resolve_architectures(LlamaSrvConfig())
    # Duplicates collapsed -- two identical cards are one SASS target -- order kept.
    assert arch == "86;89"
    assert found == ["86", "86", "89"]


def test_an_architecture_the_host_does_not_have_is_refused(monkeypatch):
    """Fails closed. A tuned tree on the wrong card can answer with degraded
    output rather than erroring, so 'it built' is not evidence of anything."""
    _caps(monkeypatch, ["86"])
    with pytest.raises(RuntimeError, match="sm_86"):
        worker.resolve_architectures(LlamaSrvConfig(cuda_architectures="90"))


def test_the_gate_can_be_overridden_explicitly(monkeypatch):
    _caps(monkeypatch, ["86"])
    arch, _ = worker.resolve_architectures(
        LlamaSrvConfig(cuda_architectures="90", require_matching_arch=False))
    assert arch == "90"


def test_an_unprobeable_host_is_refused_rather_than_assumed(monkeypatch):
    """No cards probed with an explicit architecture is a probe failure, not a
    match: building anyway would mean the gate reported 'safe' without looking."""
    _caps(monkeypatch, [])
    with pytest.raises(RuntimeError, match="no CUDA card could be probed"):
        worker.resolve_architectures(LlamaSrvConfig(cuda_architectures="86"))


def test_no_cuda_and_no_request_falls_back_to_a_cpu_build(monkeypatch):
    _caps(monkeypatch, [])
    arch, found = worker.resolve_architectures(LlamaSrvConfig())
    assert arch == "" and found == []


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def test_the_argv_carries_the_settings_that_make_this_backend_worth_having():
    cfg = LlamaSrvConfig(parallel=12, flash_attn="on", cache_type_k="q8_0",
                         cache_type_v="q8_0", spec_type="draft-mtp-adaptive",
                         spec_draft_n_max=4, n_gpu_layers=-1,
                         extra_args="--no-context-shift --jinja")
    argv = worker.build_argv(cfg, "/b/llama-server", "/m/x.gguf", 18900,
                             "127.0.0.1", "BEARER", ctx=122880)

    assert argv[0] == "/b/llama-server"
    for flag, val in (("--port", "18900"), ("-m", "/m/x.gguf"),
                      ("--api-key", "BEARER"), ("-c", "122880"),
                      ("-ngl", "999"), ("-np", "12"), ("-fa", "on"),
                      ("-ctk", "q8_0"), ("-ctv", "q8_0"),
                      ("--spec-type", "draft-mtp-adaptive"),
                      ("--spec-draft-n-max", "4")):
        assert flag in argv, f"{flag} missing from {argv}"
        assert argv[argv.index(flag) + 1] == val, f"{flag} got {argv[argv.index(flag) + 1]}"
    # extra_args is split like a shell would, not passed as one blob.
    assert "--no-context-shift" in argv and "--jinja" in argv


def test_a_per_request_context_beats_the_configured_one():
    """The window the model is SERVED with has to be the one the launch gets, or
    the number reported to clients and the number enforced disagree by design."""
    cfg = LlamaSrvConfig(ctx=8192)
    argv = worker.build_argv(cfg, "/b/s", "/m/x.gguf", 1, "h", "", ctx=32768)
    assert argv[argv.index("-c") + 1] == "32768"
    argv = worker.build_argv(cfg, "/b/s", "/m/x.gguf", 1, "h", "", ctx=None)
    assert argv[argv.index("-c") + 1] == "8192"


def test_no_context_flag_at_all_leaves_the_servers_own_default():
    """Not the same as passing 0: llama-server reads the GGUF's trained window
    when -c is absent, and a 0 would be an explicit answer we do not have."""
    argv = worker.build_argv(LlamaSrvConfig(), "/b/s", "/m/x.gguf", 1, "h", "")
    assert "-c" not in argv


# ---------------------------------------------------------------------------
# the bearer
# ---------------------------------------------------------------------------

def test_every_proxy_call_site_sends_the_bearer():
    """The proxy is inherited from the vLLM backend, which talks to a server
    launched WITHOUT a key. Auth that reaches only some call sites is worse than
    none: the model answers until the first streamed request. So assert the hook
    is consulted at every requests.* in the inherited proxy, not just that the
    header is built."""
    import inspect

    from codai.backends import vllm as vllm_mod
    src = inspect.getsource(vllm_mod)
    calls = [l for l in src.splitlines()
             if ("requests.get(" in l or "requests.post(" in l)]
    assert calls, "no HTTP call sites found -- did the proxy stop using requests?"
    # Each call spans up to two lines; check the header kwarg is in the statement.
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if "requests.get(" not in line and "requests.post(" not in line:
            continue
        stmt = " ".join(lines[i:i + 3])
        assert "self._headers()" in stmt, (
            f"call site without the headers hook: {line.strip()}")


def test_the_bearer_reaches_the_authorization_header():
    from codai.backends.llamasrv import LlamaSrvBackend
    b = LlamaSrvBackend(LlamaSrvConfig())
    assert b._headers() == {}, "no server running yet -- nothing to authorise"
    b._bearer = "tok-abc"
    assert b._headers() == {"Authorization": "Bearer tok-abc"}


def test_the_inherited_proxy_sends_no_header_of_its_own():
    """The hook's default must stay empty, or enabling it here changes how the
    vLLM backend talks to its own server."""
    from codai.backends.vllm import VllmBackend
    assert VllmBackend()._headers() == {}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_every_field_survives_a_save(tmp_path, monkeypatch):
    """A config knob that does not persist is the usual way one silently stops
    working: it is set in the UI, it applies once, and it is gone next boot."""
    import dataclasses

    from codai.config import ConfigManager

    cm = ConfigManager(str(tmp_path))
    cm.load()
    cm.config.llamasrv.enabled = True
    cm.config.llamasrv.tree = "repo"
    cm.config.llamasrv.repo_url = "https://example.invalid/llama-fork"
    cm.config.llamasrv.git_ref = "v0.6.4"
    cm.config.llamasrv.cuda_architectures = "86"
    cm.config.llamasrv.parallel = 12
    cm.config.llamasrv.spec_type = "draft-mtp-adaptive"
    cm.save_config()

    cm2 = ConfigManager(str(tmp_path))
    cm2.load()
    reloaded = cm2.config.llamasrv
    assert reloaded.enabled is True
    assert reloaded.tree == "repo"
    assert reloaded.git_ref == "v0.6.4"
    assert reloaded.parallel == 12
    assert reloaded.spec_type == "draft-mtp-adaptive"

    saved = json.loads((tmp_path / "config.json").read_text())["llamasrv"]
    missing = [f.name for f in dataclasses.fields(LlamaSrvConfig)
               if f.name not in saved]
    assert not missing, f"fields dropped on save: {missing}"


def test_a_per_model_block_cannot_switch_off_the_architecture_gate():
    """A per-model block refines how a model is served. It must not be able to
    disable a safety check for everyone sharing the build."""
    from codai.backends.llamasrv import _LLAMASRV_FIELDS
    for forbidden in ("require_matching_arch", "install_dir", "auto_build",
                      "service_url", "api_key"):
        assert forbidden not in _LLAMASRV_FIELDS


def test_each_tree_builds_in_its_own_directory():
    """Switching tree or bumping the ref must not build over the old tree: a
    half-overwritten build directory looks like a broken fork, not a dirty tree."""
    a = worker._install_dir(LlamaSrvConfig())
    b = worker._install_dir(LlamaSrvConfig(
        tree="repo", repo_url="https://example.invalid/llama-fork", git_ref="v0.6.4"))
    c = worker._install_dir(LlamaSrvConfig(
        tree="repo", repo_url="https://example.invalid/llama-fork", git_ref="v0.6.5"))
    assert a != b != c and a != c
    assert a.name == "bindings"
    assert "v0.6.4" in b.name and "llama-fork" in b.name


# ---------------------------------------------------------------------------
# the dedicated pod
# ---------------------------------------------------------------------------

def test_the_pod_engine_and_image_resolve():
    from codai.api.runpod_worker import (default_capability_image,
                                         engine_pod_backend)

    class M:
        engine = "coderai-llamasrv"

    assert engine_pod_backend({}, M()) == "llamasrv"
    # ...and a locally-pinned model that bursts to RunPod carries the pin itself.
    assert engine_pod_backend({"backend": "llamasrv"}, None) == "llamasrv"
    assert default_capability_image("llamasrv").endswith("coderai-llamasrv:latest")


def test_the_pod_is_never_told_how_to_build_its_own_server():
    """The binary is compiled in the image's toolchain stage and the runtime
    carries no compiler. Forwarding the renter's tree, ref or architectures
    would describe a build the pod cannot perform -- and an architecture it was
    not compiled for, which is the failure that answers instead of erroring."""
    from codai.api.runpod_worker import _ENGINE_FORWARDED_FIELDS

    forwarded = set(_ENGINE_FORWARDED_FIELDS["llamasrv"])
    for build_knob in ("tree", "repo_url", "git_ref", "cuda_architectures",
                       "extra_cmake_args", "require_matching_arch", "auto_build",
                       "install_dir", "api_key", "service_url"):
        assert build_knob not in forwarded, f"{build_knob} must not reach a pod"


def test_a_pod_enables_the_backend_from_its_environment(tmp_path, monkeypatch):
    from codai.config import ConfigManager

    monkeypatch.setenv("CODERAI_LLAMASRV_ENABLED", "1")
    monkeypatch.setenv("CODERAI_LLAMASRV_CONFIG", json.dumps({
        "ctx": 122880, "parallel": 12, "spec_type": "draft-mtp-adaptive",
        # Must be ignored: this describes the RENTER's build, not the image's.
        "cuda_architectures": "120", "tree": "repo",
        "repo_url": "https://example.invalid/other-fork",
    }))
    cm = ConfigManager(str(tmp_path))
    cm.load()

    srv = cm.config.llamasrv
    assert srv.enabled is True
    assert srv.ctx == 122880 and srv.parallel == 12
    assert srv.spec_type == "draft-mtp-adaptive"
    assert srv.cuda_architectures == "", "a pod took the renter's architecture"
    assert srv.tree == "bindings", "a pod took the renter's source tree"
    assert srv.repo_url == ""


def test_the_image_tells_the_worker_where_the_compiled_binary_is(monkeypatch):
    monkeypatch.setenv("CODERAI_LLAMASRV_BIN",
                       "/opt/coderai/llamasrv/bin/llama-server")
    got = worker._server_bin(worker._install_dir(LlamaSrvConfig(tree="repo")))
    assert str(got) == "/opt/coderai/llamasrv/bin/llama-server", (
        "the env override must win over any path derived from the tree: the "
        "binary was compiled by build args this runtime cannot see")


def test_a_missing_prebuilt_binary_says_so_instead_of_building_on_the_pod(
        monkeypatch, tmp_path):
    missing = tmp_path / "nowhere" / "llama-server"
    monkeypatch.setenv("CODERAI_LLAMASRV_BIN", str(missing))
    # If this tried to build, it would fail much later inside cmake -- with a
    # message about a compiler rather than about a path that moved.
    with pytest.raises(RuntimeError, match="CODERAI_LLAMASRV_BIN"):
        worker.ensure_built(LlamaSrvConfig())


def test_a_per_model_block_travels_to_the_pod():
    """Without this the pod serves the model with the install-wide settings,
    and a model asking for 12 slots gets whatever the orchestrator uses."""
    import inspect

    from codai.api import runpod_worker
    src = inspect.getsource(runpod_worker.seed_model_env)
    assert '"llamasrv"' in src, (
        "the llamasrv block is not in seed_model_env's keep list, so a pod "
        "would never see this model's own server settings")
