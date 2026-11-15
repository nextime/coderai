"""Stock llama.cpp and a fork are two engines, not one switch.

``llamasrv`` runs llama.cpp's own ``llama-server``; ``llamasrv_fork`` runs the
same server built from a fork. They exist separately because the reason to run a
fork is to find out whether it is actually better, and that needs both serving at
once -- one model on stock, another on the fork, same card, same weights. A
single ``tree`` setting can only answer "which one is everything using".

Separation has to hold in four places, and these tests hold it there:

* routing -- a pin reaches the engine it names, and neither engine ever claims a
  GGUF by itself (both can serve every GGUF, so inference would steal the
  catalogue);
* per-engine state -- build directory, running-server key and prebuilt-binary
  variable, or the second engine silently reuses the first one's server and a
  comparison between them becomes a comparison of one with itself;
* the image -- the runtime carries no CUDA toolchain, so BOTH binaries are baked
  or the engine has nothing to launch;
* the config and the UI -- two blocks, independently enabled, every field
  readable.
"""

import dataclasses
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "packaging" / "linux" / "Dockerfile.oci"
UPDATE_LLAMA = ROOT / "packaging" / "linux" / "Dockerfile.update-llama"
LAUNCHER = ROOT / "packaging" / "linux" / "launcher" / "coderai-oci"
BUILD_SH = ROOT / "packaging" / "linux" / "build_oci_image.sh"
RPC_SH = ROOT / "packaging" / "build-rpc-server.sh"
SETTINGS = ROOT / "codai" / "admin" / "templates" / "settings.html"

STOCK_BIN = "/opt/coderai/llamasrv/bin/llama-server"
FORK_BIN = "/opt/coderai/llamasrv-fork/bin/llama-server"


def _cfgs():
    from codai.config import LlamaSrvConfig, LlamaSrvForkConfig
    return LlamaSrvConfig(), LlamaSrvForkConfig()


# --------------------------------------------------------------- the config --

def test_the_two_engines_have_the_same_knobs():
    """The fork inherits, so a field added for one is there for both. Divergent
    field sets would make the second engine quietly less configurable."""
    stock, fork = _cfgs()
    assert {f.name for f in dataclasses.fields(stock)} == \
           {f.name for f in dataclasses.fields(fork)}


def test_only_the_defaults_differ():
    stock, fork = _cfgs()
    assert stock.tree == "bindings" and fork.tree == "repo"
    assert fork.repo_url and not stock.repo_url
    assert fork.git_ref and not stock.git_ref
    # Not "this host's cards": a tuned tree built for untested hardware produces
    # plausible output rather than an error.
    assert fork.cuda_architectures == "86" and stock.cuda_architectures == ""


def test_the_engine_key_never_reaches_config_json():
    """It identifies the engine; an operator editing it would repoint an engine
    at the other one's state."""
    stock, fork = _cfgs()
    assert stock.ENGINE_KEY == "llamasrv" and fork.ENGINE_KEY == "llamasrv_fork"
    assert "ENGINE_KEY" not in {f.name for f in dataclasses.fields(fork)}


def test_both_blocks_round_trip_every_field(tmp_path):
    """The save path is field-driven for both: a hand-written key list is how a
    setting loads, is used, and is dropped on the next save."""
    import json
    from codai.config import ConfigManager
    cm = ConfigManager(config_dir=str(tmp_path))
    cm.load()
    cm.config.llamasrv.enabled = True
    cm.config.llamasrv_fork.enabled = True
    cm.config.llamasrv_fork.spec_type = "mtp"
    cm.config.llamasrv.parallel = 4
    cm.save_config()
    raw = json.loads((tmp_path / "config.json").read_text())
    for key in ("llamasrv", "llamasrv_fork"):
        assert key in raw
        assert set(raw[key]) == {f.name for f in dataclasses.fields(_cfgs()[0])}
    again = ConfigManager(config_dir=str(tmp_path))
    again.load()
    assert again.config.llamasrv_fork.spec_type == "mtp"
    assert again.config.llamasrv.parallel == 4
    assert again.config.llamasrv_fork.enabled and again.config.llamasrv.enabled


def test_one_can_be_enabled_without_the_other(tmp_path):
    import json
    from codai.config import ConfigManager
    cm = ConfigManager(config_dir=str(tmp_path))
    cm.load()
    cm.config.llamasrv.enabled = False
    cm.config.llamasrv_fork.enabled = True
    cm.save_config()
    raw = json.loads((tmp_path / "config.json").read_text())
    assert raw["llamasrv"]["enabled"] is False
    assert raw["llamasrv_fork"]["enabled"] is True


# -------------------------------------------------------------- the routing --

def test_both_engines_are_known_to_the_resolver():
    import codai.models.manager as mgr
    assert "llamasrv" in mgr._ENGINE_BACKENDS
    assert "llamasrv_fork" in mgr._ENGINE_BACKENDS
    # every engine needs a config getter and a claims rule, or resolution
    # raises or silently skips it
    assert set(mgr._ENGINE_NAME_CLAIMS) == set(mgr._ENGINE_BACKENDS)


def test_neither_engine_ever_claims_a_gguf_by_itself():
    """Both can serve every GGUF. If either inferred, enabling it would move the
    whole local catalogue off the in-process default path."""
    import codai.models.manager as mgr
    for name in ("some-model.gguf", "qwen3.8-27b", "anything"):
        assert mgr._ENGINE_NAME_CLAIMS["llamasrv"](name) is False
        assert mgr._ENGINE_NAME_CLAIMS["llamasrv_fork"](name) is False


@pytest.mark.parametrize("pin", ["llamasrv", "llamasrv_fork"])
def test_a_pin_routes_to_the_engine_it_names(pin):
    from codai.frontproxy.router import required_capability
    cap = required_capability("m", path="/v1/chat/completions", backend=pin)
    assert cap == pin


def test_each_engines_model_id_alias_routes_to_it():
    from codai.frontproxy.router import required_capability
    assert required_capability("stocky", path="/v1/chat/completions",
                               llamasrv_model_id="stocky",
                               llamasrv_enabled=True) == "llamasrv"
    assert required_capability("forky", path="/v1/chat/completions",
                               llamasrv_fork_model_id="forky",
                               llamasrv_fork_enabled=True) == "llamasrv_fork"


def test_a_disabled_engines_alias_does_not_route():
    from codai.frontproxy.router import required_capability
    assert required_capability("forky", path="/v1/chat/completions",
                               llamasrv_fork_model_id="forky",
                               llamasrv_fork_enabled=False) != "llamasrv_fork"


def test_the_front_resolves_both_engines_capabilities():
    """`app.py` passed neither, so a model routed by a configured model_id (not
    by a backend pin) resolved to no capability and was left unassigned."""
    src = (ROOT / "codai" / "frontproxy" / "app.py").read_text()
    assert "llamasrv_model_id=" in src and "llamasrv_fork_model_id=" in src


def test_the_assignment_gets_both_configs():
    src = (ROOT / "codai" / "frontproxy" / "assignment.py").read_text()
    assert "llamasrv_fork_cfg" in src
    sup = (ROOT / "codai" / "frontproxy" / "engine_supervisor.py").read_text()
    assert sup.count("llamasrv_fork") >= 2, "both assignment call sites"


def test_an_engine_node_exists_for_each():
    sup = (ROOT / "codai" / "frontproxy" / "engine_supervisor.py").read_text()
    assert '("llamasrv", "llamasrv")' in sup and \
           '("llamasrv_fork", "llamasrv_fork")' in sup


# ------------------------------------------------------- per-engine state --

def test_the_same_gguf_on_both_engines_is_two_servers():
    """THE bug this split could have shipped with: keyed by file path alone, the
    fork's request would be handed the stock engine's running server, and the
    comparison would silently be one engine against itself."""
    import codai.api.llamasrv_worker as w
    stock, fork = _cfgs()
    _, a = w.resolve_service_key(stock, "/AI/model.gguf")
    _, b = w.resolve_service_key(fork, "/AI/model.gguf")
    assert a != b, "stock and fork share a service key"
    assert a.startswith("llamasrv::") and b.startswith("llamasrv_fork::")


def test_each_engine_builds_in_its_own_directory():
    """Per tree, so bumping a ref or switching engines builds beside the old one
    instead of over it: a half-overwritten build directory looks like a broken
    fork rather than a dirty tree."""
    import codai.api.llamasrv_worker as w
    stock, fork = _cfgs()
    assert w._install_dir(stock) != w._install_dir(fork)
    assert w._install_dir(fork).name == "llamAmpere-v0.6.4"


def test_each_engine_has_its_own_prebuilt_binary_variable():
    import codai.api.llamasrv_worker as w
    stock, fork = _cfgs()
    assert w.bin_env_var(stock) == "CODERAI_LLAMASRV_BIN"
    assert w.bin_env_var(fork) == "CODERAI_LLAMASRV_FORK_BIN"


def test_the_fork_ignores_the_stock_binary_variable(tmp_path, monkeypatch):
    """One variable for both would serve whichever binary was baked last to every
    pinned model, defeating the whole split."""
    import codai.api.llamasrv_worker as w
    stock, fork = _cfgs()
    stray = tmp_path / "stock-llama-server"
    stray.write_text("#!/bin/sh\n")
    monkeypatch.setenv("CODERAI_LLAMASRV_BIN", str(stray))
    monkeypatch.delenv("CODERAI_LLAMASRV_FORK_BIN", raising=False)
    assert w._server_bin(w._install_dir(stock), stock) == stray
    assert w._server_bin(w._install_dir(fork), fork) != stray


def test_a_models_override_block_belongs_to_one_engine():
    """A `llamasrv_fork` block must not rewrite the stock engine's launch."""
    src = (ROOT / "codai" / "backends" / "llamasrv.py").read_text()
    assert 'model_block(model_name, _key)' in src
    assert 'model_block(model_name, "llamasrv")' not in src


# ------------------------------------------------------------- the image --

def test_the_image_bakes_a_binary_for_each_engine():
    """The runtime image has no nvcc and no cmake, so `auto_build` cannot run
    there: a binary that is not baked is an engine with nothing to launch."""
    df = DOCKERFILE.read_text()
    assert "--llama-server /opt/coderai/llamasrv/bin" in df, "stock not baked"
    assert "/opt/coderai/llamasrv-fork/bin/llama-server" in df, "fork not baked"


def test_the_update_overlay_bakes_both_too():
    """An install that only ever gets this overlay (not a from-scratch build)
    must still end up with both engines usable."""
    up = UPDATE_LLAMA.read_text()
    assert "--llama-server /wheels/llamasrv/bin" in up
    assert "/wheels/llamasrv-fork/bin/llama-server" in up
    assert STOCK_BIN in up and FORK_BIN in up, "not installed into the image"


def test_the_rpc_script_turns_the_server_target_on_only_when_asked():
    """It configures with LLAMA_BUILD_SERVER=OFF for its own purposes; with
    --llama-server the target would not exist and cmake would fail with
    'No rule to make target'."""
    sh = RPC_SH.read_text()
    assert "SERVER_FLAG=-DLLAMA_BUILD_SERVER=OFF" in sh
    assert '[ -n "$LLAMA_SERVER_DEST" ] && SERVER_FLAG=-DLLAMA_BUILD_SERVER=ON' in sh


def test_the_launcher_points_each_engine_at_its_own_binary():
    text = LAUNCHER.read_text()
    assert f"export CODERAI_LLAMASRV_BIN={STOCK_BIN}" in text
    assert f"export CODERAI_LLAMASRV_FORK_BIN={FORK_BIN}" in text


@pytest.mark.parametrize("var,path", [("CODERAI_LLAMASRV_BIN", STOCK_BIN),
                                      ("CODERAI_LLAMASRV_FORK_BIN", FORK_BIN)])
def test_each_export_is_guarded_and_overridable(var, path):
    """Guarded, because the fork build is non-fatal and pointing the variable at
    a missing file turns 'unavailable' into a hard error. Overridable, because a
    pod or an operator may name a binary elsewhere."""
    text = LAUNCHER.read_text()
    assert f"-x {path}" in text
    assert re.search(r'-z\s+"\$\{' + var + r':-\}"', text), f"{var} is overwritten"


def test_the_launcher_trusts_no_binary_off_the_path():
    """A `llama-server` on PATH may be any llama.cpp, and serving a tree other
    than the pinned one reads as the fork working badly."""
    text = LAUNCHER.read_text()
    for probe in ("command -v llama-server", "which llama-server",
                  "/usr/local/bin/llama-server"):
        assert probe not in text


def test_the_forks_defaults_match_what_the_image_builds():
    """Nothing at runtime can tell that a prebuilt binary is not the tree the
    config names, so the two must be pinned to each other by a test."""
    _, fork = _cfgs()
    for df in (DOCKERFILE, UPDATE_LLAMA):
        text = df.read_text()
        for arg, want in (("LLAMASRV_FORK_REPO", fork.repo_url),
                          ("LLAMASRV_FORK_REF", fork.git_ref),
                          ("LLAMASRV_FORK_ARCHS", fork.cuda_architectures)):
            m = re.search(rf"^ARG {arg}=(\S+)", text, re.M)
            assert m, f"{df.name} declares no {arg}"
            assert m.group(1) == want, (
                f"{df.name} {arg}={m.group(1)!r} but the config default is {want!r}")


def test_the_forks_ref_is_not_a_moving_branch():
    """A fork's `main` moves under you, and a build that follows it turns a
    working install into a broken one with no config change to point at."""
    _, fork = _cfgs()
    assert fork.git_ref not in ("main", "master", "HEAD", "")
    assert (re.match(r"^v?\d+(\.\d+)+$", fork.git_ref)
            or re.match(r"^[0-9a-f]{40}$", fork.git_ref)), (
        f"{fork.git_ref!r} is neither a version tag nor a full SHA")


def test_neither_image_build_can_fail_the_whole_image():
    for df in (DOCKERFILE, UPDATE_LLAMA):
        text = df.read_text()
        i = text.index("BUILD_LLAMASRV_FORK}")
        assert "WARNING: the fork llama-server was not built" in text[i:i + 3000]


def test_the_build_script_passes_every_fork_arg():
    declared = set(re.findall(r"^ARG (LLAMASRV_FORK_\w+|BUILD_LLAMASRV_FORK)=",
                              DOCKERFILE.read_text(), re.M))
    assert declared
    sh = BUILD_SH.read_text()
    assert not [a for a in declared if f'--build-arg {a}=' not in sh]
    assert not [a for a in declared if f'{a}="${{{a}:-' not in sh]


def test_each_baked_binary_carries_a_marker():
    for text in (DOCKERFILE.read_text(), UPDATE_LLAMA.read_text()):
        assert "BUILD_INFO" in text
        for field in ("repo=", "ref=", "cuda_architectures="):
            assert field in text


# ---------------------------------------------------------------- the UI --

def test_the_settings_page_has_a_card_for_each_engine():
    html = SETTINGS.read_text()
    assert "s-llamasrv-enabled" in html and "s-llamasrv-fork-enabled" in html
    assert "llama-server from a fork (second engine)" in html


def test_the_form_is_prefix_driven_for_both():
    """Two hand-written copies drift, and a field read for one engine and
    forgotten for the other looks exactly like a setting that did not save."""
    html = SETTINGS.read_text()
    assert "['llamasrv', 's-llamasrv-']" in html
    assert "['llamasrv_fork', 's-llamasrv-fork-']" in html


def test_the_admin_api_reads_and_writes_both():
    src = (ROOT / "codai" / "admin" / "routes.py").read_text()
    assert '"llamasrv": _engine_block(c.llamasrv)' in src
    assert '"llamasrv_fork": _engine_block(c.llamasrv_fork)' in src
    assert '("llamasrv", c.llamasrv), ("llamasrv_fork", c.llamasrv_fork)' in src


def test_a_bad_tree_value_does_not_turn_the_fork_into_the_stock_engine():
    src = (ROOT / "codai" / "admin" / "routes.py").read_text()
    assert "v.tree = tree if tree in (\"bindings\", \"repo\") else v.tree" in src
