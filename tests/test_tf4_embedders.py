"""jina-clip-v2 answered every image with 1024 NaNs, and said nothing about it.

The model's remote code is pinned by its own config to revision 39e6a55 (July
2025) and jinaai's jina-clip-implementation has had no commit since, so it is
not transformers-5 code and will not become it. On the server's transformers
5.12 it needs two shims merely to import -- `clip_loss`, deleted from
modeling_clip, and `dtype=` for `torch_dtype=` -- and with both applied it loads
and returns non-finite image vectors at fp32, bf16 and fp16, with xattn on and
off, inside coderai and outside it.

Measured the same day, same weights, same images, on transformers 4.49:

    text : shape=(2, 1024) finite=True nan=0 norm=0.9992
    image: shape=(1, 1024) finite=True nan=0 norm=1.0007
    cos(text, image): 0.258 / 0.260

on CUDA and on CPU, and on python 3.11 and 3.13 alike. So the fix is a venv that
pins transformers and nothing else, built with the server's own interpreter --
no second python to find -- and the model runs in it as a subprocess.

What is asserted here: the venv is described and located without building
anything, the probe refuses a venv that has drifted to transformers 5, the
backend drives the subprocess (and can replace a dead one), and the model is
never loaded in-process again.
"""

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EMB = (ROOT / "codai/api/embeddings.py").read_text(encoding="utf-8")
SERVER = (ROOT / "codai/api/tf4_embed_server.py").read_text(encoding="utf-8")
SERVER_SRC = SERVER


@pytest.fixture
def w():
    from codai.api import tf4_embedders
    return tf4_embedders


# ── where it lives ───────────────────────────────────────────────────────────

def test_an_explicit_venv_path_wins(w, tmp_path, monkeypatch):
    monkeypatch.setenv("CODERAI_TF4_EMBEDDERS_VENV", str(tmp_path / "v"))
    assert w.resolve_venv_dir("4.49.0") == tmp_path / "v"


def test_otherwise_it_sits_under_a_cache_root(w, tmp_path, monkeypatch):
    """Rebuildable derived state, several GB of it -- not the config root.

    Which cache root is a separate question (the roomiest, see below), so this
    pins the only one on offer.
    """
    monkeypatch.delenv("CODERAI_TF4_EMBEDDERS_VENV", raising=False)
    monkeypatch.delenv("CODERAI_JINACLIP_VENV", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("CODERAI_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(w, "_BAKED_VENV", tmp_path / "nothing-here")
    monkeypatch.setattr(w, "_pointer_file", lambda pin: tmp_path / "none")
    import codai.platform_paths as pp
    monkeypatch.setattr(pp, "legacy_style_cache_root", lambda: tmp_path / "xdg")
    assert w.resolve_venv_dir("4.49.0") == tmp_path / "embedders-tf-4.49.0-venv"


def test_a_baked_venv_is_preferred_over_building_one(w, tmp_path, monkeypatch):
    monkeypatch.delenv("CODERAI_TF4_EMBEDDERS_VENV", raising=False)
    monkeypatch.delenv("CODERAI_JINACLIP_VENV", raising=False)
    baked = tmp_path / "baked"
    (baked / "bin").mkdir(parents=True)
    monkeypatch.setattr(w, "_BAKED_VENV", baked)
    assert w.resolve_venv_dir("4.49.0") == baked


def test_status_needs_neither_network_nor_build(w, monkeypatch, tmp_path):
    monkeypatch.setenv("CODERAI_TF4_EMBEDDERS_VENV", str(tmp_path / "absent"))
    st = [v for v in w.state() if v["pin"] == "4.49.0"][0]
    assert st["name"] == "transformers 4.49.0 (embedders)"
    assert st["installed"] is False and st["available"] is False
    assert "transformers" in st["why"]
    assert not (tmp_path / "absent").exists(), "state() built something"


# ── the one thing the venv is for ────────────────────────────────────────────

def test_the_requirements_pin_nothing_hard_and_not_transformers():
    """transformers is the one line that differs between these venvs, so it
    comes from the registry on the pip command line and one file serves every
    pin. Nothing else is implicated, so nothing else is pinned."""
    req = (ROOT / "requirements-embedders-tf4.txt").read_text(encoding="utf-8")
    body = [l.strip() for l in req.splitlines()
            if l.strip() and not l.strip().startswith("#")]
    assert not [l for l in body if l.startswith("transformers")], body
    assert not [l for l in body if "==" in l], f"no hard pins here: {body}"
    # The vision tower IS a timm model, so this is not optional.
    assert any(l.startswith("timm") for l in body)


def test_a_venv_that_drifted_to_transformers_5_is_not_usable(w, monkeypatch):
    """--system-site-packages or a later `pip install -U` would take it to 5,
    and it would serve NaNs again with every file in place."""
    calls = []

    def _ok(py, modules):
        return True, ""

    def _run(cmd, **kw):
        calls.append(cmd)
        # The version gate is the second call; report "not 4".
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(w.venv_env, "import_ok", _ok)
    monkeypatch.setattr(w.venv_env, "run", _run)
    assert w._transformers_ok(Path("/nope/python"), "4.49.0") is False
    assert any("transformers" in str(c) for c in calls)


def test_a_venv_on_transformers_4_is_usable(w, monkeypatch):
    monkeypatch.setattr(w.venv_env, "import_ok", lambda py, m: (True, ""))
    monkeypatch.setattr(w.venv_env, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0))
    assert w._transformers_ok(Path("/nope/python"), "4.49.0") is True


def test_an_unimportable_venv_is_not_usable(w, monkeypatch):
    monkeypatch.setattr(w.venv_env, "import_ok",
                        lambda py, m: (False, "No module named 'timm'"))
    monkeypatch.setattr(w.venv_env, "run",
                        lambda cmd, **kw: pytest.fail("version-checked a venv "
                                                      "that cannot import"))
    assert w._transformers_ok(Path("/nope/python"), "4.49.0") is False


# ── the server protocol ──────────────────────────────────────────────────────

def test_the_server_announces_its_width_rather_than_assuming_it():
    assert '"ready": True' in SERVER or "'ready': True" in SERVER
    assert "encode_text" in SERVER and "shape[-1]" in SERVER


def test_the_server_names_no_dtype_at_all():
    """Both names fail, on opposite stacks, because the remote code forwards
    **kwargs into its own __init__:

        transformers 5.12 + torch_dtype= -> attribute name must be string
        transformers 4.49 + dtype=       -> unexpected keyword argument 'dtype'

    and the checkpoint's config.json already names bfloat16, which is what the
    verified run loaded in and what its vision tower needs (fp16 overflows and
    returns NaNs)."""
    code = [l for l in SERVER.splitlines()
            if l.strip() and not l.strip().startswith("#")]
    # np.asarray(..., dtype=) is not what this is about; from_pretrained is.
    passed = [l for l in code
              if "dtype=" in l and "from_pretrained" in l]
    assert not passed, f"a dtype kwarg fails on one of the two stacks: {passed}"
    assert 'kw = {"trust_remote_code": True}' in SERVER
    load = [l for l in code if "from_pretrained(model_id, **kw)" in l]
    assert load and all("dtype" not in l for l in load), load


def test_the_server_refuses_non_finite_vectors_itself():
    assert "isfinite" in SERVER


def test_the_server_falls_back_to_cpu_rather_than_dying():
    assert "TF4_FORCE_CPU" in SERVER
    assert 'device = "cpu"' in SERVER


def test_spawn_waits_for_ready_and_reports_a_load_failure(w, tmp_path, monkeypatch):
    """A handle that is still loading would turn the first embedding request
    into the load error."""
    monkeypatch.setattr(w, "ensure_installed",
                        lambda pin="4.49.0", log=None: Path(sys.executable))
    script = tmp_path / "fake_server.py"
    script.write_text("import sys\nsys.exit(3)\n")
    monkeypatch.setattr(w, "_SERVER_SCRIPT", script)
    with pytest.raises(RuntimeError) as e:
        w.spawn("jinaai/jina-clip-v2", load_timeout=30)
    assert "exited while loading" in str(e.value)


def test_spawn_returns_once_the_child_is_ready(w, tmp_path, monkeypatch):
    monkeypatch.setattr(w, "ensure_installed",
                        lambda pin="4.49.0", log=None: Path(sys.executable))
    script = tmp_path / "fake_server.py"
    script.write_text(
        "import json,sys\n"
        "print(json.dumps({'ready': True, 'device': 'cpu', 'dim': 1024,\n"
        "                  'transformers': '4.49.0'}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line)\n"
        "    n = len(req.get('texts') or req.get('images') or [])\n"
        "    print(json.dumps({'embeddings': [[1.0, 0.0]] * n}), flush=True)\n")
    monkeypatch.setattr(w, "_SERVER_SCRIPT", script)
    proc = w.spawn("jinaai/jina-clip-v2", device="cpu", load_timeout=60)
    try:
        assert w.request(proc, {"texts": ["a", "b"]}) == [[1.0, 0.0], [1.0, 0.0]]
    finally:
        proc.kill()


def test_an_error_reply_is_raised_not_returned(w, tmp_path, monkeypatch):
    monkeypatch.setattr(w, "ensure_installed",
                        lambda pin="4.49.0", log=None: Path(sys.executable))
    script = tmp_path / "fake_server.py"
    script.write_text(
        "import json,sys\n"
        "print(json.dumps({'ready': True, 'dim': 4}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    print(json.dumps({'error': 'the model returned non-finite values'}),\n"
        "          flush=True)\n")
    monkeypatch.setattr(w, "_SERVER_SCRIPT", script)
    proc = w.spawn("nvidia/MM-Embed", device="cpu", load_timeout=60)
    try:
        with pytest.raises(ValueError) as e:
            w.request(proc, {"images": ["/tmp/x.png"]})
        assert "non-finite" in str(e.value)
    finally:
        proc.kill()


# ── how the embeddings backend uses it ───────────────────────────────────────

def test_the_model_is_no_longer_loaded_in_this_process():
    """The in-process path is kept only as far as the error message: there is no
    stack this process can give the model that yields a usable vector."""
    i = EMB.index("def _load_tf4_embedder(")
    body = EMB[i:EMB.index("def _tf4_proc(")]
    assert "AutoModel.from_pretrained" not in body
    assert "tf4_embedders" in body


def test_its_vram_is_reclaimable_by_eviction():
    """A subprocess holding part of the card must be killed by cleanup(), or the
    manager's eviction frees nothing and the ledger is wrong."""
    i = EMB.index("def _cleanup_locked(")
    assert "'jinaclip'" in EMB[i:i + 400]


def test_calls_into_it_are_serialised():
    """One stdin/stdout pipe: two threads interleaving requests would read each
    other's answers."""
    i = EMB.index("if backend in ('llama', 'llama-vl'")
    assert "'jinaclip'" in EMB[i:i + 200]


def test_a_dead_server_is_replaced_in_place():
    i = EMB.index("def _tf4_proc(")
    body = EMB[i:EMB.index("def _tf4_embed_texts(")]
    assert "respawn" in body and "poll()" in body


def test_both_embed_paths_get_the_handle_not_the_process():
    """They need its lock and its respawn hook, which only the wrapper has."""
    for call in ("_tf4_embed_texts(model_obj, texts, dimensions)",
                 "_tf4_embed_images(model_obj, pil_images, dimensions)"):
        assert call in EMB, call


def test_images_reach_the_child_as_files_and_are_cleaned_up():
    i = EMB.index("def _tf4_embed_images(")
    body = EMB[i:i + 1400]
    assert "NamedTemporaryFile" in body
    assert "unlink" in body and "finally" in body


def test_a_model_pinned_off_the_gpu_stays_off_it():
    i = EMB.index("def _load_tf4_embedder(")
    body = EMB[i:EMB.index("def _tf4_proc(")]
    assert "n_gpu_layers" in body and "'cpu'" in body


# ── the file the container cannot see ────────────────────────────────────────

def test_the_fallback_pins_mirror_the_requirements_file(w):
    """Two copies of a pin drift, and the second one exists because the first is
    sometimes not there: the container mounts only codai/, tools/ and
    third_party/ over the image, so a repo-ROOT requirements file added after
    the running image was built is absent until the container is recreated.
    Measured here as a venv that built perfectly and then refused to install."""
    req = (ROOT / "requirements-embedders-tf4.txt").read_text(encoding="utf-8")
    in_file = [l.strip() for l in req.splitlines()
               if l.strip() and not l.strip().startswith("#")]
    assert sorted(in_file) == sorted(w._FALLBACK_PINS), (
        "requirements-embedders-tf4.txt and _FALLBACK_PINS disagree:\n"
        f"  file: {sorted(in_file)}\n  code: {sorted(w._FALLBACK_PINS)}")


def test_a_missing_requirements_file_installs_the_pins_instead(w, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(w, "_REQUIREMENTS", tmp_path / "not-here.txt")
    # Into tmp_path, NOT the real config root: this test wrote a pointer file
    # into ~/.coderai naming a pytest directory, which a later process would
    # then have resolved the venv from.
    monkeypatch.setattr(w, "_pointer_file", lambda pin: tmp_path / "ptr")
    monkeypatch.setattr(w, "resolve_venv_dir", lambda pin="4.49.0": tmp_path / "venv")
    monkeypatch.setattr(w.venv_env, "create_venv",
                        lambda d, **k: Path(sys.executable))
    monkeypatch.setattr(w, "_transformers_ok",
                        lambda py, pin: bool(seen))      # false, then true
    monkeypatch.setattr(w.venv_env, "run",
                        lambda cmd, **kw: (seen.append(cmd)
                                           or subprocess.CompletedProcess(cmd, 0)))
    monkeypatch.setattr(w, "_bootstrapped", {})
    w.ensure_installed()
    install = [c for c in seen if "install" in c and "-U" not in c][0]
    assert "-r" not in install, install
    assert "transformers==4.49.0" in install, install


# ── the library path of a venv that brings its own CUDA ──────────────────────

def test_the_child_does_not_inherit_the_servers_library_path(w, monkeypatch, tmp_path):
    """This venv is built by the server's interpreter -- so venv_env would keep
    LD_LIBRARY_PATH, which is right for a venv that shares our native libraries
    and wrong for this one: it ships its own torch and its own CUDA/cuDNN
    wheels. Measured live: the text tower answered on the GPU and every image
    failed with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED, because only the vision
    tower runs convolutions."""
    seen = {}

    def _env_for_venv(venv, extra=None, hf=True, own_libs=False):
        seen["own_libs"] = own_libs
        return {"PATH": "/usr/bin"}

    monkeypatch.setattr(w.venv_env, "env_for_venv", _env_for_venv)
    monkeypatch.setattr(w, "ensure_installed",
                        lambda pin="4.49.0", log=None: Path(sys.executable))
    script = tmp_path / "fake_server.py"
    script.write_text("import json\nprint(json.dumps({'ready': True, 'dim': 4}),"
                      " flush=True)\nimport sys\nsys.stdin.read()\n")
    monkeypatch.setattr(w, "_SERVER_SCRIPT", script)
    proc = w.spawn("nvidia/MM-Embed", device="cpu", load_timeout=60)
    try:
        assert seen.get("own_libs") is True
    finally:
        proc.kill()


def test_own_libs_strips_our_lib_dirs_even_for_our_own_interpreter(monkeypatch, tmp_path):
    from codai.api import venv_env
    venv = tmp_path / "v"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {os.path.dirname(sys.executable)}\n")
    # A directory that is part of the server's install whatever python is
    # running the test -- sys.prefix is /usr here and is deliberately treated as
    # the whole system, not as ours.
    ours = "/opt/coderai/local-libs"
    monkeypatch.setenv("LD_LIBRARY_PATH", ours + ":/opt/vendor/cuda")
    kept = venv_env.env_for_venv(venv, own_libs=True).get("LD_LIBRARY_PATH", "")
    assert ours not in kept.split(":"), kept
    # An entry outside our install is the operator's and is left alone.
    assert "/opt/vendor/cuda" in kept.split(":"), kept


# ── two models, one venv ─────────────────────────────────────────────────────

def test_both_models_are_registered(w):
    """MM-Embed joined jina-clip-v2 here rather than collecting a fourth shim:
    its remote code registers LlavaNextConfig process-wide (which broke e5-v)
    and then asks for names transformers 5 deleted."""
    assert w.model_for("jinaai/jina-clip-v2").family == "jinaclip"
    assert w.model_for("nvidia/MM-Embed").family == "mmembed"


def test_a_model_is_matched_however_it_was_configured(w):
    """'MM-Embed', 'nvidia/MM-Embed' and 'mm-embed' are one model."""
    for name in ("nvidia/MM-Embed", "MM-Embed", "mm-embed", "NVIDIA/mm-embed"):
        assert w.model_for(name) is not None, name
    for name in ("jina-clip-v2", "jinaai/jina-clip-v2", "JINA-CLIP-V2"):
        assert w.model_for(name) is not None, name


def test_a_model_that_is_not_ours_is_not_claimed(w):
    for name in ("royokong/e5-v", "BAAI/BGE-VL-MLLM-S2", "", None):
        assert w.model_for(name) is None, name


def test_spawning_an_unregistered_model_says_so(w, monkeypatch):
    monkeypatch.setattr(w, "ensure_installed",
                        lambda pin="4.49.0", log=None: Path(sys.executable))
    with pytest.raises(ValueError) as e:
        w.spawn("royokong/e5-v")
    assert "not served out of a pinned-transformers venv" in str(e.value)


def test_the_state_lists_the_models_that_depend_on_each_pin(w, monkeypatch, tmp_path):
    monkeypatch.setenv("CODERAI_TF4_EMBEDDERS_VENV", str(tmp_path / "absent"))
    by_pin = {v["pin"]: {m["repo"] for m in v["models"]} for v in w.state()}
    assert by_pin["4.49.0"] == {"jinaai/jina-clip-v2"}
    assert by_pin["4.42.4"] == {"nvidia/MM-Embed"}
    assert all(m["why"] for v in w.state() for m in v["models"])


def test_a_venv_built_under_the_old_name_is_adopted(w, monkeypatch, tmp_path):
    """It was called jinaclip-venv while this served one model. Several GB
    already on disk are used, not built again under the new name."""
    monkeypatch.delenv("CODERAI_TF4_EMBEDDERS_VENV", raising=False)
    monkeypatch.delenv("CODERAI_JINACLIP_VENV", raising=False)
    monkeypatch.setenv("CODERAI_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(w, "_BAKED_VENV", tmp_path / "nothing-here")
    legacy = tmp_path / "jinaclip-venv" / "bin"
    legacy.mkdir(parents=True)
    (legacy / "python").write_text("#!/bin/sh\n")
    assert w.resolve_venv_dir("4.49.0") == tmp_path / "jinaclip-venv"


# ── the quantization the parent decides ──────────────────────────────────────

def test_the_parent_hands_the_child_its_quantization(w, monkeypatch, tmp_path):
    seen = {}

    def _popen(cmd, **kw):
        seen.update(kw.get("env") or {})
        raise RuntimeError("stop here")

    monkeypatch.setattr(w, "ensure_installed",
                        lambda pin="4.49.0", log=None: Path(sys.executable))
    monkeypatch.setattr(w.subprocess, "Popen", _popen)
    with pytest.raises(RuntimeError):
        w.spawn("nvidia/MM-Embed", device="cuda",
                quantization={"load_in_4bit": True, "skip_modules": ["visual"],
                              "compute_dtype": "bf16"})
    assert seen.get("TF4_FAMILY") == "mmembed"
    assert seen.get("TF4_MODEL") == "nvidia/MM-Embed"
    spec = json.loads(seen["TF4_QUANTIZATION"])
    assert spec["load_in_4bit"] is True and spec["skip_modules"] == ["visual"]


def test_the_policy_comes_from_models_json():
    """The child cannot import coderai to read it, so the parent must — and the
    skip list is the part that matters for a vision-language embedder."""
    emb = (ROOT / "codai/api/embeddings.py").read_text(encoding="utf-8")
    i = emb.index("def _tf4_quantization(")
    body = emb[i:emb.index("def _load_tf4_embedder(")]
    assert "quant_skip_modules" in body
    assert "load_in_4bit" in body and "load_in_8bit" in body


def test_the_child_leaves_the_named_modules_at_full_precision():
    assert "llm_int8_skip_modules" in SERVER
    assert "skip_modules" in SERVER


def test_the_child_does_not_quantize_on_the_cpu():
    """bitsandbytes cannot, and a model that silently fell back to fp16 there
    would be four times the size the planner reserved for it."""
    i = SERVER.index("quant = _quantization_config(")
    assert "if want_gpu else None" in SERVER[i:i + 160]


def test_mm_embeds_remote_code_is_not_imported_in_this_process():
    """The structural fix that replaced a shim.

    MM-Embed's module registers LlavaNextConfig against its own class at import
    time, process-wide, so e5-v — a LLaVA-NeXT checkpoint — started answering
    "'LlavaNextConfig' object has no attribute 'retriever'" after MM-Embed had
    been touched. coderai shimmed that by recording and undoing the mapping;
    moving the model out of the process removes the possibility instead, so the
    shim is gone and this is what keeps it gone.
    """
    emb = (ROOT / "codai/api/embeddings.py").read_text(encoding="utf-8")
    assert "_allow_remote_automodel_reregister" not in emb, \
        "the shim is back — which means something imports MM-Embed here again"
    assert "def _load_mmembed" not in emb
    i = emb.index("if _is_mmembed(model_name):")
    assert "_load_tf4_embedder" in emb[i:i + 220]


def test_the_probe_covers_every_requirement_the_models_need(w):
    """The probe decides whether an EXISTING venv is usable, so a dependency
    the requirements gained reaches an installed venv only if its absence is
    noticed here. Measured: bitsandbytes/accelerate/peft were added for
    MM-Embed, the probe still passed on the old venv, ensure_installed returned
    early, and the model failed with "requires ... peft"."""
    pins = {p.split(">")[0].split("=")[0].strip().lower()
            for p in w._FALLBACK_PINS}
    # torch/transformers are probed under their import names already.
    for pkg, module in (("timm", "timm"), ("einops", "einops")):
        assert pkg in pins, pkg
        assert module in w._PROBE_IMPORTS, module
    # …and the per-generation extras likewise, against their own pin's probe.
    for pin, extras in w.PIN_EXTRAS.items():
        for spec in extras:
            # A spec is either `pkg==x.y` or `pkg>=x.y`.
            name = spec.split("=")[0].rstrip("><!~")
            assert name in w.probe_imports(pin), (pin, name)
        assert name not in w._PROBE_IMPORTS, \
            f"{name} is pinned per generation; probing it for every pin would " \
            f"call a venv that must not have it broken"


# ── one venv per transformers pin ────────────────────────────────────────────

def test_the_two_models_do_not_share_a_transformers(w):
    """They cannot. MM-Embed's retriever runs its own decoder loop and calls
    each Mistral layer without `position_embeddings`, which transformers
    computed inside the attention layer until the rotary refactor of 4.43 — on
    4.49 it dies in transformers' own code:

        modeling_mistral.py:166, in forward
            cos, sin = position_embeddings
        TypeError: cannot unpack non-iterable NoneType object

    and NV-Embed-v1's own card says transformers==4.42.4."""
    assert w.MODELS["jinaai/jina-clip-v2"].transformers == "4.49.0"
    assert w.MODELS["nvidia/mm-embed"].transformers == "4.42.4"
    assert set(w.pins()) == {"4.49.0", "4.42.4"}


def test_each_pin_gets_its_own_venv(w, monkeypatch, tmp_path):
    monkeypatch.delenv("CODERAI_TF4_EMBEDDERS_VENV", raising=False)
    monkeypatch.delenv("CODERAI_JINACLIP_VENV", raising=False)
    monkeypatch.setenv("CODERAI_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(w, "_BAKED_VENV", tmp_path / "nothing-here")
    monkeypatch.setattr(w, "_pointer_file", lambda pin: tmp_path / f"{pin}.missing")
    dirs = {p: w.resolve_venv_dir(p) for p in w.pins()}
    assert len(set(dirs.values())) == len(dirs), dirs
    assert "4.42.4" in str(dirs["4.42.4"])


def test_the_old_unversioned_names_are_only_candidates_for_4_49(w, monkeypatch, tmp_path):
    """The venv was called jinaclip-venv while one model used it, on 4.49. A
    4.42 install must not adopt it — it holds the wrong transformers."""
    monkeypatch.setenv("CODERAI_CACHE_DIR", str(tmp_path))
    names_449 = [d.name for d in w._candidate_dirs("4.49.0")]
    names_442 = [d.name for d in w._candidate_dirs("4.42.4")]
    assert "jinaclip-venv" in names_449
    assert "jinaclip-venv" not in names_442
    assert "embedders-tf-4.42.4-venv" in names_442


def test_the_pin_is_checked_exactly_not_as_a_major(w, monkeypatch):
    """A venv carrying the other pin's transformers would fail inside
    transformers itself, with every file in place and nothing to see."""
    probes = []
    monkeypatch.setattr(w.venv_env, "import_ok", lambda py, m: (True, ""))
    monkeypatch.setattr(w.venv_env, "run",
                        lambda cmd, **kw: (probes.append(cmd[-1])
                                           or subprocess.CompletedProcess(cmd, 0)))
    w._transformers_ok(Path("/nope/python"), "4.42.4")
    assert "'4.42.4'" in probes[-1], probes
    assert "split" not in probes[-1], "a major-version check is not enough"


def test_the_install_passes_the_pin_on_the_command_line(w, monkeypatch, tmp_path):
    cmds = []
    monkeypatch.setattr(w, "resolve_venv_dir", lambda pin="x": tmp_path / "venv")
    monkeypatch.setattr(w.venv_env, "create_venv",
                        lambda d, **k: Path(sys.executable))
    # 4.42.4 asks for python 3.12, which this test is not about.
    monkeypatch.setattr(w, "_create_venv_on",
                        lambda version, venv, say: Path(sys.executable))
    monkeypatch.setattr(w, "_transformers_ok", lambda py, pin: bool(cmds))
    monkeypatch.setattr(w.venv_env, "run",
                        lambda cmd, **kw: (cmds.append(cmd)
                                           or subprocess.CompletedProcess(cmd, 0)))
    monkeypatch.setattr(w, "_bootstrapped", {})
    monkeypatch.setattr(w, "_pointer_file", lambda pin: tmp_path / "ptr")
    w.ensure_installed("4.42.4")
    install = [c for c in cmds if "install" in c and "-U" not in c][0]
    assert "transformers==4.42.4" in install, install


def test_spawn_builds_the_venv_the_model_asks_for(w, monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(w, "ensure_installed",
                        lambda pin="4.49.0", log=None: (asked.append(pin)
                                                        or Path(sys.executable)))
    monkeypatch.setattr(w, "resolve_venv_dir", lambda pin="4.49.0": tmp_path)
    script = tmp_path / "fake.py"
    script.write_text("import json,sys\nprint(json.dumps({'ready': True,"
                      " 'dim': 4}), flush=True)\nsys.stdin.read()\n")
    monkeypatch.setattr(w, "_SERVER_SCRIPT", script)
    proc = w.spawn("nvidia/MM-Embed", device="cpu", load_timeout=60)
    try:
        assert asked == ["4.42.4"], asked
    finally:
        proc.kill()


def test_a_pin_that_needs_another_python_says_which(w):
    """transformers 4.42.4 pins tokenizers<0.20, and tokenizers 0.19.1 ships no
    wheel past cp312 — so on the server's 3.13 pip builds it from source and
    PyO3 refuses. The venv has to be built on 3.12."""
    assert w.python_for("4.42.4") == "3.12"
    assert w.python_for("4.49.0") == ""


def test_an_interpreter_that_cannot_build_a_venv_is_not_the_end_of_it(w, monkeypatch, tmp_path):
    """The shipped image carries /usr/bin/python3.12 WITHOUT its -venv package,
    so the interpreter runs and `python3.12 -m venv` ends in "After installing
    the python3-venv package…". Measured on zeiss 2026-10-31."""
    import subprocess as sp
    tried = []

    def _run(cmd, **kw):
        head = list(cmd[1:3])
        if head == ["-m", "venv"]:
            tried.append(str(cmd[0]))
            return sp.CompletedProcess(cmd, 0, "", "")   # exits 0 either way
        if head == ["-m", "pip"]:
            # The real test: only the venv the SECOND interpreter made has pip.
            # The image's python3.12 exits 0 from `-m venv` and leaves none.
            last = tried[-1] if tried else ""
            return sp.CompletedProcess(cmd, 1 if "broken" in last else 0)
        return sp.CompletedProcess(cmd, 0, "3.12\n", "")  # version probes

    monkeypatch.setattr(w, "_interpreters",
                        lambda v: [Path("/usr/bin/broken-python"),
                                   Path(sys.executable)])
    monkeypatch.setattr(w.venv_env, "run", _run)
    made = tmp_path / "x"
    made.write_text("")
    monkeypatch.setattr(w.venv_env, "venv_python", lambda d: made)
    # No early return: this test is about building one, not adopting one.
    monkeypatch.setattr(w.venv_env, "interpreter_version", lambda py: None)
    monkeypatch.setattr(w.venv_env, "fetch_standalone_python",
                        lambda v, log=None: pytest.fail("fetched too early"))
    w._create_venv_on("3.12", tmp_path / "venv", lambda m: None)
    assert tried[0].endswith("broken-python") and len(tried) == 2, tried


def test_when_nothing_can_build_it_an_interpreter_is_fetched(w, monkeypatch, tmp_path):
    """Not optional: otherwise the model is uninstallable on an image whose
    distribution python ships without ensurepip — which is the image coderai
    ships."""
    import subprocess as sp
    fetched = []
    monkeypatch.setattr(w, "_interpreters", lambda v: [])
    monkeypatch.setattr(w.venv_env, "fetch_standalone_python",
                        lambda v, log=None: (fetched.append(v)
                                             or Path(sys.executable)))
    monkeypatch.setattr(w.venv_env, "run",
                        lambda cmd, **kw: sp.CompletedProcess(cmd, 0))
    made = tmp_path / "x"
    made.write_text("")
    monkeypatch.setattr(w.venv_env, "venv_python", lambda d: made)
    w._create_venv_on("3.12", tmp_path / "venv", lambda m: None)
    assert fetched == ["3.12"]


def test_an_interpreter_provisioned_for_another_venv_is_reused(w, monkeypatch, tmp_path):
    """One provisioned python serves every isolated venv: the bundle builds
    /opt/coderai/py310 for the wav2lip venv, and a lip-sync install provisions
    <config>/py311."""
    import codai.platform_paths as pp
    monkeypatch.setattr(pp, "legacy_style_config_dir", lambda: tmp_path)
    mine = f"{sys.version_info[0]}.{sys.version_info[1]}"
    root = tmp_path / ("py" + mine.replace(".", ""))
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "python3").symlink_to(sys.executable)
    monkeypatch.setattr(w.shutil, "which", lambda n: None)
    monkeypatch.setattr(w.venv_env.shutil, "which", lambda n: None)
    assert (root / "bin" / "python3") in w._interpreters(mine)


def test_a_venv_with_no_pip_is_not_a_venv(w, monkeypatch, tmp_path):
    """`python3.12 -m venv` on the shipped image exits 0 and leaves bin/python
    with no pip ("After installing the python3-venv package…"), and the next
    step failed with "No module named pip" after the exit code had already been
    believed. So the check is pip, not the exit code."""
    src = (ROOT / "codai/api/tf4_embedders.py").read_text(encoding="utf-8")
    i = src.index("def _create_venv_on(")
    body = src[i:src.index("\ndef ", i + 10)]
    assert '"-m", "pip", "--version"' in body
    assert "r.returncode == 0 and made.is_file() and _usable(made)" in body


def test_the_2024_companions_are_pinned_to_the_2024_transformers(w):
    """transformers 4.42.4 pins numpy<2, and the current releases cannot live
    with that: peft 0.21 imports EncoderDecoderCache (transformers >= 4.43) and
    datasets 5 pulls pyarrow 26, which "requires NumPy 2.0 or newer". Both were
    measured in a venv where everything else imported cleanly."""
    extras = w.PIN_EXTRAS["4.42.4"]
    assert any(e.startswith("peft==") for e in extras), extras
    assert any(e.startswith("datasets==") for e in extras), extras
    # And pyarrow, which is what actually refuses numpy 1.x: datasets 2.20
    # accepts pyarrow>=15, so pinning datasets alone left pyarrow 26 in place.
    assert any(e.startswith("pyarrow==") for e in extras), extras
    # The other pin has no business carrying either.
    assert "4.49.0" not in w.PIN_EXTRAS
    assert "peft" not in w.probe_imports("4.49.0")


def test_the_extras_reach_the_pip_command(w, monkeypatch, tmp_path):
    cmds = []
    monkeypatch.setattr(w, "resolve_venv_dir", lambda pin="x": tmp_path / "venv")
    monkeypatch.setattr(w, "_create_venv_on",
                        lambda version, venv, say: Path(sys.executable))
    monkeypatch.setattr(w.venv_env, "create_venv",
                        lambda d, **k: Path(sys.executable))
    monkeypatch.setattr(w, "_transformers_ok", lambda py, pin: bool(cmds))
    monkeypatch.setattr(w.venv_env, "run",
                        lambda cmd, **kw: (cmds.append(cmd)
                                           or subprocess.CompletedProcess(cmd, 0)))
    monkeypatch.setattr(w, "_bootstrapped", {})
    monkeypatch.setattr(w, "_pointer_file", lambda pin: tmp_path / "ptr")
    w.ensure_installed("4.42.4")
    install = [c for c in cmds if "install" in c and "-U" not in c][0]
    assert "peft==0.11.1" in install and "datasets==2.20.0" in install, install


# ── what the child says, and to whom ─────────────────────────────────────────

def test_the_childs_stderr_reaches_the_log(w, capsys):
    """A reason nobody reads is a reason nobody has. MM-Embed fell back from the
    GPU to the CPU on zeiss 2026-10-31 — "ready on cpu" in the log, 33 GB of
    host RAM, 1100% CPU — and said why on a stderr pipe that was only ever read
    if the child exited silently."""
    import io
    import types
    proc = types.SimpleNamespace(
        stderr=io.StringIO("would not load on the GPU (CUDA error)\nretrying\n"))
    w._pump_stderr(proc, "nvidia/MM-Embed")
    import time
    for _ in range(50):
        if "would not load" in capsys.readouterr().out:
            return
        time.sleep(0.02)
    pytest.fail("the child's stderr never reached the log")


def test_the_pipe_is_drained_for_the_childs_whole_life():
    """Not only on a silent exit: a pipe nobody drains fills at 64 KB and the
    child blocks writing to it, which looks like a model that hangs."""
    src = (ROOT / "codai/api/tf4_embedders.py").read_text(encoding="utf-8")
    i = src.index("proc = subprocess.Popen(")
    assert "_pump_stderr(proc, model_id)" in src[i:i + 1400]
    # …and the failure path no longer competes with the pump for the pipe.
    assert "proc.stderr.read()" not in src


def test_a_cpu_fallback_is_loud_and_says_why():
    """A model pinned to the CPU and a model that fell to it look identical in
    the log otherwise, and the difference is 30x and tens of GB."""
    src = (ROOT / "codai/api/tf4_embedders.py").read_text(encoding="utf-8")
    # The phrase is split across two source lines by the f-string.
    assert "IS ON THE CPU because the GPU " in src and "load failed" in src
    assert "gpu_error" in SERVER_SRC and "fell_back" in SERVER_SRC


def test_a_failed_gpu_load_gives_the_card_back():
    """It has already allocated by the time it fails, and those allocations live
    in the process's CUDA context for the rest of its life: measured at
    9756 MiB held by a model that was running on the CPU, invisible to the
    manager's ledger and unusable by anything else."""
    i = SERVER_SRC.index("fell_back = f\"{type(exc).__name__}")
    body = SERVER_SRC[i:i + 1200]
    assert "empty_cache()" in body
    assert body.index("empty_cache()") < body.index('device = "cpu"')


def test_no_test_here_writes_into_the_real_config_root(w):
    """One did: it ran ensure_installed() without patching _pointer_file and
    left ~/.coderai/embedders-tf-4.49.0-venv.path naming a pytest tmpdir, which
    a later process would have resolved the venv from."""
    src = (ROOT / "tests/test_tf4_embedders.py").read_text(encoding="utf-8")
    for block in src.split("\ndef test_")[1:]:
        if "ensure_installed(" in block and "_pointer_file" not in block:
            pytest.fail("a test calls ensure_installed() without redirecting "
                        "_pointer_file:\n  " + block.splitlines()[0])


def test_a_new_venv_goes_on_the_roomiest_root(w, monkeypatch, tmp_path):
    """Six GB has to go somewhere, and the cache roots are not equivalent. On
    zeiss the engines' CODERAI_CACHE_DIR is /AI/guffcache (197 GB free) while
    the default cache root had 15 GB — building there took that partition to
    11 MB free, and HF_HOME lives on it too, so every shard download failed
    with "No space left on device" and MM-Embed fell back to the CPU."""
    small = tmp_path / "small"
    big = tmp_path / "big"
    small.mkdir(); big.mkdir()
    monkeypatch.setenv("CODERAI_CACHE_DIR", str(small))
    monkeypatch.setenv("XDG_CACHE_HOME", str(big))
    free = {str(small): 15 * 2**30, str(big): 400 * 2**30}

    class _St:
        f_frsize = 1

        def __init__(self, n):
            self.f_bavail = n

    monkeypatch.setattr(w.os, "statvfs",
                        lambda d: _St(free.get(str(d), 2**30)))
    first = w._candidate_dirs("4.42.4")[0]
    assert str(first).startswith(str(big)), first


def test_an_existing_venv_still_wins_over_a_roomier_root(w, monkeypatch, tmp_path):
    """Otherwise a second copy gets built beside the one already on disk."""
    small = tmp_path / "small"
    (small / "embedders-tf-4.42.4-venv" / "bin").mkdir(parents=True)
    (small / "embedders-tf-4.42.4-venv" / "bin" / "python").write_text("")
    big = tmp_path / "big"
    big.mkdir()
    monkeypatch.setenv("CODERAI_CACHE_DIR", str(small))
    monkeypatch.setenv("XDG_CACHE_HOME", str(big))
    monkeypatch.setattr(w, "_pointer_file", lambda pin: tmp_path / "none")
    monkeypatch.setattr(w, "_BAKED_VENV", tmp_path / "nothing")
    assert w.resolve_venv_dir("4.42.4") == small / "embedders-tf-4.42.4-venv"
