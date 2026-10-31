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
SERVER = (ROOT / "codai/api/jina_clip_server.py").read_text(encoding="utf-8")


@pytest.fixture
def w():
    from codai.api import jinaclip_worker
    return jinaclip_worker


# ── where it lives ───────────────────────────────────────────────────────────

def test_an_explicit_venv_path_wins(w, tmp_path, monkeypatch):
    monkeypatch.setenv("CODERAI_JINACLIP_VENV", str(tmp_path / "v"))
    assert w.resolve_venv_dir() == tmp_path / "v"


def test_otherwise_it_sits_under_the_cache_root(w, tmp_path, monkeypatch):
    """Rebuildable derived state, several GB of it -- not the config root."""
    monkeypatch.delenv("CODERAI_JINACLIP_VENV", raising=False)
    monkeypatch.setenv("CODERAI_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(w, "_BAKED_VENV", tmp_path / "nothing-here")
    assert w.resolve_venv_dir() == tmp_path / "jinaclip-venv"


def test_a_baked_venv_is_preferred_over_building_one(w, tmp_path, monkeypatch):
    monkeypatch.delenv("CODERAI_JINACLIP_VENV", raising=False)
    baked = tmp_path / "baked"
    (baked / "bin").mkdir(parents=True)
    monkeypatch.setattr(w, "_BAKED_VENV", baked)
    assert w.resolve_venv_dir() == baked


def test_status_needs_neither_network_nor_build(w, monkeypatch, tmp_path):
    monkeypatch.setenv("CODERAI_JINACLIP_VENV", str(tmp_path / "absent"))
    st = w.state()
    assert st["name"] == "jina-clip-v2"
    assert st["installed"] is False and st["available"] is False
    assert "transformers" in st["why"]
    assert not (tmp_path / "absent").exists(), "state() built something"


# ── the one thing the venv is for ────────────────────────────────────────────

def test_the_requirements_pin_transformers_4_and_little_else():
    req = (ROOT / "requirements-jinaclip.txt").read_text(encoding="utf-8")
    assert "transformers==4.49.0" in req
    hard = [l for l in req.splitlines()
            if l.strip() and not l.startswith("#") and "==" in l]
    assert hard == ["transformers==4.49.0"], \
        f"only transformers is implicated; pinning more invites drift: {hard}"
    # The vision tower IS a timm model, so this is not optional.
    assert "timm" in req


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
    assert w._transformers_ok(Path("/nope/python")) is False
    assert any("transformers" in str(c) for c in calls)


def test_a_venv_on_transformers_4_is_usable(w, monkeypatch):
    monkeypatch.setattr(w.venv_env, "import_ok", lambda py, m: (True, ""))
    monkeypatch.setattr(w.venv_env, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0))
    assert w._transformers_ok(Path("/nope/python")) is True


def test_an_unimportable_venv_is_not_usable(w, monkeypatch):
    monkeypatch.setattr(w.venv_env, "import_ok",
                        lambda py, m: (False, "No module named 'timm'"))
    monkeypatch.setattr(w.venv_env, "run",
                        lambda cmd, **kw: pytest.fail("version-checked a venv "
                                                      "that cannot import"))
    assert w._transformers_ok(Path("/nope/python")) is False


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
    assert "from_pretrained(model_id, trust_remote_code=True)" in SERVER
    load = [l for l in code if "from_pretrained(" in l]
    assert len(load) == 1 and "dtype" not in load[0], load


def test_the_server_refuses_non_finite_vectors_itself():
    assert "isfinite" in SERVER


def test_the_server_falls_back_to_cpu_rather_than_dying():
    assert "JINACLIP_FORCE_CPU" in SERVER
    assert 'device = "cpu"' in SERVER


def test_spawn_waits_for_ready_and_reports_a_load_failure(w, tmp_path, monkeypatch):
    """A handle that is still loading would turn the first embedding request
    into the load error."""
    monkeypatch.setattr(w, "ensure_installed", lambda log=None: Path(sys.executable))
    script = tmp_path / "fake_server.py"
    script.write_text("import sys\nsys.exit(3)\n")
    monkeypatch.setattr(w, "_SERVER_SCRIPT", script)
    with pytest.raises(RuntimeError) as e:
        w.spawn("jinaai/jina-clip-v2", load_timeout=30)
    assert "exited while loading" in str(e.value)


def test_spawn_returns_once_the_child_is_ready(w, tmp_path, monkeypatch):
    monkeypatch.setattr(w, "ensure_installed", lambda log=None: Path(sys.executable))
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
    monkeypatch.setattr(w, "ensure_installed", lambda log=None: Path(sys.executable))
    script = tmp_path / "fake_server.py"
    script.write_text(
        "import json,sys\n"
        "print(json.dumps({'ready': True, 'dim': 4}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    print(json.dumps({'error': 'the model returned non-finite values'}),\n"
        "          flush=True)\n")
    monkeypatch.setattr(w, "_SERVER_SCRIPT", script)
    proc = w.spawn("x", device="cpu", load_timeout=60)
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
    i = EMB.index("def _load_jinaclip(")
    body = EMB[i:EMB.index("def _jinaclip_proc(")]
    assert "AutoModel.from_pretrained" not in body
    assert "jinaclip_worker" in body


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
    i = EMB.index("def _jinaclip_proc(")
    body = EMB[i:EMB.index("def _jinaclip_embed_texts(")]
    assert "respawn" in body and "poll()" in body


def test_both_embed_paths_get_the_handle_not_the_process():
    """They need its lock and its respawn hook, which only the wrapper has."""
    for call in ("_jinaclip_embed_texts(model_obj, texts, dimensions)",
                 "_jinaclip_embed_images(model_obj, pil_images, dimensions)"):
        assert call in EMB, call


def test_images_reach_the_child_as_files_and_are_cleaned_up():
    i = EMB.index("def _jinaclip_embed_images(")
    body = EMB[i:i + 1400]
    assert "NamedTemporaryFile" in body
    assert "unlink" in body and "finally" in body


def test_a_model_pinned_off_the_gpu_stays_off_it():
    i = EMB.index("def _load_jinaclip(")
    body = EMB[i:EMB.index("def _jinaclip_proc(")]
    assert "n_gpu_layers" in body and "'cpu'" in body


# ── the file the container cannot see ────────────────────────────────────────

def test_the_fallback_pins_mirror_the_requirements_file(w):
    """Two copies of a pin drift, and the second one exists because the first is
    sometimes not there: the container mounts only codai/, tools/ and
    third_party/ over the image, so a repo-ROOT requirements file added after
    the running image was built is absent until the container is recreated.
    Measured here as a venv that built perfectly and then refused to install."""
    req = (ROOT / "requirements-jinaclip.txt").read_text(encoding="utf-8")
    in_file = [l.strip() for l in req.splitlines()
               if l.strip() and not l.strip().startswith("#")]
    assert sorted(in_file) == sorted(w._FALLBACK_PINS), (
        "requirements-jinaclip.txt and _FALLBACK_PINS disagree:\n"
        f"  file: {sorted(in_file)}\n  code: {sorted(w._FALLBACK_PINS)}")


def test_a_missing_requirements_file_installs_the_pins_instead(w, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(w, "_REQUIREMENTS", tmp_path / "not-here.txt")
    monkeypatch.setattr(w, "resolve_venv_dir", lambda: tmp_path / "venv")
    monkeypatch.setattr(w.venv_env, "create_venv",
                        lambda d, **k: Path(sys.executable))
    monkeypatch.setattr(w, "_transformers_ok",
                        lambda py: bool(seen))      # false, then true
    monkeypatch.setattr(w.venv_env, "run",
                        lambda cmd, **kw: (seen.append(cmd)
                                           or subprocess.CompletedProcess(cmd, 0)))
    monkeypatch.setattr(w, "_bootstrapped", False)
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
    monkeypatch.setattr(w, "ensure_installed", lambda log=None: Path(sys.executable))
    script = tmp_path / "fake_server.py"
    script.write_text("import json\nprint(json.dumps({'ready': True, 'dim': 4}),"
                      " flush=True)\nimport sys\nsys.stdin.read()\n")
    monkeypatch.setattr(w, "_SERVER_SCRIPT", script)
    proc = w.spawn("x", device="cpu", load_timeout=60)
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
