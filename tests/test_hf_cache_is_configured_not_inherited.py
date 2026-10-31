"""One HuggingFace cache, the configured one, in every process.

A container has to point HF_HOME somewhere before any config exists, so the
image bakes one in. That value then wins in every process that does not know
better, and `models.hf_cache_dir` was applied in exactly one place -- inside the
server's main(), by mutating os.environ. os.environ does not reach a sibling
process, so supervisord's other children, and any subprocess that runs in its
own venv, kept using the image's directory. The result is two caches holding the
same models: measured on one install, 22 GB in the image's directory with 13 GB
of it a byte-for-byte duplicate of the configured cache -- on the smaller disk,
which was 99% full.

Applying it from codai/__init__.py fixes that for anything that imports the
package, which is every coderai process, without the launcher being involved.
The two remaining cases are covered here as well: the OCR fleet supervisor,
which is deliberately stdlib-only and built the path itself, and the
isolated-venv workers, which never import the package and were given only two
of the four variables that decide where a download lands.
"""
import json
import os
import subprocess
import sys

import pytest


@pytest.fixture(autouse=True)
def _restore_hf_env():
    """Put the four variables back exactly as they were.

    monkeypatch is not enough on its own: ``delenv(name, raising=False)`` on a
    name that is not set records nothing, so a variable the code under test
    CREATES survives the test and leaks into the next module -- which is how an
    earlier version of this file broke three longcat tests that set HF_HOME and
    expect it to be honoured."""
    from codai.hf_cache import _VAR_NAMES
    names = _VAR_NAMES + ("CODERAI_CONFIG_DIR", "CODERAI_CACHE_DIR")
    before = {k: os.environ.get(k) for k in names}
    try:
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _config(tmp_path, hf_cache_dir="/AI/huggingface", extra=None):
    """A deployment config dir, as CODERAI_CONFIG_DIR names it."""
    d = tmp_path / "coderai"
    d.mkdir(parents=True, exist_ok=True)
    doc = {"models": {"hf_cache_dir": hf_cache_dir}}
    if extra:
        doc.update(extra)
    (d / "config.json").write_text(json.dumps(doc))
    return d


@pytest.fixture
def image_env(monkeypatch, tmp_path):
    """An install whose IMAGE points HuggingFace at the wrong disk."""
    monkeypatch.setenv("HF_HOME", "/cache/huggingface")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "/cache/huggingface/hub")
    monkeypatch.setenv("TRANSFORMERS_CACHE", "/cache/huggingface/transformers")
    monkeypatch.setenv("CODERAI_CACHE_DIR", "/cache")
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("CODERAI_NO_HF_CACHE_ENV", raising=False)
    _config(tmp_path)
    monkeypatch.setenv("CODERAI_CONFIG_DIR", str(tmp_path))
    return tmp_path


def test_the_configured_dir_is_read_from_config_json(image_env):
    from codai import hf_cache
    assert hf_cache.configured_hf_cache_dir() == "/AI/huggingface"


def test_config_beats_the_image(image_env):
    """The defect in one line: the image said /cache, the config said /AI."""
    from codai import hf_cache
    assert hf_cache.apply_hf_cache_env() == "/AI/huggingface"
    assert os.environ["HF_HOME"] == "/AI/huggingface"
    assert os.environ["HF_HUB_CACHE"] == "/AI/huggingface/hub"
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == "/AI/huggingface/hub"
    # The hub directory, not a `transformers` one beside it: transformers 4.x
    # honours TRANSFORMERS_CACHE as a HUB-layout cache, so pointing it at its
    # own corner gives the machine two hub caches and a worker that cannot see
    # what the server downloaded (measured: 122 models in one, 2 in the other).
    assert os.environ["TRANSFORMERS_CACHE"] == "/AI/huggingface/hub"


def test_all_four_variables_or_the_cache_splits_again(image_env):
    """TRANSFORMERS_CACHE is deprecated upstream and still honoured; leaving it
    behind is what put 3.1 GB in the image's directory."""
    from codai import hf_cache
    hf_cache.apply_hf_cache_env()
    assert not any(v.startswith("/cache/huggingface")
                   for v in (os.environ[k] for k in hf_cache._VAR_NAMES))


def test_nothing_configured_leaves_the_environment_alone(monkeypatch, tmp_path):
    """An install on the HuggingFace default keeps it: no directory is invented."""
    from codai import hf_cache
    _config(tmp_path, hf_cache_dir="")
    monkeypatch.setenv("CODERAI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HF_HOME", "/cache/huggingface")
    assert hf_cache.apply_hf_cache_env() is None
    assert os.environ["HF_HOME"] == "/cache/huggingface"


def test_a_missing_config_file_is_not_an_error(monkeypatch, tmp_path):
    from codai import hf_cache
    monkeypatch.setenv("CODERAI_CONFIG_DIR", str(tmp_path / "nothing-here"))
    assert hf_cache.configured_hf_cache_dir() is None
    assert hf_cache.apply_hf_cache_env() is None


def test_opting_out_is_respected(image_env, monkeypatch):
    from codai import hf_cache
    monkeypatch.setenv(hf_cache.OPT_OUT, "1")
    assert hf_cache.apply_hf_cache_env() is None
    assert os.environ["HF_HOME"] == "/cache/huggingface"


def test_an_explicit_dir_wins_over_the_config(image_env):
    """main() re-applies against the config it actually loaded, which may come
    from a --config dir other than CODERAI_CONFIG_DIR."""
    from codai import hf_cache
    assert hf_cache.apply_hf_cache_env("/elsewhere/hf") == "/elsewhere/hf"
    assert os.environ["HF_HUB_CACHE"] == "/elsewhere/hf/hub"


def test_huggingface_hub_constants_are_patched_when_already_imported(image_env, monkeypatch):
    """The library computes its constants once, at import time."""
    import types
    fake = types.ModuleType("huggingface_hub.constants")
    fake.HF_HUB_CACHE = "/cache/huggingface/hub"
    fake.HF_HOME = "/cache/huggingface"
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake)
    from codai import hf_cache
    hf_cache.apply_hf_cache_env()
    assert fake.HF_HUB_CACHE == "/AI/huggingface/hub"
    assert fake.HF_HOME == "/AI/huggingface"


def test_applying_twice_changes_nothing(image_env):
    from codai import hf_cache
    hf_cache.apply_hf_cache_env()
    before = {k: os.environ[k] for k in hf_cache._VAR_NAMES}
    hf_cache.apply_hf_cache_env()
    assert {k: os.environ[k] for k in hf_cache._VAR_NAMES} == before


# ------------------------------------------------- the two processes that do
#                                                   not import the package

def test_a_venv_child_gets_all_four(image_env):
    """An isolated worker never imports codai, so the parent hands it the lot."""
    from codai import hf_cache
    hf_cache.apply_hf_cache_env()
    env = hf_cache.child_env()
    assert env["HF_HOME"] == "/AI/huggingface"
    assert env["TRANSFORMERS_CACHE"] == "/AI/huggingface/hub"
    assert env["HF_HUB_CACHE"] == env["HUGGINGFACE_HUB_CACHE"] == "/AI/huggingface/hub"


def test_a_venv_child_follows_this_process(image_env, monkeypatch):
    """Not the config: a caller that pointed THIS process somewhere means it."""
    from codai import hf_cache
    monkeypatch.setenv("HF_HOME", "/somewhere/else")
    assert hf_cache.child_env()["HF_HUB_CACHE"] == "/somewhere/else/hub"


def test_current_dir_is_derived_from_the_hub_path(monkeypatch):
    from codai import hf_cache
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", "/AI/huggingface/hub")
    assert hf_cache.current_hf_cache_dir() == "/AI/huggingface"


@pytest.mark.parametrize("worker", [
    "canary", "crisperwhisper", "h3", "longcat", "melotts", "parler", "pyannote",
])
def test_no_worker_sets_only_the_hub_pair(worker):
    """The regression this guards: two of four variables set, model downloaded
    twice. Any worker that spawns a venv must build its environment with a
    helper that applies all four -- never by hand.

    The helper moved: the workers now call ``venv_env.env_for_python``, which
    applies ``hf_cache.child_env`` AND decides the library path from whose
    interpreter the venv holds. What is asserted is that the environment comes
    from one of those helpers, not which one."""
    src = open(f"codai/api/{worker}_worker.py").read()
    assert 'env["HF_HUB_CACHE"]' not in src, f"{worker}: sets the hub pair by hand"
    assert any(h in src for h in ("child_env", "env_for_python", "env_for_venv")), (
        f"{worker}: builds its environment without a helper that applies all "
        f"four HF variables")
    assert "dict(os.environ)" not in src and "{**os.environ" not in src, (
        f"{worker}: builds its own environment dict, so it inherits both the "
        f"image's HF variables and the engine's PYTHONHOME")


def test_the_ocr_supervisor_uses_the_configured_dir(image_env):
    """It is stdlib-only on purpose, and used to build the path from
    CODERAI_CACHE_DIR -- which is the image's directory, not the config's."""
    sys.path.insert(0, "codai")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ocrfw_under_test", "codai/ocr_fleet_workers.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    env = mod.worker_env(0, 2)
    assert env["HF_HOME"] == "/AI/huggingface"
    assert env["HF_HUB_CACHE"] == "/AI/huggingface/hub"
    assert env["TRANSFORMERS_CACHE"] == "/AI/huggingface/hub"
    # and it still keeps the cache dir it genuinely owns
    assert env["XDG_CACHE_HOME"] == "/cache"


def test_the_ocr_supervisor_imports_nothing_from_the_package():
    """Importing codai into the supervisor would cost what it exists to save."""
    src = open("codai/ocr_fleet_workers.py").read()
    assert "from codai" not in src and "import codai" not in src


def test_importing_the_package_applies_it(tmp_path):
    """The whole point: no launcher involved. A fresh interpreter whose image
    env is wrong, importing codai, ends up on the configured cache."""
    _config(tmp_path)
    env = dict(os.environ)
    env.update({
        "CODERAI_CONFIG_DIR": str(tmp_path),
        "HF_HOME": "/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/cache/huggingface/hub",
        "TRANSFORMERS_CACHE": "/cache/huggingface/transformers",
    })
    env.pop("CODERAI_NO_HF_CACHE_ENV", None)
    out = subprocess.run(
        [sys.executable, "-c",
         "import codai, os, json;"
         "print(json.dumps({k: os.environ.get(k) for k in "
         "('HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE')}))"],
        capture_output=True, text=True, env=env, cwd=os.getcwd(), timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got == {
        "HF_HOME": "/AI/huggingface",
        "HF_HUB_CACHE": "/AI/huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": "/AI/huggingface/hub",
        "TRANSFORMERS_CACHE": "/AI/huggingface/hub",
    }
