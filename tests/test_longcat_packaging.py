"""Step 1 of the LongCat-Video integration: the isolated Python 3.10 venv and its
packaging.

LongCat is the strictest isolation in the tree — Python 3.10, torch 2.6+cu124,
transformers 4.41 and numpy 1.26, four simultaneous conflicts with the main venv
(3.13 / 2.11+cu130 / 5.x / 2.4). Nothing in codai/ can even create the venv, because
every bootstrap there uses `sys.executable -m venv`. So the interpreter is the standalone
3.10 already bundled for the lip-sync tools, and these tests pin the contract that makes
that work: the pins that must not drift, and the four packaging hooks that carry the venv
into the image.

No venv, no weights, no GPU: this reads the declared files.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.config import Config, LongcatConfig


REQ = (ROOT / "requirements-longcat.txt").read_text()


def _pin(name: str):
    m = re.search(rf"^{re.escape(name)}==([^\s#]+)", REQ, re.M)
    return m.group(1) if m else None


# ------------------------------------------------------------ the pins
def test_the_four_conflicting_pins_are_exact():
    """These are the reason the venv exists. A range here would let pip resolve them to
    the main venv's versions and the isolation would quietly stop being isolation."""
    assert _pin("torch") == "2.6.0"
    assert _pin("transformers") == "4.41.0"
    assert _pin("numpy") == "1.26.4"
    assert _pin("diffusers") == "0.35.1"


def test_it_installs_from_the_cu124_index():
    """torch 2.6.0 from PyPI is not the cu124 build; LongCat's CUDA stack is 12.4."""
    assert "download.pytorch.org/whl/cu124" in REQ


def test_flash_attn_is_not_installed_by_default():
    """flash-attn==2.7.4.post1 is a SOURCE build needing ninja and a CUDA toolchain, and
    upstream accepts xformers instead. Pinning it here would make the venv unbuildable on
    any host without nvcc wired up."""
    assert not re.search(r"^flash-attn==", REQ, re.M)
    assert _pin("xformers") is not None


def test_streamlit_is_not_installed():
    """Upstream's demo UI. coderai serves the API itself."""
    assert not re.search(r"^streamlit==", REQ, re.M)


def test_the_opencv_clash_is_recorded():
    """opencv-python against the bundled opencv-contrib is the fourth conflict; it is
    installed deliberately IN THIS VENV."""
    assert _pin("opencv-python") == "4.9.0.80"


# ------------------------------------------------------------ the config section
def test_the_config_section_is_attached():
    c = Config()
    assert isinstance(c.longcat, LongcatConfig)


def test_it_ships_disabled_and_does_not_auto_build():
    """auto_build pulls torch 2.6 plus its CUDA runtime — minutes and gigabytes on a
    first request, so the operator opts in."""
    lc = LongcatConfig()
    assert lc.enabled is False
    assert lc.auto_build is False


def test_the_attention_default_matches_what_the_requirements_install():
    assert LongcatConfig().attention == "xformers"


def test_the_ready_timeout_allows_for_a_13b_load():
    assert LongcatConfig().ready_timeout >= 600.0


def test_the_remote_escape_hatch_exists():
    """A service already running elsewhere (a pod, another host) must be reachable
    without building or downloading anything locally."""
    assert LongcatConfig().service_url == ""
    assert hasattr(LongcatConfig(), "service_url")


def test_the_section_round_trips_through_config_json(tmp_path):
    """Save and reload: a field that save_config does not write is one the operator can
    set in the interface and lose on the next restart."""
    from codai.config import ConfigManager
    cm = ConfigManager(str(tmp_path))
    cm.config = cm.load()
    cm.config.longcat.enabled = True
    cm.config.longcat.venv = "/opt/coderai/longcat_venv"
    cm.config.longcat.python = "/opt/coderai/py310/bin/python3.10"
    cm.config.longcat.auto_build = True
    cm.config.longcat.attention = "flash"
    cm.config.longcat.ready_timeout = 2400.0
    cm.config.longcat.service_url = "http://pod:8000"
    cm.config.longcat.extra_env = "NCCL_DEBUG=WARN"
    cm.save_config()

    again = ConfigManager(str(tmp_path)).load()
    assert again.longcat.enabled is True
    assert again.longcat.venv == "/opt/coderai/longcat_venv"
    assert again.longcat.python == "/opt/coderai/py310/bin/python3.10"
    assert again.longcat.auto_build is True
    assert again.longcat.attention == "flash"
    assert again.longcat.ready_timeout == 2400.0
    assert again.longcat.service_url == "http://pod:8000"
    assert again.longcat.extra_env == "NCCL_DEBUG=WARN"


def test_the_settings_api_exposes_the_section():
    """The admin settings route is what the interface reads; a section missing there
    cannot be configured from the UI at all."""
    src = (ROOT / "codai/admin/routes.py").read_text()
    assert '"longcat": {' in src
    assert 'if "longcat" in data:' in src          # and it is parsed back on save


# ------------------------------------------------------------ packaging hooks
BUILD = (ROOT / "packaging/linux/build_oci_image.sh").read_text()
DOCKER = (ROOT / "packaging/linux/Dockerfile.oci-venv").read_text()
SMOKE = (ROOT / "packaging/linux/smoke_test_services.sh").read_text()


def test_the_build_declares_and_bundles_the_venv():
    assert "LONGCAT_VENV=" in BUILD
    assert 'rsync -a "${_venv_excl[@]}" "$LONGCAT_VENV/" "$bundle/longcat_venv/"' in BUILD


def test_the_interpreter_is_found_from_either_py310_venv():
    """It used to be read only from the lip-sync venv's pyvenv.cfg, so a host with
    LongCat but no lip-sync venv would bundle the venv and no interpreter for it."""
    assert 'for _v in "$LIPSYNC_VENV" "$LONGCAT_VENV"' in BUILD


def test_the_image_copies_the_venv_and_repoints_it():
    """A bundled venv still carries the BUILD host's `home =`; without the rewrite its
    bin/python points at an interpreter that does not exist in the image."""
    assert "for d in lipsync_venv longcat_venv" in DOCKER
    assert "for v in lipsync_venv longcat_venv" in DOCKER
    assert "home = /opt/coderai/py310/bin" in DOCKER


def test_the_smoke_test_checks_the_versions_not_just_the_import():
    """A venv re-pointed at the WRONG interpreter passes a bare `import torch` while
    being useless for LongCat, so the check asserts 3.10 / 2.6 / 4.41."""
    assert "longcat_venv" in SMOKE
    assert "(3,10)" in SMOKE
    assert "'2.6'" in SMOKE
    assert "'4.41'" in SMOKE
