"""Step 5: LongCat is configurable from the interface, and the settings survive.

The requirement is that a supported model have "full configuration in the interface". The
trap is specific and silent: `api_model_configure` rebuilds the entry from scratch and
copies only an explicit whitelist, so a key with a UI control but no whitelist entry is
written once and **deleted on the next save from the models page**. That is exactly why
H3's documented `h3_venv` / `in_process` / `dtype` keys do not survive, and it fails
quietly — the field looks saved until you reload.

So these tests tie the three layers together: a control exists, the save sends it, and the
whitelist keeps it.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HTML = (ROOT / "codai/admin/templates/models.html").read_text()
ROUTES = (ROOT / "codai/admin/routes.py").read_text()
VIDEO = (ROOT / "codai/api/video.py").read_text()

# Every per-model LongCat setting: the control id, and the entry key it persists as.
FIELDS = {
    "cfg-longcat-variant": "variant",
    "cfg-longcat-quality": "quality",
    "cfg-longcat-offload": "offload_strategy",
    "cfg-longcat-segments": "num_segments",
    "cfg-longcat-cond-frames": "num_cond_frames",
    "cfg-longcat-cp": "cp_size",
    "cfg-longcat-source": "longcat_source",
    "cfg-longcat-venv": "longcat_venv",
    "cfg-longcat-ref-img": "ref_img_index",
    "cfg-longcat-mask-range": "mask_frame_range",
    "cfg-longcat-int8": "use_int8",
    "cfg-longcat-distill": "use_distill",
    "cfg-longcat-offload-kv": "offload_kv_cache",
    # Quantising the text encoder is what decides whether the pipeline can stay
    # resident on a 24 GB card; BSA whether attention is sparse; base_model where an
    # avatar family borrows tokenizer/text_encoder/vae; cp_split_hw the CP tile.
    "cfg-longcat-te-quant": "text_encoder_quant",
    "cfg-longcat-bsa": "bsa",
    "cfg-longcat-base-model": "base_model",
    "cfg-longcat-cp-split": "cp_split_hw",
}


def _whitelist() -> set:
    """The keys api_model_configure copies onto a rebuilt entry."""
    start = ROUTES.index('for key in ("alias", "config_name", "backend"')
    block = ROUTES[start:ROUTES.index("):", start)]
    return set(re.findall(r'"([a-z0-9_]+)"', block))


# ------------------------------------------------------------ the backend is selectable
def test_longcat_is_in_the_backend_select():
    """The select is a hard-coded option list; a backend absent from it cannot be pinned
    from the UI at all — which is why H3's `backend: "h3"` is unreachable."""
    sel = HTML[HTML.index('id="cfg-backend"'):]
    sel = sel[:sel.index("</select>")]
    assert 'value="longcat"' in sel


def test_the_field_group_is_shown_for_that_backend():
    assert 'id="cfg-longcat-row"' in HTML
    assert "lcrow.style.display = (backend === 'longcat')" in HTML


def test_the_group_is_hidden_by_default():
    """It belongs to one backend; showing it for every model would be noise."""
    row = HTML[HTML.index('id="cfg-longcat-row"'):]
    assert "display:none" in row[:400]


# ------------------------------------------------------------ control → save → whitelist
@pytest.mark.parametrize("control,key", sorted(FIELDS.items()))
def test_each_control_exists(control, key):
    assert f'id="{control}"' in HTML, f"no control {control} for {key}"


@pytest.mark.parametrize("control,key", sorted(FIELDS.items()))
def test_each_control_is_populated_on_open(control, key):
    assert f"'{control}'" in HTML.split("function saveModelConfig")[0], \
        f"{control} is never populated, so it always opens blank"


@pytest.mark.parametrize("control,key", sorted(FIELDS.items()))
def test_each_setting_is_sent_on_save(control, key):
    save = HTML[HTML.index("if (backend === 'longcat') {"):]
    save = save[:save.index("out.vllm")]
    assert f"'{control}'" in save, f"{control} is never read back on save"
    assert f"out.{key}" in save, f"{control} is not sent as {key}"


@pytest.mark.parametrize("key", sorted(set(FIELDS.values())))
def test_each_setting_survives_a_save(key):
    """The one that bites: a key outside the whitelist is dropped on the NEXT save."""
    assert key in _whitelist(), (
        f"'{key}' has a UI control but is not in api_model_configure's whitelist — it "
        f"would be silently deleted the next time the model is saved")


def test_the_whitelist_still_carries_the_pre_existing_keys():
    """Guard against an edit that replaces the tuple instead of extending it."""
    for key in ("alias", "backend", "load_mode", "max_instances", "acceleration",
                "offload_strategy", "capabilities", "placement"):
        assert key in _whitelist()


# ------------------------------------------------------------ the settings reach the engine
def test_the_dispatch_reads_the_entry_defaults():
    """A knob that is saved but never read is worse than no knob. num_segments,
    num_cond_frames and offload_kv_cache are the ones whose feature exists today."""
    fn = VIDEO[VIDEO.index("async def _generate_longcat"):]
    fn = fn[:fn.index("\nasync def ")]
    assert 'model_cfg.get("num_segments")' in fn
    assert 'model_cfg.get("num_cond_frames")' in fn
    assert 'model_cfg.get("offload_kv_cache")' in fn
    assert 'model_cfg.get("quality")' in fn


def test_the_request_still_wins_over_the_entry():
    fn = VIDEO[VIDEO.index("async def _generate_longcat"):]
    fn = fn[:fn.index("\nasync def ")]
    seg = fn[fn.index("_segments ="):fn.index("_cond =")]
    # request first, entry as the fallback
    assert seg.index('getattr(request, "num_segments"') < seg.index("model_cfg.get")


def test_the_variant_reaches_the_service_identity():
    from codai.api import longcat_worker as W
    assert W.service_key("/w", {"variant": "fp8"}) != W.service_key("/w", {"variant": "bf16"})


# ------------------------------------------------------------ global settings section
def test_the_global_section_is_in_the_settings_api():
    assert '"longcat": {' in ROUTES
    assert 'if "longcat" in data:' in ROUTES


def test_the_runtime_fields_round_trip(tmp_path):
    from codai.config import ConfigManager
    cm = ConfigManager(str(tmp_path))
    cm.config = cm.load()
    cm.config.longcat.venv = "/v"
    cm.config.longcat.attention = "flash"
    cm.config.longcat.evict_drain_timeout_s = 120.0
    cm.save_config()
    again = ConfigManager(str(tmp_path)).load()
    assert (again.longcat.venv, again.longcat.attention,
            again.longcat.evict_drain_timeout_s) == ("/v", "flash", 120.0)


# ------------------------------------------------------------ docs
def test_the_docs_name_the_repos_and_the_download_route():
    doc = (ROOT / "docs/longcat-video.md").read_text()
    assert "meituan-longcat/LongCat-Video" in doc
    assert "model-download" in doc and "file_pattern" in doc
    # and are honest about what is not done
    assert "no generation has been verified against real weights" in doc.lower()
