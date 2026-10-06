"""Building INT8 LongCat weights from the model page.

A 13.6B bf16 DiT does not fit a 24 GB card and streams across PCIe every step; the
INT8 copy does fit. It is an ACTION, never a default — it is lossy and slow — and
the bf16 weights are kept, so un-ticking "INT8 DiT" goes straight back to them.

The quantiser itself runs in LongCat's isolated py3.10 venv (it needs upstream's
helpers and torch 2.6), so it is a worker script like training, not in-process.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "longcat_quantize.py"
SERVICE = ROOT / "tools" / "longcat_service.py"
WORKER = ROOT / "codai" / "api" / "longcat_worker.py"
ROUTES = ROOT / "codai" / "admin" / "routes.py"
PAGE = ROOT / "codai" / "admin" / "templates" / "models.html"


def test_the_quantiser_script_exists_and_parses():
    import ast
    assert SCRIPT.is_file()
    ast.parse(SCRIPT.read_text(encoding="utf-8"))


def test_it_never_replaces_the_bf16_weights():
    """It writes a sibling directory; nothing removes or overwrites dit/."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "base_model_int8" in src
    for destructive in ("shutil.rmtree", "os.remove", "os.unlink", "unlink("):
        assert destructive not in src, f"{destructive} has no business here"


def test_it_quantises_on_the_cpu():
    """The whole point is to make a model that does not fit the card — so getting
    there must not need the card. The streaming path reads tensors straight to CPU
    and never touches CUDA."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert 'device="cpu"' in src
    assert "cuda" not in src.lower()


def test_an_existing_build_is_not_silently_redone():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "overwrite" in src and "already quantised" in src


def test_the_worker_runs_it_in_the_inference_venv():
    src = WORKER.read_text(encoding="utf-8")
    assert "_QUANTIZE_SCRIPT" in src
    assert "def quantize_int8(" in src
    # ensure_built() is the inference venv; ensure_train_built() is SimpleTuner's.
    body = src[src.index("def quantize_int8("):]
    body = body[:body.index("\ndef ")]
    assert "ensure_built(config)" in body
    assert "ensure_train_built" not in body


def test_the_service_can_load_a_quantised_base_dit():
    """Upstream's load_quantized_dit hardcodes the AVATAR transformer, which is
    wrong for a quantised base checkpoint — so the base family needs its own."""
    src = SERVICE.read_text(encoding="utf-8")
    assert "_load_quantized_base_dit" in src
    assert "LongCatVideoTransformer3DModel(**config)" in src
    # the avatar path must still use upstream's loader
    assert "load_quantized_dit(root, subfolder=LC.INT8_SUBDIR" in src


def test_the_endpoint_routes_longcat_models_to_int8():
    src = ROUTES.read_text(encoding="utf-8")
    assert '_model_backend(model_id) == "longcat"' in src
    assert "_start_longcat_quantize" in src
    # and a model whose weights are absent is refused rather than started
    assert "not downloaded yet" in src


def test_the_status_endpoint_reports_longcat_jobs():
    src = ROUTES.read_text(encoding="utf-8")
    assert "_longcat_quant_jobs" in src


def test_the_page_offers_the_action_next_to_the_int8_switch():
    html = PAGE.read_text(encoding="utf-8")
    assert 'id="cfg-longcat-int8"' in html, "the use-int8 switch is gone"
    assert 'id="cfg-longcat-quant-btn"' in html
    assert "startLongcatQuantize" in html
    # it must read the same hidden field the other quantize button uses
    js = html[html.index("async function startLongcatQuantize"):]
    assert "getElementById('cfg-path')" in js[:600]


def test_quantising_is_not_the_default_anywhere():
    """It must stay opt-in: nothing may turn use_int8 on by itself."""
    for path in (SERVICE, WORKER):
        src = path.read_text(encoding="utf-8")
        assert not re.search(r'use_int8["\']?\s*[:=]\s*True', src), path.name


# ── the gate that decides whether INT8 may be SERVED ────────────────────────

import sys as _sys
_sys.path.insert(0, str(ROOT / "tools"))
LC = pytest.importorskip("longcat_common")


def _ckpt(tmp_path, with_int8=False):
    if with_int8:
        (tmp_path / LC.INT8_SUBDIR).mkdir(parents=True)
    return str(tmp_path)


def test_int8_is_refused_on_a_base_checkpoint_that_has_none(tmp_path):
    """Upstream ships INT8 only for avatar-1.5, so without built weights this is a
    request for something that does not exist."""
    problems = LC.variant_problems("int8", "", checkpoint_dir=_ckpt(tmp_path))
    assert problems and "Build INT8 weights" in problems[0]


def test_int8_is_allowed_once_the_weights_have_been_built(tmp_path):
    """The regression this guards: the quantise button produces base_model_int8/ for
    a BASE checkpoint, and the gate used to reject it anyway because the family was
    not avatar-1.5 — so the weights could be built and never served."""
    assert LC.variant_problems("int8", "", checkpoint_dir=_ckpt(tmp_path, True)) == []


def test_use_int8_follows_the_same_rule_as_the_variant(tmp_path):
    assert LC.variant_problems("bf16", "", use_int8=True,
                               checkpoint_dir=_ckpt(tmp_path)) != []
    assert LC.variant_problems("bf16", "", use_int8=True,
                               checkpoint_dir=_ckpt(tmp_path, True)) == []


def test_upstreams_avatar_int8_still_needs_no_local_build():
    assert LC.variant_problems("int8", "avatar-1.5") == []


def test_bf16_is_never_gated(tmp_path):
    assert LC.variant_problems("bf16", "", checkpoint_dir=_ckpt(tmp_path)) == []


def test_the_service_passes_the_checkpoint_to_the_gate():
    src = SERVICE.read_text(encoding="utf-8")
    assert "checkpoint_dir=checkpoint" in src, "the gate cannot see the built weights"


# ── it must not take the machine down ───────────────────────────────────────
# Measured: on a 54 GB host this filled RAM and all 4 GB of swap and had to be
# killed to keep the server alive. The peak holds the bf16 model, the INT8 copy
# and the state dict the save builds.

import importlib.util as _ilu2

_qspec = _ilu2.spec_from_file_location("lc_quant", SCRIPT)
LQ = _ilu2.module_from_spec(_qspec)
_qspec.loader.exec_module(LQ)


def test_it_estimates_the_peak_from_the_weights_on_disk(tmp_path):
    d = tmp_path / "dit"
    d.mkdir()
    # 8 GB of fp32 shards -> 4 GB bf16 -> 1.5x + 4 floor
    (d / "a.safetensors").write_bytes(b"\0" * 1024)
    size = 8 * 1024 ** 3
    import os as _os
    _os.truncate(d / "a.safetensors", size)
    got = LQ._needed_gb(str(tmp_path), "dit")
    assert abs(got - (4 * 1.5 + 4)) < 0.1, got


def test_an_unreadable_checkpoint_does_not_block_the_run(tmp_path):
    """0 means 'unknown', and unknown must not refuse."""
    assert LQ._needed_gb(str(tmp_path), "nope") == 0.0


def test_available_memory_is_read_from_the_kernel():
    got = LQ._available_gb()
    assert got >= 0.0
    if got:
        assert got < 10000, "that is not gigabytes"


def test_the_job_still_checks_the_host_has_room():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "not enough host memory" in src
    assert "_available_gb()" in src
    assert "min_free_gb" in src


def test_nothing_loads_the_weights_into_a_model():
    """Superseded by the streaming rewrite: there is no load left to make cheap."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "low_cpu_mem_usage" not in src


def test_memory_is_released_as_it_goes():
    src = SCRIPT.read_text(encoding="utf-8")
    assert src.count("gc.collect()") >= 2
    assert "del tensor" in src


# ── the streaming rewrite ───────────────────────────────────────────────────
# Materialising the model needed ~42 GB and exhausted a 54 GB host. The peak is
# now one tensor plus the 4 GB shard being accumulated.

def test_the_model_is_never_materialised():
    src = SCRIPT.read_text(encoding="utf-8")
    # matched as CALLS, so prose explaining what upstream does does not trip it
    import re as _re
    for call in ("from_pretrained", "quantize_model", "save_quantized_state_dict"):
        assert not _re.search(r"^\s*(?:\w+\s*=\s*)?" + call + r"\(", src, _re.M), call


def test_the_layout_is_learned_on_the_meta_device():
    """Free: a 13.6B transformer costs no memory there, and it gives the exact set
    of modules upstream would quantise rather than a guess from tensor shapes."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert 'torch.device("meta")' in src
    assert "DEFAULT_SKIP_PATTERNS" in src, "upstream's skip list must still apply"


def test_tensors_are_streamed_one_at_a_time():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "safe_open(" in src and "get_tensor(" in src


def test_the_emitted_keys_match_what_the_loader_expects():
    """load_quantized_dit builds QuantizedLinear modules whose buffers are named
    weight_int8 / weight_scale / bias — the output has to use those exact names."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert '.weight_int8"' in src
    assert '.weight_scale"' in src
    assert "quantized_model.safetensors.index.json" in src
    assert "quantization_config.json" in src


def test_the_quantisation_matches_upstreams_arithmetic():
    """Per-channel symmetric, /127, clamped — same as QuantizedLinear.from_linear."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "abs().amax(dim=1).clamp(min=1e-8) / 127.0" in src
    assert "round().clamp(-128, 127)" in src


def test_shards_are_capped_so_the_buffer_cannot_grow_unbounded():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "4 * 1024 * 1024 * 1024" in src
    assert "_flush()" in src


def test_the_memory_floor_is_now_a_sanity_check_not_the_constraint():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "or 8.0" in src, "the 42 GB ceiling should be gone"
