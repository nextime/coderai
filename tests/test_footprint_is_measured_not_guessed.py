# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""The footprint must come from the model, and must survive quantization.

Replaces a process-RSS delta that was wrong by construction. Ground truth from
a production host, 2026-10-29:

    gemma-2-9b-it.Q4_K_M    5.37 GB on disk  ->  reported 30.81 GB   (5.7x)
    LTX-2.5, six loads, one unchanged config:
        22.11, 22.11, 23.28, 4.50, 19.02, 29.80 GB                   (6.6x)

RSS counts mmap'd weight-file pages and, summed over a process tree, counts
pages shared between those processes once per process; as a delta it also
absorbs whatever else moved in the window. The 4.50 GB reading came from a load
that placed one layer and then raised, and the 29.80 GB one was taken after an
abandoned resident attempt had pushed a transformer and text encoder to host
RAM. Both were persisted under force_vram_update and planned against.
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.models import footprint as F

torch = pytest.importorskip("torch")
nn = torch.nn


# ---------------------------------------------------------------- weights ---

def test_a_plain_module_measures_its_real_bytes():
    m = nn.Linear(1000, 1000, bias=False).to(torch.float32)
    r = F.walk_module_bytes(m)
    assert r["gpu"] == 0
    assert r["cpu"] == 1000 * 1000 * 4          # fp32, exactly
    assert r["declared"] == 0


def test_dtype_is_read_from_the_tensor_not_assumed():
    """Half precision must halve the figure -- no factor table involved."""
    big = F.walk_module_bytes(nn.Linear(512, 512, bias=False).to(torch.float32))
    small = F.walk_module_bytes(nn.Linear(512, 512, bias=False).to(torch.bfloat16))
    assert big["cpu"] == 2 * small["cpu"]


def test_a_four_bit_parameter_counts_its_packed_bytes_not_its_shape():
    """A bitsandbytes Params4bit IS a uint8 tensor, so numel x element_size
    already yields the packed size. This is why measuring the object needs no
    quantization correction factor, unlike an estimate from the config."""
    m = nn.Module()
    packed = torch.empty(512 * 512 // 2, dtype=torch.uint8)   # 4 bits/weight
    m.register_parameter("weight", nn.Parameter(packed, requires_grad=False))
    r = F.walk_module_bytes(m)
    assert r["cpu"] == 512 * 512 // 2
    # ... and that is a quarter of the fp16 equivalent, not the same.
    assert r["cpu"] == (512 * 512 * 2) // 4


def test_the_bnb_quant_state_is_counted():
    """absmax / the code book / the nested second-level state are real memory
    and live OUTSIDE parameters(), so a plain walk misses them."""
    class _QS:
        def __init__(self, nested=None):
            self.absmax = torch.empty(4096, dtype=torch.float32)   # 16 KiB
            self.code = torch.empty(256, dtype=torch.float32)      # 1 KiB
            self.state2 = nested

    m = nn.Module()
    p = nn.Parameter(torch.empty(1024, dtype=torch.uint8), requires_grad=False)
    p.quant_state = _QS(nested=_QS())          # double quantization
    m.register_parameter("weight", p)
    r = F.walk_module_bytes(m)
    expected = 1024 + 2 * (4096 * 4 + 256 * 4)
    assert r["cpu"] == expected, "quant_state tensors were not counted"


def test_prequantized_buffers_are_counted():
    """GPTQ/AWQ keep the payload in BUFFERS (qweight/qzeros/scales), so a
    parameters-only walk reports a 4-bit model as very nearly empty."""
    m = nn.Module()
    m.register_buffer("qweight", torch.empty((512, 64), dtype=torch.int32))
    m.register_buffer("qzeros", torch.empty((4, 64), dtype=torch.int32))
    m.register_buffer("scales", torch.empty((4, 512), dtype=torch.float16))
    r = F.walk_module_bytes(m)
    expected = 512 * 64 * 4 + 4 * 64 * 4 + 4 * 512 * 2
    assert r["cpu"] == expected
    assert r["tensors"] == 3


def test_tied_weights_are_counted_once():
    """An embedding shared with the LM head is ONE allocation. Counting the
    same storage twice inflates every model that ties them."""
    emb = nn.Embedding(5000, 256)
    head = nn.Linear(256, 5000, bias=False)
    head.weight = emb.weight                    # tied, as in most LLMs
    m = nn.Module()
    m.emb, m.head = emb, head
    r = F.walk_module_bytes(m)
    assert r["cpu"] == 5000 * 256 * 4
    assert r["tensors"] == 1


def test_a_meta_placeholder_is_sized_from_its_dtype_and_flagged():
    """accelerate leaves a meta tensor where it moved the real weights to host
    RAM or disk. The model still needs those bytes (the footprint is
    placement-independent), but they are DECLARED, not measured."""
    m = nn.Module()
    with torch.device("meta"):
        w = torch.empty((1000, 1000), dtype=torch.bfloat16)
    m.register_parameter("weight", nn.Parameter(w, requires_grad=False))
    r = F.walk_module_bytes(m)
    assert r["cpu"] == 1000 * 1000 * 2
    assert r["declared"] == r["cpu"], "meta bytes were not flagged as declared"


def test_a_pipeline_is_walked_through_its_components():
    """A diffusers pipeline is not an nn.Module; its weights are in components,
    beside schedulers and tokenizers that hold none. Walking the container
    alone would report ~0 for every video and image model."""
    class _Pipe:
        pass
    p = _Pipe()
    p.components = {
        "transformer": nn.Linear(1000, 1000, bias=False).to(torch.bfloat16),
        "vae": nn.Linear(500, 500, bias=False).to(torch.bfloat16),
        "scheduler": object(),                  # no parameters()
        "tokenizer": None,
    }
    gpu, cpu, notes, ok = F.measure_weights(p)
    assert ok
    assert cpu == pytest.approx((1000 * 1000 * 2 + 500 * 500 * 2) / 1e9, rel=1e-6)


# --------------------------------------------------------------------- KV ---

def test_kv_uses_key_value_heads_not_attention_heads():
    """Under GQA many query heads share one KV head. Sizing by attention heads
    over-states the cache by the grouping factor -- 4x on this geometry."""
    gqa = {"num_hidden_layers": 32, "num_attention_heads": 32,
           "num_key_value_heads": 8, "head_dim": 128}
    mha = dict(gqa, num_key_value_heads=32)
    assert F.kv_cache_gb(mha, 4096) == pytest.approx(4 * F.kv_cache_gb(gqa, 4096))


def test_kv_scales_with_the_configured_context():
    g = {"num_hidden_layers": 32, "num_key_value_heads": 8, "head_dim": 128}
    assert F.kv_cache_gb(g, 32768) == pytest.approx(8 * F.kv_cache_gb(g, 4096))


def test_kv_is_exact_for_a_known_geometry():
    """2 (K+V) x layers x kv_heads x head_dim x 2 bytes x tokens."""
    g = {"num_hidden_layers": 32, "num_key_value_heads": 8, "head_dim": 128}
    expect = 32 * 8 * (128 * 16 + 128 * 16) / 8.0 * 32000 / 1e9
    assert F.kv_cache_gb(g, 32000) == pytest.approx(expect, rel=1e-9)


def test_a_quantized_kv_cache_is_smaller():
    """cache_type_k/v=q8_0 roughly halves the cache; reserving an f16-sized one
    forces needless CPU offload at long context."""
    g = {"num_hidden_layers": 32, "num_key_value_heads": 8, "head_dim": 128}
    f16 = F.kv_cache_gb(g, 32000)
    q8 = F.kv_cache_gb(g, 32000, "q8_0", "q8_0")
    assert q8 < f16
    assert q8 == pytest.approx(f16 * 8.5 / 16.0, rel=1e-6)


def test_unknown_geometry_reports_zero_rather_than_guessing():
    assert F.kv_cache_gb({}, 32000) == 0.0
    assert F.kv_cache_gb({"num_hidden_layers": 32}, 32000) == 0.0


def test_a_diffusion_model_gets_no_kv_term():
    pipe = type("P", (), {})()
    pipe.components = {"transformer": nn.Linear(100, 100).to(torch.bfloat16)}
    fp = F.measure(pipe, {"n_ctx": 32768}, "video")
    assert fp.kv_gb == 0.0, "a diffusion pipeline has no KV cache"


# ------------------------------------------------------------ integration ---

def test_the_gemma_weights_are_the_file_not_six_times_it():
    """The production reading: a 5.37 GB Q4_K_M GGUF reported 30.81 GB total.
    A GGUF is loaded as-is with no dtype conversion, so its WEIGHTS are exactly
    the file -- that 5.7x was the RSS delta, not the model."""
    g = {"num_hidden_layers": 42, "num_key_value_heads": 8, "head_dim": 256}
    fp = F.measure(None, {"n_ctx": 32000}, "text", geom=g, file_size_gb=5.37)
    assert fp.complete
    assert fp.weights_gb == pytest.approx(5.37), (
        f"the weights are no longer the file size: {fp.describe()}")


def test_the_kv_estimate_is_labelled_as_one():
    """Predicting KV from metadata has been wrong by 5x in both directions on
    this codebase's own models, so a computed figure must announce itself."""
    g = {"num_hidden_layers": 42, "num_key_value_heads": 8, "head_dim": 256}
    fp = F.measure(None, {"n_ctx": 32000}, "text", geom=g, file_size_gb=5.37)
    assert fp.kv_gb > 0
    assert not fp.kv_measured
    assert any("ESTIMATE" in n for n in fp.notes), fp.notes
    assert "(est)" in fp.describe()


def test_a_measured_kv_size_beats_the_estimate():
    """llama.cpp reports its own KV buffer size exactly. When that has been
    recorded for this context it must win over any computation."""
    g = {"num_hidden_layers": 42, "num_key_value_heads": 8, "head_dim": 256}
    est = F.measure(None, {"n_ctx": 32000}, "text", geom=g, file_size_gb=5.37)
    meas = F.measure(None, {"n_ctx": 32000}, "text", geom=g, file_size_gb=5.37,
                     measured_kv_gb={"32000": 2.2})
    assert meas.kv_gb == pytest.approx(2.2)
    assert meas.kv_measured
    assert meas.kv_gb < est.kv_gb
    assert "(est)" not in meas.describe()


def test_a_measured_kv_size_is_per_context():
    """A figure taken at 4k says nothing about 128k, so it must not be reused
    for a context it was not measured at."""
    g = {"num_hidden_layers": 42, "num_key_value_heads": 8, "head_dim": 256}
    fp = F.measure(None, {"n_ctx": 32000}, "text", geom=g, file_size_gb=5.37,
                   measured_kv_gb={"4096": 0.3})
    assert not fp.kv_measured, "a 4k measurement was reused for a 32k context"


def test_sliding_window_layers_are_sized_to_their_window():
    """Gemma alternates a windowed layer with a global one. Sizing every layer
    at the full context over-states the windowed half by n_ctx/window -- which
    is why llama.cpp's own figure was far below the naive calculation."""
    g = {"num_hidden_layers": 42, "num_key_value_heads": 8, "head_dim": 256,
         "sliding_window": 4096}
    full = F.kv_cache_gb(g, 32000, swa_full=True)
    windowed = F.kv_cache_gb(g, 32000, swa_full=False)
    assert windowed < full
    # 21 global layers at 32000 tokens + 21 windowed layers at 4096.
    expect_ratio = (21 * 32000 + 21 * 4096) / (42 * 32000)
    assert windowed / full == pytest.approx(expect_ratio, rel=1e-6)


def test_a_known_sliding_layer_count_is_honoured():
    """llama.cpp prints is_swa per layer, so a caller that counted them should
    not be overridden by the alternating assumption."""
    g = {"num_hidden_layers": 42, "num_key_value_heads": 8, "head_dim": 256,
         "sliding_window": 4096, "swa_layer_count": 36}
    assumed = F.kv_cache_gb(dict(g, swa_layer_count=0), 32000, swa_full=False)
    stated = F.kv_cache_gb(g, 32000, swa_full=False)
    assert stated < assumed, "swa_layer_count was ignored"


def test_a_window_wider_than_the_context_changes_nothing():
    g = {"num_hidden_layers": 32, "num_key_value_heads": 8, "head_dim": 128,
         "sliding_window": 8192}
    assert (F.kv_cache_gb(g, 4096, swa_full=False)
            == pytest.approx(F.kv_cache_gb(g, 4096, swa_full=True)))


def test_an_unaccountable_model_is_not_reported_as_complete():
    """The 4.50 GB production reading came from a load that placed one layer
    and then raised. A footprint that could not be accounted for must announce
    that, so the caller declines to persist it as authoritative."""
    fp = F.measure(object(), {}, "video")
    assert not fp.complete
    assert fp.notes


def test_the_total_is_placement_independent():
    """Moving weights GPU<->host must not change what the model needs -- the
    contract measured_vram_gb already documents."""
    m = nn.Linear(1000, 1000, bias=False).to(torch.bfloat16)
    pipe = type("P", (), {})()
    pipe.components = {"transformer": m}
    on_host = F.measure(pipe, {}, "video").total_gb
    if torch.cuda.is_available():
        pipe.components = {"transformer": m.cuda()}
        assert F.measure(pipe, {}, "video").total_gb == pytest.approx(on_host)
        m.cpu()
    else:
        assert on_host > 0


def test_companion_files_are_added(tmp_path):
    """An mmproj projector loads ON TOP of the model and is held by a separate
    handler, so walking the model cannot see it."""
    p = tmp_path / "mmproj.gguf"
    p.write_bytes(b"\0" * 1_000_000)
    assert F.extra_file_gb(str(p)) == pytest.approx(0.001)
    assert F.extra_file_gb(None, "", "/nonexistent") == 0.0


def test_a_gguf_wrapper_falls_back_to_the_file_and_stays_complete():
    """A GGUF is not a torch object: llama.cpp owns its memory and the registry
    holds a ModelManager wrapper with no parameters() to walk. The weights are
    the file byte for byte -- loaded as stored, no dtype conversion -- so with
    that size and llama.cpp's own measured KV the production 30.81 GB reading
    for a 5.37 GB Q4_K_M becomes explainable."""
    class _ManagerLike:
        def __init__(self):
            self.backend = object()

    fp = F.measure(_ManagerLike(), {"n_ctx": 32000}, "text",
                   geom={"num_hidden_layers": 42, "num_key_value_heads": 8,
                         "head_dim": 256},
                   file_size_gb=5.37, measured_kv_gb={"32000": 2.2})
    assert fp.complete, fp.notes
    assert fp.weights_gb == pytest.approx(5.37)
    assert fp.kv_gb == pytest.approx(2.2)
    assert fp.kv_measured
    assert fp.total_gb < 10.0, f"still inflated: {fp.describe()}"
    assert any("on-disk size" in n for n in fp.notes)


def test_without_a_file_size_an_unwalkable_object_is_incomplete():
    """Better to decline than to persist a guess: this is the path that wrote
    4.50 GB for a 22 GB model and planned against it four seconds later."""
    class _ManagerLike:
        pass
    fp = F.measure(_ManagerLike(), {"n_ctx": 4096}, "text")
    assert not fp.complete


# ------------------------------------------------- the partial-load guard ---

def _manager():
    from codai.models.manager import multi_model_manager
    return multi_model_manager


def _load(m, key, width):
    class Pipe:
        pass
    p = Pipe()
    p.components = {"transformer": nn.Linear(width, width, bias=False).to(torch.bfloat16)}
    m.models[key] = p
    m.active_in_vram = key
    m.record_vram_delta(key, free_before=10.0, ram_before=1.0)


def test_a_partial_load_cannot_overwrite_a_good_measurement():
    """The production failure: a load placed ONE layer, raised, and still wrote
    4.505 GB to disk under force_vram_update -- and 'Full-GPU load looks viable
    (4.5 GB peak need)' was planned off it four seconds later."""
    m = _manager()
    key = "video:__guard_partial"
    m.config[key] = {"force_vram_update": True}
    try:
        _load(m, key, 20000)
        good = m.config[key].get("measured_vram_gb")
        assert good and good > 0.5
        _load(m, key, 6000)                       # a fraction of the real model
        assert m.config[key].get("measured_vram_gb") == good, (
            "a partial load overwrote the good measurement")
    finally:
        m.models.pop(key, None)
        m.config.pop(key, None)


def test_a_real_drop_with_a_quantization_change_is_accepted():
    """Wan2.2 genuinely went 40.04 -> 11.73 GB once its transformers were
    quantized. That must not be mistaken for a partial load -- and it is told
    apart by the quantization signature changing with it."""
    m = _manager()
    key = "video:__guard_quant"
    m.config[key] = {"force_vram_update": True}
    try:
        _load(m, key, 20000)
        before = m.config[key].get("measured_vram_gb")
        m.config[key]["load_in_4bit"] = True      # signature changes with it
        _load(m, key, 6000)
        after = m.config[key].get("measured_vram_gb")
        assert after < before, "the quantization escape hatch did not open"
    finally:
        m.models.pop(key, None)
        m.config.pop(key, None)


def test_an_empty_pipeline_is_reported_as_a_failed_load():
    """Walked every component and none holds weights: that is a failed load,
    not an object we cannot introspect. The difference decides whether a
    fallback estimate is legitimate -- and it is not, because a size taken from
    process memory measures everything EXCEPT this model."""
    class Pipe:
        pass
    p = Pipe()
    p.components = {"scheduler": object(), "tokenizer": None}
    fp = F.measure(p, {}, "video")
    assert not fp.complete
    assert fp.empty_load
    assert any(n.startswith("EMPTY_PIPELINE") for n in fp.notes), fp.notes


def test_an_unintrospectable_object_is_not_called_an_empty_load():
    """A GGUF wrapper is not a failed load -- it is a model we cannot walk, and
    its file size IS a legitimate answer."""
    class _ManagerLike:
        pass
    fp = F.measure(_ManagerLike(), {}, "text", file_size_gb=5.37)
    assert not fp.empty_load
    assert fp.complete
    assert fp.weights_gb == pytest.approx(5.37)


def test_a_failed_load_persists_nothing_and_says_why():
    """Deterministic, unlike a check on the resulting number: the previous
    figure must survive and the history must record the refusal. Asserting on
    process memory made this pass alone and fail in the full suite, because the
    fallback delta grows with whatever else the process has loaded."""
    m = _manager()
    key = "video:__guard_zero"
    m.config[key] = {"force_vram_update": True, "measured_vram_gb": 22.11}
    try:
        class Pipe:
            pass
        p = Pipe()
        p.components = {"scheduler": object()}     # nothing with weights
        m.models[key] = p
        m.active_in_vram = key
        m.record_vram_delta(key, free_before=10.0, ram_before=1.0)
        assert m.config[key].get("measured_vram_gb") == 22.11, (
            "a failed load overwrote the previous good measurement")
        last = m.get_history(limit=1)[0]
        assert last["model"] == key and "no weights" in last.get("reason", "")
    finally:
        m.models.pop(key, None)
        m.config.pop(key, None)


def test_the_rejection_names_which_check_fired():
    """Two different conditions reject a measurement -- an implausible absolute
    size and a collapse against the previous one -- and the log must say which,
    or it sends the reader after the wrong thing."""
    src = io.open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "codai/models/manager.py"), encoding="utf-8").read()
    blk = src.split("Reject a partial load", 1)[1][:2200]
    assert "not a plausible size for a loaded model" in blk
    assert "less than half the" in blk


def test_a_text_model_wrapper_is_walked_through_its_backend():
    """A text/vision model is registered as a ModelManager wrapper, so its real
    module hangs off .backend.model. Without reaching it, an HF LLM falls
    through to the process-delta fallback despite being perfectly walkable."""
    class _Backend:
        def __init__(self, m):
            self.model = m

    class _Wrapper:
        def __init__(self, m):
            self.backend = _Backend(m)

    inner = nn.Linear(1000, 1000, bias=False).to(torch.bfloat16)
    gpu, cpu, notes, ok = F.measure_weights(_Wrapper(inner))
    assert ok, notes
    assert cpu == pytest.approx(1000 * 1000 * 2 / 1e9, rel=1e-6)


def test_a_vision_projector_beside_the_lm_is_counted():
    """An mmproj loads ON TOP of the language model as a separate module."""
    class _Backend:
        def __init__(self, m):
            self.model = m

    class _Wrapper:
        def __init__(self, m, proj):
            self.backend = _Backend(m)
            self.mmproj = proj

    lm = nn.Linear(500, 500, bias=False).to(torch.bfloat16)
    proj = nn.Linear(200, 200, bias=False).to(torch.bfloat16)
    gpu, cpu, notes, ok = F.measure_weights(_Wrapper(lm, proj))
    assert ok
    assert cpu == pytest.approx((500 * 500 * 2 + 200 * 200 * 2) / 1e9, rel=1e-6)


def test_geometry_is_found_through_the_backend_wrapper_too():
    """Walking a wrapper's weights but missing its config would drop the KV
    term for the one model kind that definitely has one."""
    class _Cfg:
        num_hidden_layers = 32
        num_key_value_heads = 8
        head_dim = 128
        num_attention_heads = 32
        hidden_size = 4096

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 100)
            self.config = _Cfg()

    class _Backend:
        def __init__(self, m):
            self.model = m

    class _Wrapper:
        def __init__(self, m):
            self.backend = _Backend(m)

    w = _Wrapper(_Inner())
    geom = F.hf_geometry(w)
    assert geom.get("num_key_value_heads") == 8, geom
    fp = F.measure(w, {"n_ctx": 8192}, "text")
    assert fp.kv_gb > 0, f"the KV term was dropped: {fp.describe()}"


# ---------------------------------------------------------------------------
# A GGUF's weights total is the file; its PLACEMENT is not this helper's to say
# ---------------------------------------------------------------------------
#
# Measured on zeiss 2026-10-30. Gemma-4-14B-A4B-Heretic IQ4_XS, loaded with
# `n_gpu_layers: all` and holding 10.1 GB of VRAM by nvtop, described itself as
#
#     footprint: 8.48 GB total (weights 7.71 GB (0.00 on GPU, 7.71 on host), …)
#
# and the Tasks page called it "CPU offload 7.71 GB". The file size was banked
# in `weights_cpu_gb` -- which keeps the TOTAL right, and the total is what
# sizing consumes -- but nothing had measured a split, and "0.00 on GPU" is the
# opposite of the truth for a fully-resident model. It is also precisely the
# line an operator reads when chasing a CPU-spill problem.

def test_a_gguf_total_is_still_the_file_size():
    """The total is load-bearing for sizing and eviction; this must not change
    it. Under-reporting a footprint is how eviction stops freeing room."""
    fp = F.measure(None, cfg={"n_ctx": 32000}, model_type="text_models",
                 file_size_gb=7.71, measured_kv_gb=0.5)
    assert abs(fp.weights_gb - 7.71) < 0.001
    assert fp.total_gb > 7.71


def test_a_gguf_does_not_claim_a_placement():
    fp = F.measure(None, cfg={"n_ctx": 32000}, model_type="text_models",
                 file_size_gb=7.71, measured_kv_gb=0.5)
    assert fp.weights_split_known is False
    assert "0.00 on GPU" not in fp.describe()
    assert "on host" not in fp.describe()
    assert "placement not measured" in fp.describe()


def test_a_real_per_module_measurement_still_reports_its_split():
    """The case where the numbers ARE measured must keep saying so -- an
    actually-offloaded model is the thing this breakdown exists for."""
    fp = F.Footprint(weights_gpu_gb=3.0, weights_cpu_gb=1.0)
    assert fp.weights_split_known is True
    assert "(3.00 on GPU, 1.00 on host)" in fp.describe()


def test_the_on_disk_fallback_also_stops_claiming_a_split():
    """The other path to the same figure: an object that could not be walked,
    falling back to the file size."""
    class Unwalkable:
        pass

    fp = F.measure(Unwalkable(), cfg={"n_ctx": 4096}, model_type="text_models",
                   file_size_gb=5.0)
    if not fp.complete:
        pytest.skip("this object is rejected outright, not fallen back for")
    assert fp.weights_split_known is False
    assert "placement not measured" in fp.describe()


def test_the_manager_does_not_call_an_unsplit_total_an_offload():
    import io as _io
    import os as _os
    mgr = _io.open(_os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), "codai/models/manager.py"),
        encoding="utf-8").read()
    assert "ram_kind = 'unsplit'" in mgr
    assert "GPU/host split was NOT measured here" in mgr
    # And the Tasks page only states a SIZE for a genuine measurement.
    assert "if measured_ram and ram_kind is None:" in mgr
