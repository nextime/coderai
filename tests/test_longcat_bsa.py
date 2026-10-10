"""Block-sparse attention, and the cp_split_hw pair the DiT cannot run without.

At video token counts attention dominates: ~37k tokens for 93 frames at 480x832, and
attention is O(N^2) in that. The checkpoint already carries bsa_params with sparsity
0.9375 — roughly a 16x cut — and upstream's implementation is pure Triton, so no
flash-attn source build is involved. It ships disabled.

cp_split_hw is in here because it is the same kind of defect: a DiT construction
argument whose checkpoint value cannot be used as shipped.
"""
import importlib.util
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVICE = ROOT / "tools" / "longcat_service.py"
WORKER = ROOT / "codai" / "api" / "longcat_worker.py"


@pytest.fixture
def service(monkeypatch):
    spec = importlib.util.spec_from_file_location("lc_service_bsa", SERVICE)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    monkeypatch.delenv("LONGCAT_BSA", raising=False)
    return mod


# ------------------------------------------------------------------ cp_split_hw
# LongCatVideoTransformer3DModel.forward does
#   if self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
# with no None guard, while dit/config.json ships "cp_split_hw": null.

def test_the_default_is_a_pair_not_none(service):
    """None is what the checkpoint gives, and it raises TypeError on the first forward."""
    assert service._cp_split_hw() == [1, 1]


def test_without_context_parallelism_the_split_is_inert(service):
    h, w = service._cp_split_hw(1)
    assert h * w == 1, "anything else takes the split_cp_2d branch on one GPU"


def test_the_process_count_is_not_the_split(service):
    """The bug: cp_size was stored straight into cp_split_hw, so the DiT indexed an int."""
    pair = service._cp_split_hw(2)
    assert isinstance(pair, list) and len(pair) == 2
    assert pair[0] * pair[1] == 2


@pytest.mark.parametrize("text,want", [("1x2", [1, 2]), ("2,1", [2, 1]), ("1x4", [1, 4])])
def test_an_explicit_tile_is_honoured(service, text, want):
    assert service._cp_split_hw(want[0] * want[1], text) == want


def test_a_tile_that_does_not_cover_the_group_is_rejected(service):
    """2x2 under cp_size 2 leaves two ranks with nothing and two tiles unowned."""
    with pytest.raises(ValueError, match="every rank must own one tile"):
        service._cp_split_hw(2, "2x2")


@pytest.mark.parametrize("bad", ["1", "0x2", "1x2x3", "-1x2"])
def test_a_malformed_tile_is_rejected(service, bad):
    with pytest.raises(ValueError):
        service._cp_split_hw(4, bad)


# ------------------------------------------------------------------ BSA

def test_bsa_is_off_unless_asked_for(service):
    """It is an approximation, and video shows approximation as temporal flicker."""
    assert service._bsa_enabled() is False
    assert service._bsa_enabled("off") is False


@pytest.mark.parametrize("on", ["on", "true", "1", "yes", "auto"])
def test_bsa_turns_on_when_triton_is_there(service, on):
    pytest.importorskip("triton")
    assert service._bsa_enabled(on) is True


def test_auto_stays_off_without_triton_but_on_is_an_error(service, monkeypatch):
    """"auto" means "if you can"; "on" means the user asked, so failing quietly would
    leave them wondering why nothing got faster."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda *a, **k: None)
    assert service._bsa_enabled("auto") is False
    with pytest.raises(RuntimeError, match="needs triton"):
        service._bsa_enabled("on")


def test_a_nonsense_value_is_rejected(service):
    with pytest.raises(ValueError, match="bsa must be"):
        service._bsa_enabled("sparse-ish")


def test_bsa_joins_the_dense_backend_rather_than_replacing_it(service, monkeypatch):
    """BSA covers SELF-attention only. _process_cross_attn has no enable_bsa branch and
    ends in `raise RuntimeError("Unsupported attention operations.")`, so turning the
    dense backend off to let BSA have the field kills every cross-attention call — which
    is exactly what happened on the first real run."""
    pytest.importorskip("triton")
    monkeypatch.setenv("LONGCAT_BSA", "on")

    kwargs = service._dit_kwargs()

    assert kwargs["enable_bsa"] is True
    assert (kwargs["enable_xformers"] or kwargs["enable_flashattn2"]
            or kwargs["enable_flashattn3"]), "cross-attention still needs a dense path"


def test_exactly_one_dense_backend_remains_under_bsa(service, monkeypatch):
    pytest.importorskip("triton")
    monkeypatch.setenv("LONGCAT_BSA", "on")

    kwargs = service._dit_kwargs()

    dense = sum(bool(kwargs[k]) for k in ("enable_xformers", "enable_flashattn2",
                                          "enable_flashattn3"))
    assert dense == 1


def test_the_dit_kwargs_always_carry_a_usable_split(service):
    assert service._dit_kwargs()["cp_split_hw"] == [1, 1]


def test_both_load_paths_build_from_the_same_overrides(service):
    """bf16 reads dit/config.json, INT8 reads the copy beside the quantised weights;
    both are wrong about this machine in the same two ways."""
    src = SERVICE.read_text(encoding="utf-8")
    body = src[src.index("def load_pipeline"):src.index("def _cp_split_hw")]
    assert body.count("_dit_kwargs()") == 2


# ------------------------------------------------------------------ plumbing

def test_the_worker_passes_bsa_and_the_tile_through(service):
    src = WORKER.read_text(encoding="utf-8")
    assert '"--bsa"' in src
    assert '"--cp-split-hw"' in src


def test_a_model_can_override_the_server_wide_setting(service):
    """Per-model first: one checkpoint being too lossy under BSA must not force it off
    for every other model on the box."""
    src = WORKER.read_text(encoding="utf-8")
    body = src[src.index('bsa = str(config.get("bsa")'):]
    assert body.index('config.get("bsa")') < body.index('getattr(sec, "bsa"')


# ------------------------------------------------------------------ geometry
# flash_attn_bsa_3d tiles the latent into 3D chunks and asserts it divides evenly:
#   assert Tq % tq == 0 and Hq % hq == 0 and Wq % wq == 0
# With the shipped (4,4,4) that is sides divisible by 64 and a latent depth divisible
# by 4. Upstream's own default 480x832 fails it, which is how a bare AssertionError
# came out of the first denoising step.

@pytest.fixture
def LC():
    spec = importlib.util.spec_from_file_location(
        "lc_common_bsa", ROOT / "tools" / "longcat_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("frames,h,w", [(93, 512, 832), (93, 448, 832), (29, 512, 832),
                                        (13, 64, 64), (45, 512, 1280), (93, 704, 1280)])
def test_a_tileable_geometry_passes(LC, frames, h, w):
    assert LC.bsa_problems(frames, h, w) == []


def test_the_shipped_default_geometry_is_not_tileable(LC):
    """480/16 = 30, and 30 % 4 != 0. This is the real failure, reproduced."""
    problems = LC.bsa_problems(93, 480, 832)
    assert problems
    assert any("height" in p and "64" in p for p in problems)


def test_the_frame_count_we_tested_with_is_caught_too(LC):
    """33 frames gives a latent depth of 9, and 9 % 4 != 0."""
    problems = LC.bsa_problems(33, 512, 832)
    assert any("num_frames" in p for p in problems)


def test_the_message_suggests_something_that_actually_works(LC):
    """"invalid" without a usable number means guessing at multiples of the VAE stride."""
    import re
    for frames, h, w in [(33, 480, 832), (50, 500, 900)]:
        problems = LC.bsa_problems(frames, h, w)
        suggested = [int(n) for p in problems for n in re.findall(r"try (\d+)", p)]
        assert suggested, f"no suggestion for {frames} {h}x{w}"
        # every suggested side must itself be tileable
        for s in suggested:
            assert s % 64 == 0 or ((s - 1) // 4 + 1) % 4 == 0, s


def test_a_custom_chunk_changes_the_requirement(LC):
    """The chunk comes from the checkpoint's bsa_params, not a constant here."""
    assert LC.bsa_problems(93, 480, 832, chunk=(4, 2, 2)) == []   # 30 % 2 == 0
    assert LC.bsa_problems(93, 480, 832, chunk=(4, 4, 4)) != []


def test_non_numeric_geometry_is_reported_not_raised(LC):
    assert LC.bsa_problems("x", None, 832)


def test_the_service_falls_back_instead_of_asserting(service):
    """Turning bsa on in a config must not break requests that already work: the pass
    runs dense and logs why, rather than dying inside the first step."""
    src = SERVICE.read_text(encoding="utf-8")
    body = src[src.index("def _bsa_for"):src.index("def _place_pipeline")]
    assert "LC.bsa_problems" in body
    assert "enable_bsa = False" in body
    assert "enable_bsa = True" in body, "it must be restored for the next request"
    assert "WARNING" in body, "a silent fallback hides that nothing got faster"
    # and it is actually applied around the generation
    assert "with _bsa_for(pipe, ctx, task):" in src


# ──────────────────────────────── the conditioning branch ─────────────────────
# A geometry can be perfectly tileable and still assert, because a conditioning pass
# tiles the cond and noise blocks SEPARATELY (modules/attention.py:123-134). This is
# what killed a 93-frame i2v run with 64-divisible sides: the guard checked the noise
# depth, passed, and flash_attn_bsa_3d asserted on Tq=1 in the first step.

def test_the_i2v_hardcode_can_never_tile(LC):
    """generate_i2v passes num_cond_latents=1, and 1 % 4 != 0 under any frame count."""
    for frames in (13, 29, 45, 61, 77, 93, 173):
        problems = LC.bsa_problems(frames, 512, 832,
                                   num_cond_latents=LC.I2V_COND_LATENTS)
        assert any("conditioning" in p for p in problems), frames


def test_a_pass_with_no_conditioning_is_unaffected(LC):
    """t2v has no cond branch, so the old answer must not change."""
    assert LC.bsa_problems(93, 512, 832, num_cond_latents=0) == []
    assert LC.bsa_problems(93, 512, 832, num_cond_latents=None) == []
    assert LC.bsa_problems(93, 512, 832) == []


def test_both_halves_of_the_split_must_tile(LC):
    """Latent depth 24: 4 cond + 20 noise tiles; 8 cond + 16 noise tiles; 6 does not."""
    frames = 93                                   # (93-1)/4 + 1 = 24 latent frames
    assert LC.bsa_problems(frames, 512, 832, num_cond_latents=4) == []
    assert LC.bsa_problems(frames, 512, 832, num_cond_latents=8) == []
    assert LC.bsa_problems(frames, 512, 832, num_cond_latents=6) != []


def test_a_conditioning_block_that_swallows_the_noise_is_rejected(LC):
    """Padding must not be allowed to leave zero noise latents to denoise."""
    assert LC.bsa_problems(13, 512, 832, num_cond_latents=4) != []   # depth 4, noise 0


def test_the_message_names_the_way_out(LC):
    problems = LC.bsa_problems(93, 512, 832, num_cond_latents=1)
    joined = " ".join(problems)
    assert "bsa_pad_cond" in joined
    assert "4" in joined


def test_the_continuation_count_matches_the_pipeline(LC):
    """pipeline_longcat_video.py:1016 — 1 + (num_cond_frames - 1) // 4."""
    assert LC.cond_latents_for(1) == 1
    assert LC.cond_latents_for(13) == 4
    assert LC.cond_latents_for(0) == 0
    assert LC.cond_latents_for(None) == 0


def test_rounding_matches_what_generate_refine_does(LC):
    """pipeline_longcat_video.py:1245 — ceil(n / granularity) * granularity."""
    assert LC.round_to_granularity(1) == 4
    assert LC.round_to_granularity(4) == 4
    assert LC.round_to_granularity(5) == 8
    assert LC.round_to_granularity(3, chunk=(2, 4, 4)) == 4


# ───────────────────────── (a) the guard refuses a cond pass ──────────────────

def test_an_i2v_request_counts_one_conditioning_latent(service):
    assert service._cond_latent_passes("i2v", {"num_frames": 93}) == [1]


def test_a_t2v_request_counts_none(service):
    assert service._cond_latent_passes("t2v", {"num_frames": 93}) == [0]


def test_multiple_segments_add_a_continuation_pass(service):
    """Segment 0 is t2v, every later one is generate_vc — one DiT, one BSA flag."""
    counts = service._cond_latent_passes(
        "t2v", {"num_frames": 93, "num_segments": 3, "num_cond_frames": 13})
    assert 0 in counts and 4 in counts


def test_a_continuation_request_uses_its_cond_frames(service):
    counts = service._cond_latent_passes(
        "vc", {"num_frames": 93, "num_cond_frames": 13})
    assert counts == [4]


def test_an_avatar_continuation_allows_for_the_reference_image(service):
    """pipeline_longcat_video_avatar.py:1375 adds one for a ref image."""
    counts = service._cond_latent_passes(
        "ai2v", {"num_frames": 93, "num_cond_frames": 13})
    assert 5 in counts


def test_the_guard_checks_every_pass_the_request_will_run(service):
    src = SERVICE.read_text(encoding="utf-8")
    body = src[src.index("def _bsa_for"):src.index("def _place_pipeline")]
    assert "_cond_latent_passes" in body
    assert "num_cond_latents=cond" in body
    assert "with _bsa_for(pipe, ctx, task):" in src, "the pass kind must reach the guard"


# ───────────────────────── (b) padding, off by default ────────────────────────

def test_padding_is_off_by_default(service, monkeypatch):
    monkeypatch.delenv("LONGCAT_BSA_PAD_COND", raising=False)
    assert service._bsa_pad_cond() is False


@pytest.mark.parametrize("value", ["", "off", "no", "false", "0", "none"])
def test_every_spelling_of_off_is_off(service, value):
    assert service._bsa_pad_cond(value) is False


@pytest.mark.parametrize("value", ["on", "yes", "true", "1"])
def test_it_can_be_turned_on_explicitly(service, value):
    assert service._bsa_pad_cond(value) is True


def test_a_nonsense_setting_is_rejected_before_the_load(service):
    """A bad value must not surface after a multi-minute checkpoint load."""
    with pytest.raises(ValueError):
        service._bsa_pad_cond("maybe")
    src = SERVICE.read_text(encoding="utf-8")
    assert "_bsa_pad_cond()" in src[src.index("_bsa_enabled()\n    _bsa_pad_cond()"):]


def test_the_env_var_is_read(service, monkeypatch):
    monkeypatch.setenv("LONGCAT_BSA_PAD_COND", "on")
    assert service._bsa_pad_cond() is True


def test_with_padding_on_the_guard_checks_the_padded_count(service, monkeypatch):
    """1 cond latent cannot tile; padded to 4 against a depth-24 latent, it can."""
    monkeypatch.setenv("LONGCAT_BSA_PAD_COND", "on")
    pipe, dit = _fake_pipe()
    with service._bsa_for(pipe, {"num_frames": 93, "height": 512, "width": 832},
                          "i2v") as on:
        assert on is True, "padding is the whole point: BSA must survive an i2v pass"


def test_with_padding_off_an_i2v_pass_falls_back_to_dense(service, monkeypatch):
    monkeypatch.delenv("LONGCAT_BSA_PAD_COND", raising=False)
    pipe, dit = _fake_pipe()
    with service._bsa_for(pipe, {"num_frames": 93, "height": 512, "width": 832},
                          "i2v") as on:
        assert on is False
        assert dit.block.enable_bsa is False, "it must actually be turned off"
    assert dit.block.enable_bsa is True, "and restored for the next request"


def test_padding_is_not_installed_unless_both_flags_are_on(service):
    """bsa off + padding on must not patch the DiT: there is nothing to pad for."""
    src = SERVICE.read_text(encoding="utf-8")
    assert "_bsa_enabled() and _bsa_pad_cond() and _install_cond_padding(pipe)" in src


def test_the_wrapper_rounds_the_count_on_the_way_into_the_dit(service):
    pipe, dit = _fake_pipe()
    assert service._install_cond_padding(pipe) is True
    pipe.dit(hidden_states=_FakeLatent(depth=24), num_cond_latents=1)
    assert dit.seen[-1] == 4


def test_the_wrapper_is_idempotent(service):
    """A reload must not stack wrappers, each rounding the previous one's answer."""
    pipe, dit = _fake_pipe()
    assert service._install_cond_padding(pipe) is True
    assert service._install_cond_padding(pipe) is False


def test_the_wrapper_leaves_a_pass_with_no_conditioning_alone(service):
    pipe, dit = _fake_pipe()
    service._install_cond_padding(pipe)
    pipe.dit(hidden_states=_FakeLatent(depth=24), num_cond_latents=0)
    assert dit.seen[-1] == 0
    pipe.dit(hidden_states=_FakeLatent(depth=24))
    assert dit.seen[-1] is None


def test_the_wrapper_does_not_swallow_the_noise_block(service):
    """Depth 4 padded to 4 leaves nothing to denoise — hand the real count through."""
    pipe, dit = _fake_pipe()
    service._install_cond_padding(pipe)
    pipe.dit(hidden_states=_FakeLatent(depth=4), num_cond_latents=1)
    assert dit.seen[-1] == 1


def test_the_wrapper_uses_the_checkpoints_own_chunk(service):
    pipe, dit = _fake_pipe(chunk=[2, 4, 4])
    service._install_cond_padding(pipe)
    pipe.dit(hidden_states=_FakeLatent(depth=24), num_cond_latents=1)
    assert dit.seen[-1] == 2


def test_the_guard_wraps_only_the_segment_loop(service):
    """generate_refine rounds both counts itself (pipeline:1245-1250), so it is safe
    with BSA on and must not be forced down the dense path. The guard therefore wraps
    _run_segments and nothing else."""
    src = SERVICE.read_text(encoding="utf-8")
    sites = [m.start() for m in re.finditer(r"with _bsa_for\(", src)]
    assert len(sites) == 1, "one wrap point, or a pass can slip past the guard"
    after = src[sites[0]:].split("\n", 2)[1]
    assert "_run_segments(" in after


# ────────────────────────────── config plumbing ───────────────────────────────

def test_the_worker_passes_the_flag_through():
    src = WORKER.read_text(encoding="utf-8")
    assert '"bsa_pad_cond": "off"' in src, "it must be documented AND default off"
    assert '"--bsa-pad-cond"' in src


def test_the_admin_save_does_not_drop_it():
    """A key the whitelist does not name is silently lost on every GUI save."""
    routes = (ROOT / "codai" / "admin" / "routes.py").read_text(encoding="utf-8")
    assert '"bsa_pad_cond"' in routes
    html = (ROOT / "codai" / "admin" / "templates"
            / "models.html").read_text(encoding="utf-8")
    assert "cfg-longcat-bsa-pad-cond" in html
    assert "out.bsa_pad_cond" in html
    assert "s.bsa_pad_cond" in html


# ───────────────────────────────── fakes ──────────────────────────────────────

class _FakeLatent:
    """Stands in for the [B, C, T, H, W] latent the DiT is handed."""

    def __init__(self, depth):
        self.shape = (1, 16, depth, 32, 52)
        self.ndim = 5


class _FakeBlock:
    def __init__(self, chunk):
        self.enable_bsa = True
        self.bsa_params = {"chunk_3d_shape_q": list(chunk),
                           "chunk_3d_shape_k": list(chunk)}


class _FakeDit:
    def __init__(self, chunk):
        self.block = _FakeBlock(chunk)
        self.seen = []

    def modules(self):
        return [self, self.block]

    def forward(self, *args, **kwargs):
        self.seen.append(kwargs.get("num_cond_latents"))
        return None

    def __call__(self, *args, **kwargs):
        # nn.Module._call_impl reads self.forward, which is what the wrapper rebinds.
        return self.forward(*args, **kwargs)


class _FakePipe:
    def __init__(self, dit):
        self.dit = dit


def _fake_pipe(chunk=(4, 4, 4)):
    dit = _FakeDit(chunk)
    return _FakePipe(dit), dit
