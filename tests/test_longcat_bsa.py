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


def test_bsa_replaces_the_dense_backend_rather_than_joining_it(service, monkeypatch):
    """Attention checks enable_bsa first, so leaving xformers on too would be a silent
    second opinion about which path is running."""
    pytest.importorskip("triton")
    monkeypatch.setenv("LONGCAT_BSA", "on")

    kwargs = service._dit_kwargs()

    assert kwargs["enable_bsa"] is True
    assert not kwargs["enable_xformers"]
    assert not kwargs["enable_flashattn2"]
    assert not kwargs["enable_flashattn3"]


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
    assert "with _bsa_for(pipe, ctx):" in src
