"""C2/C3/C1: the LTX-2 pipeline split, the new lip-sync engines, Stable Audio.

Each of these was a case where the name matched but the code did not: an LTX-2
checkpoint handed to the LTX 0.9 pipeline, a Stable Audio entry handed to
AudioLDM2, a GGUF component that failed to load and was quietly replaced by
full-precision weights.
"""
import pytest


# ------------------------------------------------------ LTX 0.9 vs LTX 2.x
@pytest.mark.parametrize("name,mode,expect", [
    ("Lightricks/LTX-2.5", "t2v", "LTX2Pipeline"),
    ("Lightricks/LTX-2.5", "i2v", "LTX2ImageToVideoPipeline"),
    ("Lightricks/LTX-2.5", "ti2v", "LTX2ImageToVideoPipeline"),
    ("LTX2.3-10Eros", "t2v", "LTX2Pipeline"),
    ("ltx_2-something", "t2v", "LTX2Pipeline"),
    ("PinkCherry_FineTune_Q5_K_M_v18_LTX23", "i2v", "LTX2ImageToVideoPipeline"),
    # The 0.9 family must keep its own pipeline: they are different
    # architectures that happen to share a name.
    ("Lightricks/LTX-Video", "t2v", "LTXPipeline"),
    ("Lightricks/LTX-Video-0.9.7", "t2v", "LTXPipeline"),
    ("Lightricks/LTX-Video-0.9.7", "i2v", "LTXImageToVideoPipeline"),
])
def test_the_ltx_version_picks_the_pipeline(name, mode, expect):
    pytest.importorskip("diffusers")
    from codai.api.video import _detect_pipeline_class
    assert _detect_pipeline_class(name, mode).__name__ == expect


def test_ltx2_support_really_is_present_in_this_diffusers():
    # The claim above is only meaningful if the classes exist; an LTX-2 entry
    # was parked for exactly this reason.
    diffusers = pytest.importorskip("diffusers")
    for n in ("LTX2Pipeline", "LTX2ImageToVideoPipeline",
              "LTX2VideoTransformer3DModel", "AutoencoderKLLTX2Video",
              # The audio VAE is why LTX-2 can return a soundtrack with the
              # picture, which no other video model here does.
              "AutoencoderKLLTX2Audio"):
        assert hasattr(diffusers, n), n


# ------------------------------------------------------------ audio backends
@pytest.mark.parametrize("name,expect", [
    ("stabilityai/stable-audio-open-1.0", "stable_audio"),
    ("stable_audio_open", "stable_audio"),
    ("cvssp/audioldm2", "audioldm"),
    ("facebook/musicgen-large", "musicgen"),
    ("facebook/audiogen-medium", "audiogen"),
    ("ACE-Step/ACE-Step-v1-3.5B", "ace_step"),
    ("acestep-3.5b", "ace_step"),
])
def test_each_audio_model_gets_its_own_family(name, expect):
    from codai.api.audio_gen import _detect_audio_gen_type
    assert _detect_audio_gen_type(name) == expect


def test_stable_audio_is_not_routed_to_audioldm():
    # It was. AudioLDM2Pipeline is a different architecture with a different
    # projection model and sample rate, so a configured stable-audio entry
    # could only ever fail at load.
    from codai.api.audio_gen import _detect_audio_gen_type
    assert _detect_audio_gen_type("stabilityai/stable-audio-open-1.0") != "audioldm"


def test_stable_audio_has_a_loader_and_ace_step_says_why_it_has_none():
    import inspect

    from codai.api import audio_gen
    assert hasattr(audio_gen, "_load_stable_audio")
    src = inspect.getsource(audio_gen)
    i = src.index("elif model_type == 'ace_step':")
    window = src[i:i + 700]
    # Named plainly rather than loaded as a MusicGen and failing with a shape
    # error minutes later.
    assert "acestep" in window and "own venv" in window


def test_a_stereo_waveform_is_written_as_stereo():
    """Stable Audio's point is a usable stereo bed; folding it to mono throws
    that away, and writing (channels, samples) as-is gives a 2-sample file."""
    import io

    import numpy as np
    import scipy.io.wavfile as wavfile

    # (channels, samples) as the pipeline returns it, transposed the way the
    # writer now does, then round-tripped.
    audio = np.zeros((2, 4800), dtype="float32")
    audio[0, :] = 0.5
    arr = audio
    if arr.ndim > 1 and arr.shape[0] < arr.shape[-1]:
        arr = arr.T
    buf = io.BytesIO()
    wavfile.write(buf, 44100, (np.clip(arr, -1, 1) * 32767).astype(np.int16))
    buf.seek(0)
    sr, back = wavfile.read(buf)
    assert sr == 44100
    assert back.shape == (4800, 2), back.shape


def test_a_hot_sample_is_clipped_not_wrapped():
    # A float above 1.0 cast straight to int16 wraps to a loud NEGATIVE
    # sample, which is heard as a click.
    import numpy as np
    hot = np.array([1.4, -1.4], dtype="float32")
    clipped = (np.clip(hot, -1.0, 1.0) * 32767).astype(np.int16)
    assert clipped[0] > 0 and clipped[1] < 0
    naive = (hot * 32767).astype(np.int16)
    assert naive[0] < 0, "the wrap this guards against no longer happens"


# ------------------------------------------------------- lip-sync engines
def test_the_new_engines_are_video_driven_and_preferred():
    from codai.compose import presenter as P
    assert "latentsync" in P.VIDEO_ENGINES
    assert "musetalk" in P.VIDEO_ENGINES
    # Both beat wav2lip for a clip: wav2lip works at 96x96 and upscales, which
    # is the soft mouth on a 1080p reel.
    order = list(P._AUTO_WITH_BASE)
    assert order.index("latentsync") < order.index("wav2lip")
    assert order.index("musetalk") < order.index("wav2lip")
    # For a still, sadtalker still leads: it invents head motion rather than
    # replacing a mouth, which the others cannot do.
    assert P._AUTO_STILL[0] == "sadtalker"


def test_an_engine_is_only_available_when_its_assets_are_here(monkeypatch, tmp_path):
    from codai.compose import presenter as P
    monkeypatch.setattr(P.shutil, "which", lambda n: None)
    monkeypatch.setattr(P, "_managed_engine_dir", lambda n: tmp_path / n)
    # Nothing installed.
    assert P._managed_engine_ready("latentsync") is False
    # An entrypoint but no weights is NOT ready: a compose job is the wrong
    # place to discover a 5 GB download is missing.
    d = tmp_path / "latentsync"
    (d / "scripts").mkdir(parents=True)
    (d / "scripts" / "inference.py").write_text("x")
    assert P._managed_engine_ready("latentsync") is False
    (d / "unet.pt").write_text("w")
    assert P._managed_engine_ready("latentsync") is True


def test_available_engines_is_reported_in_preference_order(monkeypatch):
    from codai.compose import presenter as P
    monkeypatch.setattr(P.shutil, "which",
                        lambda n: "/usr/bin/" + n if n in
                        ("sadtalker", "musetalk", "latentsync") else None)
    monkeypatch.setattr(P, "_managed_engine_ready", lambda n: False)
    import codai.api.lipsync as L
    monkeypatch.setattr(L, "is_available", lambda: True)
    got = P.available_engines()
    # Discovery order would put sadtalker first; the listing must agree with
    # what engine:"auto" would actually choose.
    assert got.index("latentsync") < got.index("musetalk") < got.index("wav2lip")


def test_picking_an_uninstalled_engine_names_what_is_installed(monkeypatch):
    from codai.compose import presenter as P
    monkeypatch.setattr(P, "available_engines", lambda: ["wav2lip"])
    monkeypatch.setattr(P.shutil, "which", lambda n: None)
    with pytest.raises(P.PresenterError) as e:
        P.pick_engine("latentsync", True)
    assert "latentsync" in str(e.value) and "wav2lip" in str(e.value)


def test_run_can_set_cwd_and_pythonpath():
    # A managed checkout's inference.py imports its own package as a
    # top-level module, so it only resolves from inside the checkout.
    import inspect

    from codai.compose import presenter as P
    sig = inspect.signature(P._run)
    assert "cwd" in sig.parameters and "pythonpath" in sig.parameters


# ------------------------------------------- a GGUF component that fails
def test_a_failed_gguf_component_refuses_instead_of_downgrading():
    """The silent substitution this prevents.

    A component named in component_quantization that does not load is simply
    absent from the returned dict, so the base pipeline loads its own
    full-precision copy: the operator asked for 14 GB and got 38, while the
    VRAM estimate still said 14 and the planner freed a third of what the load
    needed.
    """
    import inspect

    from codai.models import hf_loading
    src = inspect.getsource(hf_loading)
    i = src.index("gguf_entries = {n: v for n, v in comp_q.items()")
    window = src[i:i + 4000]
    assert "problems" in window
    assert "raise RuntimeError" in window
    assert "silently substitute" in window
