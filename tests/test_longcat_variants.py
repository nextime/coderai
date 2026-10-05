"""Step 6: weight variants and the audio-driven avatar tasks.

Read from upstream's avatar demo rather than assumed, because almost nothing carries over
from the base model:

- avatar is a DIFFERENT pipeline class, ``LongCatVideoAvatarPipeline``, with its own
  methods ``generate_at2v`` / ``generate_ai2v`` / ``generate_avc``;
- it takes TWO guidance scales (text and audio) where the base pipeline takes one;
- its distilled pass is ``dmd_lora`` at **8** steps, not ``cfg_step_lora`` at 16;
- ``use_int8`` is a separate ``load_quantized_dit(subfolder="base_model_int8")``, and both
  it and the DMD LoRA exist only for avatar-1.5 — which also *requires* distilled sampling.

Asking for an unsupported combination has to be an error, because the alternative is
quietly loading something that is not what was asked for.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LC = _mod("longcat_common", "tools/longcat_common.py")
SVC = _mod("longcat_service", "tools/longcat_service.py")
VIDEO = (ROOT / "codai/api/video.py").read_text()
SERVICE_SRC = (ROOT / "tools/longcat_service.py").read_text()


# ------------------------------------------------------------ family detection
@pytest.mark.parametrize("path,family", [
    ("meituan-longcat/LongCat-Video", ""),
    ("/w/LongCat-Video-Avatar", "avatar"),
    ("/w/LongCat-Video-Avatar-1.5", "avatar-1.5"),
    ("/w/longcat_video_avatar_1.5", "avatar-1.5"),
    ("szwagros/LongCat-Video-Avatar-1.5-fp8", "avatar-1.5"),
])
def test_the_family_is_read_from_the_path(path, family):
    """The family IS the repo — three separate downloads with different audio encoders."""
    assert LC.family_of(path) == family


def test_each_family_has_its_own_audio_encoder():
    assert LC.AVATAR_ENCODERS["avatar"] == "chinese-wav2vec2-base"
    assert LC.AVATAR_ENCODERS["avatar-1.5"] == "whisper-large-v3"


# ------------------------------------------------------------ variant gating
def test_int8_is_refused_on_the_base_model():
    probs = LC.variant_problems("int8", "")
    assert probs and "avatar-1.5" in probs[0]


def test_int8_is_accepted_on_avatar_15():
    assert LC.variant_problems("int8", "avatar-1.5", use_int8=True) == []


def test_the_dmd_distillation_is_refused_on_the_plain_avatar():
    probs = LC.variant_problems("bf16", "avatar", use_distill=True)
    assert any("DMD" in p for p in probs)


def test_avatar_15_requires_distilled_sampling():
    """Upstream states it; without the flag the 8-step DMD path is not taken."""
    probs = LC.variant_problems("bf16", "avatar-1.5")
    assert probs and "requires distilled sampling" in probs[0]
    assert LC.variant_problems("bf16", "avatar-1.5", use_distill=True) == []


def test_gguf_is_refused_with_the_reason():
    """They are packaged for ComfyUI's gguf loader, which is not implemented."""
    probs = LC.variant_problems("gguf", "")
    assert probs and "ComfyUI" in probs[0]


def test_an_unknown_variant_is_refused():
    assert LC.variant_problems("int4", "")


def test_bf16_on_the_base_model_needs_nothing():
    assert LC.variant_problems("bf16", "") == []


def test_fp8_is_allowed_everywhere_it_is_published():
    assert LC.variant_problems("fp8", "") == []
    assert LC.variant_problems("fp8", "avatar-1.5", use_distill=True) == []


# ------------------------------------------------------------ avatar checkpoint checks
def test_int8_needs_its_subfolder(tmp_path):
    probs = LC.avatar_problems(str(tmp_path), "avatar-1.5", use_int8=True)
    assert probs and LC.INT8_SUBDIR in probs[0]
    (tmp_path / LC.INT8_SUBDIR).mkdir()
    assert LC.avatar_problems(str(tmp_path), "avatar-1.5", use_int8=True) == []


def test_the_dmd_lora_must_be_present(tmp_path):
    probs = LC.avatar_problems(str(tmp_path), "avatar-1.5", use_distill=True)
    assert probs and "dmd_lora" in probs[0]


def test_a_non_avatar_checkpoint_cannot_serve_the_audio_tasks():
    probs = LC.avatar_problems("/w/LongCat-Video", "")
    assert probs and "audio-driven" in probs[0]


# ------------------------------------------------------------ the avatar step table
def test_the_avatar_distilled_pass_is_eight_steps_not_sixteen():
    """The base model's distilled pass is 16 steps against cfg_step_lora; the avatar's is
    8 against dmd_lora. Using the base numbers would run it wrong."""
    assert LC.avatar_stage_params("distill")["num_inference_steps"] == 8
    assert LC.stage_params("distill")["num_inference_steps"] == 16


def test_the_avatar_pass_has_two_guidance_scales():
    p = LC.avatar_stage_params("base")
    assert "text_guidance_scale" in p and "audio_guidance_scale" in p
    assert "guidance_scale" not in p
    assert LC.avatar_stage_params("distill")["audio_guidance_scale"] == 1.0


def test_the_dmd_lora_dimensions_are_pinned():
    assert (LC.DMD_NETWORK_DIM, LC.DMD_NETWORK_ALPHA) == (128, 64)


# ------------------------------------------------------------ the service calls
class _AvatarPipe:
    def __init__(self):
        self.calls = []

    def _out(self):
        import numpy as np
        return [np.zeros((4, 8, 8, 3), dtype="float32")]

    def generate_at2v(self, **kw):
        self.calls.append(("at2v", kw)); return self._out()

    def generate_ai2v(self, **kw):
        self.calls.append(("ai2v", kw)); return self._out()

    def generate_avc(self, **kw):
        self.calls.append(("avc", kw)); return self._out()


def _ctx(**kw):
    base = {"prompt": "speak", "negative_prompt": "", "height": 480, "width": 832,
            "num_frames": 93, "num_cond_frames": 13, "audio_emb": "EMB",
            "image": None, "generator": None, "use_distill": False,
            "num_inference_steps": None, "text_guidance_scale": None,
            "audio_guidance_scale": None, "offload_kv_cache": False,
            "ref_img_index": None, "mask_frame_range": None, "stage": "base"}
    base.update(kw)
    return base


def test_at2v_is_called_for_audio_only():
    p = _AvatarPipe()
    SVC._avatar_pass(p, "at2v", _ctx())
    name, kw = p.calls[0]
    assert name == "at2v"
    assert kw["audio_emb"] == "EMB" and kw["output_type"] == "both"
    assert "text_guidance_scale" in kw and "audio_guidance_scale" in kw
    assert "guidance_scale" not in kw


def test_ai2v_passes_the_reference_image():
    p = _AvatarPipe()
    SVC._avatar_pass(p, "ai2v", _ctx(image="REF"))
    name, kw = p.calls[0]
    assert name == "ai2v" and kw["image"] == "REF"


def test_continuation_uses_avc_with_the_avatar_options():
    p = _AvatarPipe()
    SVC._avatar_pass(p, "at2v", _ctx(ref_img_index=4, mask_frame_range="0:10"),
                     cond=["prev"])
    name, kw = p.calls[0]
    assert name == "avc"
    assert kw["video"] == ["prev"] and kw["num_cond_frames"] == 13
    assert kw["use_kv_cache"] is True
    assert kw["ref_img_index"] == 4 and kw["mask_frame_range"] == "0:10"


def test_the_distill_flag_is_forwarded():
    p = _AvatarPipe()
    SVC._avatar_pass(p, "at2v", _ctx(use_distill=True, stage="distill"))
    _n, kw = p.calls[0]
    assert kw["use_distill"] is True
    assert kw["num_inference_steps"] == 8


def test_avatar_tasks_route_through_one_pass():
    p = _AvatarPipe()
    SVC._one_pass(p, "at2v", "base", _ctx())
    assert p.calls and p.calls[0][0] == "at2v"


# ------------------------------------------------------------ loading
def test_the_avatar_pipeline_class_is_used():
    assert "LongCatVideoAvatarPipeline" in SERVICE_SRC
    assert "pipeline_longcat_video_avatar" in SERVICE_SRC


def test_int8_uses_the_repos_quantised_loader():
    assert "load_quantized_dit" in SERVICE_SRC
    assert "base_model_int8" in (LC.INT8_SUBDIR,)


def test_fp8_loads_as_that_dtype_and_says_so():
    """Community FP8 layouts differ between publishers, so this must fail with the real
    reason rather than silently produce noise."""
    seg = SERVICE_SRC[SERVICE_SRC.index('== "fp8"'):]
    assert "float8_e4m3fn" in seg[:600]


def test_the_audio_is_resampled_to_the_encoders_rate():
    assert "sr=16000" in SERVICE_SRC
    assert "get_audio_embedding" in SERVICE_SRC


def test_librosa_is_a_declared_dependency():
    """The house rule: a new pip dependency goes in the requirements file, not only in
    the code that imports it."""
    req = (ROOT / "requirements-longcat.txt").read_text()
    assert "librosa" in req


# ------------------------------------------------------------ dispatch
def test_the_avatar_modes_are_mapped():
    seg = VIDEO[VIDEO.index('_task = {"t2v"'):VIDEO.index("if _task is None")]
    for mode, task in (("at2v", "at2v"), ("ai2v", "ai2v"), ("avatar", "at2v")):
        assert f'"{mode}": "{task}"' in seg, mode


def test_the_audio_comes_from_audio_file_not_the_add_audio_boolean():
    """add_audio is a BOOLEAN for the post-processing path; using it would pass True
    into the base64 encoder."""
    fn = VIDEO[VIDEO.index("async def _generate_longcat"):]
    fn = fn[:fn.index("\nasync def ")]
    assert "request.audio_file" in fn
    assert "add_audio" not in fn.split("_audio =")[1][:200]


def test_an_avatar_request_without_audio_is_refused():
    fn = VIDEO[VIDEO.index("async def _generate_longcat"):]
    fn = fn[:fn.index("\nasync def ")]
    seg = fn[fn.index('if _task in ("at2v"'):]
    assert "status_code=400" in seg[:900]


def test_the_variant_flags_are_forwarded_to_the_service():
    fn = VIDEO[VIDEO.index("async def _generate_longcat"):]
    fn = fn[:fn.index("\nasync def ")]
    for key in ("variant", "use_int8", "use_distill", "ref_img_index",
                "mask_frame_range"):
        assert f'"{key}"' in fn, key


def test_the_request_model_carries_the_avatar_scales():
    from codai.pydantic.videorequest import VideoGenerationRequest as R
    r = R(model="m", prompt="p", text_guidance_scale=4.0, audio_guidance_scale=3.0)
    assert (r.text_guidance_scale, r.audio_guidance_scale) == (4.0, 3.0)
