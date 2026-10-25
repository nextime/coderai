"""A published model id can be a base checkpoint plus its own LoRAs, and can be
one checkpoint FILE rather than a repo.

Two gaps found while serving a request for community fine-tunes.

**Per-model LoRA stacks.** A diffusion pipeline could carry an acceleration
LoRA (from the ``acceleration`` block) and per-request LoRAs (sent by the
caller), but nothing in between: there was no way to say "this id is that base
plus this style LoRA at 0.8". The field ``lora_path`` existed in the model
config and in the admin UI, and on the diffusers image path nothing read it, so
configuring a LoRA on an image model did precisely nothing and said so nowhere.

Worse, ``set_adapters`` replaces the whole active adapter set, and the image
path called it with only the current request's adapters — so a request that
carried a LoRA silently switched the acceleration LoRA off, leaving a 4-step
preset running on an un-distilled model. The video path had always re-included
them.

**Single-file checkpoints.** ``from_single_file`` was wired up only for GGUF
*components*. A whole-pipeline checkpoint — which is how Civitai ships every
community fine-tune — could not be opened at all: ``from_pretrained`` wants a
directory tree. The model that prompted this (CyberRealistic Pony v18) has
converted copies on the Hub only up to v14.1, so "wait for someone to convert
it" is not a plan.
"""
import types

import pytest

from codai.api import images as images_mod
from codai.models import diffusion_loras as dl
from codai.models import hf_loading
from codai.models.text_loras import configured_specs


# --------------------------------------------------------------------------- #
#  a pipeline that records what was done to it
# --------------------------------------------------------------------------- #

class FakePipe:
    def __init__(self, fail_on=(), has_lora=True):
        self.loaded = []           # (repo, kwargs)
        self.active = None         # (names, weights) of the last set_adapters
        self.deleted = []
        self._fail_on = set(fail_on)
        if not has_lora:
            del FakePipe.load_lora_weights   # pragma: no cover - see subclass

    def load_lora_weights(self, repo, **kw):
        if repo in self._fail_on:
            raise RuntimeError(f"no such adapter: {repo}")
        self.loaded.append((repo, kw))

    def set_adapters(self, names, weights):
        self.active = (list(names), list(weights))

    def delete_adapters(self, names):
        self.deleted.append(names)


class NoLoraPipe:
    """A pipeline class that cannot take LoRAs at all."""


def _cfg(**kw):
    return kw


# --------------------------------------------------------------------------- #
#  which adapters does this model want
# --------------------------------------------------------------------------- #

def test_a_model_can_name_several_loras_with_weights():
    specs = dl.configured(_cfg(loras=[
        {"model": "org/style-lora", "weight": 0.8, "name": "style"},
        {"model": "org/detail-lora", "weight": 0.5},
    ]))
    assert [(s["source"], s["weight"], s["name"]) for s in specs] == [
        ("org/style-lora", 0.8, "style"),
        ("org/detail-lora", 0.5, "detail-lora"),
    ]


def test_the_long_ignored_lora_path_field_is_honoured():
    """It was in the config and in the UI; the image path never read it."""
    specs = dl.configured(_cfg(lora_path="org/one-lora", lora_scale=0.6))
    assert [(s["source"], s["weight"]) for s in specs] == [("org/one-lora", 0.6)]


def test_a_directory_of_adapters_is_not_swept_for_a_diffusion_model(tmp_path):
    """Config shapes are shared with text models, but "every file in this
    directory" is not a sane answer for a stack applied by name and weight:
    whatever else is in a shared LoRA directory would come along."""
    (tmp_path / "a.safetensors").write_bytes(b"")
    (tmp_path / "b.safetensors").write_bytes(b"")
    cfg = _cfg(lora_model_dir=str(tmp_path))
    assert dl.configured(cfg) == []
    # ...while the text path, which wants exactly that, still gets it.
    assert len(configured_specs(cfg)) == 2


def test_no_loras_configured_is_not_an_error():
    assert dl.configured({}) == []
    assert dl.configured(None) == []


# --------------------------------------------------------------------------- #
#  applying them
# --------------------------------------------------------------------------- #

def test_the_adapters_are_namespaced_and_activated_at_their_weights():
    pipe = FakePipe()
    names = dl.apply_model_loras(pipe, _cfg(loras=[
        {"model": "org/nsfw-lora", "weight": 0.8, "name": "nsfw"}]))
    assert names == ["__model_nsfw"]
    assert pipe.loaded == [("org/nsfw-lora", {"adapter_name": "__model_nsfw"})]
    assert pipe.active == (["__model_nsfw"], [0.8])


def test_a_weight_file_inside_a_repo_is_passed_as_weight_name():
    pipe = FakePipe()
    dl.apply_model_loras(pipe, _cfg(loras=[
        {"model": "org/repo:some_lora_v2.safetensors", "weight": 1.0}]))
    repo, kw = pipe.loaded[0]
    assert repo == "org/repo"
    assert kw["weight_name"] == "some_lora_v2.safetensors"


def test_the_acceleration_lora_stays_active_alongside_the_model_lora():
    """The case that motivated this: base + NSFW LoRA + Lightning as one id.
    Activating one group must not deactivate the other."""
    pipe = FakePipe()
    pipe._coderai_accel_adapters = ["__accel__"]
    pipe._coderai_accel_weight = 1.0
    dl.apply_model_loras(pipe, _cfg(loras=[{"model": "org/nsfw", "weight": 0.8}]))
    names, weights = pipe.active
    assert set(names) == {"__accel__", "__model_nsfw"}
    assert weights[names.index("__model_nsfw")] == 0.8
    assert weights[names.index("__accel__")] == 1.0


def test_re_finalising_a_pipeline_drops_the_previous_model_adapters():
    """A second config of the same checkpoint, or a reload after eviction,
    would otherwise hit "Adapter name already in use"."""
    pipe = FakePipe()
    dl.apply_model_loras(pipe, _cfg(loras=[{"model": "org/nsfw", "weight": 0.8}]))
    dl.apply_model_loras(pipe, _cfg(loras=[{"model": "org/nsfw", "weight": 0.8}]))
    assert ["__model_nsfw"] in pipe.deleted


def test_a_failing_model_lora_raises_instead_of_serving_the_base_model():
    """The whole point of the id is the fine-tune. Loading the base instead
    answers 200 with a picture that is not what was asked for, and nothing
    anywhere says so — so this one is loud."""
    pipe = FakePipe(fail_on={"org/missing"})
    with pytest.raises(RuntimeError, match="org/missing"):
        dl.apply_model_loras(pipe, _cfg(loras=[{"model": "org/missing"}]))


def test_a_pipeline_that_cannot_take_loras_is_a_refusal_not_a_shrug():
    with pytest.raises(RuntimeError, match="load_lora_weights"):
        dl.apply_model_loras(NoLoraPipe(), _cfg(loras=[{"model": "org/x"}]))


def test_an_unactivatable_adapter_set_raises():
    pipe = FakePipe()

    def boom(names, weights):
        raise RuntimeError("peft said no")
    pipe.set_adapters = boom
    with pytest.raises(RuntimeError, match="peft said no"):
        dl.apply_model_loras(pipe, _cfg(loras=[{"model": "org/x"}]))


# --------------------------------------------------------------------------- #
#  a request's own LoRAs must not turn the model's off
# --------------------------------------------------------------------------- #

def _lora(model, weight=1.0, name=None):
    return types.SimpleNamespace(model=model, weight=weight, name=name)


def test_a_request_lora_keeps_the_models_own_adapters_active():
    pipe = FakePipe()
    pipe._coderai_accel_adapters = ["__accel__"]
    pipe._coderai_accel_weight = 1.0
    pipe._coderai_model_adapters = ["__model_nsfw"]
    pipe._coderai_model_weights = [0.8]

    images_mod._apply_loras(pipe, [_lora("org/character", 0.7, "char")])

    names, weights = pipe.active
    assert names[0] == "char" and weights[0] == 0.7
    assert set(names[1:]) == {"__accel__", "__model_nsfw"}, \
        "a request LoRA used to silently un-distil the model and drop its style"


def test_dropping_a_requests_adapters_re_activates_the_models_own():
    """Next request carries no LoRA: the previous adapters are deleted, which
    must not leave the model running without what defines it."""
    pipe = FakePipe()
    pipe._coderai_model_adapters = ["__model_nsfw"]
    pipe._coderai_model_weights = [0.8]
    pipe._coderai_request_adapters = ["char"]

    images_mod._apply_loras(pipe, [])

    assert pipe.active == (["__model_nsfw"], [0.8])


def test_persistent_adapters_defaults_a_missing_weight_rather_than_dropping_it():
    pipe = FakePipe()
    pipe._coderai_model_adapters = ["__model_a", "__model_b"]
    pipe._coderai_model_weights = [0.8]
    names, weights = dl.persistent_adapters(pipe)
    assert names == ["__model_a", "__model_b"]
    assert weights == [0.8, 1.0]


def test_both_image_load_paths_finish_through_the_combined_step():
    """Acceleration alone used to be the last thing a load did; the per-model
    LoRA stack has to happen on every path, including --no-ram."""
    src = open("codai/api/images.py").read()
    assert src.count("return _finish_image_load(pipeline, _mc)") == 2
    assert "return _apply_image_acceleration(pipeline, _mc)" not in src
    body = src[src.index("def _finish_image_load"):][:1400]
    assert body.index("_apply_image_acceleration") < body.index("apply_model_loras"), \
        "accel adapters must be registered before the union is activated"


# --------------------------------------------------------------------------- #
#  single-file checkpoints
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ref,expected", [
    ("/AI/models/someMix_v18.safetensors", True),
    ("/AI/models/old_model.ckpt", True),
    ("org/repo:someMix_v18.safetensors", True),
    ("org/repo", False),
    ("Qwen/Qwen-Image-Edit-2511", False),
    ("", False),
    (None, False),
])
def test_a_single_file_reference_is_told_from_a_repo_id(ref, expected):
    assert hf_loading.is_single_file_ref(ref) is expected


def test_a_local_checkpoint_resolves_to_itself(tmp_path):
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"not really a checkpoint")
    assert hf_loading.resolve_single_file(str(f)) == str(f)


def test_one_file_out_of_a_repo_is_fetched_by_name(monkeypatch):
    """Community repos hold every version of a checkpoint side by side, so the
    id alone does not say which file to serve."""
    calls = []

    def fake_download(repo, filename):
        calls.append((repo, filename))
        return "/cache/" + filename
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    got = hf_loading.resolve_single_file("org/repo:CyberRealisticPony_V18.0_F16.safetensors")
    assert got == "/cache/CyberRealisticPony_V18.0_F16.safetensors"
    assert calls == [("org/repo", "CyberRealisticPony_V18.0_F16.safetensors")]


def test_a_missing_checkpoint_is_reported_as_missing():
    with pytest.raises(FileNotFoundError):
        hf_loading.resolve_single_file("/nowhere/at/all.safetensors")


def test_an_ordinary_repo_load_is_left_alone():
    """The branch returns None rather than raising, so every existing model
    keeps loading exactly as before."""
    assert hf_loading.build_single_file_pipeline(
        "Qwen/Qwen-Image-Edit-2511", {}, None) is None


def test_the_configured_pipeline_class_wins(monkeypatch, tmp_path):
    f = tmp_path / "m.safetensors"
    f.write_bytes(b"")
    seen = []

    class Fake:
        @staticmethod
        def from_single_file(path, **kw):
            seen.append(("FluxPipeline", path, kw))
            return "flux-pipe"
    import diffusers
    monkeypatch.setattr(diffusers, "FluxPipeline", Fake, raising=False)

    out = hf_loading.build_single_file_pipeline(
        str(f), {"pipeline_class": "FluxPipeline"}, "dtype")
    assert out == "flux-pipe"
    assert seen[0][1] == str(f)


def test_sdxl_is_tried_first_and_the_next_class_covers_its_refusal(monkeypatch, tmp_path):
    f = tmp_path / "m.safetensors"
    f.write_bytes(b"")
    tried = []

    class Xl:
        @staticmethod
        def from_single_file(path, **kw):
            tried.append("xl")
            raise ValueError("not an SDXL checkpoint")

    class Sd:
        @staticmethod
        def from_single_file(path, **kw):
            tried.append("sd")
            return "sd-pipe"
    import diffusers
    monkeypatch.setattr(diffusers, "StableDiffusionXLPipeline", Xl)
    monkeypatch.setattr(diffusers, "StableDiffusionPipeline", Sd)

    assert hf_loading.build_single_file_pipeline(str(f), {}, "dtype") == "sd-pipe"
    assert tried == ["xl", "sd"]


def test_quantization_is_offered_then_dropped_rather_than_failing_the_load(monkeypatch, tmp_path):
    """A class whose from_single_file does not take quantization_config would
    otherwise fail a load that works perfectly well unquantized."""
    f = tmp_path / "m.safetensors"
    f.write_bytes(b"")
    attempts = []

    class Xl:
        @staticmethod
        def from_single_file(path, **kw):
            attempts.append(sorted(kw))
            if "quantization_config" in kw:
                raise TypeError("unexpected keyword argument 'quantization_config'")
            return "plain-pipe"
    import diffusers
    monkeypatch.setattr(diffusers, "StableDiffusionXLPipeline", Xl)

    out = hf_loading.build_single_file_pipeline(
        str(f), {}, "dtype", {"quantization_config": object()})
    assert out == "plain-pipe"
    assert attempts == [["quantization_config", "torch_dtype"], ["torch_dtype"]]


def test_an_unloadable_checkpoint_says_what_was_tried_and_what_to_set(monkeypatch, tmp_path):
    f = tmp_path / "m.safetensors"
    f.write_bytes(b"")

    class Boom:
        @staticmethod
        def from_single_file(path, **kw):
            raise ValueError("cannot infer architecture")
    import diffusers
    monkeypatch.setattr(diffusers, "StableDiffusionXLPipeline", Boom)
    monkeypatch.setattr(diffusers, "StableDiffusionPipeline", Boom)

    with pytest.raises(RuntimeError) as e:
        hf_loading.build_single_file_pipeline(str(f), {}, "dtype")
    msg = str(e.value)
    assert "cannot infer architecture" in msg
    assert "pipeline_class" in msg, "the message has to say what to do next"


def test_both_load_sites_branch_on_a_single_file_checkpoint():
    src = open("codai/api/images.py").read()
    assert src.count("build_single_file_pipeline(model_name, _mc, dtype, _xtra)") == 2, \
        "--no-ram and the ordinary path both have to open one"


def test_the_new_fields_are_savable_from_the_admin_page():
    """A config field the admin route does not copy is silently dropped on the
    next save, which is how a configured model quietly reverts."""
    src = open("codai/admin/routes.py").read()
    for field in ('"loras"', '"pipeline_class"'):
        assert field in src, f"{field} is not persisted by the admin route"


# --------------------------------------------------------------------------- #
#  a sampler is a class PLUS its options
# --------------------------------------------------------------------------- #

def test_the_samplers_own_options_survive_resolution():
    """`scheduler_config` has to reach apply_accel_to_pipeline: resolve builds a
    fixed key set, so a field it does not copy is dropped before anything reads
    it — the quietest way for a configured setting to do nothing."""
    from codai.models.acceleration import resolve_acceleration
    accel = resolve_acceleration({"acceleration": {
        "enabled": True, "preset": "custom",
        "scheduler": "DPMSolverMultistepScheduler",
        "scheduler_config": {"algorithm_type": "sde-dpmsolver++",
                             "use_karras_sigmas": True},
    }})
    assert accel["scheduler_config"] == {"algorithm_type": "sde-dpmsolver++",
                                         "use_karras_sigmas": True}


def test_a_missing_scheduler_config_is_an_empty_mapping_not_none():
    from codai.models.acceleration import resolve_acceleration
    accel = resolve_acceleration({"acceleration": {"enabled": True, "steps": 4}})
    assert accel["scheduler_config"] == {}


def test_the_sampler_options_are_merged_over_the_current_scheduler(monkeypatch):
    """Without this, naming DPMSolverMultistepScheduler silently gives plain
    DPM++ 2M, not the SDE Karras variant the checkpoint was tuned for."""
    from codai.models import acceleration as accel_mod

    built = {}

    class FakeSched:
        @classmethod
        def from_config(cls, cfg):
            built.update(cfg)
            return "new-scheduler"

    class Pipe:
        scheduler = type("S", (), {"config": {"beta_start": 0.00085,
                                              "use_karras_sigmas": False}})()

    import diffusers
    monkeypatch.setattr(diffusers, "DPMSolverMultistepScheduler", FakeSched,
                        raising=False)
    pipe = Pipe()
    accel_mod.apply_accel_to_pipeline(pipe, {
        "preset": "custom", "scheduler": "DPMSolverMultistepScheduler",
        "scheduler_config": {"algorithm_type": "sde-dpmsolver++",
                             "use_karras_sigmas": True},
    })
    assert pipe.scheduler == "new-scheduler"
    assert built["algorithm_type"] == "sde-dpmsolver++"
    assert built["use_karras_sigmas"] is True
    assert built["beta_start"] == 0.00085, "the rest of the config must survive"
