"""Where each component of the LongCat pipeline actually sits on the card.

Upstream's LongCatVideoPipeline is a plain object whose docstring claims it inherits
from DiffusionPipeline. It does not, so it has no enable_model_cpu_offload and no
enable_sequential_cpu_offload: asking for offload logged a warning and then loaded
everything onto the card anyway, and the offload setting in the model config did
nothing. On a 24 GB card that is ~25 GB of components before activations even with an
INT8 DiT, because the UMT5-XXL text encoder is ~11 GB on its own.

The encoder is used once, to encode the prompt. These tests hold it off the card for
everything else.
"""
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVICE = ROOT / "tools" / "longcat_service.py"


class FakeModule:
    """Records every device it is moved to, in order."""

    def __init__(self, name):
        self.name = name
        self.device = "cpu"
        self.history = ["cpu"]

    def to(self, device, non_blocking=False):
        self.device = str(device)
        self.history.append(str(device))
        return self


class FakePipeline:
    """Upstream's shape: a plain object with its own to(), no offload hooks at all."""

    def __init__(self):
        self.dit = FakeModule("dit")
        self.vae = FakeModule("vae")
        self.text_encoder = FakeModule("text_encoder")
        self.device = "cpu"
        self.encode_calls = []
        # What was on the card at the moment each component was moved there.
        self.coresident = []

    def to(self, device):
        self.device = str(device)
        for attr in ("dit", "text_encoder", "vae"):
            mod = getattr(self, attr)
            if mod is not None:
                mod.to(device, non_blocking=True)
        self._snapshot()
        return self

    def _snapshot(self):
        self.coresident.append({
            name: getattr(self, name).device
            for name in ("dit", "vae", "text_encoder")
            if getattr(self, name) is not None
        })

    def encode_prompt(self, prompt, device=None, **kwargs):
        self.encode_calls.append(prompt)
        self._snapshot()
        return ("embeds", "mask")


@pytest.fixture
def service(monkeypatch):
    torch = pytest.importorskip("torch")
    # Never touch a real card from a test: production holds this one.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    spec = importlib.util.spec_from_file_location("lc_service_placement", SERVICE)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def test_without_offload_everything_is_resident(service):
    """The existing behaviour, for a card that can hold it all."""
    pipe = FakePipeline()

    service._place_pipeline(pipe, None)

    assert pipe.dit.device == "cuda"
    assert pipe.vae.device == "cuda"
    assert pipe.text_encoder.device == "cuda"


@pytest.mark.parametrize("mode", ["model", "sequential"])
def test_offload_leaves_the_text_encoder_on_the_host(service, mode):
    """~11 GB that is dead weight for every denoising step."""
    pipe = FakePipeline()

    service._place_pipeline(pipe, mode)

    assert pipe.dit.device == "cuda", "the DiT must stay resident — it runs every step"
    assert pipe.vae.device == "cuda"
    assert pipe.text_encoder.device == "cpu"


def test_the_encoder_is_never_on_the_card_beside_the_dit_during_placement(service):
    """Moving it over and straight back would still need the peak we are avoiding."""
    pipe = FakePipeline()

    service._place_pipeline(pipe, "model")

    for snapshot in pipe.coresident:
        both = snapshot.get("dit") == "cuda" and snapshot.get("text_encoder") == "cuda"
        assert not both, f"DiT and text encoder were both on the card: {snapshot}"
    assert "cuda" not in pipe.text_encoder.history


def test_the_encoder_visits_the_card_to_encode_and_leaves_again(service):
    pipe = FakePipeline()
    service._place_pipeline(pipe, "model")

    out = pipe.encode_prompt("a fighter entering the ring", device="cuda")

    assert out == ("embeds", "mask")
    assert pipe.encode_calls == ["a fighter entering the ring"]
    during = pipe.coresident[-1]
    assert during["text_encoder"] == "cuda", "it has to be on the card to encode"
    assert pipe.text_encoder.device == "cpu", "and has to go back afterwards"


def test_a_failed_encode_still_puts_the_encoder_back(service):
    """Stranding 11 GB on a failure makes the retry the thing that runs out of memory."""

    class Failing(FakePipeline):
        def encode_prompt(self, prompt, device=None, **kwargs):
            self._snapshot()
            raise RuntimeError("tokenizer blew up")

    pipe = Failing()
    service._place_pipeline(pipe, "model")

    with pytest.raises(RuntimeError, match="tokenizer blew up"):
        pipe.encode_prompt("x")

    assert pipe.coresident[-1]["text_encoder"] == "cuda", "it did reach the card"
    assert pipe.text_encoder.device == "cpu", "and was put back despite the failure"


def test_every_generate_path_goes_through_the_wrapper(service):
    """All four generate_* methods call self.encode_prompt, so the wrapper must be set
    on the instance, where it shadows the class method."""
    pipe = FakePipeline()
    service._place_pipeline(pipe, "model")

    assert "encode_prompt" in pipe.__dict__
    assert pipe.encode_prompt is not FakePipeline.encode_prompt


def test_no_cuda_is_reported_not_silently_tolerated(service, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    pipe = FakePipeline()

    service._place_pipeline(pipe, "model")

    assert pipe.dit.device == "cpu"
