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

    def __init__(self, name, children=()):
        self.name = name
        self.device = "cpu"
        self.history = ["cpu"]
        self.children = list(children) or [self]

    def to(self, device, non_blocking=False):
        self.device = str(device)
        self.history.append(str(device))
        return self

    def modules(self):
        """nn.Module's walk, which _encoder_is_quantised uses to spot bnb layers."""
        return iter(self.children)


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
    assert pipe.dit.device == "cuda", "the DiT has to come back for the denoising loop"


def test_the_dit_and_the_encoder_are_never_on_the_card_together(service):
    """The whole point. 13.6 GB of INT8 DiT plus ~11 GB of UMT5-XXL is 24.8 GB on a
    24 GB card: bringing the encoder over WITHOUT moving the DiT off is an OOM, and it
    is the one this fixture failed to catch the first time. Peak must be max(), not sum.
    """
    pipe = FakePipeline()
    service._place_pipeline(pipe, "model")

    pipe.encode_prompt("two fighters circling", device="cuda")

    for snapshot in pipe.coresident:
        on_card = [name for name, dev in snapshot.items() if dev == "cuda"]
        assert not ("dit" in on_card and "text_encoder" in on_card), (
            f"both were resident at once: {snapshot}")


def test_the_dit_comes_back_even_if_the_encode_fails(service):
    """Otherwise a failed prompt leaves the DiT on the host and the next generation
    runs at PCIe speed with no indication why."""

    class Failing(FakePipeline):
        def encode_prompt(self, prompt, device=None, **kwargs):
            self._snapshot()
            raise RuntimeError("bad prompt")

    pipe = Failing()
    service._place_pipeline(pipe, "model")

    with pytest.raises(RuntimeError, match="bad prompt"):
        pipe.encode_prompt("x")

    assert pipe.dit.device == "cuda"
    assert pipe.text_encoder.device == "cpu"


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


# ---------------------------------------------------------------- attention backend
# The checkpoint config asks for FlashAttention-2; the venv installs xformers, because
# flash-attn is a source build. Upstream's attention raises on anything unselected, and
# the --attention setting was written to the environment and never read — so the config
# won and imported a module that is not there.

def _fake_find_spec(installed):
    def find_spec(name, *a, **k):
        return object() if name in installed else None
    return find_spec


@pytest.fixture
def pick(service, monkeypatch):
    import importlib.util

    def choose(installed, want=None):
        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(installed))
        if want is None:
            monkeypatch.delenv("LONGCAT_ATTENTION", raising=False)
        else:
            monkeypatch.setenv("LONGCAT_ATTENTION", want)
        return service._attention_kwargs()
    return choose


def test_exactly_one_backend_is_ever_enabled(pick):
    """Two enabled flags would silently take whichever branch is checked first."""
    flags = pick({"xformers", "flash_attn"})
    assert sum(bool(v) for v in flags.values()) == 1
    assert set(flags) == {"enable_flashattn3", "enable_flashattn2", "enable_xformers"}


def test_it_picks_what_is_installed_not_what_the_checkpoint_asks_for(pick):
    """dit/config.json says enable_flashattn2: true and flash_attn is not in the venv."""
    flags = pick({"xformers"})
    assert flags["enable_xformers"] is True
    assert flags["enable_flashattn2"] is False


def test_an_explicit_choice_is_honoured_when_it_is_available(pick):
    flags = pick({"xformers", "flash_attn"}, want="flash")
    assert flags["enable_flashattn2"] is True
    assert flags["enable_xformers"] is False


def test_asking_for_a_backend_that_is_absent_falls_back(pick):
    flags = pick({"xformers"}, want="flash")
    assert flags["enable_xformers"] is True


@pytest.mark.parametrize("alias", ["flashattn2", "flash2", "fa2"])
def test_the_usual_names_for_flashattention_all_work(pick, alias):
    """A config saying "flashattn2" should not silently land on xformers."""
    flags = pick({"xformers", "flash_attn"}, want=alias)
    assert flags["enable_flashattn2"] is True


def test_no_backend_at_all_is_an_error_here_not_deep_in_the_first_step(pick):
    """Upstream's message is "Unsupported attention operations." from inside attention,
    48 blocks into a load that already cost minutes."""
    with pytest.raises(RuntimeError, match="no usable attention backend"):
        pick(set())


def test_both_load_paths_get_the_backend(service):
    """bf16 via from_pretrained and INT8 via the config dict — a fix to one only would
    leave the other raising on the first attention call. Both go through _dit_kwargs(),
    which is where the backend is chosen."""
    src = SERVICE.read_text(encoding="utf-8")
    body = src[src.index("def load_pipeline"):src.index("def _cp_split_hw")]
    assert body.count("_dit_kwargs()") == 2
    assert "_attention_kwargs()" in src[src.index("def _dit_kwargs"):]


# ------------------------------------------------- one encode per request, not per segment
# _run_segments calls generate_t2v once PER SEGMENT and each calls self.encode_prompt.
# A one-minute video at 15 fps is 12 segments, so that was 12 identical encodes — and
# under offload, 12 round trips of ~35 GB over PCIe for an answer that cannot change.

def test_the_same_prompt_is_encoded_once(service):
    pipe = FakePipeline()
    service._place_pipeline(pipe, "model")

    for _ in range(12):                       # twelve segments, one prompt
        out = pipe.encode_prompt("a boxer", device="cuda")

    assert pipe.encode_calls == ["a boxer"], "encoded once per request, not per segment"
    assert out == ("embeds", "mask")


def test_the_encoder_stays_off_the_card_after_the_first_segment(service):
    """The point of caching here: for the rest of a long generation the card holds
    nothing but the DiT and the VAE."""
    pipe = FakePipeline()
    service._place_pipeline(pipe, "model")
    pipe.encode_prompt("a boxer", device="cuda")
    visits_after_first = len([d for d in pipe.text_encoder.history if d == "cuda"])

    for _ in range(11):
        pipe.encode_prompt("a boxer", device="cuda")

    assert len([d for d in pipe.text_encoder.history if d == "cuda"]) == visits_after_first
    assert pipe.dit.device == "cuda"


def test_a_different_prompt_is_encoded_again(service):
    pipe = FakePipeline()
    service._place_pipeline(pipe, "model")

    pipe.encode_prompt("a boxer", device="cuda")
    pipe.encode_prompt("a dancer", device="cuda")
    pipe.encode_prompt("a boxer", device="cuda")

    assert pipe.encode_calls == ["a boxer", "a dancer"]


def test_differing_keyword_arguments_are_not_conflated(service):
    """Same text, different guidance or sequence length is a different encode."""
    pipe = FakePipeline()
    service._place_pipeline(pipe, "model")

    pipe.encode_prompt("a boxer", device="cuda", max_sequence_length=512)
    pipe.encode_prompt("a boxer", device="cuda", max_sequence_length=226)

    assert len(pipe.encode_calls) == 2


def test_the_cache_is_bounded(service):
    """A long-lived service must not accumulate embeddings for every prompt it sees."""
    pipe = FakePipeline()
    service._place_pipeline(pipe, "model")

    for i in range(10):
        pipe.encode_prompt(f"prompt {i}", device="cuda")
    pipe.encode_prompt("prompt 0", device="cuda")      # evicted by now

    assert len(pipe.encode_calls) == 11


def test_caching_also_applies_without_offload(service):
    """Fully resident still runs 12 redundant encodes otherwise."""
    pipe = FakePipeline()
    service._place_pipeline(pipe, None)

    for _ in range(5):
        pipe.encode_prompt("a boxer", device="cuda")

    assert pipe.encode_calls == ["a boxer"]
    assert pipe.text_encoder.device == "cuda", "nothing was offloaded"


def test_an_unhashable_argument_still_encodes(service):
    """A dict or tensor among the arguments must not break the call, just skip the cache."""
    pipe = FakePipeline()
    service._place_pipeline(pipe, "model")

    pipe.encode_prompt("a boxer", device="cuda", extra={"weights": [1, 2]})
    pipe.encode_prompt("a boxer", device="cuda", extra={"weights": [1, 2]})

    assert len(pipe.encode_calls) == 2
