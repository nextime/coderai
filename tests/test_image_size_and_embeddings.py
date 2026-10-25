"""Request size, and the learned prompt tokens a prompt needs loaded.

Two sets of silent failures, both of which answered 200 with an image:

* **size.** A size that did not parse fell through to the default, so
  ``"1024"`` generated a square nobody asked for. A side that was not a
  multiple of 8 went to the pipeline as-is, where the VAE cannot represent it.
  And nothing capped it, so ``4096x4096`` came back as a CUDA OOM -- which
  reads like a busy GPU rather than a size the caller chose.
* **textual inversions.** ``easynegative`` and the rest are vectors living in
  CLIP's embedding table under a word of their own. Until the file is loaded
  the word is an ordinary unknown token, so half the ecosystem's negative
  prompts asked for nothing at all. A multi-vector embedding needs its prompt
  expanded too, or only the first of its vectors is ever used.
"""
import os
import types

import pytest
from fastapi import HTTPException

from codai.api import images as images_mod
from codai.models import textual_inversion as ti


@pytest.fixture
def no_limits():
    """An install with the shipped defaults: 4096 px a side, 4.2 MP."""
    return types.SimpleNamespace()


# --------------------------------------------------------------------------- #
#  size
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("size,expected", [
    ("1024x1024", (1024, 1024)),
    ("832x1216",  (832, 1216)),          # the usual SDXL portrait
    (" 768 x 512 ", (768, 512)),
    ("768X512",   (768, 512)),
    ("768×512",   (768, 512)),           # the unicode multiplication sign
])
def test_a_good_size_is_taken_as_written(size, expected, no_limits):
    assert images_mod._parse_size(size, 1024, 1024, no_limits) == expected


@pytest.mark.parametrize("size", [None, "", "auto", "AUTO"])
def test_nothing_asked_for_means_the_default(size, no_limits):
    assert images_mod._parse_size(size, 640, 480, no_limits) == (640, 480)


@pytest.mark.parametrize("size", ["1024", "1k x 1k", "1024x", "x1024", "big",
                                  "1024*1024*1024", "-512x512", "0x0x0"])
def test_a_size_that_does_not_parse_is_a_400(size, no_limits):
    """It used to silently generate at the default instead."""
    with pytest.raises(HTTPException) as exc:
        images_mod._parse_size(size, 1024, 1024, no_limits)
    assert exc.value.status_code == 400
    assert "WIDTHxHEIGHT" in exc.value.detail


def test_the_error_says_what_was_sent(no_limits):
    with pytest.raises(HTTPException) as exc:
        images_mod._parse_size("1k x 1k", 1024, 1024, no_limits)
    assert "'1k x 1k'" in exc.value.detail


@pytest.mark.parametrize("size,expected", [
    ("1000x1000", (1000, 1000)),         # already a multiple of 8
    ("1001x1001", (1000, 1000)),         # nearest
    ("1023x1023", (1024, 1024)),
    ("3x3",       (8, 8)),               # never below one latent pixel
])
def test_a_side_off_the_grid_is_snapped(size, expected, no_limits):
    """The VAE downsamples by 8: an odd side is not representable."""
    assert images_mod._parse_size(size, 1024, 1024, no_limits) == expected


@pytest.mark.parametrize("size", ["0x0", "0x512", "512x0"])
def test_a_zero_dimension_is_a_400(size, no_limits):
    """Snapping it up would answer an uninitialised client variable with an
    8x8 image."""
    with pytest.raises(HTTPException) as exc:
        images_mod._parse_size(size, 1024, 1024, no_limits)
    assert exc.value.status_code == 400
    assert "zero dimension" in exc.value.detail


def test_an_oversized_side_is_a_400_naming_the_limit(no_limits):
    with pytest.raises(HTTPException) as exc:
        images_mod._parse_size("8192x512", 1024, 1024, no_limits)
    assert exc.value.status_code == 400
    assert "maximum side" in exc.value.detail and "4096" in exc.value.detail


def test_too_many_pixels_is_a_400_not_an_oom(no_limits):
    """4096x4096 used to be accepted and come back as a CUDA OOM."""
    with pytest.raises(HTTPException) as exc:
        images_mod._parse_size("4096x4096", 1024, 1024, no_limits)
    assert exc.value.status_code == 400
    assert "MP" in exc.value.detail and "image.max_pixels" in exc.value.detail


@pytest.mark.parametrize("size", ["2048x2048", "2560x1440", "1920x1080",
                                  "1024x1024", "832x1216"])
def test_every_ordinary_size_is_under_the_default_ceiling(size, no_limits):
    images_mod._parse_size(size, 1024, 1024, no_limits)   # no raise


def test_the_ceiling_is_configurable_upwards():
    """An operator with the VRAM for it can raise both."""
    big = types.SimpleNamespace(image_max_side=8192, image_max_pixels=8192 * 8192)
    assert images_mod._parse_size("8192x8192", 1024, 1024, big) == (8192, 8192)


def test_the_dead_nan_guard_is_gone():
    """`if width != width` on a value that is already int can never fire."""
    src = open("codai/api/images.py").read()
    assert "if width != width or width == float('inf')" not in src


def test_inpaint_no_longer_defaults_to_a_square():
    """A 1024x1024 default squashed every non-square photo -- and pulled the
    mask out of alignment with it, which is worse than on a plain edit."""
    assert images_mod.ImageInpaintRequest.model_fields["size"].default is None


# --------------------------------------------------------------------------- #
#  textual inversions: resolving
# --------------------------------------------------------------------------- #

@pytest.fixture
def embeddings_dir(tmp_path):
    d = tmp_path / "embeddings"
    d.mkdir()
    for name in ("easynegative.safetensors", "bad-hands-5.pt", "notes.txt"):
        (d / name).write_bytes(b"\x00")
    return str(d)


def test_a_bare_name_is_looked_up_in_the_directory(embeddings_dir):
    """What a prompt copied off a model card means by `easynegative`."""
    source, token = ti.resolve("easynegative", embeddings_dir)
    assert source == os.path.join(embeddings_dir, "easynegative.safetensors")
    assert token == "easynegative"


def test_a_name_with_its_extension_also_resolves(embeddings_dir):
    source, token = ti.resolve("bad-hands-5.pt", embeddings_dir)
    assert source.endswith("bad-hands-5.pt")
    assert token == "bad-hands-5"


def test_a_path_is_taken_as_given(embeddings_dir, tmp_path):
    path = tmp_path / "elsewhere.safetensors"
    path.write_bytes(b"\x00")
    source, token = ti.resolve(str(path), None)
    assert source == str(path)
    assert token == "elsewhere"


def test_a_trigger_word_can_be_renamed(embeddings_dir):
    source, token = ti.resolve("easynegative:myneg", embeddings_dir)
    assert source.endswith("easynegative.safetensors")
    assert token == "myneg"


def test_a_repo_id_is_left_for_diffusers_to_fetch():
    source, token = ti.resolve("sd-concepts-library/cat-toy", None)
    assert source == "sd-concepts-library/cat-toy"
    assert token == "cat-toy"


def test_a_missing_embedding_says_where_it_looked(embeddings_dir):
    with pytest.raises(FileNotFoundError) as exc:
        ti.resolve("nosuchthing", embeddings_dir)
    assert "nosuchthing" in str(exc.value)
    assert embeddings_dir in str(exc.value)


def test_a_missing_embedding_with_no_directory_says_so():
    with pytest.raises(FileNotFoundError) as exc:
        ti.resolve("nosuchthing", None)
    assert "image.embeddings_dir is not set" in str(exc.value)


def test_discovery_takes_the_embeddings_and_leaves_the_rest(embeddings_dir):
    found = ti.discover(embeddings_dir)
    assert set(found) == {"easynegative", "bad-hands-5"}     # not notes.txt


def test_discovery_of_a_directory_that_is_not_there_is_not_an_error():
    assert ti.discover("/nonexistent/embeddings") == {}
    assert ti.discover(None) == {}


# --------------------------------------------------------------------------- #
#  textual inversions: loading
# --------------------------------------------------------------------------- #

class _FakePipeline:
    """Records what was loaded, and into which encoder pair."""

    def __init__(self, sdxl=False, fail=()):
        self.calls = []
        self.fail = set(fail)
        self.tokenizer = "tok"
        self.text_encoder = "enc"
        if sdxl:
            self.tokenizer_2 = "tok2"
            self.text_encoder_2 = "enc2"

    def load_textual_inversion(self, source, token=None, text_encoder=None,
                               tokenizer=None):
        if token in self.fail:
            raise RuntimeError("bad file")
        self.calls.append((source, token, tokenizer))


def test_loading_registers_the_trigger_word(embeddings_dir):
    pipe = _FakePipeline()
    tokens = ti.apply(pipe, ["easynegative"], embeddings_dir)
    assert "easynegative" in tokens
    assert pipe.calls[0][1] == "easynegative"


def test_everything_in_the_directory_is_loaded_unasked(embeddings_dir):
    """A1111's behaviour, and what a copied prompt assumes."""
    pipe = _FakePipeline()
    tokens = ti.apply(pipe, [], embeddings_dir)
    assert set(tokens) == {"easynegative", "bad-hands-5"}


def test_a_second_request_does_not_load_them_again(embeddings_dir):
    """The pipeline outlives the request; diffusers raises on a re-register."""
    pipe = _FakePipeline()
    ti.apply(pipe, ["easynegative"], embeddings_dir)
    before = len(pipe.calls)
    tokens = ti.apply(pipe, ["easynegative"], embeddings_dir)
    assert len(pipe.calls) == before
    assert "easynegative" in tokens


def test_a_name_the_caller_gave_that_cannot_load_is_an_error(embeddings_dir):
    """Rendering without it would look like a success."""
    pipe = _FakePipeline(fail=("easynegative",))
    with pytest.raises(RuntimeError):
        ti.apply(pipe, ["easynegative"], embeddings_dir, strict=True)


def test_one_bad_file_in_the_directory_does_not_break_the_request(embeddings_dir):
    """A sweep is never strict: the other embeddings still load."""
    pipe = _FakePipeline(fail=("bad-hands-5",))
    tokens = ti.apply(pipe, [], embeddings_dir, strict=True)
    assert tokens == ["easynegative"]


def test_an_sdxl_embedding_goes_into_both_encoders(tmp_path, monkeypatch):
    """clip_l is 768 wide and clip_g is 1280: loading only one gives the token
    an embedding in one encoder and nothing in the other."""
    path = tmp_path / "sdxlneg.safetensors"
    path.write_bytes(b"\x00")
    monkeypatch.setattr(
        "safetensors.torch.load_file",
        lambda p: {"clip_l": "L", "clip_g": "G"})
    pipe = _FakePipeline(sdxl=True)
    ti.apply(pipe, [str(path)], None)
    assert [(c[0], c[2]) for c in pipe.calls] == [("L", "tok"), ("G", "tok2")]
    assert {c[1] for c in pipe.calls} == {"sdxlneg"}


def test_a_single_tensor_embedding_takes_the_plain_path(tmp_path, monkeypatch):
    path = tmp_path / "sd15neg.pt"
    path.write_bytes(b"\x00")
    pipe = _FakePipeline(sdxl=True)
    ti.apply(pipe, [str(path)], None)
    assert pipe.calls == [(str(path), "sd15neg", None)]


def test_a_pipeline_that_cannot_load_them_is_left_alone():
    assert ti.apply(types.SimpleNamespace(), [], None) == []
    with pytest.raises(RuntimeError):
        ti.apply(types.SimpleNamespace(), ["easynegative"], None, strict=True)


# --------------------------------------------------------------------------- #
#  textual inversions: the prompt side
# --------------------------------------------------------------------------- #

def test_a_multi_vector_embedding_is_written_out_before_tokenizing():
    """`easynegative` is eight vectors registered as easynegative,
    easynegative_1 ... If the prompt is not expanded, seven are unused."""
    from codai.models.prompt_weighting import _expand_learned_tokens

    def convert(text, tokenizer):
        return text.replace("easynegative",
                            "easynegative easynegative_1 easynegative_2")

    pipe = types.SimpleNamespace(maybe_convert_prompt=convert)
    out = _expand_learned_tokens(pipe, "blurry, easynegative", "tok")
    assert out == "blurry, easynegative easynegative_1 easynegative_2"


def test_expansion_leaves_the_weight_grammar_alone():
    from codai.models.prompt_weighting import _expand_learned_tokens
    pipe = types.SimpleNamespace(
        maybe_convert_prompt=lambda text, tok: text.replace("emb", "emb emb_1"))
    assert _expand_learned_tokens(pipe, "(emb:1.3) BREAK x", "tok") == \
        "(emb emb_1:1.3) BREAK x"


def test_a_pipeline_without_expansion_is_no_obstacle():
    from codai.models.prompt_weighting import _expand_learned_tokens
    assert _expand_learned_tokens(types.SimpleNamespace(), "a cat", "tok") == "a cat"


def test_expansion_failure_falls_back_to_the_prompt_as_written():
    from codai.models.prompt_weighting import _expand_learned_tokens

    def boom(text, tok):
        raise RuntimeError("no")

    pipe = types.SimpleNamespace(maybe_convert_prompt=boom)
    assert _expand_learned_tokens(pipe, "a cat", "tok") == "a cat"


# --------------------------------------------------------------------------- #
#  the API surface
# --------------------------------------------------------------------------- #

def test_the_request_takes_an_embeddings_list():
    from codai.pydantic.imagerequest import ImageGenerationRequest
    r = ImageGenerationRequest(model="m", prompt="p",
                               embeddings=["easynegative", "x.pt:y"])
    assert r.embeddings == ["easynegative", "x.pt:y"]
    assert ImageGenerationRequest(model="m", prompt="p").embeddings is None


def test_a_missing_embedding_a_caller_named_is_a_400(monkeypatch, embeddings_dir):
    pipe = _FakePipeline()
    req = types.SimpleNamespace(embeddings=["nosuchthing"])
    args = types.SimpleNamespace(image_embeddings_dir=embeddings_dir)
    with pytest.raises(HTTPException) as exc:
        images_mod._apply_embeddings(pipe, req, {}, args)
    assert exc.value.status_code == 400
    assert "nosuchthing" in exc.value.detail


def test_nothing_configured_and_nothing_asked_for_does_nothing():
    pipe = _FakePipeline()
    req = types.SimpleNamespace(embeddings=None)
    assert images_mod._apply_embeddings(
        pipe, req, {}, types.SimpleNamespace()) == []
    assert pipe.calls == []


def test_a_model_can_carry_its_own_embeddings(embeddings_dir):
    """So every request to it gets the trigger words, whatever the client is."""
    pipe = _FakePipeline()
    req = types.SimpleNamespace(embeddings=None)
    tokens = images_mod._apply_embeddings(
        pipe, req, {"textual_inversion": ["easynegative"]},
        types.SimpleNamespace(image_embeddings_dir=embeddings_dir))
    assert "easynegative" in tokens
    assert any(c[1] == "easynegative" for c in pipe.calls)


def test_the_loaded_token_set_is_part_of_the_embedding_cache_key():
    """The same prompt text tokenizes differently once a trigger word exists,
    so a cache entry from before the load must not be reused."""
    src = open("codai/api/images.py").read()
    assert "|ti={','.join(_ti_tokens)}" in src


def test_both_are_configurable_per_model_and_server_wide():
    assert '"textual_inversion"' in open("codai/admin/routes.py").read()
    assert "cfg-textual-inversion" in open(
        "codai/admin/templates/models.html").read()
    from codai.config import ImageConfig
    cfg = ImageConfig()
    assert cfg.embeddings_dir == ""
    assert cfg.max_side == 4096 and cfg.max_pixels == 4096 * 1024


def test_a_400_is_not_relabelled_as_a_server_fault():
    """The edit endpoints caught every Exception and answered 500, so a 400
    raised while preparing the prompt -- an embedding that does not exist --
    arrived as "Image editing failed: 400: ...", a server fault it is not."""
    src = open("codai/api/images.py").read()
    for message in ("Image editing failed", "Inpainting failed",
                    "Outfit change failed"):
        idx = src.index(message)
        assert "except HTTPException:" in src[max(0, idx - 250):idx], message


def test_a_rejected_request_is_not_mistaken_for_a_failed_backend():
    """/v1/images/generations tries diffusers, then sd.cpp, and reports which
    backends failed if both do. A 400 from the request itself (a bad size, a
    missing embedding) raised on that path used to be caught as a backend
    failure, so the caller got a report about loading instead of the reason."""
    src = open("codai/api/images.py").read()
    head = src.index("Step 4: Model not loaded")
    tail = src.index("Step 5: Both backends failed")
    section = src[head:tail]
    # one pass-through per backend attempt, each above its catch-all
    assert section.count("except HTTPException:") == 2
    for idx in (section.index("diffusers_error = str(e)\n                print(f\"diffusers error"),
                section.index("sdcpp_error = str(e)\n            print(f\"sd.cpp error")):
        assert "except HTTPException:" in section[max(0, idx - 700):idx]
