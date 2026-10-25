"""The prompt syntax the SD ecosystem speaks, and the 77-token wall.

Three defects are pinned here, all of them silent -- each one answered 200 with
a plausible-looking image:

1. A prompt longer than 77 CLIP tokens was truncated by the tokenizer and the
   rest was never seen. The booru-tag fine-tune families (Pony, Illustrious,
   NoobAI) open with a run of quality tags, so the part that got dropped was
   the subject.
2. ``(tag:1.3)``, ``[tag]`` and ``BREAK`` reached CLIP as literal text, i.e.
   as noise, because nothing parsed them.
3. The sd.cpp path hardcoded ``negative_prompt=''``: for every GGUF image
   model the caller's negative prompt was discarded between the API and the
   engine, which is half of the idiom gone.

Plus ``clip_skip``, which those models are trained at and which an
OpenAI-image-API client has no field to ask for -- so the model's own config
has to be able to.
"""
import types

import pytest

from codai.models.prompt_weighting import (
    BREAK,
    WINDOW,
    encode_weighted_prompt,
    has_weighting_syntax,
    parse_prompt_attention,
    supports_weighted_embeds,
    tokenize_weighted,
    window_count,
)


# --------------------------------------------------------------------------- #
#  the grammar
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,expected", [
    ("a (b) c",      [["a ", 1.0], ["b", 1.1], [" c", 1.0]]),
    ("(a:1.3)",      [["a", 1.3]]),
    ("(a:0.5)",      [["a", 0.5]]),
    ("[a]",          [["a", 1 / 1.1]]),
    ("((a))",        [["a", 1.1 * 1.1]]),
    ("[[a]]",        [["a", (1 / 1.1) ** 2]]),
    ("",             [["", 1.0]]),
])
def test_the_a1111_grammar(text, expected):
    got = parse_prompt_attention(text)
    assert len(got) == len(expected)
    for (gt, gw), (et, ew) in zip(got, expected):
        assert gt == et
        assert gw == pytest.approx(ew)


def test_a_backslash_escapes_a_bracket():
    """Prose with parentheses in it has an out, and it is the ecosystem's."""
    assert parse_prompt_attention(r"a \(b\) c") == [["a (b) c", 1.0]]


def test_an_unclosed_bracket_still_applies():
    """What a user who forgot the closing bracket meant."""
    assert parse_prompt_attention("a (b c") == [["a ", 1.0], ["b c", 1.1]]


def test_break_is_a_marker_not_a_word():
    parsed = parse_prompt_attention("a BREAK b")
    assert [p[0] for p in parsed] == ["a", BREAK, "b"]
    assert parsed[1][1] == -1


def test_break_is_never_merged_into_the_text_around_it():
    """Both neighbours weigh 1.0; merging equal weights must not swallow it."""
    parsed = parse_prompt_attention("one BREAK two BREAK three")
    assert [p[0] for p in parsed] == ["one", BREAK, "two", BREAK, "three"]


def test_weights_nest_through_a_sentence():
    parsed = parse_prompt_attention("score_9, (masterpiece:1.4), [blurry]")
    weights = {t.strip(" ,"): w for t, w in parsed}
    assert weights["score_9"] == pytest.approx(1.0)
    assert weights["masterpiece"] == pytest.approx(1.4)
    assert weights["blurry"] == pytest.approx(1 / 1.1)


@pytest.mark.parametrize("text,flagged", [
    ("a plain prompt", False),
    ("score_9, score_8_up, a cat", False),
    ("(a cat:1.2)", True),
    ("[blurry]", True),
    ("a cat BREAK a dog", True),
    (r"a \(parenthetical\) aside", False),
])
def test_reporting_whether_the_grammar_was_used(text, flagged):
    assert has_weighting_syntax(text) is flagged


# --------------------------------------------------------------------------- #
#  windows: the 77-token wall
# --------------------------------------------------------------------------- #

class _FakeTokenizer:
    """One id per whitespace-separated word, which makes token counts countable."""
    bos_token_id = 49406
    eos_token_id = 49407
    pad_token_id = 49407

    def __call__(self, text, add_special_tokens=True, truncation=True, **kw):
        assert add_special_tokens is False, "the markers belong to the window"
        assert truncation is False, "truncating here is the whole defect"
        ids = [1000 + (hash(w) % 1000) for w in str(text).split()]
        return types.SimpleNamespace(input_ids=ids)


def test_a_short_prompt_is_one_window():
    tok = _FakeTokenizer()
    windows, weights = tokenize_weighted(tok, "a cat on a mat")
    assert len(windows) == 1
    assert len(windows[0]) == WINDOW == len(weights[0])


def test_a_long_prompt_is_not_truncated_but_windowed():
    """The regression: 200 tokens used to become 75 and nobody was told."""
    tok = _FakeTokenizer()
    prompt = " ".join(f"tag{i}" for i in range(200))
    windows, weights = tokenize_weighted(tok, prompt)
    assert len(windows) == 3                     # 75 + 75 + 50 padded
    assert window_count(tok, prompt) == 3
    assert all(len(w) == WINDOW for w in windows)
    # every one of the 200 tokens is somewhere in those windows
    flat = [t for w in windows for t in w]
    assert sum(1 for t in flat if t != tok.pad_token_id) == 200


def test_break_closes_the_window_early():
    """Two tags separated by BREAK do not share a window, which is the point."""
    tok = _FakeTokenizer()
    windows, _ = tokenize_weighted(tok, "cat BREAK dog")
    assert len(windows) == 2
    assert windows[0][1] == tok.pad_token_id     # padded out after one token
    assert windows[1][1] == tok.pad_token_id


def test_weights_line_up_with_their_tokens():
    tok = _FakeTokenizer()
    windows, weights = tokenize_weighted(tok, "plain (heavy:2.0) plain2")
    assert weights[0][0] == pytest.approx(1.0)
    assert weights[0][1] == pytest.approx(2.0)
    assert weights[0][2] == pytest.approx(1.0)
    # the padding weighs nothing in particular
    assert weights[0][3] == pytest.approx(1.0)


def test_plain_mode_keeps_the_windows_and_drops_the_grammar():
    """`prompt_weighting: false` is an escape hatch, not a return to truncation."""
    tok = _FakeTokenizer()
    prompt = " ".join(f"tag{i}" for i in range(100)) + " (heavy:2.0)"
    windows, weights = tokenize_weighted(tok, prompt, plain=True)
    assert len(windows) == 2
    assert all(w == 1.0 for win in weights for w in win)


# --------------------------------------------------------------------------- #
#  encoding
# --------------------------------------------------------------------------- #

HIDDEN = 8
HIDDEN_2 = 12
LAYERS = 6


class _Out:
    """What a CLIP text encoder returns, indexable the way diffusers reads it."""

    def __init__(self, hidden_states, last_hidden_state, text_embeds=None):
        self.hidden_states = hidden_states
        self.last_hidden_state = last_hidden_state
        if text_embeds is not None:
            self.text_embeds = text_embeds
        self._first = last_hidden_state if text_embeds is None else text_embeds

    def __getitem__(self, i):
        return self._first if i == 0 else self.hidden_states[i]


class CLIPTextModel:
    """A text encoder whose every layer is identifiable by its own constant."""

    projection = False

    def __init__(self, hidden_size):
        import torch
        self.hidden_size = hidden_size
        self.dtype = torch.float32
        # A real final_layer_norm would change the values; marking them is
        # enough to prove it was applied on the SD 1.x clip_skip path.
        self.text_model = types.SimpleNamespace(
            final_layer_norm=lambda t: t * 100.0)

    def __call__(self, ids, output_hidden_states=False, **kw):
        import torch
        n = ids.shape[1]
        # hidden_states[i] is filled with i, so which layer was taken is
        # readable straight off the returned tensor.
        hs = tuple(torch.full((1, n, self.hidden_size), float(i))
                   for i in range(LAYERS))
        last = torch.full((1, n, self.hidden_size), -1.0)
        pooled = (torch.full((1, self.hidden_size), 7.0)
                  if self.projection else None)
        return _Out(hs, last, pooled)


class CLIPTextModelWithProjection(CLIPTextModel):
    """SDXL's second encoder: output[0] is the pooled vector, not the states."""

    projection = True


class T5EncoderModel:
    """What Flux and SD3 carry instead, with its own limit and conventions."""


def _fake_sd15_pipeline():
    pipe = types.SimpleNamespace()
    pipe.tokenizer = _FakeTokenizer()
    pipe.text_encoder = CLIPTextModel(HIDDEN)
    pipe.device = "cpu"
    return pipe


def _fake_sdxl_pipeline():
    pipe = _fake_sd15_pipeline()
    pipe.tokenizer_2 = _FakeTokenizer()
    pipe.text_encoder_2 = CLIPTextModelWithProjection(HIDDEN_2)
    return pipe


def test_sd15_is_supported_and_sdxl_is_supported():
    assert supports_weighted_embeds(_fake_sd15_pipeline())
    assert supports_weighted_embeds(_fake_sdxl_pipeline())


def test_a_t5_second_encoder_is_left_alone():
    """Flux and SD3 carry a T5 with its own limit and conventions."""
    pipe = _fake_sdxl_pipeline()
    pipe.text_encoder_2 = T5EncoderModel()
    assert not supports_weighted_embeds(pipe)
    assert encode_weighted_prompt(pipe, "a cat") is None


def test_a_pipeline_with_no_text_encoder_is_left_alone():
    assert not supports_weighted_embeds(types.SimpleNamespace())


def test_sd15_shape_grows_with_the_prompt():
    """77 tokens of room per window, concatenated -- not a cap."""
    pipe = _fake_sd15_pipeline()
    long_prompt = " ".join(f"tag{i}" for i in range(200))
    out = encode_weighted_prompt(pipe, long_prompt, "")
    assert out["prompt_embeds"].shape == (1, 3 * (WINDOW + 2), HIDDEN)


def test_the_negative_is_padded_to_the_same_length():
    """Guidance subtracts one from the other, so they must match."""
    pipe = _fake_sd15_pipeline()
    out = encode_weighted_prompt(
        pipe, " ".join(f"tag{i}" for i in range(200)), "blurry")
    assert out["prompt_embeds"].shape == out["negative_prompt_embeds"].shape


def test_sd15_clip_skip_picks_the_layer_and_normalises_it():
    pipe = _fake_sd15_pipeline()
    plain = encode_weighted_prompt(pipe, "a cat", "")
    # No clip_skip: the encoder's final output (marked -1), untouched.
    assert plain["prompt_embeds"].flatten()[0].item() == pytest.approx(-1.0)
    # clip_skip=2 means the penultimate layer -- hidden_states[-2], layer 4 of
    # 6 -- then the final layer norm, which the fake multiplies by 100.
    skipped = encode_weighted_prompt(pipe, "a cat", "", clip_skip=2)
    assert skipped["prompt_embeds"].flatten()[0].item() == pytest.approx(400.0)


@pytest.mark.parametrize("clip_skip,layer", [
    (None, 4),    # hidden_states[-2]: what SDXL is trained against
    (1,    5),    # hidden_states[-1]: the final layer
    (2,    4),    # hidden_states[-2]: what every Pony model card asks for
    (3,    3),
    (4,    2),
])
def test_clip_skip_counts_layers_from_the_end_as_users_do(clip_skip, layer):
    """``clip_skip`` is A1111's CLIP_stop_at_last_layers: 1 is the final layer,
    2 the penultimate, n the n-th from the end.

    Diffusers' argument of the same name counts something else -- layers
    skipped BEYOND the penultimate one SDXL already uses, so its SDXL path
    reads ``hidden_states[-(clip_skip + 2)]``. Passing a caller's 2 through to
    it read ``hidden_states[-4]``: two layers deeper than asked for, which on a
    Pony-family model is the difference between a photograph and mush. It
    shipped that way in 0.4.89 and this is what caught it.
    """
    pipe = _fake_sdxl_pipeline()
    out = encode_weighted_prompt(pipe, "a cat", "", clip_skip=clip_skip)
    assert out["prompt_embeds"].flatten()[0].item() == pytest.approx(float(layer))


def test_clip_skip_2_means_the_same_layer_on_both_families():
    """It is one number in one API: whichever family the model belongs to,
    clip_skip 2 is the penultimate layer. The old mapping made it
    hidden_states[-4] on SDXL and [-3] on SD 1.x -- two different depths from
    the same request."""
    sdxl = encode_weighted_prompt(_fake_sdxl_pipeline(), "a cat", "", clip_skip=2)
    sd15 = encode_weighted_prompt(_fake_sd15_pipeline(), "a cat", "", clip_skip=2)
    # layer 4 of 6 == hidden_states[-2] in both; SD 1.x also applies the norm
    assert sdxl["prompt_embeds"].flatten()[0].item() == pytest.approx(4.0)
    assert sd15["prompt_embeds"].flatten()[0].item() == pytest.approx(4.0 * 100)


def test_sdxl_concatenates_both_encoders_and_pools_the_second():
    pipe = _fake_sdxl_pipeline()
    out = encode_weighted_prompt(pipe, "a cat", "blurry")
    assert out["prompt_embeds"].shape == (1, WINDOW + 2, HIDDEN + HIDDEN_2)
    assert out["pooled_prompt_embeds"].shape == (1, HIDDEN_2)
    assert out["negative_pooled_prompt_embeds"].shape == (1, HIDDEN_2)


class CLIPTextModelNoProjection(CLIPTextModel):
    """A second encoder that offers no ``text_embeds``, but can still be pooled.

    Named for what it stands in for: ``supports_weighted_embeds`` identifies
    the encoder families by class name, so a fake called anything else is
    turned away before the pooling is ever reached.

    Stands in for anything other than a CLIPTextModelWithProjection landing in
    text_encoder_2: the pooled vector is not handed over, but the projection
    and the hidden states to build it from are both there.
    """

    projection = False

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        self.config = types.SimpleNamespace(eos_token_id=49407)
        # A real text_projection is a Linear; marking the output is enough to
        # prove it was applied rather than the raw hidden state being passed.
        self.text_projection = lambda v: v * 3.0


def test_sdxl_pooled_is_rebuilt_when_it_is_not_offered():
    """Giving up the weighting and the window count is a worse answer than
    pooling the way CLIP does: at the end-of-text token, then projected."""
    pipe = _fake_sdxl_pipeline()
    pipe.text_encoder_2 = CLIPTextModelNoProjection(HIDDEN_2)
    out = encode_weighted_prompt(pipe, "a cat", "blurry")
    assert out is not None, "fell back instead of rebuilding the pooled vector"
    assert out["pooled_prompt_embeds"].shape == (1, HIDDEN_2)
    # last_hidden_state is -1.0 in the fake, and the projection triples it
    assert out["pooled_prompt_embeds"].flatten()[0].item() == pytest.approx(-3.0)
    # and the per-token path is still the weighted, windowed one
    assert out["prompt_embeds"].shape == (1, WINDOW + 2, HIDDEN + HIDDEN_2)


def test_sdxl_falls_back_only_when_it_cannot_be_rebuilt():
    """No text_embeds and no projection: SDXL would raise on prompt_embeds
    without a pooled vector, so this returns None and the caller uses the
    pipeline's own encoder -- unweighted, but served."""
    pipe = _fake_sdxl_pipeline()
    pipe.text_encoder_2 = CLIPTextModel(HIDDEN_2)   # no projection at all
    assert supports_weighted_embeds(pipe)           # still looks encodable
    assert encode_weighted_prompt(pipe, "a cat", "") is None


def test_sdxl_never_returns_embeds_without_the_pooled_pair():
    """The invalid combination is the one that 500s instantly, so it must not
    be constructible: either all four keys, or None."""
    for second in (CLIPTextModelWithProjection(HIDDEN_2),
                   CLIPTextModelNoProjection(HIDDEN_2),
                   CLIPTextModel(HIDDEN_2)):
        pipe = _fake_sdxl_pipeline()
        pipe.text_encoder_2 = second
        out = encode_weighted_prompt(pipe, "a cat", "blurry")
        if out is None:
            continue
        assert {"prompt_embeds", "negative_prompt_embeds",
                "pooled_prompt_embeds", "negative_pooled_prompt_embeds"} <= set(out)


def test_sd15_needs_no_pooled_vector():
    """Only SDXL conditions on one; SD 1.x must not be held to it."""
    out = encode_weighted_prompt(_fake_sd15_pipeline(), "a cat", "")
    assert out is not None
    assert "pooled_prompt_embeds" not in out


def test_a_weight_changes_the_tokens_but_not_the_window_mean():
    """Scaling alone would shift the whole window's magnitude, which shows up
    as a brightness change that nobody asked for."""
    import torch
    pipe = _fake_sd15_pipeline()
    flat = encode_weighted_prompt(pipe, "cat dog", "")["prompt_embeds"]
    weighted = encode_weighted_prompt(pipe, "(cat:1.5) dog", "")["prompt_embeds"]
    assert not torch.allclose(flat, weighted)
    assert weighted.mean().item() == pytest.approx(flat.mean().item(), rel=1e-4)


def test_plain_mode_ignores_the_weights_end_to_end():
    import torch
    pipe = _fake_sd15_pipeline()
    a = encode_weighted_prompt(pipe, "(cat:1.5) dog", "", plain=True)["prompt_embeds"]
    b = encode_weighted_prompt(pipe, "cat dog", "", plain=True)["prompt_embeds"]
    # Same token count either way, and no scaling applied.
    assert a.shape == b.shape
    assert torch.allclose(a, b)


# --------------------------------------------------------------------------- #
#  the API: what a request can ask for, and what reaches the engine
# --------------------------------------------------------------------------- #

def test_the_image_request_takes_clip_skip_and_weighting():
    from codai.pydantic.imagerequest import ImageGenerationRequest
    r = ImageGenerationRequest(model="m", prompt="p", clip_skip=2,
                               prompt_weighting=False, negative_prompt="score_4")
    assert r.clip_skip == 2
    assert r.prompt_weighting is False
    assert r.negative_prompt == "score_4"


def test_defaults_leave_every_decision_to_the_model():
    from codai.pydantic.imagerequest import ImageGenerationRequest
    r = ImageGenerationRequest(model="m", prompt="p")
    assert r.clip_skip is None and r.prompt_weighting is None


@pytest.mark.parametrize("cls_name", ["ImageEditRequest", "ImageInpaintRequest"])
def test_the_edit_endpoints_accept_a_negative_prompt(cls_name):
    """They had no field for one, so an edit could only be steered positively."""
    from codai.api import images as images_mod
    cls = getattr(images_mod, cls_name)
    kw = {"model": "m", "prompt": "p", "image": "data:,x"}
    if cls_name == "ImageInpaintRequest":
        kw["mask"] = "data:,y"
    r = cls(negative_prompt="score_4, blurry", clip_skip=2, **kw)
    assert r.negative_prompt == "score_4, blurry"
    assert r.clip_skip == 2
    assert r.prompt_weighting is None


def test_the_sdcpp_path_no_longer_discards_the_negative_prompt():
    """The defect, as source: `negative_prompt=''` passed to the engine."""
    src = open("codai/api/images.py").read()
    assert "negative_prompt=''" not in src
    assert 'negative_prompt=""' not in src


def test_resolve_clip_skip_precedence():
    from codai.api import images as images_mod
    req = types.SimpleNamespace(clip_skip=3)
    model = {"clip_skip": 2}
    args = types.SimpleNamespace(image_clip_skip=4)
    assert images_mod._resolve_clip_skip(req, model, args) == 3
    assert images_mod._resolve_clip_skip(
        types.SimpleNamespace(clip_skip=None), model, args) == 2
    assert images_mod._resolve_clip_skip(
        types.SimpleNamespace(clip_skip=None), {}, args) == 4
    assert images_mod._resolve_clip_skip(
        types.SimpleNamespace(clip_skip=None), {}, types.SimpleNamespace()) is None


@pytest.mark.parametrize("value", [0, -1, "", "two", None])
def test_a_meaningless_clip_skip_means_the_default(value):
    from codai.api import images as images_mod
    assert images_mod._resolve_clip_skip(
        types.SimpleNamespace(clip_skip=value), {}, types.SimpleNamespace()) is None


def test_resolve_prompt_weighting_precedence():
    from codai.api import images as images_mod
    none = types.SimpleNamespace(prompt_weighting=None)
    assert images_mod._resolve_prompt_weighting(
        types.SimpleNamespace(prompt_weighting=False), {"prompt_weighting": True},
        types.SimpleNamespace()) is False
    assert images_mod._resolve_prompt_weighting(
        none, {"prompt_weighting": False}, types.SimpleNamespace()) is False
    assert images_mod._resolve_prompt_weighting(
        none, {}, types.SimpleNamespace(image_prompt_weighting=False)) is False
    # nothing says otherwise: on, as the ecosystem assumes
    assert images_mod._resolve_prompt_weighting(
        none, {}, types.SimpleNamespace()) is True


def test_clip_skip_is_part_of_the_embedding_cache_key():
    """Embeddings taken from different layers are different embeddings: sharing
    one cache entry would serve the first request's layer to all the rest."""
    from codai.api.images import _PromptEmbedCache
    c = _PromptEmbedCache()
    c.put("p", "n", "model#1|cs=2|w=1", {"prompt_embeds": "a"})
    assert c.get("p", "n", "model#1|cs=2|w=1") is not None
    assert c.get("p", "n", "model#1|cs=None|w=1") is None
    assert c.get("p", "n", "model#1|cs=2|w=0") is None


def test_both_knobs_are_configurable_per_model():
    """An OpenAI-API client cannot send clip_skip, so the model must carry it."""
    src = open("codai/admin/routes.py").read()
    assert '"clip_skip", "prompt_weighting",' in src
    ui = open("codai/admin/templates/models.html").read()
    assert "cfg-clip-skip" in ui and "cfg-prompt-weighting" in ui


def test_the_server_wide_defaults_round_trip_through_config():
    from codai.config import ImageConfig
    cfg = ImageConfig()
    assert cfg.clip_skip is None
    assert cfg.prompt_weighting is True


# --------------------------------------------------------------------------- #
#  the compatibility guarantee
# --------------------------------------------------------------------------- #

def test_a_short_plain_prompt_encodes_exactly_as_diffusers_did():
    """Nothing that worked before renders differently now.

    This is the whole reason the windowing is safe to turn on for everyone: for
    a prompt that fits, encoding it ourselves has to be *bit-identical* to what
    the pipeline produced, or every existing caller's images change under them.

    It is identical for a reason worth recording. diffusers tokenizes with
    ``padding="max_length"``, giving ``[bos] + tokens + [eos] + [pad]...``;
    these windows are ``[bos] + tokens + [pad]... + [eos]``. For a CLIP
    tokenizer the pad token *is* the end-of-text token, so the two id
    sequences are the same -- and if a tokenizer ever turns up where they
    differ, this test is what notices.

    Run against a real ``CLIPTextModel`` (tiny, random weights, built from a
    config so it needs no download), because the point is the model's own
    attention over the padding, which a stub cannot stand in for.
    """
    import torch
    from transformers import CLIPTextConfig, CLIPTextModel

    cfg = CLIPTextConfig(hidden_size=32, intermediate_size=37,
                         num_hidden_layers=5, num_attention_heads=4,
                         vocab_size=1000, max_position_embeddings=77,
                         bos_token_id=0, eos_token_id=999)
    torch.manual_seed(0)
    encoder = CLIPTextModel(cfg).eval()

    class _ClipLikeTokenizer(_FakeTokenizer):
        bos_token_id = 0
        eos_token_id = 999
        pad_token_id = 999          # CLIP pads with end-of-text, which is the key

        def __call__(self, text, add_special_tokens=True, truncation=True, **kw):
            ids = [1 + (abs(hash(w)) % 900) for w in str(text).split()]
            return types.SimpleNamespace(input_ids=ids)

    tok = _ClipLikeTokenizer()
    pipe = types.SimpleNamespace(tokenizer=tok, text_encoder=encoder, device="cpu")
    prompt = "a cat sitting on a mat"

    mine = encode_weighted_prompt(pipe, prompt, "")["prompt_embeds"]

    ids = tok(prompt, add_special_tokens=False, truncation=False).input_ids
    as_diffusers_does = [tok.bos_token_id] + ids + \
        [tok.pad_token_id] * (WINDOW + 2 - 1 - len(ids))
    with torch.no_grad():
        theirs = encoder(torch.tensor([as_diffusers_does]))[0]

    assert mine.shape == theirs.shape == (1, WINDOW + 2, 32)
    assert torch.equal(mine, theirs)
