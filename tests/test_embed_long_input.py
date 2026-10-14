"""A long document embedded as its opening, with nothing saying so.

Only the GGUF path ever chunked. Every other backend passed `truncation=True`
and returned a confident UNIT vector of roughly the first page:

    embeddings.py   truncation=True, max_length=1800        # GME-Qwen2-VL
                    truncation=True, max_length=max_length  # bge-m3 native
                    model.encode(texts, …)                  # sentence-transformers

Not an error — a wrong answer that looks right, which is the failure mode this
codebase keeps paying for. Now long inputs are windowed, embedded, pooled and
RE-NORMALIZED, and the response says it happened.

Tunable per-model -> env -> default, like the admission gate:
  embed_long_input  chunk | truncate | error   (default chunk)
  embed_chunk_pool  mean | first               (default mean)
"""

import math
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import embeddings as em                        # noqa: E402


# A toy tokenizer: one token per word, so the arithmetic in these tests is
# visible rather than inferred from a real vocab.
def _enc(t):
    return t.split()


def _dec(ids):
    return " ".join(ids)


def _words(n):
    return " ".join(f"w{i}" for i in range(n))


# ------------------------------------------------------------- the windows #
def test_a_short_text_is_one_window():
    assert em._token_windows(_words(10), 100, _enc, _dec) == [_words(10)]


def test_a_long_text_is_split_to_the_limit():
    w = em._token_windows(_words(250), 100, _enc, _dec)
    assert len(w) == 3
    assert [len(_enc(x)) for x in w] == [100, 100, 50]


def test_the_windows_cover_the_whole_text_in_order():
    """Nothing dropped, nothing repeated — the point of chunking is the tail."""
    text = _words(250)
    w = em._token_windows(text, 100, _enc, _dec)
    assert " ".join(w) == text


def test_an_exact_multiple_does_not_produce_an_empty_window():
    w = em._token_windows(_words(200), 100, _enc, _dec)
    assert len(w) == 2
    assert all(x.strip() for x in w)


def test_a_window_that_re_encodes_longer_is_shrunk():
    """A detokenize->retokenize round trip is NOT token-count-stable: specials
    and BOS are re-added. The GGUF path learned this by tripping llama.cpp's
    GGML_ASSERT; every backend now gets the guard."""
    def _enc_inflating(t):
        # Pretend re-encoding adds 5 specials to any non-empty text.
        toks = t.split()
        return toks + ["<s>"] * 5 if toks else toks
    w = em._token_windows(_words(300), 100, _enc_inflating, _dec)
    assert w, "shrinking gave up entirely"
    for piece in w:
        assert len(_enc_inflating(piece)) <= 100, piece


def test_a_single_untokenizable_token_does_not_loop_forever():
    """Pathological input must end, not spin. Silence would be worse than a
    short answer here."""
    def _enc_huge(t):
        return ["x"] * 1000 if t else []
    w = em._token_windows("whatever", 10, _enc_huge, _dec)
    assert isinstance(w, list)


# --------------------------------------------------------------- the pooling #
def test_one_vector_passes_through_untouched():
    assert em._pool_vectors([[0.6, 0.8]], "mean") == [0.6, 0.8]


def test_mean_pooling_renormalizes():
    """Mean-pooling unit vectors does NOT give a unit vector, and cosine
    similarity on a non-unit vector is quietly wrong. The re-normalization is
    part of the contract, not a tidy-up."""
    v = em._pool_vectors([[1.0, 0.0], [0.0, 1.0]], "mean")
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9)
    assert math.isclose(v[0], v[1], rel_tol=1e-9)


def test_first_pooling_keeps_the_opening_window():
    v = em._pool_vectors([[1.0, 0.0], [0.0, 1.0]], "first")
    assert v == [1.0, 0.0]


def test_pooling_nothing_is_empty_not_a_crash():
    assert em._pool_vectors([], "mean") == []


# ------------------------------------------------------------- the dispatch #
def _batch_of(dim=2):
    """An embedder that returns a distinct unit vector per window."""
    def _f(ws):
        out = []
        for i, _ in enumerate(ws):
            v = [0.0] * dim
            v[i % dim] = 1.0
            out.append(v)
        return out
    return _f


def test_every_window_goes_in_one_batch():
    """A 40-page document must not become 40 round trips."""
    calls = []

    def _batch(ws):
        calls.append(len(ws))
        return [[1.0, 0.0] for _ in ws]
    em._embed_long([_words(250), _words(250)], 100, _enc, _dec, _batch,
                   "chunk", "mean")
    assert calls == [6], f"expected one call with 6 windows, got {calls}"


def test_counts_are_reported_per_input():
    _, counts = em._embed_long([_words(10), _words(250)], 100, _enc, _dec,
                               _batch_of(), "chunk", "mean")
    assert counts == [1, 3]


def test_each_input_gets_exactly_one_vector():
    vecs, _ = em._embed_long([_words(10), _words(250), _words(40)], 100,
                             _enc, _dec, _batch_of(), "chunk", "mean")
    assert len(vecs) == 3
    assert all(len(v) == 2 for v in vecs)


def test_truncate_mode_keeps_only_the_first_window():
    """The pre-0.3.72 behaviour, kept reachable and now DELIBERATE: for a
    short-text encoder a 40-page input is usually a caller bug."""
    calls = []

    def _batch(ws):
        calls.append(len(ws))
        return [[1.0, 0.0] for _ in ws]
    _, counts = em._embed_long([_words(250)], 100, _enc, _dec, _batch,
                               "truncate", "mean")
    assert counts == [1]
    assert calls == [1]


def test_error_mode_refuses_instead_of_pooling():
    with pytest.raises(ValueError) as e:
        em._embed_long([_words(250)], 100, _enc, _dec, _batch_of(),
                       "error", "mean")
    assert "embed_long_input=error" in str(e.value)
    assert "3 windows" in str(e.value)


def test_error_mode_lets_a_fitting_input_through():
    vecs, counts = em._embed_long([_words(10)], 100, _enc, _dec, _batch_of(),
                                  "error", "mean")
    assert counts == [1] and len(vecs) == 1


def test_no_inputs_is_not_a_crash():
    assert em._embed_long([], 100, _enc, _dec, _batch_of()) == ([], [])


# --------------------------------------------------------------- the settings #
def test_the_default_is_chunk_and_mean(monkeypatch):
    monkeypatch.delenv("CODERAI_EMBED_LONG_INPUT", raising=False)
    monkeypatch.delenv("CODERAI_EMBED_CHUNK_POOL", raising=False)
    assert em._embed_chunk_cfg(None) == ("chunk", "mean")


def test_the_env_can_override(monkeypatch):
    monkeypatch.setenv("CODERAI_EMBED_LONG_INPUT", "truncate")
    monkeypatch.setenv("CODERAI_EMBED_CHUNK_POOL", "first")
    assert em._embed_chunk_cfg(None) == ("truncate", "first")


def test_a_nonsense_setting_falls_back_to_the_default(monkeypatch):
    """A typo must not silently pick a third behaviour."""
    monkeypatch.setenv("CODERAI_EMBED_LONG_INPUT", "chunky")
    assert em._embed_chunk_cfg(None)[0] == "chunk"


def test_a_per_model_setting_beats_the_env(monkeypatch):
    monkeypatch.setenv("CODERAI_EMBED_LONG_INPUT", "chunk")

    class _MM:
        @staticmethod
        def _config_for_model(m):
            return {"embed_long_input": "truncate"}
    monkeypatch.setattr(em, "multi_model_manager", _MM)
    assert em._embed_chunk_cfg("some-model")[0] == "truncate"


def test_a_broken_model_config_does_not_break_embedding(monkeypatch):
    class _MM:
        @staticmethod
        def _config_for_model(m):
            raise RuntimeError("no such model")
    monkeypatch.setattr(em, "multi_model_manager", _MM)
    assert em._embed_chunk_cfg("x") == ("chunk", "mean")


# ------------------------------------------------------- it is reported #
def test_the_response_reports_chunking():
    src = (ROOT / "codai/api/embeddings.py").read_text()
    assert "resp.chunked = {" in src
    assert 'if any(c > 1 for c in (_chunks or [])):' in src


def test_the_sentence_transformers_path_uses_it():
    src = (ROOT / "codai/api/embeddings.py").read_text()
    st = src.split("if backend == 'sentence_transformers':")[1].split("elif backend ==")[0]
    assert "_embed_long(" in st


def test_the_gme_text_path_uses_it():
    src = (ROOT / "codai/api/embeddings.py").read_text()
    blk = src.split("def _embed_texts(")[1].split("def _embed_images(")[0]
    gme = blk.split("elif backend == 'qwenvl':")[1].split("elif backend ==")[0]
    assert "_embed_long(" in gme


def test_the_clip_family_is_left_truncating_on_purpose():
    """CLIP-style encoders take 77 tokens BY DESIGN -- they embed captions, not
    documents. Pooling forty windows of a contract into a CLIP vector would be
    meaningless, so truncation there is correct rather than a bug. Asserted so
    that 'chunk everywhere' is not read as 'chunk where it makes no sense'."""
    src = (ROOT / "codai/api/embeddings.py").read_text()
    blk = src.split("def _embed_texts(")[1].split("def _embed_images(")[0]
    clip = blk.split("elif backend == 'clip':")[1].split("elif backend ==")[0]
    assert "_embed_long(" not in clip
