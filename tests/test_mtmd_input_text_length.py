"""Every image sent to a GGUF vision embedder answered 500.

Caught during a live sweep on 2026-10-30, reported by the 5xx logging added the
same evening:

    [radeon] mtmd_tokenize: error: number of media markers in text (0) does not
             match number of bitmaps (1)
    [radeon] [http] 500 on POST /v1/embeddings:
             'Embedding failed: mtmd_tokenize failed (2)'

The marker was in the prompt the whole time. `mtmd_default_marker()` returns
`b'<__media__>'` and `_prompt()` interpolates it -- both verified against the
running engine. What was wrong is that `mtmd_input_text` carries a `text_len`
field beside its `c_char_p`, and coderai built the struct with the constructor

    _M.mtmd_input_text(text=_prompt(marker).encode(), add_special=True, …)

which leaves `text_len` at 0. Measured in the container: that form gives
`text_len == 0` for a 38-byte prompt. llama.cpp read a zero-length prompt, found
no markers in it, and refused the one bitmap it had been given.

llama_cpp's own chat handler does it field by field and keeps a NAMED reference
to the encoded bytes:

    input_text_bytes = text.encode("utf-8")
    input_text.text = input_text_bytes
    input_text.text_len = len(input_text_bytes)

Both halves matter. `text` is a `c_char_p`, so a temporary is free to be
collected while the struct still points at it -- a second defect in the same
two lines, and one that would have been intermittent rather than total.
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = io.open(os.path.join(ROOT, 'codai/api/embeddings.py'), encoding='utf-8').read()
LOG = io.open(os.path.join(ROOT, 'codai/api/log.py'), encoding='utf-8').read()


def _image_branch():
    """The mtmd call site: the image half of `_llama_vl_embed`."""
    i = SRC.index('def _llama_vl_embed')
    body = SRC[i:SRC.index('\ndef ', i + 10)]
    return body[body.index("img = it.get('image')"):]


# ── the fix ──────────────────────────────────────────────────────────────────

def test_text_len_is_set():
    b = _image_branch()
    assert 'itext.text_len = len(_itext_bytes)' in b, (
        'text_len is unset, so llama.cpp reads a zero-length prompt and finds '
        'no media marker in it'
    )


def test_the_constructor_form_is_not_used():
    """`mtmd_input_text(text=…)` silently leaves text_len at 0."""
    b = _image_branch()
    assert 'mtmd_input_text(\n' not in b
    assert 'mtmd_input_text(text=' not in b
    assert '_M.mtmd_input_text()' in b


def test_the_encoded_bytes_are_held_in_a_named_local():
    """`text` is a c_char_p: a temporary may be collected while the struct still
    points at it."""
    b = _image_branch()
    assert '_itext_bytes = _prompt(marker).encode' in b
    assert b.index('_itext_bytes =') < b.index('itext.text = _itext_bytes')


def test_the_reference_outlives_the_tokenize_call():
    b = _image_branch()
    assert b.index('_itext_bytes =') < b.index('mtmd_tokenize(')


def test_the_special_token_flags_survive():
    """The prompt frame is chat markup; parsing it as plain text would tokenise
    `<|im_start|>` as literal characters."""
    b = _image_branch()
    assert 'itext.add_special = True' in b
    assert 'itext.parse_special = True' in b


def test_the_marker_is_still_what_is_interpolated():
    """The half that was never broken, pinned so a future prompt change cannot
    quietly drop it."""
    i = SRC.index('def _llama_vl_embed')
    body = SRC[i:SRC.index('\ndef ', i + 10)]
    assert 'marker = _M.mtmd_default_marker()' in body
    assert '_prompt(marker)' in body


def test_the_struct_really_behaves_this_way():
    """The diagnosis, not the source. Skipped where llama_cpp cannot load (no
    GPU driver in a plain test environment); it was measured in the container.
    """
    mtmd = pytest.importorskip('llama_cpp.mtmd_cpp')
    prompt = b'<|im_start|>user\n<__media__><|im_end|>'
    bad = mtmd.mtmd_input_text(text=prompt, add_special=True, parse_special=True)
    assert bad.text_len == 0, (
        'the constructor form now sets text_len; this test records WHY the '
        'field-by-field form is used and can be retired if so')
    good = mtmd.mtmd_input_text()
    good.text = prompt
    good.text_len = len(prompt)
    assert good.text_len == len(prompt)


# ── the logging gap that hid it ──────────────────────────────────────────────

def test_the_request_log_reports_the_image_count():
    """`model=gme-… inputs=4 chars=84` was read as a text-only request, so a
    failure on the IMAGE path looked impossible and the search went after text
    being mistaken for pictures."""
    assert 'images=' in LOG


def test_the_count_covers_both_shapes():
    from codai.api.log import _summarise
    assert 'images=2' in _summarise(
        {'model': 'gme', 'input': ['a', 'b'], 'image': ['x', 'y']})
    assert 'images=1' in _summarise({'model': 'gme', 'image': 'data:image/png;base64,A'})


def test_a_text_only_request_says_nothing_about_images():
    from codai.api.log import _summarise
    assert 'images=' not in _summarise({'model': 'gme', 'input': ['a', 'b']})


def test_the_log_still_reports_shape_and_not_content():
    """It is a request log on a server that handles other people's documents."""
    from codai.api.log import _summarise
    out = _summarise({'model': 'gme', 'input': ['secret text here'],
                      'image': ['data:image/png;base64,SECRETBYTES']})
    assert 'secret text here' not in out
    assert 'SECRETBYTES' not in out
    assert 'inputs=1' in out and 'images=1' in out
