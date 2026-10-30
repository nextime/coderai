"""An image-only embedder must refuse text, not hang on it.

`/v1/embeddings` has two image-only families whose contract puts the image in
`input` rather than `image`:

    vpr          EigenPlaces, decoded in-process
    vpr-server   DINOv2-SALAD, handed to an isolated subprocess

The refusal for the OTHER image-only models is keyed on the backend name:

    elif backend in ('vision', 'dinov2cpp'):
        raise ValueError("this embedding model is image-only (no text tower) …")

so `dinov2` answers text with a clean 400. `vpr-server` had no such guard: it
passed the strings straight to a subprocess that could not answer and did not
reply. Measured on zeiss 2026-10-30, mid-sweep: four sentences sent to `salad`
produced no response in 180s and the sweep recorded a 500 with no error logged
anywhere, while `dinov2` given the identical request returned 400 immediately.

A hang is worse than a refusal: it holds a request slot and tells the caller
nothing.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.api.embeddings import _looks_like_image_ref

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'codai', 'api', 'embeddings.py')


# ── what counts as an image reference ────────────────────────────────────────

@pytest.mark.parametrize('ref', [
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==',
    'data:image/jpeg;base64,/9j/4AAQ',
    'http://example.com/a.png',
    'https://example.com/a.jpg?x=1',
])
def test_the_four_accepted_shapes_pass(ref):
    assert _looks_like_image_ref(ref) is True


def test_a_real_file_path_passes(tmp_path):
    p = tmp_path / 'x.png'
    p.write_bytes(b'\x89PNG\r\n\x1a\n')
    assert _looks_like_image_ref(str(p)) is True


def test_a_long_base64_blob_passes():
    assert _looks_like_image_ref('QUJD' * 40) is True


@pytest.mark.parametrize('text', [
    'a test sentence',
    'what is in this image?',
    'Qwen3-Embedding-4B',
    '', '   ', None, 123, [],
])
def test_text_and_nonsense_are_refused(text):
    assert _looks_like_image_ref(text) is False


def test_a_short_base64_string_is_treated_as_text():
    """'aGVsbG8=' decodes as base64 but is the word "hello". No picture is 8
    bytes, and treating every short token as an image would send text to the
    subprocess again."""
    assert _looks_like_image_ref('aGVsbG8=') is False


def test_a_path_that_does_not_exist_is_refused():
    assert _looks_like_image_ref('/no/such/file.png') is False


def test_an_absurdly_long_string_does_not_raise():
    """os.path.isfile raises rather than returning False past the OS name limit,
    and this runs on every input of every request."""
    assert _looks_like_image_ref('x' * 100000) in (True, False)


def test_it_never_decodes_to_check():
    """A real image must not be decoded twice — once to validate and once to
    embed. The check reads the prefix and the base64 alphabet only."""
    src = open(SRC).read()
    i = src.index('def _looks_like_image_ref')
    body = src[i:src.index('def _decode_image', i)]
    assert 'Image.open' not in body
    assert 'convert(' not in body


# ── the guard is wired, and placed before dispatch ───────────────────────────

def test_both_image_only_input_backends_are_guarded():
    src = open(SRC).read()
    assert "elif backend in ('vpr', 'vpr-server'):" in src, (
        'vpr-server is no longer guarded alongside vpr')


def test_the_check_happens_before_the_subprocess_is_handed_anything():
    """The whole point: vpr-server cannot refuse after dispatch, because the
    subprocess simply never replies."""
    src = open(SRC).read()
    i = src.index("elif backend in ('vpr', 'vpr-server'):")
    block = src[i:i + 1600]
    assert block.index('_looks_like_image_ref') < block.index('_embed_images('), \
        'text reaches the subprocess before being validated'
    assert block.index('_looks_like_image_ref') < block.index('_vpr_embed_images('), \
        'text reaches the in-process decoder before being validated'


def test_the_refusal_says_where_the_image_goes():
    """`input`, not `image` — the opposite of what the dinov2 message says, and
    a caller told the wrong field just fails differently."""
    src = open(SRC).read()
    i = src.index("elif backend in ('vpr', 'vpr-server'):")
    block = src[i:i + 1600]
    assert "image-only (no text tower)" in block
    assert "in 'input'" in block
    assert 'data URI' in block and 'base64' in block


def test_the_refusal_names_the_offending_input():
    """A batch of 4 with one bad entry should say which one."""
    src = open(SRC).read()
    i = src.index("elif backend in ('vpr', 'vpr-server'):")
    block = src[i:i + 1600]
    assert 'input[{_bad[0]}]' in block or '_bad[0]' in block


def test_the_other_image_only_backends_keep_their_refusal():
    """dinov2's guard was already correct; this change must not disturb it."""
    src = open(SRC).read()
    assert "elif backend in ('vision', 'dinov2cpp'):" in src
    assert "this embedding model is image-only (no text tower) — send 'image' " in src


def test_vpr_backends_are_still_listed_as_image_capable():
    """They must keep accepting images — the fix refuses text, not pictures."""
    src = open(SRC).read()
    i = src.index("if backend in ('clip', 'vision', 'qwenvl'")
    line = src[i:i + 200]
    assert "'vpr'" in line and "'vpr-server'" in line


# ── refused BEFORE the model is loaded ───────────────────────────────────────
#
# The first version of this guard sat inside the embed path, after the load. It
# stopped the hang but answered 400 in 42 seconds, because SALAD had to spawn
# its subprocess first. The call site of the mirror check records that exact
# lesson for images-to-a-text-model:
#
#   # Refuse an impossible request BEFORE loading anything. The same refusal
#   # used to come after the load: images sent to a text-only 8 GB embedder
#   # spent forty seconds loading it and then answered 400.
#
# so repeating it in the other direction was avoidable.

def test_the_model_is_identified_by_name_alone():
    """A name-only predicate is what makes a pre-load refusal possible."""
    from codai.api.embeddings import _takes_image_in_input
    assert _takes_image_in_input('salad') is True
    assert _takes_image_in_input('dinov2-salad') is True
    assert _takes_image_in_input('vpr') is True
    assert _takes_image_in_input('eigenplaces') is True
    assert _takes_image_in_input('Qwen3-Embedding-4B') is False
    assert _takes_image_in_input('dinov2-large') is False
    assert _takes_image_in_input(None) is False


def test_the_name_check_is_case_and_separator_insensitive():
    from codai.api.embeddings import _takes_image_in_input
    for n in ('SALAD', 'Salad', 'dinov2_salad', '  salad  '):
        assert _takes_image_in_input(n) is True, n


def test_it_reads_the_existing_id_sets_rather_than_a_new_list():
    """A second hand-kept list of salad/vpr aliases would drift from the ones
    the loaders actually dispatch on."""
    src = open(SRC).read()
    i = src.index('def _takes_image_in_input')
    body = src[i:src.index('def _looks_like_image_ref', i)]
    assert '_VPR_IDS' in body and '_SALAD_IDS' in body


#: The guard itself, not the function name. Searching for the bare name finds
#: the `def` line first, 1200 lines from the call and with no condition near it
#: -- the first version of these two tests asserted against the definition and
#: failed on correct code.
GUARD = 'if texts and not images and _takes_image_in_input(model_name):'


def test_the_refusal_happens_before_any_load():
    src = open(SRC).read()
    pre = src.index(GUARD)
    load = src.index('_load_embedding_locked(', pre)
    assert pre < load, 'the refusal still waits for the model to load'


def test_it_only_fires_when_there_is_no_image_at_all():
    """A request carrying a real image must not be refused, and a mixed request
    must reach the normal path."""
    assert GUARD in open(SRC).read()


def test_the_post_load_guard_is_still_there_as_a_backstop():
    """The pre-load check is keyed on the model NAME; a model reached by an
    alias the id sets do not know still has to be refused rather than hang."""
    src = open(SRC).read()
    assert "elif backend in ('vpr', 'vpr-server'):" in src
    i = src.index("elif backend in ('vpr', 'vpr-server'):")
    assert '_looks_like_image_ref' in src[i:i + 1600]
