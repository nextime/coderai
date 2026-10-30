"""A failed download must not become a permanent cache hit.

`download_model` on a URL wrote straight to its final cache path:

    with open(cached_path, 'wb') as f:          # created before any byte arrived
        for chunk in response.iter_content(...):

with no try/except and no .partial, and the cache check that guards it tested
existence only:

    if os.path.exists(cached_path):
        return cached_path

So any interruption left a file that was then returned for ever as a valid
cached model. Measured on zeiss 2026-10-30 in one cache directory:

  * 188 zero-byte `<sha256-of-url>_<name>.gguf` entries, accruing ~14 a day
    since 2026-10-16, against 2 non-empty files;
  * and one of those two was a 9.72 GB `ltx-2.5-22b-distilled-Q4_K_M.gguf`
    where the real file is 13.58 GB -- a truncated download that the
    existence-only check would have served as a complete model.

The empty ones break the load loudly. The truncated one is worse: it loads.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'codai', 'models', 'cache', '__init__.py')


def _url_branch():
    """The URL-download branch of download_model, by structure not line number."""
    src = open(SRC).read()
    start = src.index("# It's a URL - use coderai cache")
    end = src.index('def list_cached_models', start)
    return src[start:end]


# ── the cache check ──────────────────────────────────────────────────────────

def test_a_zero_byte_entry_is_not_a_cache_hit():
    body = _url_branch()
    i = body.index('if os.path.exists(cached_path):')
    guard = body[i:i + 700]
    assert 'os.path.getsize(cached_path) > 0' in guard, (
        'the cache check still tests existence only, so a failed download is '
        'returned for ever as a valid model')


def test_a_zero_byte_entry_is_removed_rather_than_left_to_rot():
    body = _url_branch()
    i = body.index('if os.path.exists(cached_path):')
    guard = body[i:i + 700]
    assert 'os.remove(cached_path)' in guard
    # And removing it must not be able to fail the request.
    assert 'except OSError' in guard


# ── the download ─────────────────────────────────────────────────────────────

def test_the_download_goes_to_a_partial_path():
    body = _url_branch()
    assert "partial_path = cached_path + '.partial'" in body
    assert "open(partial_path, 'wb')" in body
    assert "open(cached_path, 'wb')" not in body, (
        'still writing directly to the final path: an interruption leaves a '
        'file that looks cached')


def test_the_rename_is_atomic_and_last():
    body = _url_branch()
    assert 'os.replace(partial_path, cached_path)' in body
    # The rename must come after the loop that writes the bytes.
    assert body.index('for chunk in response.iter_content') < \
        body.index('os.replace(partial_path, cached_path)')


def test_an_empty_response_is_an_error_not_a_cached_model():
    body = _url_branch()
    assert 'if downloaded == 0:' in body
    assert 'raise IOError' in body


def test_a_truncated_transfer_is_rejected():
    """The 9.72 GB case. content-length said more than arrived, and the file was
    kept and served."""
    body = _url_branch()
    assert 'downloaded < total_size' in body
    assert 'truncated download' in body


def test_the_partial_is_cleaned_up_on_any_failure():
    body = _url_branch()
    i = body.index("partial_path = cached_path + '.partial'")
    tail = body[i:i + 2000]
    assert 'except BaseException:' in tail, (
        'Exception is not enough: a KeyboardInterrupt or SystemExit mid-download '
        'is how several of these were created')
    assert 'os.remove(partial_path)' in tail
    assert 'raise' in tail                      # and the caller still learns


def test_cleanup_failure_does_not_mask_the_real_error():
    body = _url_branch()
    i = body.index('except BaseException:')
    tail = body[i:i + 500]
    assert 'except OSError' in tail
    assert tail.rstrip().endswith('raise') or '\n            raise' in tail


# ── the module still imports and the helper still resolves ───────────────────

def test_the_cache_dir_resolution_is_unchanged():
    """This fix must not move anyone's cache: a changed location is a silent
    full re-download for every existing install."""
    from codai.models.cache import get_model_cache_dir
    src = open(SRC).read()
    i = src.index('def get_model_cache_dir')
    body = src[i:i + 500]
    assert "os.environ.get('CODERAI_CACHE_DIR')" in body
    assert 'default_model_cache_dir()' in body
    assert isinstance(get_model_cache_dir(), str)
