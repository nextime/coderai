"""A file this build cannot read should say so before it costs anything.

Two GGUFs on one afternoon could not be loaded, and both failed late:

  * Qwen3.8-Flash-Next, architecture ``qwen4exp``, which this llama.cpp does not
    implement. llama.cpp names it ("unknown model architecture") but only after
    the load is attempted.
  * a 27B "Flash" quant that passed every check a careful operator would make --
    right architecture, all 65 blocks present including the MTP one, an mmproj
    beside it -- and stored 402 of its 866 tensors in ggml type 143, a format
    from a fork. NOTHING in the metadata announces that; the type ids live in
    the tensor index, which no error message reads back to you.

The second one is what this module catches, and it catches it before eviction:
a model that cannot possibly load has no business turning other models out of
the card first.
"""

import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.backends import gguf_mtp, gguf_preflight

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_gguf(path, meta, tensors):
    """Minimal GGUF v3 header. ``tensors`` is [(name, ggml_type), ...]."""
    def s(val):
        b = val.encode('utf-8')
        return struct.pack('<Q', len(b)) + b

    out = [b'GGUF', struct.pack('<I', 3),
           struct.pack('<Q', len(tensors)), struct.pack('<Q', len(meta))]
    for k, v in meta.items():
        out.append(s(k))
        if isinstance(v, str):
            out.append(struct.pack('<I', 8) + s(v))
        else:
            out.append(struct.pack('<I', 4) + struct.pack('<I', int(v)))
    for name, ttype in tensors:
        out.append(s(name))
        out.append(struct.pack('<I', 1))
        out.append(struct.pack('<Q', 16))
        out.append(struct.pack('<I', int(ttype)))
        out.append(struct.pack('<Q', 0))
    with open(path, 'wb') as f:
        f.write(b''.join(out))
    return path


@pytest.fixture(autouse=True)
def _clear_caches():
    gguf_mtp._LAYOUT_CACHE.clear()
    yield
    gguf_mtp._LAYOUT_CACHE.clear()


@pytest.fixture
def gguf(tmp_path):
    def make(name, tensors, arch='qwen35', **meta):
        m = {'general.architecture': arch}
        m.update(meta)
        return _write_gguf(str(tmp_path / name), m, tensors)
    return make


# ── the exact check ──────────────────────────────────────────────────────────

def test_a_fork_quant_is_named_with_its_type_and_count(gguf):
    p = gguf('prism.gguf', [('token_embd.weight', 0)]
             + [(f'blk.0.w{i}', 143) for i in range(402)])
    bad = gguf_preflight.unsupported_tensor_types(p, 42)
    assert bad == {143: 402}


def test_a_standard_quant_passes(gguf):
    p = gguf('plain.gguf', [('token_embd.weight', 0), ('blk.0.a', 12),
                            ('blk.0.b', 14), ('blk.0.c', 8)])
    assert gguf_preflight.unsupported_tensor_types(p, 42) == {}


def test_bf16_is_allowed_although_the_binding_omits_its_constant(gguf):
    """THE trap. llama_cpp's GGML_TYPE_* constants are an incomplete mirror of
    the compiled enum: it defines 0-3, 6-29 and 39-42, so 30..38 have no
    constant -- and 30 is BF16, which files we serve today use and which loads
    perfectly. A check written as "absent from the binding's constants" would
    have rejected every one of them. Only ids at or above GGML_TYPE_COUNT are
    provably outside the enum.
    """
    p = gguf('bf16.gguf', [('token_embd.weight', 30)] * 96)
    assert gguf_preflight.unsupported_tensor_types(p, 42) == {}


@pytest.mark.parametrize('tid', [30, 31, 35, 38, 41])
def test_every_id_below_the_bound_is_left_to_llama_cpp(tid, gguf):
    """Including the deprecated holes (4, 5) and the undefined 31-38. An id the
    build cannot handle but which IS inside its enum gets refused by llama.cpp
    in its own words; inventing a verdict here could block a working model."""
    p = gguf(f'id{tid}.gguf', [('blk.0.a', tid)])
    assert gguf_preflight.unsupported_tensor_types(p, 42) == {}


def test_the_bound_comes_from_the_caller_not_from_a_list_here(gguf):
    """So it tracks the loaded library. On a build with more types, 143 would
    stop being a problem without editing this module."""
    p = gguf('future.gguf', [('blk.0.a', 143)])
    assert gguf_preflight.unsupported_tensor_types(p, 42) == {143: 1}
    assert gguf_preflight.unsupported_tensor_types(p, 200) == {}


@pytest.mark.parametrize('bound', [0, -1, None, 'x'])
def test_a_missing_or_nonsense_bound_checks_nothing(bound, gguf):
    """No bound means no verdict, not a verdict of 'unsupported'."""
    p = gguf('x.gguf', [('blk.0.a', 143)])
    assert gguf_preflight.unsupported_tensor_types(p, bound) == {}


def test_an_unreadable_file_is_not_called_unsupported(tmp_path):
    (tmp_path / 'junk.gguf').write_bytes(b'\x00' * 32)
    assert gguf_preflight.unsupported_tensor_types(
        str(tmp_path / 'junk.gguf'), 42) == {}
    assert gguf_preflight.unsupported_tensor_types('/no/such.gguf', 42) == {}


# ── the message ──────────────────────────────────────────────────────────────

def test_the_explanation_is_actionable(gguf):
    p = gguf('prism2.gguf', [('token_embd.weight', 0)]
             + [(f'blk.0.w{i}', 143) for i in range(402)])
    text = gguf_preflight.explain_tensor_types(
        p, gguf_preflight.unsupported_tensor_types(p, 42), 42)
    assert '402' in text and '403' in text          # affected / total
    assert 'type 143' in text
    assert 'GGML_TYPE_COUNT=42' in text
    assert 'standard format' in text                # what to do instead


def test_the_architecture_is_readable_for_the_load_log(gguf):
    p = gguf('arch.gguf', [('blk.0.a', 12)], arch='qwen4exp')
    assert gguf_preflight.architecture(p) == 'qwen4exp'
    assert gguf_preflight.architecture('/no/such.gguf') == ''


# ── where it runs ────────────────────────────────────────────────────────────

def test_the_check_runs_before_eviction():
    """The whole point. Placed after the already-resident short-circuit and
    before the VRAM need is computed, so nothing is turned out of the card for
    a model that cannot load."""
    src = open(os.path.join(ROOT, 'codai', 'models', 'manager.py')).read()
    call = src.index('_gguf_unreadable_reason(model_key, resolved_name, requested_id)')
    evict = src.index('evicting models to make room')
    need = src.index('needed_gb = self._get_model_used_vram_gb(')
    assert call < need < evict


def test_it_degrades_to_nothing_without_llama_cpp():
    """manager.py is imported by the front too, and the front has no CUDA
    libraries, so `import llama_cpp` raises there. That must cost the check,
    not the request."""
    src = open(os.path.join(ROOT, 'codai', 'models', 'manager.py')).read()
    start = src.index('def _gguf_unreadable_reason(')
    body = src[start:src.index('def _model_entry_for(', start)]
    assert 'import llama_cpp as _lc' in body
    assert 'except Exception:' in body
    assert "return ''" in body
    # Lazily, never at module import.
    assert 'import llama_cpp' not in src[:start]


def test_only_local_gguf_files_are_judged():
    """A remote/pod-served model, a non-GGUF checkpoint or a path that is not on
    this disk must pass straight through."""
    src = open(os.path.join(ROOT, 'codai', 'models', 'manager.py')).read()
    start = src.index('def _gguf_unreadable_reason(')
    body = src[start:src.index('def _model_entry_for(', start)]
    assert ".lower().endswith('.gguf')" in body
    assert '_os.path.isfile(path)' in body


def test_the_loader_logs_the_architecture():
    src = open(os.path.join(ROOT, 'codai', 'backends', 'vulkan.py')).read()
    assert '_pf.architecture(model_path)' in src
    assert 'arch         :' in src


def test_the_preflight_adds_no_fourth_gguf_parser():
    """Three header parsers in this package is already two too many; the
    tensor-type histogram rides along on the walk gguf_mtp already does."""
    src = open(os.path.join(ROOT, 'codai', 'backends', 'gguf_preflight.py')).read()
    assert 'from codai.backends.gguf_mtp import read_layout' in src
    assert "b'GGUF'" not in src
    assert 'struct' not in src


def test_the_shared_reader_still_answers_the_mtp_question(gguf):
    """Extending read_layout must not disturb what it was built for."""
    p = gguf('mtp.gguf',
             [(f'blk.{i}.attn_norm.weight', 12) for i in range(64)],
             arch='qwen35', **{'qwen35.block_count': 65,
                               'qwen35.nextn_predict_layers': 1})
    assert gguf_mtp.block_count_override(p) == {'qwen35.block_count': 64,
                                                'qwen35.nextn_predict_layers': 0}
    assert gguf_mtp.read_layout(p)['tensor_types'] == {12: 64}


# ── a verdict must not be swallowed as a warning ─────────────────────────────

def test_the_verdict_has_its_own_exception_type():
    """The on-demand load path turns a failed VRAM reservation into a printed
    warning and carries on -- correctly, because a reservation that could not be
    made is often still a load that works. This is not that, and a verdict
    swallowed into a warning reappears a minute later as the original confusing
    failure. Same treatment the host-RAM refusal gets, for the same reason."""
    from codai.backends.gguf_preflight import GgufUnreadable
    assert issubclass(GgufUnreadable, RuntimeError)
    e = GgufUnreadable('x', path='/a.gguf', types={143: 402})
    assert e.path == '/a.gguf' and e.types == {143: 402}


def test_the_swallowing_caller_re_raises_it():
    src = open(os.path.join(ROOT, 'codai', 'models', 'manager.py')).read()
    i = src.index('self.ensure_vram_for(model_name, requested_id=requested_id)')
    window = src[i:i + 700]
    # The specific re-raise comes BEFORE the catch-all that prints a warning.
    assert 'except GgufUnreadable:' in window
    assert window.index('except GgufUnreadable:') < window.index('except Exception as _ev_e:')
    assert 'raise' in window[window.index('except GgufUnreadable:'):
                             window.index('except Exception as _ev_e:')]


def test_it_is_reported_as_not_implemented_not_as_try_again():
    """501, not 503. Evicting and retrying cannot change the answer, so a client
    that retries a 503 forever must not be invited to."""
    src = open(os.path.join(ROOT, 'codai', 'api', 'app.py')).read()
    i = src.index('async def gguf_unreadable_handler')
    body = src[i:i + 1400]
    assert 'status_code=501' in body
    assert 'unsupported_model_format' in body
    assert 'tensor_types' in body          # the actionable part
    assert '@app.exception_handler(GgufUnreadable)' in src


def test_the_handler_is_registered_once_for_every_route():
    """The refusal can come from any path that loads a model, so it is handled
    centrally rather than caught per route -- as the host-RAM one is."""
    src = open(os.path.join(ROOT, 'codai', 'api', 'app.py')).read()
    assert src.count('@app.exception_handler(GgufUnreadable)') == 1
