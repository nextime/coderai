"""Two ways a GGUF load took the whole engine down with it.

1. A model configured ``cache_type_k/v`` and ``n_ctx=128000`` got a 47.7 GB KV
   cache placed in host RAM and the engine was SIGKILLed mid-load. The KV
   placement plan that exists to stop exactly that was skipped, because pinning a
   KV *type* was read as consent to whatever size resulted.
2. A quantizer that drops a checkpoint's NextN (multi-token-prediction) blocks
   without lowering ``block_count`` makes llama.cpp ask for a block the file does
   not have, which it reports as ``missing tensor 'blk.64.attn_norm.weight'`` —
   indistinguishable from a truncated download.
"""

import ast
import json
import os
import struct
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.backends import gguf_mtp, kv_journal


SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'codai', 'backends', 'vulkan.py')


# ── GGUF fixtures ────────────────────────────────────────────────────────────

def _write_gguf(path, meta: dict, tensor_names):
    """Minimal GGUF v3 header: metadata KV section + tensor index, no data."""
    def s(val):
        b = val.encode('utf-8')
        return struct.pack('<Q', len(b)) + b

    out = [b'GGUF', struct.pack('<I', 3),
           struct.pack('<Q', len(tensor_names)), struct.pack('<Q', len(meta))]
    for k, v in meta.items():
        out.append(s(k))
        if isinstance(v, str):
            out.append(struct.pack('<I', 8) + s(v))
        else:
            out.append(struct.pack('<I', 4) + struct.pack('<I', int(v)))
    for name in tensor_names:
        out.append(s(name))
        out.append(struct.pack('<I', 1))        # n_dims
        out.append(struct.pack('<Q', 16))       # dim 0
        out.append(struct.pack('<I', 0))        # ggml type
        out.append(struct.pack('<Q', 0))        # offset
    with open(path, 'wb') as f:
        f.write(b''.join(out))
    return path


def _blocks(lo, hi, arch_tensors=('attn_norm.weight', 'ffn_up.weight')):
    return [f'blk.{i}.{t}' for i in range(lo, hi) for t in arch_tensors]


@pytest.fixture
def tmpgguf(tmp_path):
    def make(name, meta, names):
        return _write_gguf(str(tmp_path / name), meta, names)
    return make


# ── 1. The MTP block override ────────────────────────────────────────────────

def test_quant_that_dropped_the_nextn_block_is_corrected(tmpgguf):
    """65 declared, 64 present, nextn=1 — the shape that proves the quantizer
    dropped the draft head."""
    p = tmpgguf('mtp.gguf',
                {'general.architecture': 'qwen35',
                 'qwen35.block_count': 65,
                 'qwen35.nextn_predict_layers': 1},
                ['token_embd.weight'] + _blocks(0, 64))
    assert gguf_mtp.block_count_override(p) == {'qwen35.block_count': 64,
                                                'qwen35.nextn_predict_layers': 0}


def test_the_override_also_zeroes_nextn_so_the_count_is_read_the_same_way(tmpgguf):
    """Builds disagree on whether block_count includes the NextN blocks. A build
    that subtracts them would read a corrected 64 as 63 and drop a real layer, so
    the override states both halves."""
    p = tmpgguf('mtp3.gguf',
                {'general.architecture': 'qwen35',
                 'qwen35.block_count': 65,
                 'qwen35.nextn_predict_layers': 1},
                ['token_embd.weight'] + _blocks(0, 64))
    over = gguf_mtp.block_count_override(p)
    assert over['qwen35.nextn_predict_layers'] == 0
    assert over['qwen35.block_count'] == 64


def test_intact_model_needs_no_override(tmpgguf):
    p = tmpgguf('plain.gguf',
                {'general.architecture': 'gemma4', 'gemma4.block_count': 60},
                ['token_embd.weight'] + _blocks(0, 60))
    assert gguf_mtp.block_count_override(p) is None


def test_a_model_that_declares_nextn_and_carries_it_is_left_alone(tmpgguf):
    """All 65 blocks present: nothing was dropped, so nothing is overridden."""
    p = tmpgguf('full.gguf',
                {'general.architecture': 'qwen35',
                 'qwen35.block_count': 65,
                 'qwen35.nextn_predict_layers': 1},
                _blocks(0, 65))
    assert gguf_mtp.block_count_override(p) is None


def test_a_genuinely_truncated_download_still_fails(tmpgguf):
    """Blocks missing from the MIDDLE are a broken file, not a dropped draft
    head. Reinterpreting that would turn a corrupt download into a model that
    loads and answers nonsense."""
    names = ['token_embd.weight'] + _blocks(0, 30) + _blocks(40, 64)
    p = tmpgguf('torn.gguf',
                {'general.architecture': 'qwen35',
                 'qwen35.block_count': 65,
                 'qwen35.nextn_predict_layers': 1},
                names)
    assert gguf_mtp.block_count_override(p) is None


def test_more_missing_than_nextn_declares_is_not_corrected(tmpgguf):
    """nextn=1 but three blocks short: the arithmetic doesn't add up, so the
    file is not the known case and must keep failing."""
    p = tmpgguf('short.gguf',
                {'general.architecture': 'qwen35',
                 'qwen35.block_count': 65,
                 'qwen35.nextn_predict_layers': 1},
                ['token_embd.weight'] + _blocks(0, 62))
    assert gguf_mtp.block_count_override(p) is None


def test_a_bare_draft_head_is_not_a_model(tmpgguf):
    """An MTP-only file carries just its own block (blk.48 of 49). It is a draft
    head for a base model, not something to serve, and must not be rewritten
    into a 1-layer model."""
    p = tmpgguf('draft.gguf',
                {'general.architecture': 'qwen4exp',
                 'qwen4exp.block_count': 49,
                 'qwen4exp.nextn_predict_layers': 1},
                _blocks(48, 49))
    assert gguf_mtp.block_count_override(p) is None


def test_non_gguf_file_is_ignored(tmp_path):
    p = tmp_path / 'notgguf.bin'
    p.write_bytes(b'\x00' * 64)
    assert gguf_mtp.read_layout(str(p)) == {}
    assert gguf_mtp.block_count_override(str(p)) is None


def test_missing_file_is_ignored():
    assert gguf_mtp.block_count_override('/nonexistent/none.gguf') is None
    assert gguf_mtp.block_count_override(None) is None


def test_explain_names_the_counts_and_the_architecture(tmpgguf):
    p = tmpgguf('mtp2.gguf',
                {'general.architecture': 'qwen35',
                 'qwen35.block_count': 65,
                 'qwen35.nextn_predict_layers': 1},
                ['token_embd.weight'] + _blocks(0, 64))
    text = gguf_mtp.explain(p)
    assert '65' in text and '64' in text and 'qwen35.block_count' in text


def test_loader_passes_the_override_to_llama_cpp():
    src = open(SRC).read()
    assert 'gguf_mtp.block_count_override(model_path)' in src
    assert "llama_kwargs['kv_overrides'] = _kv_over" in src


def test_loader_lets_a_config_override_win_over_the_automatic_one():
    """A hand-written kv_overrides entry for the same key must not be replaced."""
    src = open(SRC).read()
    assert 'if _mtp_over and not (set(_mtp_over) & set(_kv_over)):' in src


def test_the_fix_can_be_switched_off_per_model():
    assert "_raw_cfg.get('mtp_block_fix', True)" in open(SRC).read()


# ── 2. A pinned KV type no longer disables the fits-check ────────────────────

def _plan_block():
    """The KV placement plan's source, by structure rather than line window: a
    fixed window broke twice before as the file moved around it."""
    src = open(SRC).read()
    start = src.index('# ── KV-cache placement plan')
    end = src.index('# Batch size (llama.cpp -b / n_batch)', start)
    return src[start:end]


def test_type_pin_and_placement_pin_are_tracked_separately():
    block = _plan_block()
    assert '_type_pinned = (_ck is not None or _cv is not None)' in block
    assert '_place_pinned = (_kv_off is not None)' in block


def test_the_plan_runs_even_when_the_config_pinned_something():
    """The regression: `and not _explicit` on this condition is what let a
    47.7 GB host KV through on a model that pinned q8_0."""
    block = _plan_block()
    assert 'if _meas > 0 and _ctx > 0:' in block
    assert 'if _meas > 0 and _ctx > 0 and not _explicit:' not in block


def test_a_pinned_type_is_never_requantized():
    """The plan may move the KV and may drop swa_full, but overriding a type the
    operator chose is the one thing it must not do."""
    block = _plan_block()
    assert '_ladder = () if (_placed or _type_pinned) else _KV_FALLBACK_TYPES' in block


def test_a_pinned_placement_is_never_moved():
    block = _plan_block()
    assert block.count('if not _place_pinned:') >= 1
    assert "if _where == 'RAM' and not _place_pinned:" in block


def test_a_pinned_type_that_cannot_fit_is_refused_by_name():
    """Refusing beats being OOM-killed, and the message has to say that the pin
    is what removed the remaining lever."""
    block = _plan_block()
    assert 'if _type_pinned:' in block
    assert 'cache_type_k/v pinned to' in block
    assert 'unpin cache_type_k/v' in block


def test_the_unconditional_refusal_survives():
    block = _plan_block()
    assert 'Refusing the load instead of' in block
    assert 'letting the OOM killer take the engine down' in block


# ── 3. The journal that survives the kill ────────────────────────────────────

def test_journal_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_journal, '_PATH_OVERRIDE',
                        str(tmp_path / 'kv.json'), raising=False)
    assert kv_journal.record('/m/a.gguf', 128000, 47.73) is True
    assert kv_journal.lookup('/m/a.gguf', 128000) == 47.73


def test_journal_is_keyed_by_context_length(tmp_path, monkeypatch):
    """The same model at a different n_ctx is a different measurement; reusing
    one for the other is how a 128k figure would refuse a 4k load."""
    monkeypatch.setattr(kv_journal, '_PATH_OVERRIDE',
                        str(tmp_path / 'kv.json'), raising=False)
    kv_journal.record('/m/a.gguf', 128000, 47.7)
    kv_journal.record('/m/a.gguf', 8192, 3.1)
    assert kv_journal.lookup('/m/a.gguf', 128000) == 47.7
    assert kv_journal.lookup('/m/a.gguf', 8192) == 3.1
    assert kv_journal.lookup('/m/a.gguf', 32000) == 0.0


def test_journal_separates_models(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_journal, '_PATH_OVERRIDE',
                        str(tmp_path / 'kv.json'), raising=False)
    kv_journal.record('/m/a.gguf', 4096, 1.0)
    kv_journal.record('/m/b.gguf', 4096, 2.0)
    assert kv_journal.lookup('/m/a.gguf', 4096) == 1.0
    assert kv_journal.lookup('/m/b.gguf', 4096) == 2.0


def test_journal_rejects_nonsense(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_journal, '_PATH_OVERRIDE',
                        str(tmp_path / 'kv.json'), raising=False)
    assert kv_journal.record('', 4096, 1.0) is False
    assert kv_journal.record('/m/a.gguf', 0, 1.0) is False
    assert kv_journal.record('/m/a.gguf', 4096, 0) is False
    assert kv_journal.record('/m/a.gguf', 4096, -3) is False


def test_journal_never_raises_on_an_unwritable_path(monkeypatch):
    """A journal is a convenience. It must not fail a load that would work."""
    monkeypatch.setattr(kv_journal, '_PATH_OVERRIDE',
                        '/proc/definitely/not/writable/kv.json', raising=False)
    assert kv_journal.record('/m/a.gguf', 4096, 1.0) is False
    assert kv_journal.lookup('/m/a.gguf', 4096) == 0.0


def test_journal_survives_a_corrupt_file(tmp_path, monkeypatch):
    jp = tmp_path / 'kv.json'
    jp.write_text('{ this is not json')
    monkeypatch.setattr(kv_journal, '_PATH_OVERRIDE', str(jp), raising=False)
    assert kv_journal.lookup('/m/a.gguf', 4096) == 0.0
    assert kv_journal.record('/m/a.gguf', 4096, 1.5) is True
    assert kv_journal.lookup('/m/a.gguf', 4096) == 1.5


def test_journal_write_is_atomic(tmp_path, monkeypatch):
    """A partial write would leave the plan reading a truncated number."""
    jp = tmp_path / 'kv.json'
    monkeypatch.setattr(kv_journal, '_PATH_OVERRIDE', str(jp), raising=False)
    kv_journal.record('/m/a.gguf', 4096, 1.0)
    kv_journal.record('/m/b.gguf', 4096, 2.0)
    assert json.loads(jp.read_text())            # parses as a whole
    assert not list(tmp_path.glob('*.tmp'))      # no leftovers


def test_the_host_resident_size_is_recorded_from_the_log_callback():
    """Recorded during the load, not after it: a KV cache this size is one the
    OOM killer may take the process for while it is still loading."""
    src = open(SRC).read()
    start = src.index('def _install_layer_log_callback(')
    end = src.index('return _cb  # caller must hold this reference', start)
    cb = src[start:end]
    assert 'kv_journal.record(model_path, n_ctx, _total)' in cb
    assert '_KV_OBSERVED["cpu"] > 1.0' in cb


def test_the_callback_is_given_the_model_identity():
    src = open(SRC).read()
    assert '_install_layer_log_callback(model_path, int(self.n_ctx or 0))' in src


def test_the_plan_reads_the_journal_when_the_config_has_no_measurement():
    block = _plan_block()
    assert 'kv_journal.lookup(model_path, _ctx)' in block


def test_config_measurement_still_wins_over_the_journal():
    """measured_kv_gb is configuration; the journal is a cache. The journal is
    consulted only when the config said nothing."""
    block = _plan_block()
    cfg = block.index("_raw_cfg.get('measured_kv_gb')")
    jrnl = block.index('kv_journal.lookup(model_path, _ctx)')
    assert cfg < jrnl
    assert 'if _meas <= 0:' in block


def test_success_path_journals_too():
    src = open(SRC).read()
    assert 'kv_journal.record(model_path, self.last_kv_ctx,' in src


def test_the_copy_this_by_hand_hint_is_gone():
    """It told whoever was watching to write the number down. Nobody is watching
    when the engine is killed, which is the case that mattered."""
    assert 'record as measured_kv_gb' not in open(SRC).read()


# ── 4. Both modules import without the heavy backend ─────────────────────────

def test_modules_have_no_heavy_imports():
    """These run on the load path and inside a ctypes callback; pulling torch or
    llama_cpp in there would be a new way to fail."""
    for mod in ('gguf_mtp.py', 'kv_journal.py'):
        path = os.path.join(os.path.dirname(SRC), mod)
        tree = ast.parse(open(path).read())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split('.')[0])
        assert not (names & {'torch', 'llama_cpp', 'transformers', 'diffusers'}), names


# ── 5. The knobs are reachable from the model page ───────────────────────────
#
# A lever the loader reads but the model page cannot set is a lever that only
# exists for whoever edits models.json by hand. All four of these decide whether
# a load fits in host RAM at all, so they belong in the UI — and the savable-list
# in the admin routes is where a field silently stops persisting.

ROUTES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'codai', 'admin', 'routes.py')
MODELS_HTML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'codai', 'admin', 'templates', 'models.html')

KV_FIELDS = ('kv_autoplan', 'swa_full', 'measured_kv_gb', 'mtp_block_fix')


@pytest.mark.parametrize('field', KV_FIELDS)
def test_field_is_savable(field):
    assert f'"{field}"' in open(ROUTES).read(), f'{field} is dropped on save'


@pytest.mark.parametrize('field', KV_FIELDS)
def test_field_is_sent_by_the_form(field):
    assert f'{field}:' in open(MODELS_HTML).read(), f'{field} is never posted'


@pytest.mark.parametrize('el', ('cfg-kv-autoplan', 'cfg-swa-full',
                                'cfg-measured-kv', 'cfg-mtp-fix'))
def test_control_exists_is_populated_and_is_gguf_only(el):
    html = open(MODELS_HTML).read()
    assert f'id="{el}"' in html, f'{el} has no control'
    assert f"getElementById('{el}')" in html, f'{el} is never read or written'
    assert f"show('{el}-row'" in html or f"show('{el.replace('cfg-', 'cfg-')}-row'" in html \
        or f"'{el}-row'" in html, f'{el} row is not gated to GGUF models'


def test_the_loader_reads_each_field_it_offers():
    """The other half of the round trip: the UI must not offer a setting the
    loader ignores."""
    src = open(SRC).read()
    for field in ('kv_autoplan', 'swa_full', 'measured_kv_gb', 'mtp_block_fix'):
        assert f"'{field}'" in src, f'{field} is offered but never read'


def test_measured_kv_accepts_both_stored_shapes():
    """It has been written by hand as {n_ctx: gb} and is now journalled as a
    plain number; the form must not render one of them as [object Object]."""
    html = open(MODELS_HTML).read()
    assert "typeof mk === 'object'" in html
