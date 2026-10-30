"""Two reasons a 31B ran 14 of 60 layers on the CPU with the GPU at 6%.

Measured on zeiss 2026-10-30, one request, five minutes, fourteen cores:

    Loading Artemis-31B: need 22.7 GB VRAM, have 10.6 GB free — evicting
    [cosite-evict] released 4 embedding models
    On-request VRAM eviction: external releaser freed 14.2 GB     <- 22.9 GB free
    KV plan: 7.5 GB journalled for this model at n_ctx=32000 by an earlier load
    Auto-offload: model needs ~22.7 GB but only 22.9 GB free
                  — placing 46/60 layers on GPU, 14 on CPU
    ...
    KV measured: 1.0 GB at n_ctx=32000 — journalled for the next load

Eviction was never the problem: it freed 14.2 GB and handed the loader 22.9 GB.
Two separate defects then threw that away.

1. The journal key was `<path>@<n_ctx>`, so a 7.5 GB measurement taken before
   swa_full=False stayed reachable after it. The real KV was 1.0 GB. The
   inflated figure put the requirement at 22.7 GB instead of ~16.2 GB.

2. The sizing never asked for more room. It sampled free VRAM once, found
   itself 0.2 GB inside a 95% trigger, and gave away layers — while another
   engine still held VRAM it would have released on request.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.backends import kv_journal as J

VULKAN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'codai', 'backends', 'vulkan.py')

FULL = {'swa_full': True, 'cache_type_k': 'q4_0', 'cache_type_v': 'q4_0'}
WIN = {'swa_full': False, 'cache_type_k': 'q4_0', 'cache_type_v': 'q4_0'}
P = '/AI/guffcache/TheDrummer_Artemis-31B-v1.2-Q4_K_M.gguf'


@pytest.fixture
def journal(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr(J, '_PATH_OVERRIDE', os.path.join(d, 'kv.json'),
                        raising=False)
    return J


# ── 1. the journal key carries the KV settings ───────────────────────────────

def test_the_exact_stale_hit_is_now_a_miss(journal):
    """The incident, reproduced: 7.5 GB recorded under full SWA must not be
    served to a lookup made with swa_full=False."""
    journal.record(P, 32000, 7.5, journal.signature(FULL))
    assert journal.lookup(P, 32000, journal.signature(FULL)) == 7.5
    assert journal.lookup(P, 32000, journal.signature(WIN)) == 0.0


def test_the_new_measurement_is_then_found(journal):
    journal.record(P, 32000, 7.5, journal.signature(FULL))
    journal.record(P, 32000, 1.0, journal.signature(WIN))
    assert journal.lookup(P, 32000, journal.signature(WIN)) == 1.0
    # and the old one is still there under its own key, harmlessly
    assert journal.lookup(P, 32000, journal.signature(FULL)) == 7.5


@pytest.mark.parametrize('field,other', [
    ('swa_full', True),
    ('cache_type_k', 'q8_0'),
    ('cache_type_v', 'f16'),
    ('kv_offload', False),
])
def test_each_kv_setting_changes_the_signature(field, other):
    changed = dict(WIN)
    changed[field] = other
    assert J.signature(changed) != J.signature(WIN), field


def test_the_two_spellings_of_kv_offload_agree():
    """The loader reads kv_offload or offload_kqv; one must not key differently
    from the other, or a card written one way loses its measurement."""
    assert J.signature({'kv_offload': False}) == J.signature({'offload_kqv': False})


def test_an_unrelated_field_keeps_the_measurement():
    """Over-invalidating throws away every measurement and puts models back on
    the OOM-ladder discovery path, which is the cost the journal exists to
    avoid."""
    for field, val in (('alias', 'Artemis-31B'), ('n_batch', 256),
                       ('n_gpu_layers', -1), ('measured_vram_gb', 21.47)):
        changed = dict(WIN)
        changed[field] = val
        assert J.signature(changed) == J.signature(WIN), field


def test_n_ctx_still_separates_entries(journal):
    """It is in the key already; the signature must not collapse contexts."""
    sig = journal.signature(WIN)
    journal.record(P, 32000, 1.0, sig)
    journal.record(P, 8192, 0.3, sig)
    assert journal.lookup(P, 32000, sig) == 1.0
    assert journal.lookup(P, 8192, sig) == 0.3


def test_a_bad_config_still_yields_a_signature():
    """An unreadable config must orphan old entries, not fall back to the bare
    key and resurrect them."""
    for cfg in (None, {}, {'swa_full': object()}):
        assert isinstance(J.signature(cfg), str)
        assert J.signature(cfg) != ''


def test_every_call_site_passes_a_signature():
    src = open(VULKAN).read()
    assert 'kv_journal.lookup(\n' in src or 'kv_journal.lookup(' in src
    # No bare two-argument calls left.
    import re
    for m in re.finditer(r'kv_journal\.(record|lookup)\(([^)]*)\)', src, re.S):
        args = m.group(2)
        assert 'sig' in args or 'signature' in args, \
            f'{m.group(1)} called without a signature: {args[:80]}'


# ── 2. evict before spilling ─────────────────────────────────────────────────

def _sizing_block():
    src = open(VULKAN).read()
    i = src.index('# Spilling to CPU is a LAST resort')
    return src[i:src.index('except Exception as _off_e', i)]


def test_the_sizing_asks_for_more_vram_before_giving_away_layers():
    body = _sizing_block()
    assert '_evict_models_for_vram(_want)' in body, (
        'the sizing still spills without ever asking for more room')
    assert body.index('_evict_models_for_vram') < body.index('self.n_gpu_layers = _fit')


def test_it_re_samples_free_vram_after_evicting():
    """Asking without re-measuring would change nothing."""
    body = _sizing_block()
    assert '_pooled_free_vram_gb(' in body
    assert body.index('_evict_models_for_vram') < body.index('_pooled_free_vram_gb(')


def test_a_full_placement_is_possible_after_eviction():
    """The old code ran `min(_nlayers - 1, _fit)` unconditionally, so entering
    this branch guaranteed at least one CPU layer however much was freed."""
    body = _sizing_block()
    assert 'if _exp <= _free * 0.95:' in body
    assert 'keeping every layer on GPU' in body
    # and the spill is now the else-branch, not the only outcome
    assert body.index('if _exp <= _free * 0.95:') < body.index('self.n_gpu_layers = _fit')


def test_it_asks_for_what_a_full_placement_needs():
    """Evicting to the bare estimate would leave the same thin margin that
    triggered the spill."""
    body = _sizing_block()
    assert '_want = _exp * 1.20 + _headroom' in body


def test_a_failed_eviction_is_not_fatal():
    """A model that cannot get more room must still load, spilled, exactly as
    it did before."""
    body = _sizing_block()
    assert 'except Exception as _ev_e' in body
    assert 'could not ask for more VRAM' in body
    assert 'self.n_gpu_layers = _fit' in body       # the spill path survives


def test_the_message_says_eviction_was_already_tried():
    """So a reader does not conclude, as I did, that eviction never ran.

    Matched against the message with its line breaks collapsed: it is built
    from adjacent f-string literals, so the words a reader sees on one line do
    not appear contiguously in the source.
    """
    import re
    # Joined across adjacent f-string literals, so the quote/f/quote boundary
    # has to come out before the sentence reads as one.
    body = re.sub(r'"\s*f?"', '', re.sub(r'\s+', ' ', _sizing_block()))
    assert 'even after evicting what could be evicted' in body


def test_the_evictor_it_calls_skips_busy_models():
    """This call is only safe because _evict_models_for_vram consults
    _is_key_busy and waits for idle — and that check only became trustworthy
    for embedding models when external_busy was wired into the embeddings path
    earlier the same day."""
    mgr = open(os.path.join(os.path.dirname(VULKAN), '..', 'models',
                            'manager.py')).read()
    i = mgr.index('def _evict_models_for_vram')
    body = mgr[i:i + 4000]
    assert '_is_key_busy' in body and '_wait_until_idle' in body
