"""An embedding request must be visible to the evictor, or the engine segfaults.

`_is_key_busy()` is what stops eviction unloading a model mid-request, and its
own docstring states the hazard: "Evicting a busy model would move its tensors to
CPU mid-forward-pass". It reads two sources — the instance pool's ref counts, and
`external_busy`.

An embedding request appeared in NEITHER. It takes no pool ref, and nothing on
that path called `external_busy`, whose only caller in the whole codebase was the
lipsync engine. So to the evictor an embedding model was permanently idle.

Measured on zeiss 2026-10-30 during a capability sweep:

    12:27:41  Loading Qwen3.6-14B-A3B: need 9.3 GB, have 4.8 GB free — evicting
    12:27:41  [req] POST /v1/embeddings model=Qwen3-Embedding-4B
    12:27:42  Response status: 200
    12:27:42  On-request VRAM eviction: unloading ACTIVE model
              'embedding:…Qwen3-Embedding-4B-Q4_K_M.gguf'  backend.model=None
    12:27:44  engine#1 exited (code -11)          <- SIGSEGV

code -11 is a segfault, not the -9 OOM-kill: the native model was freed under a
running request. Every other model on that engine died with it, and the 502s on
/v1/embeddings that had been appearing for hours were this.
"""

import ast
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(ROOT, 'codai', 'api', 'embeddings.py')
MGR = os.path.join(ROOT, 'codai', 'models', 'manager.py')


# ── the mechanism works, and is keyed the way eviction reads it ──────────────

def test_external_busy_makes_a_key_busy():
    from codai.models.manager import MultiModelManager
    m = MultiModelManager.__new__(MultiModelManager)
    m._external_busy = {}
    m._external_busy_lock = threading.RLock()
    m.model_pools = {}
    key = "embedding:/AI/guffcache/x.gguf"
    assert m._is_key_busy(key) is False
    with m.external_busy(key):
        assert m._is_key_busy(key) is True
    assert m._is_key_busy(key) is False


def test_it_nests_so_concurrent_requests_do_not_release_early():
    """Two in-flight embedding requests on one model: the first to finish must
    not mark it idle while the second is still running."""
    from codai.models.manager import MultiModelManager
    m = MultiModelManager.__new__(MultiModelManager)
    m._external_busy = {}
    m._external_busy_lock = threading.RLock()
    m.model_pools = {}
    key = "embedding:x"
    with m.external_busy(key):
        with m.external_busy(key):
            assert m._is_key_busy(key) is True
        assert m._is_key_busy(key) is True, "inner exit released the outer hold"
    assert m._is_key_busy(key) is False


def test_it_releases_even_when_the_request_raises():
    """An embedding request that 400s or 500s must not leave the model pinned
    busy for ever — that would block every later eviction."""
    from codai.models.manager import MultiModelManager
    m = MultiModelManager.__new__(MultiModelManager)
    m._external_busy = {}
    m._external_busy_lock = threading.RLock()
    m.model_pools = {}
    key = "embedding:x"
    with pytest.raises(ValueError):
        with m.external_busy(key):
            raise ValueError("boom")
    assert m._is_key_busy(key) is False


# ── the embeddings path uses it ──────────────────────────────────────────────

def test_the_embeddings_path_marks_the_model_busy():
    src = open(EMB).read()
    assert 'multi_model_manager.external_busy(model_key)' in src, (
        "the embeddings path does not mark its model busy — eviction will free "
        "the native model under a live request")


def test_the_guard_wraps_the_inference_rather_than_the_load():
    """The window to protect is "from having the model to being finished with
    it". Guarding only the load would leave every inference call exposed."""
    tree = ast.parse(open(EMB).read())
    fn = next(n for n in ast.walk(tree)
              if getattr(n, 'name', None) == '_run_embeddings')
    withs = [n for n in ast.walk(fn)
             if isinstance(n, ast.With)
             and 'external_busy' in ast.dump(n.items[0].context_expr)]
    assert withs, "no external_busy block inside _run_embeddings"
    guard = withs[0]
    # Everything from the guard to the end of the function is inside it.
    assert guard.end_lineno >= fn.end_lineno - 1, (
        "the guard does not cover the rest of the function; some inference "
        "call sites are still unprotected")


def test_every_inference_call_site_is_inside_the_guard():
    """Seven call sites and several early returns, which is why it is one block
    and not a wrapper per call."""
    tree = ast.parse(open(EMB).read())
    fn = next(n for n in ast.walk(tree)
              if getattr(n, 'name', None) == '_run_embeddings')
    guard = next(n for n in ast.walk(fn)
                 if isinstance(n, ast.With)
                 and 'external_busy' in ast.dump(n.items[0].context_expr))
    inside = {id(n) for n in ast.walk(guard)}
    # Exact names only. Matching substrings flagged `_may_embed_images` —- a
    # name/config capability check that runs BEFORE the model is loaded and
    # touches no tensors — as an unprotected inference call.
    INFERENCE = {'_embed_texts', '_embed_images', '_bge_m3_response'}
    seen = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and n.func.id in INFERENCE:
            seen.add(n.func.id)
            assert id(n) in inside, (
                f"{n.func.id} at line {n.lineno} is outside the busy guard")
    assert seen, "found no inference calls at all — this test has stopped checking"


def test_the_key_is_the_one_eviction_checks():
    """request_model builds `<model_type>:<resolved_name>` for a non-text model,
    which is the key the evictor logs and _is_key_busy looks up. Marking busy
    under any other spelling would be a no-op that looks like a fix."""
    mgr = open(MGR).read()
    assert 'model_key = f"{model_type}:{resolved_name}"' in mgr
    emb = open(EMB).read()
    assert "model_key = model_info['model_key']" in emb
    # And the guard uses exactly that variable.
    assert 'external_busy(model_key)' in emb


def test_the_evictor_still_consults_both_sources():
    """If _is_key_busy ever stops reading external_busy, this fix silently
    stops working."""
    mgr = open(MGR).read()
    i = mgr.index('def _is_key_busy')
    body = mgr[i:i + 900]
    assert '_external_busy' in body
    assert 'ref_counts' in body
