"""A model cannot be "available" when its python package is absent.

`vosk-en`/`vosk-it` answered every transcription with `No module named 'vosk'`;
`geoclip`/`geoclip-location` answered 500 in half a second. Both were listed by
`/v1/models` the whole time, because nothing ever asked whether they could load.
Both packages were declared in requirements.txt and neither reached an
incrementally-updated image.

And the companion fault, from the same evening: a footprint line that said

    21.47 GB total … of which 19.52 GB was offloaded to host RAM this load;
    n_gpu_layers=46

for a model with 46 of 60 layers on the GPU and ~15 GB resident on the card.
19.52 GB is the process's RSS growth, and llama.cpp mmaps the whole file, so a
GGUF grows RSS by about its file size whatever the layer split. The total is
usable; the decomposition is not, and a reader chasing a CPU-spill problem is
exactly who reads that line.
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.models import deps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MGR = io.open(os.path.join(ROOT, 'codai/models/manager.py'), encoding='utf-8').read()
EMB = io.open(os.path.join(ROOT, 'codai/api/embeddings.py'), encoding='utf-8').read()
SYS = io.open(os.path.join(ROOT, 'codai/system_app.py'), encoding='utf-8').read()


@pytest.fixture(autouse=True)
def _clean():
    deps.reset_cache()
    yield
    deps.reset_cache()


# ── the check itself ─────────────────────────────────────────────────────────

def test_a_present_package_is_not_reported_missing():
    """`json` stands in for an installed requirement: the point is that a
    satisfied requirement produces no complaint."""
    deps._CACHE.clear()
    assert deps._installed('json') is True


def test_an_absent_package_is_reported():
    assert deps._installed('definitely_not_a_real_module_xyz') is False


def test_a_model_with_no_declared_requirement_is_left_alone():
    for n in ('bge-m3', 'Qwen3-Embedding-4B', 'gemma-2-9b', '', None):
        assert deps.missing_dependency(n) is None, n


def test_the_two_models_that_prompted_this_are_matched(monkeypatch):
    monkeypatch.setattr(deps, '_installed', lambda m: False)
    assert deps.missing_dependency('geoclip') == 'geoclip'
    assert deps.missing_dependency('geoclip-location') == 'geoclip'
    assert deps.missing_dependency('vosk-it') == 'vosk'
    assert deps.missing_dependency('/AI/guffcache/vosk/vosk-model-small-it-0.22') == 'vosk'


def test_nothing_is_reported_when_the_packages_are_installed(monkeypatch):
    monkeypatch.setattr(deps, '_installed', lambda m: True)
    for n in ('geoclip', 'geoclip-location', 'vosk-it'):
        assert deps.missing_dependency(n) is None, n


def test_the_reason_names_the_package_and_where_to_add_it(monkeypatch):
    monkeypatch.setattr(deps, '_installed', lambda m: False)
    why = deps.unavailable_reason('geoclip')
    assert "'geoclip'" in why
    assert 'requirements.txt' in why
    assert 'Dockerfile.update' in why, (
        'the overlay is the place it was missing from; saying only '
        'requirements.txt sends the reader where it already is')


def test_it_never_imports_the_package_or_loads_the_model():
    """An availability probe that loads what it is probing turned /v1/models
    into a 350 MB loader once already."""
    src = io.open(os.path.join(ROOT, 'codai/models/deps.py'), encoding='utf-8').read()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith('#'))
    assert 'find_spec' in code
    assert 'import_module' not in code and '__import__' not in code


def test_the_answer_is_memoised():
    """The listing is polled; a per-request walk of sys.path is still a
    per-request walk of sys.path."""
    calls = []
    real = deps.importlib.util.find_spec

    def counting(name):
        calls.append(name)
        return real(name)

    deps.importlib.util.find_spec = counting
    try:
        deps._installed('json')
        deps._installed('json')
        deps._installed('json')
    finally:
        deps.importlib.util.find_spec = real
    assert len(calls) == 1, f'find_spec called {len(calls)} times'


def test_an_unresolvable_name_is_treated_as_present():
    """Conservative in one direction only: hiding a model that works is the
    worse mistake, so anything undeterminable is reported as fine."""
    real = deps.importlib.util.find_spec
    deps.importlib.util.find_spec = lambda n: (_ for _ in ()).throw(ValueError('boom'))
    try:
        assert deps._installed('whatever') is True
    finally:
        deps.importlib.util.find_spec = real


def test_only_main_venv_dependencies_are_declared():
    """A package that lives in an ISOLATED worker venv (SALAD, NeMo-Canary,
    pyannote) is absent from this venv by design, so declaring it here would
    report a working model as unavailable."""
    mods = {m for _p, m, _w in deps._REQUIREMENTS}
    for forbidden in ('pytorch_lightning', 'nemo', 'nemo_toolkit', 'pyannote',
                      'pyannote.audio', 'ultralytics'):
        assert forbidden not in mods, forbidden


# ── wired into both places ───────────────────────────────────────────────────

def test_the_request_path_refuses_before_loading():
    i = EMB.index('models_deps.unavailable_reason')
    load = EMB.index('_load_embedding_locked(', i)
    assert i < load, 'the dependency check runs after the load'


def test_the_request_path_answers_503_not_500():
    """The model is configured and the deployment is incomplete; the request was
    fine. A 500 says the request broke the server."""
    i = EMB.index('models_deps.unavailable_reason')
    block = EMB[i:i + 700]
    assert 'status_code=503' in block


def test_the_listing_annotates_rather_than_hides():
    assert 'unavailable_reason' in SYS
    i = SYS.index('_deps.unavailable_reason')
    block = SYS[max(0, i - 900):i + 500]
    assert 'entry["unavailable_reason"]' in block
    # The model stays in the list.
    assert 'out.remove' not in block and 'del out' not in block


def test_the_listing_never_fails_because_of_the_check():
    i = SYS.index('_deps.unavailable_reason')
    block = SYS[max(0, i - 900):i + 700]
    assert 'except Exception' in block


def test_the_field_exists_on_the_published_schema():
    from codai.pydantic.textrequest import ModelInfo
    m = ModelInfo(id='x')
    assert m.unavailable_reason is None
    assert ModelInfo(id='x', unavailable_reason='nope').unavailable_reason == 'nope'


# ── the footprint label ──────────────────────────────────────────────────────

def test_the_host_figure_states_which_of_the_three_it_is():
    """Three different things land in `measured_ram` and only one is offloaded
    weights: a per-module measurement, a GGUF's whole file size with no split
    measured, and the process's RSS growth."""
    assert "ram_kind = 'unsplit'" in MGR
    assert "ram_kind = 'process-delta'" in MGR
    assert 'host RSS grew' in MGR
    assert 'NOT the offloaded weights' in MGR
    assert 'GPU/host split was NOT measured here' in MGR


def test_the_mmap_reason_is_recorded_where_the_number_is_made():
    i = MGR.index('ram_kind = None')
    head = MGR[max(0, i - 2000):i]
    assert 'MMAPS' in head or 'mmap' in head.lower()


def test_a_real_per_module_measurement_keeps_the_plain_wording():
    """fp.weights_cpu_gb IS offloaded weights; this must not relabel it."""
    i = MGR.index('_host = f"{measured_ram:.2f} GB was offloaded to host RAM')
    assert i > 0


def test_the_tasks_page_only_prints_a_measured_offload_size():
    i = MGR.index('if measured_ram and ram_kind is None:')
    block = MGR[i:i + 400]
    assert 'CPU offload {measured_ram:.1f} GB' in block
    assert 'elif offloaded:' in block


def test_the_arithmetic_is_untouched():
    """The TOTAL drove a sizing decision that was correct, and under-reporting a
    model's footprint is how eviction stops freeing room and the forward OOMs.
    Only the label changes."""
    assert 'measured_vram = round(gpu_gb + (measured_ram or 0.0) + reserve_gb, 3)' in MGR
