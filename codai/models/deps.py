# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.

"""Which python package a model needs, answered without loading the model.

Two models were advertised by ``/v1/models`` for as long as they existed and
could never serve a single request. `vosk-en`/`vosk-it` answered every
transcription with ``No module named 'vosk'``; `geoclip` and `geoclip-location`
answered 500 in half a second, because they share one loader and its import was
missing. Both packages were declared in requirements.txt, and neither reached an
incrementally-updated image, because the app overlay never installs from there.

The catalogue could not tell, because nothing asked. The check that would have
is one line -- ``importlib.util.find_spec`` -- and the reason to put it here
rather than at each call site is that it has to be answerable in two places at
once: the listing, so an operator sees it before a client does, and the load
path, so the error names the package instead of being a bare ImportError from
three frames down.

Rules this follows, both learned the hard way:

* It never imports the package and never loads the model. ``find_spec`` reads
  the finders, not the module. An availability probe that loads what it is
  probing turned ``/v1/models`` into a 350 MB loader once already.
* It is conservative in one direction only. A model whose requirement cannot be
  determined is reported as fine: hiding a model that works is worse than
  listing one that does not, and this is advice, never a gate on serving.
"""

import importlib.util
import threading
from typing import Optional

#: (predicate on the lowered model id, required module, what it is for).
#: Deliberately short. Everything in it is a HARD import in the MAIN venv --
#: a dependency that lives in an isolated worker venv (SALAD, NeMo-Canary,
#: pyannote) is not listed, because its absence from this venv says nothing and
#: a false "unavailable" is the one failure mode that costs a working model.
_REQUIREMENTS = (
    (lambda n: n.startswith('geoclip'), 'geoclip',
     'GeoCLIP image/coordinate embeddings'),
    (lambda n: 'vosk' in n, 'vosk',
     'the Vosk CPU speech-to-text backend'),
)

_LOCK = threading.Lock()
_CACHE: dict = {}


def _installed(module: str) -> bool:
    """Whether ``module`` is importable, asked once per process.

    Memoised because the listing is polled: `scan_cache_dir` on a polled
    endpoint cost 27 s on a 900 GB cache once, and while `find_spec` is nothing
    like that, a per-request filesystem walk of sys.path is still a per-request
    filesystem walk.
    """
    with _LOCK:
        if module in _CACHE:
            return _CACHE[module]
    try:
        ok = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # A namespace package or a broken parent raises rather than returning
        # None. Unknown means "do not claim it is missing".
        ok = True
    except Exception:
        ok = True
    with _LOCK:
        _CACHE[module] = ok
    return ok


def missing_dependency(model_id) -> Optional[str]:
    """The python module this model needs and does not have, or None.

    None means "nothing known to be missing", which includes "no requirement is
    declared for this model" -- see the conservative rule above.
    """
    n = str(model_id or '').strip().lower()
    if not n:
        return None
    for matches, module, _what in _REQUIREMENTS:
        try:
            if matches(n) and not _installed(module):
                return module
        except Exception:
            continue
    return None


def unavailable_reason(model_id) -> Optional[str]:
    """A sentence an operator can act on, or None when nothing is missing."""
    module = missing_dependency(model_id)
    if not module:
        return None
    what = next((w for m, mod, w in _REQUIREMENTS if mod == module), module)
    return (f"the python package '{module}' is not installed in this image, so "
            f"this model cannot load ({what}). It is declared in "
            f"requirements.txt; an incrementally-updated image also needs it in "
            f"packaging/linux/Dockerfile.update.")


def reset_cache() -> None:
    """Forget what was found. For tests, and for after an install at run time."""
    with _LOCK:
        _CACHE.clear()
