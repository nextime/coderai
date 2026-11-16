"""Two training jobs submitted together both died on the same import.

    14:50:24  job-4883184e3eb5: FAILED: ImportError: cannot import name
              'AutoTokenizer' from 'transformers'
    14:50:24  job-2899e56a8838: FAILED: (the same)

Same second, two jobs, on a package that imports perfectly a moment later --
transformers 5.12.1, `from transformers import AutoTokenizer` fine from a
shell in the same container. It is not a broken install: transformers resolves
its public names through a lazy module, and that machinery is not safe against
two threads taking it at once. Each text job runs in its own thread
(`asyncio.to_thread`), so submitting two together races the FIRST import.

The cost of the fix is one lock acquisition per job. The cost of not having it
is a failure that reads as a broken dependency and sends you to pip.
"""

import inspect
import threading

import pytest

from codai.api import loras


def test_the_lock_exists():
    assert isinstance(loras._TRANSFORMERS_IMPORT_LOCK,
                      type(threading.Lock()))


def test_the_import_is_taken_under_it():
    src = inspect.getsource(loras._text_train_blocking)
    i = src.index("with _TRANSFORMERS_IMPORT_LOCK:")
    assert "from transformers import AutoTokenizer" in src[i:i + 200]


def test_the_lock_is_not_held_for_the_work():
    """Serialising the import is cheap; serialising the dataset pass would
    make two jobs take turns for minutes."""
    src = inspect.getsource(loras._text_train_blocking)
    lines = src.splitlines()
    i = next(n for n, l in enumerate(lines)
             if "with _TRANSFORMERS_IMPORT_LOCK:" in l)
    indent = len(lines[i]) - len(lines[i].lstrip())
    # Every line more deeply indented than the `with` is inside it. Counted by
    # INDENTATION rather than by a character window: my first version measured
    # newlines in a fixed slice, which ran past the block and failed on code
    # that was already correct.
    inside = []
    for line in lines[i + 1:]:
        if not line.strip():
            continue
        if (len(line) - len(line.lstrip())) <= indent:
            break
        inside.append(line.strip())
    assert inside == ["from transformers import AutoTokenizer"], inside


def test_the_reason_is_recorded_where_the_lock_is_taken():
    """Otherwise the next person removes it as pointless ceremony."""
    src = inspect.getsource(loras._text_train_blocking)
    # The note travelled with the import when it moved INSIDE the deferred
    # dataset pass -- it had to move, because sitting before the distributed
    # dispatch it made the rental wait minutes for torch to load.
    i = src.index("_TRANSFORMERS_IMPORT_LOCK")
    note = src[max(0, i - 900):i + 200]
    assert "lazily" in note
    assert "two threads" in note
    assert "cannot import name" in note


def test_concurrent_first_imports_are_serialised():
    """The property, exercised: many threads taking the same guarded import
    must all succeed."""
    errors = []
    ready = threading.Barrier(8)

    def _one():
        try:
            ready.wait(timeout=5)
            with loras._TRANSFORMERS_IMPORT_LOCK:
                from transformers import AutoTokenizer  # noqa: F401
        except Exception as exc:                        # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=_one) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert errors == [], f"concurrent guarded imports failed: {errors}"
