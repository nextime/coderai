"""The admission-gate warning must fire on the trap and stay silent otherwise.

Context: server.max_parallel_requests defaults to 2 in flight / 6 queued,
right for one local card answering chat and the throughput ceiling for an OCR
install dispatching pages to rented pods. One A40 at a gate of 2 sat at
`Running: 1, Waiting: 0` and read 219 pages in 1210s -- the same as reading
them one at a time -- while ocr.max_concurrency said 4096.

The first version of this check compared the gate to ocr.max_concurrency
(gate * 4 < pages). Those are different units, a request against a page, and
for a 4096-page budget it demanded a gate of 1024 -- memory-unsafe, and more
than the derivation itself will ever choose. It duly fired against the 256 that
the same file had just derived. These tests pin the corrected rule and, above
all, that the warning never contradicts the derivation.
"""
import pytest

GATE_FLOOR, GATE_CAP = 16, 256


def warn(ocr_enabled, gate):
    """Mirror of the check in codai/main.py."""
    return bool(ocr_enabled) and int(gate or 0) < GATE_FLOOR


def derive(page_budget):
    """Mirror of the derivation in codai/main.py."""
    return max(GATE_FLOOR, min(GATE_CAP, page_budget // 8))


# --- the trap it exists for ---

def test_the_shipped_default_with_ocr_enabled_is_flagged():
    assert warn(True, 2) is True


def test_a_gate_of_one_is_flagged():
    assert warn(True, 1) is True


# --- the regression that shipped: never contradict the derivation ---

@pytest.mark.parametrize("page_budget", [8, 128, 1024, 2048, 4096, 99999])
def test_a_derived_gate_is_never_warned_about(page_budget):
    """The bug: 256 was derived from a 4096-page budget and then warned about."""
    assert warn(True, derive(page_budget)) is False


def test_the_observed_production_values_are_silent():
    """ocr.max_concurrency=4096 with the gate at 256 is correct, not a problem."""
    assert warn(True, 256) is False


def test_a_large_page_budget_does_not_by_itself_trigger_a_warning():
    """The page budget is not the unit; only the gate's own size is."""
    assert warn(True, GATE_FLOOR) is False


# --- not everyone's problem ---

def test_ocr_disabled_is_never_flagged():
    assert warn(False, 2) is False
    assert warn(False, 1) is False


def test_the_boundary_is_the_shared_floor():
    assert warn(True, GATE_FLOOR - 1) is True
    assert warn(True, GATE_FLOOR) is False


# --- the rules above must be the rules main.py applies ---

def test_main_uses_one_shared_floor_for_both_rules():
    """A separate constant in each place is how they came to disagree."""
    src = open("codai/main.py").read()
    assert "_GATE_FLOOR, _GATE_CAP = 16, 256" in src
    assert "_gate < _GATE_FLOOR" in src, "warning must use the shared floor"
    assert "max(_GATE_FLOOR, min(_GATE_CAP, _page_budget // 8))" in src, \
        "derivation must use the shared floor and cap"


def test_main_no_longer_compares_requests_against_pages():
    """The unit error that caused the false positive must not come back."""
    src = open("codai/main.py").read()
    assert "_gate * 4 < _pages" not in src


def test_the_warning_is_exception_guarded():
    src = open("codai/main.py").read()
    block = src[src.index("Say so when an EXPLICIT gate is small enough"):]
    assert "except Exception:" in block
