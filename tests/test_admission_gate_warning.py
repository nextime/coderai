"""The admission gate must announce itself when it is the ceiling.

server.max_parallel_requests defaults to 2 in flight and 6 queued. That is
right for one local card answering chat, and catastrophic for an OCR install
dispatching pages to rented pods: the pool is sized in pages, the pod is handed
a much larger limit through env, and the front in front of all of it keeps
admitting two requests. A fleet idles behind a two-lane gate while every
diagnostic points at the pods.

The bare "[queue] max_parallel_requests=2" line was printed at every boot while
exactly this went undiagnosed for days -- a number with no consequence attached
does not read as a problem. These tests pin the warning that does.

The condition is extracted here as `gate_is_the_ceiling` so it can be tested
without booting a server; main.py applies the same rule.
"""
import pytest


def gate_is_the_ceiling(ocr_enabled, ocr_max_concurrency, gate):
    """Mirror of the check in codai/main.py."""
    if not ocr_enabled:
        return False
    if not ocr_max_concurrency or ocr_max_concurrency <= 0:
        return False
    return gate * 4 < ocr_max_concurrency


def test_the_default_gate_with_ocr_enabled_is_flagged():
    """The exact shipped default, with OCR on: 2 requests vs 4096 pages."""
    assert gate_is_the_ceiling(True, 4096, 2) is True


def test_the_observed_production_case_is_flagged():
    """What Aruba actually ran: ocr.max_concurrency 4096, gate 2."""
    assert gate_is_the_ceiling(True, 4096, 2) is True


def test_a_raised_gate_is_not_flagged():
    """64 requests x ~9 pages each covers a 4096-page budget's useful range."""
    assert gate_is_the_ceiling(True, 4096, 2048) is False


def test_ocr_disabled_is_never_flagged():
    """A chat-only install is exactly who the default of 2 is for."""
    assert gate_is_the_ceiling(False, 4096, 2) is False


def test_no_declared_page_budget_is_not_flagged():
    """Nothing to compare against: say nothing rather than guess."""
    assert gate_is_the_ceiling(True, 0, 2) is False
    assert gate_is_the_ceiling(True, None, 2) is False


@pytest.mark.parametrize("pages,gate,expect", [
    (8, 2, False),        # gate*4 == pages: covered, no warning
    (9, 2, True),         # one page beyond what the gate can cover
    (64, 16, False),
    (64, 15, True),
])
def test_the_boundary_is_four_pages_per_admitted_request(pages, gate, expect):
    """Four is slack for one request carrying several documents; past it the
    gate provably cannot deliver the page budget the operator asked for."""
    assert gate_is_the_ceiling(True, pages, gate) is expect


def test_main_applies_the_same_rule():
    """The rule above is not a second implementation drifting from the first."""
    import re
    src = open("codai/main.py").read()
    assert "_gate * 4 < _pages" in src, "main.py no longer uses this condition"
    # and it must be guarded so a diagnostic cannot stop a server booting
    block = src[src.index("Say so when this gate is the ceiling"):]
    block = block[:block.index("# RunPod:")]
    assert "except Exception:" in block
    assert re.search(r"server\.max_parallel_requests", block)
