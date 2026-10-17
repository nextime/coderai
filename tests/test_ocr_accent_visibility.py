"""Why a fixed accent bug could come back unseen, and what now prevents it.

The repair exists and works (see test_ocr_accent_repair.py). On 2026-10-17 it
was not running on rented pods, and NOTHING said so:

* the worker's diagnostics went to STDERR, and `stderr=None` let the worker
  inherit the engine's raw fd 2 -- while the pod's log ring tees the Python
  `sys.stdout`/`sys.stderr` OBJECTS. A subprocess holding the descriptor writes
  straight past the tee, so `grep -r "\\[surya\\]"` across 343 drained pod logs
  returned nothing at all;
* the byte trace, the one thing that says whether the model or the serving path
  lost the characters, was gated on an opt-in flag -- so a page full of
  replacement characters came back with no explanation unless someone had
  turned the flag on before the run.

A measured page read `societ<FFFD>` with 25 markers, zero 0xC3 lead bytes and a
surviving 3-byte em-dash: 2-byte UTF-8 sequences lost in detokenisation, which
is precisely what the repair is for.
"""

import inspect

from codai.ocr import subprocess_engine
from codai.ocr.workers import ocr_worker


# --------------------------------------------------- the worker can be heard #

def test_the_workers_stderr_is_captured_not_inherited():
    """Asserted on CODE lines, not on prose: the comment that explains the bug
    quotes `stderr=None`, and a test that cannot tell the two apart fails for
    the wrong reason (it did)."""
    src = inspect.getsource(subprocess_engine.SubprocessOcrEngine.load)
    code = [ln.split("#", 1)[0] for ln in src.splitlines()]
    code = "\n".join(ln for ln in code if ln.strip())
    assert "stderr=subprocess.PIPE" in code
    assert "stderr=None" not in code


def test_the_stderr_pump_exists_and_is_a_daemon():
    src = inspect.getsource(subprocess_engine.SubprocessOcrEngine._pump_stderr)
    assert "daemon=True" in src
    # Through print(), i.e. the Python stream the ring tees -- not os.write.
    assert "print(" in src
    assert "flush=True" in src


def test_stderr_is_not_merged_into_the_protocol_pipe():
    """stdout carries newline-JSON; a log line there would desynchronise it."""
    src = inspect.getsource(subprocess_engine.SubprocessOcrEngine.load)
    assert "stderr=subprocess.STDOUT" not in src


def test_the_pump_keeps_draining(monkeypatch):
    """A full stderr pipe would eventually block the worker mid-page."""
    import io
    import threading

    class _Proc:
        def __init__(self):
            self.stderr = io.StringIO("[surya] one\n[surya] two\n")

    eng = subprocess_engine.SubprocessOcrEngine.__new__(
        subprocess_engine.SubprocessOcrEngine)
    eng.name = "surya"
    seen = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: seen.append(a[0] if a else ""))
    eng._pump_stderr(_Proc())
    for _ in range(50):
        if len(seen) >= 2:
            break
        threading.Event().wait(0.02)
    assert any("one" in s for s in seen) and any("two" in s for s in seen)
    assert all(s.startswith("[surya:worker]") for s in seen)


# ------------------------------------------- the diagnosis is not opt-in now #

def test_the_byte_trace_runs_whenever_markers_are_present():
    """It used to require `token_trace`, so the diagnosis was missing exactly
    when it was needed."""
    src = inspect.getsource(ocr_worker.SuryaWorker.ocr)
    assert "if UNREADABLE in page[\"text\"]:" in src
    assert "if self._trace and UNREADABLE" not in src


def test_a_corrupt_page_reports_itself_on_stdout():
    src = inspect.getsource(ocr_worker.SuryaWorker.ocr)
    i = src.index("replacement character(s) after")
    line = src[i - 400:i + 500]
    assert "flush=True" in line
    assert "file=sys.stderr" not in line
    # It must name all three things a reader needs.
    assert "accent repair" in line
    assert "byte trace" in line
    assert "blame" in line


def test_the_report_distinguishes_repaired_from_off_from_ineffective():
    src = inspect.getsource(ocr_worker.SuryaWorker.ocr)
    for phrase in ("recovered", "ran and recovered nothing", "is OFF"):
        assert phrase in src, phrase


def test_a_missing_hook_target_is_loud_and_on_stdout():
    """The single line that explains an accent regression; it used to go to a
    stream a pod throws away."""
    src = inspect.getsource(ocr_worker.SuryaWorker._install_trace_hook)
    assert "ACCENT REPAIR AND TOKEN TRACE ARE OFF" in src
    i = src.index("ACCENT REPAIR AND TOKEN TRACE ARE OFF")
    assert "file=sys.stderr" not in src[i:i + 400]


def test_repair_state_is_reportable():
    assert ocr_worker._oc_repair_on() in (True, False)
