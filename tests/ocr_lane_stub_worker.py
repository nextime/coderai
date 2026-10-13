# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.

"""Stands in for an OCR engine's venv python, running the REAL worker protocol.

The point of the lane tests is the protocol — ids, out-of-order replies, one page
per lane object — so the protocol must be the shipped code and only the engine
stubbed. ``codai/ocr/workers/ocr_worker.py`` is loaded by path (it is a script,
not a module on sys.path), its two worker classes are replaced with a sleepy stub
that reports which lane served the page, and its own ``main()`` is run.

Driven by tests/test_ocr_worker_lanes.py; not a test itself.
"""

import importlib.util
import os
import time


class _Stub:
    """One object per lane, so a crossed reply shows up as a repeated lane."""

    made = 0

    def __init__(self, opts):
        type(self).made += 1
        self.lane = type(self).made

    def load(self):
        pass

    def ocr(self, img):
        # Long enough that four sequential pages could not fit in the window the
        # test allows, so overlapping is the only way to pass it.
        time.sleep(float(os.environ.get("OCR_STUB_PAGE_S") or 0.25))
        return {"index": 0, "text": f"lane{self.lane}", "width": 1, "height": 1,
                "lines": [], "regions": []}


def main():
    path = os.environ["OCR_WORKER_PATH"]
    spec = importlib.util.spec_from_file_location("ocr_worker_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.SuryaWorker = _Stub
    mod.PaddleWorker = _Stub
    mod.main()


if __name__ == "__main__":
    main()
