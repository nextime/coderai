"""Pages in flight must stop being the process count.

The OCR worker protocol was one request, one reply, strictly in order — so the
only way to read two pages at once was to run two processes, and a process costs
~600 MB of imported torch. `surya_instances: 64` therefore meant 64 processes:
on 2026-10-12 that was 37 GB of RSS, swap exhausted and load 25 on a 16-core
host, and a batch that gave one 19-second burst and then stalled for nine
minutes with the rented GPU idle at 2% KV-cache usage.

In the served modes that memory buys nothing — the recognition VLM is on the pod,
the predictor holds no weights, and the process rasterises a page and then waits
on a socket. So a lane is a worker OBJECT inside the one process, and these tests
pin the protocol that makes that safe: ids, out-of-order replies, one page per
lane object, and a trace that stays with its own page.
"""

import json
import os
import pathlib
import sys
import threading
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.ocr.base import OcrError                              # noqa: E402
from codai.ocr.subprocess_engine import SubprocessOcrEngine      # noqa: E402

WORKER = ROOT / "codai" / "ocr" / "workers" / "ocr_worker.py"
#: Runs the real protocol with a stubbed engine — see its docstring.
FAKE = ROOT / "tests" / "ocr_lane_stub_worker.py"


#: A real 1x1 PNG: the worker decodes what it is sent before handing it on, so
#: a plausible-looking header is not enough.
def _png_bytes():
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 255, 255)).save(buf, "PNG")
    return buf.getvalue()


class _Img:
    """The minimum of a PIL image the engine touches before sending it."""

    def convert(self, mode):
        return self

    def save(self, buf, fmt):
        buf.write(_png_bytes())


class _Engine(SubprocessOcrEngine):
    """The real engine, driving the real worker in this interpreter's venv."""

    name = "surya"
    _worker_engine = "surya"

    def __init__(self, lanes):
        super().__init__(cfg=None)
        self._lanes = lanes

    def lanes(self):
        return self._lanes

    def _venv_dir(self):
        return "unused"

    def _requirements(self):
        return ""

    def _venv_python(self):
        return sys.executable

    def _ensure_venv(self):
        return sys.executable

    def _worker_opts(self):
        return {}

    def _worker_env(self):
        return {"OCR_WORKER_PATH": str(WORKER)}


@pytest.fixture
def engine(monkeypatch):
    """A live worker subprocess, torn down with the test."""
    import codai.ocr.subprocess_engine as se
    monkeypatch.setattr(se, "_WORKER", str(FAKE))
    made = []

    def _make(lanes):
        eng = _Engine(lanes)
        eng.load()
        made.append(eng)
        return eng

    yield _make
    for eng in made:
        eng.cleanup()


def test_four_lanes_read_four_pages_at_once(engine):
    """The whole point: four pages of 0.25s each take about a quarter of a
    second, not a second. Sequentially this is the same code path."""
    eng = engine(4)
    assert eng._mux is True
    out = {}

    def _one(i):
        out[i] = eng.recognize_image(_Img()).text

    started = time.time()
    threads = [threading.Thread(target=_one, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    took = time.time() - started

    assert len(out) == 4                      # every page answered
    assert took < 0.75                        # overlapped, not queued (4 x 0.25s)
    # Four pages at once means four lane objects, each used by one page.
    assert len(set(out.values())) == 4


def test_one_lane_still_speaks_the_old_protocol(engine):
    """A single-lane engine sends no ids and reads replies in order, exactly as
    before — paddle, docTR and surya-local are driven this way."""
    eng = engine(1)
    assert eng._mux is False
    assert eng.recognize_image(_Img()).text.startswith("lane")


def test_a_lane_is_built_only_when_a_page_needs_it(engine):
    """A quiet install pays for one. The 37 GB came from building for the peak."""
    eng = engine(8)
    first = eng.recognize_image(_Img()).text
    second = eng.recognize_image(_Img()).text
    assert first == second == "lane1"          # reused, never a second lane


def test_pages_beyond_the_lane_count_wait_rather_than_pile_up(engine):
    """Lanes are a bound, not a hint: six pages over two lanes is three rounds."""
    eng = engine(2)
    out = []
    lock = threading.Lock()

    def _one():
        page = eng.recognize_image(_Img())
        with lock:
            out.append(page.text)

    threads = [threading.Thread(target=_one) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(out) == 6
    assert len(set(out)) <= 2                  # never more lanes than allowed


def test_a_reply_goes_to_the_page_that_asked_for_it(engine):
    """Attributing one page's text to another request is the one failure worse
    than losing the page, so each request carries an id and the reader matches
    on it. The stub answers with its lane, so a crossed reply would show up as
    two pages claiming the same lane while both were in flight."""
    eng = engine(3)
    seen = {}

    def _one(i):
        seen[i] = eng.recognize_image(_Img()).text

    threads = [threading.Thread(target=_one, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(seen) == 3
    assert len(set(seen.values())) == 3


def test_a_dead_worker_fails_the_pages_instead_of_hanging(engine):
    """With no reply ever coming, a page used to block on readline for ever. The
    wait is broken up so a worker that died is noticed."""
    eng = engine(4)
    eng.recognize_image(_Img())            # prove it works first
    eng._proc.kill()
    eng._proc.wait(timeout=10)
    with pytest.raises(OcrError) as got:
        eng.recognize_image(_Img())
    assert "worker" in str(got.value)


def test_killing_the_engine_wakes_everything_waiting(engine):
    """cleanup() during a batch must not leave threads parked on an Event."""
    eng = engine(2)
    errs = []

    def _one():
        try:
            eng.recognize_image(_Img())
        except OcrError as e:
            errs.append(str(e))

    threads = [threading.Thread(target=_one) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    eng.cleanup()
    for t in threads:
        t.join(20)
        assert not t.is_alive()


# ----------------------------------------------------- the trace is per page #
def test_the_token_trace_is_per_thread_not_per_worker():
    """surya's hook is module-level: one `_generate_one` per process. The sink
    used to belong to whichever worker object installed it last, which with
    lanes would file one page's logprobs — and its accent repairs — under
    another page."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ocr_worker_mod", str(WORKER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    seen = {}

    def _in_thread(name):
        calls, repairs = mod._trace_begin()
        calls.append(name)
        time.sleep(0.05)                   # overlap with the other thread
        seen[name] = list(mod._trace_calls())

    a = threading.Thread(target=_in_thread, args=("A",))
    b = threading.Thread(target=_in_thread, args=("B",))
    a.start(); b.start(); a.join(10); b.join(10)

    assert seen["A"] == ["A"]              # no cross-talk
    assert seen["B"] == ["B"]


def test_lanes_default_to_one_for_an_engine_that_has_not_opted_in():
    """Concurrency is opt-in per engine: driving one that is not safe for it
    concurrently would corrupt pages rather than slow them down."""
    from codai.ocr.base import OcrEngine

    class _Plain(OcrEngine):
        def load(self):
            pass

        def recognize_image(self, image):
            raise NotImplementedError

    assert _Plain(cfg=None).lanes() == 1


def test_surya_takes_lanes_from_config_in_served_mode_only():
    from codai.ocr.surya import SuryaEngine

    class _Cfg:
        surya_serve = "model"
        surya_lanes = 6
        surya_langs = "it"

    eng = SuryaEngine.__new__(SuryaEngine)
    eng.cfg = _Cfg()
    assert eng.lanes() == 6

    # Local mode: a lane would be a second copy of the model on the card.
    eng.cfg.surya_serve = "local"
    assert eng.lanes() == 1


def test_the_pool_offers_an_instance_once_per_lane():
    """What turns lanes into concurrency: the queue is the bound, so a 4-lane
    instance has to be in it four times."""
    import asyncio
    from codai.ocr import manager

    class _Eng:
        def __init__(self):
            self.loaded = 0

        def lanes(self):
            return 4

        def ensure_loaded(self):
            self.loaded += 1

        def prelaunch_vram_gb(self):
            return 0.0

        def vram_gb(self):
            return 0.0

    made = []

    def _factory(cfg):
        eng = _Eng()
        made.append(eng)
        return eng

    manager._ENGINE_FACTORIES["_lanetest"] = _factory
    try:
        pool = manager._Pool("_lanetest", cfg=None, size=2)
        asyncio.run(pool._ensure_built())
        assert len(made) == 2                      # two processes
        assert pool._q.qsize() == 8                # eight pages at once
    finally:
        manager._ENGINE_FACTORIES.pop("_lanetest", None)
