"""`conf` on the VLM engines: a measurement, or honestly absent.

chandra and olmOCR used to report nothing — chandra hard-coded `conf=0.0` with a
comment saying the model reported none and inventing one would be read as a
measurement. The first half was right and the second was wrong: the number IS
obtainable, by the route surya already uses. Both are served over an OpenAI chat
endpoint, so asking for `logprobs` and averaging `exp(logprob)` gives a genuine
figure, and the same computation surya's own client does means it is comparable
with surya's rather than a second notion of confidence.

Two things the tests hold to, both learned the hard way in this subsystem:

* a 0.0 that means "not measured" must be distinguishable from a 0.0 that means
  "unsure" — the empty-bytes episode came from exactly that confusion;
* a page-level number must not be dressed up as per-line precision it lacks.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest                                                      # noqa: E402

from codai.config import OcrConfig                                 # noqa: E402
from codai.ocr import chandra as ch                                # noqa: E402
from codai.ocr import olmocr as om                                 # noqa: E402
from codai.ocr.base import logprobs_of, mean_token_prob            # noqa: E402


def _cfg(engine, **kw):
    cfg = OcrConfig()
    cfg.enabled = True
    setattr(cfg, f"{engine}_enabled", True)
    if engine == "chandra":
        cfg.chandra_accept_license = True
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class _Reply:
    def __init__(self, payload, status=200, text=""):
        self._p, self.status_code, self.text = payload, status, text

    def json(self):
        return self._p


def _body(content, logprobs=None):
    choice = {"message": {"content": content}}
    if logprobs is not None:
        choice["logprobs"] = {"content": logprobs}
    return {"choices": [choice]}


def _lp(*values):
    return [{"token": "x", "logprob": v} for v in values]


# ------------------------------------------------------------- the arithmetic #
def test_the_mean_is_of_probabilities_not_logprobs():
    """exp() first, then average — averaging the logs and exponentiating gives a
    geometric mean, a different (and smaller) number."""
    import math
    got = mean_token_prob(_lp(-0.1, -0.2))
    assert got == pytest.approx((math.exp(-0.1) + math.exp(-0.2)) / 2)


def test_a_certain_reply_is_near_one():
    assert mean_token_prob(_lp(0.0, 0.0)) == pytest.approx(1.0)


def test_nothing_reported_is_none_not_zero():
    """THE distinction. A 0.0 would be indistinguishable from a model that was
    certain it was unsure, and that confusion already produced one confidently
    wrong verdict in this subsystem."""
    assert mean_token_prob(None) is None
    assert mean_token_prob([]) is None
    assert mean_token_prob([{"token": "a"}]) is None          # no logprob field


def test_entries_without_a_logprob_are_skipped_not_counted_as_zero():
    import math
    got = mean_token_prob([{"logprob": -0.1}, {"token": "a"}])
    assert got == pytest.approx(math.exp(-0.1))


def test_model_objects_work_as_well_as_dicts():
    class _E:
        logprob = 0.0
    assert mean_token_prob([_E()]) == pytest.approx(1.0)


def test_logprobs_are_dug_out_of_the_openai_shape():
    assert logprobs_of(_body("x", _lp(-0.1))) == _lp(-0.1)
    assert logprobs_of(_body("x")) is None
    assert logprobs_of({"choices": []}) is None
    assert logprobs_of(None) is None


# ------------------------------------------------------------------- chandra #
def test_chandra_asks_for_logprobs(monkeypatch):
    from codai.api import pod_http
    seen = {}
    monkeypatch.setattr(pod_http, "post",
                        lambda url, json=None, **kw: (seen.update(json=json),
                                                      _Reply(_body("<p>x</p>",
                                                                   _lp(0.0))))[1])
    eng = ch.ChandraEngine(_cfg("chandra", chandra_serve="server",
                                chandra_server_url="http://h:8000"))
    eng.load()
    eng._ask("data:,", 0.0, 0.1)
    assert seen["json"]["logprobs"] is True


def test_chandra_can_be_told_not_to(monkeypatch):
    from codai.api import pod_http
    seen = {}
    monkeypatch.setattr(pod_http, "post",
                        lambda url, json=None, **kw: (seen.update(json=json),
                                                      _Reply(_body("<p>x</p>")))[1])
    eng = ch.ChandraEngine(_cfg("chandra", chandra_serve="server",
                                chandra_server_url="http://h:8000",
                                chandra_logprobs=False))
    eng.load()
    eng._ask("data:,", 0.0, 0.1)
    assert "logprobs" not in seen["json"]


def test_chandra_reports_the_measured_figure(monkeypatch):
    from PIL import Image
    eng = ch.ChandraEngine(_cfg("chandra"))
    eng._mode = lambda: "server"
    eng._ask = lambda uri, t, p: ('<div data-label="Text" data-bbox="0 0 500 500">'
                                  "<p>società</p></div>", 0.9312)
    page = eng.recognize_image(Image.new("RGB", (100, 100)))
    assert page.meta["confidence"] == 0.9312
    assert page.meta["confidence_measured"] is True
    assert page.meta["confidence_scope"] == "page"
    # ...and it reaches the lines and regions, which is what `unreadable`'s
    # per-spot conf reads.
    assert page.lines[0].conf == 0.9312
    assert page.regions[0].conf == 0.9312


def test_chandra_says_when_it_was_not_measured(monkeypatch):
    """A server that declines logprobs leaves conf at 0.0 — and the flag is the
    only thing that stops that reading as "the model was hopeless"."""
    from PIL import Image
    eng = ch.ChandraEngine(_cfg("chandra"))
    eng._mode = lambda: "server"
    eng._ask = lambda uri, t, p: ("<p>x</p>", None)
    page = eng.recognize_image(Image.new("RGB", (100, 100)))
    assert page.meta["confidence"] == 0.0
    assert page.meta["confidence_measured"] is False


def test_chandras_scope_is_page_because_it_generates_a_page_at_once():
    """Unlike surya, which calls per block and so has a genuinely per-block
    number. Every line carrying the same figure is only honest if it says so."""
    page = ch.parse_page('<div data-label="Text" data-bbox="0 0 100 100">a</div>'
                         '<div data-label="Text" data-bbox="0 100 100 200">b</div>',
                         1000, 1000, conf=0.8)
    assert page.meta["confidence_scope"] == "page"
    assert [l.conf for l in page.lines] == [0.8, 0.8]


def test_the_local_model_path_reports_no_confidence():
    """coderai's in-process chat handler surfaces no logprobs, so this path must
    not pretend otherwise."""
    import inspect
    src = inspect.getsource(ch.ChandraEngine._ask_local)
    assert "return content, None" in src


# -------------------------------------------------------------------- olmOCR #
def test_olmocr_asks_for_logprobs(monkeypatch):
    from codai.api import pod_http
    seen = {}
    monkeypatch.setattr(pod_http, "post",
                        lambda url, json=None, **kw: (seen.update(json=json),
                                                      _Reply(_body("text",
                                                                   _lp(0.0))))[1])
    eng = om.OlmOcrEngine(_cfg("olmocr", olmocr_serve="server",
                               olmocr_server_url="http://h:8000"))
    eng.load()
    eng._ask_http([{"role": "user", "content": "x"}], 64, 0.0)
    assert seen["json"]["logprobs"] is True


def test_olmocr_reports_a_page_level_figure(monkeypatch):
    from PIL import Image
    eng = om.OlmOcrEngine(_cfg("olmocr"))
    eng._ask = lambda image: ("primary_language: it\n---\nLa società", 0.77)
    page = eng.recognize_image(Image.new("RGB", (50, 50)))
    assert page.meta["confidence"] == 0.77
    assert page.meta["confidence_measured"] is True
    assert page.meta["confidence_scope"] == "page"


def test_olmocr_has_no_lines_so_the_figure_can_only_be_page_level():
    """Not a shortcut — there is nowhere per-line to put one and nothing per-line
    to measure."""
    from PIL import Image
    eng = om.OlmOcrEngine(_cfg("olmocr"))
    eng._ask = lambda image: ("hello", 0.5)
    page = eng.recognize_image(Image.new("RGB", (50, 50)))
    assert page.lines == []


def test_olmocr_says_when_it_was_not_measured():
    from PIL import Image
    eng = om.OlmOcrEngine(_cfg("olmocr"))
    eng._ask = lambda image: ("hello", None)
    page = eng.recognize_image(Image.new("RGB", (50, 50)))
    assert page.meta["confidence"] == 0.0
    assert page.meta["confidence_measured"] is False


def test_a_rotated_retry_reports_the_retrys_confidence():
    """The retry's text is the one kept, so its confidence is the one that
    describes the answer."""
    from PIL import Image
    eng = om.OlmOcrEngine(_cfg("olmocr", olmocr_retry_rotation=True))
    calls = []

    def _ask(image):
        calls.append(image.size)
        if len(calls) == 1:
            return ("is_rotation_valid: false\nrotation_correction: 90\n---\nbad", 0.11)
        return ("primary_language: it\n---\ngood", 0.95)
    eng._ask = _ask
    page = eng.recognize_image(Image.new("RGB", (40, 60)))
    assert len(calls) == 2
    assert page.text == "good"
    assert page.meta["confidence"] == 0.95


# ----------------------------------------------------------- and the catalogue #
def test_the_catalogue_says_what_conf_means_per_engine():
    """So a client gating on confidence knows what it is getting BEFORE it sends
    a document — a per-line score can single out the line to re-read, a
    page-level one can only say the page went badly."""
    from codai.ocr.manager import OcrManager
    cfg = OcrConfig()
    cfg.enabled = True
    mgr = OcrManager()
    mgr.configure(cfg)
    rows = {r["id"]: r["confidence"] for r in mgr.engine_catalogue()["data"]}
    assert rows["paddle"]["scope"] == "line"
    assert rows["doctr"]["scope"] == "line"
    assert rows["surya"]["scope"] == "block"
    assert rows["chandra"]["scope"] == "page"
    assert rows["olmocr"]["scope"] == "page"
    assert all(v["measured"] for v in rows.values())


def test_every_engine_in_the_registry_has_a_confidence_row():
    """A new engine must not silently arrive with no answer to this."""
    from codai.ocr.manager import _CONFIDENCE, _ENGINE_FACTORIES
    assert set(_CONFIDENCE) == set(_ENGINE_FACTORIES)


def test_toggling_logprobs_rebuilds_the_pools():
    from codai.ocr.manager import OcrManager
    base = OcrManager._pool_params(OcrConfig())
    for field in ("chandra_logprobs", "olmocr_logprobs"):
        cfg = OcrConfig()
        setattr(cfg, field, False)
        assert OcrManager._pool_params(cfg) != base, field


def test_max_concurrency_is_a_ceiling_not_a_throttle():
    """It shipped at 4, which made it the binding limit for every install — a
    quarter of what ONE pod sustains. Measured on an A40: 16 concurrent pages is
    the knee for both surya and chandra, and both fail outright by 32. The real
    limiter is meant to be each engine's `*_instances`, so this only has to stay
    out of the way while remaining finite."""
    from codai.config import OcrConfig
    cfg = OcrConfig()
    assert cfg.max_concurrency >= 16, "must not bind below one pod's measured knee"
    assert cfg.max_concurrency <= 128, (
        "every admitted page holds a rasterised bitmap (~18 MB for A4 at 250 DPI), "
        "so an unbounded ceiling trades one failure mode for another")
