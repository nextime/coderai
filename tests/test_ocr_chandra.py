"""Chandra: a document VLM that is ASKED for its layout.

Chandra comes from the same lab as Surya and is driven the same way olmOCR is — one
page image over an OpenAI chat endpoint — but its ``ocr_layout`` prompt asks for one
``<div>`` per block carrying ``data-bbox`` and ``data-label``. So it is the only engine
here that returns text, boxes, labelled regions, tables AND page metadata in one pass.

What these tests are actually guarding:

- **The prompt is a contract.** It is mirrored verbatim from chandra/prompts.py, down to
  the Python repr of the tag list, because the model was trained on that wording.
- **The parse must not lose page.** A reply in an unexpected shape (no blocks, an
  unclosed div, a bbox that will not parse) is still a transcription of somebody's
  document, and the worst possible reading of "unexpected shape" is to drop it.
- **A repetition loop answers 200.** A VLM that loses its place repeats itself until the
  token budget runs out, and the result looks like a successful page. It has to be caught
  and flagged rather than handed on.
- **The licence is narrower than Surya's.** OpenRAIL-M with a $2M ceiling and no
  competitive use, so nothing loads un-accepted, and a rented pod can never accept on the
  operator's behalf.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest                                                      # noqa: E402

from codai.config import OcrConfig                                 # noqa: E402
from codai.ocr import chandra as ch                                # noqa: E402
from codai.ocr import manager as om                                # noqa: E402
from codai.ocr.base import OcrError, count_unreadable              # noqa: E402


def _cfg(**kw):
    cfg = OcrConfig()
    cfg.enabled = True
    cfg.chandra_enabled = True
    cfg.chandra_accept_license = True
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# A page of the kind this is actually for: an Italian judgment, with the accents
# that started this whole investigation, a table, inline maths and page furniture.
REPLY = (
    '<div data-bbox="100 50 900 90" data-label="Page-Header">'
    '<p>TRIBUNALE DI MILANO &mdash; Sez. II</p></div>\n'
    '<div data-bbox="100 120 900 200" data-label="Section-Header">'
    '<h2>Motivi della decisione</h2></div>\n'
    '<div data-bbox="100 220 900 420" data-label="Text">'
    "<p>La domanda &egrave; pi&ugrave; che fondata, e la facolt&agrave; "
    'prevista non &egrave; stata esercitata.<br>Ne consegue l&apos;accoglimento.</p></div>\n'
    '<div data-bbox="100 450 900 650" data-label="Table">'
    '<table><tr><th>Voce</th><th>Importo</th></tr>'
    '<tr><td>Spese</td><td>1.200,00</td></tr></table></div>\n'
    '<div data-bbox="100 950 900 990" data-label="Page-Footer"><p>pag. 3</p></div>'
)


# --------------------------------------------------------------- the prompt #
def test_the_layout_prompt_asks_for_boxes_and_labels():
    """Without these two attributes the reply is one unlocated blob, and the engine
    has no reason to exist alongside olmOCR."""
    assert "data-bbox" in ch.OCR_LAYOUT_PROMPT
    assert "data-label" in ch.OCR_LAYOUT_PROMPT
    assert "normalized 0-1000" in ch.OCR_LAYOUT_PROMPT
    for label in ("Section-Header", "Table", "Figure", "Blank-Page"):
        assert label in ch.OCR_LAYOUT_PROMPT


def test_the_plain_prompt_does_not_ask_for_layout():
    """`data-bbox` is still in its shared attribute whitelist — that list is one
    constant in both prompts. What the plain prompt lacks is the INSTRUCTION to
    arrange the page as blocks and the label set to use."""
    assert ch.OCR_PROMPT.startswith("OCR this image to HTML.")
    assert "arranged as layout blocks" not in ch.OCR_PROMPT
    assert "Section-Header" not in ch.OCR_PROMPT
    assert "arranged as layout blocks" in ch.OCR_LAYOUT_PROMPT


def test_both_prompts_carry_the_shared_guidelines():
    """The tag/attribute whitelist and the maths, table and chemistry rules are
    trained-on wording, not advice — they belong to both prompts."""
    for prompt in (ch.OCR_PROMPT, ch.OCR_LAYOUT_PROMPT):
        assert "<math>" in prompt and "KaTeX" in prompt
        assert "<chem>" in prompt
        assert "colspan" in prompt


def test_the_tag_list_is_rendered_as_pythons_repr():
    """Mirrored from chandra/prompts.py, where it is an f-string of a Python list. The
    quoting and spacing are part of the prompt the model was trained on."""
    assert "['math', 'br', 'i'" in ch.PROMPT_ENDING
    assert "'data-bbox'" in ch.PROMPT_ENDING


def test_layout_is_the_default_and_the_switch_turns_it_off():
    eng = ch.ChandraEngine(_cfg())
    assert eng._prompt() is ch.OCR_LAYOUT_PROMPT
    assert ch.ChandraEngine(_cfg(chandra_layout=False))._prompt() is ch.OCR_PROMPT


# ---------------------------------------------------------------- the parse #
def test_a_block_becomes_a_line_a_region_and_the_page_text():
    page = ch.parse_page(REPLY, 1000, 1000)
    assert len(page.lines) == 5
    assert len(page.regions) == 5
    assert "Motivi della decisione" in page.text
    assert page.meta["labels"]["Text"] == 1


def test_accents_survive_the_html_entity_decode():
    """The reason this engine is interesting at all: a second model's reading of a page
    whose accents Surya loses. If our own parse ate them the comparison would be
    worthless — and entity-encoded accents are exactly what an HTML-emitting model sends."""
    page = ch.parse_page(REPLY, 1000, 1000)
    assert "più" in page.text
    assert "facoltà" in page.text
    assert "è" in page.text
    assert count_unreadable(page.text) == 0


def test_boxes_are_denormalised_against_the_image_the_model_saw():
    page = ch.parse_page(REPLY, 2000, 4000)
    # data-bbox="100 50 900 90" of 1000 → 20%..90% across, 1.25%..2.25% down.
    x0, y0, x1, y1 = page.regions[0].bbox
    assert (round(x0), round(y0), round(x1), round(y1)) == (200, 200, 1800, 360)


def test_a_box_that_will_not_parse_keeps_the_text_and_drops_the_box():
    """A full-page default would look like a real answer, and anything cropping by it
    would be silently wrong. Missing is better than wrong."""
    page = ch.parse_page(
        '<div data-bbox="nonsense" data-label="Text"><p>Cfr. Cass.</p></div>', 100, 100)
    assert page.lines[0].text == "Cfr. Cass."
    assert page.lines[0].bbox == [0.0, 0.0, 0.0, 0.0]
    assert page.regions == []


def test_a_degenerate_box_is_refused_too():
    assert ch.parse_bbox("500 500 100 100", 1000, 1000) is None     # x1 < x0
    assert ch.parse_bbox("0 0 0 0", 1000, 1000) is None             # empty
    assert ch.parse_bbox("1 2 3", 1000, 1000) is None               # three numbers


def test_a_table_keeps_its_html():
    """Flattened to text a table has lost the only thing that made it a table."""
    page = ch.parse_page(REPLY, 1000, 1000)
    assert len(page.tables) == 1
    assert "<table>" in page.tables[0]["html"]
    assert page.tables[0]["bbox"] is not None


def test_table_cells_do_not_weld_together_in_the_text():
    assert ch.html_to_text(
        "<table><tr><th>Voce</th><th>Importo</th></tr></table>") == "Voce Importo"


def test_a_blank_page_block_is_dropped():
    page = ch.parse_page(
        '<div data-bbox="0 0 1000 1000" data-label="Blank-Page"></div>', 100, 100)
    assert page.lines == [] and page.regions == [] and page.text == ""


def test_headers_and_footers_can_be_left_out():
    keep = ch.parse_page(REPLY, 1000, 1000, include_furniture=True)
    drop = ch.parse_page(REPLY, 1000, 1000, include_furniture=False)
    assert "TRIBUNALE DI MILANO" in keep.text and "pag. 3" in keep.text
    assert "TRIBUNALE DI MILANO" not in drop.text and "pag. 3" not in drop.text
    assert "Motivi della decisione" in drop.text


def test_nested_divs_are_one_block_not_several():
    """Layout blocks nest — a Form or a Table block holds divs — and only the
    outermost carries the page-level box."""
    page = ch.parse_page(
        '<div data-bbox="0 0 1000 1000" data-label="Form">'
        '<div>interno</div>esterno</div>', 100, 100)
    assert len(page.lines) == 1
    assert "interno" in page.lines[0].text and "esterno" in page.lines[0].text


def test_a_reply_with_no_blocks_at_all_is_still_a_transcription():
    """The model ignored the layout instruction. It still read the page, and dropping
    that would be the worst possible response to an unexpected shape."""
    page = ch.parse_page("<p>Solo testo, senza blocchi.</p>", 100, 100)
    assert page.text == "Solo testo, senza blocchi."
    assert len(page.lines) == 1


def test_content_outside_any_block_is_not_lost():
    page = ch.parse_page("preambolo<div data-label=\"Text\"></div>", 100, 100)
    assert "preambolo" in page.text


def test_an_unclosed_final_div_still_yields_its_block():
    """Truncated at the token limit, which is routine on a dense page."""
    page = ch.parse_page(
        '<div data-bbox="0 0 500 500" data-label="Text"><p>mezza frase', 100, 100)
    assert "mezza frase" in page.text


def test_math_and_chemistry_survive_as_their_source():
    """Unrendered LaTeX beats a formula silently deleted."""
    page = ch.parse_page(
        '<div data-label="Equation-Block"><math>E = mc^2</math></div>', 100, 100)
    assert "E = mc^2" in page.text


def test_void_tags_do_not_unbalance_the_block_split():
    """<br> and <img> arrive unclosed; counting them as open tags would swallow the
    rest of the page into one block."""
    page = ch.parse_page(
        '<div data-label="Text">a<br>b<img alt="x">c</div>'
        '<div data-label="Text">secondo</div>', 100, 100)
    assert len(page.lines) == 2


def test_conf_is_zero_because_the_model_reports_none():
    """A number nobody computed must not be handed to a caller that gates on it."""
    page = ch.parse_page(REPLY, 1000, 1000)
    assert all(l.conf == 0.0 for l in page.lines)


# ------------------------------------------------------------- the scaling #
def test_the_input_is_capped_at_the_same_pixels_as_surya():
    """3072x2048 = 6,291,456 — reached by A4 at about 255 DPI, which is why
    chandra_dpi is 250 and not higher."""
    from PIL import Image
    out = ch.scale_to_fit(Image.new("RGB", (4960, 7016)))           # A4 at 600 DPI
    assert out.size[0] * out.size[1] <= 3072 * 2048


def test_dimensions_land_on_the_vision_towers_28_pixel_grid():
    from PIL import Image
    for size in ((2480, 3508), (800, 600), (1, 1)):
        out = ch.scale_to_fit(Image.new("RGB", size))
        assert out.size[0] % 28 == 0 and out.size[1] % 28 == 0, size


def test_the_boxes_are_read_against_the_scaled_size_not_the_original():
    """They come back normalised against the image the MODEL saw. De-normalising
    against the pre-scaling size would put every box on a scaled page in the wrong
    place."""
    from PIL import Image
    eng = ch.ChandraEngine(_cfg())
    eng._mode = lambda: "server"
    seen = {}

    def _ask(uri, temp, top_p):
        return '<div data-bbox="0 0 1000 1000" data-label="Text"><p>x</p></div>'
    eng._ask = _ask
    page = eng.recognize_image(Image.new("RGB", (4960, 7016)))
    assert page.width * page.height <= 3072 * 2048
    assert page.regions[0].bbox == [0.0, 0.0, float(page.width), float(page.height)]
    # …and the size the caller rasterised is still reported, so the scaling is visible.
    assert page.meta["source_size"] == [4960, 7016]


# ------------------------------------------------------- repetition loops #
def test_a_repeated_tail_is_detected():
    assert ch.detect_repeat_token("testo normale " + "pag. 3 " * 40)


def test_ordinary_prose_is_not():
    assert not ch.detect_repeat_token(
        "La domanda è più che fondata, e la facoltà prevista non è stata esercitata.")


def test_a_loop_is_re_asked_with_a_higher_temperature():
    """A greedy decode that fell into a loop falls into the same one again, so the
    retry has to change something."""
    eng = ch.ChandraEngine(_cfg(chandra_max_retries=2))
    eng._mode = lambda: "server"
    temps, replies = [], ["loop " * 60, "<div data-label=\"Text\"><p>buono</p></div>"]

    def _ask(uri, temp, top_p):
        temps.append(temp)
        return replies[min(len(temps) - 1, len(replies) - 1)]
    eng._ask = _ask
    content, meta = eng._ask_with_retries("data:,")
    assert temps == [0.0, 0.2]
    assert meta["retries"] == 1
    assert "buono" in content


def test_a_page_that_loops_to_the_end_is_returned_flagged_not_dropped():
    """A looping transcription usually has a correct beginning, and the flag is what
    lets a caller decide instead of being handed a silent mess."""
    eng = ch.ChandraEngine(_cfg(chandra_max_retries=1))
    eng._mode = lambda: "server"
    eng._ask = lambda uri, temp, top_p: "loop " * 60
    content, meta = eng._ask_with_retries("data:,")
    assert meta["repeat_detected"] is True
    assert meta["retries"] == 1
    assert content


def test_retries_off_accepts_the_first_reply():
    eng = ch.ChandraEngine(_cfg(chandra_max_retries=0))
    eng._mode = lambda: "server"
    calls = []

    def _ask(uri, temp, top_p):
        calls.append(temp)
        return "loop " * 60
    eng._ask = _ask
    _, meta = eng._ask_with_retries("data:,")
    assert len(calls) == 1
    assert meta["repeat_detected"] is True


def test_a_clean_first_reply_reports_no_retries():
    eng = ch.ChandraEngine(_cfg())
    eng._mode = lambda: "server"
    eng._ask = lambda uri, temp, top_p: '<div data-label="Text"><p>ok</p></div>'
    _, meta = eng._ask_with_retries("data:,")
    assert "retries" not in meta and "repeat_detected" not in meta


# ------------------------------------------------------------- the licence #
def test_nothing_loads_until_the_licence_is_accepted():
    cfg = _cfg(chandra_accept_license=False)
    with pytest.raises(OcrError) as e:
        ch.ChandraEngine(cfg).load()
    assert e.value.status == 400
    assert "OpenRAIL" in str(e.value)


def test_the_refusal_says_what_the_terms_actually_are():
    """An operator deciding whether to tick a box needs the terms, not the name of a
    config flag — these are narrower than Surya's GPL."""
    cfg = _cfg(chandra_accept_license=False)
    with pytest.raises(OcrError) as e:
        ch.ChandraEngine(cfg).load()
    msg = str(e.value)
    assert "$2M" in msg
    assert "competing" in msg
    assert "chandra_accept_license" in msg


def test_the_engine_is_not_enabled_without_the_licence():
    assert not om._engine_enabled(_cfg(chandra_accept_license=False), "chandra")
    assert om._engine_enabled(_cfg(), "chandra")


def test_the_catalogue_explains_the_refusal():
    mgr = om.OcrManager()
    mgr.configure(_cfg(chandra_accept_license=False))
    row = next(r for r in mgr.engine_catalogue()["data"] if r["id"] == "chandra")
    assert row["enabled"] is False
    assert "OpenRAIL" in row["reason"]


# -------------------------------------------------------------- the modes #
def test_model_mode_needs_to_be_told_where_the_model_is():
    with pytest.raises(OcrError) as e:
        ch.ChandraEngine(_cfg(chandra_serve="model", chandra_model_id="")).load()
    assert "chandra_model_id" in str(e.value)


def test_server_mode_needs_a_url():
    with pytest.raises(OcrError) as e:
        ch.ChandraEngine(_cfg(chandra_serve="server", chandra_server_url="")).load()
    assert "chandra_server_url" in str(e.value)


def test_a_server_url_gains_the_v1_it_was_given_without():
    eng = ch.ChandraEngine(_cfg(chandra_serve="server",
                                chandra_server_url="http://10.0.0.5:8000"))
    eng.load()
    assert eng._base == "http://10.0.0.5:8000/v1"


def test_a_url_that_already_ends_in_v1_is_not_doubled():
    eng = ch.ChandraEngine(_cfg(chandra_serve="server",
                                chandra_server_url="http://10.0.0.5:8000/v1/"))
    eng.load()
    assert eng._base == "http://10.0.0.5:8000/v1"


def test_an_unknown_serve_mode_is_refused_by_name():
    with pytest.raises(OcrError) as e:
        ch.ChandraEngine(_cfg(chandra_serve="telepathy")).load()
    assert "telepathy" in str(e.value)


def test_model_mode_is_the_only_one_that_needs_no_local_gpu():
    """And it is the mode the Digesta orchestrator runs: a box with no card, whose
    whole job is renting one."""
    eng = ch.ChandraEngine(_cfg(chandra_serve="model", chandra_model_id="chandra"))
    eng.load()                                   # must not raise and must not need vLLM
    assert eng._base == ""


# ------------------------------------------------------------ the transport #
class _Reply:
    def __init__(self, payload, status=200, text=""):
        self._p, self.status_code, self.text = payload, status, text

    def json(self):
        return self._p


def _ok(content):
    return _Reply({"choices": [{"message": {"content": content}}]})


def test_the_request_goes_through_pod_http_so_a_pods_own_ca_is_trusted(monkeypatch):
    """Chandra 'model' mode reaches the upstream the dispatcher chose, which can be a
    RunPod pod on the direct-TCP path: https to a bare IP with a certificate signed by
    THIS install's CA. Plain requests fails that twice over, which is exactly how
    olmOCR came to be unreachable (CERTIFICATE_VERIFY_FAILED)."""
    from codai.api import pod_http
    seen = {}

    def _post(url, json=None, headers=None, timeout=None, **kw):
        seen.update(url=url, json=json, headers=headers or {})
        return _ok("<div data-label=\"Text\"><p>ok</p></div>")
    monkeypatch.setattr(pod_http, "post", _post)
    eng = ch.ChandraEngine(_cfg(chandra_serve="server",
                                chandra_server_url="https://10.0.0.9:8000",
                                chandra_api_key="tok"))
    eng.load()
    assert "ok" in eng._ask("data:image/png;base64,AA", 0.0, 0.1)
    assert seen["url"] == "https://10.0.0.9:8000/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer tok"


def test_the_image_is_sent_before_the_instruction(monkeypatch):
    """The order chandra's own client builds. For a VLM the order it was trained on is
    part of the prompt."""
    from codai.api import pod_http
    seen = {}
    monkeypatch.setattr(pod_http, "post",
                        lambda url, json=None, **kw: (seen.update(json=json),
                                                      _ok("<p>x</p>"))[1])
    eng = ch.ChandraEngine(_cfg(chandra_serve="server",
                                chandra_server_url="http://h:8000"))
    eng.load()
    eng._ask("data:image/png;base64,AA", 0.0, 0.1)
    parts = seen["json"]["messages"][0]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[1]["type"] == "text"


def test_sampling_is_greedy_by_default(monkeypatch):
    from codai.api import pod_http
    seen = {}
    monkeypatch.setattr(pod_http, "post",
                        lambda url, json=None, **kw: (seen.update(json=json),
                                                      _ok("<p>x</p>"))[1])
    eng = ch.ChandraEngine(_cfg(chandra_serve="server",
                                chandra_server_url="http://h:8000"))
    eng.load()
    eng._ask("data:,", 0.0, 0.1)
    assert seen["json"]["temperature"] == 0.0
    assert seen["json"]["top_p"] == 0.1
    assert seen["json"]["max_tokens"] == ch.DEFAULT_MAX_TOKENS


def test_an_unreachable_server_is_a_503_not_a_500(monkeypatch):
    from codai.api import pod_http

    def _boom(*a, **kw):
        raise OSError("connection refused")
    monkeypatch.setattr(pod_http, "post", _boom)
    eng = ch.ChandraEngine(_cfg(chandra_serve="server",
                                chandra_server_url="http://h:8000"))
    eng.load()
    with pytest.raises(OcrError) as e:
        eng._ask("data:,", 0.0, 0.1)
    assert e.value.status == 503


def test_an_http_error_is_reported_with_its_body(monkeypatch):
    from codai.api import pod_http
    monkeypatch.setattr(pod_http, "post",
                        lambda *a, **kw: _Reply({}, status=500, text="CUDA OOM"))
    eng = ch.ChandraEngine(_cfg(chandra_serve="server",
                                chandra_server_url="http://h:8000"))
    eng.load()
    with pytest.raises(OcrError) as e:
        eng._ask("data:,", 0.0, 0.1)
    assert e.value.status == 502
    assert "CUDA OOM" in str(e.value)


def test_an_empty_reply_is_an_error_not_an_empty_page(monkeypatch):
    """A silently empty page is how a whole document goes missing."""
    from codai.api import pod_http
    monkeypatch.setattr(pod_http, "post", lambda *a, **kw: _ok("   "))
    eng = ch.ChandraEngine(_cfg(chandra_serve="server",
                                chandra_server_url="http://h:8000"))
    eng.load()
    with pytest.raises(OcrError):
        eng._ask("data:,", 0.0, 0.1)


def test_a_multipart_reply_is_joined():
    assert ch._content_of(
        {"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}) == "ab"


# --------------------------------------------------------------- the manager #
def test_the_manager_knows_the_engine():
    assert "chandra" in om._ENGINE_FACTORIES


def test_it_declares_no_per_instance_vram():
    """An instance is one in-flight page against a shared server. Counting a GB each
    asked a 24-instance surya pool to free 24 GB."""
    assert ch.ChandraEngine(_cfg()).vram_gb() == 0.0


def test_only_the_vllm_mode_demands_room_before_it_loads():
    assert ch.ChandraEngine(_cfg(chandra_serve="server",
                                 chandra_server_url="http://h:1")).prelaunch_vram_gb() == 0.0
    assert ch.ChandraEngine(_cfg(chandra_serve="model",
                                 chandra_model_id="x")).prelaunch_vram_gb() == 0.0


def test_the_catalogue_says_it_returns_boxes_regions_and_tables():
    mgr = om.OcrManager()
    mgr.configure(_cfg())
    row = next(r for r in mgr.engine_catalogue()["data"] if r["id"] == "chandra")
    assert row["returns"] == {"boxes": True, "regions": True, "tables": True,
                              "meta": True}
    assert row["kind"] == "pipeline"


def test_with_layout_off_the_catalogue_stops_promising_boxes():
    """Because with it off there genuinely are none, and a client choosing on
    capability would be choosing on a lie."""
    mgr = om.OcrManager()
    mgr.configure(_cfg(chandra_layout=False))
    row = next(r for r in mgr.engine_catalogue()["data"] if r["id"] == "chandra")
    assert row["returns"]["boxes"] is False
    assert row["returns"]["regions"] is False


def test_it_is_reachable_by_the_checkpoint_name_as_well_as_the_engine_name():
    """A client asks for OCR with the model id it saw in /v1/models, not with the name
    of the pipeline we happen to drive it with."""
    mgr = om.OcrManager()
    mgr.configure(_cfg())
    assert mgr.resolve_engine("datalab-to/chandra-ocr-2") == "chandra"
    assert mgr.resolve_engine("chandra") == "chandra"


def test_it_has_its_own_rasterisation_dpi_for_the_same_reason_surya_does():
    """The pixel ceiling is identical — chandra/model/util.py uses the same
    3072x2048 — so feeding it the shared 200 wastes the same 38%."""
    mgr = om.OcrManager()
    mgr.configure(_cfg(dpi=200))
    assert mgr.engine_dpi("chandra") == 250
    assert mgr.engine_dpi("paddle") == 200


def test_a_dpi_of_zero_falls_back_to_the_shared_one():
    mgr = om.OcrManager()
    mgr.configure(_cfg(dpi=180, chandra_dpi=0))
    assert mgr.engine_dpi("chandra") == 180


def test_lang_is_accepted_but_does_not_steer_it():
    """Its prompt has no language slot, like surya-2's and olmOCR's. Saying so is what
    lets a caller tell a language that took effect from one merely accepted."""
    assert om.engine_uses_language("chandra") is False
    mgr = om.OcrManager()
    mgr.configure(_cfg())
    row = next(r for r in mgr.engine_catalogue()["data"] if r["id"] == "chandra")
    assert row["language_aware"] is False


def test_it_keeps_one_pool_whatever_language_is_asked_for():
    """A language-keyed pool is a second set of loaded instances holding VRAM, for a
    parameter this engine cannot act on."""
    import asyncio
    mgr = om.OcrManager()
    mgr.configure(_cfg())

    async def _go():
        a = await mgr._get_pool("chandra", "it")
        b = await mgr._get_pool("chandra", "de")
        return a is b
    assert asyncio.run(_go())


def test_a_config_change_invalidates_the_pools():
    """Otherwise a settings change appears to save and changes nothing until a
    restart."""
    base = om.OcrManager._pool_params(_cfg())
    for field, value in (("chandra_serve", "server"), ("chandra_model", "other"),
                         ("chandra_layout", False), ("chandra_instances", 4),
                         ("chandra_accept_license", False)):
        assert om.OcrManager._pool_params(_cfg(**{field: value})) != base, field


def test_it_is_in_the_automatic_fallback_chain():
    """A 503 means a document nobody read, which is strictly worse than a
    transcription from the second-best engine."""
    mgr = om.OcrManager()
    cfg = _cfg(paddle_enabled=True, fallback_engines="auto")
    mgr.configure(cfg)
    assert "chandra" in mgr._fallback_order("surya")
    # …and it is tried after the engines with fewer ways to fail.
    order = mgr._fallback_order("surya")
    assert order.index("paddle") < order.index("chandra")


# ------------------------------------------------------------ a rented pod #
def test_a_pod_cannot_accept_the_licence_on_the_operators_behalf():
    """Asking for surya by name IS the GPL acceptance (that is the deployment's own
    decision reaching the pod). Chandra's terms are narrower — a revenue ceiling and a
    non-compete — so only a decision already recorded in this install's config travels."""
    import inspect
    from codai.api import runpod_worker as rw
    src = inspect.getsource(rw._plan)
    assert '_asked == "chandra" and _chandra_accepted()' in src
    assert '_asked == "surya" or _surya_accepted()' in src


def test_a_pod_image_without_a_vlm_server_downgrades_chandra_to_doctr():
    from codai.api import runpod_worker as rw
    assert rw.pod_ocr_engine("chandra") == "doctr"
    assert rw.pod_ocr_engine("paddle") == "paddle"
