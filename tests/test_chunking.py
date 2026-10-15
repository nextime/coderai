"""Retrieval chunking — `POST /v1/chunk`.

Splits a document into the pieces a retrieval index stores. NOT embedding
windowing: `embed_long_input` makes one over-long input fit a model window and
pools the pieces into ONE vector; this produces MANY chunks, each indexable,
each carrying its section, parent and references. This must never pool and
never drop text.

The eight cases below are the fidelity list from docs/chunking.md. What they
CANNOT do is prove byte-identity with the production chunker they port: that
needs fixtures from the live corpus, which are not in this repo. They prove the
algorithm's documented behaviour on each path; the identity check is a separate
step against real documents, and it is the one that decides whether a re-chunk
is safe.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import chunking as ch                          # noqa: E402


DEFAULTS = dict(chunk_tokens=800, overlap_tokens=120, chars_per_token=4)
TARGET = 800 * 4          # 3200
OVERLAP = 120 * 4         # 480


# ------------------------------------------------------------ 1. a ruling #
RULING = """TRIBUNALE ORDINARIO DI MILANO
Sezione Prima Civile

SVOLGIMENTO DEL PROCESSO
Con atto di citazione notificato il 3 marzo l'attore ha convenuto in giudizio.
Il convenuto si e costituito chiedendo il rigetto della domanda.

MOTIVI DELLA DECISIONE
La domanda e fondata. Ai sensi dell'art. 2043 c.c. il danno va risarcito.
Si richiama altresi il d.lgs. 209/2005 in materia assicurativa.

P.Q.M.
Condanna il convenuto al pagamento delle spese.
"""


def test_a_ruling_is_sectioned_into_its_three_parts():
    r = ch.chunk_document(RULING, **DEFAULTS)
    labels = [s["section"] for s in r["sections"]]
    assert labels == ["intestazione", "fatto", "motivazione", "dispositivo"]
    assert r["structural_applied"] is True
    assert r["structure"] == "it-legal"


def test_every_chunk_knows_its_section_and_parent():
    r = ch.chunk_document(RULING, **DEFAULTS)
    for c in r["chunks"]:
        assert c["section"] == r["sections"][c["parent"]]["section"]


def test_the_header_line_stays_in_its_own_section():
    r = ch.chunk_document(RULING, **DEFAULTS)
    moti = [c for c in r["chunks"] if c["section"] == "motivazione"]
    assert moti and "MOTIVI DELLA DECISIONE" in moti[0]["text"]


# --------------------------------------- 2. a filing, conclusions twice #
FILING = """ECC.MO TRIBUNALE DI ROMA

PREMESSO CHE
la parte ricorrente ha subito un danno.

CONCLUSIONI
Nel merito, accertare la responsabilita.

Voglia il Tribunale condannare il resistente alle spese.
"""


def test_adjacent_duplicate_sections_merge():
    """Filings routinely open the conclusions twice. Left separate, one block
    becomes two and the `parent` ids multiply, defeating the point of
    `parent`."""
    r = ch.chunk_document(FILING, **DEFAULTS)
    labels = [s["section"] for s in r["sections"]]
    assert labels.count("conclusioni") == 1, labels
    conc = [s for s in r["sections"] if s["section"] == "conclusioni"][0]
    assert conc["n_chars"] > 0


def test_conclusioni_is_never_folded_into_dispositivo():
    """In a ruling the dispositivo is what the judge DECIDED; in a filing the
    conclusioni are what a party ASKED FOR. Merging them would let a lawyer's
    demand surface as a holding -- the most damaging mistake this index can
    make."""
    labels = [s["label"] for s in ch.IT_LEGAL["sections"]]
    assert "conclusioni" in labels and "dispositivo" in labels
    assert labels.index("conclusioni") != labels.index("dispositivo")
    r = ch.chunk_document(FILING, **DEFAULTS)
    assert "dispositivo" not in [s["section"] for s in r["sections"]]


# -------------------------------------------------- 3. the flat fallback #
def test_a_document_with_no_header_falls_back_flat_and_says_so():
    r = ch.chunk_document("Una lettera qualunque.\n\nSenza intestazioni note.",
                          **DEFAULTS)
    assert r["structural_applied"] is False
    assert all(c["section"] is None for c in r["chunks"])
    assert len(r["sections"]) == 1


def test_structural_false_skips_sectioning_entirely():
    r = ch.chunk_document(RULING, structural=False, **DEFAULTS)
    assert r["structural_applied"] is False
    assert [s["section"] for s in r["sections"]] == [None]


# ------------------------------- 4. one paragraph far over the target #
#: The real ceiling is target + overlap + 2, not target + overlap. The carried
#: tail is exactly `overlap` characters, the next piece can be `target`, and they
#: are joined with "\n\n" -- so 3200 + 480 + 2 = 3682 with the defaults.
#: docs/chunking.md says "3680" in prose and "observed max on the live corpus is
#: 3682" two lines later; the arithmetic and the measurement agree with each
#: other, and the prose is the odd one out. Asserting 3680 would fail a correct
#: port.
CEILING = TARGET + OVERLAP + 2


def test_an_over_long_paragraph_splits_on_sentences():
    para = " ".join(f"Frase numero {i} del paragrafo." for i in range(400))
    r = ch.chunk_document(para, structural=False, **DEFAULTS)
    assert r["n_chunks"] > 1
    assert all(c["n_chars"] <= CEILING for c in r["chunks"])
    assert max(c["n_chars"] for c in r["chunks"]) > TARGET, (
        "the carried tail should push at least one chunk past the target")


def test_a_single_sentence_over_the_target_is_sliced():
    """Nothing is ever discarded -- a chunker that drops text is worse than one
    that chunks badly, because the loss is invisible in the index."""
    one = "x" * (TARGET * 3)          # no sentence boundary at all
    r = ch.chunk_document(one, structural=False, refs=False, **DEFAULTS)
    assert r["n_chunks"] >= 3
    assert sum(1 for c in r["chunks"] for _ in c["text"]) >= len(one)


def test_no_text_is_lost_across_the_pieces():
    para = " ".join(f"Periodo {i}." for i in range(2000))
    r = ch.chunk_document(para, structural=False, refs=False,
                          overlap_tokens=0, chunk_tokens=50, chars_per_token=4)
    joined = "".join(c["text"] for c in r["chunks"])
    for i in (0, 999, 1999):
        assert f"Periodo {i}." in joined


# --------------------------------------------- 5/6. overlap and markdown #
def test_the_overlap_is_a_raw_character_tail():
    """Not sentence- or token-aligned, and it may begin mid-word. That is the
    production behaviour; 'improving' it moves every existing boundary."""
    pieces = ["A" * 100, "B" * 100, "C" * 100]
    out = ch.pack(pieces, target=150, overlap=20)
    assert len(out) >= 2
    assert out[1].startswith("A" * 20), out[1][:30]


def test_zero_overlap_starts_the_next_chunk_clean():
    out = ch.pack(["A" * 100, "B" * 100], target=150, overlap=0)
    assert out[1] == "B" * 100


def test_a_chunk_may_reach_target_plus_overlap_but_no_more():
    pieces = ["Z" * 300 for _ in range(20)]
    out = ch.pack(pieces, target=1000, overlap=200)
    assert out, "nothing packed"
    assert max(len(c) for c in out) <= 1000 + 200


def test_a_quoted_or_prefixed_header_is_recognised():
    """LEADING decoration is tolerated: `line_prefix` ([ \\t>*_#]*) plus the
    `[*#>_ ]*` some patterns carry themselves."""
    doc = "Intestazione\n\n> motivazione\nLe ragioni.\n\n* premesso che\nI fatti.\n"
    r = ch.chunk_document(doc, **DEFAULTS)
    labels = [s["section"] for s in r["sections"]]
    assert "motivazione" in labels and "premessa" in labels, labels


@pytest.mark.parametrize("header,label", [
    ("**FATTO**", "fatto"),
    ("*FATTO*", "fatto"),
    ("## FATTO", "fatto"),
    ("> fatto", "fatto"),
    ("FATTO", "fatto"),
    ("  fatto  ", "fatto"),
    ("__DISPOSITIVO__", "dispositivo"),
    ("# Dispositivo #", "dispositivo"),
    ("**PREMESSA**", "premessa"),
    ("> motivazione", "motivazione"),
    ("* premesso che", "premessa"),
])
def test_a_markdown_decorated_header_is_recognised(header, label):
    """Decoration on EITHER side of a header is tolerated.

    `**FATTO**` used to be missed twice over: `^\\s*fatto\\s*$` tolerates only
    whitespace before the word and requires the line to end right after it, so
    both the leading and the trailing asterisks defeated it. The patterns that
    already coped (`premesso che`, `conclusioni`) spell it `^\\s*[*#>_ ]*` for
    exactly this reason; these three never did.
    """
    doc = f"Testa\n\n{header}\nCorpo del testo.\n"
    got = [s["section"] for s in ch.chunk_document(doc, **DEFAULTS)["sections"]]
    assert label in got, got


def test_widening_the_anchors_is_purely_additive():
    """Every line that matched the old `^\\s*X\\s*$` still matches the new
    `^[\\s*#>_]*X[\\s*#>_]*$`, because \\s is a subset of that class. So no
    EXISTING match moves; what changes is which documents get sectioned at all.

    That distinction is the whole risk of the change: a document whose headers
    were markdown-emphasised used to fall back to flat chunking and now gets
    labels, so its chunks differ from an index built before this and it needs
    re-chunking. A document already sectioned is untouched.
    """
    import re
    for word in ("fatto", "dispositivo", "premessa"):
        old = re.compile(rf"^\s*{word}\s*$", re.I | re.M)
        new = re.compile(rf"^{ch._LEAD}{word}{ch._TRAIL}$", re.I | re.M)
        for line in (word, f"  {word}", f"{word}  ", f"\t{word}\t"):
            if old.search(line):
                assert new.search(line), f"{word}: {line!r} no longer matches"


def test_an_already_sectioned_document_is_unchanged_by_the_widening():
    """The ruling has plain headers, so its chunking must be identical to what
    it was before the anchors were widened -- asserted by boundary, not by
    eyeballing."""
    r = ch.chunk_document(RULING, **DEFAULTS)
    assert [s["section"] for s in r["sections"]] == [
        "intestazione", "fatto", "motivazione", "dispositivo"]


# ----------------------------------------------------- 7. binary input #
@pytest.mark.parametrize("raw,why", [
    (b"PK\x03\x04rest of a zip", "zip/OOXML header"),
    (b"%PDF-1.7\nstuff", "pdf header"),
    ("[Content_Types].xml and more".encode(), "ooxml marker"),
])
def test_container_bytes_are_refused(raw, why):
    """A chunker that accepts those bytes embeds garbage and the index cannot
    tell afterwards. Measured on a production corpus: 0.59% of chunks with
    control characters, 0.04% raw OOXML."""
    assert ch.looks_binary(raw), why
    with pytest.raises(ch.ChunkError) as e:
        ch.chunk_document(raw.decode("utf-8", "ignore"), **DEFAULTS)
    assert e.value.code == "not_text"


def test_control_characters_above_a_small_ratio_are_refused():
    assert ch.looks_binary("text" + "\x00" * 50)


def test_ordinary_text_with_tabs_and_newlines_is_fine():
    assert ch.looks_binary("riga uno\n\triga due\n\fpagina\n") is None


def test_require_text_false_opts_out_deliberately():
    r = ch.chunk_document("%PDF-1.7 pretend", require_text=False,
                          structural=False, refs=False, **DEFAULTS)
    assert r["n_chunks"] == 1


# ------------------------------------------------- 8. degenerate paths #
def test_empty_input_is_an_error_not_an_empty_answer():
    for bad in ("", "   \n\n  "):
        with pytest.raises(ch.ChunkError) as e:
            ch.chunk_document(bad, **DEFAULTS)
        assert e.value.code == "empty_input"


def test_carriage_returns_are_normalised_away():
    r = ch.chunk_document("riga\r\nuno\r\n\r\nriga due", structural=False,
                          refs=False, **DEFAULTS)
    assert "\r" not in r["chunks"][0]["text"]


# ------------------------------------------------------------ templates #
def test_an_unknown_template_name_is_an_error_never_a_fallback():
    """A caller who asked for their own template and silently got Italian legal
    sectioning would index the wrong boundaries and never know."""
    with pytest.raises(ch.ChunkError) as e:
        ch.chunk_document(RULING, structure="no-such-template", **DEFAULTS)
    assert e.value.code == "unknown_template"
    assert e.value.status == 404


def test_an_inline_template_is_accepted():
    tpl = {"name": "mine", "head_label": "head", "merge_adjacent": True,
           "sections": [{"label": "uno", "pattern": r"^CAPO\s+I\b"},
                        {"label": "due", "pattern": r"^CAPO\s+II\b"}]}
    doc = "prefazione\n\nCAPO I\nprimo testo\n\nCAPO II\nsecondo testo\n"
    r = ch.chunk_document(doc, structure=tpl, **DEFAULTS)
    assert [s["section"] for s in r["sections"]] == ["head", "uno", "due"]
    assert r["structure"] == "mine"


def test_an_inline_template_beats_a_registered_one():
    reg = {"mine": {"name": "registered", "sections":
                    [{"label": "r", "pattern": "^NOPE"}]}}
    tpl = {"name": "inline", "sections": [{"label": "i", "pattern": r"^CAPO"}]}
    r = ch.chunk_document("x\n\nCAPO\ny\n", structure=tpl, registered=reg,
                          **DEFAULTS)
    assert r["structure"] == "inline"


def test_a_registered_template_is_selectable_by_name():
    reg = {"mio": {"name": "mio", "head_label": None,
                   "sections": [{"label": "a", "pattern": r"^SEZIONE"}]}}
    r = ch.chunk_document("testa\n\nSEZIONE\ncorpo\n", structure="mio",
                          registered=reg, **DEFAULTS)
    assert r["structure"] == "mio"
    # head_label None drops the text before the first match
    assert [s["section"] for s in r["sections"]] == ["a"]


@pytest.mark.parametrize("tpl", [
    {"sections": []},
    {"sections": [{"label": "x"}]},
    {"sections": [{"label": "x", "pattern": "([unclosed"}]},
    {"sections": [{"label": "x", "pattern": "a" * 3000}]},
    "not-an-object-and-not-registered",
])
def test_a_malformed_template_is_bad_template_not_a_crash(tpl):
    with pytest.raises(ch.ChunkError) as e:
        ch.chunk_document(RULING, structure=tpl, **DEFAULTS)
    assert e.value.code in ("bad_template", "unknown_template")


def test_too_many_sections_is_refused():
    tpl = {"sections": [{"label": f"s{i}", "pattern": f"^X{i}"}
                        for i in range(ch.MAX_TEMPLATE_SECTIONS + 1)]}
    with pytest.raises(ch.ChunkError) as e:
        ch.chunk_document(RULING, structure=tpl, **DEFAULTS)
    assert e.value.code == "bad_template"


def test_one_alternation_is_scanned_not_one_pass_per_pattern():
    """Scanning per pattern and merging changes WHICH header claims an
    overlapping position -- a different segmentation for the same document."""
    src = (ROOT / "codai/api/chunking.py").read_text()
    fn = src.split("def compile_template")[1].split("\ndef ")[0]
    assert '"|".join(parts)' in fn
    assert "finditer" in src.split("def split_sections")[1].split("\ndef ")[0]


# ----------------------------------------------------------- references #
def test_articles_normalise_with_their_code():
    assert ch.extract_refs("ai sensi dell'art. 2043 c.c.") == ["art.2043 c.c."]
    assert ch.extract_refs("art.112 cpc") == ["art.112 c.p.c."]


def test_an_article_suffix_keeps_its_form_without_spaces():
    assert ch.extract_refs("art. 2055 bis c.c.") == ["art.2055bis c.c."]


def test_acts_normalise_to_kind_number_year():
    assert ch.extract_refs("il d.lgs. 209/2005") == ["d.lgs.209/2005"]
    assert ch.extract_refs("legge 24/2017") == ["l.24/2017"]


# Cases taken one by one from a diff against the implementation this ports,
# run side by side on the same documents. Each was a real divergence.
def test_an_article_enumeration_resolves_to_its_first_article():
    """`artt. 163, 164 e 167 c.p.c.` must reach the code at the end and emit
    the FIRST article only. Requiring the code to sit immediately after the
    number dropped the whole citation instead."""
    assert ch.extract_refs("artt. 115 e 116 c.p.c.") == ["art.115 c.p.c."]
    assert ch.extract_refs("artt. 163, 164 e 167 c.p.c.") == ["art.163 c.p.c."]


def test_an_article_suffix_may_be_hyphenated():
    """`art. 2055-bis c.c.` keeps its hyphen; `art. 2055 bis c.c.` loses its
    space. Only whitespace is squeezed out of the number."""
    assert ch.extract_refs("art. 2055-bis c.c.") == ["art.2055-bis c.c."]
    assert ch.extract_refs("art. 2055 bis c.c.") == ["art.2055bis c.c."]


def test_the_code_may_be_spelled_out():
    assert ch.extract_refs("art. 2043 cod. civ.") == ["art.2043 c.c."]


def test_articolo_spelled_out_is_not_an_article_citation():
    """`art`/`artt` only. Accepting `articolo` adds references the corpus this
    was measured against does not have, which is a divergence even though it
    looks like an improvement."""
    assert ch.extract_refs("articolo 2043 c.c.") == []
    assert ch.extract_refs("visto l'art. 2043 della c.c.") == []


def test_an_act_may_be_written_with_del_instead_of_a_slash():
    assert ch.extract_refs("legge 241 del 1990") == ["l.241/1990"]
    assert ch.extract_refs("l 241/1990") == []        # the dot is required


def test_a_bare_article_number_is_not_a_reference():
    """Reported from production: `art. 380` and `art. 360` came back as
    `['art.380']` -- an unqualified number is ambiguous (every code has an
    art. 360), so it would put three different articles under one exact-match
    key. The code suffix is required, and a citation without one is dropped."""
    assert ch.extract_refs("si veda l'art. 360 n. 3") == []
    assert ch.extract_refs("art.380, art.360, art.115, art.395") == []
    assert ch.extract_refs("violazione dell'art. 380 c.p.c.") == ["art.380 c.p.c."]


def test_an_act_year_may_be_two_digits_and_is_kept_verbatim():
    """`d.p.r. 115/02` was extracted as nothing, because the year was pinned to
    four digits. It is kept as written: expanding `02` to `2002` would file the
    same act under two keys depending on the source's style."""
    assert ch.extract_refs("ai sensi del d.p.r. 115/02") == ["d.p.r.115/02"]
    assert ch.extract_refs(
        "del d.p.r. 115/02 e del d.lgs. 446/1998") == ["d.p.r.115/02",
                                                       "d.lgs.446/1998"]


# ------------------------------------------------- section bodies are strip #
def test_two_adjacent_sections_merge_into_one_paragraph_not_two():
    """THE off-by-one. An unstripped section body ends with the blank line
    before the next header, so merging two of them inserted a PARAGRAPH break
    where the original puts a line break -- and a paragraph break is packed as
    `\n\n` while a line break rides along inside one piece. One character per
    merged section, which is enough to make a re-chunked index incomparable."""
    txt = ("CONCLUSIONI\n\nchiede il rigetto.\n\n"
           "Voglia il Tribunale\n\naccogliere le domande.\n")
    secs, applied, _ = ch.split_sections(txt, ch.IT_LEGAL)
    assert applied
    merged = [s for s in secs if s["label"] == "conclusioni"]
    assert len(merged) == 1, [s["label"] for s in secs]
    body = merged[0]["text"]
    assert not body.startswith(("\n", " ")) and not body.endswith(("\n", " "))
    assert "chiede il rigetto.\nVoglia il Tribunale" in body


def test_an_empty_section_does_not_spend_a_parent_id():
    """Two headers back to back leave an empty body between them. Kept, it
    would take a `parent` and shift every id after it."""
    txt = "testa.\n\nSVOLGIMENTO DEL PROCESSO\nP.Q.M.\n\ncondanna."
    secs, _, _ = ch.split_sections(txt, ch.IT_LEGAL)
    assert all(s["text"].strip() for s in secs)


# -------------------------------------------------------- strict template #
def test_strict_template_keeps_the_original_anchors():
    """`it-legal` matches a markdown-emphasised header; the implementation this
    ports does not, so a corpus indexed with it must be re-chunked the old way
    or not at all. `it-legal-strict` is that old way, and it is the ONLY
    deliberate behavioural difference between the two."""
    txt = "testa.\n\n**FATTO**\n\ncorpo.\n\n**IN DIRITTO**\n\nmotivi."
    wide = ch.chunk_document(txt, structure="it-legal", **DEFAULTS)
    strict = ch.chunk_document(txt, structure="it-legal-strict", **DEFAULTS)
    assert [c["section"] for c in wide["chunks"]] == ["intestazione", "fatto",
                                                      "motivazione"]
    assert [c["section"] for c in strict["chunks"]] == ["intestazione",
                                                        "motivazione"]
    assert strict["structure"] == "it-legal-strict"


def test_strict_and_wide_agree_on_an_undecorated_document():
    """The widening is purely additive: a document whose headers carry no
    markdown is sectioned identically by both."""
    wide = ch.chunk_document(RULING, structure="it-legal", **DEFAULTS)
    strict = ch.chunk_document(RULING, structure="it-legal-strict", **DEFAULTS)
    assert [(c["text"], c["section"], c["parent"]) for c in wide["chunks"]] == \
           [(c["text"], c["section"], c["parent"]) for c in strict["chunks"]]


def test_references_are_deduplicated_in_first_seen_order():
    txt = "art. 2054 c.c., poi art. 2043 c.c., di nuovo art. 2054 c.c."
    assert ch.extract_refs(txt) == ["art.2054 c.c.", "art.2043 c.c."]


def test_extract_refs_false_skips_the_work():
    r = ch.chunk_document(RULING, refs=False, **DEFAULTS)
    assert all("norm_refs" not in c for c in r["chunks"])


def test_refs_are_per_chunk_not_per_document():
    r = ch.chunk_document(RULING, **DEFAULTS)
    assert any(c.get("norm_refs") for c in r["chunks"])
    assert any(c.get("norm_refs") == [] for c in r["chunks"])


# -------------------------------------------------------------- offsets #
def test_offsets_are_omitted_by_default():
    r = ch.chunk_document(RULING, **DEFAULTS)
    assert all("start" not in c for c in r["chunks"])


def test_offsets_point_into_the_normalised_text():
    r = ch.chunk_document(RULING, include_offsets=True, **DEFAULTS)
    norm = RULING.replace("\r", "")
    for c in r["chunks"]:
        if "start" in c:
            assert norm[c["start"]:c["end"]] == c["text"]


# ------------------------------------------------- it is not the windower #
def test_chunking_never_pools_and_never_drops():
    """The distinction from embed_long_input, asserted: many chunks out, and
    the text accounted for."""
    r = ch.chunk_document(RULING, overlap_tokens=0, **{
        k: v for k, v in DEFAULTS.items() if k != "overlap_tokens"})
    assert r["n_chunks"] > 1
    for c in r["chunks"]:
        assert c["text"].strip()


def test_the_endpoint_needs_no_model_or_tokenizer():
    """Token counts are approximated as characters / chars_per_token so this
    stays a pure CPU operation -- which is what lets a light-core image serve
    it with no GPU and no pod."""
    src = (ROOT / "codai/api/chunking.py").read_text()
    for forbidden in ("AutoTokenizer", "import torch", "from transformers"):
        assert forbidden not in src, forbidden


def test_the_params_are_echoed_so_a_caller_can_tell_what_applied():
    r = ch.chunk_document(RULING, chunk_tokens=100, overlap_tokens=10,
                          chars_per_token=5)
    assert r["params"] == {"chunk_tokens": 100, "overlap_tokens": 10,
                           "chars_per_token": 5}


# ------------------------------------------- forwarding it to a remote #
#
# /v1/chunk rides the EMBEDDINGS capability: chunks are on their way to be
# embedded, the embeddings image is full-core so it carries the chunker, and a
# `chunk` capability of its own would be one nothing advertises -- or would rent
# a GPU pod to run a regex split.
def test_the_chunk_endpoints_are_forwardable():
    from codai.api.remote_gateway import capability_for, _SKIP_PATHS
    for p in ("/v1/chunk", "/v1/chunk/file", "/v1/chunk/templates"):
        assert capability_for(p) == "embeddings", p
        assert p not in _SKIP_PATHS, p


def test_a_named_registered_template_is_inlined_before_it_leaves():
    """A template registered HERE does not exist on the remote. Forwarded by
    name it comes back 404 -- or, worse, matches a DIFFERENT template with the
    same name and returns different boundaries with nothing to say so."""
    import json
    from codai.api import remote_gateway as rg
    import codai.api.chunking as chk

    mine = {"name": "mine", "sections": [{"label": "x", "pattern": "^CAPO"}]}
    orig = chk._registered_templates
    chk._registered_templates = lambda: {"mine": mine}
    try:
        body = json.dumps({"text": "t", "structure": "mine"}).encode()
        out = rg._inline_chunk_template("/v1/chunk",
                                        {"content-type": "application/json"}, body)
        assert json.loads(out)["structure"] == mine
    finally:
        chk._registered_templates = orig


def test_a_builtin_name_is_left_alone():
    """Every coderai has the built-ins; sending the object would only make the
    request bigger."""
    import json
    from codai.api import remote_gateway as rg
    body = json.dumps({"text": "t", "structure": "it-legal"}).encode()
    out = rg._inline_chunk_template("/v1/chunk",
                                    {"content-type": "application/json"}, body)
    assert json.loads(out)["structure"] == "it-legal"


def test_an_inline_template_is_passed_through_untouched():
    import json
    from codai.api import remote_gateway as rg
    tpl = {"name": "inline", "sections": [{"label": "a", "pattern": "^X"}]}
    body = json.dumps({"text": "t", "structure": tpl}).encode()
    out = rg._inline_chunk_template("/v1/chunk",
                                    {"content-type": "application/json"}, body)
    assert json.loads(out)["structure"] == tpl


@pytest.mark.parametrize("path,ctype,body", [
    ("/v1/embeddings", "application/json", b'{"structure":"mine"}'),
    ("/v1/chunk", "multipart/form-data; boundary=x", b"--x--"),
    ("/v1/chunk", "application/json", b"not json at all"),
    ("/v1/chunk", "application/json", b'[]'),
    ("/v1/chunk", "application/json", b'{}'),
])
def test_inlining_never_raises_and_leaves_what_it_cannot_handle(path, ctype, body):
    """A body it cannot parse is forwarded untouched and the remote answers for
    itself -- a rewrite hook that can throw would take down every forwarded
    request, not just a chunk one."""
    from codai.api import remote_gateway as rg
    assert rg._inline_chunk_template(path, {"content-type": ctype}, body) == body
