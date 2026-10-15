# Retrieval chunking — `POST /v1/chunk`

Splitting a document into the pieces a retrieval index stores, with the
document's own structure decided **per request** by a named (or inline)
*structure template*.

This is a port of the chunker that runs in Digesta (`app/chunking.py`), which
has been chunking Italian case law and party filings in production. The port has
one hard requirement: with the built-in `it-legal` template and default sizes it
must produce **byte-identical** output to Digesta, so Digesta can delegate
chunking here and keep its existing index comparable. Everything that is
hardcoded Italian-legal there becomes a template here.

## Not to be confused with embedding windowing

CoderAI already splits over-long embedding inputs — `embed_long_input`
(`chunk | truncate | error`) and `embed_chunk_pool` (`mean | first`) in
`codai/api/embeddings.py`. That solves a different problem and must stay as it
is:

| | `embed_long_input` (exists) | `/v1/chunk` (this doc) |
|---|---|---|
| Purpose | make one input fit the model window | split a document for retrieval |
| Output | **one** vector per input (windows pooled) | **many** chunks, each indexable |
| Metadata | none | section, parent, references, offsets |
| Chosen by | model window | the corpus and the retrieval design |

A caller may well use both: `/v1/chunk` to cut the document, then
`/v1/embeddings` per chunk (where windowing is the safety net for a chunk that
still overflows). `/v1/chunk` must never pool and must never drop text.

## Why structure-aware at all

Length-only chunking cuts a legal argument wherever the character budget runs
out. The measured consequence in Digesta was that a chunk could straddle the
`fatto` → `motivazione` boundary, so a retrieved passage mixed what the parties
alleged with what the court held. Sectioning first, then length-chunking *inside*
each section, removes that class of error, and gives every chunk a label and a
recoverable parent section.

Two findings from production worth carrying into the design:

- **One marker family is not enough.** With ruling markers only
  (`fatto`/`motivazione`/`dispositivo`), just **8–10%** of 400 merito documents
  were recognised — party filings simply do not have that structure. The
  `it-legal` template therefore carries two families (see below).
- **Adjacent duplicate sections must merge.** Filings routinely open the
  conclusions twice (`CONCLUSIONI`, then `Voglia il Tribunale…`). Left separate,
  one block becomes two and the `parent` ids multiply, defeating the purpose of
  `parent`.

Current behaviour on the live corpus (0.05% sample of ~27M chunks): ~78% of
chunks carry a section label, ~22% fall back to flat chunking.

## Request

```http
POST /v1/chunk
Authorization: Bearer <key>
Content-Type: application/json
```

```jsonc
{
  "text": "…",                  // or "file" (multipart) — exactly one of them
  "chunk_tokens": 800,          // target size, in approximated tokens
  "overlap_tokens": 120,        // carried tail, in approximated tokens
  "chars_per_token": 4,         // the approximation divisor
  "structural": true,           // false = flat length chunking, no sectioning
  "structure": "it-legal",      // template name, or an inline object (below)
  "extract_refs": "it-legal",   // reference extractor: name, inline, or false
  "require_text": true,         // reject binary input instead of chunking it
  "include_offsets": false      // add char offsets into the source text
}
```

Every field except the input is optional; omitted fields take the server
defaults, which are the Digesta defaults. Multipart form submission uses the
same field names as string values, matching how `/v1/ocr` takes its options.

### Defaults and config keys

| Field | Default | Config key | Env |
|---|---|---|---|
| `chunk_tokens` | 800 | `chunk_tokens` | `CODERAI_CHUNK_TOKENS` |
| `overlap_tokens` | 120 | `chunk_overlap` | `CODERAI_CHUNK_OVERLAP` |
| `chars_per_token` | 4 | `chunk_chars_per_token` | `CODERAI_CHUNK_CHARS_PER_TOKEN` |
| `structural` | `true` | `chunk_structural` | `CODERAI_CHUNK_STRUCTURAL` |
| `structure` | `it-legal` | `chunk_structure` | `CODERAI_CHUNK_STRUCTURE` |
| `require_text` | `true` | `chunk_require_text` | `CODERAI_CHUNK_REQUIRE_TEXT` |

Token counts are **approximated as characters ÷ `chars_per_token`** so the
endpoint needs no tokenizer and no model loaded — `/v1/chunk` must stay a pure
CPU text operation, callable with no GPU and no pod. With the defaults the real
budget is `800 × 4 = 3200` characters and the overlap `120 × 4 = 480`.

## Response

```jsonc
{
  "chunks": [
    {
      "ord": 0,                       // 0-based, document order
      "text": "…",
      "n_chars": 2464,
      "section": "motivazione",       // null when unlabelled
      "parent": 2,                    // 0-based section index; shared by its chunks
      "norm_refs": ["art.2043 c.c.", "d.lgs.209/2005"],
      "start": 10433, "end": 12897    // only with include_offsets
    }
  ],
  "n_chunks": 37,
  "sections": [                       // the segmentation, for context recovery
    {"parent": 0, "section": "intestazione", "n_chars": 1204}
  ],
  "structure": "it-legal",            // template actually applied
  "structural_applied": true,         // false = fell back to flat
  "params": {"chunk_tokens": 800, "overlap_tokens": 120, "chars_per_token": 4}
}
```

`parent` is the index into `sections`, so a caller can retrieve a chunk and then
pull the whole section it came from as context — the behaviour Digesta's
retrieval depends on.

## The structure template

A template is an ordered list of sections, each a label plus a pattern that
*opens* that section. Order in the list is only for registration; **order in the
document drives the segmentation**.

```jsonc
{
  "name": "it-legal",
  "head_label": "intestazione",   // text before the first match; null = drop it
  "flags": "im",                  // applied to every pattern
  "line_prefix": "[ \\t>*_#]*",   // tolerated before a header (markdown, quotes)
  "merge_adjacent": true,         // merge consecutive same-label sections
  "keep_header_line": true,       // the header stays in its section's text
  "sections": [
    {"label": "premessa",    "pattern": "tanto\\s+premesso|…"},
    {"label": "fatto",       "pattern": "svolgimento\\s+del\\s+processo|…"}
  ]
}
```

Rules an implementation must honour:

- Patterns are anchored at a line start, after `line_prefix`, and are matched
  case-insensitively and multi-line (`flags` default `im`). This is what lets a
  header survive markdown emphasis or quoted text.
- All patterns compile into **one** alternation with named groups, so a single
  left-to-right scan yields the marks in document order and the first matching
  alternative wins at a given position. Do not scan per-pattern and merge: that
  changes which header claims an overlapping position.
- `keep_header_line: true` means a section runs from its match **start** to the
  next match start, header included.
- `merge_adjacent: true` joins a section to the previous one with `"\n"` when
  the labels are equal.
- No match anywhere → `structural_applied: false` and the whole text is chunked
  flat as a single unlabelled section.

Templates may be supplied inline per request, or registered server-side as
`chunk_templates` in the config (same JSON shape) and selected by name. Inline
beats registered; an unknown name is an error, never a silent fallback.

### Built-in `it-legal`

Carries both families, because party filings outnumber rulings in the merito
corpus.

**`it-legal` is deliberately NOT byte-identical to the original.** Its
single-word anchors accept a DECORATED heading — `**P.Q.M.**`, `_DISPOSITIVO_`,
`## FATTO` — because OCR engines that emit markdown (chandra, olmOCR) produce
exactly those, and the original's `^\s*dispositivo\s*$` does not match one. The
difference is `[\s*#>_]*` where the original has `\s*`, on the three anchored
patterns (`premess[ae]`, `fatto`, `dispositivo`).

That difference is VISIBLE in the output: a decorated `**P.Q.M.**` becomes its
own `dispositivo` section here and stays inside `motivazione` there — one extra
chunk, usually a short one, and the preceding chunk shorter by the same text.
If you are re-chunking a corpus indexed with the original, use
[`it-legal-strict`](#built-in-it-legal-strict), which IS byte-identical; a
boundary that moves makes the new index incomparable with the old.

Patterns:

| Label | Pattern |
|---|---|
| `premessa` | `tanto\s+premesso\|ci[oò]\s+premesso\|^\s*[*#>_ ]*premesso\s+che\b\|^[\s*#>_]*premess[ae][\s*#>_]*$` |
| `fatto` | `svolgimento\s+del\s+processo\|ragioni\s+di\s+fatto\|in\s+fatto\b\|fatto\s+e\s+diritto\|^[\s*#>_]*fatto[\s*#>_]*$\|esposizione\s+del\s+fatto` |
| `motivazione` | `motivi\s+della\s+decisione\|ragioni\s+della\s+decisione\|motivazione\|considerato\s+in\s+diritto\|in\s+diritto\b\|motivi\s+in\s+diritto\|ragioni\s+in\s+diritto` |
| `dispositivo` | `p\.?\s*q\.?\s*m\.?\|p\.?\s*t\.?\s*m\.?\|per\s+questi\s+motivi\|^[\s*#>_]*dispositivo[\s*#>_]*$\|così\s+deciso\|così\s+provvede` |
| `conclusioni` | `^\s*[*#>_ ]*conclusion[ei]\b\|\bconclusioni\s*:\|voglia\s+(?:l['’]ill\.?mo\s+)?(?:il\s+)?(?:tribunale\|giudice\|giudicante)\|^\s*[*#>_ ]*chiede\b\|salvis\s+iuribus` |

**`conclusioni` is a label of its own and must never be folded into
`dispositivo`.** In a ruling the `dispositivo` is what the judge decided; in a
filing the `conclusioni` are what a party *asked for*. Merging them would let a
lawyer's demand surface as if it were a holding — the most damaging mistake this
index can make. A template that wants them merged must say so explicitly by
giving both patterns the same label.

## Algorithm

Implement exactly this order; the overlap and packing rules are where a port
silently diverges.

**1. Normalise.** `text = text.replace("\r", "")`. Empty after strip → `chunks: []`.

**2. Hygiene** (`require_text`). Reject input that is not text: a ZIP/OOXML
header (`PK\x03\x04`, `[Content_Types].xml`), `%PDF-`, or control characters
outside `\t\n\f` above a small ratio → `400 not_text`. This guard exists because
the Digesta corpus contains chunks holding raw container bytes — measured at
0.59% of chunks with control characters and 0.04% raw OOXML, concentrated in PCT
party documents whose container was chunked instead of its extracted text. A
chunker that accepts those bytes embeds garbage and the index cannot tell.
`require_text: false` opts out deliberately.

**3. Sections.** Per the template above. `structural: false` → one unlabelled
section holding everything.

**4. Paragraphs, per section.** Split on `\n\s*\n`; strip; drop empties.

**5. Over-long paragraphs.** A paragraph longer than `target` is split on
sentence boundaries `(?<=[.;:!?])\s+`, greedily refilled to `target`; a single
sentence still longer than `target` is cut into fixed `target`-sized slices.
Nothing is ever discarded.

**6. Pack with a carried tail.** Within a section:

```python
chunks, cur = [], ""
for p in pieces:
    if cur and len(cur) + len(p) + 2 > target:
        chunks.append(cur)
        cur = (cur[-overlap:] + "\n\n" + p) if overlap else p
    else:
        cur = (cur + "\n\n" + p).strip() if cur else p
if cur:
    chunks.append(cur)
```

Three details that must be reproduced:

- Paragraphs are joined with `"\n\n"`, and the `+ 2` in the test accounts for it.
- The overlap is a **raw character tail** of the emitted chunk, `cur[-overlap:]`,
  prepended to the next one. It is not sentence- or token-aligned and may begin
  mid-word. That is the production behaviour; "improving" it changes every
  existing chunk boundary.
- `overlap == 0` starts the next chunk at `p` with no carry.

Consequence to expect in tests: a chunk may reach `target + overlap` characters
(3680 with the defaults). Observed max on the live corpus is 3682, mean 2158,
median 2464.

**7. Emit.** For each chunk, in document order: `ord`, `text`, `n_chars`,
`section` (`null` when the label is empty), `parent` (section index), and
`norm_refs` when an extractor is configured.

## Reference extraction

Optional, per request, and template-driven like the structure. `it-legal`
extracts article and act citations, normalises them, and de-duplicates while
**preserving first-seen order**:

- Articles → `art.<num> <code>`, e.g. `art.2043 c.c.`, where `num` keeps a
  `bis`/`ter`/`quater`/`quinquies`/`sexies` suffix with spaces removed, and the
  code normalises through `codciv|cc → c.c.`, `cpc → c.p.c.`, `cpp → c.p.p.`,
  `cp → c.p.`, `cost → cost.`
  The code is **required**: a bare `art. 360` is not emitted, because every
  procedural code has one and the field exists to be exact-matched — an
  unqualified number files three different articles under a single key.
  Only `art`/`artt` open a citation (not `articolo`), only `del` links it to
  its code (not `della`), the code may be spelled `cod. civ.`, and a suffix may
  be hyphenated (`art.2055-bis c.c.`) — the hyphen is kept, a space is not.
  An **enumeration** (`artt. 163, 164 e 167 c.p.c.`) resolves to its first
  article only: `art.163 c.p.c.`
- Acts → `<kind><num>/<year>`, e.g. `d.lgs.209/2005`, with
  `dlgs → d.lgs.`, `dl → d.l.`, `dpr → d.p.r.`, `legge|l → l.`
  The year is two to four digits and is kept **verbatim**: `d.p.r. 115/02`
  normalises to `d.p.r.115/02`, never to `/2002`, so the same act does not
  split across two keys depending on how the source wrote it. `del` separates
  as well as `/` (`legge 241 del 1990`), and the bare `l` needs its dot.

### Built-in `it-legal-strict`

`it-legal` widens the line-anchored headers to tolerate markdown emphasis, so
`**FATTO**` opens a section. The implementation this ports does not: its
`^\s*fatto\s*$` cannot match once the `**` is consumed, and such a document is
not sectioned at all. That is the **only** deliberate behavioural difference
between the two, and it is purely additive — no line that matched before stops
matching, and on an undecorated document the two templates agree byte for byte,
including sections, parents and refs.

`it-legal-strict` keeps the original anchors. Use it to re-chunk a corpus that
was indexed with the original, where a boundary that moves makes the new index
incomparable with the old. Anyone starting fresh wants `it-legal`.

These are a filter field and an exact-match signal for hybrid search — the thing
dense embeddings blur, since `art. 2043` and `art. 2054` sit close in vector
space and are entirely different claims. A template may replace the patterns;
`extract_refs: false` skips the work.

## Fidelity tests

The port is correct when, for the same input with `it-legal` and the defaults,
it matches Digesta's `chunk_document()` output exactly — same count, same
boundaries, same labels, same `parent`, same refs. Build the fixtures from the
live corpus rather than synthetic text, and include:

1. A ruling with `fatto`/`motivazione`/`dispositivo`.
2. A filing with `premessa`/`conclusioni`, **with the conclusions opened twice**
   (merge path).
3. A document with no recognisable header (flat fallback, `section: null`).
4. A single paragraph far over `target` (sentence split, then slice path).
5. A section boundary that lands mid-paragraph (the overlap carry).
6. A document whose header is markdown-decorated (`**FATTO**`, `> fatto`).
7. Binary input (OOXML) → `400 not_text`.
8. `overlap_tokens: 0` and `structural: false` → degenerate paths.

Assert on the `target + overlap` ceiling too: a chunk longer than that means the
packing test was ported wrong.

## Errors

| Status | Code | When |
|---|---|---|
| 400 | `not_text` | `require_text` and the input is binary |
| 400 | `empty_input` | no `text` and no `file`, or empty after normalisation |
| 400 | `bad_template` | inline template malformed, or a pattern fails to compile |
| 404 | `unknown_template` | named template not registered |
| 413 | `too_large` | input above the configured ceiling |

Patterns come from callers, so compile every template with a size limit and
treat a pathological pattern as `bad_template` rather than letting a scan run
unbounded on operator-supplied input.

## Notes for the Digesta side

Digesta keeps chunk text in Postgres (`chunks.text`, `store='clear'`, currently
54 GB) and vectors in Qdrant, which is what makes a re-chunk possible without
re-OCR. When it switches to this endpoint, the migration is only worthwhile if
output is identical — otherwise old and new chunks coexist in one collection with
different boundaries and retrieval quality becomes unmeasurable. So: port,
verify byte-identity on fixtures, then cut over; do not mix.
