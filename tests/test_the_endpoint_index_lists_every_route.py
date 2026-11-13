"""The endpoint index must be the whole HTTP surface, not a remembered subset.

The index in ``CODERAI_API_DOCUMENTATION.md`` was hand-kept and drifted: 135 of
the 257 routes the server defines were missing from it, including whole
subsystems (``/v1/ocr``, ``/v1/chunk``, ``/v1/rerank``, ``/v1/moderation``,
speaker recognition, video composition, the client RunPod API). A documented
surface that silently omits half of itself is worse than none: a client reads it
as the list of what exists.

So the index is generated (``tools/gen_endpoint_index.py``) and these tests are
what keep it honest — they fail when a route is added to the code and not to the
doc, which is the exact drift that happened.
"""

import importlib.util
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC = ROOT / "CODERAI_API_DOCUMENTATION.md"
GEN = ROOT / "tools" / "gen_endpoint_index.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_endpoint_index", GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _index_rows():
    """Every ``| `METHOD` | `path` |`` row in the Endpoint Index section."""
    text = DOC.read_text()
    start = text.index("## Endpoint Index")
    section = text[start:]
    return {(m.group(1), m.group(2)) for m in re.finditer(
        r"^\| `(GET|POST|PUT|PATCH|DELETE)` \| `([^`]+)` \|", section, re.M)}


def test_every_route_in_the_code_appears_in_the_index():
    gen = _load_generator()
    routes = {(method, path) for method, path in gen.collect()}
    documented = _index_rows()
    missing = sorted(f"{m} {p}" for m, p in routes - documented)
    assert not missing, (
        f"{len(missing)} route(s) exist in codai/ but not in the endpoint index. "
        f"Run `python3 tools/gen_endpoint_index.py --write`: {missing[:20]}")


def test_the_index_invents_no_routes():
    """A stale row is the other half of the same failure: a client calls a path
    the server stopped serving and reads the 404 as an outage."""
    gen = _load_generator()
    routes = {(method, path) for method, path in gen.collect()}
    stale = sorted(f"{m} {p}" for m, p in _index_rows() - routes)
    assert not stale, (
        f"the endpoint index lists {len(stale)} route(s) the code does not "
        f"define: {stale[:20]}")


def test_the_checked_in_index_is_what_the_generator_produces():
    """`--check` is the one a release can run; this proves it passes on master,
    so a failure means the doc, not the checker."""
    proc = subprocess.run([sys.executable, str(GEN), "--check"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_a_new_route_would_be_caught():
    """The test above only means something if adding a route breaks it. Prove
    that by generating an index from a route set with one extra entry."""
    gen = _load_generator()
    routes = gen.collect()
    routes[("POST", "/v1/a-route-nobody-documented")] = {
        "summary": "", "sources": ["codai/api/app.py"]}
    section = gen.render(routes)
    assert "/v1/a-route-nobody-documented" in section
    assert section != DOC.read_text()[DOC.read_text().index("## Endpoint Index"):]


def test_every_route_carries_a_purpose():
    """A route with neither a `summary=` nor an OVERRIDES line renders as an
    em dash, which is a row that tells a reader nothing."""
    gen = _load_generator()
    undescribed = sorted(
        f"{method} {path}" for (method, path), entry in gen.collect().items()
        if not entry["summary"] and f"{method} {path}" not in gen.OVERRIDES)
    assert not undescribed, (
        f"{len(undescribed)} route(s) have no summary and no OVERRIDES entry: "
        f"{undescribed[:20]}")


def test_the_subsystems_that_were_missing_are_documented_in_prose():
    """Not just listed in the index: a reader needs the fields and the shape.

    These are the families that were absent from the document entirely, each
    pinned to the heading that now covers it.
    """
    text = DOC.read_text()
    for heading in ("## Reranking", "## Retrieval Chunking", "## Document OCR",
                    "## Moderation", "## Speaker Recognition and Diarization",
                    "## Video Composition", "## Model and Weight Transfer",
                    "## Test Runs", "## Remote GPUs: the Client API",
                    "## Service and Operational Endpoints",
                    "## Cluster Endpoints", "## Internal Endpoints"):
        assert heading in text, f"{heading!r} is missing from {DOC.name}"


def test_the_doc_links_resolve():
    """A relative link in these docs is a promise about a path in the repo."""
    import os
    broken = []
    for doc in (DOC, ROOT / "README.md", ROOT / "docs" / "README.md"):
        base = doc.parent
        for m in re.finditer(r"\]\(([^)#]+?)(?:#[^)]*)?\)", doc.read_text()):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (base / target).exists():
                broken.append(f"{doc.name} -> {target}")
    assert not broken, broken


def test_a_decorator_whose_path_is_on_the_next_line_is_still_found(tmp_path):
    """The scanner reads the decorator's first argument. If it only looked at
    the same line, a route registered as

        @router.post(
            "/v1/thing", summary="…")

    would be invisible to BOTH the index and the tests above — a silent omission
    is exactly the drift this generator exists to prevent. No route in the tree
    is written that way today, which is why this has to be tested deliberately
    rather than noticed.
    """
    gen = _load_generator()
    module = tmp_path / "codai" / "api"
    module.mkdir(parents=True)
    (module / "probe.py").write_text(
        'from fastapi import APIRouter\n'
        'router = APIRouter()\n\n'
        '@router.post(\n'
        '    "/v1/on-the-next-line",\n'
        '    summary="Found anyway",\n'
        ')\n'
        'async def probe():\n'
        '    return {}\n')
    original = gen.PKG
    try:
        gen.PKG = tmp_path / "codai"
        gen.ROOT = tmp_path
        found = gen.collect()
    finally:
        gen.PKG = original
    assert ("POST", "/v1/on-the-next-line") in found
    assert found[("POST", "/v1/on-the-next-line")]["summary"] == "Found anyway"
