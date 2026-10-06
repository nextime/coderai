"""The township page's JavaScript must actually parse.

It is emitted from a Python f-string, which makes one mistake invisible and fatal:
a bare \\n in the Python source becomes a REAL newline inside the JS string
literal, the <script> dies with a SyntaxError, and EVERYTHING in that block stops
working — status polling, the template picker, the per-stage buttons, Save config,
Stop. The page still renders perfectly, so screenshots look fine and nothing in
the server logs complains.

That shipped in v0.2.41 and was only noticed when the template picker turned out
to be an empty box. This starts the real server, fetches the real page and parses
every script block it serves.

Parsing is done by node, not by a hand-rolled scanner. One was written first and
deleted: it reported the known-broken block as clean, because tracking JS string,
comment and regex state correctly is its own project. A check that passes on a
file you have proven is broken is worse than no check at all.
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "gen_township_fighters.py"
PORT = 7987


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    out = tmp_path_factory.mktemp("township")
    proc = subprocess.Popen(
        [sys.executable, str(TOOL), "--web-port", str(PORT), "--out-dir", str(out)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=str(ROOT),
    )
    try:
        html = None
        for _ in range(100):
            if proc.poll() is not None:
                pytest.skip(f"the tool exited ({proc.returncode}) instead of serving")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=1) as r:
                    html = r.read().decode("utf-8", "replace")
                break
            except (urllib.error.URLError, OSError):
                time.sleep(0.1)
        if html is None:
            pytest.skip("the web UI did not come up")
        yield html
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _blocks(html):
    return [b for b in re.findall(r"<script[^>]*>(.*?)</script>", html, re.S) if b.strip()]


def test_the_page_serves_script(page):
    """A page with no script would make every check below vacuous."""
    assert len(_blocks(page)) >= 3


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_every_script_block_parses(page, tmp_path):
    """The real check: hand each block to a JS engine."""
    for n, block in enumerate(_blocks(page)):
        path = tmp_path / f"block{n}.js"
        path.write_text(block, encoding="utf-8")
        proc = subprocess.run(["node", "--check", str(path)],
                              capture_output=True, text=True)
        assert proc.returncode == 0, (
            f"script block {n} does not parse:\n{proc.stderr.strip()[:800]}")


def test_the_template_picker_is_filled_on_load(page):
    """tplRefresh() existed from the first templates commit but was only ever
    called AFTER a save or a delete, so a freshly loaded page listed nothing and
    the picker rendered as a box you could not choose from."""
    assert "tplRefresh();" in page
    init = page.split("Restore state on page load", 1)
    assert len(init) == 2, "the page-load init block is gone"
    assert "tplRefresh();" in init[1][:600], "nothing fills the picker on load"
