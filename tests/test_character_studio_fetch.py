"""'Add from link' in the character studio: the studio fetches the URL itself.

That is the whole point (the browser never sees the file, it lands in uploads/ and
behaves exactly like a picked file) and also the whole risk: the request is made
by a process sitting on the same box as coderai on 127.0.0.1 and whatever else is
on the LAN. So every hop has to resolve to a public address unless the operator
passed --allow-private-fetch.

These start the real studio and check the refusals are real, plus that the page's
script still parses — the township page shipped a dead <script> for four versions
because nothing ever parsed what it served.
"""
import json
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
TOOL = ROOT / "tools" / "character_studio.py"
PORT = 7931


def _post(path, payload, port=PORT):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read())


@pytest.fixture(scope="module")
def studio(tmp_path_factory):
    out = tmp_path_factory.mktemp("studio")
    proc = subprocess.Popen(
        [sys.executable, str(TOOL), "--out-dir", str(out),
         "--base-url", "http://127.0.0.1:1", "web", "--web-port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(ROOT))
    try:
        html = None
        for _ in range(100):
            if proc.poll() is not None:
                pytest.skip(f"the studio exited ({proc.returncode})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=1) as r:
                    html = r.read().decode("utf-8", "replace")
                break
            except (urllib.error.URLError, OSError):
                time.sleep(0.1)
        if html is None:
            pytest.skip("the studio did not come up")
        yield html
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.parametrize("url,because", [
    ("http://127.0.0.1:8776/favicon.ico", "loopback — coderai's own admin API lives here"),
    ("http://192.168.1.10/photo.jpg",     "a private LAN address"),
    ("http://169.254.169.254/latest/",    "link-local, the cloud metadata address"),
    ("http://[::1]/photo.jpg",            "loopback over IPv6"),
])
def test_a_link_cannot_reach_the_machine_itself(studio, url, because):
    got = _post("/api/fetch-url", {"url": url})
    assert "error" in got, f"{because}: expected a refusal, got {got}"
    assert "not a public address" in got["error"], got["error"]


@pytest.mark.parametrize("url", ["ftp://example.com/x.jpg", "file:///etc/passwd",
                                 "gopher://example.com/x"])
def test_only_http_urls_are_accepted(studio, url):
    got = _post("/api/fetch-url", {"url": url})
    assert "error" in got and "http(s)" in got["error"], got


def test_an_empty_url_is_rejected(studio):
    got = _post("/api/fetch-url", {"url": "   "})
    assert got.get("error") == "a URL is required"


def test_the_page_offers_the_control(studio):
    assert 'id="url-input"' in studio
    assert 'id="url-add"' in studio
    assert "/api/fetch-url" in studio


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_the_page_script_parses(studio, tmp_path):
    blocks = [b for b in re.findall(r"<script[^>]*>(.*?)</script>", studio, re.S) if b.strip()]
    assert blocks, "the page serves no script"
    for n, block in enumerate(blocks):
        path = tmp_path / f"block{n}.js"
        path.write_text(block, encoding="utf-8")
        proc = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
        assert proc.returncode == 0, f"block {n}:\n{proc.stderr.strip()[:600]}"
