"""`response_format: "b64_json"` must actually return the bytes.

The request model documents OpenAI's values -- `url` and `b64_json` -- and
every OpenAI SDK sends the latter. `save_image_response` compared against this
codebase's own older internal spelling, `base64`, so when `--file-path` is
configured a client asking for the bytes the standard way got a URL and no
bytes, with nothing in the response to say why. Found by generating a real
image against the live server and getting `{url, path}` back from a request
that asked for b64_json.
"""
import base64
import io

import pytest
from PIL import Image

import codai.api.images as I


@pytest.fixture
def img():
    return Image.new("RGB", (8, 8), (10, 120, 200))


@pytest.mark.parametrize("fmt", ["b64_json", "base64", "B64_JSON", " b64_json "])
def test_every_spelling_of_base64_is_honoured(fmt):
    assert I._wants_b64(fmt) is True


@pytest.mark.parametrize("fmt", ["url", "URL", None, "", "png"])
def test_url_and_nonsense_are_not_base64(fmt):
    assert I._wants_b64(fmt) is False


def test_with_a_file_path_b64_json_returns_both(monkeypatch, tmp_path, img):
    monkeypatch.setattr(I, "global_file_path", str(tmp_path))
    out = I.save_image_response(img, "b64_json")
    assert "url" in out and "path" in out
    assert "b64_json" in out, "the bytes the caller explicitly asked for"
    # And it is a real PNG, not an empty string.
    raw = base64.b64decode(out["b64_json"])
    assert Image.open(io.BytesIO(raw)).size == (8, 8)


def test_with_a_file_path_url_returns_no_bytes(monkeypatch, tmp_path, img):
    # Unchanged: the point of --file-path is not to ship megabytes inline.
    monkeypatch.setattr(I, "global_file_path", str(tmp_path))
    out = I.save_image_response(img, "url")
    assert "url" in out and "path" in out
    assert "b64_json" not in out


def test_without_a_file_path_the_bytes_come_back_either_way(monkeypatch, img):
    monkeypatch.setattr(I, "global_file_path", None)
    for fmt in ("url", "b64_json"):
        out = I.save_image_response(img, fmt)
        assert "b64_json" in out, fmt
        assert "url" not in out


def test_the_path_is_relative_so_a_caller_need_not_know_the_hostname(
        monkeypatch, tmp_path, img):
    monkeypatch.setattr(I, "global_file_path", str(tmp_path))
    out = I.save_image_response(img, "b64_json")
    assert out["path"].startswith("/v1/files/")


def test_the_file_is_actually_written(monkeypatch, tmp_path, img):
    import os
    monkeypatch.setattr(I, "global_file_path", str(tmp_path))
    out = I.save_image_response(img, "url")
    name = out["path"].rsplit("/", 1)[-1]
    p = os.path.join(str(tmp_path), name)
    assert os.path.isfile(p)
    assert Image.open(p).size == (8, 8)
