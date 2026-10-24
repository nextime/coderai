"""Uploading media to an install that configured no output directory.

`POST /v1/files/upload` exists so a remote client sends a file once and then
names it: inlining a 40 MB clip in every request is the waste the endpoint was
built to remove. But it stored its content-addressed files in the output
directory, and refused with 501 when there was none configured -- so on such an
install the only way to send media was the inlining it was meant to replace,
and a client that had already switched to uploading saw its photo edits fail
outright.

An install always has a cache directory, so uploads fall back to it. What has
to keep working from the caller's side, and is asserted here: the upload is
accepted, the `sha256:` id resolves, the `/v1/files/<name>` URL serves the
bytes back, and a generated file is NOT served out of the upload store.
"""
import hashlib
import io
import os

import pytest


def _png_bytes(w=6, h=6, colour=(3, 140, 90)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def no_output_dir(tmp_path, monkeypatch):
    """An install with no --file-path, whose cache directory is ``tmp_path``."""
    from codai.compose import media
    from codai import platform_paths
    monkeypatch.setattr(media, "global_file_path", None, raising=False)
    for mod in ("codai.api.app", "codai.api.video", "codai.api.images"):
        m = __import__(mod, fromlist=["x"])
        monkeypatch.setattr(m, "global_file_path", None, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(platform_paths, "user_cache_dir",
                        lambda: tmp_path / "coderai", raising=False)
    return tmp_path / "coderai" / "uploads"


def test_uploads_dir_falls_back_to_the_cache_dir(no_output_dir):
    from codai.compose import media
    assert media.files_dir() is None
    assert media.uploads_dir() == str(no_output_dir)


def test_uploads_dir_is_the_output_dir_when_there_is_one(tmp_path, monkeypatch):
    """Nothing moves on an install that is already configured."""
    from codai.compose import media
    monkeypatch.setattr(media, "global_file_path", str(tmp_path), raising=False)
    assert media.uploads_dir() == str(tmp_path)


def test_store_accepts_the_upload_instead_of_501(no_output_dir):
    """The defect, from the client's side: this used to raise 501."""
    from codai.api import uploads
    data = _png_bytes()
    doc = uploads.store(data, "shot.png", "image/png")
    h = hashlib.sha256(data).hexdigest()
    assert doc["id"] == f"sha256:{h}"
    assert doc["existed"] is False
    assert (no_output_dir / doc["name"]).read_bytes() == data


def test_the_sha256_id_resolves_back(no_output_dir):
    from codai.api import uploads
    from codai.compose import media
    data = _png_bytes()
    doc = uploads.store(data)
    assert media.fetch_bytes(doc["id"]) == data
    assert media.blob_path(doc["id"].split(":", 1)[1]) is not None


def test_an_image_endpoint_takes_the_uploaded_id(no_output_dir):
    """What the failing photo edit actually needed."""
    from codai.api import images, uploads
    data = _png_bytes()
    doc = uploads.store(data)
    assert images._image_bytes(doc["id"]) == data


def test_the_files_url_resolves_back(no_output_dir):
    from codai.api import uploads
    from codai.compose import media
    data = _png_bytes()
    doc = uploads.store(data)
    assert media.fetch_bytes(f"/v1/files/{doc['name']}") == data


def test_re_uploading_the_same_bytes_is_free(no_output_dir):
    from codai.api import uploads
    data = _png_bytes()
    uploads.store(data)
    again = uploads.store(data)
    assert again["existed"] is True


def test_an_upload_is_served_back_over_http(no_output_dir):
    """/v1/files/<name> is how a client fetches its own file; accepting an
    upload we then cannot serve would be a different bug."""
    from fastapi import HTTPException
    from codai.api import app as app_mod
    from codai.api import uploads
    import asyncio
    data = _png_bytes()
    doc = uploads.store(data)
    resp = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        app_mod.get_file(doc["name"]))
    assert open(resp.path, "rb").read() == data


def test_a_generated_name_is_not_served_from_the_upload_store(no_output_dir):
    """Only ``up-`` names fall back; the cache dir is not a second output dir."""
    import asyncio
    from fastapi import HTTPException
    from codai.api import app as app_mod
    os.makedirs(no_output_dir, exist_ok=True)
    (no_output_dir / "secret.png").write_bytes(_png_bytes())
    loop = asyncio.get_event_loop_policy().new_event_loop()
    with pytest.raises(HTTPException) as exc:
        loop.run_until_complete(app_mod.get_file("secret.png"))
    assert exc.value.status_code == 404


def test_traversal_out_of_the_upload_store_is_refused(no_output_dir):
    from codai.compose import media
    from codai.api import uploads
    uploads.store(_png_bytes())
    assert media.local_path_for_files_url("/v1/files/up-../../etc/passwd") is None


def test_an_upload_stored_before_a_dir_was_configured_still_resolves(
        no_output_dir, tmp_path, monkeypatch):
    """An install that gains --file-path later does not lose what it holds."""
    from codai.api import uploads
    from codai.compose import media
    data = _png_bytes()
    doc = uploads.store(data)
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setattr(media, "global_file_path", str(out), raising=False)
    assert media.fetch_bytes(doc["id"]) == data
