"""Four sentences to the image half of geoclip answered 500, twice, unlogged.

Reproduced against the running server on 2026-10-30:

    geoclip + 4 texts      : HTTP 500  {"detail":"Embedding failed: cannot
                             identify image file <_io.BytesIO object at ...>"}
    geoclip-location + text: HTTP 400  {"detail":"geoclip-location expects
                             'lat,lon' in decimal degrees, got 'una frase'"}

Three separate faults, in the order they bite:

1. The refusal came AFTER a 1.9 GB load. geoclip's image half takes its image
   in ``input`` -- the same contract as vpr/salad -- so a text ``input`` is
   answerable from the model NAME alone, before anything is loaded.

2. It was a 500 rather than a 400. ``"una frase"`` loses its space, and
   ``"unafrase"`` is eight characters of the base64 alphabet: it decodes to six
   bytes, and only PIL can tell it is not a picture. PIL raises
   ``UnidentifiedImageError``, which is NOT a ValueError, so it escaped the
   caller's ``except ValueError -> 400`` and became "the server broke" for what
   is bad input.

3. Nothing was logged. The detail existed and went only to the client. Finding
   it meant replaying the request by hand.
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.api import embeddings as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = io.open(os.path.join(ROOT, 'codai/api/embeddings.py'), encoding='utf-8').read()
APP = io.open(os.path.join(ROOT, 'codai/api/app.py'), encoding='utf-8').read()


# ── 1. refused before the load ───────────────────────────────────────────────

def test_the_image_half_of_geoclip_refuses_text_before_loading():
    assert E._takes_image_in_input('geoclip') is True
    assert E._takes_image_in_input('geoclip-image') is True
    assert E._takes_image_in_input('geoclip-img') is True


def test_the_location_half_is_not_treated_as_image_only():
    """It takes COORDINATES in `input` and already refuses prose with a 400 that
    names the format. Folding it in here would refuse the requests it exists to
    serve."""
    assert E._takes_image_in_input('geoclip-location') is False
    assert E._takes_image_in_input('geoclip-loc') is False
    assert E._takes_image_in_input('geoclip-gps') is False


def test_the_other_image_in_input_models_are_untouched():
    for n in ('salad', 'dinov2-salad', 'vpr', 'eigenplaces'):
        assert E._takes_image_in_input(n) is True, n
    for n in ('Qwen3-Embedding-4B', 'dinov2-large', 'bge-m3', None):
        assert E._takes_image_in_input(n) is False, n


def test_it_reads_the_existing_id_set():
    """A second hand-kept list of geoclip aliases would drift from the one the
    loader dispatches on."""
    i = SRC.index('def _takes_image_in_input')
    body = SRC[i:SRC.index('def _looks_like_image_ref', i)]
    assert '_GEOCLIP_IDS_IMAGE' in body


def test_geoclip_still_accepts_images():
    """The fix refuses text, not pictures."""
    assert E._may_embed_images('geoclip', {}) is True


# ── 2. a non-image is bad input, not a server fault ──────────────────────────

@pytest.mark.parametrize('text', [
    'una frase',          # the exact input from the incident
    'unafrase',           # what it decodes as: valid base64, six bytes
    'aGVsbG8=',           # "hello"
    'QUJDRA==',
])
def test_a_base64_decodable_non_image_raises_valueerror(text):
    """ValueError is what the call site maps to 400. Anything else is a 500."""
    with pytest.raises(ValueError):
        E._decode_image(text)


def test_the_message_quotes_what_arrived_and_says_what_is_accepted():
    try:
        E._decode_image('una frase')
    except ValueError as e:
        msg = str(e)
    assert 'una frase' in msg
    assert 'not an image' in msg
    assert 'data URI' in msg and 'base64' in msg


def test_a_real_image_still_decodes():
    import base64
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', (8, 8), (10, 20, 30)).save(buf, 'PNG')
    raw = base64.b64encode(buf.getvalue()).decode()
    assert E._decode_image(raw).size == (8, 8)
    assert E._decode_image('data:image/png;base64,' + raw).size == (8, 8)


def test_an_undecodable_string_still_raises_valueerror():
    """The pre-existing branch; it must not regress while the new one is added."""
    with pytest.raises(ValueError):
        E._decode_image('not base64 at all !!!')


def test_pil_is_no_longer_allowed_to_escape_as_a_500():
    i = SRC.index('def _decode_image')
    body = SRC[i:SRC.index('def _truncate_dims', i)]
    tail = body[body.index('validate=True'):]
    assert 'Image.open' in tail
    assert 'raise ValueError' in tail, (
        'Image.open can still raise UnidentifiedImageError, which is not a '
        'ValueError and so becomes a 500 for what is bad input')


# ── 3. a 5xx says why, in the log ────────────────────────────────────────────

def test_the_handler_is_registered_on_the_exception_class():
    """Starlette moves a handler registered for `500` (or `Exception`) out of
    the status table and into ServerErrorMiddleware, which fires only for an
    UNHANDLED exception. `raise HTTPException(status_code=500, ...)` -- how
    nearly every 500 here is raised -- went straight past it and logged
    nothing. Verified live: two geoclip 500s, no `[http] 500` line."""
    assert '@app.exception_handler(StarletteHTTPException)' in APP
    assert '@app.exception_handler(500)' not in APP, (
        'the status-500 registration is back, and it does not catch '
        'HTTPException(500)')


def test_the_starlette_class_is_imported_not_the_fastapi_one():
    """Starlette raises its own HTTPException (404 on an unknown route), and the
    FastAPI class is a subclass of it -- registering on the Starlette class
    covers both."""
    assert 'from starlette.exceptions import HTTPException as StarletteHTTPException' in APP


def test_only_5xx_is_logged():
    i = APP.index('@app.exception_handler(StarletteHTTPException)')
    body = APP[i:i + 2600]
    assert 'if status >= 500:' in body, 'every 404 would now print a line'
    assert '[http] ' in body


def test_the_status_and_detail_are_passed_through_unchanged():
    i = APP.index('@app.exception_handler(StarletteHTTPException)')
    body = APP[i:i + 2600]
    assert 'status_code=status' in body
    assert 'headers=headers' in body, (
        'a 401 WWW-Authenticate or a 429 Retry-After would be dropped')


def test_the_400_handler_keeps_its_own_line():
    """A status-keyed handler wins over a class-keyed one, so the richer 400
    message (content-type, length, peer) survives."""
    assert '@app.exception_handler(400)' in APP
    assert '[http] 400 on' in APP


def test_a_500_is_actually_logged_and_a_400_still_takes_its_own_path():
    """The registration semantics are the whole fix, so they are exercised
    rather than asserted: the first version of this handler passed every
    source-level check and caught nothing."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from starlette.exceptions import HTTPException as SHE
    from starlette.testclient import TestClient

    seen = []
    app = FastAPI()

    @app.exception_handler(400)
    async def four(request: Request, exc: HTTPException):
        seen.append(('400', exc.detail))
        return JSONResponse(status_code=400, content={'detail': exc.detail})

    @app.exception_handler(SHE)
    async def cls(request: Request, exc: SHE):
        if exc.status_code >= 500:
            seen.append(('5xx', exc.detail))
        return JSONResponse(status_code=exc.status_code,
                            content={'detail': exc.detail})

    @app.get('/e500')
    def e500(): raise HTTPException(status_code=500, detail='Embedding failed: x')

    @app.get('/e400')
    def e400(): raise HTTPException(status_code=400, detail='bad')

    c = TestClient(app, raise_server_exceptions=False)
    assert c.get('/e500').status_code == 500
    assert c.get('/e400').status_code == 400
    assert ('5xx', 'Embedding failed: x') in seen
    assert ('400', 'bad') in seen
