"""Two model ids answered 500 for as long as they existed.

`/v1/models` advertised `geoclip` and `geoclip-location`; both answered

    POST /v1/embeddings model=geoclip          -> 500
    POST /v1/embeddings model=geoclip-location -> 500

in half a second, with no load line and nothing logged. The cause was
`No module named 'geoclip'`: the package is declared in requirements.txt, but
the incremental overlay never touches the venv, so an image updated that way
ships without it. Exactly the fault vosk had, for exactly the same reason, in
exactly the same file:

    # Vosk, the CPU STT backend. It is declared in requirements.txt, which the
    # FULL build picks up through the pre-built venv bundle -- but this overlay
    # never touches the venv […] while /v1/models advertised vosk-en and
    # vosk-it as available.

So the test is not "is it in requirements.txt" -- it was. It is "does the
overlay install it", which is the thing that was missing, and it was found by
listing the built image's packages rather than by reading the source.
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKERFILE = os.path.join(ROOT, 'packaging/linux/Dockerfile.update')
REQS = os.path.join(ROOT, 'requirements.txt')
EMB = os.path.join(ROOT, 'codai/api/embeddings.py')
APP = os.path.join(ROOT, 'codai/api/app.py')


def _overlay():
    return open(DOCKERFILE).read()


# ── the dependency reaches an incrementally-updated image ────────────────────

def test_the_overlay_installs_geoclip():
    assert re.search(r'pip install[^\n]*(\n[^\n]*)*?geoclip', _overlay()), (
        'the overlay does not install geoclip, so an image updated '
        'incrementally serves two model ids that cannot load')


def test_its_own_dependencies_come_with_it():
    """geopy and geographiclib are geoclip's; installing only geoclip moves the
    ModuleNotFoundError one import deeper."""
    src = _overlay()
    assert 'geopy' in src and 'geographiclib' in src


def test_it_goes_into_the_venv_not_the_user_site():
    """The venv is not writable by the container user, so a plain install lands
    in ~/.local and the engine -- which does not load the user site -- goes on
    reporting the module as missing while pip reports success. vosk already
    paid for this lesson."""
    src = _overlay()
    i = src.index('geoclip>=')
    block = src[max(0, i - 400):i]
    assert '--target /opt/coderai/python/lib/python3.13/site-packages' in block


def test_a_failed_install_does_not_break_an_offline_build():
    src = _overlay()
    i = src.index('geoclip>=')
    assert '|| true' in src[i:i + 200]


def test_it_is_still_declared_for_the_full_build():
    """The overlay is the second place it has to be, not the only one: the full
    build gets it from here, through the venv bundle."""
    reqs = open(REQS).read()
    assert 'geoclip>=' in reqs and 'geopy>=' in reqs and 'geographiclib>=' in reqs


def test_the_pod_profile_that_serves_embeddings_has_it_too():
    p = os.path.join(ROOT, 'packaging/runpod/profiles/embeddings.txt')
    assert 'geoclip' in open(p).read()


def test_one_missing_import_is_what_took_out_both_ids():
    """Both ids share _load_geoclip, which is why the sweep saw two failures
    and not one -- worth stating, because two independent 500s would have
    pointed somewhere else entirely."""
    src = open(EMB).read()
    i = src.index('def _load_geoclip')
    body = src[i:src.index('\ndef ', i + 10)]
    assert "mode = ('location'" in body, 'the two ids no longer share a loader'
    assert 'from geoclip' in body or 'import geoclip' in body


# ── and the next one of these is visible in the log ──────────────────────────

def test_a_500_is_logged_with_its_reason():
    """The detail existed and went only to the client. In a sweep that is a row
    in somebody else's report, so the cause had to be found by running the
    import against the image by hand."""
    src = open(APP).read()
    assert '@app.exception_handler(500)' in src, (
        '400s are logged with their reason and 500s are not, which is the '
        'wrong way round'
    )
    i = src.index('@app.exception_handler(500)')
    body = src[i:i + 2000]
    assert '[http] 500 on' in body
    assert 'detail' in body and 'request.url.path' in body


def test_it_answers_the_same_500_it_was_given():
    """A diagnostic that changes the response is not a diagnostic."""
    src = open(APP).read()
    i = src.index('@app.exception_handler(500)')
    body = src[i:i + 2000]
    assert 'status_code=500' in body


def test_the_400_handler_is_untouched():
    src = open(APP).read()
    assert '@app.exception_handler(400)' in src
    assert '[http] 400 on' in src
