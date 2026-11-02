# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""How long a page may spend on a pod has to be reachable from the config.

Measured on a production orchestrator 2026-11-02, a 512-page run at 128 lanes
against a single pod:

    [ocr] 1 of 1 page(s) could not be sent to any pod for
          'datalab-to/chandra-ocr-2' after 3 attempts each (pod OCR at
          https://.../v1/ocr is unreachable: Read timed out.
          (read timeout=600))
    [http] 503 on POST /v1/ocr: "no pod could read this document ... and
           ocr.pod_pipeline is 'require', so this box will not read it itself."

448 pages ended that way. The pod had refused nothing -- no admission line in
its whole life -- and answered 200 to everything it finished, so those were
pages waiting behind a service time longer than the timeout: work paid for and
discarded.

A deep queue is NOT a misconfiguration here; `ocr.max_concurrency` is wide on
purpose when the point of a run is to measure the fleet's peak rate rather
than this process's throttling. But the queue depth and the per-page timeout
have to be chosen together, and only one of the two was reachable: the timeout
was the literal 600 in two call sites.

So: the number comes from the config, the default is unchanged for every
install that says nothing, and both transports honour it -- the loop one and
the thread one, which passes it POSITIONALLY through `run_on_pod_pool` and is
therefore the one a later signature change breaks silently.
"""

import asyncio
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.config import OcrConfig
from codai.ocr import manager as M


class _Cfg:
    """Only what the reader looks at."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _fake_pod_http(monkeypatch, fake):
    """Stand in for `codai.api.pod_http` on BOTH lookup routes.

    The forwards do `from codai.api import pod_http`, which resolves the
    ATTRIBUTE on the already-imported `codai.api` package and only falls back
    to ``sys.modules``. So patching ``sys.modules`` alone works when this file
    runs on its own and is silently bypassed in the full suite, where an
    earlier test has already imported the real module -- at which point these
    tests dial a real address instead of testing anything. Measured: three
    passes alone, three failures in the suite.
    """
    import codai.api
    monkeypatch.setattr(codai.api, "pod_http", fake, raising=False)
    monkeypatch.setitem(sys.modules, "codai.api.pod_http", fake)


# ---------------------------------------------------------------------------
# the setting
# ---------------------------------------------------------------------------

def test_the_config_declares_the_field():
    """An undeclared key is DROPPED on load: the dataclass is the schema, so
    `ocr.pod_page_timeout_s: 1800` in a config file would have been read,
    discarded, and the install would have kept waiting 600 s with no sign."""
    assert hasattr(OcrConfig(), "pod_page_timeout_s")


def test_the_default_is_the_behaviour_every_install_had():
    assert OcrConfig().pod_page_timeout_s == 600.0
    assert M.POD_PAGE_TIMEOUT_S == 600.0
    assert M._pod_page_timeout_s(OcrConfig()) == 600.0


def test_a_configured_value_is_used():
    assert M._pod_page_timeout_s(_Cfg(pod_page_timeout_s=1800)) == 1800.0


@pytest.mark.parametrize("bad", [0, 0.0, None, "", -1, -600, "nonsense", object()])
def test_an_unusable_value_falls_back_to_the_default(bad):
    """A timeout of 0 means "no timeout" to some HTTP clients and "expire at
    once" to others. Neither is what an operator typing 0 wants, so 0 and
    anything unreadable mean "use the default" -- stated here because the
    alternative is a config that can hang every page or fail every page."""
    assert M._pod_page_timeout_s(_Cfg(pod_page_timeout_s=bad)) == 600.0


def test_a_config_object_without_the_attribute_at_all_still_works():
    """Config objects are also built by pods from a seed, so the reader must
    not require the attribute to exist."""
    assert M._pod_page_timeout_s(object()) == 600.0


def test_it_is_exposed_where_the_other_pod_settings_are():
    """Reachable from the settings surface, not only by editing a file: the
    whole failure was a number nobody could reach."""
    import codai.config as C
    src = inspect.getsource(C)
    assert '"pod_page_timeout_s"' in src


# ---------------------------------------------------------------------------
# both transports have to honour it
# ---------------------------------------------------------------------------

def test_both_forwards_accept_the_timeout():
    for fn in (M._apost_page_to_pod, M._post_page_to_pod):
        params = inspect.signature(fn).parameters
        assert "timeout_s" in params, fn.__name__
        assert params["timeout_s"].default == M.POD_PAGE_TIMEOUT_S


def test_the_loop_transport_passes_it_to_the_request(monkeypatch):
    seen = {}

    class _FakePodHttp:
        @staticmethod
        async def apost(url, **kw):
            seen.update(kw)
            raise RuntimeError("stop here: the request is what was under test")

    _fake_pod_http(monkeypatch, _FakePodHttp)
    monkeypatch.setattr(M, "POD_PAGE_ATTEMPTS", 1)
    with pytest.raises(M.PodUnreachable):
        asyncio.run(M._apost_page_to_pod(
            "https://pod/v1", "k", b"%PDF-", "1.pdf", 200, {}, None,
            "chandra", timeout_s=1800))
    assert seen.get("timeout") == 1800


def test_the_thread_transport_passes_it_to_the_request(monkeypatch):
    seen = {}

    class _FakePodHttp:
        @staticmethod
        def post(url, **kw):
            seen.update(kw)
            raise RuntimeError("stop here: the request is what was under test")

    _fake_pod_http(monkeypatch, _FakePodHttp)
    monkeypatch.setattr(M, "POD_PAGE_ATTEMPTS", 1)
    with pytest.raises(M.PodUnreachable):
        M._post_page_to_pod("https://pod/v1", "k", b"%PDF-", "1.pdf", 200, {},
                            None, "chandra", timeout_s=1800)
    assert seen.get("timeout") == 1800


def test_the_default_still_reaches_the_request_when_nobody_passes_one(monkeypatch):
    """The two forwards are also called directly elsewhere; the default must
    be the old literal, not None."""
    seen = {}

    class _FakePodHttp:
        @staticmethod
        def post(url, **kw):
            seen.update(kw)
            raise RuntimeError("stop")

    _fake_pod_http(monkeypatch, _FakePodHttp)
    monkeypatch.setattr(M, "POD_PAGE_ATTEMPTS", 1)
    with pytest.raises(M.PodUnreachable):
        M._post_page_to_pod("https://pod/v1", "k", b"%PDF-", "1.pdf", 200, {},
                            None, "chandra")
    assert seen.get("timeout") == 600.0


def test_the_thread_path_passes_the_timeout_in_the_right_position():
    """The dispatcher hands `_post_page_to_pod` its arguments POSITIONALLY to
    `run_on_pod_pool`, so the timeout rides in the slot after `engine`. If a
    parameter is ever inserted before it, the timeout silently becomes the
    engine name -- or the engine name becomes a float -- and no test of the
    keyword form would notice."""
    names = list(inspect.signature(M._post_page_to_pod).parameters)
    assert names.index("timeout_s") == names.index("engine") + 1
    # and the dispatcher really does pass it positionally there
    src = inspect.getsource(M)
    assert "detect, engine_name,\n                            page_timeout)" in src
