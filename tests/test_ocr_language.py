"""`lang=` was accepted on /v1/ocr and marked "reserved" -- it did nothing.

Now a client names the document's language and we respect it, and when it names
none the default is AUTO-RECOGNITION rather than any particular language. That
asymmetry is the whole design:

* surya-2, olmOCR and docTR are multilingual and have NO language input. Surya-2's
  entire prompt is "OCR this block image to HTML." and its own module warns the
  wording is the training-time contract, so there is nowhere to put a language and
  nothing to gain. For them auto is simply what they already do.
* PaddleOCR loads a language-specific recognition model, and the options are sent
  once at load -- an instance cannot be re-languaged. So it is the only engine
  that needs a pool per language, and the only one with something to translate
  `auto` into.

Keeping that set small is not tidiness: a language-keyed pool is a SECOND set of
loaded instances holding VRAM, and a production install runs 24 surya instances.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.ocr import manager as om                                # noqa: E402


class _Cfg:
    def __init__(self, **kw):
        self.lang = "auto"
        self.paddle_lang = ""
        self.paddle_auto_lang = ""
        for k, v in kw.items():
            setattr(self, k, v)


# ------------------------------------------------------------ what auto means #
def test_nothing_specified_means_auto():
    assert om.resolve_lang(_Cfg(), "surya", None) == "auto"
    assert om.DEFAULT_LANG == "auto"


def test_the_ways_of_saying_auto_all_mean_auto():
    for word in ("", "auto", "AUTO", " any ", "multi", "multilingual", "detect"):
        assert om.normalize_lang(word) == "auto", word


def test_a_named_language_is_normalised_not_mangled():
    assert om.normalize_lang("  EN ") == "en"
    assert om.normalize_lang("pt_BR") == "pt-br"


# ------------------------------------------------------------- precedence #
def test_the_request_wins_over_every_configured_language():
    """A caller naming a language knows something about the document in hand;
    a configured one is only an install's default."""
    cfg = _Cfg(lang="it", paddle_lang="french")
    assert om.resolve_lang(cfg, "paddle", "japan") == "japan"


def test_the_engine_specific_setting_beats_the_global_one():
    cfg = _Cfg(lang="it", paddle_lang="french")
    assert om.resolve_lang(cfg, "paddle", None) == "french"


def test_the_global_setting_is_used_when_there_is_nothing_else():
    assert om.resolve_lang(_Cfg(lang="it"), "surya", None) == "it"


def test_asking_for_auto_explicitly_is_not_the_same_as_asking_for_nothing():
    """An install configured for Italian must still be able to serve a request
    that says "work it out yourself"."""
    cfg = _Cfg(lang="it")
    assert om.resolve_lang(cfg, "surya", None) == "it"
    assert om.resolve_lang(cfg, "surya", "auto") == "auto"


# --------------------------------------------- which engines can act on it #
def test_only_paddle_reads_by_language():
    assert om.engine_uses_language("paddle")
    for name in ("surya", "olmocr", "doctr"):
        assert not om.engine_uses_language(name), name


def test_auto_becomes_a_script_family_for_paddle_which_has_no_auto_mode():
    assert om.engine_lang(_Cfg(), "paddle", "auto") == "latin"
    assert om.AUTO_PADDLE_LANG == "latin"


def test_the_paddle_auto_language_is_configurable():
    assert om.engine_lang(_Cfg(paddle_auto_lang="arabic"), "paddle", "auto") == "arabic"


def test_a_named_language_is_passed_to_paddle_untouched():
    assert om.engine_lang(_Cfg(), "paddle", "japan") == "japan"


def test_auto_stays_auto_for_an_engine_that_needs_no_translation():
    assert om.engine_lang(_Cfg(), "surya", "auto") == "auto"


# ------------------------------------------------------------ pool identity #
def _pools_for(monkeypatch, cfg, asks):
    """Resolve each (engine, lang) ask against a fresh manager; return the pool
    objects and the keys they landed under."""
    import asyncio

    class _StubPool:
        def __init__(self, name, cfg, size):
            self.name, self.cfg, self.size = name, cfg, size

    monkeypatch.setattr(om, "_Pool", _StubPool)
    monkeypatch.setattr(om, "_engine_instances", lambda c, n: 1)
    m = om.OcrManager()
    m._require_cfg = lambda: cfg
    m._pools = {}
    m._lock = asyncio.Lock()

    async def _run():
        return [await m._get_pool(n, lg) for (n, lg) in asks]
    got = asyncio.run(_run())
    return got, set(m._pools)


def test_two_languages_are_two_paddle_pools(monkeypatch):
    got, keys = _pools_for(monkeypatch, _Cfg(),
                           [("paddle", "it"), ("paddle", "japan")])
    assert got[0] is not got[1]
    assert keys == {"paddle:it", "paddle:japan"}


def test_auto_and_the_language_it_maps_to_share_one_pool(monkeypatch):
    """They are the same loaded model; two pools would be two copies in VRAM."""
    got, keys = _pools_for(monkeypatch, _Cfg(),
                           [("paddle", "auto"), ("paddle", "latin")])
    assert got[0] is got[1]
    assert keys == {"paddle:latin"}


def test_a_multilingual_engine_keeps_one_pool_whatever_is_asked(monkeypatch):
    """24 surya instances must not become 24 per language for a parameter the
    engine cannot use."""
    got, keys = _pools_for(monkeypatch, _Cfg(),
                           [("surya", "it"), ("surya", "japan")])
    assert got[0] is got[1]
    assert keys == {"surya"}


def test_a_paddle_pool_is_built_with_the_language_resolved_into_its_config(monkeypatch):
    from codai.config import OcrConfig
    got, _ = _pools_for(monkeypatch, OcrConfig(), [("paddle", "japan")])
    assert got[0].cfg.lang == "japan"


# ------------------------------------------- the config handed to an engine #
def test_the_resolved_language_is_written_into_the_config_the_engine_reads():
    """The pool builds engines with factory(cfg) and each engine reads its own
    language out of the config, so resolving it here means no engine has to know
    a per-request language exists."""
    import dataclasses
    from codai.config import OcrConfig
    cfg = OcrConfig(lang="it", paddle_lang="french")
    out = om.OcrManager._cfg_for(cfg, "paddle", "japan")
    assert out.lang == "japan"
    # the request has to beat paddle_lang too, or the override would be ignored
    assert out.paddle_lang == "japan"
    assert dataclasses.is_dataclass(out)
    # and the original is untouched -- other pools are still serving from it
    assert cfg.lang == "it" and cfg.paddle_lang == "french"


def test_no_language_leaves_the_config_exactly_as_it_was():
    from codai.config import OcrConfig
    cfg = OcrConfig(lang="it")
    assert om.OcrManager._cfg_for(cfg, "surya", "") is cfg


def test_a_config_that_is_not_a_dataclass_does_not_crash_the_request():
    assert om.OcrManager._cfg_for(_Cfg(), "paddle", "japan") is not None


# ------------------------------------------------------------- what we report #
def test_the_catalogue_says_which_engines_language_actually_steers():
    from codai.config import OcrConfig
    m = om.OcrManager()
    m._require_cfg = lambda: OcrConfig(paddle_enabled=True, surya_enabled=True,
                                       surya_accept_license=True)
    rows = {r["id"]: r for r in m.engine_catalogue()["data"]}
    assert rows["paddle"]["language_aware"] is True
    assert rows["surya"]["language_aware"] is False


def test_the_shipped_default_is_auto_not_a_language():
    from codai.config import OcrConfig
    assert OcrConfig().lang == "auto"
    assert OcrConfig().paddle_auto_lang == "latin"
