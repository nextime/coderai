"""Loading MM-Embed broke e5-v, in the same process, for the rest of its life.

Found during a customer test run on zeiss 2026-10-31. The test asked for four
embedders in a row; the third was MM-Embed, which failed on its gated retriever
(nvidia/NV-Embed-v1), and the fourth was e5-v, which answered

    500 Failed to load embedding model: 'LlavaNextConfig' object has no
        attribute 'retriever'

e5-v had worked all day on its own. The cause is process-wide state: MM-Embed's
`modeling_nvmmembed` calls `AutoModel.register(LlavaNextConfig, NVMMEmbedModel)`
at import time, and e5-v IS a LLaVA-NeXT checkpoint — so after MM-Embed has been
touched, every LLaVA-NeXT model in the process is constructed as MM-Embed and
asks its config for MM-Embed's field.

coderai has to allow that registration (transformers 5 refuses a duplicate, and
the refusal happens during the import, so loading the class directly does not
avoid it) — but it must put the mapping back afterwards. The first version of
the shim restored the `register` METHOD and left the entry it had written.
"""

import sys

import pytest


@pytest.fixture
def E():
    from codai.api import embeddings
    return embeddings


class _FakeLazyMapping:
    """Enough of _LazyAutoMapping: an override dict plus a stock lookup.

    Deliberately WITHOUT a one-argument `get`, because the real one requires a
    default and raises without it — a shim that calls `.get(key)` on the real
    mapping would fail here too.
    """

    def __init__(self, stock):
        self._stock = dict(stock)
        self._extra_content = {}

    def __contains__(self, key):
        return key in self._extra_content or key in self._stock

    def get(self, key, default):
        if key in self._extra_content:
            return self._extra_content[key]
        return self._stock.get(key, default)


def _fake_auto(monkeypatch, stock):
    """Install a fake _BaseAutoModelClass whose register writes _extra_content."""
    import transformers.models.auto.auto_factory as af

    class _Auto:
        _model_mapping = _FakeLazyMapping(stock)

        @classmethod
        def register(cls, config_class, model_class, exist_ok=False):
            if config_class in cls._model_mapping and not exist_ok:
                raise ValueError(f"{config_class} is already used by a "
                                 f"Transformers model.")
            cls._model_mapping._extra_content[config_class] = model_class

    monkeypatch.setattr(af, "_BaseAutoModelClass", _Auto)
    return _Auto


class _LlavaNextConfig:      # stands in for the real config class
    pass


class _StockLlavaNext:
    pass


class _NVMMEmbed:
    pass


def test_the_registration_is_allowed_while_the_shim_is_up(E, monkeypatch):
    auto = _fake_auto(monkeypatch, {_LlavaNextConfig: _StockLlavaNext})
    with pytest.raises(ValueError):
        auto.register(_LlavaNextConfig, _NVMMEmbed)     # transformers 5 refuses
    restore = E._allow_remote_automodel_reregister()
    try:
        auto.register(_LlavaNextConfig, _NVMMEmbed)     # the remote code's call
        assert auto._model_mapping.get(_LlavaNextConfig, None) is _NVMMEmbed
    finally:
        restore()


def test_the_mapping_is_put_back_afterwards(E, monkeypatch):
    """The defect: e5-v loaded as MM-Embed because this entry stayed."""
    auto = _fake_auto(monkeypatch, {_LlavaNextConfig: _StockLlavaNext})
    restore = E._allow_remote_automodel_reregister()
    auto.register(_LlavaNextConfig, _NVMMEmbed)
    restore()
    assert auto._model_mapping.get(_LlavaNextConfig, None) is _StockLlavaNext, \
        "the next LLaVA-NeXT model in this process still loads as MM-Embed"
    assert _LlavaNextConfig not in auto._model_mapping._extra_content


def test_an_override_that_was_already_there_is_restored_not_deleted(E, monkeypatch):
    auto = _fake_auto(monkeypatch, {})
    auto._model_mapping._extra_content[_LlavaNextConfig] = _StockLlavaNext
    restore = E._allow_remote_automodel_reregister()
    auto.register(_LlavaNextConfig, _NVMMEmbed)
    restore()
    assert auto._model_mapping._extra_content[_LlavaNextConfig] is _StockLlavaNext


def test_the_register_method_is_restored_too(E, monkeypatch):
    auto = _fake_auto(monkeypatch, {_LlavaNextConfig: _StockLlavaNext})
    before = auto.register
    restore = E._allow_remote_automodel_reregister()
    restore()
    assert auto.register == before or auto.register.__func__ is before.__func__


def test_restoring_twice_is_harmless(E, monkeypatch):
    auto = _fake_auto(monkeypatch, {_LlavaNextConfig: _StockLlavaNext})
    restore = E._allow_remote_automodel_reregister()
    auto.register(_LlavaNextConfig, _NVMMEmbed)
    restore()
    restore()
    assert auto._model_mapping.get(_LlavaNextConfig, None) is _StockLlavaNext


def test_a_failed_mmembed_load_still_restores(E):
    """MM-Embed failed on its gated retriever and STILL left the mapping
    rewritten, which is how one model's 500 became another's."""
    src = open(E.__file__, encoding="utf-8").read()
    i = src.index("def _load_mmembed(")
    body = src[i:src.index("def _load_jinaclip(")]
    assert "finally:" in body and "_restore()" in body
    # The direct-class attempt swallows its own exception, so the restore must
    # not hang off that branch.
    assert body.index("_restore()") > body.index("direct class load failed")
