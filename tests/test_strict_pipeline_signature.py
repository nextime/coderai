"""A strict pipeline signature must not turn an unsupported kwarg into a 500.

Measured 2026-11-01 on zeiss:

    [http] 500 on POST /v1/images/edits: "Image editing failed:
    Flux2KleinPipeline.__call__() got an unexpected keyword argument
    'negative_prompt'"

The edit path passes a negative prompt on every call, which is right -- the
endpoint used to have nowhere to put one. The assumption that made it safe is
stated elsewhere in the module: every diffusers ``__call__`` ends in
``**kwargs``, so an unknown argument is accepted and ignored. FLUX.2 Klein's
does not, so the request died instead.
"""
import codai.api.images as images


class _Tolerant:
    def __call__(self, prompt=None, **kwargs):
        pass


class _Strict:
    def __call__(self, prompt=None, image=None):
        pass


class _Pipe:
    def __init__(self, inner):
        self._inner = inner
    @property
    def __call__(self):
        return self._inner.__call__


def _kw():
    return {"prompt": "a cat", "negative_prompt": "blurry"}


def test_a_strict_signature_loses_only_what_it_cannot_take():
    got = images._drop_unsupported_call_kwargs(_Pipe(_Strict()), _kw())
    assert got == {"prompt": "a cat"}


def test_a_tolerant_signature_is_left_alone():
    """It may receive things it ignores; filtering would change working behaviour."""
    got = images._drop_unsupported_call_kwargs(_Pipe(_Tolerant()), _kw())
    assert got == _kw()


def test_an_uninspectable_pipeline_is_left_alone():
    class _Odd:
        __call__ = "not callable"
    assert images._drop_unsupported_call_kwargs(_Odd(), _kw()) == _kw()


def test_empty_kwargs_are_returned_unchanged():
    assert images._drop_unsupported_call_kwargs(_Pipe(_Strict()), {}) == {}
    assert images._drop_unsupported_call_kwargs(_Pipe(_Strict()), None) is None


def test_the_prompt_path_filters_before_returning():
    import inspect
    src = inspect.getsource(images._prompt_call_kwargs)
    assert src.count("_drop_unsupported_call_kwargs") == 2, (
        "both the weighted-embeddings return and the plain-text return must "
        "be filtered; a strict pipeline refuses either"
    )
