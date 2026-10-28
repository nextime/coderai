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

"""A step callback only goes to a pipeline that takes one.

diffusers added `callback_on_step_end` per pipeline rather than across the
board. StableAudioPipeline does not take it, and passing it raised

    StableAudioPipeline.__call__() got an unexpected keyword argument
    'callback_on_step_end'

AFTER the model had loaded -- so one unsupported argument cost the whole load
and read like a broken model. Asked of the signature rather than kept as a list
of pipeline names, because such a list is wrong again at the next release.
"""

from codai.api.audio_gen import _step_callback_kwargs


def _cb(pipe, i, t, kw):
    return kw


def test_a_pipeline_that_takes_the_callback_gets_it():
    class _P:
        def __call__(self, prompt, num_inference_steps=1,
                     callback_on_step_end=None):
            pass

    got = _step_callback_kwargs(_P(), _cb)
    assert set(got) == {"callback_on_step_end"}
    assert got["callback_on_step_end"] is _cb


def test_a_pipeline_without_it_gets_nothing(capsys):
    """Stable Audio: the generation must run, without progress."""
    class _P:
        def __call__(self, prompt, num_inference_steps=1, audio_end_in_s=10.0):
            pass

    assert _step_callback_kwargs(_P(), _cb) == {}
    out = capsys.readouterr().out
    assert "no step callback" in out, (
        "a pipeline silently losing progress reporting must say so")


def test_the_older_spelling_is_adapted():
    """Pre-0.25 pipelines expose `callback` instead; progress still works."""
    class _P:
        def __call__(self, prompt, callback=None):
            pass

    got = _step_callback_kwargs(_P(), _cb)
    assert set(got) == {"callback"}
    seen = {}

    def _cb2(pipe, i, t, kw):
        seen["i"] = i
        return kw

    got2 = _step_callback_kwargs(_P(), _cb2)
    got2["callback"](7, 0.5, None)
    assert seen["i"] == 7, "the adapter did not forward the step index"


def test_an_unintrospectable_pipeline_does_not_raise():
    """A C-implemented or wrapped __call__ must degrade, not explode."""
    class _P:
        __call__ = print          # builtin: signature may be unavailable

    assert isinstance(_step_callback_kwargs(_P(), _cb), dict)


def test_both_audio_call_sites_use_the_guard():
    import inspect
    from codai.api import audio_gen
    src = inspect.getsource(audio_gen)
    # The `**` form, so the definition line is not counted as a use.
    assert src.count("**_step_callback_kwargs(pipe,") == 2, (
        "an audio generation path still passes the callback unconditionally")
    assert "callback_on_step_end=_sa_step_cb" not in src
    assert "callback_on_step_end=_aud_step_cb" not in src
