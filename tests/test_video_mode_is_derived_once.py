"""The pipeline class and the call kwargs must agree about the request.

`mode` is optional on /v1/video/generations, and a client that sends a
keyframe without naming a mode means image-to-video. That inference lived in
_generate_video, which builds the call kwargs, but not at the load site,
which picks the pipeline CLASS from the raw field. The two disagreed, and
nothing noticed until the pipeline was called — after the weights were
loaded and the prompt encoded. Measured 2026-11-01 on an LTX-2.3 fine-tune
sent an image with no mode:

    [video] generation (initial) FAILED for 'diffusers/LTX-2.3-Diffusers':
    LTX2Pipeline.__call__() got an unexpected keyword argument 'image'

LTX2Pipeline is the text-to-video class. The image-to-video sibling exists
and was passed over because the load site read `mode` as None.
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import codai.api.video as video

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = io.open(os.path.join(ROOT, 'codai/api/video.py'), encoding='utf-8').read()


class _Req:
    def __init__(self, mode=None, image=None, init_image=None, video=None):
        self.mode = mode
        self.image = image
        self.init_image = init_image
        self.video = video


def test_an_image_with_no_mode_is_image_to_video():
    assert video._effective_mode(_Req(image='data:image/png;base64,AA')) == 'i2v'


def test_an_init_image_with_no_mode_is_image_to_video():
    assert video._effective_mode(_Req(init_image='x')) == 'i2v'


def test_a_video_with_no_mode_is_video_to_video():
    assert video._effective_mode(_Req(video='x')) == 'v2v'


def test_nothing_but_a_prompt_is_text_to_video():
    assert video._effective_mode(_Req()) == 't2v'


def test_an_explicit_mode_always_wins():
    """The keyframe-identity bridge sets mode='ti2v' after rendering a first
    frame; inferring 'i2v' from the image it just attached would undo it."""
    assert video._effective_mode(_Req(mode='ti2v', init_image='x')) == 'ti2v'
    assert video._effective_mode(_Req(mode='t2v', image='x')) == 't2v'
    assert video._effective_mode(_Req(mode='extend')) == 'extend'


def test_no_load_site_reads_the_raw_field_any_more():
    """A load site passing request.mode picks the class for a mode the caller
    never stated, which is the whole defect."""
    i = 0
    sites = []
    while True:
        j = SRC.find('_load_video_pipeline,', i)
        if j < 0:
            break
        sites.append(SRC[j:j + 220])
        i = j + 1
    assert sites, 'no _load_video_pipeline call sites found — did they move?'
    for s in sites:
        head = s[:s.index(')')] if ')' in s else s
        assert 'request.mode' not in head, head


def test_the_generator_and_the_load_site_use_the_same_derivation():
    """Two copies of the inference drift apart; that is how they disagreed."""
    assert SRC.count('def _effective_mode(') == 1
    i = SRC.index('def _generate_video(')
    body = SRC[i:SRC.index('\n    kw = _build_call_kwargs', i)]
    assert '_effective_mode(request)' in body


def test_the_measurement_is_recorded_with_the_helper():
    i = SRC.index('def _effective_mode(')
    body = SRC[i:SRC.index('\ndef ', i + 10)]
    assert 'LTX2Pipeline' in body and 'unexpected keyword' in body, (
        'without the failure it reads as a tidy-up and gets inlined again')


# --- the refusal ----------------------------------------------------------

class _NoImage:
    def __call__(self, prompt=None, num_frames=None):
        pass


class _WithImage:
    def __call__(self, prompt=None, image=None):
        pass


def test_a_pipeline_that_takes_an_image_passes_the_guard():
    video._require_image_conditioning(_WithImage(), 'i2v')


def test_a_pipeline_that_takes_no_image_is_refused_by_name():
    try:
        video._require_image_conditioning(_NoImage(), 'ti2v')
    except ValueError as e:
        msg = str(e)
    else:
        raise AssertionError('a t2v class must not accept a keyframe run')
    assert '_NoImage' in msg, 'name the class that cannot do it'
    assert 'ti2v' in msg, 'name the mode that was asked for'


def test_the_keyframe_is_never_silently_dropped():
    """Answering 200 with a clip the picture never touched is worse than
    failing: the caller cannot tell the frame was ignored."""
    i = SRC.index('def _require_image_conditioning(')
    body = SRC[i:SRC.index('\ndef ', i + 10)]
    assert 'raise' in body
    assert "kw.pop('image'" not in body


def test_both_image_modes_are_guarded_before_the_kwarg_is_set():
    i = SRC.index("elif mode == 'i2v' and init_src:")
    body = SRC[i:SRC.index("elif mode == 'interp'", i)]
    assert body.count('_require_image_conditioning(pipe, mode)') == 2, body
    for part in body.split('_require_image_conditioning(pipe, mode)')[1:]:
        assert "kw['image']" in part.split('elif')[0], (
            'the guard must come before the kwarg, not after')


# --- the teardown must not run on the event loop --------------------------

def test_the_teardown_is_handed_to_a_thread():
    """_free_pipeline_vram ends in the host teardown-leak breaker, which walks
    gc.get_referrers over every surviving tensor. Measured 2026-11-01 with it
    on the loop, after an LTX-2.3 generation failed:

        [loop] STALL IN PROGRESS: the event loop has not run for 249.1s
        ... File "codai/api/video.py", in _free_pipeline_vram
            for ref in gc.get_referrers(obj):

    and ~11 minutes later:

        host teardown-leak breaker: 23.9 GB of this pipeline's host-resident
        tensors survived teardown; nulled 7020 external reference(s)

    Nothing was served in that window, /healthz included, so one slow video
    read from outside as a hung server and took every unrelated client with
    it. The dedicated AST guard covers the general rule; this pins the two
    sites that were measured doing it.
    """
    i = SRC.index('async def video_generations(')
    body = SRC[i:]
    body = body[:body.index('\n@router.') if '\n@router.' in body else len(body)]
    # The nested thread target calls it directly — that is what it is for.
    # Anything else naming it must hand it to an executor.
    handoff = ('run_in_executor', 'to_thread')
    checked = 0
    in_target = False
    for line in body.splitlines():
        t = line.strip()
        if t.startswith('def _teardown('):
            in_target = True
            continue
        if in_target:
            if t and not line.startswith(' ' * 12):
                in_target = False
            else:
                continue
        if t.startswith('#') or not ('_free_pipeline_vram' in t or '_teardown' in t):
            continue
        assert any(h in t for h in handoff), f'called straight from the loop: {t}'
        checked += 1
    assert checked >= 2, f'expected both measured sites, saw {checked}'


def test_the_gc_and_cuda_sweep_went_with_it():
    """A full gc pass and cuda.synchronize on a 24 GB card are not free
    either, and leaving them behind keeps part of the stall."""
    i = SRC.index('def _teardown(_p):')
    body = SRC[i:SRC.index('run_in_executor', i)]
    assert '_gc.collect()' in body
    assert 'cuda.synchronize()' in body
    assert 'empty_cache()' in body
