"""C5: a model's own recommended settings, and the sampler machinery.

Without these a client keeps a table of per-model step counts keyed on the
model NAME -- which breaks the moment a second quantization of the same model
is added -- and a client without one runs an 8-step distilled model for 30
steps: four times the cost for a worse picture.
"""
import pytest

from codai.api.images import SAMPLERS, _apply_sampler, model_defaults


# ------------------------------------------------------------------ defaults
def test_defaults_are_read_from_the_model_entry():
    d = model_defaults({"defaults": {"steps": 8, "guidance_scale": 1.0,
                                     "sampler": "euler"}})
    assert d["steps"] == 8 and d["guidance_scale"] == 1.0


@pytest.mark.parametrize("cfg", [None, {}, {"defaults": None},
                                 {"defaults": "not a dict"}])
def test_a_model_without_defaults_is_not_an_error(cfg):
    assert model_defaults(cfg) == {}


def test_defaults_are_not_call_kwargs():
    # `call_kwargs` applies on EVERY call because the card says the model does
    # not work without it; `defaults` fill in only what the caller omitted.
    # Confused for one another, a caller's explicit steps would be overridden.
    import inspect

    from codai.api import images
    src = inspect.getsource(images.model_defaults)
    assert "never override" in src


def test_the_request_beats_the_card_and_the_card_beats_the_floor():
    import inspect

    from codai.api import images
    src = inspect.getsource(images._generate_with_diffusers)
    # The window is the step/guidance decision itself -- from where the card is
    # read to where the guidance ends -- not a fixed slice. A fixed slice fails
    # the moment another layer is inserted between the two, which says nothing
    # about the precedence this test is here to protect.
    i = src.index("_dflt = model_defaults(model_config)")
    window = src[i:src.index("_progress_reset(", i)]
    # request.steps first, then the fused-accel preset, then the card, then 30.
    assert "request.steps if request.steps" in window
    assert "_accel_steps if _accel_steps" in window
    assert "int(_d_steps) if _d_steps" in window
    assert window.index("request.steps if request.steps") < \
        window.index("_accel_steps if _accel_steps") < \
        window.index("int(_d_steps) if _d_steps") < \
        window.index("30 if quality")


def test_video_layers_the_card_under_the_accel_preset_and_over_the_floor():
    import inspect

    from codai.api import video
    src = inspect.getsource(video._generate_video)
    i_accel = src.index("accel_call_defaults")
    i_card = src.index("model_defaults(")
    i_floor = src.index("kw.setdefault('num_inference_steps', 25)")
    assert i_accel < i_card < i_floor, "the precedence order is wrong"


def test_build_call_kwargs_still_sets_only_what_the_request_asked():
    # The accel and card layers both use setdefault, so an unset key has to
    # keep meaning "the caller did not ask". Breaking this makes the card
    # silently outrank a fused distill preset.
    import types

    from codai.api.video import _build_call_kwargs
    r = types.SimpleNamespace(prompt="x", negative_prompt=None,
                              num_inference_steps=None, guidance_scale=None,
                              num_frames=None, width=None, height=None,
                              seed=None)
    kw = _build_call_kwargs(r)
    assert "num_inference_steps" not in kw
    assert "guidance_scale" not in kw


# ------------------------------------------------------------------ samplers
class _Pipe:
    def __init__(self, sched):
        self.scheduler = sched


def _sched(name):
    import diffusers
    return getattr(diffusers, name)()


def test_a_known_sampler_swaps_the_scheduler():
    p = _Pipe(_sched("EulerDiscreteScheduler"))
    assert _apply_sampler(p, "euler_a") == "EulerAncestralDiscreteScheduler"
    assert type(p.scheduler).__name__ == "EulerAncestralDiscreteScheduler"


def test_the_spelling_is_forgiving():
    for spelling in ("DPMPP-2M", "dpmpp_2m", " dpmpp 2m "):
        p = _Pipe(_sched("EulerDiscreteScheduler"))
        assert _apply_sampler(p, spelling) == "DPMSolverMultistepScheduler"


def test_an_unknown_sampler_names_the_valid_ones():
    p = _Pipe(_sched("EulerDiscreteScheduler"))
    with pytest.raises(ValueError) as e:
        _apply_sampler(p, "nonsense")
    assert "unknown sampler" in str(e.value)
    assert "euler" in str(e.value)


def test_crossing_the_flow_matching_boundary_is_refused():
    """Not a worse picture -- noise.

    Flow-matching and DDPM-family schedulers parameterise the trajectory
    differently, so substituting one for the other does not degrade the
    result, it destroys it. A silent swap would look like the model being
    broken.
    """
    flow = _Pipe(_sched("FlowMatchEulerDiscreteScheduler"))
    with pytest.raises(ValueError) as e:
        _apply_sampler(flow, "ddim")
    assert "not compatible" in str(e.value)

    classic = _Pipe(_sched("EulerDiscreteScheduler"))
    with pytest.raises(ValueError):
        _apply_sampler(classic, "flow_euler")

    # Within a family it is fine.
    assert _apply_sampler(flow, "flow_heun") == "FlowMatchHeunDiscreteScheduler"


def test_asking_for_the_scheduler_already_in_place_is_a_no_op():
    p = _Pipe(_sched("EulerDiscreteScheduler"))
    before = p.scheduler
    assert _apply_sampler(p, "euler") == "EulerDiscreteScheduler"
    assert p.scheduler is before


def test_an_empty_sampler_does_nothing():
    p = _Pipe(_sched("EulerDiscreteScheduler"))
    before = p.scheduler
    assert _apply_sampler(p, "") == ""
    assert _apply_sampler(p, None) == ""
    assert p.scheduler is before


def test_every_mapped_sampler_exists_in_this_diffusers():
    import diffusers
    missing = sorted({c for c in SAMPLERS.values()
                      if not hasattr(diffusers, c)})
    assert not missing, missing


# ------------------------------------------------- what /v1/models publishes
def test_the_listing_can_carry_defaults_modes_and_licence():
    from codai.pydantic.textrequest import ModelInfo
    m = ModelInfo(id="x", defaults={"steps": 8}, modes=["t2v", "i2v"],
                  license="Apache-2.0", available=True)
    d = m.model_dump()
    assert d["defaults"] == {"steps": 8}
    assert d["modes"] == ["t2v", "i2v"]
    assert d["license"] == "Apache-2.0"
    assert d["available"] is True


def test_modes_and_licence_are_read_from_either_place():
    """An operator should not have to remember whether these go at the top
    level of the entry or inside `defaults`.

    Asserted over the whole ``ModelInfo(...)`` call rather than a fixed slice
    of source: the first version of this test read 700 characters from an
    anchor and broke the moment a comment was added inside the block, which
    tested the comment density rather than the behaviour.
    """
    import inspect

    from codai.models import manager
    src = inspect.getsource(manager)
    anchor = 'defaults=(meta.get("defaults") or None)'
    i = src.index(anchor)
    # The enclosing ModelInfo(...) call, delimited by BALANCED parentheses
    # rather than by the first "))" after the anchor. Scanning for "))" found
    # the end of any nested call that happened to sit inside the block -- a
    # mode-inference fallback written as
    # `or _modes_from_caps(caps, resolved_type))` closed two parens and
    # truncated the slice, so fields further down went missing and this test
    # failed for a field it was not about.
    start = src.rindex("models.append(ModelInfo(", 0, i)
    depth, end = 0, None
    for pos in range(start, len(src)):
        if src[pos] == "(":
            depth += 1
        elif src[pos] == ")":
            depth -= 1
            if depth == 0:
                end = pos
                break
    assert end is not None, "unbalanced ModelInfo(...) call"
    block = src[start:end]
    for spelling in ('meta.get("modes")', 'meta.get("video_subtypes")',
                     'meta.get("license")', 'meta.get("licence")',
                     '(meta.get("defaults") or {}).get("modes")',
                     '(meta.get("defaults") or {}).get("license")'):
        assert spelling in block, spelling
