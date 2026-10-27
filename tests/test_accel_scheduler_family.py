"""A model card must not be able to cross the flow/DDPM line either.

The request path (`images._apply_sampler`) refuses a cross-family sampler with
an explanation. The acceleration config had no such guard, so a card could
swap a flow-matching model's scheduler for a DDPM one silently — and the model
then failed at call time with

    EulerDiscreteScheduler.set_timesteps() got an unexpected keyword
    argument 'mu'

which names neither the card nor the model and reads like a diffusers bug.
Measured on Krea-2-Turbo, whose entry asked for EulerDiscreteScheduler.
"""
import logging

import pytest

from codai.models import acceleration as accel_mod


class _Sched:
    def __init__(self, name, cfg=None):
        self.__class__.__name__ = name
        self.config = cfg or {}


class _FlowSched:
    config = {"num_train_timesteps": 1000}


class _Pipe:
    def __init__(self, scheduler):
        self.scheduler = scheduler


def test_the_two_paths_share_one_definition_of_the_families():
    """Two copies of this set would drift, and the drift is invisible until a
    model produces noise."""
    from codai.api import images

    assert images._FLOW_SCHEDULERS is accel_mod.FLOW_SCHEDULERS
    assert "FlowMatchEulerDiscreteScheduler" in accel_mod.FLOW_SCHEDULERS


def test_a_ddpm_scheduler_is_refused_on_a_flow_matching_model(caplog):
    """Krea 2's own scheduler is FlowMatchEulerDiscreteScheduler; the card
    asked for EulerDiscreteScheduler."""
    pipe = _Pipe(_FlowSched())
    type(pipe.scheduler).__name__ = "FlowMatchEulerDiscreteScheduler"
    with caplog.at_level(logging.WARNING):
        accel_mod.apply_accel_to_pipeline(
            pipe, {"scheduler": "EulerDiscreteScheduler", "lora": None})
    assert type(pipe.scheduler).__name__ == "FlowMatchEulerDiscreteScheduler"
    assert any("cannot be substituted" in r.getMessage() for r in caplog.records)


def test_a_same_family_scheduler_is_still_applied():
    """LCM on an SD model is the reason this feature exists."""
    pipe = _Pipe(_FlowSched())
    type(pipe.scheduler).__name__ = "EulerDiscreteScheduler"
    accel_mod.apply_accel_to_pipeline(
        pipe, {"scheduler": "LCMScheduler", "lora": None})
    assert type(pipe.scheduler).__name__ == "LCMScheduler"
