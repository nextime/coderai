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

"""Two bugs with the same shape: a signal that existed and was not consulted.

1. Same-path siblings. Two cards can share one weights path and differ only in
   settings -- Wan2.2-I2V-A14B-Lightning and the plain Wan2.2-I2V-A14B both
   point at Wan-AI/Wan2.2-I2V-A14B-Diffusers, declaring 22 GB and 151 GB. The
   manager has disambiguated these since the lisa/lisa-32k case, via
   `requested_id`. The placement callers never passed it, so the Lightning
   variant was sized at 151 GB, read as ~127 GB over free VRAM, and sent
   straight to CPU offload -- it then computed on the CPU at 0% GPU.

2. The leak watcher's load signal. It consulted only the manager's ready event,
   which diffusers image/video loads never touch, so a 12-minute multi-shard
   load looked idle and its rising RSS looked like a leak.
"""

import inspect
import re

import pytest


# ------------------------------------------------- sizing the right sibling

def test_the_video_placement_passes_the_requested_id():
    from codai.api import video
    src = inspect.getsource(video.video_generations) \
        if hasattr(video, "video_generations") else inspect.getsource(video)
    m = re.search(r"_get_model_used_vram_gb\(\s*model_key,\s*model_name,\s*"
                  r"requested_id=", inspect.getsource(video))
    assert m, ("the video placement decision sizes the model without "
               "requested_id, so same-path siblings cannot be told apart")


def test_the_image_placement_passes_the_requested_id():
    from codai.api import images
    assert re.search(r"_get_model_used_vram_gb\([^)]*requested_id=",
                     inspect.getsource(images), re.S), (
        "the image placement decision sizes the model without requested_id")


def test_the_manager_prefers_the_requested_cards_own_size():
    """The disambiguation itself: an alias that matches the requested id wins."""
    from codai.models.manager import multi_model_manager as m
    saved = dict(m.config or {})
    try:
        m.config.clear()
        m.config.update({
            # load_in_4bit as the real cards carry it. Without it this pair
            # only "fit" a 24 GiB card because the estimator was also halving a
            # configured VRAM figure by the checkpoint's storage dtype -- fixed
            # 2026-10-31 after that same halving let a 7B embedder load with
            # 954 MiB left and OOM on its first batch.
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers": {
                "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "used_vram_gb": 151.42, "load_in_4bit": True, "alias": None},
            "__lightning": {
                "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "used_vram_gb": 22.0, "load_in_4bit": True,
                "alias": "Wan2.2-I2V-A14B-Lightning"},
        })
        big = m._get_model_used_vram_gb("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
        small = m._get_model_used_vram_gb(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            requested_id="Wan2.2-I2V-A14B-Lightning")
        # Not the declared figures: the estimator scales them by a quantization
        # factor. What matters is which CARD was consulted, and that the answer
        # is small enough to stay on a 24 GiB device instead of being exiled.
        assert small < big / 2, (
            f"requested_id made no difference ({small} vs {big}) -- the "
            f"Lightning card is still sized by its 151 GB sibling")
        assert small < 24.0, (
            f"the Lightning card is sized at {small} GB, which still does not "
            f"fit a 24 GiB card, so it goes straight to CPU offload")
    finally:
        m.config.clear()
        m.config.update(saved)


# --------------------------------------------- knowing a load is in progress

def test_a_loading_reservation_counts_as_a_load():
    """Diffusers image/video loads never touch the manager's ready event, so
    the reservation is the only signal that covers them."""
    from codai.models import ram_monitor
    from codai.models.manager import multi_model_manager as m
    saved = dict(m._loading_reservations)
    try:
        m._loading_reservations.clear()
        m._model_ready_event.set()
        assert ram_monitor._load_in_progress() is False
        m._loading_reservations["some-model"] = 19.0
        assert ram_monitor._load_in_progress() is True, (
            "a reserved load window is not seen as loading, so its rising RSS "
            "is mistaken for a leak")
    finally:
        m._loading_reservations.clear()
        m._loading_reservations.update(saved)


def test_the_ready_event_still_counts():
    from codai.models import ram_monitor
    from codai.models.manager import multi_model_manager as m
    saved = dict(m._loading_reservations)
    try:
        m._loading_reservations.clear()
        m._model_ready_event.clear()
        assert ram_monitor._load_in_progress() is True
    finally:
        m._model_ready_event.set()
        m._loading_reservations.update(saved)


def test_a_leak_is_never_suspected_while_loading():
    """The heuristic resets its trend while loading; that only works if the
    load is detected in the first place."""
    from codai.models import ram_monitor
    src = inspect.getsource(ram_monitor._loop)
    assert "loading = _load_in_progress()" in src
    assert "idle = _scheduler_idle() and not loading" in src
