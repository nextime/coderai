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

"""A run must say which trainer image its pods are launched from.

Half of this subsystem lives in that image -- the dataset code, the trainer,
the runner -- and the log named everything except it: the cards, the prices,
the clouds, the regions, the VRAM floor.

What that cost, measured: the fix for a bug that killed every distributed run
shipped at 19:45 and the image was rebuilt, pushed and cosign-signed three
times over the following five hours, while the `lora_train` block pinned an
explicit `:v0.27.1-r2` tag from before the fix. Every pod all day ran the old
code and failed identically. The log said "round 1 failed" and nothing else.
It was found at 01:00 by reading the config, not the log.

Whether the tag FLOATS is part of the finding, not decoration: with `:latest`
a rebuild reaches the next pod, and with an explicit tag it never will however
many times it is pushed. A reader has to be told which case they are in before
they can trust that rebuilding changed anything.
"""

import inspect

from codai.api import loras


SRC = inspect.getsource(loras._text_train_distributed)


def test_the_image_is_logged_at_rental():
    assert "trainer image" in SRC
    i = SRC.index("trainer image")
    block = SRC[max(0, i - 600):i + 500]
    assert "cfg.image" in block


def test_a_pinned_tag_is_called_out_as_pinned():
    """The actionable half: a rebuilt image will not reach these pods."""
    i = SRC.index("trainer image")
    block = SRC[i:i + 600]
    assert "PINNED" in block
    assert "will NOT reach these pods" in block


def test_a_floating_tag_is_called_out_too():
    i = SRC.index("trainer image")
    block = SRC[i:i + 600]
    assert "floating tag" in block


def test_the_floating_test_handles_a_bare_repository():
    """`ghcr.io/x/y` with no tag means `:latest` to a registry, so it floats.

    Checked on the repository segment, not the whole string -- a registry host
    with a port (`host:5000/x/y`) has a colon that is not a tag, and reading
    that as pinned would report the opposite of the truth.
    """
    i = SRC.index("_floating =")
    line = SRC[i:i + 200]
    assert 'rsplit("/", 1)[-1]' in line, "a host:port would read as a tag"


def test_the_image_reaches_the_job_record():
    """So a client, and a later reader, can see it without the log."""
    assert "trainer_image=_img" in SRC
    assert "trainer_image" in loras._MIRROR_FIELDS


def test_the_two_adapter_failures_are_distinguishable():
    """The pre-fix loader and a genuine mismatch printed the SAME sentence.

    So a fleet failing on an old image looked exactly like an adapter trained
    against the wrong base, which is where the diagnosis went wrong.
    """
    from codai.api import text_lora_trainer as tlt
    src = inspect.getsource(tlt.run_training)
    i = src.index("PEFT refused")
    block = src[i:i + 900]
    assert "no place for" in block, (
        "the new message should name the OLD wording so a reader can tell "
        "which loader produced the failure they are looking at")
    assert "old trainer image" in block, (
        "the phrase must survive line wrapping -- a grep for it is how\n         someone finds this")
    # and the new message's own phrasing must not be the old one
    assert 'f"PEFT refused {len(unexpected)} of {len(incoming)} tensors "' in src
