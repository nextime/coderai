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

"""One HuggingFace cache, not two.

``TRANSFORMERS_CACHE`` is deprecated, but transformers 4.x still honours it --
and honours it as the ``cache_dir`` passed to ``hf_hub_download``, i.e. as a
hub-layout cache. Pointing it at ``<dir>/transformers`` does not give
transformers a corner of one cache; it gives the machine two hub caches, and
which one a process reads depends on the transformers version in its venv.

Measured 2026-10-28 on a production host: ``<dir>/hub`` held 122 models and
``<dir>/transformers`` held 2, both ``models--org--name``. A worker in an
isolated venv (transformers 4.46) looked only in the second one, found nothing,
and -- forced offline so a missing model fails fast -- reported "We couldn't
connect to https://huggingface.co" for a model that was present the whole time.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.hf_cache import hf_cache_env


def test_every_variable_names_the_same_hub_directory():
    env = hf_cache_env("/AI/huggingface")
    hub = "/AI/huggingface/hub"
    assert env["HF_HUB_CACHE"] == hub
    assert env["HUGGINGFACE_HUB_CACHE"] == hub
    assert env["TRANSFORMERS_CACHE"] == hub, (
        "transformers 4.x would read a different cache from huggingface_hub, "
        "so a model downloaded by the server is invisible to the worker")
    assert env["HF_HOME"] == "/AI/huggingface"


def test_no_variable_points_at_a_second_store():
    env = hf_cache_env("/AI/huggingface")
    caches = {v for k, v in env.items() if k != "HF_HOME"}
    assert len(caches) == 1, f"more than one cache directory in play: {caches}"


def test_a_user_path_is_expanded():
    env = hf_cache_env("~/hfcache")
    assert env["HF_HOME"] == os.path.expanduser("~/hfcache")
    assert not env["HF_HUB_CACHE"].startswith("~")


def test_a_worker_env_carries_the_hub_cache(monkeypatch):
    """child_env is what the isolated workers actually receive."""
    from codai.hf_cache import child_env
    monkeypatch.setenv("HF_HOME", "/AI/huggingface")
    env = child_env({})
    assert env["TRANSFORMERS_CACHE"] == env["HF_HUB_CACHE"] == "/AI/huggingface/hub"
