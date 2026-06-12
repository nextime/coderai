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

"""Central task registry for long-running operations."""

from codai.tasks.registry import (
    Task,
    TaskCancelled,
    TaskRegistry,
    task_registry,
    raise_if_cancelled,
    wait_if_paused,
    loading_task,
)

__all__ = [
    "Task",
    "TaskCancelled",
    "TaskRegistry",
    "task_registry",
    "raise_if_cancelled",
    "wait_if_paused",
    "loading_task",
]
