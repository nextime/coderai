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

"""Global state for codai API modules."""
from typing import Any, Optional

# Global state variables
_args: Optional[Any] = None
_debug: bool = False
_file_path: Optional[str] = None
_system_prompt: Optional[str] = None
_tools_closer_prompt: bool = False
_grammar_guided_gen: bool = False
_load_mode: str = "ondemand"


def set_global_args(args: Any) -> None:
    """Set global arguments."""
    global _args
    _args = args


def get_global_args() -> Any:
    """Get global arguments."""
    return _args


def set_global_debug(debug: bool) -> None:
    """Set global debug flag."""
    global _debug
    _debug = debug


def get_global_debug() -> bool:
    """Get global debug flag."""
    return _debug


def set_global_file_path(path: Optional[str]) -> None:
    """Set global file path."""
    global _file_path
    _file_path = path


def get_global_file_path() -> Optional[str]:
    """Get global file path."""
    return _file_path


def set_global_system_prompt(prompt: Optional[str]) -> None:
    """Set global system prompt."""
    global _system_prompt
    _system_prompt = prompt


def get_global_system_prompt() -> Optional[str]:
    """Get global system prompt."""
    return _system_prompt


def set_global_tools_closer_prompt(enabled: bool) -> None:
    """Set global tools closer prompt flag."""
    global _tools_closer_prompt
    _tools_closer_prompt = enabled


def get_global_tools_closer_prompt() -> bool:
    """Get global tools closer prompt flag."""
    return _tools_closer_prompt


def set_grammar_guided_gen(enabled: bool) -> None:
    """Set grammar-guided generation flag."""
    global _grammar_guided_gen
    _grammar_guided_gen = enabled


def get_grammar_guided_gen() -> bool:
    """Get grammar-guided generation flag."""
    return _grammar_guided_gen


def set_load_mode(mode: str) -> None:
    """Set load mode (loadall, loadswap, nopreload, ondemand)."""
    global _load_mode
    _load_mode = mode


def get_load_mode() -> str:
    """Get load mode."""
    return _load_mode