# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Front proxy package: always-responsive web/API front + supervised engines.

See ``docs/frontend-engine-split.md`` and ``docs/process-isolation-plans.md``.
"""

from codai.frontproxy.app import run_front, build_app

__all__ = ["run_front", "build_app"]
