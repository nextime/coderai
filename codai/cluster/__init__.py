# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Several coderai installs acting as one.

* :mod:`codai.cluster.nodes` — other installs used as engines of this front
  (the head polls them, assigns models to them, routes to them by capability).
* :mod:`codai.cluster.rpc` — the llama.cpp ``rpc-server`` processes this
  machine contributes, so one GGUF can span cards on several machines.
* :mod:`codai.cluster.compat` — which engine can run which backend, shared by
  the Models page (live filtering) and the save-time check.
"""
