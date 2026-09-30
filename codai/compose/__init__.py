# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Server-side video composition — turning the pieces into a finished reel.

CoderAI can generate every *piece* of a short video (narration, stills, clips,
music) but a client that is not on this machine cannot assemble them: that
needs ffmpeg, the narration's exact duration, and word timings. This package
is that assembly, driven by ``POST /v1/video/compose``
(:mod:`codai.api.compose`):

* :mod:`codai.compose.media` — resolving a media reference (URL, ``/v1/files``
  path, data URI, base64) to a local file, probing duration, finding fonts;
* :mod:`codai.compose.captions` — word timings (from STT or estimated) and the
  ASS / SRT / VTT the renderer burns in and returns;
* :mod:`codai.compose.render` — the ffmpeg pipeline: one normalised clip per
  visual, concat or crossfade, narration + ducked music, burned captions,
  overlays, thumbnail.

Everything here is synchronous and side-effect-free apart from the files it
writes into the job's scratch directory, so the whole pipeline is testable
without a server and without a GPU.
"""
