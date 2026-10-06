# Vendored third-party source

## `longcat_video/`

Upstream: <https://github.com/meituan-longcat/LongCat-Video>
Commit:   `6b3f4b8582a8bc3f20f795735f5383716c4ba794`
Licence:  MIT (see `LONGCAT-VIDEO-LICENSE`) — © 2025 Meituan

`LongCatVideoPipeline` and the DiT, VAE, scheduler and INT8 quantisation helpers
live in a package *inside* the upstream repo and are not published on PyPI. They
are vendored here, rather than cloned at install time, so that:

* an image someone else pulls already has them — no network, no git, and no
  "download the upstream sources" step in anyone's setup;
* the version is pinned to a commit we tested against instead of whatever `main`
  happens to be on the day of the install;
* the isolated Python 3.10 venv still installs its own dependencies from
  `requirements-longcat.txt` (torch 2.6+cu124, transformers 4.41, …) — only the
  26 source files are vendored, no weights and no wheels.

496 KB, 26 Python files, nothing compiled. To refresh it, copy `longcat_video/`
from a clean checkout of the commit you intend to pin and update the hash above.
