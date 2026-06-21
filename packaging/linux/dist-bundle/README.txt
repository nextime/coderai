CoderAI — Docker distribution
==============================

This bundle contains everything needed to run CoderAI from a prebuilt
all-in-one Docker image (server + API + admin UI + video editor + videogen
studio + township fighters, all behind one nginx port).

Contents
--------
  install.sh           Loads the image into Docker and installs the runner.
  uninstall.sh         Removes the runner and (optionally) the image.
  coderai-docker       The run wrapper (installed to your PATH by install.sh).
  coderai-dist.tar.gz  The Docker image (gzip-compressed `docker save`).
  README.txt           This file.

Requirements
------------
  - Docker (or Podman: set CONTAINER_ENGINE=podman).
  - For GPU: NVIDIA Container Toolkit (--nvidia) or /dev/dri access (--vulkan).

Install
-------
  ./install.sh            (prompts to confirm; add --yes to skip the prompt)

  Installs `coderai-docker` to:
    - /usr/local/bin            when run as root
    - ~/.local/usr/bin          when run as a normal user (added to PATH via
                                ~/.bashrc if it isn't there already)

Uninstall
---------
  ./uninstall.sh          Removes the runner from both locations and (after a
                          prompt) the image. --keep-image keeps the image;
                          --yes skips all prompts. Runtime data is left untouched.

Run
---
  coderai-docker --nvidia                 # CUDA; or --vulkan (AMD/Intel) / --cpu
  coderai-docker --nvidia --vulkan        # both backends (also: --all)
  coderai-docker --help                   # all options

  Then open http://127.0.0.1:8776/admin

Useful options (see --help)
---------------------------
  --nvidia / --vulkan     GPU backends; ADDITIVE (pass both, or --all). --vulkan
                          auto-maps the host libcuda.so.1 (the bundled llama-cpp
                          is a CUDA build); --with-libcuda[=PATH] overrides it.
  -p, --port PORT         Host port (default 8776).
  --host ADDR             Bind the published port to a specific interface
                          (e.g. 127.0.0.1 for localhost-only; default all).
  --data-dir PATH         Where config/models/cache live (default ./coderai-runtime).
  --local                 Run against your existing ~/.coderai config.
  --map HOST[:CONT]       Bind-mount a host dir at the same path (for absolute
                          model paths in models.json), e.g. --map /AI/guffcache.
  --debug[=engine,ws,...] Run with coderai debug flags + a host-tailable file log.
  --log-file PATH         In-container log path (default /cache/logs/coderai.log,
                          visible on the host under the cache mount).
  --no-tools              Disable the bundled demo tool web UIs (video editor,
                          videogen, township). They're on by default.
  --enable-tool NAME      Force-enable a demo tool (also turns on parler TTS,
                          which is off by default). NAME: video-editor | videogen
                          | township | parler. Repeatable.
  --disable-tool NAME     Disable a single demo tool. Repeatable.
  -d, --detach            Run in the background.

Manual image load (without install.sh)
---------------------------------------
  docker load < coderai-dist.tar.gz       # restores the coderai:dist tag
