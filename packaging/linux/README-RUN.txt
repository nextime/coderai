CoderAI Linux Tarball
=====================

Run:
  ./bin/coderai

Then open:
  http://127.0.0.1:8776/admin

Default first-run credentials are created by the app:
  admin / admin

State directories inside this bundle:
  config/   app config and auth
  models/   model storage / data path
  cache/    Hugging Face, diffusers, and runtime caches

NVIDIA CUDA:
  Install a compatible NVIDIA driver on the host. The bundle includes Python/CUDA
  runtime packages from the source venv, but not the host GPU driver.

AMD/Intel Vulkan:
  Install Vulkan runtime/ICD packages on the host, for example on Debian/Ubuntu:
    sudo apt install libvulkan1 mesa-vulkan-drivers

CPU:
  No GPU setup is required.


================================================================================
Running the Docker / OCI image
================================================================================

The image publishes ONE port (8776). nginx inside the container fronts:
  http://HOST:8776/            CoderAI server + OpenAI-compatible API + admin UI
  http://HOST:8776/editor/     Video editor
  http://HOST:8776/videogen/   Videogen studio
  http://HOST:8776/township/   Township fighters

Three volumes hold all mutable state (everything else in the image is read-only):
  /config   app config + auth (small)
  /models   model storage / data path
  /cache    Hugging Face/diffusers caches + tool outputs (LARGE)

The examples below run the container as YOUR user (recommended). Create and own
the state dirs once, up front:

  mkdir -p coderai-config coderai-models coderai-cache
  sudo chown -R "$(id -u):$(id -g)" coderai-config coderai-models coderai-cache

Basic run (NVIDIA):
  docker run --gpus all --ipc=host -p 8776:8776 \
    --user "$(id -u):$(id -g)" \
    -v "$PWD/coderai-config:/config" \
    -v "$PWD/coderai-models:/models" \
    -v "$PWD/coderai-cache:/cache" \
    coderai:local

AMD/Intel Vulkan: replace `--gpus all` with `--device /dev/dri`.
CPU only: drop the GPU flag entirely.
Run as container-root instead: just omit the `--user` line (see "Running as a
non-root user" below for rootless / userns-remap alternatives).


External storage for /models and /cache
----------------------------------------
/models and /cache are where the big data lives, so put them on your large
storage and bind-mount them onto the defaults. The in-container paths never
change — only the host side does.

1) Host-mounted big disk / SAN (a path already mounted on the host):

   # Make the big-storage dirs owned by the UID you run as:
   sudo chown -R "$(id -u):$(id -g)" /srv/coderai/config \
     /mnt/bigstorage/coderai/models /mnt/bigstorage/coderai/cache

   docker run --gpus all --ipc=host -p 8776:8776 \
     --user "$(id -u):$(id -g)" \
     -v /srv/coderai/config:/config \
     -v /mnt/bigstorage/coderai/models:/models \
     -v /mnt/bigstorage/coderai/cache:/cache \
     coderai:local

   The launcher points HF_HOME at /cache/huggingface and writes tool outputs to
   /cache/{videogen_output,township_output}, so /cache on the big disk captures
   downloads AND produced artifacts.

2) NFS (shared across machines) — back a Docker volume with the NFS driver:

   docker volume create --driver local \
     --opt type=nfs --opt o=addr=10.0.0.5,rw,nfsvers=4 \
     --opt device=:/export/coderai/models  coderai-models
   docker volume create --driver local \
     --opt type=nfs --opt o=addr=10.0.0.5,rw,nfsvers=4 \
     --opt device=:/export/coderai/cache   coderai-cache

   docker run --gpus all --ipc=host -p 8776:8776 \
     --user "$(id -u):$(id -g)" \
     -v "$PWD/coderai-config:/config" \
     -v coderai-models:/models \
     -v coderai-cache:/cache \
     coderai:local

   (SMB/CIFS works the same way with `--opt type=cifs` and credentials.)
   For NFS, the export must let the UID you pass to --user write (e.g. map it,
   or set the right ownership on the export); don't rely on no_root_squash.

Performance note: NFS/CIFS are fine as a model LIBRARY, but mmap-heavy weight
loads and KV-cache spill are much faster on local NVMe or a fast SAN. Keep the
active inference weights on fast storage if you can.


Running as a non-root user
--------------------------
The image works as root (PID 1 sets up nginx + the services) AND as an arbitrary
UID. nginx pid/temp and the supervisor socket live under /tmp, logs go to
stdout/stderr, and Python doesn't write .pyc, so no part of the runtime needs to
write outside the mounted volumes.

Option A — run as your own UID/GID (recommended for bind mounts):

   # The mounted dirs must be owned by that UID so the app can write to them:
   mkdir -p coderai-config coderai-models coderai-cache
   sudo chown -R "$(id -u):$(id -g)" coderai-config coderai-models coderai-cache

   docker run --gpus all --ipc=host -p 8776:8776 \
     --user "$(id -u):$(id -g)" \
     -v "$PWD/coderai-config:/config" \
     -v "$PWD/coderai-models:/models" \
     -v "$PWD/coderai-cache:/cache" \
     coderai:local

   Caveat: with --user, the in-image standalone Python and app tree stay
   root-owned but world-readable, which is all the runtime needs. For NFS, the
   export must allow that UID to write (no_root_squash is NOT required when you
   pass a real --user; map/allow the UID you run as).

Option B — keep container-root but map it to an unprivileged host UID, with no
image changes. Best when you don't want to manage UIDs/ownership by hand:

   * Rootless Docker (run the daemon as a normal user), or
   * userns-remap: add  { "userns-remap": "default" }  to
     /etc/docker/daemon.json and restart Docker. Container root (UID 0) then maps
     to a high, unprivileged host subordinate UID automatically.

   In both cases run the normal command (no --user needed); the container thinks
   it is root, but the kernel sees an unprivileged user on the host.

GPU + non-root: NVIDIA Container Toolkit and /dev/dri both work under --user and
under rootless/userns-remap; no extra flags are needed beyond the usual
--gpus all (NVIDIA) or --device /dev/dri (Vulkan).
