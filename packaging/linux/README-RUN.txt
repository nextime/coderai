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
