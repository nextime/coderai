# CoderAI on Windows

CoderAI runs on Windows as the published Linux image under **Docker Desktop
(WSL2 backend)** with the NVIDIA GPU passed through (CUDA on WSL). Everything
works — text, images, video, audio, the cluster head or node, RunPod
escalation — except the AMD/Intel Vulkan path, which WSL2 does not expose.

## Install

Either run `CoderAI-Setup-<version>.exe` from the GitHub release, or in an
elevated PowerShell:

```powershell
irm https://raw.githubusercontent.com/nextime/coderai/master/packaging/windows/install-coderai.ps1 | iex
```

It enables WSL2, installs Docker Desktop with winget when missing, checks the
NVIDIA driver (470+), installs the `coderai` launcher on PATH and pulls
`ghcr.io/nextime/coderai:latest` (~28 GB). A reboot may be needed after the
WSL2 / Docker Desktop step; then run `coderai`.

## Use

```
coderai                    start (pull if needed) and open http://localhost:8776/admin
coderai -Stop              stop
coderai -Logs              follow the log (the first-run admin login is printed there)
coderai -Upgrade           pull the production branch into the image in place
coderai -DataDir D:\coderai -Port 9000 -Image ghcr.io/nextime/coderai:0.2.20
coderai -NoGpu             CPU only
```

Data (config, models, cache) lives under `%LOCALAPPDATA%\CoderAI` unless
`-DataDir` says otherwise; put it on the drive with the most space.

## Notes

* Give the WSL2 VM enough memory: `%USERPROFILE%\.wslconfig` with
  `[wsl2]` / `memory=32GB` (more for video models), then `wsl --shutdown`.
* Docker Desktop is NAT'd: as a cluster *node*, RPC server or Ray worker the
  machine needs its ports published (`-Port`) and the head must use the
  Windows host's address. As a cluster *head* nothing special is needed.
* Building the .exe: `iscc CoderAI.iss` with Inno Setup 6 on Windows, or
  `packaging/windows/build-installer.sh` under wine.
