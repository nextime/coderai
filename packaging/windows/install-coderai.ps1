<#
.SYNOPSIS
  Install CoderAI on Windows: WSL2 + Docker Desktop + the published image + the
  `coderai` launcher on PATH.

.DESCRIPTION
  Run in an ELEVATED PowerShell (the WSL and Docker Desktop steps need it):

      irm https://raw.githubusercontent.com/nextime/coderai/master/packaging/windows/install-coderai.ps1 | iex

  or, from the release zip / the .exe installer's directory:

      .\install-coderai.ps1 [-Image ghcr.io/nextime/coderai:latest] [-InstallDir ...] [-NoPull]

  What it does, each step skipped when already done:
    1. enables the Windows features WSL2 needs and installs the WSL kernel;
    2. installs Docker Desktop with winget if `docker` is not on PATH;
    3. checks the NVIDIA driver (CUDA on WSL2 needs a 2021+ driver, 470 or newer);
    4. copies coderai.ps1 / coderai.cmd to the install dir and adds it to PATH;
    5. pulls the image (unless -NoPull) — ~28 GB.

  Docker Desktop must be on the WSL2 backend (its default). After a fresh
  WSL install Windows may need a reboot before Docker Desktop starts; the
  script says so and the launcher takes over from there.
#>
[CmdletBinding()]
param(
  [string]$Image = "ghcr.io/nextime/coderai:latest",
  [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "Programs\CoderAI"),
  [switch]$NoPull,
  [switch]$NoDocker
)
$ErrorActionPreference = "Stop"
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { Write-Host "Please run this from an elevated (Administrator) PowerShell." -ForegroundColor Red; exit 2 }

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }

# 1. WSL2
Step "WSL2"
$needReboot = $false
foreach ($f in @("Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform")) {
  $state = (Get-WindowsOptionalFeature -Online -FeatureName $f).State
  if ($state -ne "Enabled") {
    Write-Host "enabling $f"
    $r = Enable-WindowsOptionalFeature -Online -FeatureName $f -NoRestart
    if ($r.RestartNeeded) { $needReboot = $true }
  } else { Write-Host "$f already enabled" }
}
wsl --status *> $null
if ($LASTEXITCODE -eq 0) { Write-Host "wsl present" } else {
  Write-Host "installing the WSL kernel"; wsl --install --no-distribution 2>$null; $needReboot = $true
}
wsl --set-default-version 2 *> $null

# 2. Docker Desktop
Step "Docker Desktop"
if ($NoDocker) { Write-Host "skipped (-NoDocker)" }
elseif (Get-Command docker -ErrorAction SilentlyContinue) { Write-Host "docker already installed: $(docker --version)" }
else {
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Host "winget is not available; install Docker Desktop from https://www.docker.com/products/docker-desktop/ and re-run." -ForegroundColor Yellow
    exit 3
  }
  winget install --id Docker.DockerDesktop -e --accept-package-agreements --accept-source-agreements
  if ($LASTEXITCODE -ne 0) { Write-Host "winget could not install Docker Desktop" -ForegroundColor Red; exit 3 }
  $needReboot = $true
}

# 3. NVIDIA driver
Step "NVIDIA driver"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
  $ver = (& nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>$null | Select-Object -First 1)
  Write-Host "driver/card: $ver"
  $major = [int](($ver -split ",")[0].Trim().Split(".")[0])
  if ($major -lt 470) { Write-Host "driver $major is too old for CUDA on WSL2 (need 470+); update it from nvidia.com" -ForegroundColor Yellow }
} else {
  Write-Host "nvidia-smi not found: no NVIDIA driver, or not on PATH. CoderAI will run on CPU only (coderai -NoGpu)." -ForegroundColor Yellow
}

# 4. Launcher
Step "Launcher"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
foreach ($f in @("coderai.ps1", "coderai.cmd")) {
  $src = Join-Path $here $f
  if (Test-Path $src) { Copy-Item $src (Join-Path $InstallDir $f) -Force }
  else {
    # Piped through `iex`: fetch the launcher from the repository.
    Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/nextime/coderai/master/packaging/windows/$f" -OutFile (Join-Path $InstallDir $f)
  }
}
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (($userPath -split ";") -notcontains $InstallDir) {
  [Environment]::SetEnvironmentVariable("Path", "$userPath;$InstallDir", "User")
  Write-Host "added $InstallDir to the user PATH (open a new terminal to use 'coderai')"
}
$sm = Join-Path ([Environment]::GetFolderPath("Programs")) "CoderAI.lnk"
try {
  $ws = New-Object -ComObject WScript.Shell
  $lnk = $ws.CreateShortcut($sm)
  $lnk.TargetPath = Join-Path $InstallDir "coderai.cmd"
  $lnk.WorkingDirectory = $InstallDir
  $lnk.Description = "CoderAI — complete orchestration, distribution and escalation of remotizable advanced inference"
  $lnk.Save()
} catch {}

# 5. Image
Step "Image"
if ($needReboot) {
  Write-Host "Windows needs a reboot to finish WSL2 / Docker Desktop. After it, open a terminal and run:  coderai" -ForegroundColor Yellow
  Write-Host "(the first run pulls $Image, ~28 GB)"
  exit 0
}
if ($NoPull) { Write-Host "skipped (-NoPull); the first 'coderai' run pulls it" }
else {
  docker info *> $null
  if ($LASTEXITCODE -ne 0) { Write-Host "Docker Desktop is not running yet; start it, then run: coderai" -ForegroundColor Yellow; exit 0 }
  docker pull $Image
}
Write-Host "`nDone. Run:  coderai" -ForegroundColor Green
