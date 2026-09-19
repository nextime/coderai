<#
.SYNOPSIS
  CoderAI on Windows: run the published Linux image under Docker Desktop (WSL2)
  with the NVIDIA GPU passed through.

.DESCRIPTION
  The Windows twin of packaging/linux/run_oci.sh. Same container shape:
  /config, /models, /cache bind-mounted from a data directory, the API on one
  published port, --gpus all for CUDA-on-WSL. There is no /dev/nvidia* under
  WSL2 (--gpus all is the whole GPU story there) and no Vulkan path.

.EXAMPLE
  coderai                 # start (pull the image if needed) and open the admin page
  coderai -Stop           # stop the container
  coderai -Logs           # follow the log
  coderai -Upgrade        # pull the production branch into the image in place
  coderai -DataDir D:\coderai -Port 9000 -Image ghcr.io/nextime/coderai:0.2.20
#>
[CmdletBinding()]
param(
  [string]$Image = $(if ($env:CODERAI_IMAGE) { $env:CODERAI_IMAGE } else { "ghcr.io/nextime/coderai:latest" }),
  [string]$DataDir = $(if ($env:CODERAI_DATA) { $env:CODERAI_DATA } else { Join-Path $env:LOCALAPPDATA "CoderAI" }),
  [int]$Port = 8776,
  [string]$Name = "coderai",
  [switch]$Stop,
  [switch]$Logs,
  [switch]$Status,
  [switch]$Upgrade,
  [switch]$Pull,
  [switch]$NoGpu,
  [switch]$NoOpen,
  [string]$ExtraArgs = ""
)
$ErrorActionPreference = "Stop"

function Need-Docker {
  if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker is not installed. Run install-coderai.ps1 (elevated) first, or install Docker Desktop." -ForegroundColor Red
    exit 2
  }
  docker info *> $null
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker Desktop is not running — starting it..." -ForegroundColor Yellow
    $dd = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) { Start-Process $dd }
    $deadline = (Get-Date).AddMinutes(3)
    while ((Get-Date) -lt $deadline) {
      Start-Sleep 5
      docker info *> $null
      if ($LASTEXITCODE -eq 0) { break }
    }
    docker info *> $null
    if ($LASTEXITCODE -ne 0) { Write-Host "Docker did not come up." -ForegroundColor Red; exit 2 }
  }
}

Need-Docker

if ($Stop) {
  docker stop $Name 2>$null | Out-Null
  Write-Host "stopped $Name"
  exit 0
}
if ($Logs) { docker logs -f $Name; exit $LASTEXITCODE }
if ($Status) {
  docker ps --filter "name=^$Name$" --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
  exit 0
}
if ($Pull) { docker pull $Image; exit $LASTEXITCODE }

# Data layout: the same three mounts the Linux runner uses.
foreach ($d in @("config", "models", "cache")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $DataDir $d) | Out-Null
}

if ($Upgrade) {
  # The in-image upgrader (coderai-upgrade) pulls the production branch and
  # commits the result back onto the same tag; restart afterwards.
  Write-Host "upgrading $Image in place (production branch)..."
  docker stop $Name 2>$null | Out-Null
  docker run --rm --name "$Name-upgrade" --entrypoint /usr/local/bin/coderai-upgrade `
    -e CODERAI_UPGRADE_REF=production `
    -v "$($DataDir)\config:/config" -v "$($DataDir)\models:/models" -v "$($DataDir)\cache:/cache" `
    $Image
  if ($LASTEXITCODE -ne 0) { Write-Host "upgrade failed" -ForegroundColor Red; exit $LASTEXITCODE }
  Write-Host "upgraded; start it again with: coderai"
  exit 0
}

docker image inspect $Image *> $null
if ($LASTEXITCODE -ne 0) {
  Write-Host "pulling $Image (this is a ~28 GB download the first time)..."
  docker pull $Image
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$running = docker ps -q --filter "name=^$Name$"
if ($running) {
  Write-Host "$Name is already running on http://localhost:$Port"
} else {
  docker rm -f $Name 2>$null | Out-Null
  $gpu = @()
  if (-not $NoGpu) { $gpu = @("--gpus", "all", "-e", "NVIDIA_DRIVER_CAPABILITIES=all") }
  $dargs = @("run", "-d", "--name", $Name, "--ipc=host", "--restart", "unless-stopped") + $gpu + @(
    "-e", "CODERAI_HOST=0.0.0.0", "-e", "CODERAI_PORT=8776",
    "-p", "$($Port):8776",
    "-v", "$($DataDir)\config:/config", "-v", "$($DataDir)\models:/models", "-v", "$($DataDir)\cache:/cache")
  if ($ExtraArgs) { $dargs += @("-e", "CODERAI_EXTRA_ARGS=$ExtraArgs") }
  $dargs += $Image
  Write-Host "docker $($dargs -join ' ')"
  & docker @dargs | Out-Null
  if ($LASTEXITCODE -ne 0) { Write-Host "docker run failed" -ForegroundColor Red; exit $LASTEXITCODE }
  Write-Host "CoderAI starting on http://localhost:$Port  (data: $DataDir)"
  Write-Host "first-run admin credentials are printed in the log: coderai -Logs"
}

if (-not $NoOpen) {
  $deadline = (Get-Date).AddMinutes(5)
  while ((Get-Date) -lt $deadline) {
    try { Invoke-WebRequest -UseBasicParsing "http://localhost:$Port/healthz" -TimeoutSec 3 | Out-Null; break } catch { Start-Sleep 3 }
  }
  Start-Process "http://localhost:$Port/admin"
}
