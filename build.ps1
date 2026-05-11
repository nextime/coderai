param(
    [string]$Backend = "all",
    [string]$Venv = "",
    [switch]$Package
)

$ErrorActionPreference = "Stop"
$Backend = $Backend.ToLowerInvariant()
$validBackends = @("cuda", "cpu", "all")
if ($validBackends -notcontains $Backend) {
    throw "Invalid backend '$Backend'. Use one of: cuda, cpu, all"
}

function Write-Info($msg) { Write-Host $msg -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Ok($msg) { Write-Host $msg -ForegroundColor Green }

Write-Info "========================================"
Write-Info "  CoderAI Windows Build Script"
Write-Info "  Backend: $Backend"
Write-Info "========================================"

$python = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
$pythonArgs = if ($python -eq "py") { @("-3") } else { @() }

if ([string]::IsNullOrWhiteSpace($Venv)) {
    switch ($Backend) {
        "cuda" { $Venv = "venv_win_cuda" }
        "cpu" { $Venv = "venv_win_cpu" }
        default { $Venv = "venv_win_all" }
    }
}

if (-not (Test-Path $Venv)) {
    Write-Info "Creating virtual environment: $Venv"
    & $python @pythonArgs -m venv $Venv
} else {
    Write-Warn "Using existing virtual environment: $Venv"
}

$venvPython = Join-Path $Venv "Scripts\python.exe"
$venvActivate = Join-Path $Venv "Scripts\Activate.ps1"

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r requirements.txt

function Install-CommonMLStack {
    & $venvPython -m pip install "imageio[ffmpeg]" scipy soundfile sentence-transformers openai-whisper argostranslate edge-tts kokoro-tts timm
    & $venvPython -m pip install realesrgan basicsr
    & $venvPython -m pip install demucs deepfilternet rnnoise voicefixer
    & $venvPython -m pip install insightface onnxruntime-gpu
    & $venvPython -m pip install f5-tts seed-vc
    try { & $venvPython -m pip install audiocraft } catch { Write-Warn "audiocraft install failed; continuing" }
}

function Install-CudaStack {
    Write-Info "Installing PyTorch with CUDA support..."
    & $venvPython -m pip install torch torchvision torchaudio

    Write-Info "Installing NVIDIA-specific requirements..."
    try { & $venvPython -m pip install -r requirements-nvidia.txt } catch { Write-Warn "Some NVIDIA-specific packages failed; continuing" }

    Write-Info "Installing llama-cpp-python with CUDA support..."
    $env:CMAKE_ARGS = "-DGGML_CUDA=ON"
    try { & $venvPython -m pip install --upgrade llama-cpp-python --no-cache-dir } finally { Remove-Item Env:CMAKE_ARGS -ErrorAction SilentlyContinue }

    Write-Info "Installing stable-diffusion-cpp-python with CUDA support..."
    $env:CMAKE_ARGS = "-DSD_CUDA=ON -DSD_WEBM=OFF"
    try { & $venvPython -m pip install stable-diffusion-cpp-python --no-cache-dir } catch { Write-Warn "stable-diffusion-cpp-python CUDA build failed; continuing" } finally { Remove-Item Env:CMAKE_ARGS -ErrorAction SilentlyContinue }
}

function Install-CpuStack {
    Write-Info "Installing CPU-oriented runtime..."
    & $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    & $venvPython -m pip install --upgrade llama-cpp-python
    try { & $venvPython -m pip install stable-diffusion-cpp-python } catch { Write-Warn "stable-diffusion-cpp-python CPU install failed; continuing" }
    try { & $venvPython -m pip install onnxruntime } catch { Write-Warn "onnxruntime CPU install failed; continuing" }
}

function Invoke-PackageBuild {
    Write-Info "Packaging CoderAI with PyInstaller..."
    & $venvPython -m pip install pyinstaller
    New-Item -ItemType Directory -Force -Path "dist-package" | Out-Null
    & $venvPython -m PyInstaller --clean --noconfirm --onefile --name coderai `
        --collect-all codai `
        --collect-all fastapi `
        --collect-all uvicorn `
        --collect-all pydantic `
        --collect-all transformers `
        --collect-all diffusers `
        --collect-all sentence_transformers `
        --collect-all whispercpp `
        --collect-all insightface `
        --collect-all onnxruntime `
        --collect-all PIL `
        coderai
    Copy-Item "dist\coderai.exe" "dist-package\coderai.exe" -Force
    Write-Ok "Packaged executable: dist-package\coderai.exe"
    Write-Warn "Target systems still need compatible GPU/runtime drivers such as CUDA when GPU backends are used."
}

switch ($Backend) {
    "cuda" {
        Install-CudaStack
        Install-CommonMLStack
    }
    "cpu" {
        Install-CpuStack
        Install-CommonMLStack
    }
    default {
        Install-CudaStack
        Install-CommonMLStack
    }
}

Set-Content -Path ".backend" -Value $Backend
if ($Package) {
    Invoke-PackageBuild
}
Write-Ok "Build completed successfully!"
Write-Host ""
Write-Host "To activate the environment in the future, run:"
Write-Host "  $venvActivate"
Write-Host ""
Write-Host "Recommended runtime notes:"
Write-Host "  - Windows keeps CUDA as the primary NVIDIA acceleration path"
Write-Host "  - Metal is macOS-only and not used on Windows"
