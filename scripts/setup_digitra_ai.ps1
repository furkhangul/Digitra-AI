$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$apiRoot = Join-Path $projectRoot "apps\api"
$webRoot = Join-Path $projectRoot "apps\web"
$venv = Join-Path $apiRoot ".venv-digitra"
$python = Join-Path $venv "Scripts\python.exe"

function Invoke-Checked {
    param([string]$FilePath, [string[]]$Arguments)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Komut başarısız oldu: $FilePath $($Arguments -join ' ')"
    }
}

Write-Host "== Digitra kurulumu =="

# 1) Python 3.12
if (-not (Test-Path -LiteralPath $python)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python 3.12 bulunamadı. https://www.python.org/downloads/ adresinden Python 3.12 kurun (kurulumda 'py launcher' seçili olsun)."
    }
    Write-Host "Python ortamı oluşturuluyor..."
    Invoke-Checked "py" @("-3.12", "-m", "venv", $venv)
}

Write-Host "Python paketleri kuruluyor (birkaç dakika sürebilir)..."
Invoke-Checked $python @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $python @("-m", "pip", "install", "-r", (Join-Path $apiRoot "requirements.txt"))

# 2) NVIDIA ekran kartı varsa CUDA'lı PyTorch
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "NVIDIA GPU bulundu, CUDA'lı PyTorch kuruluyor..."
    Invoke-Checked $python @("-m", "pip", "install", "--force-reinstall", "--no-deps", "-r", (Join-Path $apiRoot "requirements-cuda.txt"))
}
else {
    Write-Host "NVIDIA GPU bulunamadı, model CPU üzerinde çalışacak."
}

# 3) API ayar dosyası
$envFile = Join-Path $apiRoot ".env"
if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath (Join-Path $apiRoot ".env.example") -Destination $envFile
}

# 4) Web bağımlılıkları
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "Node.js bulunamadı. https://nodejs.org adresinden Node.js (LTS) kurun."
}
Write-Host "Web paketleri kuruluyor..."
Push-Location $webRoot
try {
    Invoke-Checked "npm.cmd" @("ci")
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Kurulum tamamlandı. Başlatmak için run_digitra_ai.cmd dosyasına çift tıklayın."
