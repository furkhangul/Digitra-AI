param([switch]$NoBrowser)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$apiRoot = Join-Path $projectRoot "apps\api"
$webRoot = Join-Path $projectRoot "apps\web"
$python = Join-Path $apiRoot ".venv-digitra\Scripts\python.exe"
$logRoot = Join-Path $projectRoot "output\runtime-logs"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Digitra API ortamı bulunamadı: $python"
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

function Get-JsonEndpoint {
    param([string]$Uri, [int]$TimeoutSeconds = 3)
    try {
        return Invoke-RestMethod -Uri $Uri -TimeoutSec $TimeoutSeconds
    }
    catch {
        return $null
    }
}

function Test-WebEndpoint {
    param([string]$Uri)
    try {
        $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 3
        return [int]$response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

$health = Get-JsonEndpoint -Uri "http://127.0.0.1:8001/health"
if ($null -eq $health) {
    $apiOut = Join-Path $logRoot "api-$timestamp.out.log"
    $apiErr = Join-Path $logRoot "api-$timestamp.err.log"
    $apiProcess = Start-Process -FilePath $python `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8001") `
        -WorkingDirectory $apiRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $apiOut `
        -RedirectStandardError $apiErr `
        -PassThru

    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if ($apiProcess.HasExited) {
            $details = if (Test-Path -LiteralPath $apiErr) {
                (Get-Content -LiteralPath $apiErr -Tail 30) -join [Environment]::NewLine
            }
            else {
                "API hata günlüğü oluşturulamadı."
            }
            throw "Digitra API başlatılamadı.$([Environment]::NewLine)$details"
        }
        $health = Get-JsonEndpoint -Uri "http://127.0.0.1:8001/health" -TimeoutSeconds 20
        if ($null -ne $health) {
            break
        }
        Start-Sleep -Seconds 1
    }
}

if ($null -eq $health) {
    throw "Digitra API 30 saniye içinde yanıt vermedi. Günlükler: $logRoot"
}
if (-not $health.model.ready) {
    throw "Digitra modeli yüklenemedi: $($health.model.error)"
}
if ($health.model.version -ne "5.0.0") {
    throw "8001 portunda beklenmeyen model çalışıyor: $($health.model.version)"
}

$v6Health = Get-JsonEndpoint -Uri "http://127.0.0.1:8006/health"
if ($null -eq $v6Health) {
    $v6Out = Join-Path $logRoot "v6-$timestamp.out.log"
    $v6Err = Join-Path $logRoot "v6-$timestamp.err.log"
    $v6Process = Start-Process -FilePath $python `
        -ArgumentList @("-m", "uvicorn", "live_api:app", "--app-dir", "ml/training/tid_v6", "--host", "127.0.0.1", "--port", "8006", "--ws-max-size", "1000000") `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $v6Out `
        -RedirectStandardError $v6Err `
        -PassThru
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if ($v6Process.HasExited) {
            throw "V6 tanıma servisi başlatılamadı. Günlük: $v6Err"
        }
        $v6Health = Get-JsonEndpoint -Uri "http://127.0.0.1:8006/health" -TimeoutSeconds 20
        if ($null -ne $v6Health) { break }
        Start-Sleep -Seconds 1
    }
}
if ($null -eq $v6Health -or -not $v6Health.model.ready -or $v6Health.model.version -ne "6.0.0") {
    throw "Çeviri ve canlı eğitim için V6 tanıma servisi hazır değil. Günlükler: $logRoot"
}

if (-not (Test-WebEndpoint -Uri "http://127.0.0.1:3000/")) {
    $webOut = Join-Path $logRoot "web-$timestamp.out.log"
    $webErr = Join-Path $logRoot "web-$timestamp.err.log"
    $webProcess = Start-Process -FilePath "npm.cmd" `
        -ArgumentList @("run", "dev") `
        -WorkingDirectory $webRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $webOut `
        -RedirectStandardError $webErr `
        -PassThru

    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if ($webProcess.HasExited) {
            $details = if (Test-Path -LiteralPath $webErr) {
                (Get-Content -LiteralPath $webErr -Tail 30) -join [Environment]::NewLine
            }
            else {
                "Web hata günlüğü oluşturulamadı."
            }
            throw "Digitra web başlatılamadı.$([Environment]::NewLine)$details"
        }
        if (Test-WebEndpoint -Uri "http://127.0.0.1:3000/") {
            break
        }
        Start-Sleep -Seconds 1
    }
}

if (-not (Test-WebEndpoint -Uri "http://127.0.0.1:3000/")) {
    throw "Digitra web 30 saniye içinde yanıt vermedi. Günlükler: $logRoot"
}

Write-Host "Digitra hazır: TİD v$($v6Health.model.version) · sabit + hareket ($($v6Health.model.device))"
if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:3000/#ceviri"
}
