@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_digitra_ai.ps1"
if errorlevel 1 (
  echo.
  echo Digitra baslatilamadi. Yukaridaki hata ayrintisini kontrol edin.
  pause
)
