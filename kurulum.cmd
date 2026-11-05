@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_digitra_ai.ps1"
if errorlevel 1 (
  echo.
  echo Kurulum tamamlanamadi. Yukaridaki hata ayrintisini kontrol edin.
)
pause
