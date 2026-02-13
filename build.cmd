@echo off
cd /d "%~dp0"
echo === Live Captions Build ===
echo.
powershell -ExecutionPolicy Bypass -File ".\build.ps1"
echo.
echo Build bitti. Kapatmak icin bir tusa basin...
pause >nul
