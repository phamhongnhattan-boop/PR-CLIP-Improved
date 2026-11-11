@echo off
rem Double-click to run the full A3 protocol on this PC's GPU (see RUN_A3_GPU.ps1).
rem Keep this window open and the laptop plugged in; re-run it if it stops: finished runs are skipped.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_A3_GPU.ps1"
echo.
echo Finished (or stopped). Press any key to close.
pause >nul
