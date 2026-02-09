@echo off
setlocal EnableExtensions

:: Require admin
net session >nul 2>&1
if not %errorlevel%==0 (
  echo This script must be run as Administrator.
  exit /b 1
)

set "TASK_NAME=EdgeKiosk"

schtasks /delete /tn "%TASK_NAME%" /f

if %errorlevel%==0 (
  echo Task "%TASK_NAME%" unregistered successfully.
) else (
  echo Task not found or failed to unregister.
)

pause
exit /b 0
