@echo off
setlocal EnableExtensions

:: Require admin
net session >nul 2>&1
if not %errorlevel%==0 (
  echo This script must be run as Administrator.
  exit /b 1
)

set "TASK_NAME=SystemMaintenance"
set "SCRIPT_PATH=%~dp0SystemMaintenance.bat"

:: Create scheduled task - runs at startup with highest privileges
schtasks /create /tn "%TASK_NAME%" /tr "\"%SCRIPT_PATH%\"" /sc onstart /ru SYSTEM /rl highest /f

if %errorlevel%==0 (
  echo.
  echo Task registered successfully.
  echo   Name: %TASK_NAME%
  echo   Path: %SCRIPT_PATH%
  echo   Trigger: At system startup
  echo   Run as: SYSTEM (highest privileges)
) else (
  echo Failed to register task.
)

pause
exit /b 0
