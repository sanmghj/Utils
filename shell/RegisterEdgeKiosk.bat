@echo off
setlocal EnableExtensions

:: Require admin
net session >nul 2>&1
if not %errorlevel%==0 (
  echo This script must be run as Administrator.
  exit /b 1
)

set "TASK_NAME=EdgeKiosk"
set "EDGE_PATH=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

:: ============ CONFIGURATION ============
set "DMAPS_IP=192.168.0.100"
set "DMAPS_USER=dmaps"
:: ========================================

set "KIOSK_URL=http://%DMAPS_IP%/login"
set "USER_DATA_DIR=C:\Users\%DMAPS_USER%\AppData\Local\Microsoft\Edge\User Data"
set "PROFILE_DIR=Default"
set "KIOSK_ARGS=--kiosk "%KIOSK_URL%" --edge-kiosk-type=fullscreen --user-data-dir="%USER_DATA_DIR%" --profile-directory="%PROFILE_DIR%""

:: Check Edge path (try both x86 and x64)
if not exist "%EDGE_PATH%" (
  set "EDGE_PATH=C:\Program Files\Microsoft\Edge\Application\msedge.exe"
)

:: Get current username
for /f "tokens=*" %%u in ('whoami') do set "CURRENT_USER=%%u"

:: Create XML for task with no delay
set "XML_PATH=%TEMP%\EdgeKiosk.xml"

(
echo ^<?xml version="1.0" encoding="UTF-16"?^>
echo ^<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"^>
echo   ^<Triggers^>
echo     ^<LogonTrigger^>
echo       ^<Enabled^>true^</Enabled^>
echo       ^<UserId^>%CURRENT_USER%^</UserId^>
echo     ^</LogonTrigger^>
echo   ^</Triggers^>
echo   ^<Principals^>
echo     ^<Principal id="Author"^>
echo       ^<UserId^>%CURRENT_USER%^</UserId^>
echo       ^<LogonType^>InteractiveToken^</LogonType^>
echo       ^<RunLevel^>LeastPrivilege^</RunLevel^>
echo     ^</Principal^>
echo   ^</Principals^>
echo   ^<Settings^>
echo     ^<MultipleInstancesPolicy^>IgnoreNew^</MultipleInstancesPolicy^>
echo     ^<DisallowStartIfOnBatteries^>false^</DisallowStartIfOnBatteries^>
echo     ^<StopIfGoingOnBatteries^>false^</StopIfGoingOnBatteries^>
echo     ^<AllowHardTerminate^>true^</AllowHardTerminate^>
echo     ^<StartWhenAvailable^>false^</StartWhenAvailable^>
echo     ^<RunOnlyIfNetworkAvailable^>false^</RunOnlyIfNetworkAvailable^>
echo     ^<AllowStartOnDemand^>true^</AllowStartOnDemand^>
echo     ^<Enabled^>true^</Enabled^>
echo     ^<Hidden^>false^</Hidden^>
echo     ^<RunOnlyIfIdle^>false^</RunOnlyIfIdle^>
echo     ^<WakeToRun^>false^</WakeToRun^>
echo     ^<ExecutionTimeLimit^>PT0S^</ExecutionTimeLimit^>
echo     ^<Priority^>5^</Priority^>
echo   ^</Settings^>
echo   ^<Actions Context="Author"^>
echo     ^<Exec^>
echo       ^<Command^>"%EDGE_PATH%"^</Command^>
echo       ^<Arguments^>--kiosk "%KIOSK_URL%" --edge-kiosk-type=fullscreen --user-data-dir="%USER_DATA_DIR%" --profile-directory="%PROFILE_DIR%"^</Arguments^>
echo     ^</Exec^>
echo   ^</Actions^>
echo ^</Task^>
) > "%XML_PATH%"

:: Register task from XML
schtasks /create /tn "%TASK_NAME%" /xml "%XML_PATH%" /f

if %errorlevel%==0 (
  echo.
  echo Task registered successfully.
  echo   Name: %TASK_NAME%
  echo   Trigger: At logon (no delay)
  echo   URL: %KIOSK_URL%
  echo   Profile: %USER_DATA_DIR%\%PROFILE_DIR%
  echo   User: %CURRENT_USER%
  echo.
  echo To change settings, edit DMAPS_IP and DMAPS_USER in this script.
) else (
  echo Failed to register task.
)

del "%XML_PATH%" >nul 2>&1

pause
exit /b 0
