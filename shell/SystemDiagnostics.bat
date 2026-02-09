@echo off
setlocal EnableExtensions

:: Require admin
net session >nul 2>&1
if not %errorlevel%==0 (
  echo This script must be run as Administrator.
  exit /b 1
)

echo ============================================
echo  System Diagnostics - Restore Test Setup
echo ============================================
echo.

:: (1) Restore registry to allow driver installation
echo [1/4] Restoring registry settings...
:: Remove driver update exclusion
reg delete "HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate" /v ExcludeWUDriversInQualityUpdate /f >nul 2>&1
:: Enable driver auto search (1 = enabled)
reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\DriverSearching" /v SearchOrderConfig /t REG_DWORD /d 1 /f >nul
:: Remove device install restrictions (delete entire key)
reg delete "HKLM\SOFTWARE\Policies\Microsoft\Windows\DeviceInstall\Restrictions" /f >nul 2>&1
:: Also remove parent key if empty
reg delete "HKLM\SOFTWARE\Policies\Microsoft\Windows\DeviceInstall" /f >nul 2>&1
:: Enable driver installation via Windows Update
reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Device Metadata" /v PreventDeviceMetadataFromNetwork /t REG_DWORD /d 0 /f >nul 2>&1
reg add "HKLM\SOFTWARE\Microsoft\PolicyManager\default\Update\ExcludeWUDriversInQualityUpdate" /v value /t REG_DWORD /d 0 /f >nul 2>&1
echo     Done.

:: (2) Enable and start Wi-Fi and Bluetooth services
echo [2/4] Enabling services...
for %%S in (WlanSvc BthServ BluetoothUserService) do (
  sc config %%S start= demand >nul 2>&1
  sc start %%S >nul 2>&1
)
echo     Done.

:: (3) Enable devices and scan for hardware changes
echo [3/4] Enabling devices and scanning...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-WmiObject Win32_PnPEntity | Where-Object { $_.Name -match 'Bluetooth|Wi.Fi|Wireless|WLAN' -and $_.Name -notmatch 'WAN|Realtek|Virtual|Graphics|UHD|Iris|Arc|Ethernet|LAN|GbE' } | ForEach-Object { pnputil /enable-device \"\"\"$($_.DeviceID)\"\"\" 2>&1 | Out-Null }"
netsh interface set interface "Wi-Fi" admin=enable >nul 2>&1
pnputil /scan-devices >nul 2>&1
echo     Done.

:: (4) Show current status
echo [4/4] Current status:
echo.
echo --- Services ---
for %%S in (WlanSvc BthServ) do (
  sc query %%S 2>nul | findstr "STATE"
)
echo.
echo --- Wi-Fi Devices ---
pnputil /enum-devices /class Net /connected 2>nul | findstr /i "Wi-Fi Wireless Instance"
echo.
echo --- Bluetooth Devices ---
pnputil /enum-devices /class Bluetooth /connected 2>nul | findstr /i "Bluetooth Instance"
echo.
echo ============================================
echo  Test setup complete. Reboot to verify
echo  SystemMaintenance.bat removes devices.
echo ============================================
pause
exit /b 0
