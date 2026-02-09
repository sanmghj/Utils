@echo off
setlocal EnableExtensions

:: Require admin
net session >nul 2>&1
if not %errorlevel%==0 (
  echo This script must be run as Administrator.
  exit /b 1
)

:: (1) Stop and disable Wi-Fi and Bluetooth services
sc stop WlanSvc >nul 2>&1
sc stop BthServ >nul 2>&1
sc stop BluetoothUserService >nul 2>&1
sc config WlanSvc start= disabled >nul 2>&1
sc config BthServ start= disabled >nul 2>&1

:: (2) Disable Wi-Fi adapter via netsh ONLY
netsh interface set interface "Wi-Fi" admin=disable >nul 2>&1

:: (3) Disable and remove Bluetooth devices + delete their drivers (class=Bluetooth ONLY)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$devs = Get-PnpDevice -Class Bluetooth -ErrorAction SilentlyContinue; foreach($d in $devs) { $id = $d.InstanceId; $inf = (Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_Device_DriverInfPath' -ErrorAction SilentlyContinue).Data; pnputil /disable-device $id 2>&1 | Out-Null; pnputil /remove-device $id 2>&1 | Out-Null; if($inf -and $inf -match 'oem') { pnputil /delete-driver $inf /uninstall /force 2>&1 | Out-Null } }"

:: (4) Disable and remove Wi-Fi device + delete driver (Net class, name=Wi-Fi ONLY)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$devs = Get-PnpDevice -Class Net -FriendlyName '*Wi-Fi*' -ErrorAction SilentlyContinue; foreach($d in $devs) { $id = $d.InstanceId; $inf = (Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_Device_DriverInfPath' -ErrorAction SilentlyContinue).Data; pnputil /disable-device $id 2>&1 | Out-Null; pnputil /remove-device $id 2>&1 | Out-Null; if($inf -and $inf -match 'oem') { pnputil /delete-driver $inf /uninstall /force 2>&1 | Out-Null } }"

:: (5) Registry policies to block driver installation via Windows Update
reg add "HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate" /v ExcludeWUDriversInQualityUpdate /t REG_DWORD /d 1 /f >nul
reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\DriverSearching" /v SearchOrderConfig /t REG_DWORD /d 0 /f >nul
reg add "HKLM\SOFTWARE\Policies\Microsoft\Windows\DeviceInstall\Restrictions" /v DenyUnspecified /t REG_DWORD /d 1 /f >nul

echo Done.
exit /b 0
