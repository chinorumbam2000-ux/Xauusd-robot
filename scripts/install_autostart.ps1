# Register the supervisor as a Windows scheduled task so the robot comes back
# after a reboot without anyone logging in and clicking anything.
#
# Run from an ELEVATED PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
#
# Remove it again with:
#   Unregister-ScheduledTask -TaskName XauusdRobot -Confirm:$false
#
# The task runs in DRY RUN unless -Live is passed. On an unattended machine that
# is the safer default: the robot starts, analyses, and waits to be armed from
# the dashboard rather than placing orders the moment the box boots.

param(
    [switch]$Live,
    [string]$TaskName = 'XauusdRobot',
    [string]$Symbols = 'XAUUSD',
    [double]$Risk = 2.0
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$supervisor = Join-Path $root 'scripts\supervise.ps1'

if (-not (Test-Path (Join-Path $root 'state\mt5_credentials.xml'))) {
    Write-Warning "No saved credentials. Run scripts\save_credentials.ps1 first, as the"
    Write-Warning "user this task will run as -- DPAPI ties the file to that account."
}

$argLine = "-ExecutionPolicy Bypass -NoProfile -File `"$supervisor`" -Symbols $Symbols -Risk $Risk"
if ($Live) { $argLine += ' -Live' }

$action   = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argLine -WorkingDirectory $root
# At startup rather than at logon, so an unattended VPS recovers from a reboot
# without anyone signing in.
$trigger  = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
            -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "Registered scheduled task '$TaskName'" -ForegroundColor Green
Write-Host "  runs   : $argLine"
Write-Host "  mode   : $(if ($Live) { 'LIVE ORDER PLACEMENT' } else { 'DRY RUN -- arm from the dashboard' })"
Write-Host ""
Write-Host "Start it now with : Start-ScheduledTask -TaskName $TaskName"
Write-Host "Check status with : Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "Dashboard         : http://127.0.0.1:8765 (on that machine)"
