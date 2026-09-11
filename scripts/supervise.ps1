# Keep the robot running unattended: start it, and restart it if it dies.
#
# The robot already survives restarts by design -- safety state, equity peak,
# locks and the trade ledger are all persisted each bar, and any open position
# is re-adopted by its magic number on startup. Stop-loss and take-profit are
# attached to the order itself, so a crash never leaves a position unprotected.
# This script simply makes sure something is always running.
#
#   powershell -ExecutionPolicy Bypass -File scripts\supervise.ps1
#
# Pass -Live to arm real order placement at startup. Without it the robot comes
# up in dry run and must be armed from the dashboard, which is the safer default
# for an unattended machine.

param(
    [switch]$Live,
    [string]$Symbols = 'XAUUSD',
    [double]$Risk = 2.0,
    [int]$Port = 8765,
    [int]$RestartDelaySeconds = 30
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root

$logDir = Join-Path $root 'results\supervisor'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Log($message) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $message
    Write-Host $line
    Add-Content -Path (Join-Path $logDir ("supervisor_{0}.log" -f (Get-Date -Format 'yyyy-MM-dd'))) -Value $line -Encoding utf8
}

# --- credentials -----------------------------------------------------------
$credFile = Join-Path $root 'state\mt5_credentials.xml'
if (Test-Path $credFile) {
    $c = Import-Clixml $credFile
    $env:MT5_LOGIN  = $c.Login
    $env:MT5_SERVER = $c.Server
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR((ConvertTo-SecureString $c.Password))
    $env:MT5_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    Write-Log "credentials loaded for login $($c.Login) on $($c.Server)"
} elseif ($env:MT5_LOGIN -and $env:MT5_PASSWORD -and $env:MT5_SERVER) {
    Write-Log "using credentials from the environment"
} else {
    Write-Log "ERROR: no credentials. Run scripts\save_credentials.ps1 first."
    exit 1
}

# --- make sure the terminal is up -----------------------------------------
$terminal = 'C:\Program Files\MetaTrader 5\terminal64.exe'
if (-not (Get-Process -Name 'terminal64' -ErrorAction SilentlyContinue)) {
    if (Test-Path $terminal) {
        Write-Log "starting MetaTrader 5"
        Start-Process -FilePath $terminal
        Start-Sleep -Seconds 25
    } else {
        Write-Log "WARNING: terminal not found at $terminal"
    }
}

# --- supervise -------------------------------------------------------------
$robotArgs = @('scripts/run_dashboard.py', '--symbols') + $Symbols.Split(',') +
        @('--risk', $Risk, '--port', $Port, '--no-browser')
if ($Live) { $robotArgs += '--live' }

Write-Log "supervising: python $($robotArgs -join ' ')"
Write-Log ("mode: {0}" -f $(if ($Live) { 'LIVE ORDER PLACEMENT' } else { 'DRY RUN (arm from the dashboard)' }))

$restarts = 0
while ($true) {
    $started = Get-Date
    try {
        $p = Start-Process -FilePath 'python' -ArgumentList $robotArgs -NoNewWindow -PassThru `
             -RedirectStandardOutput (Join-Path $logDir 'robot.out') `
             -RedirectStandardError  (Join-Path $logDir 'robot.err')
        $p.WaitForExit()
        $code = $p.ExitCode
    } catch {
        $code = -1
        Write-Log "launch failed: $_"
    }

    $ranFor = [int]((Get-Date) - $started).TotalSeconds
    $restarts++
    Write-Log "robot exited (code $code) after ${ranFor}s -- restart #$restarts in ${RestartDelaySeconds}s"

    # A process that dies immediately is usually misconfigured rather than
    # crashed; backing off avoids hammering the broker with login attempts.
    if ($ranFor -lt 60) {
        Write-Log "exited quickly; backing off to 5 minutes. Check results\supervisor\robot.err"
        Start-Sleep -Seconds 300
    } else {
        Start-Sleep -Seconds $RestartDelaySeconds
    }
}
