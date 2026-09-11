# Store MT5 credentials encrypted at rest, so the robot can start unattended
# without a password sitting in a script or an environment variable.
#
# ConvertFrom-SecureString uses Windows DPAPI: the result can only be decrypted
# by the SAME Windows user on the SAME machine. Copying the file to another
# machine gives an attacker nothing. Re-run this on the VPS after setting it up.
#
#   powershell -ExecutionPolicy Bypass -File scripts\save_credentials.ps1

$ErrorActionPreference = 'Stop'
$stateDir = Join-Path $PSScriptRoot '..\state'
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
$target = Join-Path $stateDir 'mt5_credentials.xml'

Write-Host "MT5 credentials (stored DPAPI-encrypted for $env:USERNAME on $env:COMPUTERNAME)" -ForegroundColor Cyan
$login  = Read-Host "  Login"
$server = Read-Host "  Server (e.g. Deriv-Demo)"
$secure = Read-Host "  Password" -AsSecureString

[pscustomobject]@{
    Login    = $login
    Server   = $server
    Password = (ConvertFrom-SecureString $secure)
} | Export-Clixml -Path $target

Write-Host ""
Write-Host "Saved to $target" -ForegroundColor Green
Write-Host "This file is gitignored and is useless on any other machine or account."
Write-Host "If the account is ever a live-money one, review who can log in as $env:USERNAME."
