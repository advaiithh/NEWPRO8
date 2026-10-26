# OPTIONAL (needs an elevated PowerShell): block all outbound traffic for the venv python and ollama.exe.
# Remove with:  scripts\offline_fw.ps1 -Remove
param([switch]$Remove)
$root = Split-Path -Parent $PSScriptRoot
$targets = @{ "PocketVoice python" = (Resolve-Path "$root\.venv\Scripts\python.exe").Path
              "PocketVoice ollama" = (Get-Command ollama).Source }
foreach ($name in $targets.Keys) {
    Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    if (-not $Remove) {
        New-NetFirewallRule -DisplayName $name -Direction Outbound -Program $targets[$name] -Action Block | Out-Null
        Write-Host "blocked outbound for $($targets[$name])"
    }
}
