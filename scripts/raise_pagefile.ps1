# OPTIONAL, needs an ELEVATED PowerShell (Run as administrator) and a reboot afterwards.
# Raises the Windows page file so the larger models (qwen3.5:4b needs ~5 GB, plus ~2 GB for the speech stack) can load while
# browsers etc. stay open. It does not make the CPU faster; it stops "out of memory" failures and the "low memory mode" fallbacks.
param([int]$InitialMB = 16384, [int]$MaximumMB = 32768)
$ErrorActionPreference = "Stop"
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { throw "Run this in an elevated PowerShell (right-click PowerShell > Run as administrator)." }
$free = [int]((Get-PSDrive C).Free / 1GB)
if ($free -lt ($MaximumMB / 1024 + 10)) { throw "C: has only $free GB free; need at least $([int]($MaximumMB/1024) + 10) GB." }

$cs = Get-CimInstance Win32_ComputerSystem
if ($cs.AutomaticManagedPagefile) { $cs | Set-CimInstance -Property @{ AutomaticManagedPagefile = $false } }
$pf = Get-CimInstance Win32_PageFileSetting | Where-Object { $_.Name -like "C:*" }
if ($pf) { $pf | Set-CimInstance -Property @{ InitialSize = $InitialMB; MaximumSize = $MaximumMB } }
else { New-CimInstance -ClassName Win32_PageFileSetting -Property @{ Name = "C:\pagefile.sys"; InitialSize = $InitialMB; MaximumSize = $MaximumMB } | Out-Null }
Write-Host "Page file on C: set to $InitialMB-$MaximumMB MB. REBOOT for it to take effect." -ForegroundColor Green
