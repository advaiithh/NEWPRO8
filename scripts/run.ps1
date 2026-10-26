# Runtime (offline). Starts the CPU-only Ollama, then the app (limiter, dashboard, governor) in this console.
param([string]$Profile = "quality", [string]$Mode = "always", [switch]$OpenDash, [string[]]$Extra = @())
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"; $env:CUDA_VISIBLE_DEVICES = "-1"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\launch_ollama.ps1 -IfDown
$argsList = @("-m", "pv.app", "--mode", $Mode)
if ($Profile) { $argsList += @("--profile", $Profile) }
if ($OpenDash) { $argsList += "--open-dash" }
& .\.venv\Scripts\python.exe @argsList @Extra
