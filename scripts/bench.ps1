# Baseline, ablation ladder, quantization sweep, degradation test, cache + quality evaluation, report.
# Takes a while (real-time audio). Close other heavy apps first; leave the laptop plugged in.
param([int]$Reps = 3)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = ".\.venv\Scripts\python.exe"
$env:HF_HUB_OFFLINE = "1"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\launch_ollama.ps1 -IfDown
& $py -m bench.make_testset
& $py -m bench.run_ablation --reps $Reps
& $py -m bench.run_quant_sweep
& $py -m bench.run_degrade
& $py -m bench.cache_eval
& $py -m bench.quality_eval
& $py -m bench.make_report
Write-Host "Reports in reports\" -ForegroundColor Green
