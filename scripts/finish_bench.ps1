# Remaining spec benchmarks: 5-limit x 40-request degradation, per-tier quality, quality-profile run, report.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = ".\.venv\Scripts\python.exe"
& $py -m bench.run_degrade --n 40
& $py -m bench.quality_eval
& $py -m bench.run_bench --system FULL --profile quality --subset q40 --reps 1 --warmup-turns 1 --idle-s 3 --tag quality_profile --restart-ollama
& $py -m bench.make_report
