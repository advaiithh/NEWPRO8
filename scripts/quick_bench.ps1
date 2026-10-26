# Shortened benchmark: ladder A2-A7 (2 reps; A0 and A1 reused if present), reduced degradation test, cache eval, report.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = ".\.venv\Scripts\python.exe"
& $py -m bench.run_ablation --reps 2
& $py -m bench.run_degrade --n 20 --steps "4:4096,2:2048,2:1536"
& $py -m bench.cache_eval
& $py -m bench.make_report
