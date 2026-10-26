# Setup phase (internet allowed). Safe to re-run: every step skips what is already done.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# 1. Python 3.12 side by side
$have = (& py -0 2>&1 | Out-String) -match "3\.12"
if (-not $have) {
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
}
py -3.12 --version

# 2. venv + packages (CPU only; no torch)
if (-not (Test-Path .venv\Scripts\python.exe)) { py -3.12 -m venv .venv }
$py = Join-Path $root ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip wheel
& $py -m pip install -r requirements.lock
& $py --version

# 3. Ollama models (our own server serves them from the same model store)
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen2.5:0.5b-instruct

# 4. Other models, pre-rendered answers, manifest, machine probe
& $py scripts\setup_models.py
& $py scripts\download_extra.py     # Whisper small.en + Simple English Wikipedia (online, ~600 MB)
& $py scripts\build_kb.py           # offline knowledge index (needs pyarrow, installed from requirements.lock)
& $py scripts\download_tamil.py     # Tamil voices + Whisper small (multilingual), online, ~600 MB
& $py scripts\render_cache.py
& $py scripts\verify_assets.py
& $py scripts\probe.py | Out-Null
& $py -m bench.make_testset
Write-Host "Setup complete. Run scripts\run.ps1" -ForegroundColor Green
