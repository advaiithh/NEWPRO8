# Starts OUR CPU-only Ollama server on 127.0.0.1:11435 (the tray instance on 11434 is left alone).
param(
    [string]$KvCache = "q8_0",
    [string]$FlashAttention = "1",
    [int]$MaxLoaded = 1,
    [switch]$IfDown      # do nothing when our server already answers
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root "data\ollama.pid"
if ($IfDown) {
    try { Invoke-RestMethod "http://127.0.0.1:11435/api/version" | Out-Null; Write-Host "Ollama already up"; return } catch {}
}
New-Item -ItemType Directory -Force (Split-Path $pidFile) | Out-Null

if (Test-Path $pidFile) {
    $old = [int](Get-Content $pidFile)
    Stop-Process -Id $old -Force -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$old" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Remove-Item $pidFile
}
# Any leftover server already bound to our port is ours from an earlier run.
$busy = Get-NetTCPConnection -LocalPort 11435 -State Listen -ErrorAction SilentlyContinue
if ($busy) { Stop-Process -Id $busy.OwningProcess -Force; Start-Sleep 1 }

$env:OLLAMA_HOST = "127.0.0.1:11435"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:HIP_VISIBLE_DEVICES = "-1"
$env:GGML_VK_VISIBLE_DEVICES = "-1"
$env:OLLAMA_LLM_LIBRARY = "cpu"
$env:OLLAMA_VULKAN = "0"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "$MaxLoaded"
$env:OLLAMA_KEEP_ALIVE = "-1"
$env:OLLAMA_FLASH_ATTENTION = $FlashAttention
$env:OLLAMA_KV_CACHE_TYPE = $KvCache
$env:OLLAMA_NO_CLOUD = "1"

$log = Join-Path $root "data\ollama.log"
$p = Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput $log -RedirectStandardError "$log.err"
$p.Id | Set-Content $pidFile

for ($i = 0; $i -lt 40; $i++) {
    try { Invoke-RestMethod "http://127.0.0.1:11435/api/version" | Out-Null; break } catch { Start-Sleep -Milliseconds 500 }
}
Write-Host ("Ollama CPU-only server up on 127.0.0.1:11435 (pid {0}) version {1}" -f $p.Id, (Invoke-RestMethod "http://127.0.0.1:11435/api/version").version)
