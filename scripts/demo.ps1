# Demo mode: app + dashboard opened in the browser. Keys: SPACE push-to-talk, T tighten limit, R restore, Q quit.
$root = Split-Path -Parent $PSScriptRoot
& "$root\scripts\run.ps1" -OpenDash @args
