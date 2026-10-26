"""Offline + CPU-only + limit proofs (SPEC 13). Prints the evidence and writes reports/proofs.json."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pv.guard import gpucheck, netguard  # noqa: E402
from pv.config import config  # noqa: E402

CAP_CHILD = ("import sys; sys.stdin.readline(); x = bytearray(int(sys.argv[1]) * 2**20); "
             "x[::4096] = b'1' * len(x[::4096]); print('ALLOCATED')")


def limit_proof() -> dict:
    code = f"""
import json, subprocess, sys, psutil
from pv.guard.limiter import Limiter
lim = Limiter(2, 400, include_ollama=False)
res = {{}}
for mb in (100, 900):
    p = subprocess.Popen([sys.executable, "-c", {CAP_CHILD!r}, str(mb)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lim._tree_t = 0; lim.enforce()
    aff = psutil.Process(p.pid).cpu_affinity()
    o, e = p.communicate("go\\n", timeout=60)
    res[str(mb)] = dict(allocated="ALLOCATED" in o, memory_error="MemoryError" in e, affinity=aff)
print(json.dumps(dict(cap_mb=400, banner=lim.banner(), results=res)))
"""
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> None:
    pip = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True).stdout.lower()
    banned = [m for m in ("torch", "torchaudio", "tensorflow", "onnxruntime-gpu", "cuda") if any(l.split()[:1] == [m] for l in pip.splitlines())]
    # exercise the real stack on a loopback request, then read the proofs
    from pv.llm.ollama_client import OllamaClient, OllamaOptions
    netguard.install()
    netguard.reset()
    c = OllamaClient()
    "".join(c.stream([{"role": "user", "content": "Say hello."}], options=OllamaOptions(config["ollama"]["full_model"], 512, 8)))
    import numpy as np
    from pv.vad.silero_onnx import SileroVAD
    from pv.tts.piper_tts import PiperTTS
    SileroVAD().prob(np.zeros(512, np.int16))
    PiperTTS(config["tts"]["voice_low"], threads=1).synth("test")
    gpu = gpucheck.proof(config["ollama"]["url"])
    loop_attempts = netguard.status()["attempts"]
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        blocked = False
    except netguard.OfflineViolation:
        blocked = True
    limit = limit_proof()
    ok = (not banned and gpu["pass"] and loop_attempts == 0 and blocked and
          limit["results"]["100"]["allocated"] and limit["results"]["900"]["memory_error"])
    summary = (f"GPU-free: forbidden packages installed = {banned or 'none'}; ONNX providers = "
               f"{sorted({p for v in gpu['ort_sessions'].values() for p in v})}; Ollama `size_vram` = "
               f"{[m['size_vram'] for m in gpu['ollama']]} (model {gpu['ollama'][0]['model']}). Offline: external network attempts during the "
               f"run = {loop_attempts}; a deliberate connect to 8.8.8.8 was blocked and counted = {blocked} "
               f"(counter now {netguard.status()['attempts']}). Limit: a child inside the 400 MB Job Object allocated 100 MB = "
               f"{limit['results']['100']['allocated']} and was refused 900 MB (MemoryError) = {limit['results']['900']['memory_error']}; "
               f"affinity {limit['results']['100']['affinity']}.")
    out = {"ok": ok, "summary": summary, "gpu": gpu, "banned_packages": banned, "network_attempts_during_run": loop_attempts,
           "external_probe_blocked": blocked, "limit": limit}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "proofs.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))
    print("PROOFS", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
