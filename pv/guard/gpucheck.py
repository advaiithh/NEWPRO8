"""CPU-only gate: env vars at import, provider/device assertions, Ollama VRAM check."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["MOONSHINE_ORT_SINGLE_THREAD"] = "1"      # Moonshine's ORT spin-waits badly on pinned cores
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))

import sys
from typing import Any

import httpx

CPU_ONLY = ["CPUExecutionProvider"]
_providers: dict[str, list[str]] = {}


def set_threads(n: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n)


def ort_session(path: str, threads: int = 1):
    """The only way the project creates ONNX sessions: CPU provider, recorded for the proof."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False        # the arena grows with every new input shape (Piper leaked ~1.7 MB per sentence)
    so.enable_mem_pattern = False
    s = ort.InferenceSession(str(path), so, providers=CPU_ONLY)
    got = s.get_providers()
    assert got == CPU_ONLY, f"non-CPU providers for {path}: {got}"
    _providers[os.path.basename(str(path))] = got
    return s


def ollama_vram(url: str) -> list[dict[str, Any]]:
    models = httpx.get(f"{url}/api/ps", timeout=5).json().get("models", [])
    return [{"model": m["name"], "size": m["size"], "size_vram": m["size_vram"]} for m in models]


def proof(ollama_url: str | None = None) -> dict[str, Any]:
    import onnxruntime as ort
    rec: dict[str, Any] = {
        "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")},
        "ort_available_providers": ort.get_available_providers(),
        "ort_sessions": dict(_providers),
        "torch_imported": "torch" in sys.modules,
        "ctranslate2_device": "cpu (explicit)",
    }
    if ollama_url:
        rec["ollama"] = ollama_vram(ollama_url)
        rec["ollama_vram_zero"] = all(m["size_vram"] == 0 for m in rec["ollama"])
    rec["pass"] = (rec["env"]["CUDA_VISIBLE_DEVICES"] == "-1" and not rec["torch_imported"]
                   and all(p == CPU_ONLY for p in _providers.values()) and rec.get("ollama_vram_zero", True))
    return rec
