"""R3 gate: CPU-only. Providers, env, no torch, Ollama reports zero VRAM."""
import importlib.util
import os

from pv.guard import gpucheck


def test_env_hides_gpus():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["HIP_VISIBLE_DEVICES"] == "-1"


def test_no_torch_or_gpu_runtimes_installed():
    for mod in ("torch", "torchaudio", "tensorflow", "onnxruntime_gpu"):
        assert importlib.util.find_spec(mod) is None, mod


def test_every_onnx_session_is_cpu_only():
    import numpy as np
    from pv.vad.silero_onnx import SileroVAD
    from pv.wake.oww import WakeWord
    from pv.tts.piper_tts import PiperTTS
    SileroVAD().prob(np.zeros(512, np.int16))
    WakeWord().process(np.zeros(1280, np.int16))
    PiperTTS("en_US-lessac-low", threads=1)
    rec = gpucheck.proof()
    assert rec["pass"] and rec["ort_sessions"]
    assert all(p == ["CPUExecutionProvider"] for p in rec["ort_sessions"].values())
    assert not rec["torch_imported"]


def test_ollama_vram_is_zero(ollama_up):
    from pv.llm.ollama_client import OllamaClient, OllamaOptions
    c = OllamaClient()
    "".join(c.stream([{"role": "user", "content": "Say hi."}], options=OllamaOptions("qwen2.5:0.5b-instruct", 512, 8)))
    rec = gpucheck.proof(ollama_up)
    assert rec["ollama"], "a model must be loaded to be measured"
    assert rec["ollama_vram_zero"] and all(m["size_vram"] == 0 for m in rec["ollama"])
    assert rec["pass"]
