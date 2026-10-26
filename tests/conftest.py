import subprocess
from pathlib import Path

import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()          # per test: a test that joins the server to its Job Object takes it down on exit
def ollama_up() -> str:
    """Our CPU-only Ollama server on :11435 (started if not already answering)."""
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(ROOT / "scripts" / "launch_ollama.ps1"), "-IfDown"], check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "http://127.0.0.1:11435"


@pytest.fixture(scope="session")
def make_wav(tmp_path_factory):
    """Synthesize a spoken test request with Piper (offline)."""
    from pv.tts.piper_tts import PiperTTS
    tts = PiperTTS("en_US-lessac-medium", threads=2, length_scale=1.0)
    out = tmp_path_factory.mktemp("wavs")

    def make(text: str, name: str) -> Path:
        p = out / f"{name}.wav"
        sf.write(p, tts.synth(text), tts.sample_rate)
        return p

    return make
