"""Setup phase (online): Tamil voices (community Piper voices, Apache-2.0) and multilingual Whisper small. Safe to re-run."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODELS = Path(__file__).resolve().parents[1] / "models"
REPO = "Jeyaram-K/piper-tamil-voices"
VOICES = ["ta_IN-ValluvarNeural-medium", "ta_IN-HemaLatha-medium"]


def main() -> None:
    (MODELS / "tts").mkdir(parents=True, exist_ok=True)
    for v in VOICES:
        for ext in (".onnx", ".onnx.json"):
            src = hf_hub_download(REPO, f"{v}/{v}{ext}")
            (MODELS / "tts" / f"{v}{ext}").write_bytes(Path(src).read_bytes())
    snapshot_download("Systran/faster-whisper-small", local_dir=MODELS / "asr" / "whisper-small")
    print("tamil downloads done")


if __name__ == "__main__":
    main()
