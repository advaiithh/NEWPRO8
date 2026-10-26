"""Setup phase (online): download every model into models/. Safe to re-run."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SILERO_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
PIPER_VOICES = ["en_US-lessac-medium", "en_US-lessac-low", "en_GB-alba-medium"]
WHISPER = {"tiny.en": "Systran/faster-whisper-tiny.en", "base.en": "Systran/faster-whisper-base.en",
           "base": "Systran/faster-whisper-base"}
EMBED = "minishlab/potion-base-8M"


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("download", url)
    urllib.request.urlretrieve(url, dest)


def hindi_voices() -> list[str]:
    from piper.download_voices import VOICES_JSON
    with urllib.request.urlopen(VOICES_JSON) as r:
        voices = json.load(r)
    hi = sorted(v for v in voices if v.startswith("hi_IN"))
    print("hindi voices available:", hi)
    return hi[:1]


def main() -> int:
    from huggingface_hub import snapshot_download
    from piper.download_voices import download_voice

    fetch(SILERO_URL, MODELS / "vad" / "silero_vad.onnx")

    import openwakeword.utils as owu
    owu.download_models(["hey_jarvis"], target_directory=str(MODELS / "wakeword"))

    for name, repo in WHISPER.items():
        snapshot_download(repo, local_dir=MODELS / "asr" / f"whisper-{name}")

    import moonshine_voice.download as md
    from moonshine_voice.moonshine_api import ModelArch
    for lang, arch in [("en", ModelArch.TINY_STREAMING), ("en", ModelArch.SMALL_STREAMING)]:
        path, _ = md.get_model_for_language(lang, arch, cache_root=MODELS / "asr" / "moonshine")
        print("moonshine", lang, arch.name, path)

    (MODELS / "tts").mkdir(parents=True, exist_ok=True)
    for voice in PIPER_VOICES + hindi_voices():
        download_voice(voice, MODELS / "tts")

    snapshot_download(EMBED, local_dir=MODELS / "embed" / "potion-base-8M")
    print("setup_models done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
