"""Setup: pre-render every cached answer to cache/audio/<lang>/<id>_<k>.wav (Piper, offline)."""
from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pv.config import config  # noqa: E402
from pv.router.cache import INTENTS, INTENTS_TA  # noqa: E402
from pv.tts.piper_tts import PiperTTS  # noqa: E402


def main() -> None:
    hindi = next((ROOT / "models" / "tts").glob("hi_IN-*.onnx")).stem
    voices = {"en": PiperTTS(config["tts"]["voice_full"], threads=2), "hi": PiperTTS(hindi, threads=2),
              "ta": PiperTTS(config["tts"]["voice_ta"], threads=2)}
    n = 0
    for it in [i for f in (INTENTS, INTENTS_TA) for i in yaml.safe_load(f.read_text(encoding="utf-8"))]:
        tts = voices[it["lang"]]
        out = ROOT / config["paths"]["cache"] / "audio" / it["lang"]
        out.mkdir(parents=True, exist_ok=True)
        for k, text in enumerate(it["a"]):
            sf.write(out / f"{it['id']}_{k}.wav", tts.synth(text, do_normalize=it["lang"] == "en"), tts.sample_rate)
            n += 1
    print(f"rendered {n} answers")


if __name__ == "__main__":
    main()
