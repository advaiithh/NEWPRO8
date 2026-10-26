"""Compare recognisers on YOUR saved utterances (data/live_wavs): prints each transcript and the time it took.

  python scripts/compare_asr.py [folder]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pv.asr.moonshine_asr import MoonshineASR  # noqa: E402
from pv.asr.whisper_asr import WhisperASR  # noqa: E402
from pv.audio.io import load_wav16k, normalize_gain  # noqa: E402


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "live_wavs"
    files = sorted(folder.glob("*.wav"))[-15:]
    if not files:
        raise SystemExit(f"no wav files in {folder}: talk to the assistant first")
    engines = {"moonshine-small": MoonshineASR("small"), "whisper base.en": WhisperASR("base.en", 4, fast=True),
               "whisper small.en": WhisperASR("small.en", 4, fast=True)}
    for e in engines.values():
        e.warmup()
    for f in files:
        x = load_wav16k(f)
        peak = int(np.abs(x).max())
        print(f"\n{f.name}  ({len(x) / 16000:.1f}s, peak {peak})")
        for name, e in engines.items():
            t = time.perf_counter()
            if isinstance(e, MoonshineASR):
                e.start_stream()
                for i in range(0, len(x) - 512, 512):
                    e.feed(x[i:i + 512])
                text = e.finalize().text
            else:
                text = e.transcribe(normalize_gain(x))[0]
            print(f"  {name:17} {time.perf_counter() - t:5.2f}s  {text}")


if __name__ == "__main__":
    main()
