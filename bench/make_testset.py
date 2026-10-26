"""D11: synthesise the spoken test set offline with Piper (3 voices x 2 speeds, optional noise).

Writes bench/testset/<id>.wav (16 kHz mono) and bench/testset/manifest.json. Real-microphone recordings,
if any, go in bench/testset/real/ (see bench/record_real.py) and are picked up by run_bench --real.
"""
from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from pv.audio.io import SR
from pv.audio.playback import resample
from pv.tts.piper_tts import PiperTTS

HERE = Path(__file__).resolve().parent
VOICES = ["en_US-lessac-medium", "en_GB-alba-medium", "en_US-lessac-low"]
SPEEDS = [1.0, 0.85]                  # Piper length_scale: <1 faster


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p = np.mean(x.astype(np.float64) ** 2) + 1e-9
    noise = rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(x))
    return np.clip(x + noise, -32768, 32767).astype(np.int16)


def load_requests() -> list[dict]:
    reqs = yaml.safe_load((HERE / "requests.yaml").read_text(encoding="utf-8"))
    reqs += [dict(r, heldout=True) for r in yaml.safe_load((HERE / "heldout.yaml").read_text(encoding="utf-8"))]
    return reqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-snr", type=float, default=None, help="add white noise at this SNR (dB) to every file")
    args = ap.parse_args()
    out = HERE / "testset"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    engines = {v: PiperTTS(v, threads=2, length_scale=1.0) for v in VOICES}
    hindi = next((HERE.parent / "models" / "tts").glob("hi_IN-*.onnx")).stem
    engines[hindi] = PiperTTS(hindi, threads=2, length_scale=1.0)
    combos = cycle([(v, s) for s in SPEEDS for v in VOICES])
    manifest = []
    for r in load_requests():
        voice, speed = (hindi, 1.0) if r.get("lang") == "hi" else next(combos)
        eng = engines[voice]
        pcm = resample(eng.synth(r["text"], speed, do_normalize=False), eng.sample_rate, SR)
        if args.noise_snr:
            pcm = add_noise(pcm, args.noise_snr, rng)
        sf.write(out / f"{r['id']}.wav", pcm, SR)
        manifest.append(dict(r, voice=voice, speed=speed, seconds=round(len(pcm) / SR, 2), noise_snr=args.noise_snr))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    total = sum(m["seconds"] for m in manifest)
    print(f"{len(manifest)} requests, {total:.0f} s of audio -> {out}")


if __name__ == "__main__":
    main()
