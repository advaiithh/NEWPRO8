"""Check every required model file exists and write models/MANIFEST.json (size + SHA-256)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REQUIRED = [
    "vad/silero_vad.onnx",
    "wakeword/hey_jarvis_v0.1.onnx", "wakeword/melspectrogram.onnx", "wakeword/embedding_model.onnx",
    "asr/whisper-base.en/model.bin", "asr/whisper-tiny.en/model.bin", "asr/whisper-base/model.bin",
    "tts/en_US-lessac-medium.onnx", "tts/en_US-lessac-medium.onnx.json",
    "tts/en_US-lessac-low.onnx", "tts/en_US-lessac-low.onnx.json",
    "tts/en_GB-alba-medium.onnx", "tts/en_GB-alba-medium.onnx.json",
    "embed/potion-base-8M/model.safetensors",
    "asr/whisper-small.en/model.bin", "asr/whisper-small/model.bin",                          # accurate English / multilingual
    "kb/emb.npy", "kb/titles.json", "kb/leads.json",                                          # offline knowledge base
    "tts/ta_IN-ValluvarNeural-medium.onnx", "tts/ta_IN-ValluvarNeural-medium.onnx.json",       # Tamil voice
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    missing = [r for r in REQUIRED if not (MODELS / r).is_file()]
    hindi = list((MODELS / "tts").glob("hi_IN-*.onnx"))
    moon = [p for p in (MODELS / "asr" / "moonshine").rglob("*") if p.is_file()]
    if not hindi:
        missing.append("tts/hi_IN-*.onnx")
    if not moon:
        missing.append("asr/moonshine/**")
    if missing:
        print("MISSING:", *missing, sep="\n  ")
        return 1
    files = sorted(p for p in MODELS.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    manifest = {p.relative_to(MODELS).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files}
    (MODELS / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total = sum(v["bytes"] for v in manifest.values()) / 2**20
    print(f"OK: {len(manifest)} files, {total:.0f} MB, manifest written to models/MANIFEST.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
