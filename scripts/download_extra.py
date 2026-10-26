"""Setup phase (online): extra models for accuracy + knowledge. Safe to re-run."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODELS = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    snapshot_download("Systran/faster-whisper-small.en", local_dir=MODELS / "asr" / "whisper-small.en")
    hf_hub_download("wikimedia/wikipedia", "20231101.simple/train-00000-of-00001.parquet", repo_type="dataset",
                    local_dir=MODELS / "kb" / "raw")
    print("extra downloads done")


if __name__ == "__main__":
    main()
