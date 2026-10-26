"""Single global config (the only module-level mutable state)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def load(profile: str | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    profile = profile or os.environ.get("PV_PROFILE")
    if profile:
        cfg = _merge(cfg, yaml.safe_load((ROOT / "config" / "profiles" / f"{profile}.yaml").read_text(encoding="utf-8")))
        cfg["profile"] = profile
    return cfg


config: dict[str, Any] = load()


def reload(profile: str | None) -> None:
    config.clear()
    config.update(load(profile))


def path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def model(*parts: str) -> Path:
    return ROOT / config["paths"]["models"] / Path(*parts)
