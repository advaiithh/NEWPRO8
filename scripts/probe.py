"""Machine probe: writes reports/machine.json and prints it."""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable: {e}"


def cim(cls: str, props: str) -> str:
    return run(["powershell", "-NoProfile", "-Command",
                f"Get-CimInstance {cls} | Select-Object {props} | ConvertTo-Json -Compress"])


def main() -> None:
    vm = psutil.virtual_memory()
    info = {
        "windows": platform.platform(),
        "cpu": platform.processor(),
        "cpu_name": cim("Win32_Processor", "Name"),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_total_mb": vm.total // 2**20,
        "ram_free_mb": vm.available // 2**20,
        "battery": psutil.sensors_battery() is not None,
        "gpus_informational_only": cim("Win32_VideoController", "Name"),
        "python_launcher": run(["py", "-0"]),
        "ollama": run(["ollama", "--version"]),
    }
    phys, ram = info["physical_cores"], info["ram_total_mb"]
    info["declared_limit"] = {
        "cores": 4 if phys >= 4 else phys,
        "ram_mb": 4096 if ram >= 8192 else ram // 2,
        "reason": "default 4c/4096MB is satisfiable on this machine" if phys >= 4 and ram >= 8192
        else "machine too small for default; halved",
    }
    out = ROOT / "reports" / "machine.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
