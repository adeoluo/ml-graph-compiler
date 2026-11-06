"""Shared paths and helpers for the benchmark scripts."""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MGC_DATA", ROOT.parent / ".bench-data" / "ml-graph-compiler")).resolve()
MODELS = DATA / "models"
RESULTS = ROOT / "benchmarks" / "results"

# name -> (file, symbolic dims) for the four exported models
MODEL_FILES = {
    "resnet50": "resnet50.onnx",
    "mobilenetv2": "mobilenetv2.onnx",
    "bert-base": "bert-base-uncased.onnx",
    "gpt2": "gpt2.onnx",
}


def model_path(name: str) -> Path:
    return MODELS / MODEL_FILES[name]


def machine() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpus": os.cpu_count(),
        "load_average_1m": round(os.getloadavg()[0], 2),
    }
    try:
        info["cpu"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        info["memory_gb"] = round(
            int(
                subprocess.run(
                    ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
                ).stdout.strip()
                or 0
            )
            / 2**30,
            1,
        )
    except OSError:
        pass
    return info


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def write_result(name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"machine": machine(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
