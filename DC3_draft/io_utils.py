"""Small I/O / reproducibility helpers shared by both applications."""

from __future__ import annotations

import json
import os
import random
import subprocess
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DC3_ROOT = os.path.join(REPO_ROOT, "DC3")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> torch.device:
    """`auto` prefers CUDA, then Apple MPS, then CPU.

    NOTE: MPS does not support float64.  Callers that run in float64 must pass
    `cpu` explicitly (see `DC3Config.resolve_device`).
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def json_default(obj: Any):
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def save_json(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=json_default)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_csv(path: str, columns: Mapping[str, Any]) -> None:
    """Minimal CSV writer (keeps the repo dependency-free of pandas at runtime).

    Matches the column-per-method layout used by
    `data/MVEE_data/benchmark_results/*.csv`.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    names = list(columns.keys())
    cols = [list(np.asarray(columns[k], dtype=object).reshape(-1)) for k in names]
    nrow = max((len(c) for c in cols), default=0)

    def cell(v):
        if v is None:
            return ""
        if isinstance(v, str):
            return v.replace(",", ";")
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v).replace(",", ";")
        return "" if not np.isfinite(f) else repr(f)

    with open(path, "w") as f:
        f.write(",".join(names) + "\n")
        for i in range(nrow):
            f.write(",".join("" if i >= len(c) else cell(c[i]) for c in cols) + "\n")


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # pragma: no cover - git may be unavailable
        return "unknown"


def environment_report(device: torch.device, dtype: torch.dtype) -> dict:
    import platform

    return {
        "git_revision": git_revision(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "dtype": str(dtype),
        "torch_num_threads": torch.get_num_threads(),
    }
