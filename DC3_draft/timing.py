"""Latency measurement with explicit warm-up and device synchronisation.

Timing boundaries used throughout the benchmark (documented in the READMEs):

* ``predict``      : network forward pass only (features -> partial vars)
* ``complete``     : equality completion only
* ``correct``      : test-time correction loop only
* ``end_to_end``   : features -> predict -> complete -> correct -> final complete
  i.e. exactly what a deployment would have to pay per query.  Moving instance
  parameters onto the device is *excluded* (they are assumed resident), which
  matches how the Julia baselines are timed (``JuMP.solve_time``/`time_ns` around
  the solve only, parameters already built into the model).
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(fn: Callable[[], object], device: torch.device, n_warmup: int = 10, n_repeat: int = 100) -> np.ndarray:
    """Return an array of ``n_repeat`` wall-clock timings in seconds."""
    for _ in range(n_warmup):
        fn()
    sync(device)
    out = np.empty(n_repeat)
    for i in range(n_repeat):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        out[i] = time.perf_counter() - t0
    return out


def summarize(times_s: np.ndarray) -> dict:
    t = np.asarray(times_s, dtype=float) * 1e3   # -> milliseconds (repo convention)
    return {
        "n": int(t.size),
        "mean_ms": float(t.mean()),
        "std_ms": float(t.std(ddof=1)) if t.size > 1 else 0.0,
        "median_ms": float(np.median(t)),
        "p10_ms": float(np.percentile(t, 10)),
        "p90_ms": float(np.percentile(t, 90)),
        "min_ms": float(t.min()),
        "max_ms": float(t.max()),
    }
