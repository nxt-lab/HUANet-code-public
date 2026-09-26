"""Instance generation for the economic-MPC benchmark.

`energy_mag()` in ``examples/power_grid/power_system.jl`` defines a *single*
instance: ``x0 = 0.5`` and the first ``N = 96`` samples of

    data/micro_grid/PV_48h_15-min_150kW_San_Diego.csv        (column 2, kW)
    data/micro_grid/load_15min_max100kW_SanDiego_Building.csv (column 2, kW)

DC3 is a *parametric* solver, so a family of instances is required.  The family
used here keeps the plant and the cost function untouched and varies only the
quantities that the JuMP model already exposes as ``MOI.Parameter``:

    x0           ~ U(x0_lo, x0_hi)          default [0.25, 0.75]
    (load, gen)  = the length-N window of the CSVs starting at offset s,
                   s ~ Uniform{0, ..., 192 - N}

New experiments split raw time blocks before constructing windows, preserving
chronological order and optionally leaving a gap. They require aligned CSV
timestamps and enough data for all three horizons. The bundled positional
pairing is retained only under explicit ``legacy_offsets`` for historical
interpolation experiments (including nominal test instance 0).

The ``x0`` range brackets the pool used for the ADMM training data in
``data_eMPC_power.jl`` (``train_pool = [1/2, 2/3, 3/4]``, ``test_pool = [3/5]``).
"""

from __future__ import annotations

import csv
import os
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from ..common.io_utils import REPO_ROOT
from .problem import GridParams

GEN_CSV = os.path.join(REPO_ROOT, "data", "micro_grid", "PV_48h_15-min_150kW_San_Diego.csv")
LOAD_CSV = os.path.join(REPO_ROOT, "data", "micro_grid", "load_15min_max100kW_SanDiego_Building.csv")
SPLIT_ID = {"train": 0, "valid": 1, "test": 2}


def read_series(load_csv=LOAD_CSV, gen_csv=GEN_CSV, require_aligned=False) -> tuple[np.ndarray, np.ndarray]:
    """Second column of each CSV, exactly as ``CSV.read(...)[:, 2]`` in Julia."""
    def col2(path):
        with open(path) as f:
            rows = list(csv.reader(f))
        return np.array([float(r[1]) for r in rows[1:]], dtype=float), [r[0] for r in rows[1:]]

    (load, load_times), (gen, gen_times) = col2(load_csv), col2(gen_csv)
    if require_aligned and load_times != gen_times:
        raise ValueError("Temporal data requires aligned load/PV timestamps; CSV timestamps differ. "
                         "Supply longer aligned CSV files. Legacy positional pairing is interpolation only.")
    if load.size != gen.size or not (np.isfinite(load).all() and np.isfinite(gen).all()):
        raise ValueError("Load/PV series must have equal lengths, finite values, and aligned timestamps")
    return load, gen


def offset_grid(N: int, n_samples: int) -> np.ndarray:
    return np.arange(0, n_samples - N + 1, dtype=int)


def split_offsets(N: int, n_samples: int, seed: int,
                  strategy: str = "temporal", gap: int = 0) -> dict[str, np.ndarray]:
    """Split raw time blocks before windowing; no timestamps cross splits.

    Each block reserves N samples; remaining samples are allocated 60/20/20.
    Optional gaps separate blocks. Legacy offset splitting is interpolation only.
    """
    if N <= 0 or gap < 0:
        raise ValueError("N must be positive and split_gap nonnegative")
    if strategy == "temporal":
        required = 3 * N + 2 * gap
        if n_samples < required:
            raise ValueError(f"Temporal split needs at least {required} aligned samples for N={N}; "
                             f"only {n_samples} available. Supply longer load_csv/gen_csv files, "
                             "or explicitly select a shorter horizon. legacy_offsets is interpolation only.")
        extra = n_samples - required
        lengths = [N + int(.6 * extra), N + int(.2 * extra)]
        lengths.append(n_samples - 2 * gap - sum(lengths))
        start = 0
        pools = {}
        for split, length in zip(SPLIT_ID, lengths):
            pools[split] = np.arange(start, start + length - N + 1, dtype=int)
            start += length + gap
        return pools
    if strategy != "legacy_offsets":
        raise ValueError(f"Unknown split strategy: {strategy}")
    warnings.warn("legacy_offsets shares timestamps across splits: interpolation only, "
                  "not independent temporal generalization", UserWarning, stacklevel=2)
    offs = offset_grid(N, n_samples)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(offs[offs != 0])
    n = perm.size
    n_tr = int(round(0.6 * n))
    n_va = int(round(0.2 * n))
    return {
        "train": np.sort(perm[:n_tr]),
        "valid": np.sort(perm[n_tr : n_tr + n_va]),
        "test": np.sort(np.concatenate([[0], perm[n_tr + n_va :]])),
    }


def generate_numpy(
    N: int, count: int, seed: int, split: str,
    x0_lo: float = 0.25, x0_hi: float = 0.75,
    split_strategy: str = "temporal", split_gap: int = 0,
    load_csv: str = LOAD_CSV, gen_csv: str = GEN_CSV,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x0, load, gen, offset)`` for `count` instances of `split`."""
    load_all, gen_all = read_series(load_csv, gen_csv, require_aligned=split_strategy == "temporal")
    n_samples = min(load_all.size, gen_all.size)
    pools = split_offsets(N, n_samples, seed, split_strategy, split_gap)
    pool = pools[split]
    if pool.size == 0:
        raise RuntimeError(f"empty offset pool for split {split!r} (N={N}, samples={n_samples})")

    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=(SPLIT_ID[split], 7))
    rng = np.random.default_rng(ss)
    offs = rng.choice(pool, size=count, replace=True)
    x0 = rng.uniform(x0_lo, x0_hi, size=count)
    if split == "test" and split_strategy == "legacy_offsets":
        offs[0], x0[0] = 0, 0.5            # nominal `energy_mag()` instance

    load = np.stack([load_all[s : s + N] for s in offs])
    gen = np.stack([gen_all[s : s + N] for s in offs])
    return x0, load, gen, offs


def to_params(x0, load, gen, device, dtype) -> GridParams:
    return GridParams(
        x0=torch.as_tensor(x0, dtype=dtype, device=device),
        load=torch.as_tensor(load, dtype=dtype, device=device),
        gen=torch.as_tensor(gen, dtype=dtype, device=device),
    )


def make_split(N, count, seed, split, device, dtype, **kw) -> GridParams:
    x0, load, gen, offsets = generate_numpy(N, count, seed, split, **kw)
    params = to_params(x0, load, gen, device, dtype)
    params.offsets = torch.as_tensor(offsets, dtype=torch.long, device=device)
    return params


def instances_path(root: str, N: int, count: int, seed: int, split: str = "test") -> str:
    return os.path.join(root, f"power_{split}_N={N}_n={count}_seed={seed}.npz")


def save_instances(path, x0, load, gen, offs, meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, x0=x0, load=load, gen=gen, offset=offs,
                        **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})


def load_instances(path):
    d = np.load(path)
    return d["x0"], d["load"], d["gen"], d["offset"]
