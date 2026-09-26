"""Benchmark entry point.  Run from the repository root::

    python -m DC3.power_grid.benchmark [--config ...] [--set key=value ...]
"""

from __future__ import annotations

import argparse

from ..common.runner import add_common_args, apply_overrides, load_config, run_benchmark
from ..report import write_report
from .experiment import SPEC


def main():
    ap = argparse.ArgumentParser(description="benchmark DC3 on power_grid")
    add_common_args(ap)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--force-reference", action="store_true")
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--n-repeat", type=int, default=50)
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 8, 32, 100])
    a = ap.parse_args()
    cfg = apply_overrides(load_config(SPEC, a.config), a.set)
    run_benchmark(SPEC, cfg, tag=a.tag, checkpoint=a.checkpoint,
                  force_reference=a.force_reference, skip_reference=a.skip_reference,
                  n_repeat=a.n_repeat, n_warmup=a.n_warmup, batch_sizes=tuple(a.batch_sizes))
    write_report(SPEC.name, tag=a.tag)


if __name__ == "__main__":
    main()
