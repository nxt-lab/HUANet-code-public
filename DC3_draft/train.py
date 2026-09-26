"""Training entry point.  Run from the repository root::

    python -m DC3.power_grid.train [--config ...] [--set key=value ...]
"""

from __future__ import annotations

import argparse

from ..common.runner import add_common_args, apply_overrides, load_config, run_training
from .experiment import SPEC


def main():
    ap = argparse.ArgumentParser(description="train DC3 on power_grid")
    add_common_args(ap)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cfg = apply_overrides(load_config(SPEC, a.config), a.set)
    run_training(SPEC, cfg, tag=a.tag, quiet=a.quiet)


if __name__ == "__main__":
    main()
