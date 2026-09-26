"""AppSpec wiring for the economic-MPC problem."""

from __future__ import annotations

import os

import numpy as np
import torch

from ..common.runner import AppSpec
from . import data as D
from . import reference as R
from .problem import EcoMPCProblem, index_grid

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def build_problem(pc: dict, device, dtype) -> EcoMPCProblem:
    overrides = {k: v for k, v in pc.items()
                 if k not in ("N", "feature_mode", "p_margin", "bound_margin", "p_safe_eps")}
    return EcoMPCProblem(
        N=int(pc.get("N", 96)),
        feature_mode=pc.get("feature_mode", "x0_load_gen"),
        p_margin=float(pc.get("p_margin", 1e-3)),
        bound_margin=float(pc.get("bound_margin", 0.0)),
        p_safe_eps=float(pc.get("p_safe_eps", 1e-9)),
        dtype=dtype, device=device, **overrides,
    )


def partition_other_vars(prob: EcoMPCProblem, pcfg: dict):
    return list(pcfg.get("other_vars", prob.default_other_vars()))


def build_split(dc: dict, split: str, count: int, device, dtype):
    return D.make_split(int(dc["N"]), count, int(dc["seed"]), split, device, dtype,
                        x0_lo=float(dc.get("x0_lo", 0.25)), x0_hi=float(dc.get("x0_hi", 0.75)),
                        split_strategy=dc.get("split_strategy", "temporal"),
                        split_gap=int(dc.get("split_gap", 0)),
                        load_csv=dc.get("load_csv", D.LOAD_CSV), gen_csv=dc.get("gen_csv", D.GEN_CSV))


def reference_solve(params, cfg: dict):
    rc = cfg.get("reference", {})
    x0 = params.x0.detach().cpu().double().numpy()
    load = params.load.detach().cpu().double().numpy()
    gen = params.gen.detach().cpu().double().numpy()
    return R.solve_batch(x0, load, gen, c=build_problem(cfg["problem"], torch.device("cpu"), torch.float64).c, solver=rc.get("solver", "CLARABEL"),
                         tol=float(rc.get("tol", 1e-9)))


def export_instances(params, out_dir: str, cfg: dict) -> str:
    x0 = params.x0.detach().cpu().double().numpy()
    load = params.load.detach().cpu().double().numpy()
    gen = params.gen.detach().cpu().double().numpy()
    path = os.path.join(out_dir, "test_instances.npz")
    offsets = getattr(params, "offsets", None)
    offsets = np.full(x0.shape, -1, dtype=int) if offsets is None else offsets.detach().cpu().numpy()
    D.save_instances(path, x0, load, gen, offsets,
                     {"N": load.shape[1], "count": x0.size, "seed": int(cfg["data"]["seed"])})
    return path


SPEC = AppSpec(
    name="power_grid",
    build_problem=build_problem,
    partition_other_vars=partition_other_vars,
    build_split=build_split,
    index_fn=index_grid,
    reference_solve=reference_solve,
    export_instances=export_instances,
    default_config=os.path.join(CONFIG_DIR, "default.json"),
)
