"""Application-agnostic training and benchmarking drivers.

Each application supplies an :class:`AppSpec` describing how to build its
problem, its variable partition, its instance splits and its reference solver;
`run_training` and `run_benchmark` then do the same thing for both.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch

from .completion import LinearCompletion
from .dc3 import DC3Config, DC3Solver, train_dc3
from .io_utils import DC3_ROOT, environment_report, load_json, save_csv, save_json, set_seed
from .metrics import aggregate, format_table, per_instance_metrics
from .problem import ParametricProblem
from .timing import measure, summarize, sync


@dataclass
class AppSpec:
    name: str
    build_problem: Callable[[dict, torch.device, torch.dtype], ParametricProblem]
    partition_other_vars: Callable[[ParametricProblem, dict], Optional[list]]
    build_split: Callable[[dict, str, int, torch.device, torch.dtype], Any]
    index_fn: Callable[[Any, torch.Tensor], Any]
    reference_solve: Callable[[Any, dict], tuple]   # -> (Y, times_s, J, status)
    export_instances: Callable[[Any, str, dict], str]
    default_config: str


# ---------------------------------------------------------------------------
def load_config(spec: AppSpec, path: Optional[str]) -> dict:
    path = path or spec.default_config
    cfg = load_json(path)
    cfg.setdefault("problem", {})
    cfg.setdefault("data", {})
    cfg.setdefault("partition", {"strategy": "explicit"})
    cfg.setdefault("dc3", {})
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def build(spec: AppSpec, cfg: dict):
    dc3cfg = DC3Config.from_dict(cfg["dc3"])
    device = dc3cfg.resolve_device()
    dtype = dc3cfg.torch_dtype()
    set_seed(dc3cfg.seed)
    problem = spec.build_problem(cfg["problem"], device, dtype)
    strategy = cfg["partition"].get("strategy", "explicit")
    other = spec.partition_other_vars(problem, cfg["partition"]) if strategy == "explicit" else None
    comp = LinearCompletion(problem.A_eq, strategy=strategy, other_vars=other,
                            seed=dc3cfg.seed,
                            cond_warn=float(cfg["partition"].get("cond_warn", 1e10)))
    comp.check(strict=bool(cfg["partition"].get("strict", True)))
    comp.to(device, dtype)
    solver = DC3Solver(problem, comp, dc3cfg).to(device=device, dtype=dtype)
    return problem, comp, solver, dc3cfg, device, dtype


@torch.no_grad()
def _fit_input_norm_chunked(solver, problem, spec, params, n, chunk: int = 64) -> None:
    """Streaming mean/std of the network input.

    Materialising the whole feature matrix at once is fine for the eco-MPC
    (193 columns) but not for the cone program at n=1000, m=100, where one
    feature row is 100_100 numbers and the training set is hundreds of MB.
    """
    device = next(solver.parameters()).device
    total = torch.zeros(problem.x_dim, dtype=torch.float64, device=device)
    total_sq = torch.zeros_like(total)
    for s0 in range(0, n, chunk):
        idx = torch.arange(s0, min(s0 + chunk, n), device=device)
        X = problem.features(spec.index_fn(params, idx)).double()
        total += X.sum(dim=0)
        total_sq += (X * X).sum(dim=0)
    mean = total / n
    var = (total_sq / n - mean * mean).clamp(min=0)
    std = var.sqrt()
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    solver.net.x_mean.copy_(mean.to(solver.net.x_mean.dtype))
    solver.net.x_std.copy_(std.to(solver.net.x_std.dtype))


def results_dir(spec: AppSpec, cfg: dict, tag: str = "") -> str:
    d = os.path.join(DC3_ROOT, "results", spec.name + (f"-{tag}" if tag else ""))
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
def run_training(spec: AppSpec, cfg: dict, tag: str = "", quiet: bool = False) -> dict:
    problem, comp, solver, dc3cfg, device, dtype = build(spec, cfg)
    d = cfg["data"]
    t0 = time.perf_counter()
    train_p = spec.build_split(d, "train", int(d["n_train"]), device, dtype)
    valid_p = spec.build_split(d, "valid", int(d["n_valid"]), device, dtype)
    data_time = time.perf_counter() - t0

    _fit_input_norm_chunked(solver, problem, spec, train_p, len(train_p))
    if dc3cfg.output_init_target and dc3cfg.output_weight_scale == 0:
        probe = spec.index_fn(train_p, torch.arange(min(32, len(train_p)), device=device))
        solver.eval()
        with torch.no_grad():
            if not bool(problem.domain_valid(probe, solver(probe)).all()):
                raise ValueError("Completed initialization leaves the objective domain; check the partition/target")
    n_params = solver.net.n_params()
    print(f"[{spec.name}] {comp.info.summary()}")
    print(f"[{spec.name}] x_dim={problem.x_dim}  n_y={problem.n_y}  n_eq={problem.n_eq}  "
          f"n_ineq={problem.n_ineq}  partial={comp.n_partial}  net params={n_params:,}")
    print(f"[{spec.name}] device={device} dtype={dtype} "
          f"train={len(train_p)} valid={len(valid_p)} (data build {data_time:.1f}s)")

    hist = train_dc3(solver, train_p, valid_p, dc3cfg, spec.index_fn, len(train_p),
                     log_fn=(lambda *_: None) if quiet else print)

    out = results_dir(spec, cfg, tag)
    ckpt = os.path.join(out, "checkpoint.pt")
    torch.save({"state_dict": solver.state_dict(), "config": cfg,
                "partition": comp.info.__dict__, "history": hist}, ckpt)
    save_json(os.path.join(out, "train_history.json"), {
        "config": cfg, "history": hist, "n_params": n_params,
        "partition": comp.info.__dict__,
        "data_build_time_s": data_time,
        "environment": environment_report(device, dtype),
    })
    print(f"[{spec.name}] training time {hist['train_time_s']:.1f}s "
          f"(best epoch {hist['best_epoch']}) -> {ckpt}")
    return {"checkpoint": ckpt, "history": hist, "out_dir": out}


# ---------------------------------------------------------------------------
def _reference_fingerprint(spec, cfg, params):
    """Bind cached optima to exact ordered inputs, formulation, solver and source."""
    digest = hashlib.sha256()
    digest.update(json.dumps({"version": 1, "app": spec.name,
                              "problem": cfg["problem"], "reference": cfg.get("reference", {})},
                             sort_keys=True).encode())
    for name, value in sorted(vars(params).items()):
        digest.update(name.encode())
        if value is None:
            digest.update(b"None")
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    module = inspect.getmodule(spec.reference_solve)
    for source in (module, getattr(module, "R", None)):
        path = getattr(source, "__file__", None)
        if path and Path(path).is_file():
            digest.update(Path(path).read_bytes())
            problem_path = Path(path).with_name("problem.py")
            if problem_path.is_file():
                digest.update(problem_path.read_bytes())
    return digest.hexdigest()


def _reference(spec: AppSpec, cfg: dict, test_p, out_dir: str, force: bool) -> dict:
    path = os.path.join(out_dir, "reference.npz")
    fingerprint = _reference_fingerprint(spec, cfg, test_p)
    n = len(test_p)
    n_y = spec.build_problem(cfg["problem"], torch.device("cpu"), torch.float64).n_y
    if os.path.exists(path) and not force:
        try:
            with np.load(path, allow_pickle=False) as z:
                if ("fingerprint" in z and str(z["fingerprint"].item()) == fingerprint
                        and z["Y"].shape == (n, n_y)
                        and all(z[k].shape == (n,) for k in ("J", "time_s", "status"))):
                    return {"Y": z["Y"], "time_s": z["time_s"], "J": z["J"],
                            "status": z["status"].astype(str).tolist()}
        except (ValueError, KeyError, OSError):
            pass
        print(f"[{spec.name}] reference cache is stale or unverifiable; recomputing")
    print(f"[{spec.name}] solving {n} test instances with the reference solver ...")
    t0 = time.perf_counter()
    Y, times, J, status = spec.reference_solve(test_p, cfg)
    print(f"[{spec.name}] reference solve took {time.perf_counter()-t0:.1f}s "
          f"(median {1e3*np.median(times):.2f} ms/instance)")
    np.savez_compressed(path, Y=Y, time_s=times, J=J, status=np.asarray(status, dtype=str),
                        fingerprint=fingerprint)
    return {"Y": Y, "time_s": times, "J": J, "status": status}


def check_checkpoint_config(checkpoint, cfg, spec):
    """Reject same-shaped but semantically incompatible checkpoints before inference."""
    saved = checkpoint["config"]
    if saved["problem"] != cfg["problem"]:
        raise ValueError("Checkpoint problem differs from evaluation problem; use the checkpoint config or retrain")
    if saved.get("partition", {}) != cfg.get("partition", {}):
        raise ValueError("Checkpoint completion partition differs from evaluation partition")
    old, new = DC3Config.from_dict(saved["dc3"]), DC3Config.from_dict(cfg["dc3"])
    for key in ("hidden_size", "n_hidden", "batch_norm", "input_norm", "output_transform", "use_compl"):
        if getattr(old, key) != getattr(new, key):
            raise ValueError(f"Checkpoint architecture mismatch: {key}")
    if spec.name == "power_grid":
        # Pre-fix checkpoints omitted the strategy and used overlapping offset pools.
        old_data = dict(saved["data"]); new_data = dict(cfg["data"])
        old_data.setdefault("split_strategy", "legacy_offsets")
        new_data.setdefault("split_strategy", "temporal")
        for data in (old_data, new_data):
            for key in ("n_train", "n_valid", "n_test"):
                data.pop(key, None)
            data.setdefault("split_gap", 0)
        if old_data != new_data:
            raise ValueError("MPC checkpoint data protocol differs; an overlapping-split checkpoint cannot "
                             "be relabeled as temporal generalization. Retrain with the new protocol.")


def _dc3_eval(solver, eval_problem, params, eval_params, J_ref, tol,
              index_fn, device, n_warmup=10) -> dict:
    """Score the exact batch=1 outputs whose solve calls are timed, on every instance."""
    first_param = index_fn(params, torch.tensor([0], device=device))
    for _ in range(n_warmup):
        solver.solve(first_param)
    ys, raw, steps, conv, first, times = [], [], [], [], [], []
    for i in range(len(params)):
        pb = index_fn(params, torch.tensor([i], device=device))
        sync(device)
        start = time.perf_counter()
        out = solver.solve(pb)
        sync(device)
        times.append(time.perf_counter() - start)
        ys.append(out["Y"].detach().to(device="cpu", dtype=torch.float64))
        raw.append(out["Y_raw"].detach().to(device="cpu", dtype=torch.float64))
        steps.append(out["steps"])
        conv.append(out["converged"].detach().cpu().numpy())
        first.append(out["first_feasible_step"].detach().cpu().numpy())
    Yc, Yr = torch.cat(ys), torch.cat(raw)
    return {
        "corrected": per_instance_metrics(eval_problem, eval_params, Yc, J_ref, tol),
        "raw": per_instance_metrics(eval_problem, eval_params, Yr, J_ref, tol),
        "corr_steps": np.asarray(steps),
        "corr_converged": np.concatenate(conv).astype(float),
        "first_feasible_step": np.concatenate(first).astype(float),
        "time_s": np.asarray(times),
        "Y": Yc.numpy(),
    }


def _latency(spec: AppSpec, solver: DC3Solver, test_p, device, batch_sizes, n_warmup, n_repeat) -> dict:
    res = {}
    n = len(test_p)
    for bs in batch_sizes:
        if bs > n:
            continue
        idx = torch.arange(bs, device=device)
        pb = spec.index_fn(test_p, idx)
        s = summarize(measure(lambda: solver.solve(pb), device, n_warmup, max(10, n_repeat // 4)))
        s["batch_size"] = bs
        s["per_instance_ms"] = s["median_ms"] / bs
        s["throughput_inst_per_s"] = bs / (s["median_ms"] / 1e3)
        res[f"batch_{bs}"] = s

    # stage breakdown at batch = 1
    p1 = spec.index_fn(test_p, torch.tensor([0], device=device))
    with torch.no_grad():
        res["stage_predict_b1"] = summarize(measure(lambda: solver.predict_partial(p1), device, n_warmup, n_repeat))
        Z1 = solver.predict_partial(p1)
        res["stage_complete_b1"] = summarize(measure(lambda: solver.complete(p1, Z1), device, n_warmup, n_repeat))
    res["stage_correct_b1"] = summarize(measure(lambda: solver.correct_test(p1, Z1), device, n_warmup, n_repeat))
    return res


def run_benchmark(spec: AppSpec, cfg: dict, tag: str = "", checkpoint: Optional[str] = None,
                  force_reference: bool = False, n_repeat: int = 50, n_warmup: int = 10,
                  batch_sizes=(1, 8, 32, 100), skip_reference: bool = False) -> dict:
    problem, comp, solver, dc3cfg, device, dtype = build(spec, cfg)
    out_dir = results_dir(spec, cfg, tag)
    ckpt_path = checkpoint or os.path.join(out_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}; run the training entry point first")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    check_checkpoint_config(ck, cfg, spec)
    solver.load_state_dict(ck["state_dict"])
    solver.eval()
    train_time = float(ck.get("history", {}).get("train_time_s", float("nan")))

    d = cfg["data"]
    test_p = spec.build_split(d, "test", int(d["n_test"]), device, dtype)
    # float64 / CPU twins used exclusively for scoring (see _dc3_eval)
    cpu64 = torch.device("cpu")
    eval_problem = spec.build_problem(cfg["problem"], cpu64, torch.float64)
    eval_params = spec.build_split(d, "test", int(d["n_test"]), cpu64, torch.float64)
    inst_path = spec.export_instances(eval_params, out_dir, cfg)
    print(f"[{spec.name}] test instances -> {inst_path}")

    tol = dc3cfg.feas_tol
    report: dict[str, Any] = {
        "schema_version": 2,
        "evaluation_protocol": "batch=1; quality and latency from the same calls on all test instances",
        "app": spec.name,
        "config": cfg,
        "partition": comp.info.__dict__,
        "environment": environment_report(device, dtype),
        "feas_tol": tol,
        "train_time_s": train_time,
        "n_net_params": solver.net.n_params(),
        "instances_file": inst_path,
    }

    J_ref = None
    if not skip_reference:
        ref = _reference(spec, cfg, eval_params, out_dir, force_reference)
        J_ref = np.asarray(ref["J"], dtype=float)
        report["reference"] = {
            "solver": cfg.get("reference", {}).get("solver", "CLARABEL"),
            "status_counts": {s: int(sum(1 for x in ref["status"] if x == s)) for s in set(ref["status"])},
            "latency_ms": summarize(np.asarray(ref["time_s"])),
            "obj_mean": float(np.nanmean(J_ref)),
        }
        Yref = torch.as_tensor(ref["Y"], dtype=torch.float64, device=cpu64)
        report["reference"]["feasibility"] = aggregate(
            per_instance_metrics(eval_problem, eval_params, Yref, J_ref, tol), tol)

    ev = _dc3_eval(solver, eval_problem, test_p, eval_params, J_ref, tol,
                   spec.index_fn, device, n_warmup)
    report["dc3"] = {
        "corrected": aggregate(ev["corrected"], tol),
        "raw_no_correction": aggregate(ev["raw"], tol),
        "correction": {
            "steps_mean": float(np.mean(ev["corr_steps"])),
            "steps_max": int(np.max(ev["corr_steps"])),
            "max_steps_allowed": dc3cfg.corr_test_max_steps,
            "converged_rate": float(np.mean(ev["corr_converged"])),
            "correction_failures": int(np.sum(ev["corr_converged"] < 0.5)),
            "first_feasible_step_mean": float(np.mean(ev["first_feasible_step"][ev["first_feasible_step"] >= 0]))
            if np.any(ev["first_feasible_step"] >= 0) else float("nan"),
            "first_feasible_step_max": float(np.max(ev["first_feasible_step"])),
            "note": "Per-instance correction statistics from the timed batch=1 solves.",
        },
    }

    print(f"[{spec.name}] measuring inference latency ...")
    report["dc3"]["latency"] = _latency(spec, solver, test_p, device, batch_sizes, n_warmup, n_repeat)

    report["dc3"]["latency"]["single_instance"] = summarize(ev["time_s"])
    report["dc3"]["latency"]["single_instance"]["note"] = report["evaluation_protocol"]

    # per-instance dump for downstream plots / tables
    cols = {f"dc3_{k}": v for k, v in ev["corrected"].items() if isinstance(v, np.ndarray)}
    cols["dc3_corr_converged"] = ev["corr_converged"]
    cols["dc3_corr_steps"] = ev["corr_steps"]
    cols["dc3_time_ms"] = 1e3 * ev["time_s"]
    if J_ref is not None:
        cols["J_ref"] = J_ref
    save_csv(os.path.join(out_dir, "per_instance.csv"), cols)
    np.savez_compressed(os.path.join(out_dir, "dc3_solutions.npz"), Y=ev["Y"])

    save_json(os.path.join(out_dir, "benchmark.json"), report)
    print(f"[{spec.name}] wrote {os.path.join(out_dir, 'benchmark.json')}")
    return report


# ---------------------------------------------------------------------------
def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--set", type=str, nargs="*", default=[],
                    help="config overrides, e.g. --set dc3.epochs=5 problem.n=100")


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for ov in overrides:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            node[parts[-1]] = json.loads(val)
        except json.JSONDecodeError:
            node[parts[-1]] = val
    return cfg
