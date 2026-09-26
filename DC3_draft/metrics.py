"""Solution quality / feasibility metrics shared by both applications.

Conventions
-----------
* Objective values are reported in the *same* convention as the Julia code
  (`sum w log w` for the cone program; the raw eco-MPC cost for the power grid).
* ``gap_pct`` is ``100 * |J - J_ref| / |J_ref|``, which is what
  `examples/*/benchmark*.jl` and `preprocess.jl` print.  Because that absolute
  value hides the *direction* of the error - and a constraint-violating point
  can easily undercut the true optimum - ``signed_gap_pct`` is reported next to
  it, and all gap aggregates are additionally computed over the feasible subset
  only.
* Feasibility requires constraint residuals within ``tol`` AND exact objective
  domain membership. Undefined objectives/gaps are NaN, never clamped values.
  All-instance means remain undefined if any sample is outside the domain;
  explicitly named domain-valid and feasible subset statistics are separate.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().double().numpy()


def per_instance_metrics(
    problem,
    params: Any,
    Y: torch.Tensor,
    J_ref: Optional[np.ndarray] = None,
    tol: float = 1e-4,
) -> dict:
    with torch.no_grad():
        surrogate_obj = _np(problem.obj_fn(params, Y, safe=True))
        domain_valid = _np(problem.domain_valid(params, Y)).astype(bool)
        obj = _np(problem.obj_fn(params, Y, safe=False))
        domain_valid &= np.isfinite(obj)
        obj = np.where(domain_valid, obj, np.nan)
        eq = problem.eq_resid(params, Y).abs()
        eq_max = _np(eq.amax(dim=1)) if eq.shape[1] else np.zeros(Y.shape[0])
        eq_mean = _np(eq.mean(dim=1)) if eq.shape[1] else np.zeros(Y.shape[0])
        ineq = problem.ineq_dist(params, Y)                # original constraints
        ineq_max = _np(ineq.amax(dim=1))
        ineq_mean = _np(ineq.mean(dim=1))
        n_ineq_viol = _np((ineq > tol).sum(dim=1).double())
        dom = problem.domain_resid(params, Y)
        dom_max = _np(dom.amax(dim=1)) if dom.shape[1] else np.zeros(Y.shape[0])

    constraint_feasible = (eq_max <= tol) & (ineq_max <= tol)
    feasible = constraint_feasible & domain_valid
    out = {
        "obj": obj,
        "surrogate_obj": surrogate_obj,
        "domain_valid": domain_valid.astype(float),
        "constraint_feasible": constraint_feasible.astype(float),
        "eq_max": eq_max,
        "eq_mean": eq_mean,
        "ineq_max": ineq_max,
        "ineq_mean": ineq_mean,
        "n_ineq_viol": n_ineq_viol,
        "domain_max": dom_max,
        "feasible": feasible.astype(float),
    }
    if J_ref is not None:
        J_ref = np.asarray(J_ref, dtype=float)
        denom = np.abs(J_ref)
        denom = np.where(denom < 1e-12, 1.0, denom)
        out["J_ref"] = J_ref
        out["gap_pct"] = 100.0 * np.abs(obj - J_ref) / denom
        out["signed_gap_pct"] = 100.0 * (obj - J_ref) / denom
    return out


def _stats(x: np.ndarray, prefix: str) -> dict:
    if x.size == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("mean", "median", "p90", "max", "min")}
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_p90": float(np.percentile(x, 90)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_min": float(np.min(x)),
    }


def aggregate(m: dict, tol: float = 1e-4, extra_tols=(1e-6, 1e-4, 1e-3, 1e-2)) -> dict:
    n = int(m["obj"].size)
    feas = m["feasible"] > 0.5
    agg = {"n_instances": n, "feas_tol": tol, "feasible_rate": float(feas.mean())}
    valid = m["domain_valid"] > 0.5
    agg["domain_valid_rate"] = float(valid.mean())
    agg["domain_valid_count"] = int(valid.sum())
    agg["constraint_feasible_rate"] = float(m["constraint_feasible"].mean())
    agg.update(_stats(m["obj"], "obj"))
    agg.update(_stats(m["obj"][valid], "obj_domain_valid"))
    agg.update(_stats(m["eq_max"], "eq_max"))
    agg.update(_stats(m["ineq_max"], "ineq_max"))
    agg.update(_stats(m["ineq_mean"], "ineq_mean"))
    agg.update(_stats(m["domain_max"], "domain_max"))
    agg["n_ineq_viol_mean"] = float(np.mean(m["n_ineq_viol"]))
    agg["n_ineq_viol_max"] = float(np.max(m["n_ineq_viol"])) if n else float("nan")
    for t in extra_tols:
        ok = (m["eq_max"] <= t) & (m["ineq_max"] <= t) & valid
        agg[f"feasible_rate@{t:g}"] = float(ok.mean())
    if "gap_pct" in m:
        agg.update(_stats(m["gap_pct"], "gap_pct"))
        agg.update(_stats(m["gap_pct"][valid], "gap_pct_domain_valid"))
        agg.update(_stats(m["signed_gap_pct"], "signed_gap_pct"))
        agg.update(_stats(m["gap_pct"][feas], "gap_pct_feasible"))
        agg.update(_stats(m["signed_gap_pct"][feas], "signed_gap_pct_feasible"))
    return agg


def format_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    """Render a list of dicts as a fixed-width table.

    `columns` is a list of ``(key, header, format)`` triples.
    """
    header = [h for _, h, _ in columns]
    body = []
    for r in rows:
        cells = []
        for k, _, fmt in columns:
            v = r.get(k, None)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("-")
            elif isinstance(v, str):
                cells.append(v)
            else:
                cells.append(format(v, fmt))
        body.append(cells)
    widths = [max(len(header[i]), *(len(b[i]) for b in body)) if body else len(header[i])
              for i in range(len(columns))]
    sep = "  "
    lines = [sep.join(h.ljust(w) for h, w in zip(header, widths)),
             sep.join("-" * w for w in widths)]
    for b in body:
        lines.append(sep.join(c.ljust(w) for c, w in zip(b, widths)))
    return "\n".join(line.rstrip() for line in lines)
