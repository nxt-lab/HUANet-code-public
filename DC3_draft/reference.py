"""Reference solver for the economic-MPC instance family.

The cost has three non-smooth convex pieces; the ``a/p - 1`` term makes the
problem an SOCP (``a/p <= t`` is the rotated cone ``p*t >= a, p > 0``), which is
exactly what the JuMP model hands to Ipopt.  In cvxpy the same epigraph form is
DCP as ``cp.pos(a * cp.inv_pos(p) - 1)``.

The Julia/Ipopt reference on the *same* saved instances is produced by
``DC3/julia/reference_power.jl``; the benchmark cross-checks the two.
"""

from __future__ import annotations

import time

import numpy as np

from .problem import DEFAULTS


def solve_instance(x0: float, load: np.ndarray, gen: np.ndarray,
                   c: dict | None = None, solver: str = "CLARABEL",
                   tol: float = 1e-9, verbose: bool = False):
    import cvxpy as cp

    c = dict(DEFAULTS) if c is None else c
    N = load.size
    Ad, Bd = c["A"], -c["dT"] / c["BESS"]
    kappa = (1.0 - c["eta"]) / (2.0 * np.sqrt(c["eta"]))

    m = cp.Variable(N)
    u = cp.Variable(N)
    p = cp.Variable(N, pos=True)
    x = cp.Variable(N + 1)

    cons = [
        x[0] == x0,
        x[N] == x0,
        x[1:] == Ad * x[:-1] + Bd * u,
        m + u + gen - load - p == 0,
        u >= c["u_min"], u <= c["u_max"],
        p >= 0,
        x >= c["x_min"], x <= c["x_max"],
    ]
    obj = cp.Minimize(
        cp.sum(c["r_ec"] * c["dT"] * (m + kappa * cp.abs(u))
               + c["r_op"] * cp.pos(m)
               + c["r_df"] * cp.pos(c["a"] * cp.inv_pos(p) - 1.0))
    )
    prob = cp.Problem(obj, cons)
    kwargs = {}
    if solver == "CLARABEL":
        kwargs = dict(tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
    t0 = time.perf_counter()
    prob.solve(solver=solver, verbose=verbose, **kwargs)
    t = time.perf_counter() - t0
    if m.value is None:
        return None, t, float("nan"), prob.status
    y = np.concatenate([m.value, u.value, p.value, x.value[1:]])
    J = objective_numpy(y, N, c)
    return y, t, float(J), prob.status


def objective_numpy(y: np.ndarray, N: int, c: dict | None = None) -> float:
    c = dict(DEFAULTS) if c is None else c
    kappa = (1.0 - c["eta"]) / (2.0 * np.sqrt(c["eta"]))
    m, u, p = y[:N], y[N:2 * N], y[2 * N:3 * N]
    return float(np.sum(
        c["r_ec"] * c["dT"] * (m + kappa * np.abs(u))
        + c["r_op"] * np.maximum(m, 0.0)
        + c["r_df"] * np.maximum(c["a"] / np.clip(p, 1e-12, None) - 1.0, 0.0)
    ))


def solve_batch(x0: np.ndarray, load: np.ndarray, gen: np.ndarray, **kw):
    sols, times, objs, status = [], [], [], []
    N = load.shape[1]
    for i in range(x0.size):
        y, t, J, st = solve_instance(float(x0[i]), load[i], gen[i], **kw)
        sols.append(np.full(4 * N, np.nan) if y is None else y)
        times.append(t)
        objs.append(J)
        status.append(st)
    return np.array(sols), np.array(times), np.array(objs), status
