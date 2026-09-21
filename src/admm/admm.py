"""ADMM updates for problems with a nonnegative slack variable."""

import time
from collections.abc import Callable

import numpy as np


def admm(
    solve_primal: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, float]],
    n_ineq: int,
    max_iter: int,
    tol: float,
    rho: float,
) -> tuple[np.ndarray, float]:
    """Solve one instance; solve_primal(q) returns x, s, and solver time."""
    w = np.zeros(n_ineq)
    v = np.zeros(n_ineq)
    elapsed = 0.0

    for _ in range(max_iter):
        start = time.perf_counter()
        q = w - v / rho
        elapsed += time.perf_counter() - start

        x, s, solve_time = solve_primal(q)
        elapsed += solve_time

        start = time.perf_counter()
        w_next = np.maximum(0.0, s + v / rho)
        v_next = v + rho * (s - w_next)

        r = np.max(np.abs(s - w_next))
        t = rho * np.max(np.abs(w_next - w))
        w, v = w_next, v_next
        elapsed += time.perf_counter() - start
        if r <= tol and t <= tol:
            break

    return x, elapsed
