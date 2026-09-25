from typing import Any, Mapping

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from src.huanet.neural_layer import _feedforward

jax.config.update("jax_enable_x64", True)


def energy_slices(N: int) -> tuple[slice, slice, slice, slice]:
    """Slices of the physical vector x = [m, u, pc, soc]."""
    return (
        slice(0, N),
        slice(N, 2 * N),
        slice(2 * N, 3 * N),
        slice(3 * N, 4 * N),
    )


def recover_physical(x: np.ndarray, p: dict[str, Any]) -> dict[str, np.ndarray]:
    """Split an energy decision vector whose entries are already physical."""
    m, u, pc, soc = energy_slices(int(p["N"]))
    values = np.asarray(x, dtype=np.float64)
    return {
        "soc": values[..., soc],
        "u": values[..., u],
        "pc": values[..., pc],
        "m": values[..., m],
    }


def equality_rhs(lam: jnp.ndarray, net_demand: jnp.ndarray, N: int) -> jnp.ndarray:
    """Build b(x0) = [x0*e1, x0, net_demand]."""
    lam = jnp.asarray(lam)
    if lam.ndim == 1:
        lam = lam[:, None]
    if lam.shape[-1] != 1:
        raise ValueError(f"Expected lambda shape (..., 1), got {lam.shape}.")
    leading = lam.shape[:-1]
    dynamics = jnp.zeros((*leading, N), dtype=lam.dtype).at[..., 0].set(lam[..., 0])
    demand = jnp.broadcast_to(jnp.asarray(net_demand, dtype=lam.dtype), (*leading, N))
    return jnp.concatenate([dynamics, lam, demand], axis=-1)


def energy_proposal_bounds(p: dict[str, Any]) -> tuple[jnp.ndarray, ...]:
    """Return blockwise physical bounds for energy primal and slack proposals."""
    N = int(p["N"])
    primal_lo = jnp.concatenate([
        jnp.full(N, float(p["m_min"])), jnp.full(N, float(p["u_min"])),
        jnp.full(N, float(p["pc_min"])), jnp.full(N, float(p["s_min"])),
    ])
    primal_hi = jnp.concatenate([
        jnp.full(N, float(p["m_max"])), jnp.full(N, float(p["u_max"])),
        jnp.full(N, float(p["pc_max"])), jnp.full(N, float(p["s_max"])),
    ])
    slack_lo = jnp.zeros(5 * N)
    slack_hi = jnp.concatenate([
        jnp.full(2 * N, float(p["u_max"]) - float(p["u_min"])),
        jnp.full(N, float(p["pc_max"]) - float(p["pc_min"])),
        jnp.full(2 * N, float(p["s_max"]) - float(p["s_min"])),
    ])
    return primal_lo, primal_hi, slack_lo, slack_hi


def energy_slack_scales(p: Mapping[str, Any]) -> jnp.ndarray:
    """Characteristic scales for the physical inequality-slack blocks."""
    N = int(p["N"])
    return jnp.concatenate([
        jnp.full(2 * N, float(p["u_max"]) - float(p["u_min"])),
        jnp.full(N, float(p["pc_max"]) - float(p["pc_min"])),
        jnp.full(2 * N, float(p["s_max"]) - float(p["s_min"])),
    ])


def normalize_energy_q(q: jnp.ndarray, p: Mapping[str, Any]) -> jnp.ndarray:
    """Normalize network inputs without changing the physical ADMM state."""
    return q / energy_slack_scales(p).astype(q.dtype)


def normalize_energy_kkt(residual: jnp.ndarray, p: Mapping[str, Any]) -> jnp.ndarray:
    """Balance stationarity-loss blocks for y = [m, u, pc, soc]."""
    N = int(p["N"])
    scales = jnp.concatenate([
        jnp.full(N, float(p["m_max"]) - float(p["m_min"])),
        jnp.full(N, float(p["u_max"]) - float(p["u_min"])),
        jnp.full(N, float(p["pc_max"]) - float(p["pc_min"])),
        jnp.full(N, float(p["s_max"]) - float(p["s_min"])),
    ])
    return residual / scales.astype(residual.dtype)


def scale_energy_proposal(
    primal_unit: jnp.ndarray, slack_unit: jnp.ndarray, p: dict[str, Any]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply v_lo + (v_hi - v_lo) * sigmoid(z) to unit-sigmoid outputs."""
    primal_lo, primal_hi, slack_lo, slack_hi = energy_proposal_bounds(p)
    return (
        primal_lo + (primal_hi - primal_lo) * primal_unit,
        slack_lo + (slack_hi - slack_lo) * slack_unit,
    )


class EnergyPrimeNet(nn.Module):
    """Energy-specific primal network with physical and virtual proposal scales."""

    cfg: Mapping[str, Any]

    @nn.compact
    def __call__(self, nn_input: jnp.ndarray, eta: jnp.ndarray | None = None):
        del eta
        p = self.cfg["problem"]
        n_var, n_ineq = int(p["n_var"]), int(p["n_ineq"])
        hidden_layers = tuple(self.cfg["neural_net"]["hidden_layers"])
        logits = _feedforward(nn_input, hidden_layers, n_var + n_ineq)
        primal_logits, slack_logits = jnp.split(logits, [n_var], axis=-1)
        primal_unit = nn.sigmoid(primal_logits)
        slack_unit = nn.sigmoid(slack_logits)
        return scale_energy_proposal(primal_unit, slack_unit, p)


def _objective_terms_jax(x: jnp.ndarray, p: dict[str, Any]) -> tuple[jnp.ndarray, ...]:
    m_sl, u_sl, pc_sl, _ = energy_slices(int(p["N"]))
    u = x[..., u_sl]
    pc = x[..., pc_sl]
    imported = x[..., m_sl]
    alpha = (1.0 - float(p["mu"])) / (2.0 * np.sqrt(float(p["mu"])))
    battery = jnp.sqrt(u**2 + float(p["delta"])**2) - float(p["delta"])
    energy = float(p["c_e"]) * float(p["Delta_t"]) * jnp.sum(imported + alpha * battery, axis=-1)
    positive_import = float(p["c_p"]) * jnp.sum(
        jax.nn.softplus(float(p["kappa_m"]) * imported) / float(p["kappa_m"]), axis=-1
    )
    pc_min = float(p["pc_min"])
    pc_eval = jnp.maximum(pc, pc_min)
    discomfort_arg = float(p["a"]) / pc_eval - 1.0
    discomfort = float(p["c_d"]) * jnp.sum(
        jax.nn.softplus(float(p["kappa_c"]) * discomfort_arg) / float(p["kappa_c"]), axis=-1
    )
    return energy, positive_import, discomfort, jnp.all(pc >= pc_min, axis=-1)


def smooth_objective_jax(x: jnp.ndarray, p: dict[str, Any]) -> jnp.ndarray:
    energy, positive_import, discomfort, _ = _objective_terms_jax(x, p)
    return energy + positive_import + discomfort


def objective_components(x: np.ndarray, p: dict[str, Any]) -> dict[str, np.ndarray]:
    physical = recover_physical(x, p)
    u, pc, imported = physical["u"], physical["pc"], physical["m"]
    alpha = (1.0 - float(p["mu"])) / (2.0 * np.sqrt(float(p["mu"])))
    battery = np.sqrt(u**2 + float(p["delta"])**2) - float(p["delta"])
    energy = float(p["c_e"]) * float(p["Delta_t"]) * np.sum(imported + alpha * battery, axis=-1)
    positive_import = float(p["c_p"]) * np.sum(
        np.logaddexp(0.0, float(p["kappa_m"]) * imported) / float(p["kappa_m"]), axis=-1
    )
    pc_min = float(p["pc_min"])
    valid = np.all(pc >= pc_min, axis=-1)
    pc_eval = np.maximum(pc, pc_min)
    discomfort_arg = float(p["a"]) / pc_eval - 1.0
    discomfort = float(p["c_d"]) * np.sum(
        np.logaddexp(0.0, float(p["kappa_c"]) * discomfort_arg) / float(p["kappa_c"]), axis=-1
    )
    return {
        "energy": energy,
        "positive_import": positive_import,
        "discomfort": discomfort,
        "domain_valid": valid,
    }


def energy_cost(x: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    terms = objective_components(x, p)
    return terms["energy"] + terms["positive_import"] + terms["discomfort"]


def cvxpy_objective(x: cp.Expression, p: dict[str, Any]) -> cp.Expression:
    m_sl, u_sl, pc_sl, _ = energy_slices(int(p["N"]))
    u = x[u_sl]
    pc = x[pc_sl]
    imported = x[m_sl]
    alpha = (1.0 - float(p["mu"])) / (2.0 * np.sqrt(float(p["mu"])))
    battery = cp.norm(cp.vstack([u, float(p["delta"]) * np.ones(int(p["N"]))]), axis=0) - float(p["delta"])
    return (
        float(p["c_e"]) * float(p["Delta_t"]) * cp.sum(imported + alpha * battery)
        + float(p["c_p"]) * cp.sum(cp.logistic(float(p["kappa_m"]) * imported)) / float(p["kappa_m"])
        + float(p["c_d"]) * cp.sum(
            cp.logistic(float(p["kappa_c"]) * (float(p["a"]) * cp.inv_pos(pc) - 1.0))
        ) / float(p["kappa_c"])
    )


def compute_j_ref(
    lam: np.ndarray,
    net_demand: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    p: dict[str, Any],
    n_train: int,
    n_calibration: int = 200,
) -> float:
    """Median no-battery optimality gap used by the original energy training loss."""
    N = int(p["N"])
    _, u_sl, _, _ = energy_slices(N)
    x = cp.Variable(A.shape[1])
    slack = cp.Variable(C.shape[0], nonneg=True)
    b = cp.Parameter(A.shape[0])
    objective = cp.Minimize(cvxpy_objective(x, p))
    constraints = [A @ x == b, C @ x + slack == d]
    optimal = cp.Problem(objective, constraints)
    no_battery = cp.Problem(objective, constraints + [x[u_sl] == 0.0])
    if not optimal.is_dcp() or not no_battery.is_dcp():
        raise ValueError("J_ref calibration problems are not DCP canonicalizable.")

    gaps = []
    for value in np.asarray(lam)[:min(n_train, n_calibration)]:
        b.value = np.asarray(equality_rhs(
            jnp.asarray(value).reshape(1, 1), jnp.asarray(net_demand), N
        ))[0]
        optimal.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
        if optimal.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"J_ref optimal solve failed with status {optimal.status}.")
        optimal_value = optimal.value
        no_battery.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
        if no_battery.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"J_ref no-battery solve failed with status {no_battery.status}.")
        gaps.append(no_battery.value - optimal_value)

    j_ref = float(np.median(gaps))
    if not np.isfinite(j_ref) or j_ref <= 0.0:
        raise ValueError(f"J_ref must be finite and positive, got {j_ref}.")
    return j_ref


def energy_slack_form(
    lam: np.ndarray,
    net_demand: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    p: dict[str, Any],
    q: np.ndarray,
    rho: float,
) -> tuple[cp.Problem, cp.Variable, cp.Variable]:
    x = cp.Variable(A.shape[1])
    slack = cp.Variable(C.shape[0])
    b = np.asarray(equality_rhs(jnp.asarray(lam).reshape(1, 1), jnp.asarray(net_demand), int(p["N"])))[0]
    objective = cvxpy_objective(x, p) + 0.5 * rho * cp.sum_squares(slack - q)
    return cp.Problem(cp.Minimize(objective), [A @ x == b, C @ x + slack == d]), x, slack


def eq_viol(y: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.max(np.abs(np.asarray(y) @ A.T - b), axis=1)


def ineq_viol(y: np.ndarray, C: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.max(np.maximum(0.0, np.asarray(y) @ C.T - d), axis=1)


def physical_diagnostics(
    y: np.ndarray,
    lam: np.ndarray,
    net_demand: np.ndarray,
    p: dict[str, Any],
) -> dict[str, np.ndarray | float | int]:
    physical = recover_physical(y, p)
    soc, u, pc, imported = (physical[name] for name in ("soc", "u", "pc", "m"))
    lam_flat = np.asarray(lam).reshape(-1)
    previous_soc = np.concatenate([lam_flat[:, None], soc[:, :-1]], axis=1)
    dynamics = soc - previous_soc + float(p["Delta_t"]) * u / float(p["B"])
    balance = u - pc + imported - np.asarray(net_demand)[None, :]
    return {
        "soc_dynamics_max": np.max(np.abs(dynamics), axis=1),
        "terminal_soc_error": np.abs(soc[:, -1] - lam_flat),
        "power_balance_max": np.max(np.abs(balance), axis=1),
        "battery_bound_violation": np.max(np.maximum.reduce([
            u - float(p["u_max"]), float(p["u_min"]) - u, np.zeros_like(u)
        ]), axis=1),
        "soc_bound_violation": np.max(np.maximum.reduce([
            soc - float(p["s_max"]), float(p["s_min"]) - soc, np.zeros_like(soc)
        ]), axis=1),
        "grid_proposal_range_excursion": np.max(np.maximum.reduce([
            imported - float(p["m_max"]), float(p["m_min"]) - imported, np.zeros_like(imported)
        ]), axis=1),
        "pc_upper_proposal_range_excursion": np.max(np.maximum(pc - float(p["pc_max"]), 0.0), axis=1),
        "pc_lower_bound_violation": np.max(np.maximum(float(p["pc_min"]) - pc, 0.0), axis=1),
        "minimum_pc": np.min(pc, axis=1),
        "pc_below_min_count": int(np.count_nonzero(pc < float(p["pc_min"]))),
    }


def benchmark_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    run_time: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    b: np.ndarray,
    d: np.ndarray,
    lam: np.ndarray,
    net_demand: np.ndarray,
    p: dict[str, Any],
) -> dict[str, float | int]:
    obj_pred, obj_true = energy_cost(y_pred, p), energy_cost(y_true, p)
    eq_error, ineq_error = eq_viol(y_pred, A, b), ineq_viol(y_pred, C, d)
    physical = physical_diagnostics(y_pred, lam, net_demand, p)
    reference_valid = np.isfinite(obj_true)
    prediction_valid = np.isfinite(obj_pred) & (eq_error <= 1e-6) & (ineq_error <= 1e-6)
    valid = reference_valid & prediction_valid
    signed = np.where(valid, obj_pred - obj_true, np.nan)
    relative = np.where(valid, np.abs(signed) / np.maximum(np.abs(obj_true), 1e-10) * 100.0, np.nan)
    return {
        "Mean signed obj difference": float(np.nanmean(signed)) if np.any(valid) else float("nan"),
        "Mean obj gap (%)": float(np.nanmean(relative)) if np.any(valid) else float("nan"),
        "Valid gap samples": int(np.count_nonzero(valid)),
        "Infeasible prediction samples": int(np.count_nonzero(~prediction_valid)),
        "Mean eq violation": float(np.mean(eq_error)),
        "Max eq violation": float(np.max(eq_error)),
        "Mean ineq violation": float(np.mean(ineq_error)),
        "Max ineq violation": float(np.max(ineq_error)),
        "Max SOC dynamics residual": float(np.max(physical["soc_dynamics_max"])),
        "Max terminal SOC mismatch": float(np.max(physical["terminal_soc_error"])),
        "Max power balance residual": float(np.max(physical["power_balance_max"])),
        "Minimum physical pc": float(np.min(physical["minimum_pc"])),
        "pc entries below pc_min": int(physical["pc_below_min_count"]),
        "Mean solve time (s)": float(np.mean(run_time)),
        "Max solve time (s)": float(np.max(run_time)),
    }


def compute_metrics(y, times, y_ref, samples, matrices, p):
    A, C = matrices["A"], matrices["C"]
    b, d = samples["b"], samples["d"]
    obj = energy_cost(y, p)
    obj_ref = energy_cost(y_ref, p)
    eq_error, ineq_error = eq_viol(y, A, b), ineq_viol(y, C, d)
    gap_valid = np.isfinite(obj) & np.isfinite(obj_ref) & (eq_error <= 1e-6) & (ineq_error <= 1e-6)
    signed_difference = np.where(gap_valid, obj - obj_ref, np.nan)
    relative_gap = np.where(
        gap_valid, np.abs(signed_difference) / np.maximum(np.abs(obj_ref), 1e-10) * 100.0, np.nan
    )
    return {
        "x": y,
        "times": times,
        "obj": obj,
        "signed_obj_difference": signed_difference,
        "relative_gap_percent": relative_gap,
        "gap_valid": gap_valid,
        "eq_viol": eq_error,
        "ineq_viol": ineq_error,
        "physical": physical_diagnostics(y, samples["lam"], samples["net_demand"], p),
        "metrics": benchmark_metrics(y, y_ref, times, A, C, b, d, samples["lam"], samples["net_demand"], p),
    }


def print_metrics_table(title: str, metrics: dict[str, float | int]) -> None:
    print(f"\n{title}")
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"  {key:<36} {value:.6e}")
        else:
            print(f"  {key:<36} {value}")


def report_benchmark(summary_path, samples, matrices, solvers, sequential_time, p, J_ref) -> None:
    all_solvers = [
        ("clarabel", *solvers["clarabel"], "Clarabel (reference)"),
        ("our_method", *solvers["our_method"], "HUANet Metrics (vs Clarabel)"),
    ]
    if "admm" in solvers:
        all_solvers.append(("admm", *solvers["admm"], "ADMM Metrics (vs Clarabel)"))
    if "dc3" in solvers:
        all_solvers.append(("dc3", *solvers["dc3"], "DC3 Metrics (vs Clarabel)"))
    save_dict = {
        **samples,
        "n_var": matrices["A"].shape[1],
        "J_ref": J_ref,
        "our_method_sequential_time": sequential_time,
    }
    results = {}
    for prefix, y, times, _ in all_solvers:
        result = compute_metrics(y, times, solvers["clarabel"][0], samples, matrices, p)
        results[prefix] = result
        for key, value in result.items():
            if key != "physical":
                save_dict[f"{prefix}_{key}"] = value
    np.savez_compressed(summary_path, **save_dict)
    for prefix, _, _, title in all_solvers:
        print_metrics_table(title, results[prefix]["metrics"])
