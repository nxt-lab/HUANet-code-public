import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import cvxpy as cp
import jax.numpy as jnp
from flax import linen as nn
from typing import Mapping


with Path(__file__).with_name("cfg.yaml").open(encoding="utf-8") as file:
    cfg = yaml.safe_load(file)

problem_cfg = {**cfg["problem"], "rho": cfg["admm"]["rho"]}
dataset_cfg = cfg["data"]


class EnergyPrimeNet(nn.Module):
    cfg: Mapping[str, Any]

    @nn.compact
    def __call__(self, nn_input: jnp.ndarray, eta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        p = self.cfg["problem"]
        N = int(p["N"])
        x = nn_input
        kernel_init = nn.initializers.xavier_uniform()
        bias_init = nn.initializers.zeros_init()
        for width in self.cfg["neural_net"]["hidden_layers"]:
            x = nn.gelu(nn.Dense(width, kernel_init=kernel_init, bias_init=bias_init)(x))

        S_B, S_e, S_m = (float(p[key]) for key in ("S_B", "S_e", "S_m"))
        pb_bound = max(float(p["B_max"]) / S_B, -float(p["B_min"]) / S_B)
        pb_bar = pb_bound * nn.tanh(
            nn.Dense(N, kernel_init=kernel_init, bias_init=bias_init, name="pb_head")(x)
        )
        r_floor = float(p["r_ext"])
        r_bar = r_floor + (1.0 - r_floor) * nn.sigmoid(
            nn.Dense(N, kernel_init=kernel_init, bias_init=bias_init, name="r_head")(x)
        )
        e_steps = -jnp.cumsum((S_B / S_e) * pb_bar, axis=-1)
        e_bar = jnp.concatenate([jnp.zeros_like(e_steps[..., :1]), e_steps], axis=-1)
        demand = eta[..., N + 1:2 * N + 1]
        m_bar = demand + (float(p["gamma"]) / S_m) * r_bar - (S_B / S_m) * pb_bar
        s_bar = nn.softplus(nn.Dense(
            5 * N + 3, kernel_init=kernel_init, bias_init=nn.initializers.constant(-3.0)
        )(x))
        return jnp.concatenate([e_bar, pb_bar, r_bar, m_bar], axis=-1), s_bar


def equality_rhs(lam: jnp.ndarray, N: int, S_m: float) -> jnp.ndarray:
    return jnp.concatenate([
        jnp.zeros((lam.shape[0], N + 1), dtype=lam.dtype), lam[..., :N] / S_m,
    ], axis=-1)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root() -> str:
    return str(Path(__file__).resolve().parents[2])


def resolve_scenario_dirs(n_var: int, root_subdir: str | None = None) -> dict[str, str]:
    if root_subdir is None:
        root_subdir = dataset_cfg["root_subdir"]
    scenario_root = os.path.join(project_root(), root_subdir, f"n_var_{n_var}")
    return {
        "scenario_root": scenario_root,
        "datasets": os.path.join(scenario_root, "datasets"),
        "problem_data": os.path.join(scenario_root, "problem_data"),
        "model_params": os.path.join(scenario_root, "model_params"),
    }


def energy_slices(N: int) -> tuple[slice, slice, slice]:
    """Slices of the [e, P_B, r] blocks."""
    e = slice(0, N + 1)
    pb = slice(e.stop, e.stop + N)
    r = slice(pb.stop, pb.stop + N)
    return e, pb, r


def imported_slice(N: int) -> slice:
    """Slice of the imported-power block m."""
    return slice(3 * N + 1, 4 * N + 1)


def physical_to_scaled(
    sigma: np.ndarray,
    PB: np.ndarray,
    Pc: np.ndarray,
    S_e: float,
    S_B: float,
    gamma: float,
    sigma_0: float,
    Delta_t: float,
    b_bess: float,
) -> np.ndarray:
    e = (b_bess / Delta_t) * (np.asarray(sigma, dtype=np.float64) - sigma_0)
    return np.concatenate([
        e / S_e,
        np.asarray(PB, dtype=np.float64) / S_B,
        np.asarray(Pc, dtype=np.float64) / gamma,
    ])


def scaled_to_physical(
    x_scaled: np.ndarray,
    N: int,
    S_e: float,
    S_B: float,
    gamma: float,
    sigma_0: float,
    Delta_t: float,
    b_bess: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e_sl, pb_sl, r_sl = energy_slices(N)
    x_scaled = np.asarray(x_scaled, dtype=np.float64)
    e = S_e * x_scaled[e_sl]
    sigma = sigma_0 + (Delta_t / b_bess) * e
    PB = S_B * x_scaled[pb_sl]
    Pc = gamma * x_scaled[r_sl]
    return sigma, PB, Pc


def _read_power_series(path: Path, column: str) -> np.ndarray:
    data = pd.read_csv(path)
    if column not in data.columns:
        raise ValueError(f"Column {column!r} not found in {path}. Available columns: {list(data.columns)}")
    return data[column].astype(float).to_numpy()


def load_power_series() -> tuple[np.ndarray, np.ndarray]:
    root = Path(project_root()) / dataset_cfg["root_subdir"]
    P_u = _read_power_series(root / "load/load_30days_gp.csv", "load_kw")
    P_R = _read_power_series(root / "generator/PV_30days_gp.csv", "PV_kw")
    n_time = min(len(P_u), len(P_R))
    return np.maximum(P_u[:n_time], 0.0), np.maximum(P_R[:n_time], 0.0)


def midnight_window(day: int, N: int, Delta_t: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    Delta_t = float(problem_cfg["Delta_t"] if Delta_t is None else Delta_t)
    steps_per_day = int(round(24.0 / Delta_t))
    P_u, P_R = load_power_series()
    n_days = len(P_u) // steps_per_day
    if not 0 <= day < n_days:
        raise IndexError(f"day={day} outside the {n_days} available days.")
    start = day * steps_per_day
    if start + N > len(P_u):
        raise ValueError(f"day={day} does not have {N} steps of data after midnight.")
    sl = slice(start, start + N)
    return P_u[sl], P_R[sl]


def sample_disturbance_windows(N: int, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    P_u, P_R = load_power_series()
    starts = np.random.default_rng(seed).integers(0, len(P_u) - N + 1, size=n_samples)
    windows = starts[:, None] + np.arange(N)
    P_u_samples = P_u[windows]
    P_R_samples = P_R[windows]
    return P_u_samples - P_R_samples, P_u_samples, P_R_samples, starts


def discomfort_ratio_np(r: np.ndarray, r_ext: float) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    phi_ext = 1.0 / r_ext - 1.0
    dr = r - r_ext
    extended = phi_ext - dr / r_ext**2 + dr**2 / r_ext**3
    safe_r = np.where(r >= r_ext, np.maximum(r, r_ext), 1.0)
    return np.where(r >= r_ext, 1.0 / safe_r - 1.0, extended)


def smooth_discomfort_np(r: np.ndarray, eta: float, kappa_g: float, r_ext: float) -> np.ndarray:
    return eta * np.logaddexp(0.0, kappa_g * discomfort_ratio_np(r, r_ext)) / kappa_g


def exact_discomfort_np(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    positive = r > 0.0
    ratio = np.where(positive, 1.0 / np.where(positive, r, 1.0) - 1.0, np.inf)
    return np.maximum(ratio, 0.0)


def compute_j_ref(
    demand: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    problem: dict[str, Any],
    n_train: int,
    n_calibration: int = 200,
) -> float:
    N = int(problem["N"])
    _, pb, r = energy_slices(N)
    m = imported_slice(N)
    S_B = float(problem["S_B"])
    S_m = float(problem["S_m"])
    delta = float(problem["delta"])
    mu = float(problem["mu"])

    x = cp.Variable(A.shape[1])
    s = cp.Variable(C.shape[0], nonneg=True)
    b = cp.Parameter(A.shape[0])
    P_B = S_B * x[pb]
    cycling = (1.0 - mu) / (2.0 * np.sqrt(mu))
    abs_power = cp.norm(cp.vstack([P_B, delta * np.ones(N)]), axis=0) - delta
    kappa_D = float(problem["kappa_D"])
    kappa_g = float(problem["kappa_g"])
    f = cp.Minimize(
        float(problem["p_k"]) * float(problem["Delta_t"]) * cp.sum(S_m * x[m] + cycling * abs_power)
        + float(problem["p_p"]) * cp.sum(cp.logistic(kappa_D * S_m * x[m])) / kappa_D
        + float(problem["eta"]) * cp.sum(cp.logistic(kappa_g * (cp.inv_pos(x[r]) - 1.0))) / kappa_g
    )
    constraints = [A @ x == b, C @ x + s == d]
    optimal = cp.Problem(f, constraints)
    energy_prob = cp.Problem(f, constraints + [x[pb] == 0])

    gaps = []
    for lam in demand[:min(n_train, n_calibration)]:
        rhs = np.zeros(A.shape[0])
        rhs[N + 1:] = lam[:N] / S_m
        b.value = rhs
        optimal.solve(solver=cp.CLARABEL)
        j_opt = optimal.value
        energy_prob.solve(solver=cp.CLARABEL)
        gaps.append(energy_prob.value - j_opt)

    j_ref = float(np.median(gaps))
    return j_ref


def energy_cost(x: np.ndarray, p: dict[str, Any], *, smooth: bool = False, centered: bool = False) -> np.ndarray:
    N = int(p["N"])
    _, pb, r = energy_slices(N)
    P_B = float(p["S_B"]) * x[:, pb]
    imported = float(p["S_m"]) * x[:, imported_slice(N)]
    cycling = (1.0 - float(p["mu"])) / (2.0 * np.sqrt(float(p["mu"])))
    if smooth:
        delta = float(p["delta"])
        battery = np.sqrt(P_B**2 + delta**2) - delta
        peak = np.logaddexp(0.0, float(p["kappa_D"]) * imported) / float(p["kappa_D"])
        if centered:
            peak -= np.log(2.0) / float(p["kappa_D"])
        discomfort = smooth_discomfort_np(x[:, r], float(p["eta"]), float(p["kappa_g"]), float(p["r_ext"]))
    else:
        battery = np.abs(P_B)
        peak = np.maximum(imported, 0.0)
        discomfort = float(p["eta"]) * exact_discomfort_np(x[:, r])
    return (
        float(p["p_k"]) * float(p["Delta_t"]) * np.sum(imported + cycling * battery, axis=1)
        + float(p["p_p"]) * np.sum(peak, axis=1)
        + np.sum(discomfort, axis=1)
    )


def energy_slack_form(
    lam: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    p: dict[str, Any],
    q: np.ndarray,
    rho: float,
) -> tuple[cp.Problem, cp.Variable, cp.Variable]:
    """Build the energy epigraph problem with ADMM's inequality slack."""
    N = int(p["N"])
    _, pb, r = energy_slices(N)
    x = cp.Variable(A.shape[1])
    s = cp.Variable(C.shape[0])
    su = cp.Variable(N, nonneg=True)
    sm = cp.Variable(N, nonneg=True)
    sd = cp.Variable(N, nonneg=True)
    P_B = float(p["S_B"]) * x[pb]
    imported = float(p["S_m"]) * x[imported_slice(N)]
    cycling = (1.0 - float(p["mu"])) / (2.0 * np.sqrt(float(p["mu"])))

    b = np.zeros(A.shape[0])
    b[N + 1:] = lam[:N] / float(p["S_m"])
    f = (
        float(p["p_k"]) * float(p["Delta_t"]) * cp.sum(imported + cycling * su)
        + float(p["p_p"]) * cp.sum(sm)
        + float(p["eta"]) * cp.sum(sd)
        + 0.5 * rho * cp.sum_squares(s - q)
    )
    constraints = [
        A @ x == b,
        C @ x + s == d,
        su >= P_B,
        su >= -P_B,
        sm >= imported,
        sd >= cp.inv_pos(x[r]) - 1.0,
    ]
    return cp.Problem(cp.Minimize(f), constraints), x, s


def benchmark_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    run_time: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    b: np.ndarray,
    d: np.ndarray,
    problem_cfg: dict[str, Any],
) -> dict[str, float]:
    obj_pred = energy_cost(y_pred, problem_cfg)
    obj_true = energy_cost(y_true, problem_cfg)
    gap_percent = np.abs(obj_pred - obj_true) / np.maximum(np.abs(obj_true), 1e-10) * 100.0
    eq_error = eq_viol(y_pred, A, b)
    ineq_error = ineq_viol(y_pred, C, d)
    return {
        "Mean obj gap (%)": float(np.mean(gap_percent)),
        "Max obj gap (%)": float(np.max(gap_percent)),
        "Mean eq violation": float(np.mean(eq_error)),
        "Max eq violation": float(np.max(eq_error)),
        "Mean ineq violation": float(np.mean(ineq_error)),
        "Max ineq violation": float(np.max(ineq_error)),
        "Mean solve time (s)": float(np.mean(run_time)),
        "Max solve time (s)": float(np.max(run_time)),
    }


def eq_viol(y: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(y @ A.T - b, ord=np.inf, axis=1)


def ineq_viol(y: np.ndarray, C: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.max(np.maximum(0.0, y @ C.T - d), axis=1)


def compute_metrics(y, times, y_ref, samples, matrices, problem_cfg):
    A, C = matrices["A"], matrices["C"]
    b, d = samples["b"], samples["d"]
    return {
        "x": y,
        "times": times,
        "obj": energy_cost(y, problem_cfg),
        "smooth_obj": energy_cost(y, problem_cfg, smooth=True),
        "smooth_centered_obj": energy_cost(y, problem_cfg, smooth=True, centered=True),
        "eq_viol": eq_viol(y, A, b),
        "ineq_viol": ineq_viol(y, C, d),
        "active_counts": np.sum((d - y @ C.T) <= 1e-6, axis=1),
        "metrics": benchmark_metrics(y, y_ref, times, A, C, b, d, problem_cfg),
    }


def print_metrics_table(title: str, metrics: dict[str, float]) -> None:
    metric_width = 40
    value_width = 28

    print(f"\n{title}")
    print("+" + "-" * (metric_width + 2) + "+" + "-" * (value_width + 2) + "+")
    print(f"| {'Metric':<{metric_width}} | {'Value':>{value_width}} |")
    print("+" + "-" * (metric_width + 2) + "+" + "-" * (value_width + 2) + "+")

    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            value_str = f"{value:.6e}"
        else:
            value_str = str(value)
        print(f"| {key:<{metric_width}} | {value_str:>{value_width}} |")

    print("+" + "-" * (metric_width + 2) + "+" + "-" * (value_width + 2) + "+")


def report_benchmark(summary_path, samples, matrices, solvers, sequential_time, problem_cfg, J_ref) -> None:
    all_solvers = [
        ("clarabel", *solvers["clarabel"], "Clarabel (ground truth)"),
        ("our_method", *solvers["our_method"], "HUANet Metrics (vs Clarabel)"),
    ]
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
        result = compute_metrics(y, times, solvers["clarabel"][0], samples, matrices, problem_cfg)
        results[prefix] = result
        save_dict.update({f"{prefix}_{key}": value for key, value in result.items()})

    np.savez_compressed(summary_path, **save_dict)

    for prefix, _, _, title in all_solvers:
        print_metrics_table(title, results[prefix]["metrics"])
