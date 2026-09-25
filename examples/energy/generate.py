from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

from utils import compute_j_ref, energy_slices, equality_rhs


def _load_fixed_forecast(cfg: dict[str, Any]) -> tuple[np.ndarray, ...]:
    """Load one positionally aligned quarter-hour load/PV window."""
    p, data_cfg = cfg["problem"], cfg["data"]
    root = Path(__file__).resolve().parents[2] / data_cfg["source_subdir"]
    load_data = pd.read_csv(root / data_cfg["load_file"])
    pv_data = pd.read_csv(root / data_cfg["pv_file"])
    if not {"time_h", "load_kw"}.issubset(load_data.columns):
        raise ValueError("Load data must contain time_h and load_kw columns.")
    if not {"timestamp", "power_kw"}.issubset(pv_data.columns):
        raise ValueError("PV data must contain timestamp and power_kw columns.")

    load_time = load_data["time_h"].to_numpy(dtype=np.float64)
    load = load_data["load_kw"].to_numpy(dtype=np.float64)
    pv_time = pd.DatetimeIndex(pd.to_datetime(pv_data["timestamp"], utc=True, errors="raise"))
    pv = pv_data["power_kw"].to_numpy(dtype=np.float64)
    if not all(np.all(np.isfinite(values)) for values in (load_time, load, pv)):
        raise ValueError("Load and PV forecasts must contain only finite values.")

    dt, N = float(p["Delta_t"]), int(p["N"])
    if len(load_time) > 1 and not np.allclose(np.diff(load_time), dt, rtol=0.0, atol=1e-10):
        raise ValueError(f"Load sampling interval is not {dt} h.")
    pv_step_h = (pv_time[1:] - pv_time[:-1]).total_seconds().to_numpy() / 3600.0
    if len(pv_step_h) > 0 and not np.allclose(pv_step_h, dt, rtol=0.0, atol=1e-10):
        raise ValueError(f"PV sampling interval is not {dt} h.")

    steps_per_day = int(round(24.0 / dt))
    if not np.isclose(steps_per_day * dt, 24.0):
        raise ValueError(f"Delta_t={dt} does not divide a day into integral steps.")
    start = int(p["forecast_day"]) * steps_per_day
    stop = start + N
    if start < 0 or stop > min(len(load), len(pv)):
        raise IndexError(f"Forecast window [{start}, {stop}) is unavailable in both series.")
    load_window = np.maximum(load[start:stop], 0.0)
    pv_window = np.maximum(pv[start:stop], 0.0)
    return load_window - pv_window, load_window, pv_window, load_time[start:stop]


def build_constraint_matrices(p: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the physical A, C, and h matrices for the configured horizon."""
    N = int(p["N"])
    n_var, n_eq, n_ineq = 4 * N, 2 * N + 1, 5 * N
    m, u, pc, soc = energy_slices(N)
    steps = np.arange(N)

    A = np.zeros((n_eq, n_var), dtype=np.float64)
    A[steps, soc.start + steps] = 1.0
    A[steps[1:], soc.start + steps[:-1]] = -1.0
    A[steps, u.start + steps] = float(p["Delta_t"]) / float(p["B"])
    A[N, soc.stop - 1] = 1.0
    A[N + 1 + steps, m.start + steps] = 1.0
    A[N + 1 + steps, u.start + steps] = 1.0
    A[N + 1 + steps, pc.start + steps] = -1.0

    C = np.zeros((n_ineq, n_var), dtype=np.float64)
    variables = (u, u, pc, soc, soc)
    signs = (1.0, -1.0, -1.0, 1.0, -1.0)
    for block, (variable, sign) in enumerate(zip(variables, signs)):
        C[block * N + steps, variable.start + steps] = sign
    h = np.concatenate([
        np.full(N, float(p["u_max"])), np.full(N, -float(p["u_min"])),
        np.full(N, -float(p["pc_min"])),
        np.full(N, float(p["s_max"])), np.full(N, -float(p["s_min"])),
    ])
    return A, C, h


class EnergyMPCProblem:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.problem = cfg["problem"]
        self.N = int(self.problem["N"])

    @property
    def n_var(self) -> int:
        return 4 * self.N

    @property
    def n_eq(self) -> int:
        return 2 * self.N + 1

    @property
    def n_ineq(self) -> int:
        return 5 * self.N

    def generate_data(self, n_samples: int) -> dict[str, Any]:
        p, N = self.problem, self.N
        expected = (1, self.n_var, self.n_eq, self.n_ineq)
        configured = tuple(int(p[name]) for name in ("n_param", "n_var", "n_eq", "n_ineq"))
        if configured != expected:
            raise ValueError(f"Configured dimensions {configured} do not match {expected}.")
        m, _, pc, soc = energy_slices(N)
        A, C, d = build_constraint_matrices(p)

        net_demand, load, pv, timestamps = _load_fixed_forecast(self.cfg)
        rng = np.random.default_rng(int(p["seed"]))
        lam = rng.uniform(
            float(p["soc_initial_min"]), float(p["soc_initial_max"]), size=(n_samples, 1)
        )
        lam_features = (
            2.0 * (lam - float(p["soc_initial_min"]))
            / (float(p["soc_initial_max"]) - float(p["soc_initial_min"]))
            - 1.0
        )
        b = np.asarray(equality_rhs(jnp.asarray(lam), jnp.asarray(net_demand), N))
        d_batch = np.broadcast_to(d, (n_samples, self.n_ineq)).copy()
        n_train = min(n_samples, max(1, int(n_samples * float(self.cfg["data"]["train_percent"]))))
        J_ref = compute_j_ref(lam, net_demand, A, C, d, p, n_train)

        E = np.block([
            [A, np.zeros((self.n_eq, self.n_ineq))],
            [C, np.eye(self.n_ineq)],
        ])
        if np.linalg.matrix_rank(A) != self.n_eq or np.linalg.matrix_rank(E) != self.n_eq + self.n_ineq:
            raise ValueError("Energy constraint matrices do not have the required row rank.")
        correction_gain = np.linalg.solve(E @ E.T, E).T

        diagnostic_lam = np.array([[0.25], [0.5], [0.75]], dtype=np.float64)
        diagnostic_b = np.asarray(equality_rhs(jnp.asarray(diagnostic_lam), jnp.asarray(net_demand), N))

        feasible_x = np.zeros((len(diagnostic_lam), self.n_var), dtype=np.float64)
        feasible_x[:, soc] = diagnostic_lam
        feasible_x[:, pc] = float(p["pc_min"])
        feasible_x[:, m] = net_demand + float(p["pc_min"])
        if np.max(np.abs(feasible_x @ A.T - diagnostic_b)) > 1e-10:
            raise ValueError("Constructed constant-SOC trajectory violates the equalities.")
        if np.max(feasible_x @ C.T - d) > 1e-10:
            raise ValueError("Constructed constant-SOC trajectory violates the inequalities.")

        return {
            "A": A,
            "C": C,
            "E": E,
            "correction_gain": correction_gain,
            "b": b,
            "d": d,
            "d_ineq": d,
            "d_batch": d_batch,
            "lam": lam,
            "lambda_raw": lam,
            "lambda_features": lam_features,
            "eta": np.concatenate([b, d_batch], axis=-1),
            "J_ref": np.array(J_ref),
            "net_demand": net_demand,
            "forecast_load": load,
            "forecast_pv": pv,
            "forecast_timestamps_h": timestamps,
            "forecast_day": np.array(int(p["forecast_day"])),
            "load_file": np.array(str(self.cfg["data"]["load_file"])),
            "pv_file": np.array(str(self.cfg["data"]["pv_file"])),
            "diagnostic_lam": diagnostic_lam,
            "diagnostic_b": diagnostic_b,
            "diagnostic_feasible_x": feasible_x,
            "schema_version": np.array(str(p["schema_version"])),
            "parameter_meaning": np.array("initial_soc_fraction"),
            "x0_bounds": np.array([
                float(p["soc_initial_min"]), float(p["soc_initial_max"]),
            ]),
            "variable_ordering": np.array("m,u,pc,soc"),
            "Delta_t": np.array(float(p["Delta_t"])),
            "B": np.array(float(p["B"])),
            "c_e": np.array(float(p["c_e"])),
            "c_p": np.array(float(p["c_p"])),
            "c_d": np.array(float(p["c_d"])),
            "mu": np.array(float(p["mu"])),
            "a": np.array(float(p["a"])),
            "delta": np.array(float(p["delta"])),
            "kappa_m": np.array(float(p["kappa_m"])),
            "kappa_c": np.array(float(p["kappa_c"])),
            "u_min": np.array(float(p["u_min"])),
            "u_max": np.array(float(p["u_max"])),
            "m_min": np.array(float(p["m_min"])),
            "m_max": np.array(float(p["m_max"])),
            "s_min": np.array(float(p["s_min"])),
            "s_max": np.array(float(p["s_max"])),
            "pc_min": np.array(float(p["pc_min"])),
            "pc_max": np.array(float(p["pc_max"])),
            "primal_lo": np.concatenate([
                np.full(N, float(p["m_min"])), np.full(N, float(p["u_min"])),
                np.full(N, float(p["pc_min"])), np.full(N, float(p["s_min"])),
            ]),
            "primal_hi": np.concatenate([
                np.full(N, float(p["m_max"])), np.full(N, float(p["u_max"])),
                np.full(N, float(p["pc_max"])), np.full(N, float(p["s_max"])),
            ]),
            "slack_lo": np.zeros(5 * N),
            "slack_hi": np.concatenate([
                np.full(2 * N, float(p["u_max"]) - float(p["u_min"])),
                np.full(N, float(p["pc_max"]) - float(p["pc_min"])),
                np.full(2 * N, float(p["s_max"]) - float(p["s_min"])),
            ]),
            "row_block_starts": np.arange(5) * N,
            "row_block_ends": np.arange(1, 6) * N,
            "row_block_labels": np.array([
                "u_upper", "u_lower", "pc_lower", "soc_upper", "soc_lower",
            ]),
        }


def main() -> None:
    with Path(__file__).with_name("cfg.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    problem = EnergyMPCProblem(cfg)
    n_samples = int(cfg["data"]["n_samples"])
    scenario_root = Path(__file__).resolve().parents[2] / cfg["data"]["root_subdir"] / f"n_var_{problem.n_var}"
    dataset_dir = scenario_root / "datasets"
    model_dir = scenario_root / "model_params"
    for directory in (dataset_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_dir / f"datasets_{n_samples}.npz"
    np.savez_compressed(dataset_path, **problem.generate_data(n_samples))
    print(f"Saved dataset: {dataset_path}")
    print(f"Model directory: {model_dir}")


if __name__ == "__main__":
    main()
