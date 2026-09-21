from pathlib import Path
from typing import Any

import numpy as np
import yaml

from utils import (
    compute_j_ref,
    energy_slices,
    imported_slice,
    sample_disturbance_windows,
)


class EnergyMPCProblem:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.problem = cfg["problem"]
        self.N = int(self.problem["N"])

    @property
    def n_var(self) -> int:
        return 4 * self.N + 1

    @property
    def n_eq(self) -> int:
        return 2 * self.N + 1

    @property
    def n_ineq(self) -> int:
        return 5 * self.N + 3

    def generate_data(self, n_samples: int) -> dict[str, Any]:
        p = self.problem
        N = self.N
        S_e, S_B, S_m = (float(p[key]) for key in ("S_e", "S_B", "S_m"))
        gamma = float(p["gamma"])
        e, pb, r = energy_slices(N)
        m = imported_slice(N)
        steps = np.arange(N)

        A = np.zeros((self.n_eq, self.n_var))
        A[steps, e.start + steps] = -1.0
        A[steps, e.start + steps + 1] = 1.0
        A[steps, pb.start + steps] = S_B / S_e
        A[N, e.start] = 1.0
        A[N + 1 + steps, m.start + steps] = 1.0
        A[N + 1 + steps, r.start + steps] = -gamma / S_m
        A[N + 1 + steps, pb.start + steps] = S_B / S_m

        C = np.zeros((self.n_ineq, self.n_var))
        C[steps, pb.start + steps] = 1.0
        C[N + steps, pb.start + steps] = -1.0
        C[2 * N + np.arange(N + 1), e.start + np.arange(N + 1)] = 1.0
        C[3 * N + 1 + np.arange(N + 1), e.start + np.arange(N + 1)] = -1.0
        C[4 * N + 2, e.stop - 1] = -1.0
        C[4 * N + 3 + steps, r.start + steps] = -1.0

        soc_scale = float(p["b_bess"]) / float(p["Delta_t"]) / S_e
        d = np.concatenate([
            np.full(N, float(p["B_max"]) / S_B),
            np.full(N, -float(p["B_min"]) / S_B),
            np.full(N + 1, soc_scale * (float(p["sigma_max"]) - float(p["sigma_0"]))),
            np.full(N + 1, soc_scale * (float(p["sigma_0"]) - float(p["sigma_min"]))),
            [soc_scale * (float(p["sigma_0"]) - float(p["sigma_N_min"]))],
            np.full(N, -float(p["r_min"])),
        ])
        b = np.zeros(self.n_eq)
        E = np.block([
            [A, np.zeros((self.n_eq, self.n_ineq))],
            [C, np.eye(self.n_ineq)],
        ])
        constraint_rhs = np.concatenate([b, d])
        gain = np.linalg.solve(E @ E.T, E).T
        Pi = np.eye(self.n_var + self.n_ineq) - gain @ E
        c = gain @ constraint_rhs

        demand, P_u, P_R, starts = sample_disturbance_windows(N, n_samples, int(p["seed"]))
        n_train = min(n_samples, max(1, int(n_samples * float(self.cfg["data"]["train_percent"]))))
        center = np.mean(demand[:n_train], axis=0)
        scale = np.maximum(np.std(demand[:n_train], axis=0), 1e-8)
        J_ref = compute_j_ref(demand, A, C, d, p, n_train)

        row_starts = np.array([0, N, 2 * N, 3 * N + 1, 4 * N + 2, 4 * N + 3])
        row_ends = np.array([N, 2 * N, 3 * N + 1, 4 * N + 2, 4 * N + 3, 5 * N + 3])
        return {
            "A": A,
            "C": C,
            "E": E,
            "b": b,
            "d_ineq": d,
            "d": d,
            "constraint_rhs": constraint_rhs,
            "Pi_correction": Pi,
            "correction_gain": gain,
            "c_correction": c,
            "n_y": self.n_var + self.n_ineq,
            "e_bound": max(soc_scale * (float(p["sigma_max"]) - float(p["sigma_0"])),
                           soc_scale * (float(p["sigma_0"]) - float(p["sigma_min"]))),
            "pb_bound": max(float(p["B_max"]) / S_B, -float(p["B_min"]) / S_B),
            "J_ref": J_ref,
            "lam": demand,
            "lambda_raw": demand,
            "lambda_features": (demand - center) / scale,
            "disturbance": demand,
            "lambda_param": demand,
            "zeta": demand,
            "zeta_mean": float(np.mean(demand[:n_train])),
            "zeta_std": float(max(np.std(demand[:n_train]), 1e-8)),
            "parameter_center": center,
            "parameter_scale": scale,
            "P_u_samples": P_u,
            "P_R_samples": P_R,
            "start_indices": starts,
            "row_block_starts": row_starts,
            "row_block_ends": row_ends,
            "row_block_labels": np.array([
                "pb_upper", "pb_lower", "soc_upper", "soc_lower",
                "terminal_reserve", "r_lower",
            ]),
        }


def main() -> None:
    with Path(__file__).with_name("cfg.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    problem = EnergyMPCProblem(cfg)
    n_samples = int(cfg["data"]["n_samples"])
    scenario_root = (Path(__file__).resolve().parents[2]/ cfg["data"]["root_subdir"]/ f"n_var_{problem.n_var}")
    dataset_dir = scenario_root / "datasets"
    model_dir = scenario_root / "model_params"
    for directory in (dataset_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data = problem.generate_data(n_samples)
    dataset_path = dataset_dir / f"datasets_{n_samples}.npz"
    np.savez_compressed(dataset_path, **data)

    print(f"Saved dataset: {dataset_path}")
    print(f"Model directory: {model_dir}")


if __name__ == "__main__":
    main()
