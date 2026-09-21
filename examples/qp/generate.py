from pathlib import Path
from typing import Any

import numpy as np
import yaml



class QPProblem:
    def __init__(self, cfg: dict[str, Any]) -> None:
        problem     = cfg["problem"]
        self.n_var  = int(problem["n_var"])
        self.n_eq   = int(problem["n_eq"])
        self.n_ineq = int(problem["n_ineq"])
        self.seed   = int(problem["seed"])
        self.rng    = np.random.default_rng(self.seed)

    def generate_data(self, n_samples: int) -> dict[str, Any]:
        Fq = self.rng.standard_normal((self.n_var, self.n_var))
        lam_para = self.rng.uniform(-1.0, 1.0, (n_samples, self.n_var))
        Q  = Fq.T @ Fq + np.eye(self.n_var)
        Q_inv = np.linalg.inv(Q)
        p  = np.ones(self.n_var) + lam_para
        A = self.rng.standard_normal((self.n_eq, self.n_var))
        C = self.rng.standard_normal((self.n_ineq, self.n_var))
        eps_b = self.rng.uniform(0.0, 0.1, self.n_eq)
        eps_d = self.rng.uniform(0.0, 0.1, self.n_ineq)

        b = - (p @ Q_inv.T) @ A.T + eps_b
        d = - (p @ Q_inv.T) @ C.T + eps_d

        return {
            "Q": Q,
            "A": A,
            "C": C,
            "lam": lam_para,
            "p": p,
            "b": b,
            "d": d,
        }


def main() -> None:
    with Path(__file__).with_name("cfg.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    problem = QPProblem(cfg)
    n_samples = int(cfg["data"]["n_samples"])
    scenario_root = (
        Path(__file__).resolve().parents[2]
        / cfg["data"]["root_subdir"]
        / f"n_var_{problem.n_var}"
    )
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
