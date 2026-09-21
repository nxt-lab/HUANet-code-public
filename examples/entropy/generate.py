from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.special import softmax


class EntropyProblem:
    def __init__(self, cfg: dict[str, Any]) -> None:
        problem = cfg["problem"]
        self.n_var = int(problem["n_var"])
        self.n_eq = int(problem["n_eq"])
        self.n_ineq = int(problem["n_ineq"])
        self.rng = np.random.default_rng(int(problem["seed"]))

    def generate_data(self, n_samples: int) -> dict[str, np.ndarray]:
        A = np.ones((self.n_eq, self.n_var))
        C = self.rng.standard_normal((self.n_ineq, self.n_var))

        x = softmax(self.rng.uniform(0.001, 1.0, (n_samples, self.n_var)), axis=1)
        lam = x @ C.T + self.rng.uniform(0.0, 0.1, (n_samples, self.n_ineq))

        return {
            "A": A,
            "C": C,
            "lam": lam,
            "b": np.ones((n_samples, self.n_eq)),
            "d": lam,
        }


def main() -> None:
    with Path(__file__).with_name("cfg.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    problem = EntropyProblem(cfg)
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
