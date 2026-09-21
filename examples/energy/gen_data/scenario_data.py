import numpy as np

from data_load_gen import data_load_gen
from data_PV_gen import data_PV_gen


def sample_nonnegative_gp_trajectories(gp, time_grid, S: int, seed: int | None = None):
    rng = np.random.default_rng(seed)
    time_grid = np.asarray(time_grid, dtype=float)
    X = time_grid.reshape(-1, 1)

    mean_values, covariance = gp.predict(X, return_cov=True)
    covariance = covariance + 1e-8 * np.eye(covariance.shape[0])

    samples = rng.multivariate_normal(mean_values, covariance, size=S)
    return np.maximum(samples, 0.0)


def scn_data_gen(S: int = 1000, L: int = 2880, seed: int = 42):
    GP_load = data_load_gen(seed=seed)
    GP_PV = data_PV_gen(seed=seed)

    arr = np.arange(L) * 0.25
    scn_load = sample_nonnegative_gp_trajectories(GP_load, arr, S, seed=seed)
    scn_PV = sample_nonnegative_gp_trajectories(GP_PV, arr, S, seed=seed)

    return scn_load, scn_PV


if __name__ == "__main__":
    scn_data_gen()
