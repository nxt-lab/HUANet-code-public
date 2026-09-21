from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, ConstantKernel, WhiteKernel

def data_load_gen(seed: int = 42):
    with (Path(__file__).resolve().parents[1] / "cfg.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    load_dir = Path(__file__).resolve().parents[3] / cfg["data"]["root_subdir"] / "load"
    rng = np.random.default_rng(seed)
    df_raw = pd.read_csv(load_dir / "load_15min_max100kW_SanDiego_Building.csv")
    load_orig = df_raw.iloc[:, 1].astype(float).to_numpy()

    n_per_day = 96
    n_days = 30
    sigma_day_noise = 1.

    day_template = load_orig[:n_per_day]

    load_30d = []

    for _ in range(n_days):
        day = day_template + sigma_day_noise * rng.normal(size=n_per_day)
        load_30d.extend(day)

    load_30d = np.asarray(load_30d, dtype=float)

    n_total = len(load_30d)
    t_h = np.arange(n_total) * 0.25

    df_30d = pd.DataFrame({
        "time_h": t_h,
        "load_kw": load_30d,
    })

    df_30d.to_csv(load_dir / "load_30days.csv", index=False)
    print(f"Extended dataset ({n_total} pts) saved -> load_30days.csv")

    X = t_h.reshape(-1, 1)
    y = load_30d
    period_h = 24.0
    kernel = (ConstantKernel(1.0) * ExpSineSquared(length_scale=1.0, periodicity=period_h, periodicity_bounds="fixed")) + WhiteKernel(noise_level=sigma_day_noise**2)
    gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=False,
    n_restarts_optimizer=5,
    random_state=seed,
    )

    gp.fit(X, y)

    print(f"Optimized kernel: {gp.kernel_}")
    print(f"Log-marginal likelihood: {gp.log_marginal_likelihood_value_:.3f}")
    print(f"Fixed GP period: {period_h} h")

    mu, sigma_pred = gp.predict(X, return_std=True)

    df_gp = pd.DataFrame({
        "time_h": t_h,
        "load_kw": load_30d,
        "gp_mean": mu,
        "gp-std": sigma_pred,
        "gp_lower": mu - 2 * sigma_pred,
        "gp_upper": mu + 2 * sigma_pred,
    })

    df_gp.to_csv(load_dir / "load_30days_gp.csv")
    print(f"GP predictions saved")

    df_saved = pd.read_csv(load_dir / "load_30days_gp.csv")
    plt.figure(figsize=(12, 5))

    plt.plot(
        df_saved["time_h"],
        df_saved["gp_mean"],
        color="blue",
        linewidth=2,
        label="GP mean and 95% interval",
    )
    plt.fill_between(
    df_saved["time_h"],
    df_saved["gp_lower"],
    df_saved["gp_upper"],
    color="blue",
    alpha=0.2,
    )

    plt.plot(
        df_saved["time_h"],
        df_saved["load_kw"],
        color="black",
        linewidth=1,
        alpha=0.30,
        label="Saved load data",
    )

    plt.xlabel("Time (h)")
    plt.ylabel("load (kW)")
    plt.title("30-day load generation")
    plt.legend(loc="upper right")
    plt.grid(True)

    plot_path = load_dir / "load_30days_gp.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"30-day data plot saved -> {plot_path}")

    return gp


if __name__ == "__main__":
    data_load_gen()
