from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, ConstantKernel, WhiteKernel


def data_PV_gen(seed: int = 42):
    with (Path(__file__).resolve().parents[1] / "cfg.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    generator_dir = Path(__file__).resolve().parents[3] / cfg["data"]["root_subdir"] / "generator"
    rng = np.random.default_rng(seed)

    # -- 1. PV original 2-day PV generation data -------------------------------
    # 192 samples at 15-min intervals, PV generation
    df_raw = pd.read_csv(generator_dir / "PV_48h_15-min_150kW_San_Diego.csv")
    PV_orig = df_raw.iloc[:, 1].astype(float).to_numpy()

    # -- 2. Extend to 30 days by tiling the daily pattern with small random noise
    n_per_day = 96  # 24 h x 4 samples/h
    n_days = 30
    sigma_day_noise = 1.0  # kW, small day-to-day variation

    day_template = PV_orig[:n_per_day]  # day-1 profile used as the base template

    PV_30d = []
    for _ in range(n_days):
        day = day_template + sigma_day_noise * rng.normal(size=n_per_day)
        PV_30d.extend(day)

    PV_30d = np.asarray(PV_30d, dtype=float)

    n_total = len(PV_30d)  # 6302
    t_h = np.arange(n_total) * 0.25  # hours: 0, 0.25, ..., 1630.305
    df_30d = pd.DataFrame({
        "time_h": t_h,
        "PV_kw": PV_30d,
    })
    df_30d.to_csv(generator_dir / "PV_30days.csv", index=False)
    print(f"Extended dataset ({n_total} pts) saved -> PV_30days.csv")

    # -- 3. Build and train Gaussian Process -----------------------------------
    # Input x: (n x 1) matrix for scikit-learn 1-D inputs
    X = t_h.reshape(-1, 1)
    y = PV_30d

    # Kernel: periodic with a fixed 24-hour period.
    period_h = 24.0
    kernel = (
        ConstantKernel(1.0)
        * ExpSineSquared(
            length_scale=1.0,
            periodicity=period_h,
            periodicity_bounds="fixed",
        )
        + WhiteKernel(noise_level=sigma_day_noise**2)
    )
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

    # -- 4. Predict over the training horizon and save -------------------------
    mu, sigma_pred = gp.predict(X, return_std=True)

    df_gp = pd.DataFrame({
        "time_h": t_h,
        "PV_kw": PV_30d,
        "gp_mean": mu,
        "gp_std": sigma_pred,
        "gp_lower": mu - 2 * sigma_pred,
        "gp_upper": mu + 2 * sigma_pred,
    })
    df_gp.to_csv(generator_dir / "PV_30days_gp.csv", index=False)
    print("GP predictions saved -> PV_30days_gp.csv")

    df_saved = pd.read_csv(generator_dir / "PV_30days_gp.csv")
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
        df_saved["PV_kw"],
        color="black",
        linewidth=1,
        alpha=0.30,
        label="Saved PV data",
    )

    plt.xlabel("Time (h)")
    plt.ylabel("PV (kW)")
    plt.title("30-day PV generation")
    plt.legend(loc="upper right")
    plt.grid(True)

    plot_path = generator_dir / "PV_30days_gp.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"30-day data plot saved -> {plot_path}")

    return gp


if __name__ == "__main__":
    data_PV_gen()
