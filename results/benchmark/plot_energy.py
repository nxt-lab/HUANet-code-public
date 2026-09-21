"""Plot the HUANet (L2O) trajectories of the energy MPC problem.

The prediction is produced here rather than read from a benchmark summary: the
script rebuilds run_em.py's unrolled-ADMM solver from the scenario checkpoint
and evaluates it on one calendar day aligned to midnight, so the horizon spans
0 h to 24 h of real clock time.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
# Set before JAX is imported, matching run_em.py: inference here is a single
# sample and must not fight other jobs for the GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "examples" / "energy"))

import jax.numpy as jnp  # noqa: E402
from flax.core import freeze  # noqa: E402
from run_em import make_l2o_solver  # noqa: E402
from generate_em import (  # noqa: E402
    DATASET_CFG,
    PROBLEM_CFG,
    energy_slices,
    imported_slice,
    load_config,
    midnight_window,
    resolve_scenario_dirs,
)
from examples.energy.utils import EnergyPrimeNet  # noqa: E402
from huanet.neural_layer import HUANet  # noqa: E402

FIGURE_DIR = PROJECT_ROOT / "media"
DEFAULT_DAY = 0
CONFIG_PATH = PROJECT_ROOT / "src" / "examples" / "energy" / "config_em.yaml"

RC = {
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "mathtext.fontset": "dejavusans",
}


def load_scenario(n_var: int | None) -> tuple[np.lib.npyio.NpzFile, np.lib.npyio.NpzFile]:
    """Open the problem data and trained checkpoint of the scenario."""
    cfg = load_config(str(CONFIG_PATH))
    n_var = int(cfg["training"]["scenario_n_var"]) if n_var is None else int(n_var)

    ctx = {**PROBLEM_CFG, **DATASET_CFG, "n_var": n_var}
    scenario_dirs = resolve_scenario_dirs(n_var, str(DATASET_CFG["root_subdir"]))
    problem_path = Path(scenario_dirs["problem_data"]) / (
        "problem_data_{n_samples}_n{n_var}_eq{n_eq}_ineq{n_ineq}.npz".format(**ctx)
    )
    model_path = Path(scenario_dirs["model_params"]) / (
        "huanet_params_{n_samples}_n{n_var}_eq{n_eq}_ineq{n_ineq}.npz".format(**ctx)
    )
    if not problem_path.is_file():
        raise FileNotFoundError(f"Problem data not found: {problem_path}\nRun generate_em.py first.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}\nRun train_em.py first.")
    if model_path.stat().st_mtime < problem_path.stat().st_mtime:
        print(
            f"WARNING: {model_path.name} is older than the problem data; the checkpoint may have "
            "been trained on a different objective. Re-run train_em.py."
        )
    return np.load(problem_path), np.load(model_path, allow_pickle=True)


def solve_huanet(problem: np.lib.npyio.NpzFile, saved: np.lib.npyio.NpzFile, demand: np.ndarray) -> np.ndarray:
    """Run run_em.py's unrolled-ADMM solver on a single demand trajectory."""
    A = jnp.array(problem["A"])
    C = jnp.array(problem["C"])
    b = np.asarray(problem["b"], dtype=np.float64)
    d_ineq = np.asarray(
        problem["d_ineq"] if "d_ineq" in problem.files else problem["d"], dtype=np.float64
    )
    n_x = int(problem["N_VAR"])
    n_eq = int(problem["N_EQ"])
    n_in = int(problem["N_INEQ"])

    E = jnp.array(problem["E"])
    correction_gain = jnp.array(problem["correction_gain"])

    model_cfg = load_config(str(CONFIG_PATH))
    model = HUANet(freeze({
        "problem": {**model_cfg["problem"], "n_var": n_x, "n_eq": n_eq, "n_ineq": n_in},
        "neural_net": {
            **model_cfg["neural_net"],
            "hidden_layers": tuple(model_cfg["neural_net"]["hidden_layers"]),
        },
    }), prime_net_cls=EnergyPrimeNet)
    cfg = load_config(str(CONFIG_PATH))["training"]
    rho = float(saved["rho"]) if "rho" in saved.files else float(cfg["rho"])
    n_admm = int(saved["n_admm"]) if "n_admm" in saved.files else int(cfg["n_admm"])
    print(f"HUANet inference: n_admm={n_admm}  rho={rho:.2e}")

    solve = make_l2o_solver(
        nn_params=saved["params"].item()["nn"],
        model=model,
        E=E,
        correction_gain=correction_gain,
        d_ineq=jnp.array(d_ineq),
        n_eq=n_eq,
        S_m=float(problem["S_m"]) if "S_m" in problem.files else 1.0,
        n_x=n_x,
        n_in=n_in,
        n_admm=n_admm,
        rho=rho,
    )

    center = problem["parameter_center"]
    scale = problem["parameter_scale"]
    features = jnp.array(((demand - center) / scale)[None, None, :])
    raw = jnp.array(np.asarray(demand, dtype=np.float64)[None, None, :])
    return np.asarray(solve(features, raw)[0, 0], dtype=np.float64)


def scaled_to_horizons(
    x_sample: np.ndarray,
    demand_sample: np.ndarray,
    N: int,
    S_e: float,
    S_B: float,
    gamma: float,
    Delta_t: float,
    b_bess: float,
    sigma_0: float,
    S_m: float = 1.0,
) -> dict[str, np.ndarray]:
    e_sl, pb_sl, r_sl = energy_slices(N)
    e_physical = S_e * x_sample[e_sl]
    pb = S_B * x_sample[pb_sl]
    pc = gamma * x_sample[r_sl]
    sigma = sigma_0 + (Delta_t / b_bess) * e_physical
    imported = S_m * x_sample[imported_slice(N)]
    return {
        "sigma": sigma,
        "pb": pb,
        "pc": pc,
        "demand": demand_sample[:N],
        "imported": imported,
    }


def horizon_xticks(horizon_hours: float) -> np.ndarray:
    step = 2 if horizon_hours <= 24 else 4
    return np.arange(0.0, np.ceil(horizon_hours) + step, step)


def zoomed_limits(
    series: np.ndarray, pad_fraction: float = 0.08, min_pad: float = 0.02
) -> tuple[float, float]:
    ymin = float(np.min(series))
    ymax = float(np.max(series))
    pad = max(pad_fraction * max(ymax - ymin, np.finfo(float).eps), min_pad)
    return ymin - pad, ymax + pad


def plot_energy_prediction(
    n_var: int | None = None,
    day: int | None = None,
    output_path: Path | None = None,
) -> Path:
    plt.rcParams.update(RC)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    problem, saved = load_scenario(n_var)
    with problem, saved:
        n_x = int(problem["N_VAR"])
        N = int(problem["N"])
        S_e = float(problem["S_e"])
        S_B = float(problem["S_B"])
        gamma = float(problem["gamma"])
        Delta_t = float(problem["Delta_t"])
        b_bess = float(problem["b_bess"])
        sigma_0 = float(problem["sigma_0"])
        S_m = float(problem["S_m"]) if "S_m" in problem.files else 1.0

        day_index = DEFAULT_DAY if day is None else int(day)
        p_load, p_pv = midnight_window(day_index, N, Delta_t)
        demand = p_load - p_pv

        print(f"Running HUANet for n_var={n_x}, day {day_index} (00:00 to {N * Delta_t:g}:00)...")
        x_pred = solve_huanet(problem, saved, demand)

    huanet = scaled_to_horizons(x_pred, demand, N, S_e, S_B, gamma, Delta_t, b_bess, sigma_0, S_m)

    stage_edges_hours = np.arange(N + 1) * Delta_t
    soc_time_hours = np.arange(N + 1) * Delta_t
    horizon_hours = N * Delta_t
    xticks = horizon_xticks(horizon_hours)

    panels = [
        ("imported", r"$m$", "kW", None),
        ("pc", r"$P^c$", "kW", "pc"),
        ("pb", r"$P^B$", "kW", None),
        ("sigma", "SOC", "%", None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), dpi=300, constrained_layout=True)
    axes_flat = axes.ravel()

    huanet_color = "#2CA02C"

    for panel_id, (key, ylabel, unit, zoom_key) in enumerate(panels):
        ax = axes_flat[panel_id]
        hua_series = 100.0 * huanet[key] if key == "sigma" else huanet[key]
        if key == "sigma":
            ax.plot(
                soc_time_hours,
                hua_series,
                color=huanet_color,
                linewidth=2.0,
                label="HUANet",
            )
        else:
            ax.stairs(
                hua_series,
                stage_edges_hours,
                color=huanet_color,
                linewidth=2.0,
                label="HUANet",
            )
        ax.set_xlim(0.0, horizon_hours)
        ax.set_xticks(xticks)
        ax.set_xlabel(f"Time of day (hour)\n({chr(ord('a') + panel_id)})")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#d8d8d8", linewidth=0.7, alpha=0.8)
        ax.text(0.03, 1.03, unit, transform=ax.transAxes, fontsize=10, ha="left", va="bottom")
        if zoom_key == "pc":
            ax.set_ylim(zoomed_limits(hua_series, pad_fraction=2.0, min_pad=0.05))

    if output_path is None:
        output_path = FIGURE_DIR / f"energy-prediction-nvar-{n_x}-day-{day_index}.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved energy prediction plot to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot HUANet trajectories over one midnight-aligned 24 h day."
    )
    parser.add_argument(
        "day",
        nargs="?",
        type=int,
        default=None,
        help="Calendar day of the 30-day series to plot, aligned to midnight (default 0).",
    )
    parser.add_argument(
        "--n-var",
        type=int,
        default=None,
        help="Scenario n_var to evaluate. Defaults to scenario_n_var in config_em.yaml.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output PDF path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_energy_prediction(n_var=args.n_var, day=args.day, output_path=args.output)


if __name__ == "__main__":
    main()
