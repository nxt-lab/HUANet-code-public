"""Plot grouped box plot comparing solver times across n_var scenarios."""
import os

import numpy as np
import matplotlib

if os.environ.get("DISPLAY"):
    try:
        import tkinter  # noqa: F401
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker


DATA_FILES = [
    (
        "data/energy/n_var_385/benchmark_plots/benchmark_nvar_385_eq_193_ineq_483.npz",
        "data/energy/n_var_385/benchmark_plots/benchmark_nvar_385_admm.npz",
    ),
]

SOLVERS = [
    ("HUANet", "huanet_times", "#FF2020"),
    ("ADMM", "admm_times", "#006400"),
]

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


def load_data(benchmark_npz_path: str, admm_npz_path: str) -> dict:
    benchmark_data = np.load(benchmark_npz_path, allow_pickle=True)
    admm_data = np.load(admm_npz_path, allow_pickle=True)

    def get(data, key):
        values = np.asarray(data[key], dtype=float) if key in data else None
        if values is not None and values.size <= 1:
            source_path = admm_npz_path if key == "admm_times" else benchmark_npz_path
            print(
                f"WARNING: {key} in {source_path} has only {values.size} value; "
                "box plot will collapse to a line."
            )
        return values

    return {
        "n_var": int(benchmark_data["n_var"]) if "n_var" in benchmark_data else None,
        "admm_times": get(admm_data, "admm_times"),
        "huanet_times": get(benchmark_data, "huanet_times"),
    }


def _draw_boxes(ax, t: dict, box_width: float = 0.3) -> list:
    spacing = 0.35
    positions = np.arange(len(SOLVERS)) * spacing + 1
    legend_handles = []
    for i, (label, key, color) in enumerate(SOLVERS):
        if t[key] is None:
            continue
        bp = ax.boxplot(
            [t[key]],
            positions=[positions[i]],
            widths=box_width * 0.85,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
        )
        for patch in bp["boxes"]:
            patch.set(facecolor=color, alpha=0.7, linewidth=0.5)
        for elem in ["whiskers", "caps", "medians"]:
            for line in bp[elem]:
                line.set(color="black", linewidth=0.5)
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7,
                           edgecolor="black", linewidth=0.5, label=label)
        )
    ax.set_xticks([np.mean(positions)])
    ax.set_xticklabels([f"$n_x = {t['n_var']}$"])
    return legend_handles


def plot_grouped_box(all_data: list[dict], output_path: str | None = None) -> None:
    plt.rcParams.update(RC)

    fig, axes = plt.subplots(1, len(all_data), figsize=(2.5 * len(all_data), 2.35),
                              sharey=False)
    if len(all_data) == 1:
        axes = [axes]

    legend_handles = None
    for i, (ax, t) in enumerate(zip(axes, all_data)):
        handles = _draw_boxes(ax, t)
        if legend_handles is None:
            legend_handles = handles
        ax.set_ylabel("Time (seconds)" if i == 0 else "")
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.ScalarFormatter(useMathText=True)
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    axes[-1].legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=9, title_fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, left=0.08, right=0.98, top=0.97, bottom=0.08)

    media_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "media")
    os.makedirs(media_dir, exist_ok=True)
    save_path = output_path or os.path.join(media_dir, "box_plot_energy_admm.pdf")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def main() -> None:
    all_data = []
    for benchmark_path, admm_path in DATA_FILES:
        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")
        if not os.path.exists(admm_path):
            raise FileNotFoundError(f"ADMM benchmark file not found: {admm_path}")
        all_data.append(load_data(benchmark_path, admm_path))

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "media", "box_plot_energy_admm.pdf"
    )
    plot_grouped_box(all_data, output_path=output_path)


if __name__ == "__main__":
    main()
