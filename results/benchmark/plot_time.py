"""Plot box plots comparing energy solver times."""
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
    "data/energy_soc_day_0/n_var_384/benchmark_plots/benchmark_nvar_384_eq_193_ineq_480.npz",
]

SOLVERS = [
    ("HUANet", "our_method_times", "#2CA02C"),
    ("Clarabel", "clarabel_times", "#1F77B4"),
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


def load_data(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)

    def get(key):
        return np.asarray(data[key], dtype=float) if key in data else None

    return {
        "n_var": int(data["n_var"]) if "n_var" in data else None,
        "our_method_times": get("our_method_times"),
        "clarabel_times": get("clarabel_times"),
    }


def _draw_boxes(ax, t: dict, box_width: float = 0.5) -> None:
    positions = np.arange(len(SOLVERS))
    for i, (label, key, color) in enumerate(SOLVERS):
        if t[key] is None:
            continue
        bp = ax.boxplot(
            [t[key]],
            positions=[positions[i]],
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
        )
        for patch in bp["boxes"]:
            patch.set(facecolor=color, alpha=0.7, linewidth=0.5)
        for elem in ["whiskers", "caps", "medians"]:
            for line in bp[elem]:
                line.set(color="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([label for label, _, _ in SOLVERS])


def plot_grouped_box(all_data: list[dict], output_path: str | None = None) -> None:
    plt.rcParams.update(RC)

    fig, axes = plt.subplots(1, len(all_data), figsize=(2.5 * len(all_data), 2.35),
                              sharey=False)
    if len(all_data) == 1:
        axes = [axes]

    for i, (ax, t) in enumerate(zip(axes, all_data)):
        _draw_boxes(ax, t)
        ax.set_ylabel("Time (seconds)" if i == 0 else "")
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.ScalarFormatter(useMathText=True)
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, left=0.08, right=0.98, top=0.97, bottom=0.08)

    media_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "media")
    os.makedirs(media_dir, exist_ok=True)
    save_path = output_path or os.path.join(media_dir, "box_plot_energy.pdf")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def main() -> None:
    all_data = []
    for data_path in DATA_FILES:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Benchmark file not found: {data_path}")
        all_data.append(load_data(data_path))

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "media", "box_plot_energy.pdf"
    )
    plot_grouped_box(all_data, output_path=output_path)


if __name__ == "__main__":
    main()
