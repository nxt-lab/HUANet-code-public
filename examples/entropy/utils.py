import numpy as np


def benchmark_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    b_test: np.ndarray,
    d_test: np.ndarray,
    run_time: np.ndarray,
) -> dict[str, float]:
    x_pred = np.clip(y_pred, 1e-15, None)
    x_true = np.clip(y_true, 1e-15, None)
    obj_pred = np.sum(x_pred * np.log(x_pred), axis=1)
    obj_true = np.sum(x_true * np.log(x_true), axis=1)

    eq_error = np.max(np.abs(y_pred @ A.T - b_test), axis=1)
    ineq_error = np.max(np.maximum(0.0, y_pred @ C.T - d_test), axis=1)
    gap_percent = np.abs((obj_pred - obj_true) / np.maximum(np.abs(obj_true), 1e-10)) * 100

    return {
        "Mean obj gap (%)": float(np.mean(gap_percent)),
        "Max obj gap (%)": float(np.max(gap_percent)),
        "Mean eq violation": float(np.mean(eq_error)),
        "Max eq violation": float(np.max(eq_error)),
        "Mean ineq violation": float(np.mean(ineq_error)),
        "Max ineq violation": float(np.max(ineq_error)),
        "Mean solve time (s)": float(np.mean(run_time)),
        "Max solve time (s)": float(np.max(run_time)),
    }


def eq_viol(y: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(y @ A.T - b, ord=np.inf, axis=1)


def ineq_viol(y: np.ndarray, C: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.max(np.maximum(0.0, y @ C.T - d), axis=1)


def compute_metrics(y, times, y_ref, samples, matrices):
    A, C = matrices["A"], matrices["C"]
    b, d = samples["b"], samples["d"]
    x = np.clip(y, 1e-15, None)
    obj = np.sum(x * np.log(x), axis=1)
    return {
        "y": y,
        "times": times,
        "obj": obj,
        "eq_viol": eq_viol(y, A, b),
        "ineq_viol": ineq_viol(y, C, d),
        "metrics": benchmark_metrics(y, y_ref, A, C, b, d, times),
    }


def print_metrics_table(title: str, metrics: dict[str, float]) -> None:
    metric_width = 40
    value_width = 28

    print(f"\n{title}")
    print("+" + "-" * (metric_width + 2) + "+" + "-" * (value_width + 2) + "+")
    print(f"| {'Metric':<{metric_width}} | {'Value':>{value_width}} |")
    print("+" + "-" * (metric_width + 2) + "+" + "-" * (value_width + 2) + "+")

    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            value_str = f"{value:.6e}"
        else:
            value_str = str(value)
        print(f"| {key:<{metric_width}} | {value_str:>{value_width}} |")

    print("+" + "-" * (metric_width + 2) + "+" + "-" * (value_width + 2) + "+")


def report_benchmark(summary_path, samples, matrices, solvers, sequential_time) -> None:
    all_solvers = [
        ("clarabel", *solvers["clarabel"], "Clarabel (ground truth)"),
        ("our_method", *solvers["our_method"], "L2O Metrics (vs Clarabel)"),
        ("scs", *solvers["scs"], "SCS Metrics (vs Clarabel)"),
    ]

    save_dict = {
        **samples,
        "n_var": matrices["A"].shape[1],
        "our_method_sequential_time": sequential_time,
    }
    results = {}
    for prefix, y, times, _ in all_solvers:
        result = compute_metrics(y, times, solvers["clarabel"][0], samples, matrices)
        results[prefix] = result
        save_dict.update({f"{prefix}_{key}": value for key, value in result.items()})

    np.savez_compressed(summary_path, **save_dict)

    for prefix, _, _, title in all_solvers:
        print_metrics_table(title, results[prefix]["metrics"])
