import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=20")

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax.core import freeze

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from admm.admm import admm
from dc3.method import run_dc3
from examples.energy.utils import (
    EnergyPrimeNet,
    cvxpy_objective,
    energy_cost,
    energy_slices,
    energy_slack_form,
    equality_rhs,
    normalize_energy_q,
    report_benchmark,
)
from huanet.neural_layer import HUANet
from huanet.utils import precompute

jax.config.update("jax_enable_x64", True)


def make_l2o_solver(
    nn_params,
    model: HUANet,
    E: jnp.ndarray,
    E_pinv: jnp.ndarray,
    d_ineq: jnp.ndarray,
    net_demand: jnp.ndarray,
    N: int,
    n_var: int,
    n_ineq: int,
    n_admm: int,
    rho: float,
):
    @jax.pmap
    def solve(features_batch: jnp.ndarray, lam_batch: jnp.ndarray) -> jnp.ndarray:
        n_samples = lam_batch.shape[0]
        b = equality_rhs(lam_batch, net_demand, N)
        d = jnp.broadcast_to(d_ineq, (n_samples, n_ineq))
        eta = jnp.concatenate([b, d], axis=-1)
        w = jnp.zeros((n_samples, n_ineq))
        v = jnp.zeros_like(w)
        x = jnp.zeros((n_samples, n_var))

        def body_fn(_, carry):
            w_k, v_k, _ = carry
            q = w_k - v_k / rho
            q_network = normalize_energy_q(q, model.cfg["problem"])
            x_next, slack, _ = model.apply(
                {"params": nn_params}, q_network, features_batch, E, E_pinv, eta
            )
            w_next = jnp.maximum(0.0, slack + v_k / rho)
            v_next = v_k + rho * (slack - w_next)
            return w_next, v_next, x_next

        return jax.lax.fori_loop(0, n_admm, body_fn, (w, v, x))[2]

    return solve


def baseline_solver(
    lam_samples: np.ndarray,
    net_demand: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    p: dict,
) -> tuple[np.ndarray, np.ndarray]:
    x = cp.Variable(A.shape[1])
    slack = cp.Variable(C.shape[0], nonneg=True)
    b = cp.Parameter(A.shape[0])
    problem = cp.Problem(cp.Minimize(cvxpy_objective(x, p)), [A @ x == b, C @ x + slack == d])
    if not problem.is_dcp():
        raise ValueError("The smooth energy reference problem is not DCP canonicalizable.")

    x_batch, solve_times = [], []
    for lam in lam_samples:
        b.value = np.asarray(equality_rhs(
            jnp.asarray(lam).reshape(1, 1), jnp.asarray(net_demand), int(p["N"])
        ))[0]
        start = time.perf_counter()
        problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
        elapsed = time.perf_counter() - start
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x.value is None:
            raise RuntimeError(f"Clarabel reference failed with status {problem.status}.")
        value = np.asarray(x.value)
        eq_residual = np.max(np.abs(A @ value - b.value))
        ineq_residual = np.max(np.maximum(C @ value - d, 0.0))
        _, _, pc_slice, _ = energy_slices(int(p["N"]))
        pc = value[pc_slice]
        if (
            not np.all(np.isfinite(value))
            or np.min(pc) < float(p["pc_min"]) - 1e-6
            or max(eq_residual, ineq_residual) > 1e-5
        ):
            raise RuntimeError(
                f"Invalid Clarabel reference: eq={eq_residual:.3e}, ineq={ineq_residual:.3e}, min_pc={np.min(pc):.3e}."
            )
        x_batch.append(value)
        solve_times.append(elapsed)
    return np.stack(x_batch), np.asarray(solve_times)


def primal_solver(q, lam, net_demand, A, C, d, p, rho):
    problem, x, slack = energy_slack_form(lam, net_demand, A, C, d, p, q, rho)
    start = time.perf_counter()
    problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
    elapsed = time.perf_counter() - start
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x.value is None or slack.value is None:
        raise RuntimeError(f"Clarabel ADMM primal step failed with status {problem.status}.")
    return np.asarray(x.value), np.asarray(slack.value), elapsed


def _validate_metadata(data, saved, p: dict) -> None:
    expected = {
        "schema_version": str(p["schema_version"]),
        "forecast_day": int(p["forecast_day"]),
        "pc_min": float(p["pc_min"]),
        "pc_max": float(p["pc_max"]),
        "m_min": float(p["m_min"]),
        "m_max": float(p["m_max"]),
    }
    for name, value in expected.items():
        data_value = data[name].item()
        model_value = saved[name].item()
        if data_value != value or model_value != value:
            raise ValueError(f"Metadata mismatch for {name}: dataset={data_value}, model={model_value}, expected={value}.")
    for name in ("delta", "kappa_m", "kappa_c"):
        if saved[name].item() != float(p[name]):
            raise ValueError(f"Checkpoint smoothing parameter {name} does not match configuration.")
    for name in (
        "Delta_t", "B", "c_e", "c_p", "c_d", "mu", "a", "delta", "kappa_m", "kappa_c",
        "u_min", "u_max", "pc_min", "pc_max", "m_min", "m_max", "s_min", "s_max",
    ):
        if data[name].item() != float(p[name]):
            raise ValueError(f"Dataset parameter {name} does not match configuration.")
    for name in ("A", "C", "d", "net_demand"):
        if not np.array_equal(data[name], saved[name]):
            raise ValueError(f"Checkpoint and dataset contain different {name} values.")
    if data["J_ref"].item() != saved["J_ref"].item() or data["J_ref"].item() <= 0.0:
        raise ValueError("Checkpoint and dataset contain different or invalid J_ref values.")


def main() -> None:
    with Path(__file__).with_name("cfg.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    p = cfg["problem"]
    N = int(p["N"])
    n_var, n_eq, n_ineq = int(p["n_var"]), int(p["n_eq"]), int(p["n_ineq"])
    n_samples = int(cfg["data"]["n_samples"])
    scenario_root = Path(__file__).resolve().parents[2] / cfg["data"]["root_subdir"] / f"n_var_{n_var}"
    dataset_path = scenario_root / "datasets" / f"datasets_{n_samples}.npz"
    schema_version = str(p["schema_version"])
    model_path = scenario_root / "model_params" / f"huanet_params_{schema_version}_{n_samples}_n{n_var}_eq{n_eq}_ineq{n_ineq}.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run generate.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    with np.load(dataset_path) as data, np.load(model_path, allow_pickle=True) as saved:
        _validate_metadata(data, saved, p)
        A, C, d = (np.asarray(data[name]) for name in ("A", "C", "d"))
        net_demand = np.asarray(data["net_demand"])
        lam_all = np.asarray(data["lam"])
        features_all = np.asarray(data["lambda_features"])
        J_ref = float(data["J_ref"])
        nn_params = saved["params"].item()["nn"]

    n_total = len(lam_all)
    n_train = max(1, int(n_total * float(cfg["data"]["train_percent"])))
    n_val = max(1, int(n_total * float(cfg["data"]["val_percent"])))
    n_train = min(n_train, n_total - n_val - 1)
    lam_test = lam_all[n_train + n_val:]
    lam_eval = lam_test
    features_eval = features_all[n_train + n_val:]
    n_eval = len(lam_eval)

    model_cfg = freeze({
        "problem": p,
        "neural_net": {**cfg["neural_net"], "hidden_layers": tuple(cfg["neural_net"]["hidden_layers"])},
    })
    model = HUANet(model_cfg, prime_net_cls=EnergyPrimeNet)
    E, E_pinv = precompute(jnp.asarray(A), jnp.asarray(C))
    n_admm, rho = int(cfg["training"]["n_admm"]), float(cfg["admm"]["rho"])
    l2o_solve = make_l2o_solver(
        nn_params, model, E, E_pinv, jnp.asarray(d), jnp.asarray(net_demand), N,
        n_var, n_ineq, n_admm, rho,
    )

    n_devices = jax.local_device_count()
    print(f"Using {n_devices} CPU devices via pmap")
    pad = (n_devices - n_eval % n_devices) % n_devices
    features = jnp.asarray(features_eval)
    lam = jnp.asarray(lam_eval)
    if pad:
        lam_padding = jnp.full((pad, 1), 0.65, dtype=lam.dtype)
        feature_padding = jnp.full((pad, 1), 0.6, dtype=features.dtype)
        features = jnp.concatenate([features, feature_padding], axis=0)
        lam = jnp.concatenate([lam, lam_padding], axis=0)

    warm_features = jnp.full((n_devices, 1, 1), 0.6, dtype=features.dtype)
    warm_lam = jnp.full((n_devices, 1, 1), 0.65, dtype=lam.dtype)
    _ = l2o_solve(warm_features, warm_lam).block_until_ready()

    l2o_times, x_parts = [], []
    for start_index in range(0, len(lam), n_devices):
        stop_index = start_index + n_devices
        features_i = features[start_index:stop_index, None, :]
        lam_i = lam[start_index:stop_index, None, :]
        start = time.perf_counter()
        x_i = l2o_solve(features_i, lam_i)
        x_i.block_until_ready()
        elapsed = time.perf_counter() - start
        x_parts.append(np.asarray(x_i[:, 0, :]))
        l2o_times.extend([elapsed / n_devices] * n_devices)
    x_pred = np.vstack(x_parts)[:n_eval]
    l2o_times = np.asarray(l2o_times[:n_eval])
    sequential_time = float(np.mean(l2o_times))

    repeated = l2o_solve(
        warm_features.at[:, 0, :].set(features_eval[0]),
        warm_lam.at[:, 0, :].set(lam_eval[0]),
    )
    repeated.block_until_ready()
    repeated_np = np.asarray(repeated[:, 0, :])
    expected_np = np.broadcast_to(x_pred[0], repeated_np.shape)
    np.testing.assert_allclose(repeated_np, expected_np, rtol=1e-9, atol=1e-9)

    print("Running Clarabel with the canonical smooth energy objective...")
    x_clarabel, times_clarabel = baseline_solver(lam_eval, net_demand, A, C, d, p)
    b = np.asarray(equality_rhs(jnp.asarray(lam_eval), jnp.asarray(net_demand), N))
    d_batch = np.broadcast_to(d, (n_eval, n_ineq))
    objective_huanet = energy_cost(x_pred, p)
    objective_reference = energy_cost(x_clarabel, p)
    equality_violation = np.max(np.abs(x_pred @ A.T - b), axis=1)
    inequality_violation = np.max(np.maximum(x_pred @ C.T - d_batch, 0.0), axis=1)
    gap_valid = (
        np.isfinite(objective_huanet)
        & np.isfinite(objective_reference)
        & (equality_violation <= 1e-6)
        & (inequality_violation <= 1e-6)
    )
    relative_gap = (
        np.abs(objective_huanet[gap_valid] - objective_reference[gap_valid])
        / np.maximum(np.abs(objective_reference[gap_valid]), 1e-10)
        * 100.0
    )
    if relative_gap.size:
        print(f"HUANet mean optimality gap: {np.mean(relative_gap):.6e}%")
        print(f"HUANet max optimality gap:  {np.max(relative_gap):.6e}%")
    else:
        print("HUANet mean optimality gap: unavailable (no feasible predictions)")
        print("HUANet max optimality gap:  unavailable (no feasible predictions)")
    x_admm, times_admm = np.zeros((n_eval, n_var)), np.zeros(n_eval)
    print(f"Running ADMM on {n_eval} samples...")
    for index, lam_i in enumerate(lam_eval):
        x_admm[index], times_admm[index] = admm(
            lambda q, value=lam_i: primal_solver(q, value, net_demand, A, C, d, p, rho),
            n_ineq=n_ineq,
            max_iter=int(cfg["admm"]["max_iter"]),
            tol=float(cfg["admm"]["tolerance"]),
            rho=rho,
        )

    print("Running DC3 on the identical test instances...")
    x_dc3, times_dc3 = run_dc3(lam_eval, scenario_root)

    benchmark_dir = scenario_root / "benchmark_plots"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary_path = benchmark_dir / f"benchmark_nvar_{n_var}_eq_{n_eq}_ineq_{n_ineq}.npz"
    report_benchmark(
        summary_path,
        samples={
            "lam": lam_eval, "features": features_eval, "b": b, "d": d_batch,
            "net_demand": net_demand,
        },
        matrices={"A": A, "C": C},
        solvers={
            "clarabel": (x_clarabel, times_clarabel),
            "our_method": (x_pred, l2o_times),
            "admm": (x_admm, times_admm),
            "dc3": (x_dc3, times_dc3),
        },
        sequential_time=sequential_time,
        p=p,
        J_ref=J_ref,
    )
    print(f"Saved benchmark summary: {summary_path}")


if __name__ == "__main__":
    main()
