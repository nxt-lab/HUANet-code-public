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
from examples.energy.utils import EnergyPrimeNet, energy_slack_form, energy_slices, equality_rhs, report_benchmark
from huanet.neural_layer import HUANet
from huanet.utils import precompute

jax.config.update("jax_enable_x64", True)
print(jax.devices())

def make_l2o_solver(
    nn_params,
    model: HUANet,
    E: jnp.ndarray,
    E_pinv: jnp.ndarray,
    d_ineq: jnp.ndarray,
    S_m: float,
    n_var: int,
    n_ineq: int,
    n_admm: int,
    rho: float,
):
    @jax.pmap
    def solve(features_batch: jnp.ndarray, lam_batch: jnp.ndarray) -> jnp.ndarray:
        n_samples = lam_batch.shape[0]
        b = equality_rhs(lam_batch, lam_batch.shape[-1], S_m)
        d = jnp.broadcast_to(d_ineq, (n_samples, n_ineq))
        eta = jnp.concatenate([b, d], axis=-1)
        w_init = jnp.zeros((n_samples, n_ineq))
        v_init = jnp.zeros_like(w_init)
        x_init = jnp.zeros((n_samples, n_var))

        def body_fn(_, carry):
            w_k, v_k, _ = carry
            q = w_k - v_k / rho
            x_next, s_next, _ = model.apply({"params": nn_params}, q, features_batch, E, E_pinv, eta)
            w_next = jnp.maximum(0.0, s_next + v_k / rho)
            v_next = v_k + rho * (s_next - w_next)
            return w_next, v_next, x_next

        return jax.lax.fori_loop(0, n_admm, body_fn, (w_init, v_init, x_init))[2]

    return solve


def baseline_solver(
    lam_samples: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    problem_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    N = int(problem_cfg["N"])
    _, pb, r = energy_slices(N)
    m = slice(3 * N + 1, 4 * N + 1)
    x = cp.Variable(A.shape[1])
    s = cp.Variable(C.shape[0], nonneg=True)
    b = cp.Parameter(A.shape[0])
    P_B = float(problem_cfg["S_B"]) * x[pb]
    S_m = float(problem_cfg["S_m"])
    imported = S_m * x[m]
    cycling = (1.0 - float(problem_cfg["mu"])) / (2.0 * np.sqrt(float(problem_cfg["mu"])))
    f = (
        float(problem_cfg["p_k"]) * float(problem_cfg["Delta_t"]) * cp.sum(imported + cycling * cp.abs(P_B))
        + float(problem_cfg["p_p"]) * cp.sum(cp.pos(imported))
        + float(problem_cfg["eta"]) * cp.sum(cp.pos(cp.inv_pos(x[r]) - 1.0))
    )
    problem = cp.Problem(cp.Minimize(f), [A @ x == b, C @ x + s == d])

    x_batch, solve_times = [], []
    for lam in lam_samples:
        rhs = np.zeros(A.shape[0])
        rhs[N + 1:] = lam[:N] / S_m
        b.value = rhs
        start = time.perf_counter()
        problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
        elapsed = time.perf_counter() - start
        x_batch.append(np.array(x.value))
        solve_times.append(elapsed)
    return np.stack(x_batch), np.asarray(solve_times)


def primal_solver(q, lam, A, C, d, problem_cfg, rho):
    problem, x, s = energy_slack_form(lam, A, C, d, problem_cfg, q, rho)
    start = time.perf_counter()
    problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
    elapsed = time.perf_counter() - start
    if x.value is None or s.value is None:
        raise RuntimeError(f"Clarabel primal step failed with status {problem.status}")
    return x.value, s.value, elapsed


def main() -> None:
    with Path(__file__).with_name("cfg.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    problem_cfg = cfg["problem"]
    N = int(problem_cfg["N"])
    n_var, n_eq, n_ineq = 4 * N + 1, 2 * N + 1, 5 * N + 3
    n_samples = int(cfg["data"]["n_samples"])
    scenario_root = Path(__file__).resolve().parents[2] / cfg["data"]["root_subdir"] / f"n_var_{n_var}"
    dataset_path = scenario_root / "datasets" / f"datasets_{n_samples}.npz"
    model_path = scenario_root / "model_params" / f"huanet_params_{n_samples}_n{n_var}_eq{n_eq}_ineq{n_ineq}.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run generate.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    with np.load(dataset_path) as data:
        A, C, d = (np.asarray(data[name]) for name in ("A", "C", "d"))
        lam_all = np.asarray(data["lam"])
        features_all = np.asarray(data["lambda_features"])
        J_ref = float(data["J_ref"])
    with np.load(model_path, allow_pickle=True) as saved:
        nn_params = saved["params"].item()["nn"]

    n_total = len(lam_all)
    n_train = max(1, int(n_total * float(cfg["data"]["train_percent"])))
    n_val = max(1, int(n_total * float(cfg["data"]["val_percent"])))
    n_train = min(n_train, n_total - n_val - 1)
    test_start = n_train + n_val
    lam_test_np = lam_all[test_start:]
    features_test_np = features_all[test_start:]
    n_test = len(lam_test_np)
    if not n_test:
        raise ValueError("The configured dataset has no test samples.")

    model_cfg = freeze({
        "problem": {**problem_cfg, "n_var": n_var, "n_eq": n_eq, "n_ineq": n_ineq},
        "neural_net": {
            **cfg["neural_net"],
            "hidden_layers": tuple(cfg["neural_net"]["hidden_layers"]),
        },
    })
    model = HUANet(model_cfg, prime_net_cls=EnergyPrimeNet)
    E, E_pinv = precompute(jnp.array(A), jnp.array(C))
    n_admm = int(cfg["training"]["n_admm"])
    rho = float(cfg["admm"]["rho"])
    l2o_solve = make_l2o_solver(
        nn_params=nn_params,
        model=model,
        E=E,
        E_pinv=E_pinv,
        d_ineq=jnp.array(d),
        S_m=float(problem_cfg["S_m"]),
        n_var=n_var,
        n_ineq=n_ineq,
        n_admm=n_admm,
        rho=rho,
    )

    n_devices = jax.local_device_count()
    print(f"Using {n_devices} CPU devices via pmap")
    pad = (n_devices - n_test % n_devices) % n_devices
    features = jnp.array(features_test_np)
    lam = jnp.array(lam_test_np)
    if pad:
        features = jnp.concatenate([features, jnp.zeros((pad, N), dtype=features.dtype)], axis=0)
        lam = jnp.concatenate([lam, jnp.zeros((pad, N), dtype=lam.dtype)], axis=0)
    # Warm up
    print("Running L2O...")
    features0 = jnp.zeros((n_devices, 1, N), dtype=features.dtype)
    lam0 = jnp.zeros((n_devices, 1, N), dtype=lam.dtype)
    _ = l2o_solve(features0, lam0).block_until_ready()

    # Per-batch timing
    l2o_times = []
    x_parts = []
    for i in range(0, len(lam), n_devices):
        end = i + n_devices
        features_i = features[i:end, None, :]
        lam_i = lam[i:end, None, :]
        start = time.perf_counter()
        x_i = l2o_solve(features_i, lam_i)
        x_i.block_until_ready()
        elapsed = time.perf_counter() - start
        x_parts.append(np.array(x_i[:, 0, :]))
        l2o_times.extend([elapsed / n_devices] * n_devices)

    x_pred = np.vstack(x_parts)[:n_test]
    l2o_times = np.array(l2o_times[:n_test], dtype=float)
    sequential_time = float(np.mean(l2o_times))

    print("Running DC3...")
    x_dc3, times_dc3 = run_dc3(lam_test_np, scenario_root)

    print("Running Clarabel with the exact energy objective...")
    x_clarabel, times_clarabel = baseline_solver(lam_test_np, A, C, d, problem_cfg)
    b = np.asarray(equality_rhs(jnp.asarray(lam_test_np), N, float(problem_cfg["S_m"])))
    d_batch = np.broadcast_to(d, (n_test, n_ineq))

    # ---- ADMM ----
    x_admm = np.zeros((n_test, n_var))
    times_admm = np.zeros(n_test)
    print(f"Running ADMM on {n_test} samples...")
    for i, lam_i in enumerate(lam_test_np):
        x_admm[i], times_admm[i] = admm(
            lambda q: primal_solver(q, lam_i, A, C, d, problem_cfg, rho),
            n_ineq=n_ineq,
            max_iter=int(cfg["admm"]["max_iter"]),
            tol=float(cfg["admm"]["tolerance"]),
            rho=rho,
        )

    # ---- Save results ----
    benchmark_dir = scenario_root / "benchmark_plots"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary_path = benchmark_dir / f"benchmark_nvar_{n_var}_eq_{n_eq}_ineq_{n_ineq}.npz"
    np.savez_compressed(
        benchmark_dir / f"benchmark_nvar_{n_var}_admm.npz",
        n_var=n_var,
        admm_y=x_admm,
        admm_times=times_admm,
    )
    report_benchmark(
        summary_path,
        samples={"lam": lam_test_np, "features": features_test_np, "b": b, "d": d_batch},
        matrices={"A": A, "C": C},
        solvers={
            "clarabel": (x_clarabel, times_clarabel),
            "our_method": (x_pred, l2o_times),
            "dc3": (x_dc3, times_dc3),
        },
        sequential_time=sequential_time,
        problem_cfg=problem_cfg,
        J_ref=J_ref,
    )
    print(f"Saved benchmark summary: {summary_path}")


if __name__ == "__main__":
    main()
