import os
import sys
import time
from pathlib import Path

# Configure CPU devices before importing JAX for the pmap benchmark.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=20")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax.core import freeze

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from admm.admm import admm
from huanet.neural_layer import HUANet
from utils import report_benchmark

jax.config.update("jax_enable_x64", True)
print(jax.devices())


def make_l2o_solver(
    nn_params,
    model: HUANet,
    C: jnp.ndarray,
    n_var: int,
    n_ineq: int,
    n_admm: int,
    rho: float,
):
    @jax.pmap
    def solve(lam_batch: jnp.ndarray, d_batch: jnp.ndarray) -> jnp.ndarray:
        n_samples = lam_batch.shape[0]
        w_init = jnp.zeros((n_samples, n_ineq))
        v_init = jnp.zeros_like(w_init)
        x_init = jnp.zeros((n_samples, n_var))

        def body_fn(_, carry):
            w_k, v_k, _ = carry
            q = w_k - v_k / rho
            x_next, _ = model.apply({"params": nn_params}, q, lam_batch)
            s_next = d_batch - x_next @ C.T
            w_next = jnp.maximum(0.0, s_next + v_k / rho)
            v_next = v_k + rho * (s_next - w_next)
            return w_next, v_next, x_next

        _, _, x_final = jax.lax.fori_loop(0, n_admm, body_fn, (w_init, v_init, x_init), unroll=True)
        return x_final

    return solve


def baseline_solver(
    b_samples: np.ndarray,
    d_samples: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    n_var: int,
    solver_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_batch, solve_times = [], []
    for b_sample, d_sample in zip(b_samples, d_samples):
        x = cp.Variable(n_var)
        problem = cp.Problem(
            cp.Minimize(-cp.sum(cp.entr(x))),
            [A @ x == b_sample, C @ x <= d_sample],
        )
        start = time.perf_counter()
        problem.solve(solver=solver_name, warm_start=True, verbose=False)
        elapsed = time.perf_counter() - start
        x_batch.append(x.value if x.value is not None else np.full(n_var, np.nan))
        solve_times.append(elapsed)
    return np.array(x_batch), np.array(solve_times, dtype=float)


def primal_solver(q, A, C, b, d, rho):
    x = cp.Variable(A.shape[1])
    s = cp.Variable(C.shape[0])
    problem = cp.Problem(
        cp.Minimize(-cp.sum(cp.entr(x)) + 0.5 * rho * cp.sum_squares(s - q)),
        [A @ x == b, C @ x + s == d, x >= 0],
    )
    start = time.perf_counter()
    problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
    elapsed = time.perf_counter() - start
    return x.value, s.value, elapsed


if __name__ == "__main__":
    with Path(__file__).with_name("cfg.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    n_var = int(cfg["problem"]["n_var"])
    n_eq = int(cfg["problem"]["n_eq"])
    n_ineq = int(cfg["problem"]["n_ineq"])
    n_admm = int(cfg["training"]["n_admm"])
    max_iter = int(cfg["admm"]["max_iter"])
    tol = float(cfg["admm"]["tolerance"])
    rho = float(cfg["admm"]["rho"])

    ctx = {**cfg["problem"], **cfg["data"]}
    scenario_root = Path(__file__).resolve().parents[2] / cfg["data"]["root_subdir"] / f"n_var_{n_var}"

    dataset_file  = "datasets_{n_samples}.npz".format(**ctx)
    model_file    = "huanet_params_{n_samples}_n{n_var}_eq{n_eq}_ineq{n_ineq}.npz".format(**ctx)

    dataset_path = scenario_root / "datasets" / dataset_file
    model_path = scenario_root / "model_params" / model_file

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}\nRun generate.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}\nRun train.py first.")

    # -- Load problem matrices --
    data = np.load(dataset_path)

    A = jnp.array(data["A"])
    C = jnp.array(data["C"])

    # -- Load trained model --
    saved = np.load(model_path, allow_pickle=True)
    nn_params = saved["params"].item()["nn"]
    n_train = int(saved["n_train"])
    n_val = int(saved["n_val"])

    # -- Load test split from dataset --
    test_start = n_train + n_val
    lam_test_np = data["lam"][test_start:]
    b_test_np = data["b"][test_start:]
    d_test_np = data["d"][test_start:]
    lam = jnp.array(lam_test_np)
    d = jnp.array(d_test_np)
    print(f"Test samples: {b_test_np.shape[0]}")

    # -- Build model --
    model_cfg = freeze({
        "problem": cfg["problem"],
        "neural_net": {
            **cfg["neural_net"],
            "hidden_layers": tuple(cfg["neural_net"]["hidden_layers"]),
        },
    })
    model = HUANet(model_cfg)

    # ---- HUANet --
    l2o_solve = make_l2o_solver(
        nn_params=nn_params,
        model=model,
        C=C,
        n_var=n_var,
        n_ineq=n_ineq,
        n_admm=n_admm,
        rho=rho,
    )
    n_devices = jax.local_device_count()
    print(f"Using {n_devices} CPU devices via pmap")
    n_test = len(lam)
    pad = (n_devices - n_test % n_devices) % n_devices
    if pad:
        lam = jnp.concatenate([lam, jnp.zeros((pad, n_ineq), dtype=lam.dtype)], axis=0)
        d = jnp.concatenate([d, jnp.zeros((pad, n_ineq), dtype=d.dtype)], axis=0)
    # Warm up
    print("Running L2O...")
    lam0 = jnp.zeros((n_devices, 1, n_ineq), dtype=lam.dtype)
    d0 = jnp.zeros((n_devices, 1, n_ineq), dtype=d.dtype)
    _ = l2o_solve(lam0, d0).block_until_ready()

    # Per-batch timing
    l2o_times = []
    x_parts = []
    for i in range(0, len(lam), n_devices):
        end = i + n_devices
        lam_i = lam[i:end, None, :]
        d_i = d[i:end, None, :]
        start = time.perf_counter()
        x_i = l2o_solve(lam_i, d_i)
        x_i.block_until_ready()
        elapsed = time.perf_counter() - start
        x_parts.append(np.array(x_i[:, 0, :]))
        l2o_times.extend([elapsed / n_devices] * n_devices)

    x_pred = np.vstack(x_parts)[:n_test]
    l2o_times = np.array(l2o_times[:n_test], dtype=float)
    sequential_time = float(np.mean(l2o_times))

    A_np = np.array(A)
    C_np = np.array(C)

    # ---- Clarabel ----
    print("Running Clarabel...")
    x_clarabel, solve_times_clarabel = baseline_solver(b_test_np, d_test_np, A_np, C_np, n_var, cp.CLARABEL)
    # ---- SCS ----
    print("Running SCS...")
    x_scs, solve_times_scs = baseline_solver(b_test_np, d_test_np, A_np, C_np, n_var, cp.SCS)

    # ---- ADMM ----
    x_admm = np.zeros((n_test, n_var))
    times_admm = np.zeros(n_test)
    print(f"Running ADMM on {n_test} samples...")
    for i, (b_i, d_i) in enumerate(zip(b_test_np, d_test_np)):
        x_admm[i], times_admm[i] = admm(
            lambda q: primal_solver(q, A_np, C_np, b_i, d_i, rho),
            n_ineq=n_ineq,
            max_iter=max_iter,
            tol=tol,
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
        samples={"lam": lam_test_np, "b": b_test_np, "d": d_test_np},
        matrices={"A": A_np, "C": C_np},
        solvers={
            "clarabel": (x_clarabel, solve_times_clarabel),
            "our_method": (x_pred, l2o_times),
            "scs": (x_scs, solve_times_scs),
        },
        sequential_time=sequential_time,
    )
