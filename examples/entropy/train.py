import os
import sys
from functools import partial

import jax
import jax.numpy as jnp

import numpy as np
import optax
import yaml
from flax.core import freeze
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from huanet.neural_layer import HUANet
from huanet.huanet import Train, Unroll

jax.config.update("jax_enable_x64", True)
print(jax.devices())


def feasibility_layer(nn_params, q, batch, *, model, C):
    x, z = model.apply({"params": nn_params}, q, batch["lam"])
    s = batch["d"] - x @ C.T
    return x, s, z


def f_and_grad(x, batch):
    log_x = jnp.log(jnp.clip(x, 1e-15))
    f = jnp.sum(x * log_x, axis=-1)
    return f, log_x + 1.0


def kkt_residual(grad_f, z, s, q, batch, rho, *, A, C):
    beta = -rho * (s - q)
    return grad_f + z @ A + beta @ C


def metrics(x, data, *, A, C):
    f, _ = f_and_grad(jnp.asarray(x), data)
    A = np.asarray(A)
    C = np.asarray(C)
    eq_residual = x @ A.T - data["b"]
    ineq_residual = np.maximum(0.0, x @ C.T - data["d"])
    return {
        "f": float(jnp.mean(f)),
        "eq_viol_max": float(np.max(np.abs(eq_residual))),
        "ineq_viol_max": float(np.max(ineq_residual)),
    }


# ============================================================================
# Main Execution
# ============================================================================
def main() -> None:
    with Path(__file__).with_name("cfg.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    training_cfg   = cfg["training"]
    scenario_n_var = int(cfg["problem"]["n_var"])
    scenario_root = Path(__file__).resolve().parents[2] / cfg["data"]["root_subdir"] / f"n_var_{scenario_n_var}"
    ctx = {**cfg["problem"], **cfg["data"]}

    dataset_file   = "datasets_{n_samples}.npz".format(**ctx)
    model_filename = "huanet_params_{n_samples}_n{n_var}_eq{n_eq}_ineq{n_ineq}.npz".format(**ctx)

    dataset_path = scenario_root / "datasets" / dataset_file
    model_path = scenario_root / "model_params" / model_filename
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run generate.py first.")

    data    = np.load(dataset_path)

    lam_all = data["lam"]
    b_all   = data["b"]   # (n_total, n_eq)
    d_all   = data["d"]   # (n_total, n_ineq)
    n_total = int(b_all.shape[0])

    train_percent = float(cfg["data"]["train_percent"])
    val_percent   = float(cfg["data"]["val_percent"])
    n_train = max(1, int(n_total * train_percent))
    n_val   = max(1, int(n_total * val_percent))
    n_train = min(n_train, n_total - n_val - 1)

    train_data = {
        "lam": lam_all[:n_train],
        "b": b_all[:n_train],
        "d": d_all[:n_train],
    }
    val_data = {
        "lam": lam_all[n_train : n_train + n_val],
        "b": b_all[n_train : n_train + n_val],
        "d": d_all[n_train : n_train + n_val],
    }

    A     = jnp.array(data["A"])    # (n_eq, n_var)
    C     = jnp.array(data["C"])    # (n_ineq, n_var)
    n_ineq        = C.shape[0]

    batch_size    = int(training_cfg["batch_size"])
    n_epochs      = int(training_cfg["n_epochs"])
    n_admm        = int(training_cfg["n_admm"])
    hidden_layers = tuple(cfg["neural_net"]["hidden_layers"])
    eval_step     = int(training_cfg.get("eval_step", 5000))

    model_cfg = freeze({
        "problem": cfg["problem"],
        "neural_net": {**cfg["neural_net"], "hidden_layers": hidden_layers},
    })
    model = HUANet(model_cfg)
    key   = jax.random.PRNGKey(int(cfg["problem"]["seed"]))

    nn_params = model.init(
        key,
        jnp.zeros((1, n_ineq)),
        jnp.zeros((1, n_ineq)),
    )["params"]

    params = {"nn": nn_params}

    lr_sched = optax.exponential_decay(
        init_value=float(cfg["optimizer"]["learning_rate"]),
        decay_rate=float(cfg["optimizer"]["decay_rate"]),
        transition_steps=int(cfg["optimizer"]["transition_steps"]),
    )
    optimizer = optax.adamw(learning_rate=lr_sched, weight_decay=float(cfg["optimizer"]["weight_decay"]))
    opt_state = optimizer.init(params)

    unroll = Unroll(
        cfg,
        model_step=partial(feasibility_layer, model=model, C=C),
        f_and_grad=f_and_grad,
        kkt_residual=partial(kkt_residual, A=A, C=C),
        metrics=partial(metrics, A=A, C=C),
    )
    trainer = Train(unroll, optimizer)

    n_test = n_total - n_train - n_val
    print("=" * 80)
    print("Training Entropy — HUANet (feasibility mode)")
    print("=" * 80)
    print(f"n_var={scenario_n_var}")
    print(
        f"train={n_train} ({100.0 * n_train / n_total:.1f}%)  "
        f"val={n_val} ({100.0 * val_percent:.1f}%)  "
        f"test={n_test} ({100.0 * n_test / n_total:.1f}%)"
    )
    print(f"epochs={n_epochs}  batch={batch_size}  n_admm={n_admm}")

    rng        = np.random.default_rng(int(cfg["problem"]["seed"]))
    best_loss  = float("inf")
    best_params = params

    for epoch in range(n_epochs):
        batch_idx = rng.integers(0, n_train, size=batch_size)
        batch = {name: jnp.asarray(values[batch_idx]) for name, values in train_data.items()}

        params, opt_state, loss, f_terminal, kkt_last, viol = trainer.train_step(
            params, opt_state, batch,
        )

        if float(loss) < best_loss:
            best_loss   = float(loss)
            best_params = params

        if epoch % 1000 == 0:
            current_lr = float(lr_sched(epoch))
            print(
                f"Epoch {epoch:5d} | loss: {float(loss):.6f} | "
                f"f: {float(f_terminal):.6f} | "
                f"kkt: {float(kkt_last):.6f} | "
                f"viol: {float(viol):.6f} | "
                f"lr: {current_lr:.2e}"
            )

        if eval_step > 0 and epoch > 0 and epoch % eval_step == 0:
            val_metrics = trainer.validate(best_params, val_data)
            trainer.print_summary(epoch, val_metrics)

    val_metrics = trainer.validate(best_params, val_data)
    trainer.print_summary(n_epochs, val_metrics)

    np.savez_compressed(
        model_path,
        params=jax.tree_util.tree_map(np.array, best_params),
        best_loss=float(best_loss),
        n_epochs=n_epochs,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        scenario_n_var=scenario_n_var,
        hidden_layers=np.array(hidden_layers),
    )
    print(f"Saved model → {model_path}")


if __name__ == "__main__":
    main()
