from functools import partial
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.core import freeze

from utils import EnergyPrimeNet, energy_slices, equality_rhs, imported_slice
from src.huanet.huanet import Train, Unroll
from src.huanet.neural_layer import HUANet
from src.huanet.utils import metrics, precompute_projection

jax.config.update("jax_enable_x64", True)


def correction_layer(nn_params, q, batch, *, model, E, E_pinv):
    return model.apply({"params": nn_params}, q, batch["features"], E, E_pinv, batch["eta"])


def f_and_grad(x, batch, *, p, J_ref):
    N = int(p["N"])
    _, pb, r = energy_slices(N)
    m = imported_slice(N)
    S_B, S_m = float(p["S_B"]), float(p["S_m"])
    r_ext = float(p["r_ext"])
    kappa_D, kappa_g = float(p["kappa_D"]), float(p["kappa_g"])
    delta = float(p["delta"])
    cycling = (1.0 - float(p["mu"])) / (2.0 * np.sqrt(float(p["mu"])))

    def sample_cost(y):
        P_B = S_B * y[pb]
        imported = S_m * y[m]
        dr = y[r] - r_ext
        extended = 1.0 / r_ext - 1.0 - dr / r_ext**2 + dr**2 / r_ext**3
        safe_r = jnp.where(y[r] >= r_ext, jnp.maximum(y[r], r_ext), 1.0)
        ratio = jnp.where(y[r] >= r_ext, 1.0 / safe_r - 1.0, extended)
        f = (
            float(p["p_k"]) * float(p["Delta_t"]) * jnp.sum(imported + cycling * (jnp.sqrt(P_B**2 + delta**2) - delta))
            + float(p["p_p"]) * jnp.sum(jax.nn.softplus(kappa_D * imported) - jnp.log(2.0)) / kappa_D
            + float(p["eta"]) * jnp.sum(jax.nn.softplus(kappa_g * ratio)) / kappa_g
        )
        return f / J_ref

    return jax.vmap(jax.value_and_grad(sample_cost))(x)


def kkt_residual(grad_f, z, s, q, batch, rho, *, A, C):
    beta = -rho * (s - q)
    return grad_f + z @ A + beta @ C


def main() -> None:
    with Path(__file__).with_name("cfg.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    p, training = cfg["problem"], cfg["training"]
    N = int(p["N"])
    n_var, n_eq, n_ineq = 4 * N + 1, 2 * N + 1, 5 * N + 3
    n_samples = int(cfg["data"]["n_samples"])
    root = Path(__file__).resolve().parents[2] / cfg["data"]["root_subdir"] / f"n_var_{n_var}"
    dataset_path = root / "datasets" / f"datasets_{n_samples}.npz"
    model_path = root / "model_params" / f"huanet_params_{n_samples}_n{n_var}_eq{n_eq}_ineq{n_ineq}.npz"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(dataset_path) as dataset:
        lam = dataset["lam"]
        features = dataset["lambda_features"]
        A, C, d = (jnp.asarray(dataset[name]) for name in ("A", "C", "d"))
        J_ref = float(dataset["J_ref"])
        center, scale = dataset["parameter_center"], dataset["parameter_scale"]

    b = equality_rhs(jnp.asarray(lam), N, float(p["S_m"]))
    eta = np.concatenate([
        np.asarray(b), np.broadcast_to(np.asarray(d), (len(lam), n_ineq))
    ], axis=-1)
    n_total = len(lam)
    n_train = max(1, int(n_total * float(cfg["data"]["train_percent"])))
    n_val = max(1, int(n_total * float(cfg["data"]["val_percent"])))
    n_train = min(n_train, n_total - n_val - 1)
    data = {"features": features, "eta": eta}
    train_data = {name: values[:n_train] for name, values in data.items()}
    val_data = {name: values[n_train:n_train + n_val] for name, values in data.items()}

    hidden_layers = tuple(cfg["neural_net"]["hidden_layers"])
    problem_cfg = {**p, "n_var": n_var, "n_eq": n_eq, "n_ineq": n_ineq}
    model_cfg = freeze({
        "problem": problem_cfg,
        "neural_net": {**cfg["neural_net"], "hidden_layers": hidden_layers},
    })
    model = HUANet(model_cfg, prime_net_cls=EnergyPrimeNet)
    E, E_pinv = precompute_projection(A, C)
    nn_params = model.init(
        jax.random.PRNGKey(int(p["seed"])),
        jnp.zeros((1, n_ineq)), jnp.zeros((1, N)),
        E, E_pinv, jnp.zeros((1, n_eq + n_ineq)),
    )["params"]
    params = {"nn": nn_params}

    opt_cfg = cfg["optimizer"]
    lr_sched = optax.exponential_decay(
        float(opt_cfg["learning_rate"]), int(opt_cfg["transition_steps"]),
        float(opt_cfg["decay_rate"]),
    )
    optimizer = optax.apply_if_finite(
        optax.chain(
            optax.clip_by_global_norm(float(opt_cfg["grad_clip_norm"])),
            optax.adamw(lr_sched, weight_decay=float(opt_cfg["weight_decay"])),
        ),
        max_consecutive_errors=int(opt_cfg["max_consecutive_errors"]),
    )
    unroll_cfg = {
        **cfg, "problem": problem_cfg,
        "training": {
            **training, "gamma_r": float(training["gamma_r"]) / int(training["n_admm"]),
        },
    }
    unroll = Unroll(
        unroll_cfg,
        model_step=partial(correction_layer, model=model, E=E, E_pinv=E_pinv),
        f_and_grad=partial(f_and_grad, p=p, J_ref=J_ref),
        kkt_residual=partial(kkt_residual, A=A, C=C),
        metrics=partial(metrics, f_and_grad=partial(f_and_grad, p=p, J_ref=J_ref), A=A, C=C, d=d),
    )
    trainer = Train(unroll, optimizer)
    opt_state = optimizer.init(params)
    rng = np.random.default_rng(int(p["seed"]))
    best_loss, best_params = float("inf"), params
    batch_size = int(training["batch_size"])
    n_epochs = int(training["n_epochs"])
    eval_step = int(training["eval_step"])

    print(f"Training energy HUANet: n_var={n_var}, train={n_train}, val={n_val}")
    for epoch in range(n_epochs):
        indices = rng.integers(0, n_train, size=batch_size)
        batch = {name: jnp.asarray(values[indices]) for name, values in train_data.items()}
        params, opt_state, loss, f, r_sq, s_viol = trainer.train_step(params, opt_state, batch)
        if not np.isfinite(float(loss)):
            print(f"Stopping at epoch {epoch}: non-finite loss")
            break
        if float(loss) < best_loss:
            best_loss, best_params = float(loss), params
        if epoch % 1000 == 0:
            print(
                f"Epoch {epoch:5d} | loss: {float(loss):.6f} | f: {float(f):.6f} | "
                f"kkt: {float(r_sq):.6f} | viol: {float(s_viol):.6f} | lr: {float(lr_sched(epoch)):.2e}"
            )
        if eval_step > 0 and epoch > 0 and epoch % eval_step == 0:
            trainer.print_summary(epoch, trainer.validate(best_params, val_data))

    trainer.print_summary(n_epochs, trainer.validate(best_params, val_data))
    np.savez_compressed(
        model_path, params=jax.tree_util.tree_map(np.asarray, best_params),
        best_loss=best_loss, parameter_center=center, parameter_scale=scale,
    )
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
