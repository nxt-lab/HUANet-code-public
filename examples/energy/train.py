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

from utils import (
    EnergyPrimeNet,
    energy_slices,
    normalize_energy_kkt,
    normalize_energy_q,
    smooth_objective_jax,
)
from src.huanet.huanet import Train, Unroll
from src.huanet.neural_layer import HUANet
from src.huanet.utils import precompute_projection

jax.config.update("jax_enable_x64", True)


def correction_layer(nn_params, q, batch, *, model, E, E_pinv, p):
    q_network = normalize_energy_q(q, p)
    return model.apply({"params": nn_params}, q_network, batch["features"], E, E_pinv, batch["eta"])


def f_and_grad(x, batch, *, p, J_ref):
    del batch
    return jax.vmap(jax.value_and_grad(lambda y: smooth_objective_jax(y[None, :], p)[0] / J_ref))(x)


def kkt_residual(grad_f, z, s, q, batch, rho, *, A, C):
    del batch
    beta = -rho * (s - q)
    return grad_f + z @ A + beta @ C


def scaled_kkt_residual(grad_f, z, s, q, batch, rho, *, A, C, p):
    residual = kkt_residual(grad_f, z, s, q, batch, rho, A=A, C=C)
    return normalize_energy_kkt(residual, p)


def energy_metrics(x, data, *, p, J_ref, A, C, d):
    f, _ = f_and_grad(jnp.asarray(x), data, p=p, J_ref=J_ref)
    b = np.asarray(data["eta"])[:, :A.shape[0]]
    _, _, pc, _ = energy_slices(int(p["N"]))
    physical_pc = np.asarray(x)[:, pc]
    return {
        "f": float(jnp.mean(f)),
        "eq_viol_max": float(np.max(np.abs(np.asarray(x) @ np.asarray(A).T - b))),
        "ineq_viol_max": float(np.max(np.maximum(0.0, np.asarray(x) @ np.asarray(C).T - np.asarray(d)))),
        "minimum_pc": float(np.min(physical_pc)),
        "pc_below_min_count": int(np.count_nonzero(physical_pc < float(p["pc_min"]))),
    }


def domain_smoke_check(nn_params, samples, *, model, E, E_pinv, p, n_admm, rho):
    """Reject an initialization whose corrected iterates leave pc's objective domain."""
    n_ineq = int(p["n_ineq"])
    m, u, pc, _ = energy_slices(int(p["N"]))
    w = jnp.zeros((len(samples["features"]), n_ineq))
    v = jnp.zeros_like(w)
    minima, ranges = [], []
    for _ in range(n_admm):
        q = w - v / rho
        x, slack, _ = model.apply(
            {"params": nn_params}, normalize_energy_q(q, p),
            samples["features"], E, E_pinv, samples["eta"],
        )
        physical_pc = x[:, pc]
        minima.append(float(jnp.min(physical_pc)))
        ranges.append((
            float(jnp.min(x[:, u])), float(jnp.max(x[:, u])),
            float(jnp.min(x[:, m])), float(jnp.max(x[:, m])),
        ))
        w = jnp.maximum(0.0, slack + v / rho)
        v = v + rho * (slack - w)
    minimum_pc = min(minima)
    print(f"Initial corrected-domain check: minimum pc={minimum_pc:.6e}")
    print(f"Initial corrected signed-power ranges (last layer): u=[{ranges[-1][0]:.3f}, {ranges[-1][1]:.3f}], m=[{ranges[-1][2]:.3f}, {ranges[-1][3]:.3f}]")


def main() -> None:
    with Path(__file__).with_name("cfg.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    p, training = cfg["problem"], cfg["training"]
    N = int(p["N"])
    n_var, n_eq, n_ineq = int(p["n_var"]), int(p["n_eq"]), int(p["n_ineq"])
    n_samples = int(cfg["data"]["n_samples"])
    root = Path(__file__).resolve().parents[2] / cfg["data"]["root_subdir"] / f"n_var_{n_var}"
    dataset_path = root / "datasets" / f"datasets_{n_samples}.npz"
    schema_version = str(p["schema_version"])
    model_path = root / "model_params" / f"huanet_params_{schema_version}_{n_samples}_n{n_var}_eq{n_eq}_ineq{n_ineq}.npz"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(dataset_path) as dataset:
        schema_version = str(dataset["schema_version"])
        if schema_version != str(p["schema_version"]):
            raise ValueError(f"Dataset schema {schema_version!r} does not match {p['schema_version']!r}.")
        lam = np.asarray(dataset["lam"])
        features = np.asarray(dataset["lambda_features"])
        eta = np.asarray(dataset["eta"])
        net_demand = np.asarray(dataset["net_demand"])
        J_ref = float(dataset["J_ref"])
        A, C, d = (jnp.asarray(dataset[name]) for name in ("A", "C", "d"))
    if lam.shape != (n_samples, 1) or features.shape != (n_samples, 1):
        raise ValueError(f"Expected scalar-SOC arrays {(n_samples, 1)}, got {lam.shape} and {features.shape}.")

    n_total = len(lam)
    n_train = max(1, int(n_total * float(cfg["data"]["train_percent"])))
    n_val = max(1, int(n_total * float(cfg["data"]["val_percent"])))
    n_train = min(n_train, n_total - n_val - 1)
    data = {"features": features, "eta": eta}
    train_data = {name: values[:n_train] for name, values in data.items()}
    val_data = {name: values[n_train:n_train + n_val] for name, values in data.items()}

    hidden_layers = tuple(cfg["neural_net"]["hidden_layers"])
    model_cfg = freeze({
        "problem": p,
        "neural_net": {**cfg["neural_net"], "hidden_layers": hidden_layers},
    })
    model = HUANet(model_cfg, prime_net_cls=EnergyPrimeNet)
    E, E_pinv = precompute_projection(A, C)
    dummy_lam = jnp.array([[0.65]])
    dummy_eta = jnp.asarray(eta[:1])
    nn_params = model.init(
        jax.random.PRNGKey(int(p["seed"])),
        jnp.zeros((1, n_ineq)), dummy_lam, E, E_pinv, dummy_eta,
    )["params"]
    params = {"nn": nn_params}

    smoke_indices = np.array([0, n_train, n_train + n_val])
    domain_smoke_check(
        nn_params,
        {"features": jnp.asarray(features[smoke_indices]), "eta": jnp.asarray(eta[smoke_indices])},
        model=model, E=E, E_pinv=E_pinv, p=p,
        n_admm=int(training["n_admm"]), rho=float(cfg["admm"]["rho"]),
    )

    opt_cfg = cfg["optimizer"]
    lr_sched = optax.exponential_decay(
        float(opt_cfg["learning_rate"]), int(opt_cfg["transition_steps"]), float(opt_cfg["decay_rate"])
    )
    optimizer = optax.adamw(lr_sched, weight_decay=float(opt_cfg["weight_decay"]))
    unroll_cfg = {
        **cfg,
        "training": {**training, "gamma_r": float(training["gamma_r"]) / int(training["n_admm"])},
    }
    unroll = Unroll(
        unroll_cfg,
        model_step=partial(correction_layer, model=model, E=E, E_pinv=E_pinv, p=p),
        f_and_grad=partial(f_and_grad, p=p, J_ref=J_ref),
        kkt_residual=partial(scaled_kkt_residual, A=A, C=C, p=p),
        metrics=partial(energy_metrics, p=p, J_ref=J_ref, A=A, C=C, d=d),
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
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}; check the pc objective domain.")
        if float(loss) < best_loss:
            best_loss, best_params = float(loss), params
        if epoch % 1000 == 0:
            print(
                f"Epoch {epoch:5d} | loss: {float(loss):.6f} | f: {float(f):.6f} | "
                f"kkt: {float(r_sq):.6f} | viol: {float(s_viol):.6f} | lr: {float(lr_sched(epoch)):.2e}"
            )
        if eval_step > 0 and epoch > 0 and epoch % eval_step == 0:
            summary = trainer.validate(best_params, val_data)
            trainer.print_summary(epoch, summary)
            print(f"  Minimum physical pc: {summary['minimum_pc']:.6e}")
            if summary["pc_below_min_count"]:
                print(f"  Entries below pc_min: {summary['pc_below_min_count']}")

    summary = trainer.validate(best_params, val_data)
    trainer.print_summary(n_epochs, summary)
    np.savez_compressed(
        model_path,
        params=jax.tree_util.tree_map(np.asarray, best_params),
        best_loss=best_loss,
        J_ref=np.array(J_ref),
        A=np.asarray(A), C=np.asarray(C), d=np.asarray(d), net_demand=net_demand,
        schema_version=np.array(str(p["schema_version"])),
        forecast_day=np.array(int(p["forecast_day"])),
        pc_min=np.array(float(p["pc_min"])), pc_max=np.array(float(p["pc_max"])),
        m_min=np.array(float(p["m_min"])), m_max=np.array(float(p["m_max"])),
        delta=np.array(float(p["delta"])), kappa_m=np.array(float(p["kappa_m"])), kappa_c=np.array(float(p["kappa_c"])),
    )
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
