from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax


class Unroll:
    def __init__(
        self,
        cfg: dict,
        model_step: Callable,
        f_and_grad: Callable,
        kkt_residual: Callable,
        metrics: Callable,
    ) -> None:
        self.model_step = model_step
        self.f_and_grad = f_and_grad
        self.kkt_residual = kkt_residual
        self.metrics = metrics
        self.n_var = int(cfg["problem"]["n_var"])
        self.n_eq = int(cfg["problem"]["n_eq"])
        self.n_ineq = int(cfg["problem"]["n_ineq"])
        self.batch_size = int(cfg["training"]["batch_size"])
        self.n_admm = int(cfg["training"]["n_admm"])
        self.rho = float(cfg["admm"]["rho"])
        self.gamma_r = float(cfg["training"]["gamma_r"])
        self.gamma_s = float(cfg["training"]["gamma_s"])

    def _step(self, nn_params, batch, w, v):
        q = w - v / self.rho
        x, slack, z = self.model_step(nn_params, q, batch)
        w_next = jnp.maximum(0.0, slack + v / self.rho)
        v_next = v + self.rho * (slack - w_next)
        return x, slack, z, q, w_next, v_next

    def unrolled_admm(self, params, batch):
        """Return the loss metrics and final primal iterate."""
        nn_params = params["nn"]
        w = jnp.zeros((self.batch_size, self.n_ineq))
        v = jnp.zeros_like(w)
        x = jnp.zeros((self.batch_size, self.n_var))
        slack = jnp.zeros_like(w)
        def single_step(carry, _):
            w_k, v_k, _, _ = carry
            x_hat, s_hat, z, q, w_next, v_next = self._step(nn_params, batch, w_k, v_k)

            f, grad_f = self.f_and_grad(x_hat, batch)
            r = self.kkt_residual(grad_f, z, s_hat, q, batch, self.rho)
            r_sq_k = jnp.sum(r ** 2, axis=-1)

            return (w_next, v_next, x_hat, s_hat), (f, r_sq_k)

        (_, _, x, slack), (f_hist, r_sq_hist) = jax.lax.scan(single_step, (w, v, x, slack), None, length=self.n_admm, unroll=True)

        f = jnp.mean(f_hist[-1])
        r_sq = jnp.mean(jnp.sum(r_sq_hist, axis=0) / self.n_var)
        r_sq_N = jnp.mean(r_sq_hist[-1]) / self.n_var
        s_viol = jnp.mean(jnp.sum(jnp.maximum(0, -slack), axis=-1) / self.n_ineq)

        loss = f + self.gamma_r * r_sq + self.gamma_s * s_viol
        return loss, (f, r_sq_N, s_viol, x)


class Train:
    def __init__(self, unroll: Unroll, optimizer: optax.GradientTransformation) -> None:
        self.unroll = unroll
        self.optimizer = optimizer
        self._compiled_step = jax.jit(self._train_step)
        self._solve = jax.jit(unroll.unrolled_admm)

    def _train_step(self, params, opt_state, batch):
        def loss_fn(current_params):
            return self.unroll.unrolled_admm(current_params, batch)

        (loss, (f, r_sq_N, s_viol, _)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, f, r_sq_N, s_viol

    def train_step(self, params, opt_state, batch):
        return self._compiled_step(params, opt_state, batch)

    def validate(self, params, data):
        batch_size = self.unroll.batch_size
        all_x = []
        n_samples = len(next(iter(data.values())))
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            pad = batch_size - (stop - start)
            batch = {name: jnp.asarray(values[start:stop])for name, values in data.items()}
            if pad:
                batch = {
                    name: jnp.concatenate([values, jnp.repeat(values[-1:], pad, axis=0)])
                    for name, values in batch.items()
                }
            _, (_, _, _, x_hat) = self._solve(params, batch)
            x_hat.block_until_ready()
            all_x.append(np.array(x_hat[:stop - start]))

        x_pred = np.concatenate(all_x, axis=0)
        return self.unroll.metrics(x_pred, data)

    @staticmethod
    def print_summary(epoch, metrics):
        w = 60
        print(f"\nVALIDATION at epoch {epoch}")
        print("=" * w)
        print(f"  f:                   {metrics['f']:.6e}")
        print(f"  Eq  violation max: {metrics['eq_viol_max']:.6e}")
        print(f"  Ineq violation max: {metrics['ineq_viol_max']:.6e}")
        print("=" * w)
