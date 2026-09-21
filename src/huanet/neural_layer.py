from typing import Any, Mapping

import jax.numpy as jnp
from flax import linen as nn
from jax import vmap

from .utils import correction


def _feedforward(x: jnp.ndarray, hidden_layers: tuple[int, ...], output_dim: int) -> jnp.ndarray:
    kernel_init = nn.initializers.xavier_uniform()
    bias_init = nn.initializers.zeros_init()
    for features in hidden_layers:
        x = nn.Dense(features, kernel_init=kernel_init, bias_init=bias_init)(x)
        x = nn.gelu(x)
    return nn.Dense(output_dim, kernel_init=kernel_init, bias_init=bias_init)(x)


class PrimeNet(nn.Module):
    cfg: Mapping[str, Any]

    @nn.compact
    def __call__(self, nn_input: jnp.ndarray, eta: jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray:
        n_var = int(self.cfg["problem"]["n_var"])
        n_ineq = int(self.cfg["problem"]["n_ineq"])
        hidden_layers = tuple(self.cfg["neural_net"]["hidden_layers"])
        out = _feedforward(nn_input, hidden_layers, n_var + n_ineq)

        if self.cfg["neural_net"]["mode"] == "feasibility":
            return out[:, :n_var]

        x_bar, s_bar = jnp.split(nn.sigmoid(out), [n_var], axis=-1)
        return x_bar, s_bar


class DualNet(nn.Module):
    cfg: Mapping[str, Any]

    @nn.compact
    def __call__(self, nn_input: jnp.ndarray) -> jnp.ndarray:
        n_eq = int(self.cfg["problem"]["n_eq"])
        hidden_layers = tuple(self.cfg["neural_net"]["hidden_layers"])

        return _feedforward(nn_input, hidden_layers, n_eq)


class HUANet(nn.Module):
    cfg: Mapping[str, Any]
    prime_net_cls: type[nn.Module] = PrimeNet

    def setup(self) -> None:
        self.mode = self.cfg["neural_net"]["mode"]
        self.prime_net = self.prime_net_cls(self.cfg)
        self.dual_net = DualNet(self.cfg)

    def __call__(
        self,
        q: jnp.ndarray,
        lam_param: jnp.ndarray,
        E: jnp.ndarray | None = None,
        EtE_inv: jnp.ndarray | None = None,
        eta: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Predict primal, slack, and dual iterates for the configured mode."""
        nn_input = jnp.concatenate([q, lam_param], axis=-1)
        nn_z = self.dual_net(nn_input)

        if self.mode == "feasibility":
            x_bar = self.prime_net(nn_input, eta)
            x_hat = nn.softmax(x_bar)
            return x_hat, nn_z

        x_bar, s_bar = self.prime_net(nn_input, eta)
        if eta.ndim == 2:
            in_axes = (0, 0, 0, None, None)
        else:
            in_axes = (0, 0, None, None, None)
        x_hat, s_hat = vmap(correction, in_axes=in_axes)(x_bar, s_bar, eta, E, EtE_inv)

        return x_hat, s_hat, nn_z
