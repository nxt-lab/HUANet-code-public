import jax.numpy as jnp
import numpy as np


def precompute_projection(
    A: jnp.ndarray,
    C: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build the constraint matrix and its full-row-rank pseudoinverse.

    A has shape (n_eq, n_var), C has shape (n_in, n_var), and
    the pseudoinverse has shape (n_var + n_in, n_eq + n_in).
    """
    n_eq = A.shape[0]
    n_in = C.shape[0]

    E = jnp.block([
        [A, jnp.zeros((n_eq, n_in))],
        [C, jnp.eye(n_in)],
    ])
    E_pinv = jnp.linalg.solve(E @ E.T, E).T
    return E, E_pinv


precompute = precompute_projection


def correction(
    x_bar: jnp.ndarray,
    s_bar: jnp.ndarray,
    eta: jnp.ndarray,
    E: jnp.ndarray,
    E_pinv: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project a primal and slack pair onto the affine constraints."""
    y_bar = jnp.concatenate([x_bar, s_bar], axis=0)
    y_hat = y_bar - E_pinv @ (E @ y_bar - eta)
    x_hat, s_hat = jnp.split(y_hat, [x_bar.shape[0]])
    return x_hat, s_hat


def metrics(x, data, *, f_and_grad, A, C, d):
    f, _ = f_and_grad(jnp.asarray(x), data)
    b = np.asarray(data["eta"])[:, :A.shape[0]]
    return {
        "f": float(jnp.mean(f)),
        "eq_viol_max": float(np.max(np.abs(x @ np.asarray(A).T - b))),
        "ineq_viol_max": float(np.max(np.maximum(0.0, x @ np.asarray(C).T - np.asarray(d)))),
    }
