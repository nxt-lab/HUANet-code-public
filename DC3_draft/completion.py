# Copyright 2024-2026 the authors of this repository.
#
# Portions of this file are adapted from the official DC3 implementation
#   https://github.com/locuslab/DC3  (utils.py, class SimpleProblem)
# which is licensed under the Apache License, Version 2.0.  See
# `DC3/common/LICENSE-Apache-2.0-DC3` and `DC3/common/NOTICE.md`.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Differentiable *equality completion* for affine equality constraints.

Both applications in this repository have equality constraints that are affine
in the decision variable ``y`` with an instance-independent matrix::

    A_eq y = b_eq(param),        A_eq in R^{n_eq x n_y}

DC3 splits ``y`` into a *partial* (independent) block ``y_P`` predicted by the
network and a *dependent* block ``y_D`` obtained by solving the equalities::

    y_D = A_D^{-1} ( b_eq - A_P y_P ).

For affine ``h`` this closed form is exactly what DC3's Newton-based completion
converges to in a single step, so no inner Newton loop is needed (documented as
a deviation in the READMEs).  The map is affine in ``y_P`` and therefore
trivially differentiable; its Jacobian is the constant matrix
``dy_D/dy_P = -A_D^{-1} A_P``.

The construction requires ``A_D`` to be square and invertible.  That assumption
is checked explicitly (``check()``) and the condition number is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch


@dataclass
class PartitionInfo:
    """Diagnostics for the variable partition used by the completion."""

    strategy: str
    n_y: int
    n_eq: int
    n_partial: int
    rank_A_eq: int
    cond_A_other: float
    logabsdet_A_other: float
    completion_gain: float
    partial_vars: list
    other_vars: list

    def summary(self) -> str:
        return (
            f"partition[{self.strategy}]: n_y={self.n_y}, n_eq={self.n_eq}, "
            f"n_partial={self.n_partial}, rank(A_eq)={self.rank_A_eq}, "
            f"cond(A_D)={self.cond_A_other:.4e}, log|det(A_D)|={self.logabsdet_A_other:.4e}, "
            f"gain=||A_D^-1 A_P||_2={self.completion_gain:.4e}"
        )


def _as_numpy(A: torch.Tensor) -> np.ndarray:
    return A.detach().cpu().double().numpy()


def choose_partition(
    A_eq: torch.Tensor,
    strategy: str = "auto_qr",
    other_vars: Optional[Sequence[int]] = None,
    max_tries: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the dependent variable set ``D`` so that ``A_eq[:, D]`` is invertible.

    ``auto_qr``   column-pivoted QR of ``A_eq``; the first ``n_eq`` pivot columns
                  are the best-conditioned square block QR can find.  Deterministic.
    ``random``    DC3's original heuristic: draw random subsets until
                  ``|det(A_eq[:, D])| > 1e-4`` (see ``SimpleProblem.__init__``).
    ``explicit``  use the caller-supplied ``other_vars`` (the structured choice
                  documented in each application's README).
    """
    A = _as_numpy(A_eq)
    n_eq, n_y = A.shape
    if n_eq > n_y:
        raise ValueError(f"over-determined equalities: n_eq={n_eq} > n_y={n_y}")

    if strategy == "explicit":
        if other_vars is None:
            raise ValueError("strategy='explicit' requires other_vars")
        other = np.asarray(sorted(int(i) for i in other_vars), dtype=np.int64)
        if other.size != n_eq:
            raise ValueError(f"explicit other_vars must have {n_eq} entries, got {other.size}")
    elif strategy == "auto_qr":
        from scipy.linalg import qr

        _, _, piv = qr(A, pivoting=True, mode="economic")
        other = np.asarray(sorted(piv[:n_eq]), dtype=np.int64)
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        other = None
        # DC3 uses |det| > 1e-4; that test underflows for large n_eq, so the
        # equivalent log|det| > log(1e-4) test is used instead.
        for _ in range(max_tries):
            partial = rng.choice(n_y, n_y - n_eq, replace=False)
            cand = np.setdiff1d(np.arange(n_y), partial)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sign, logabsdet = np.linalg.slogdet(A[:, cand])
            if sign != 0 and logabsdet > np.log(1e-4):
                other = cand.astype(np.int64)
                break
        if other is None:
            raise RuntimeError("random partition search failed to find an invertible block")
    else:
        raise ValueError(f"unknown partition strategy {strategy!r}")

    partial = np.setdiff1d(np.arange(n_y), other).astype(np.int64)
    return partial, other


class LinearCompletion:
    """``y_D = A_D^{-1} (b_eq - A_P y_P)`` with cached factorisation."""

    def __init__(
        self,
        A_eq: torch.Tensor,
        strategy: str = "auto_qr",
        other_vars: Optional[Sequence[int]] = None,
        seed: int = 0,
        cond_warn: float = 1e10,
    ):
        if A_eq.dim() != 2:
            raise ValueError("A_eq must be 2-D (n_eq, n_y); instance-dependent A_eq is not supported")
        self.A_eq = A_eq
        self.n_eq, self.n_y = A_eq.shape
        partial, other = choose_partition(A_eq, strategy, other_vars, seed=seed)
        self.partial_vars = torch.as_tensor(partial, dtype=torch.long, device=A_eq.device)
        self.other_vars = torch.as_tensor(other, dtype=torch.long, device=A_eq.device)
        self.n_partial = int(partial.size)

        A_np = _as_numpy(A_eq)
        A_other = A_np[:, other]
        # numpy raises spurious FP flags from the vectorised slogdet kernel even
        # when the returned value is finite; the value itself is checked below.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            _sign, _logabsdet = np.linalg.slogdet(A_other)
            _cond = float(np.linalg.cond(A_other))
            _rank = int(np.linalg.matrix_rank(A_np))
        self.info = PartitionInfo(
            strategy=strategy,
            n_y=self.n_y,
            n_eq=self.n_eq,
            n_partial=self.n_partial,
            rank_A_eq=_rank,
            cond_A_other=_cond,
            logabsdet_A_other=float(_logabsdet) if _sign != 0 else float("-inf"),
            completion_gain=float("nan"),   # filled in below (needs A_other_inv)
            partial_vars=partial.tolist(),
            other_vars=other.tolist(),
        )
        self.cond_warn = cond_warn

        # Cache A_D^{-1} and A_D^{-1} A_P.  Both are tiny relative to the nets
        # (n_eq = 1 for the cone program, 2N+1 = 193 for the eco-MPC).
        A_p = A_eq[:, self.partial_vars]
        A_o = A_eq[:, self.other_vars]
        eye = torch.eye(self.n_eq, dtype=A_eq.dtype, device=A_eq.device)
        self.A_other_inv = torch.linalg.solve(A_o, eye)          # (n_eq, n_eq)
        self.A_other_inv_A_partial = self.A_other_inv @ A_p      # (n_eq, n_partial)
        self.A_partial = A_p
        self.A_other = A_o
        # How strongly an error in the predicted block is amplified into the
        # completed block: ||dy_D/dy_P||_2 = ||A_D^{-1} A_P||_2.  For a simplex
        # equality (1^T y = 1) this is sqrt(n-1) for *every* choice of dependent
        # variable within this coordinate-elimination parametrization. This does
        # not imply the same conditioning in other coordinates or failure of DC3.
        with np.errstate(all="ignore"):
            self.info.completion_gain = float(
                np.linalg.norm(_as_numpy(self.A_other_inv_A_partial), 2))

    # -- assumption checks -------------------------------------------------
    def check(self, strict: bool = True) -> PartitionInfo:
        info = self.info
        if info.rank_A_eq != self.n_eq:
            msg = f"A_eq is rank deficient: rank={info.rank_A_eq} < n_eq={self.n_eq}"
            if strict:
                raise ValueError(msg)
            print("WARNING:", msg)
        if not np.isfinite(info.cond_A_other) or info.cond_A_other > self.cond_warn:
            msg = (
                f"dependent block A_D is ill-conditioned: cond={info.cond_A_other:.3e} "
                f"(> {self.cond_warn:.1e}); completion may be numerically unreliable"
            )
            if strict:
                raise ValueError(msg)
            print("WARNING:", msg)
        return info

    def to(self, device, dtype) -> "LinearCompletion":
        self.A_eq = self.A_eq.to(device=device, dtype=dtype)
        self.A_partial = self.A_partial.to(device=device, dtype=dtype)
        self.A_other = self.A_other.to(device=device, dtype=dtype)
        self.A_other_inv = self.A_other_inv.to(device=device, dtype=dtype)
        self.A_other_inv_A_partial = self.A_other_inv_A_partial.to(device=device, dtype=dtype)
        self.partial_vars = self.partial_vars.to(device=device)
        self.other_vars = self.other_vars.to(device=device)
        return self

    # -- the completion itself --------------------------------------------
    def complete(self, Z: torch.Tensor, b_eq: torch.Tensor) -> torch.Tensor:
        """(B, n_partial) + (B, n_eq) -> (B, n_y).  Differentiable in ``Z`` and ``b_eq``."""
        if Z.shape[-1] != self.n_partial:
            raise ValueError(f"expected {self.n_partial} partial vars, got {Z.shape[-1]}")
        Y_other = b_eq @ self.A_other_inv.T - Z @ self.A_other_inv_A_partial.T
        Y = torch.zeros(Z.shape[0], self.n_y, dtype=Z.dtype, device=Z.device)
        Y = torch.index_copy(Y, 1, self.partial_vars, Z)
        Y = torch.index_copy(Y, 1, self.other_vars, Y_other)
        return Y

    def partial_of(self, Y: torch.Tensor) -> torch.Tensor:
        return Y[:, self.partial_vars]

    def scatter_step(self, dZ: torch.Tensor) -> torch.Tensor:
        """Map a partial-space step to full space exactly as DC3 does.

        DC3 (``SimpleProblem.ineq_partial_grad``) writes the step into the full
        vector with ``step_D = -A_D^{-1} A_P step_P``.  Because ``complete`` is
        affine, ``complete(Z - s) == complete(Z) - scatter_step(s)``; this method
        exists so that equivalence can be asserted in the validation tests.
        """
        dY = torch.zeros(dZ.shape[0], self.n_y, dtype=dZ.dtype, device=dZ.device)
        dY = torch.index_copy(dY, 1, self.partial_vars, dZ)
        dY = torch.index_copy(dY, 1, self.other_vars, -dZ @ self.A_other_inv_A_partial.T)
        return dY

    def eq_resid(self, Y: torch.Tensor, b_eq: torch.Tensor) -> torch.Tensor:
        return Y @ self.A_eq.T - b_eq
