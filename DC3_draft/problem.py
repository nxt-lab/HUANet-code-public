"""Abstract parametric-optimisation interface consumed by the DC3 solver.

A *problem instance* is a batch of parameters ``P`` (an application specific
dataclass of tensors).  The decision variable is ``y in R^{n_y}`` and the
problem solved for every instance is

    min_y   f(P, y)
    s.t.    A_eq y = b_eq(P)              (affine, instance-independent A_eq)
            g(P, y) <= 0
            y in dom f                    (see below)

**Domain handling.**  Both applications have objectives that are only defined on
an open set (``w > 0`` for ``sum w log w``; ``p > 0`` for ``a/p - 1``).  DC3 has
no notion of an open domain.  We therefore

* expose the *closed* relaxation used by the Julia models (``w >= w_floor``,
  ``p >= 0``) as ordinary inequality constraints, so that feasibility is
  measured against exactly the constraints the reference solvers see;
* additionally let DC3's own loss/correction work with a slightly *tightened*
  set (``margin=True``).  Tightening can only make DC3 more conservative - a
  point feasible for the tightened set is feasible for the original one - so
  reported feasibility rates stay valid;
* use clamped objectives (``safe=True``) only as training surrogates;
* report exact objectives (``safe=False``) only for domain-valid points.
  Domain validity is independent of constraint tolerance; invalid gaps are NaN.
  Entropy uses its continuous extension at zero.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch


class ParametricProblem(ABC):
    n_y: int
    n_eq: int
    n_ineq: int
    x_dim: int
    A_eq: torch.Tensor           # (n_eq, n_y), instance independent
    name: str = "problem"

    # ---- instance handling ----------------------------------------------
    @abstractmethod
    def features(self, params: Any) -> torch.Tensor:
        """Network input ``X`` of shape (B, x_dim) for a batch of instances."""

    @abstractmethod
    def eq_rhs(self, params: Any) -> torch.Tensor:
        """``b_eq`` of shape (B, n_eq)."""

    # ---- objective and constraints --------------------------------------
    @abstractmethod
    def obj_fn(self, params: Any, Y: torch.Tensor, safe: bool = True) -> torch.Tensor:
        """Objective, shape (B,), in the same convention as the Julia code.

        ``safe=True`` clamps the arguments into the objective's domain so the
        value stays finite; ``safe=False`` returns the mathematically exact
        value (possibly ``inf``/``nan`` outside the domain).
        """

    @abstractmethod
    def ineq_resid(self, params: Any, Y: torch.Tensor, margin: bool = False) -> torch.Tensor:
        """``g(P, y)``, shape (B, n_ineq); feasible iff <= 0.

        ``margin=False`` reproduces the constraints of the Julia model exactly
        and is what all reported metrics use.  ``margin=True`` adds the
        configured safety margins and is what DC3's loss and correction use.
        """

    def ineq_dist(self, params: Any, Y: torch.Tensor, margin: bool = False) -> torch.Tensor:
        return torch.clamp(self.ineq_resid(params, Y, margin=margin), min=0)

    def eq_resid(self, params: Any, Y: torch.Tensor) -> torch.Tensor:
        return Y @ self.A_eq.T - self.eq_rhs(params)

    def domain_resid(self, params: Any, Y: torch.Tensor) -> torch.Tensor:
        """Violation of the *open* domain of the objective, shape (B, k).

        Positive entries mean the objective was evaluated outside its domain.
        """
        return torch.zeros(Y.shape[0], 0, dtype=Y.dtype, device=Y.device)

    def domain_valid(self, params: Any, Y: torch.Tensor) -> torch.Tensor:
        """Exact objective-domain membership, independent of feasibility tolerance."""
        return torch.isfinite(Y).all(dim=1) & (self.domain_resid(params, Y) <= 0).all(dim=1)

    # ---- optional fast paths --------------------------------------------
    def ineq_partial_grad(self, params: Any, Z: torch.Tensor, completion,
                          row_scale: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Closed form of ``d/dZ || relu(s * g(P, complete(Z, b_eq))) ||^2``.

        Returning ``None`` makes the solver fall back to autograd.
        """
        return None

    def ineq_row_scale(self, completion):
        """Per-row weights for the *internal* (``margin=True``) residuals, or ``None``.

        DC3 corrects with a plain gradient step on ``||relu(g)||^2``.  When the
        rows of the reduced constraint Jacobian ``G_eff = dg/dz`` have wildly
        different norms, no single ``corr_lr`` can make progress on all of them
        (see the eco-MPC README: state-of-charge rows have gradient ``|B| = 5e-4``
        while power rows have gradient 1, a factor 2000).  Returning
        ``1 / ||G_eff_i||`` makes the reduced problem isotropic.  Reported
        metrics always use the *unscaled* residuals.
        """
        return None

    def partial_bounds(self, completion):
        """``(lo, hi)`` for the partial variables, or ``None`` for no transform.

        Entries may be ``+-inf``.  Used by the optional DC3-ACOPF style bounded
        output parametrisation (`common.nets.PartialTransform`).
        """
        return None

    def partial_init_target(self, completion):
        """Value the *initial* prediction should take, or ``None``."""
        return None

    def unpack(self, Y: torch.Tensor) -> dict:
        """Human readable decomposition of a solution (for plots / reports)."""
        return {"y": Y}

    def to(self, device, dtype):
        return self
