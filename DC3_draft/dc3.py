# Copyright 2024-2026 the authors of this repository.
#
# Adapted from the official DC3 implementation https://github.com/locuslab/DC3
# (method.py: total_loss / grad_steps / grad_steps_all / NNSolver / train_net),
# licensed under the Apache License, Version 2.0.  See
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
"""DC3: deep constraint completion and correction.

Three components, matching the paper (arXiv:2104.12225) and the official code:

1. **partial-variable prediction** - a network maps the instance parameters to
   ``n_y - n_eq`` *partial* variables ``z``;
2. **equality completion** - the remaining variables are recovered by solving
   the equality constraints (``common.completion.LinearCompletion``), so the
   equality residual is zero up to round-off *by construction*;
3. **inequality correction** - gradient descent (with momentum) on
   ``|| relu(g(y(z))) ||^2`` **in the partial-variable space**, so that every
   correction step stays on the equality manifold.

Training back-propagates through both completion and the (fixed number of)
correction steps, exactly as in ``method.py::grad_steps``.

Deviations from the official code are listed in each application's README; the
ones implemented here are:

* the correction is carried in ``z``-space rather than in the full ``y`` vector.
  For affine completion the two are algebraically identical
  (``complete(z - s) == complete(z) - scatter_step(s)``, asserted in
  `validate.py`), but the ``z``-space form also works when the closed-form
  ``ineq_partial_grad`` is unavailable and autograd has to be used;
* ``soft_loss_power`` selects between the *norm* used by the official code
  (``p=1``) and the *squared* norm written in the paper (``p=2``);
* ``obj_scale`` divides the objective term of the soft loss, which matters when
  the objective and the violations live on very different scales (the eco-MPC
  objective is ~3.6e4 while violations are O(1)).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from .completion import LinearCompletion
from .nets import PartialTransform, PartialVarNet
from .problem import ParametricProblem


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class DC3Config:
    # --- network / optimisation
    hidden_size: int = 200
    n_hidden: int = 2
    dropout: float = 0.2
    batch_norm: bool = True
    input_norm: bool = True
    output_transform: str = "bounded"   # 'bounded' (DC3-ACOPF style) or 'none'
    output_init_target: bool = True
    output_weight_scale: float = 0.0  # exact target at initialization
    lr: float = 1e-3
    lr_decay: float = 1.0            # multiplicative decay per epoch (1.0 = off)
    weight_decay: float = 0.0
    epochs: int = 200
    batch_size: int = 64
    grad_clip: float = 0.0           # 0 disables

    # --- soft loss (method.py::total_loss)
    soft_weight: float = 10.0
    soft_weight_eq_frac: float = 0.5
    soft_loss_power: int = 1         # 1 = official code (norm), 2 = paper (norm^2)
    obj_scale: float = 1.0

    # --- DC3 switches (method.py / default_args.py)
    use_compl: bool = True
    use_train_corr: bool = True
    use_test_corr: bool = True
    corr_mode: str = "partial"       # 'partial' (DC3) or 'full' (naive)
    corr_train_steps: int = 10
    corr_test_max_steps: int = 10
    corr_eps: float = 1e-4
    corr_lr: float = 1e-4
    corr_momentum: float = 0.5
    corr_freeze_converged: bool = False   # False = faithful to DC3
    corr_preconditioner: str = "none"  # or completion_metric (explicit variant)
    corr_grad_mode: str = "closed_form"   # 'closed_form' | 'autograd'
    ineq_row_scale: str = "auto"          # 'auto' (problem hook) or 'none'

    # --- bookkeeping
    seed: int = 0
    device: str = "auto"
    dtype: str = "float64"
    eval_every: int = 5
    feas_tol: float = 1e-4           # feasibility threshold used in reports

    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float64": torch.float64}[self.dtype]

    def resolve_device(self) -> torch.device:
        from .io_utils import pick_device

        dev = pick_device(self.device)
        if dev.type == "mps" and self.torch_dtype() == torch.float64:
            print("NOTE: MPS has no float64 support; falling back to CPU.")
            dev = torch.device("cpu")
        return dev

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DC3Config":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
class DC3Solver(nn.Module):
    """Network + completion + correction for one `ParametricProblem`."""

    def __init__(self, problem: ParametricProblem, completion: LinearCompletion, cfg: DC3Config):
        super().__init__()
        self.problem = problem
        self.completion = completion
        self.cfg = cfg
        if cfg.corr_mode not in ("partial", "full"):
            raise ValueError("corr_mode must be partial or full")
        if cfg.corr_mode == "partial" and not cfg.use_compl:
            raise ValueError("partial correction requires completion")
        if cfg.corr_preconditioner not in ("none", "completion_metric"):
            raise ValueError("Unknown correction preconditioner")
        if cfg.corr_preconditioner != "none" and cfg.corr_mode != "partial":
            raise ValueError("completion_metric applies only to partial correction")
        metric_factor = None
        if cfg.corr_preconditioner == "completion_metric":
            M = completion.A_other_inv_A_partial
            metric_factor = torch.linalg.solve(torch.eye(M.shape[0], device=M.device, dtype=M.dtype) + M @ M.T, M)
        self.register_buffer("_metric_factor", metric_factor, persistent=False)
        out_dim = completion.n_partial if cfg.use_compl else problem.n_y
        transform = None
        if cfg.output_transform == "bounded" and cfg.use_compl:
            bounds = problem.partial_bounds(completion)
            if bounds is not None:
                transform = PartialTransform(bounds[0].clone(), bounds[1].clone())
        elif cfg.output_transform not in ("bounded", "none"):
            raise ValueError(f"unknown output_transform {cfg.output_transform!r}")
        self.net = PartialVarNet(
            in_dim=problem.x_dim,
            out_dim=out_dim,
            hidden_size=cfg.hidden_size,
            n_hidden=cfg.n_hidden,
            dropout=cfg.dropout,
            batch_norm=cfg.batch_norm,
            input_norm=cfg.input_norm,
            transform=transform,
        )
        self.net.to(device=problem.A_eq.device, dtype=problem.A_eq.dtype)
        scale = problem.ineq_row_scale(completion) if cfg.ineq_row_scale == "auto" else None
        if cfg.ineq_row_scale not in ("auto", "none"):
            raise ValueError(f"unknown ineq_row_scale {cfg.ineq_row_scale!r}")
        self.register_buffer("row_scale", None if scale is None else scale.clone())
        if cfg.output_init_target and cfg.use_compl:
            tgt = problem.partial_init_target(completion)
            if tgt is not None:
                self.net.init_output_at(tgt.clone(), cfg.output_weight_scale)

    # -- prediction --------------------------------------------------------
    def predict_partial(self, params: Any) -> torch.Tensor:
        return self.net(self.problem.features(params))

    def complete(self, params: Any, Z: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_compl:
            return Z                      # ablation: network predicts all of y
        return self.completion.complete(Z, self.problem.eq_rhs(params))

    def forward(self, params: Any) -> torch.Tensor:
        return self.complete(params, self.predict_partial(params))

    # -- correction --------------------------------------------------------
    def ineq_dist_int(self, params: Any, Y: torch.Tensor) -> torch.Tensor:
        """Internal (margin + row-scaled) inequality violation used by DC3."""
        d = self.problem.ineq_dist(params, Y, margin=True)
        return d if self.row_scale is None else d * self.row_scale

    def _viol_grad_partial(self, params: Any, Z: torch.Tensor, create_graph: bool) -> torch.Tensor:
        """d/dZ || relu(g(P, complete(Z))) ||^2 in internal (row-scaled) units."""
        if self.cfg.corr_grad_mode == "closed_form":
            g = self.problem.ineq_partial_grad(params, Z, self.completion, self.row_scale)
            if g is not None:
                return g
        Zg = Z if (create_graph and Z.requires_grad) else Z.detach().requires_grad_(True)
        Y = self.complete(params, Zg)
        viol = (self.ineq_dist_int(params, Y) ** 2).sum()
        (grad,) = torch.autograd.grad(viol, Zg, create_graph=create_graph)
        return grad

    def _viol_grad_full(self, params: Any, Y: torch.Tensor, create_graph: bool) -> torch.Tensor:
        """Naive ('full') correction: step on both inequality and equality violation."""
        Yg = Y if (create_graph and Y.requires_grad) else Y.detach().requires_grad_(True)
        ineq = (self.ineq_dist_int(params, Yg) ** 2).sum()
        eq = (self.problem.eq_resid(params, Yg) ** 2).sum()
        frac = self.cfg.soft_weight_eq_frac
        (grad,) = torch.autograd.grad((1 - frac) * ineq + frac * eq, Yg, create_graph=create_graph)
        return grad

    def correction_output(self, params: Any, state: torch.Tensor) -> torch.Tensor:
        """Convert correction state to a decision vector without re-completing full steps."""
        return self.complete(params, state) if self.cfg.corr_mode == "partial" else state

    def _precondition(self, grad):
        if self._metric_factor is None:
            return grad
        M = self.completion.A_other_inv_A_partial
        # Woodbury: (I + M' M)^-1 grad, avoiding a dense n_partial square solve.
        return grad - (grad @ M.T) @ self._metric_factor

    def correct_train(self, params: Any, Z: torch.Tensor) -> torch.Tensor:
        """Fixed number of differentiable correction steps (``grad_steps``)."""
        cfg = self.cfg
        Z = self.complete(params, Z) if cfg.corr_mode == "full" else Z
        if not cfg.use_train_corr or cfg.corr_train_steps == 0:
            return Z
        if cfg.corr_mode == "partial" and not cfg.use_compl:
            raise ValueError("partial correction requires completion")
        Z_new = Z
        old_step = torch.zeros_like(Z)
        for _ in range(cfg.corr_train_steps):
            if cfg.corr_mode == "partial":
                d = self._viol_grad_partial(params, Z_new, create_graph=True)
            else:
                d = self._viol_grad_full(params, Z_new, create_graph=True)
            new_step = cfg.corr_lr * self._precondition(d) + cfg.corr_momentum * old_step
            Z_new = Z_new - new_step
            old_step = new_step
        return Z_new

    @torch.no_grad()
    def correct_test(self, params: Any, Z: torch.Tensor) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """Correction to tolerance (``grad_steps_all``).

        Returns ``(Z, n_steps, converged_mask, first_feasible_step)``.  Note the
        DC3 loop is *batch-global*: it stops when every instance in the batch is
        within ``corr_eps`` (or ``corr_test_max_steps`` is hit), so `n_steps` is
        a batch quantity.  `first_feasible_step` records, per instance, the step
        at which it first met the tolerance (``-1`` if never).
        """
        cfg = self.cfg
        Z = self.complete(params, Z) if cfg.corr_mode == "full" else Z
        B = Z.shape[0]
        first_feasible = torch.full((B,), -1, dtype=torch.long, device=Z.device)
        if not cfg.use_test_corr or cfg.corr_test_max_steps == 0:
            conv = self._converged(params, self.correction_output(params, Z))
            first_feasible[conv] = 0
            return Z, 0, conv, first_feasible

        if cfg.corr_mode == "partial" and not cfg.use_compl:
            raise ValueError("partial correction requires completion")

        Z_new = Z
        old_step = torch.zeros_like(Z)
        i = 0
        conv = self._converged(params, self.correction_output(params, Z_new))
        first_feasible[conv & (first_feasible < 0)] = 0
        while (not bool(conv.all())) and i < cfg.corr_test_max_steps:
            with torch.enable_grad():
                if cfg.corr_mode == "partial":
                    d = self._viol_grad_partial(params, Z_new, create_graph=False)
                else:
                    d = self._viol_grad_full(params, Z_new, create_graph=False)
            new_step = cfg.corr_lr * self._precondition(d) + cfg.corr_momentum * old_step
            if cfg.corr_freeze_converged:
                new_step = new_step * (~conv).unsqueeze(1).to(new_step.dtype)
            Z_new = Z_new - new_step
            old_step = new_step
            i += 1
            conv = self._converged(params, self.correction_output(params, Z_new))
            newly = conv & (first_feasible < 0)
            first_feasible[newly] = i
        return Z_new, i, conv, first_feasible

    def _converged(self, params: Any, Y: torch.Tensor) -> torch.Tensor:
        eps = self.cfg.corr_eps
        ineq_ok = self.ineq_dist_int(params, Y).amax(dim=1) <= eps
        eq = self.problem.eq_resid(params, Y).abs()
        eq_ok = eq.amax(dim=1) <= eps if eq.shape[1] > 0 else torch.ones_like(ineq_ok)
        return ineq_ok & eq_ok

    # -- end-to-end inference ---------------------------------------------
    def solve(self, params: Any) -> dict:
        """Full inference path: predict -> complete -> correct-to-tolerance."""
        self.eval()
        with torch.no_grad():
            Z = self.predict_partial(params)
        Y_raw = self.complete(params, Z)
        Z_corr, steps, conv, first = self.correct_test(params, Z)
        Y = self.correction_output(params, Z_corr)
        return {
            "Y": Y,
            "Y_raw": Y_raw,
            "steps": steps,
            "converged": conv,
            "first_feasible_step": first,
        }

    # -- loss --------------------------------------------------------------
    def total_loss(self, params: Any, Y: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        obj = self.problem.obj_fn(params, Y, safe=True) / cfg.obj_scale
        ineq = self.ineq_dist_int(params, Y)
        eq = self.problem.eq_resid(params, Y)
        p = cfg.soft_loss_power
        ineq_cost = torch.norm(ineq, dim=1) ** p
        eq_cost = torch.norm(eq, dim=1) ** p if eq.shape[1] > 0 else torch.zeros_like(obj)
        return (
            obj
            + cfg.soft_weight * (1 - cfg.soft_weight_eq_frac) * ineq_cost
            + cfg.soft_weight * cfg.soft_weight_eq_frac * eq_cost
        )


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_dc3(
    solver: DC3Solver,
    train_params: Any,
    valid_params: Any,
    cfg: DC3Config,
    index_fn,
    n_train: int,
    log_fn=print,
    eval_fn=None,
) -> dict:
    """Train `solver`; `index_fn(params, idx)` slices a parameter batch.

    Returns a history dict.  Model selection uses the **validation split only**
    (lowest validation soft loss among epochs where the validation feasible
    rate is not worse than the best seen minus 1e-9).
    """
    device = next(solver.parameters()).device
    opt = torch.optim.Adam(solver.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg.lr_decay) if cfg.lr_decay != 1.0 else None

    history = {"epoch": [], "train_loss": [], "valid_loss": [], "valid_obj": [],
               "valid_ineq_max": [], "valid_feas_rate": [], "epoch_time": []}
    best = {"score": np.inf, "state": None, "epoch": -1}
    n_nonfinite = 0
    g = torch.Generator().manual_seed(cfg.seed)
    t_train_total = 0.0

    for ep in range(cfg.epochs):
        solver.train()
        perm = torch.randperm(n_train, generator=g).to(device)
        losses = []
        t0 = time.perf_counter()
        for s in range(0, n_train, cfg.batch_size):
            idx = perm[s : s + cfg.batch_size]
            if idx.numel() < 2 and cfg.batch_norm:
                continue                      # BatchNorm needs >1 sample
            pb = index_fn(train_params, idx)
            opt.zero_grad(set_to_none=True)
            Z = solver.predict_partial(pb)
            Z = solver.correct_train(pb, Z)
            Y = solver.correction_output(pb, Z)
            loss = solver.total_loss(pb, Y)
            lm = loss.mean()
            if not torch.isfinite(lm):
                n_nonfinite += 1
                continue
            lm.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(solver.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss.mean().item()))
        if sched is not None:
            sched.step()
        t_train_total += time.perf_counter() - t0

        if ep % cfg.eval_every == 0 or ep == cfg.epochs - 1:
            solver.eval()
            with torch.no_grad():
                pass
            out = solver.solve(valid_params)
            Yv = out["Y"]
            with torch.no_grad():
                vloss = float(solver.total_loss(valid_params, Yv).mean().item())
                vobj = float(solver.problem.obj_fn(valid_params, Yv).mean().item())
                vineq = solver.problem.ineq_dist(valid_params, Yv).amax(dim=1)
                veq = solver.problem.eq_resid(valid_params, Yv).abs().amax(dim=1)
                vdomain = solver.problem.domain_valid(valid_params, Yv)
                vfeas = float(((vineq <= cfg.feas_tol) & (veq <= cfg.feas_tol) & vdomain).double().mean().item())
                vineq_max = float(vineq.max().item())
            history["epoch"].append(ep)
            history["train_loss"].append(float(np.mean(losses)) if losses else float("nan"))
            history["valid_loss"].append(vloss)
            history["valid_obj"].append(vobj)
            history["valid_ineq_max"].append(vineq_max)
            history["valid_feas_rate"].append(vfeas)
            history["epoch_time"].append(t_train_total)
            # selection score: infeasible validation sets are penalised first.
            # A diverged epoch (NaN/inf loss) can never win.
            score = vloss + 1e3 * (1.0 - vfeas)
            if not np.isfinite(score):
                score = np.inf
            if score < best["score"]:
                best = {
                    "score": score,
                    "state": {k: v.detach().clone() for k, v in solver.state_dict().items()},
                    "epoch": ep,
                }
            log_fn(
                f"epoch {ep:4d} | train loss {history['train_loss'][-1]:.6g} | "
                f"valid loss {vloss:.6g} | valid obj {vobj:.6g} | "
                f"valid ineq max {vineq_max:.3e} | valid feas {100*vfeas:.1f}% | "
                f"cum train {t_train_total:.1f}s"
            )
            if eval_fn is not None:
                eval_fn(ep, solver)

    if best["state"] is not None:
        solver.load_state_dict(best["state"])
    history["best_epoch"] = best["epoch"]
    history["train_time_s"] = t_train_total
    history["n_nonfinite_batches"] = n_nonfinite
    if best["state"] is None:
        log_fn("WARNING: no validation epoch produced a finite score; "
               "the final (not the best) weights are kept.")
    return history
