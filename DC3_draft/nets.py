# Copyright 2024-2026 the authors of this repository.
#
# Adapted from https://github.com/locuslab/DC3 (method.py, class NNSolver),
# licensed under the Apache License, Version 2.0.  See
# `DC3/common/LICENSE-Apache-2.0-DC3` and `DC3/common/NOTICE.md`.
"""The partial-variable prediction network.

Same recipe as DC3's ``NNSolver``: a stack of
``Linear -> BatchNorm1d -> ReLU -> Dropout`` blocks followed by a linear read-out
of ``n_y - n_eq`` partial variables, with Kaiming-normal initialisation of the
linear layers.  Depth/width/dropout/batch-norm are configurable here (DC3 hard
codes two hidden layers, dropout 0.2 and batch-norm on).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PartialVarNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_size: int = 200,
        n_hidden: int = 2,
        dropout: float = 0.2,
        batch_norm: bool = True,
        input_norm: bool = True,
        transform: "PartialTransform | None" = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.transform = transform
        # Standardisation of the (fixed) input distribution; the buffers are
        # filled from the *training* split only (see `fit_input_norm`).
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.input_norm = input_norm

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(d, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            d = hidden_size
        layers.append(nn.Linear(d, out_dim))

        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.net = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_input_norm(self, X: torch.Tensor) -> None:
        self.x_mean.copy_(X.mean(dim=0))
        std = X.std(dim=0)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        self.x_std.copy_(std)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.input_norm:
            X = (X - self.x_mean) / self.x_std
        out = self.net(X)
        return out if self.transform is None else self.transform(out)

    @torch.no_grad()
    def init_output_at(self, target: torch.Tensor, weight_scale: float = 1.0) -> None:
        """Make the *initial* prediction approximately equal to `target`.

        The read-out weights are shrunk by `weight_scale` and its bias is set to
        the pre-image of `target` under the output transform, so training starts
        from a sensible operating point instead of from a random one (for the
        eco-MPC that matters: a random `p` near zero sends `a/p` to ~1e10).
        """
        last = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        last.weight.mul_(weight_scale)
        last.bias.copy_(self.transform.inverse(target) if self.transform is not None else target)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PartialTransform(torch.nn.Module):
    """Optional bounded parametrisation of the partial variables.

    DC3's ACOPF model applies ``nn.Sigmoid()`` to the network output and then
    "interpolates between max and min values" before completion (see
    ``method.py::NNSolver.forward``).  The same idea is generalised here so that
    variables with only a *lower* bound - which is what the objective domains of
    both applications need (``w > 0``, ``p > 0``) - can be handled too:

        lo and hi finite     ->  lo + sigmoid(r) * (hi - lo)
        only lo finite       ->  lo + softplus(r)
        only hi finite       ->  hi - softplus(-r)
        neither finite       ->  r                 (identity)

    This constrains only the *prediction*.  The correction that follows operates
    on the transformed variables and may leave the box, exactly as in DC3; the
    corresponding bounds are also present in ``g`` so the correction pushes back.
    """

    def __init__(self, lo: torch.Tensor, hi: torch.Tensor):
        super().__init__()
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        self.register_buffer("both", torch.isfinite(lo) & torch.isfinite(hi))
        self.register_buffer("only_lo", torch.isfinite(lo) & ~torch.isfinite(hi))
        self.register_buffer("only_hi", ~torch.isfinite(lo) & torch.isfinite(hi))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        lo = torch.where(torch.isfinite(self.lo), self.lo, torch.zeros_like(self.lo))
        hi = torch.where(torch.isfinite(self.hi), self.hi, torch.zeros_like(self.hi))
        out = r
        out = torch.where(self.both, lo + torch.sigmoid(r) * (hi - lo), out)
        out = torch.where(self.only_lo, lo + torch.nn.functional.softplus(r), out)
        out = torch.where(self.only_hi, hi - torch.nn.functional.softplus(-r), out)
        return out

    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Pre-image of `y` (used to initialise the read-out bias)."""
        lo, hi = self.lo, self.hi
        out = y.clone()
        b = self.both
        if bool(b.any()):
            t = ((y - lo) / (hi - lo).clamp(min=1e-12)).clamp(1e-6, 1 - 1e-6)
            out = torch.where(b, torch.log(t / (1 - t)), out)
        ol = self.only_lo
        if bool(ol.any()):
            s = (y - lo).clamp(min=1e-12)
            out = torch.where(ol, s + torch.log(-torch.expm1(-s)), out)   # softplus^{-1}
        oh = self.only_hi
        if bool(oh.any()):
            s = (hi - y).clamp(min=1e-12)
            out = torch.where(oh, -(s + torch.log(-torch.expm1(-s))), out)
        return out
