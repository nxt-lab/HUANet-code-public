# Third-party attribution

This directory contains a re-implementation of

> **DC3: A learning method for optimization with hard constraints**
> Priya L. Donti, David Rolnick, J. Zico Kolter. ICLR 2021.
> Paper: <https://arxiv.org/abs/2104.12225>
> Official code: <https://github.com/locuslab/DC3>

The official implementation is released under the **Apache License, Version 2.0**;
a verbatim copy of that license is kept next to this file as
`LICENSE-Apache-2.0-DC3` (retrieved from
`https://raw.githubusercontent.com/locuslab/DC3/main/LICENSE`).

The following files in `DC3/` are *derivative works* of the official
implementation (they follow the structure and, in places, the exact algebra of
`method.py` and `utils.py` from `locuslab/DC3`) and therefore carry an
Apache-2.0 header:

* `DC3/common/dc3.py`      — adapted from `method.py` (`total_loss`, `grad_steps`,
  `grad_steps_all`, `NNSolver`, `train_net`, `eval_net`)
* `DC3/common/completion.py` — adapted from `utils.py::SimpleProblem`
  (`complete_partial`, `ineq_partial_grad`, the determinant-based search for an
  invertible set of dependent variables)
* `DC3/common/nets.py`     — adapted from `method.py::NNSolver`

Everything else (the two problem formulations, data generation, reference
solvers, benchmarking and reporting) is new code written for this repository.

The *problem formulations* themselves are transcriptions of the Julia code in
`examples/entr_max` and `examples/power_grid` of this repository, which
remains the source of truth.
