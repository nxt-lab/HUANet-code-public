import hashlib

import torch

torch.set_default_dtype(torch.float64)


def my_hash(string):
    return hashlib.sha1(bytes(string, "utf-8")).hexdigest()


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    if value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


class EnergyProblem:
    def __init__(
        self,
        A,
        C,
        d,
        b_all,
        z_L,
        z_U,
        params,
        net_demand,
        train_percent,
        val_percent,
        device,
    ):
        self.A = torch.as_tensor(A, device=device)
        self.C = torch.as_tensor(C, device=device)
        self.G = self.C
        self.d = torch.as_tensor(d, device=device)
        self.h = self.d
        self.b = torch.as_tensor(b_all, device=device)
        self.z_L = torch.as_tensor(z_L, device=device)
        self.z_U = torch.as_tensor(z_U, device=device)
        self.params = params
        self.net_demand = torch.as_tensor(net_demand, device=device)
        self.device = device

        self.num = int(self.b.shape[0])
        self.ydim = int(self.A.shape[1])
        self.xdim = int(self.b.shape[1])
        self.neq = int(self.A.shape[0])
        self.nineq = int(self.C.shape[0])
        N = int(params["N"])
        m, u, pc, soc = (
            slice(0, N), slice(N, 2 * N), slice(2 * N, 3 * N), slice(3 * N, 4 * N)
        )
        self.partial_vars = torch.as_tensor(
            list(range(u.start, u.stop - 1)) + list(range(pc.start, pc.stop)),
            dtype=torch.long, device=device,
        )
        self.other_vars = torch.as_tensor(
            [u.stop - 1] + list(range(soc.start, soc.stop)) + list(range(m.start, m.stop)),
            dtype=torch.long, device=device,
        )
        self.nknowns = int(self.other_vars.numel())
        A_partial = self.A[:, self.partial_vars]
        A_other = self.A[:, self.other_vars]
        if A_other.shape[0] != A_other.shape[1]:
            raise ValueError("DC3 dependent equality block must be square.")
        self.A_other_inv_A_partial = torch.linalg.solve(A_other, A_partial)
        self.A_other_inv = torch.linalg.inv(A_other)
        self.G_eff = self.C[:, self.partial_vars] - self.C[:, self.other_vars] @ self.A_other_inv_A_partial
        norms = torch.linalg.vector_norm(self.G_eff, dim=1)
        nonzero = norms > 1e-9 * norms.max()
        self.ineq_row_scale = torch.ones_like(norms)
        self.ineq_row_scale[nonzero] = 1.0 / norms[nonzero]
        self.ineq_row_scale /= self.ineq_row_scale[nonzero].median()
        self.ineq_row_scale[~nonzero] = 1.0

        self.train_frac = float(train_percent)
        self.valid_frac = float(val_percent)
        self.n_train = max(1, int(self.num * self.train_frac))
        self.n_val = max(1, int(self.num * self.valid_frac))
        self.n_train = min(self.n_train, self.num - self.n_val - 1)
        self.n_test = self.num - self.n_train - self.n_val

    def __str__(self):
        return f"EnergyProblem-{self.ydim}-{self.neq}-{self.nineq}-{self.num}"

    @property
    def trainX(self):
        return self.b[: self.n_train]

    @property
    def validX(self):
        return self.b[self.n_train : self.n_train + self.n_val]

    @property
    def testX(self):
        return self.b[self.n_train + self.n_val :]

    def obj_fn(self, Y, X=None):
        N = self.params["N"]
        m, u, pc = Y[:, :N], Y[:, N:2 * N], Y[:, 2 * N:3 * N]
        pc_eval = torch.clamp(pc, min=self.params["pc_min"])
        cycling_loss = (1.0 - self.params["mu"]) / (2.0 * self.params["mu"] ** 0.5)
        smooth_abs = torch.sqrt(u.square() + self.params["delta"] ** 2) - self.params["delta"]
        f = (
            self.params["c_e"] * self.params["Delta_t"]
            * torch.sum(m + cycling_loss * smooth_abs, dim=1)
            + self.params["c_p"] * torch.sum(
                torch.nn.functional.softplus(self.params["kappa_m"] * m) / self.params["kappa_m"], dim=1
            )
            + self.params["c_d"] * torch.sum(
                torch.nn.functional.softplus(
                    self.params["kappa_c"] * (self.params["a"] / pc_eval - 1.0)
                ) / self.params["kappa_c"], dim=1
            )
        )
        return f / self.params["J_ref"]

    def eq_resid(self, X, Y):
        return Y @ self.A.T - self.eq_rhs(X)

    def eq_rhs(self, X):
        N = int(self.params["N"])
        x0 = X[:, 0]
        rhs = torch.zeros((len(X), 2 * N + 1), dtype=X.dtype, device=X.device)
        rhs[:, 0] = x0
        rhs[:, N] = x0
        rhs[:, N + 1:] = self.net_demand.to(dtype=X.dtype, device=X.device)
        return rhs

    def ineq_resid(self, X, Y):
        return Y @ self.C.T - self.d

    def ineq_dist(self, X, Y):
        return torch.clamp(self.ineq_resid(X, Y), min=0.0)

    def eq_grad(self, X, Y):
        return 2.0 * self.eq_resid(X, Y) @ self.A

    def ineq_grad(self, X, Y):
        return 2.0 * self.ineq_dist(X, Y) @ self.C

    def ineq_partial_grad(self, X, Y):
        weighted_residual = self.ineq_dist(X, Y) * self.ineq_row_scale.square()
        return 2.0 * weighted_residual @ self.G_eff

    def complete_partial(self, X, Y):
        lo = self.z_L[self.partial_vars]
        hi = self.z_U[self.partial_vars]
        partial = lo + (hi - lo) * Y
        return self.complete_physical_partial(X, partial)

    def complete_physical_partial(self, X, partial):
        rhs = self.eq_rhs(X)
        dependent = (rhs - partial @ self.A[:, self.partial_vars].T) @ self.A_other_inv.T
        full = torch.empty((len(X), self.ydim), dtype=partial.dtype, device=partial.device)
        full[:, self.partial_vars] = partial
        full[:, self.other_vars] = dependent
        return full

    def process_output(self, X, Y):
        return self.z_L + (self.z_U - self.z_L) * torch.sigmoid(Y)
