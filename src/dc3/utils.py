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
        self.device = device

        self.num = int(self.b.shape[0])
        self.ydim = int(self.A.shape[1])
        self.xdim = int(self.b.shape[1])
        self.neq = int(self.A.shape[0])
        self.nineq = int(self.C.shape[0])
        self.nknowns = 0

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
        e_len = N + 1
        pb_sl = slice(e_len, e_len + N)
        r_sl = slice(e_len + N, e_len + 2 * N)

        P_B = self.params["S_B"] * Y[:, pb_sl]
        r = Y[:, r_sl]
        if self.params.get("substitute_m", False):
            if X is None:
                raise ValueError("Demand input X is required when m is substituted.")
            m = X[:, :N] + self.params["gamma"] * r - P_B
        else:
            m_sl = slice(3 * N + 1, 4 * N + 1)
            m = self.params["S_m"] * Y[:, m_sl]

        cycling_loss = (1.0 - self.params["mu"]) / (2.0 * self.params["mu"] ** 0.5)
        f = (
            self.params["p_k"] * self.params["Delta_t"]
            * torch.sum(m + cycling_loss * torch.abs(P_B), dim=1)
            + self.params["p_p"] * torch.sum(torch.relu(m), dim=1)
            + self.params["eta"] * torch.sum(torch.relu(torch.reciprocal(r) - 1.0), dim=1)
        )
        return f / self.params["J_ref"]

    def eq_resid(self, X, Y):
        if self.params.get("substitute_m", False):
            return Y @ self.A.T
        return Y @ self.A.T - X

    def ineq_resid(self, X, Y):
        return Y @ self.C.T - self.d

    def ineq_dist(self, X, Y):
        return torch.clamp(self.ineq_resid(X, Y), min=0.0)

    def eq_grad(self, X, Y):
        return 2.0 * self.eq_resid(X, Y) @ self.A

    def ineq_grad(self, X, Y):
        return 2.0 * self.ineq_dist(X, Y) @ self.C

    def ineq_partial_grad(self, X, Y):
        raise NotImplementedError("Energy DC3 uses full correction, not completion correction.")

    def complete_partial(self, X, Y):
        raise NotImplementedError("Energy DC3 does not use partial completion.")

    def process_output(self, X, Y):
        if self.params.get("substitute_m", False):
            N = self.params["N"]
            e_len = N + 1
            pb_sl = slice(e_len, e_len + N)
            r_sl = slice(e_len + N, e_len + 2 * N)
            out = torch.empty_like(Y)
            out[:, :r_sl.start] = self.z_L[:r_sl.start] + (
                self.z_U[:r_sl.start] - self.z_L[:r_sl.start]
            ) * torch.sigmoid(Y[:, :r_sl.start])
            out[:, r_sl] = self.params["r_min"] + torch.nn.functional.softplus(Y[:, r_sl])
            return out
        return self.z_L + (self.z_U - self.z_L) * torch.sigmoid(Y)
