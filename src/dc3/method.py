import argparse
import operator
import os
import pickle
import sys
import time
from functools import reduce
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.energy.utils import energy_slices
from dc3.utils import EnergyProblem, my_hash, str_to_bool

with (Path(__file__).resolve().parents[2] / "examples" / "energy" / "cfg.yaml").open(encoding="utf-8") as file:
    cfg = yaml.safe_load(file)

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def problem_variable_bounds(problem):
    N = int(problem["N"])
    m, u, pc, soc = energy_slices(N)
    z_L = np.empty(4 * N, dtype=np.float64)
    z_U = np.empty(4 * N, dtype=np.float64)
    z_L[m], z_U[m] = float(problem["m_min"]), float(problem["m_max"])
    z_L[u], z_U[u] = float(problem["u_min"]), float(problem["u_max"])
    z_L[pc], z_U[pc] = float(problem["pc_min"]), float(problem["pc_max"])
    z_L[soc], z_U[soc] = float(problem["s_min"]), float(problem["s_max"])
    return z_L, z_U


def energy_params(problem, J_ref):
    keys = (
        "N", "Delta_t", "c_e", "c_p", "c_d", "mu", "a", "delta",
        "kappa_m", "kappa_c", "pc_min",
    )
    return {**{key: problem[key] for key in keys}, "J_ref": J_ref}


def dataset_paths(problem, n_samples):
    n_var = 4 * int(problem["N"])
    root = Path(__file__).resolve().parents[2] / cfg["data"]["root_subdir"] / f"n_var_{n_var}"
    return root / "datasets" / f"datasets_{n_samples}.npz", root


def load_energy_data(args):
    problem = cfg["problem"]
    N = int(problem["N"])
    n_samples = int(args["nSamples"])
    dataset_path, root = dataset_paths(problem, n_samples)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run generate.py first.")

    z_L, z_U = problem_variable_bounds(problem)
    with np.load(dataset_path) as dataset:
        if dataset["schema_version"].item() != str(problem["schema_version"]):
            raise ValueError("DC3 dataset schema does not match the energy configuration.")
        b_all = dataset["lam"]
        A = dataset["A"]
        C = dataset["C"]
        d = dataset["d"]
        net_demand = dataset["net_demand"]
        params = energy_params(problem, float(dataset["J_ref"]))
    expected = ((2 * N + 1, 4 * N), (5 * N, 4 * N), (5 * N,))
    if (A.shape, C.shape, d.shape) != expected:
        raise ValueError(f"DC3 energy dimensions {(A.shape, C.shape, d.shape)} do not match {expected}.")

    data = EnergyProblem(
        A=A,
        C=C,
        d=d,
        b_all=b_all,
        z_L=z_L,
        z_U=z_U,
        params=params,
        net_demand=net_demand,
        train_percent=float(args["trainPercent"]),
        val_percent=float(args["valPercent"]),
        device=DEVICE,
    )
    return data, root


def main():
    parser = argparse.ArgumentParser(description="DC3 method for energy management")
    parser.add_argument("--nSamples", type=int, default=int(cfg["data"]["n_samples"]))
    parser.add_argument("--trainPercent", type=float, default=float(cfg["data"]["train_percent"]))
    parser.add_argument("--valPercent", type=float, default=float(cfg["data"]["val_percent"]))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batchSize", type=int)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hiddenSize", type=int)
    parser.add_argument("--softWeight", type=float, default=1000.0)
    parser.add_argument("--softWeightEqFrac", type=float, default=0.5)
    parser.add_argument("--useCompl", type=str_to_bool, default=True)
    parser.add_argument("--useTrainCorr", type=str_to_bool, default=True)
    parser.add_argument("--useTestCorr", type=str_to_bool, default=True)
    parser.add_argument("--corrMode", type=str, default="partial", choices=["partial", "full"])
    parser.add_argument("--corrTrainSteps", type=int, default=5)
    parser.add_argument("--corrTestMaxSteps", type=int, default=20)
    parser.add_argument("--corrEps", type=float, default=1e-4)
    parser.add_argument("--corrLr", type=float, default=1e-5)
    parser.add_argument("--corrMomentum", type=float, default=0.0)
    parser.add_argument("--resultsSaveFreq", type=int, default=100)
    parser.add_argument("--printFreq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=int(cfg["problem"]["seed"]))
    args = vars(parser.parse_args())

    training_cfg = cfg["training"]
    args["epochs"] = int(args["epochs"] or training_cfg.get("n_epochs", 500))
    args["batchSize"] = int(args["batchSize"] or training_cfg.get("batch_size", 256))
    args["hiddenSize"] = int(args["hiddenSize"] or cfg["neural_net"]["hidden_layers"][0])

    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    data, root = load_energy_data(args)
    for attr, var in vars(data).items():
        if torch.is_tensor(var):
            setattr(data, attr, var.to(DEVICE))

    save_dir = os.path.join(
        root,
        "dc3",
        "method",
        str(data),
        my_hash(str(sorted(args.items()))),
        str(time.time()).replace(".", "-"),
    )
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "args.dict"), "wb") as f:
        pickle.dump(args, f)

    print("=" * 80)
    print("Training Energy Management - DC3")
    print("=" * 80)
    print(f"device={DEVICE}")
    print(f"n_var={data.ydim}  n_eq={data.neq}  n_ineq={data.nineq}")
    print(f"train={data.n_train}  val={data.n_val}  test={data.n_test}")
    print(f"epochs={args['epochs']}  batch={args['batchSize']}  lr={args['lr']:.2e}")
    train_net(data, args, save_dir)


def train_net(data, args, save_dir):
    solver_net = NNSolver(data, args).to(DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=args["lr"])

    train_loader = DataLoader(TensorDataset(data.trainX), batch_size=args["batchSize"], shuffle=True)
    stats = {}
    best_valid = float("inf")
    best_state = None
    start_time = time.time()

    for i in range(args["epochs"]):
        solver_net.train()
        epoch_stats = {}
        for Xtrain, in train_loader:
            Xtrain = Xtrain.to(DEVICE)
            solver_opt.zero_grad()
            Yhat_train = solver_net(Xtrain)
            Ynew_train = grad_steps(data, Xtrain, Yhat_train, args)
            train_loss = total_loss(data, Xtrain, Ynew_train, args)
            train_loss = torch.mean(train_loss)
            train_loss.backward()
            solver_opt.step()

            dict_agg(epoch_stats, "train_loss", train_loss.detach().cpu().item(), op="sum")
            dict_agg(epoch_stats, "train_time", time.time() - start_time, op="sum")

        if i % args["resultsSaveFreq"] == 0:
            stats[i] = eval_net(data, data.validX, solver_net, args, prefix="valid")
            stats[i].update(eval_net(data, data.testX, solver_net, args, prefix="test"))
            stats[i].update(epoch_stats)

            if stats[i]["valid_loss"] < best_valid:
                best_valid = stats[i]["valid_loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in solver_net.state_dict().items()}

            with open(os.path.join(save_dir, "stats.dict"), "wb") as f:
                pickle.dump(stats, f)

        if i % args["printFreq"] == 0:
            valid_stats = stats.get(i) or eval_net(data, data.validX, solver_net, args, prefix="valid")
            print(
                f"Epoch {i:5d} | train loss: {epoch_stats.get('train_loss', float('nan')):.6e} | "
                f"valid obj: {valid_stats['valid_obj']:.6e} | "
                f"eq: {valid_stats['valid_eq_max']:.6e} | "
                f"ineq: {valid_stats['valid_ineq_max']:.6e}"
            )

    if best_state is not None:
        solver_net.load_state_dict(best_state)

    torch.save(solver_net.state_dict(), os.path.join(save_dir, "solver_net.dict"))
    print(f"Saved DC3 model -> {os.path.join(save_dir, 'solver_net.dict')}")


def dict_agg(stats, key, value, op="concat"):
    if key in stats:
        if op == "sum":
            stats[key] += value
        elif op == "concat":
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value


def eval_net(data, X, solver_net, args, prefix):
    solver_net.eval()
    with torch.no_grad():
        start_time = time.time()
        Y = solver_net(X)
        raw_time = time.time() - start_time
        raw = total_loss(data, X, Y, args)
        raw_obj = data.obj_fn(Y, X)
        raw_eq = torch.abs(data.eq_resid(X, Y))
        raw_ineq = data.ineq_dist(X, Y)

    start_time = time.time()
    Ycorr, steps = grad_steps_all(data, X, Y, args)
    corr_time = time.time() - start_time
    corr = total_loss(data, X, Ycorr, args)
    corr_obj = data.obj_fn(Ycorr, X)
    corr_eq = torch.abs(data.eq_resid(X, Ycorr))
    corr_ineq = data.ineq_dist(X, Ycorr)

    return {
        f"{prefix}_loss": torch.mean(corr).detach().cpu().item(),
        f"{prefix}_obj": torch.mean(corr_obj).detach().cpu().item(),
        f"{prefix}_eq_max": torch.max(corr_eq).detach().cpu().item(),
        f"{prefix}_ineq_max": torch.max(corr_ineq).detach().cpu().item(),
        f"{prefix}_raw_loss": torch.mean(raw).detach().cpu().item(),
        f"{prefix}_raw_obj": torch.mean(raw_obj).detach().cpu().item(),
        f"{prefix}_raw_eq_max": torch.max(raw_eq).detach().cpu().item(),
        f"{prefix}_raw_ineq_max": torch.max(raw_ineq).detach().cpu().item(),
        f"{prefix}_raw_time": raw_time / max(int(X.shape[0]), 1),
        f"{prefix}_corr_time": corr_time / max(int(X.shape[0]), 1),
        f"{prefix}_steps": steps,
    }


def total_loss(data, X, Y, args):
    obj_cost = data.obj_fn(Y, X)
    ineq_dist = data.ineq_dist(X, Y)
    ineq_cost = torch.norm(ineq_dist, dim=1)
    eq_cost = torch.norm(data.eq_resid(X, Y), dim=1)
    return (
        obj_cost
        + args["softWeight"] * (1.0 - args["softWeightEqFrac"]) * ineq_cost
        + args["softWeight"] * args["softWeightEqFrac"] * eq_cost
    )


def grad_steps(data, X, Y, args):
    if args["useTrainCorr"]:
        return _grad_steps(data, X, Y, args, args["corrTrainSteps"])
    return Y


def grad_steps_all(data, X, Y, args):
    if not args["useTestCorr"]:
        return Y, 0

    Y_new = Y
    for i in range(args["corrTestMaxSteps"]):
        Y_new = _grad_steps(data, X, Y_new, args, 1)
        with torch.no_grad():
            eq_max = torch.max(torch.abs(data.eq_resid(X, Y_new))).detach().cpu().item()
            ineq_max = torch.max(data.ineq_dist(X, Y_new)).detach().cpu().item()
        if max(eq_max, ineq_max) <= args["corrEps"]:
            return Y_new, i + 1
    return Y_new, args["corrTestMaxSteps"]


def _grad_steps(data, X, Y, args, steps):
    Y_new = Y
    old_step = 0.0
    for _ in range(steps):
        if args["corrMode"] == "partial":
            assert args["useCompl"]
            Y_step = data.ineq_partial_grad(X, Y_new)
        elif args["corrMode"] == "full":
            Y_step = (
                args["softWeight"] * args["softWeightEqFrac"] * data.eq_grad(X, Y_new)
                + args["softWeight"] * (1.0 - args["softWeightEqFrac"]) * data.ineq_grad(X, Y_new)
            )
        else:
            raise NotImplementedError

        new_step = args["corrLr"] * Y_step + args["corrMomentum"] * old_step
        if args["corrMode"] == "partial":
            partial = Y_new[:, data.partial_vars] - new_step
            Y_new = data.complete_physical_partial(X, partial)
        else:
            Y_new = Y_new - new_step
        old_step = new_step
    return Y_new


class NNSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        input_dim = int(data.trainX.shape[1])
        layer_sizes = [input_dim, args["hiddenSize"], args["hiddenSize"]]
        layers = reduce(
            operator.add,
            [
                [
                    nn.Linear(a, b),
                    nn.BatchNorm1d(b),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                ]
                for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])
            ],
        )
        output_dim = data.ydim - data.nknowns
        if args["useCompl"]:
            layers += [nn.Linear(layer_sizes[-1], output_dim), nn.Sigmoid()]
        else:
            layers += [nn.Linear(layer_sizes[-1], output_dim)]
        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        output_layer = self.net[-2] if args["useCompl"] else self.net[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, x):
        out = self.net(x)
        if self._data.nknowns > 0:
            return self._data.complete_partial(x, out)
        return self._data.process_output(x, out)


def expand_reduced_solution(z_reduced, lam, problem):
    del lam, problem
    return np.asarray(z_reduced)


def run_dc3(lam_test, root):
    checkpoints = list((root / "dc3" / "method").rglob("solver_net.dict"))
    if not checkpoints:
        raise FileNotFoundError(f"DC3 checkpoint not found under {root / 'dc3'}. Run method.py first.")
    checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime)
    with (checkpoint.parent / "args.dict").open("rb") as file:
        args = pickle.load(file)
    args["corrTestMaxSteps"] = 200
    args["corrEps"] = 1e-5

    data, _ = load_energy_data(args)
    if data.testX.shape != lam_test.shape or not np.array_equal(data.testX.cpu().numpy(), lam_test):
        raise ValueError("DC3 checkpoint and energy benchmark use different test samples.")

    model = NNSolver(data, args).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        raw = model(data.testX)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        y, steps = grad_steps_all(data, data.testX, raw, args)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    print(f"DC3 checkpoint: {checkpoint}")
    print(f"DC3 correction steps: {steps}")
    x = expand_reduced_solution(y.cpu().numpy(), lam_test, cfg["problem"])
    return x, np.full(len(x), elapsed / len(x))


if __name__ == "__main__":
    main()
