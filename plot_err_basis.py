import GeDiM4Py as gedim
import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from itertools import product
from weakforms import forcing_term, ones, zeros, zeros_derivative
from PiNN import MLP
from FOM import newton_solver
from PODNN import projector, train_podnn
from POD import create_snapshots, pod_base, newton_solver_pod

# setup lib & torch
lib = gedim.ImportLibrary("/content/CppToPython/release/GeDiM4Py.so")
gedim.Initialize({"GeometricTolerance": 1.0e-8}, lib)
torch.set_default_dtype(torch.float32)

n_train, n_test = 10, 100
base_dimensions = np.arange(1, n_train + 1)
error_l2 = np.zeros((2, len(base_dimensions), n_test))
error_h1 = np.zeros_like(error_l2)

order = 1
mesh_size = 1e-3
domain = {
    "SquareEdge": 1.0,
    "VerticesBoundaryCondition": [1, 1, 1, 1],
    "EdgesBoundaryCondition": [1, 1, 1, 1],
    "DiscretizationType": 1,
    "MeshCellsMaximumArea": mesh_size,
}
_, mesh = gedim.CreateDomainSquare(domain, lib)

discreteSpace = {"Order": order, "Type": 1, "BoundaryConditionsType": [1, 2]}
problem_data, dofs, strongs = gedim.Discretize(discreteSpace, lib)
n_dofs = dofs.shape[1]
n_strongs = strongs.shape[1]


def l2_error(fom, rom):
    return gedim.ComputeErrorL2(
        zeros, fom - rom, np.zeros(n_strongs), lib
    ) / gedim.ComputeErrorL2(zeros, fom, np.zeros(n_strongs), lib)


def h1_error(fom, rom):
    return gedim.ComputeErrorH1(
        zeros_derivative, fom - rom, np.zeros(n_strongs), lib
    ) / gedim.ComputeErrorH1(zeros_derivative, fom, np.zeros(n_strongs), lib)


train_set = np.random.uniform(0.1, 1, size=(n_train, 2))
test_set = np.random.uniform(0.1, 1, size=(n_test, 2))

# extract basis
inner_product, _ = gedim.AssembleStiffnessMatrix(ones, problem_data, lib)
snapshots = create_snapshots(lib, problem_data, train_set, forcing_term)
whole_basis, energy = pod_base(
    snapshots, inner_product, retained_energy=1, max_n=n_train, return_energy=True
)

_, ax = plt.subplots()
ax.set_title("Energy of the covariance Matrix")
ax.set_xlabel("Dimension of reduced basis")
ax.set_ylabel("Retained energy")
ax.plot([0] + list(base_dimensions), [0] + list(energy / energy[-1]))
ax.set_xticks([0] + list(base_dimensions))
plt.savefig("Images/rbm_energy.png", bbox_inches="tight")


# setup problem
for b, base_dim in enumerate(base_dimensions):
    basis = whole_basis[:, :base_dim]

    # train network
    network = MLP(2, basis.shape[1], 10, 5)
    proj_snapshots = projector(snapshots, basis, inner_product)

    # Neural network
    network = train_podnn(network, 1000, train_set, proj_snapshots, log=100).to("cpu")

    print(f"Evaluating FOM & PODNN on test set...")
    with torch.no_grad():
        for i, mu in enumerate(test_set):
            net_mu = torch.tensor(mu).float().unsqueeze(0)
            fom = newton_solver(lib, problem_data, forcing_term, mu)[0]
            podnn = basis @ network(net_mu).squeeze(0).detach().numpy()
            pod = newton_solver_pod(lib, problem_data, basis, mu)[0]
            error_l2[:, b, i] = np.array([l2_error(fom, pod), l2_error(fom, podnn)])
            error_h1[:, b, i] = np.array([h1_error(fom, pod), h1_error(fom, podnn)])

plt.yticks(minor=True)
_, ax = plt.subplots()
ax.set_title("H1 relative error")
ax.set_xlabel("Dimension of reduced basis")
ax.set_ylabel("avg. H1 relative error")
ax.set_yscale("log")
ax.errorbar(
    base_dimensions,
    error_h1[0, :, :].mean(axis=-1),
    yerr=error_h1[0, :, :].std(axis=-1),
    label="POD",
    fmt="o",
    capsize=2,
)
ax.errorbar(
    base_dimensions,
    error_h1[1, :, :].mean(axis=-1),
    yerr=error_h1[1, :, :].std(axis=-1),
    label="PODNN",
    fmt="o",
    capsize=2,
)
ax.legend()
plt.savefig("Images/rbm_h1_basis.png", bbox_inches="tight")


_, ax = plt.subplots()
ax.set_title("L2 relative error")
ax.set_xlabel("Dimension of reduced basis")
ax.set_ylabel("avg. L2 relative error")
ax.set_yscale("log")
ax.errorbar(
    base_dimensions,
    error_l2[0, :, :].mean(axis=-1),
    yerr=error_l2[0, :, :].std(axis=-1),
    fmt="o",
    label="POD",
    capsize=2,
)
ax.errorbar(
    base_dimensions,
    error_l2[1, :, :].mean(axis=-1),
    yerr=error_l2[1, :, :].std(axis=-1),
    fmt="o",
    label="PODNN",
    capsize=2,
)
ax.legend()
plt.savefig("Images/rbm_l2_basis.png", bbox_inches="tight")
