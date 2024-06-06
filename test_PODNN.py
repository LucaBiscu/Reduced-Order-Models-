import GeDiM4Py as gedim
import numpy as np
import time
from weakforms import forcing_term, ones
from PiNN import MLP
from FOM import newton_solver
from PODNN import projector, train_podnn 
from POD import pod_base, create_snapshots
import torch
import torch.nn as nn

# setup lib & torch
lib = gedim.ImportLibrary("/content/CppToPython/release/GeDiM4Py.so")
gedim.Initialize({"GeometricTolerance": 1.0e-8}, lib)
torch.set_default_dtype(torch.float32)

# setup problem
mesh_size = 1e-3
order = 1
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

# snapshots
n_train, n_test = 10, 100
train_set = np.random.uniform(0.1, 1, size=(n_train, 2))
test_set = np.random.uniform(0.1, 1, size=(n_test, 2))

#extract basis
inner_product, _ = gedim.AssembleStiffnessMatrix(ones, problem_data, lib)
snapshots = create_snapshots(lib, problem_data, train_set, forcing_term)
print(f"Computing pod basis...")
basis_time = time.process_time()
basis = pod_base(snapshots, inner_product)
basis_time = time.process_time() - basis_time
print(f"Computed basis in {basis_time:.2}s")

# train network
network = MLP(2, basis.shape[1], 100, 5)
proj_snapshots = projector(snapshots, basis, inner_product)

# Neural network 
network = train_podnn(network, 1000, train_set, proj_snapshots, log=100).to('cpu')

fom_solutions, podnn_solutions = (np.zeros((n_test, n_dofs)) for _ in (0, 1))
fom_times, podnn_times = (np.zeros((n_test)) for _ in (0, 1))

print(f"Evaluating FOM & PiNN on test set...")
with torch.no_grad():
    for i, mu in enumerate(test_set):
        net_mu = torch.tensor(mu).float().unsqueeze(0)
        fom_times[i] = time.process_time()
        fom_solutions[i] = newton_solver(lib, problem_data, forcing_term, mu)[0]
        fom_times[i] = time.process_time() - fom_times[i]
        podnn_times[i] = time.process_time()
        podnn_solutions[i] = basis @ network(net_mu).squeeze(0).detach().numpy()
        podnn_times[i] = time.process_time() - podnn_times[i]

error = fom_solutions - podnn_solutions
inner_product, _ = gedim.AssembleStiffnessMatrix(ones, problem_data, lib)
error_norm = np.sqrt(np.abs(np.diag(error @ inner_product @ error.T)))
fom_norm = np.sqrt(np.abs(np.diag(fom_solutions @ inner_product @ fom_solutions.T)))
relative_error = error_norm / fom_norm
speed_up = fom_times / podnn_times

gedim.PlotSolution(mesh, dofs, strongs, podnn_solutions[-1], np.zeros(problem_data['NumberStrongs']))

print(f"Speed up {np.mean(speed_up):.2} ± {np.std(speed_up):.2}")
print(
    f"""PODNN Relative Error {np.mean(relative_error):.2E} ± {
        np.std(relative_error):.2E}"""
)
