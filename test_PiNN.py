import torch.nn as nn
from PiNN import *
import GeDiM4Py as gedim
import numpy as np
import time
from weakforms import test_forcing_term, forcing_term, ones
from FOM import newton_solver

# setup lib
lib = gedim.ImportLibrary("/content/CppToPython/release/GeDiM4Py.so")
gedim.Initialize({"GeometricTolerance": 1.0e-8}, lib)

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
print(dofs.shape)

# snapshots
n_test = 100
test_set = np.random.uniform(0.1, 1, size=(n_test, 2))

# Train NN
train_steps = 10000
grid_side = 50 
print(f"Training netowrk for {train_steps} train steps on a {grid_side ** 2} points grid...")
network = MLP(4, 1, 100, 5)
network = train_pinn(
    network,
    train_steps,
    grid_side,
    torch.tensor([0.0, 1.0]),
    torch.tensor([0.1, 1.0]),
    log=1000,
).to('cpu')
print("Finished training!")

fom_solutions, pinn_solutions = (np.zeros((n_test, n_dofs)) for _ in (0, 1))
fom_times, pinn_times = (np.zeros((n_test)) for _ in (0, 1))

print(f"Evaluating FOM & PiNN on test set...")
with torch.no_grad():
    net_x = torch.tensor(dofs[:2, :]).float().T
    for i, mu in enumerate(test_set):
        net_mu = torch.tensor(mu).float().unsqueeze(0).repeat(n_dofs, 1)
        net_input = torch.cat((net_x, net_mu), dim = -1)
        fom_times[i] = time.process_time()
        fom_solutions[i] = newton_solver(lib, problem_data, forcing_term, mu)[0]
        fom_times[i] = time.process_time() - fom_times[i]
        pinn_times[i] = time.process_time()
        pinn_solutions[i] = network(net_input).squeeze().detach().numpy()
        pinn_times[i] = time.process_time() - pinn_times[i]

error = fom_solutions - pinn_solutions
inner_product, _ = gedim.AssembleStiffnessMatrix(ones, problem_data, lib)
error_norm = np.sqrt(np.abs(np.diag(error @ inner_product @ error.T)))
fom_norm = np.sqrt(np.abs(np.diag(fom_solutions @ inner_product @ fom_solutions.T)))
relative_error = error_norm / fom_norm
speed_up = fom_times / pinn_times

gedim.PlotSolution(mesh, dofs, strongs, pinn_solutions[-1], np.zeros(problem_data['NumberStrongs']))

print(f"Speed up {np.mean(speed_up):.2} ± {np.std(speed_up):.2}")
print(
    f"""PiNN Relative Error {np.mean(relative_error):.2E} ± {
        np.std(relative_error):.2E}"""
)
