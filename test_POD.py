import GeDiM4Py as gedim
import numpy as np
import time
from weakforms import forcing_term, ones
from FOM import newton_solver
from POD import create_snapshots, pod_base, newton_solver_pod

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

fom_solutions, pod_solutions = (np.zeros((n_test, n_dofs)) for _ in (0, 1))
fom_times, pod_times = (np.zeros((n_test)) for _ in (0, 1))

print(f"Evaluating FOM & POD on test set...")
for i, mu in enumerate(test_set):
    fom_times[i] = time.process_time()
    fom_solutions[i] = newton_solver(lib, problem_data, forcing_term, mu)[0]
    fom_times[i] = time.process_time() - fom_times[i]
    pod_times[i] = time.process_time()
    pod_solutions[i] = newton_solver_pod(lib, problem_data, basis, mu)[0]
    pod_times[i] = time.process_time() - pod_times[i]


error = fom_solutions - pod_solutions
inner_product, _ = gedim.AssembleStiffnessMatrix(ones, problem_data, lib)
error_norm = np.sqrt(np.abs(np.diag(error @ inner_product @ error.T)))
fom_norm = np.sqrt(np.abs(np.diag(fom_solutions @ inner_product @ fom_solutions.T)))
relative_error = error_norm / fom_norm
speed_up = fom_times / pod_times

print(f"Speed up {np.mean(speed_up):.2} ± {np.std(speed_up):.2}")
print(
    f"""POD Relative Error {np.mean(relative_error):.2E} ± {
        np.std(relative_error):.2E}"""
)
