import GeDiM4Py as gedim
import numpy as np
from weakforms import test_forcing_term, forcing_term, ones
from FOM import newton_solver
from POD import pod_base, newton_solver_pod

# setup lib
lib = gedim.ImportLibrary("/content/CppToPython/release/GeDiM4Py.so")
gedim.Initialize({"GeometricTolerance": 1.0e-8}, lib)

# setup problem
mu = np.array([0.34, 0.45])
mesh_size = 1e-2
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


# Solve FOM
u, u_strong, relative_error, k = newton_solver(lib, problem_data, forcing_term, mu)

print(f"Iterations: {k} Relative err: {relative_error}")
gedim.PlotSolution(mesh, dofs, strongs, u, u_strong)

# Extract Base
train_set = np.random.uniform(0.1, 1, size=(100, 2))
basis = pod_base(lib, problem_data, train_set)
print(f"found basis of dimensions {basis.shape}")

u_pod, u_pod_strong, _, _ = newton_solver_pod(
    lib, problem_data, forcing_term, basis, mu
)

error = u - u_pod
inner_product, _ = gedim.AssembleStiffnessMatrix(ones, problem_data, lib)
error_norm = np.sqrt(np.abs(error.T @ inner_product @ error))
fom_norm = np.sqrt(np.abs(u.T @ inner_product @ u))
print(f"POD Relative Error {error_norm / fom_norm}")
