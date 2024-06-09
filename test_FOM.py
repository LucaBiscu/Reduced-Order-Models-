import GeDiM4Py as gedim
import numpy as np
from weakforms import *
from FOM import newton_solver

# setup lib
lib = gedim.ImportLibrary("/content/CppToPython/release/GeDiM4Py.so")
gedim.Initialize({"GeometricTolerance": 1.0e-8}, lib)

for mesh_size in (1e-2, 1e-3):
    # setup problem
    order = 1
    domain = {
        "SquareEdge": 1.0,
        "VerticesBoundaryCondition": [1, 1, 1, 1],
        "EdgesBoundaryCondition": [1, 1, 1, 1],
        "DiscretizationType": 1,
        "MeshCellsMaximumArea": mesh_size,
    }
    _, mesh = gedim.CreateDomainSquare(domain, lib)

    discreteSpace = {"Order": order, "Type": 1,
                     "BoundaryConditionsType": [1, 2]}
    problem_data, dofs, strongs = gedim.Discretize(discreteSpace, lib)
    n_dofs = dofs.shape[1]

    u, u_strong, _, k = newton_solver(
        lib, problem_data, test_forcing_term, np.array([.5, .6]), tol=1e-6)
    fom_error_l2 = gedim.ComputeErrorL2(test_exact_solution, u, u_strong, lib)
    fom_norm_l2 = gedim.ComputeErrorL2(zeros, u, u_strong, lib)
    fom_error_h1 = gedim.ComputeErrorH1(
        test_exact_solution_derivative, u, u_strong, lib)
    fom_norm_h1 = gedim.ComputeErrorH1(zeros_derivative, u, u_strong, lib)

    print(f"""Test FOM converged in {k} newton iterations, with l2 error {
          fom_error_l2 / fom_norm_l2}, and h1 error {fom_error_h1 / fom_norm_h1}""")
