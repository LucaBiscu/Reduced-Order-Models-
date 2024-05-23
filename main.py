import GeDiM4Py as gedim
import numpy as np
from weakforms import test_forcing_term, forcing_term
from FOM import newton_solver

# setup lib
lib = gedim.ImportLibrary('/content/CppToPython/release/GeDiM4Py.so')
gedim.Initialize({'GeometricTolerance': 1.0e-8}, lib)

# setup problem
mu = np.array([3.34, 4.45])
mesh_size = 1e-3
order = 1
domain = {'SquareEdge': 4.0, 'VerticesBoundaryCondition': [1, 1, 1, 1], 'EdgesBoundaryCondition': [
    1, 1, 1, 1], 'DiscretizationType': 1, 'MeshCellsMaximumArea': mesh_size}
_, mesh = gedim.CreateDomainSquare(domain, lib)

discreteSpace = {'Order': order, 'Type': 1, 'BoundaryConditionsType': [1, 2]}
problem_data, dofs, strongs = gedim.Discretize(discreteSpace, lib)

u, u_strong, relative_error, k = newton_solver(
    lib, problem_data, forcing_term, mu)

print(f"Iterations: {k} Relative err: {relative_error}")
gedim.PlotSolution(mesh, dofs, strongs, u, u_strong)
