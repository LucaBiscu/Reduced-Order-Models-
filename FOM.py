import numpy as np
import GeDiM4Py as gedim
from functools import partial
from weakforms import *


def newton_solver(lib, problem_data, forcing_term, mu, max_iterations=10, tol=1e-6):
    diff_c = partial(newton_diffusion_coefficient, mu=mu)
    reac_c = partial(newton_reaction_coefficient, mu=mu)
    nl_reac_t = partial(newton_nonlinear_reaction_term, mu=mu)
    forc_t = partial(forcing_term, mu=mu)
    nl_forc_c = partial(newton_nonlinear_forcing_coefficient, mu=mu)
    nl_forc_t = partial(newton_nonlinear_forcing_term, mu=mu)
    nl_forc_d = partial(newton_nonlinear_forcing_derivative, mu=mu)

    u_k = np.zeros(problem_data["NumberDOFs"], order="F")
    u_strong = np.zeros(problem_data["NumberStrongs"], order="F")
    k = 0
    residual_norm = 1.0
    solution_norm = 10 * tol * residual_norm
    stiff_m, _ = gedim.AssembleStiffnessMatrix(diff_c, problem_data, lib)
    forc_m = gedim.AssembleForcingTerm(forc_t, problem_data, lib)
    while k < max_iterations and residual_norm > tol * solution_norm:
        react_m, _ = gedim.AssembleNonLinearReactionMatrix(
            reac_c, nl_reac_t, u_k, u_strong, problem_data, lib
        )
        nl_forc_m = gedim.AssembleNonLinearForcingTerm(
            nl_forc_c, nl_forc_t, u_k, u_strong, problem_data, lib
        )
        nl_forc_d_m = gedim.AssembleNonLinearDerivativeForcingTerm(
            ones_derivative, nl_forc_d, u_k, u_strong, problem_data, lib
        )

        du = gedim.LUSolver(stiff_m + react_m, forc_m - nl_forc_m - nl_forc_d_m, lib)
        u_k = u_k + du

        solution_norm = gedim.ComputeErrorL2(zeros, u_k, u_strong, lib)
        residual_norm = gedim.ComputeErrorL2(zeros, du, u_strong, lib)
        k = k + 1
    return u_k, u_strong, residual_norm / solution_norm, k
