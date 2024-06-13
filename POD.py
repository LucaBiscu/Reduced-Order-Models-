import numpy as np
import GeDiM4Py as gedim
from functools import partial
from weakforms import *
from FOM import newton_solver


def create_snapshots(lib, problem_data, mus, forcing_term):
    fom_solver = partial(newton_solver, lib, problem_data, forcing_term)
    return np.stack([fom_solver(mu)[0] for mu in mus])


def pod_base(
    snapshots, inner_product, retained_energy=0.99, max_n=20, return_energy=False
):
    e_vals, e_vecs = np.linalg.eig(snapshots @ inner_product @ snapshots.T)
    assert np.all(
        np.isclose(e_vals.imag, 0)
    ), "Eigen values have non zero imaginary part"
    cs = np.cumsum(e_vals.real)
    N = min(np.argmax((cs / cs[-1]) > retained_energy) + 1, max_n)
    basis = snapshots.T @ e_vecs.real[:, :N]
    norm = np.sqrt(np.diag(basis.T @ inner_product @ basis))
    if return_energy:
        return basis / norm, cs
    else:
        return basis / norm


def newton_solver_pod(lib, problem_data, basis, mu, max_iterations=10, tol=1e-6):
    diff_c = partial(newton_diffusion_coefficient, mu=mu)
    reac_c = partial(newton_reaction_coefficient, mu=mu)
    nl_reac_t = partial(newton_nonlinear_reaction_term, mu=mu)
    forc_t = partial(forcing_term, mu=mu)
    nl_forc_c = partial(newton_nonlinear_forcing_coefficient, mu=mu)
    nl_forc_t = partial(newton_nonlinear_forcing_term, mu=mu)
    nl_forc_d = partial(newton_nonlinear_forcing_derivative, mu=mu)

    u_kn = np.zeros(basis.shape[1], order="F")
    u_k = basis @ u_kn
    u_strong = np.zeros(problem_data["NumberStrongs"], order="F")
    k = 0
    residual_norm = 1.0
    solution_norm = 10 * tol * residual_norm
    stiff_m, _ = gedim.AssembleStiffnessMatrix(diff_c, problem_data, lib)
    forc_m = gedim.AssembleForcingTerm(forc_t, problem_data, lib)
    stiff_m = basis.T @ stiff_m @ basis
    forc_m = basis.T @ forc_m
    while k < max_iterations and residual_norm > tol * solution_norm:
        u_k = basis @ u_kn
        react_m, _ = gedim.AssembleNonLinearReactionMatrix(
            reac_c, nl_reac_t, u_k, u_strong, problem_data, lib
        )
        nl_forc_m = gedim.AssembleNonLinearForcingTerm(
            nl_forc_c, nl_forc_t, u_k, u_strong, problem_data, lib
        )
        nl_forc_d_m = gedim.AssembleNonLinearDerivativeForcingTerm(
            ones_derivative, nl_forc_d, u_k, u_strong, problem_data, lib
        )

        du = gedim.LUSolver(
            stiff_m + basis.T @ react_m @ basis,
            forc_m - basis.T @ (nl_forc_m + nl_forc_d_m),
            lib,
        )
        u_kn = u_kn + du
        u_k = basis @ u_kn
        solution_norm = gedim.ComputeErrorL2(zeros, u_k, u_strong, lib)
        residual_norm = gedim.ComputeErrorL2(zeros, basis @ du, u_strong, lib)
        k = k + 1
    return u_k, u_strong, residual_norm / solution_norm, k
