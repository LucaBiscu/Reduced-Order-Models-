import numpy as np
import GeDiM4Py as gedim


def __assert_mu(mu):
    assert mu is not None, "mu must not be none"
    assert isinstance(mu, np.ndarray), "mu must be an numpy ndarray"
    assert len(mu) == 2, "mu must have 2 arguments"


def newton_diffusion_coefficient(numPoints, points, mu=None):
    return np.ones(numPoints, order='F').ctypes.data


def newton_reaction_coefficient(numPoints, points, mu=None):
    __assert_mu(mu)
    return (mu[0] * np.ones(numPoints, order='F')).ctypes.data


def newton_nonlinear_reaction_term(numPoints, points, u, u_x, u_y, mu=None):
    __assert_mu(mu)
    vu = gedim.make_nd_array(u, numPoints, np.double)
    return np.exp(mu[1] * vu, order='F').ctypes.data


def test_forcing_term(numPoints, points, mu=None):
    __assert_mu(mu)
    assert mu[1] != 0, "mu 1 is equal to zero, cannot divide by zero"
    matPoints = gedim.make_nd_matrix(points, (3, numPoints), np.double)
    linear = -32 * (matPoints[0, :] ** 2 + matPoints[1, :]
                    ** 2 - matPoints[0, :] - matPoints[1, :])
    nonlinear = mu[0] / mu[1] * np.exp(16 * matPoints[0, :] * matPoints[1, :] * (
        1 - matPoints[0, :]) * (1 - matPoints[1, :]))
    return (linear + nonlinear).ctypes.data


def forcing_term(numPoints, points, mu=None):
    matPoints = gedim.make_nd_matrix(points, (3, numPoints), np.double)
    return (100 * np.sin(2 * np.pi * matPoints[0, :]) * np.cos(2 * np.pi * matPoints[1, :])).ctypes.data


def newton_nonlinear_forcing_coefficient(numPoints, points, mu=None):
    __assert_mu(mu)
    assert mu[1] != 0, "mu 1 is equal to zero, cannot divide by zero"
    return (mu[0] / mu[1] * np.ones(numPoints, order='F')).ctypes.data


def newton_nonlinear_forcing_term(numPoints, points, u, u_x, u_y, mu=None):
    __assert_mu(mu)
    vu = gedim.make_nd_array(u, numPoints, np.double)
    return (np.exp(mu[1] * vu) - 1).ctypes.data


def newton_nonlinear_forcing_derivative(numPoints, points, u, u_x, u_y, mu=None):
    vecu_x = gedim.make_nd_array(u_x, numPoints, np.double)
    vecu_y = gedim.make_nd_array(u_y, numPoints, np.double)
    values_nl_d_f = np.zeros((2, numPoints), order='F')
    values_nl_d_f[0, :] = vecu_x
    values_nl_d_f[1, :] = vecu_y
    return values_nl_d_f.ctypes.data


def zeros(numPoints, points, mu=None):
    return np.zeros(numPoints, order='F').ctypes.data


def ones(numPoints, points, mu=None):
    return np.ones(numPoints, order='F').ctypes.data


def ones_derivative(numPoints, points, mu=None):
    return np.ones((2, numPoints), order='F').ctypes.data
