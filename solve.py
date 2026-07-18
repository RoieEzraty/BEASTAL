from __future__ import annotations
import numpy as np

from numpy.typing import NDArray
from typing import Tuple, List, Optional, Union
from typing import TYPE_CHECKING

import matrix_functions

if TYPE_CHECKING:
    from Network_Structure import Network_Structure


# ==================================
# functions that solve flow
# ==================================


# @lru_cache(maxsize=20)
def solve_flow(Strctr: "Network_Structure",
               CstrTuple: Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]],
               K_vec: NDArray[np.float_], nonlinear_law: str = 'linear', nonlinear_exponent: float = 1.0,
               max_iter: int = 200, tol: float = 1e-10, relaxation: float = 0.5,
               regularization: float = 1e-12) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Solves for the pressure at nodes and flow at edges, given Lagrangian etc.
    For linear edges the solve is direct. For nonlinear monotone edges, the function performs
    an iteratively reweighted linear solve using a secant conductance.
    2nd part of State.solve_flow_given_modality, Comes after functions.setup_constraints_given_pin.

    input:
    Strctr: "Network_Structure" class instance with the input, intermediate and output nodes
    CstrTuple - Tuple consisting - Cstr_full - 2D array without last column, which is f from Rocks & Katifori 2018
                                               https://www.pnas.org/cgi/doi/10.1073/pnas.1806790116
                                   Cstr -      Cstr_full without last line
                                   f    -      constraint vector (from Rocks and Katifori 2018)1D np.arrays sized NEdges
                                               such that EI[i] is node connected to EJ[i] at certain edge
    K_vec               - [NE] 2D cubic np.array of conductances (inverse of reistances) or,
                          for nonlinear laws, prefactors in the constitutive relation
    nonlinear_law       - edge constitutive law, default='linear'
    nonlinear_exponent  - exponent for ``nonlinear_law='power_law'``. For example, 0.5 gives
                          ``u ~ sign(dp) * sqrt(|dp|)``, corresponding to ``dp ~ u|u|``.

    output:
    p - hydrostatic pressure, 1D np.array sized NNodes
    u - velocity through each edge 1D np.array sized len(EI)
    """
    if nonlinear_law == 'linear':
        p = _solve_pressure_linear(Strctr, CstrTuple, K_vec)
    else:
        p = _solve_pressure_nonlinear(Strctr, CstrTuple, K_vec, nonlinear_law=nonlinear_law,
                                      nonlinear_exponent=nonlinear_exponent, max_iter=max_iter, tol=tol,
                                      relaxation=relaxation, regularization=regularization)

    delta_p = edge_pressure_drop(Strctr, p)
    u = matrix_functions.edge_flow_from_pressure_drop(delta_p, K_vec, law=nonlinear_law,
                                                      exponent=nonlinear_exponent)
    p, u = round_small(p, u)
    return p, u


def _solve_pressure_linear(Strctr: "Network_Structure",
                           CstrTuple: Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]],
                           K_vec: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Solve the augmented linear system for node pressures.
    """
    Cstr: NDArray[np.float_] = CstrTuple[1]
    f: NDArray[np.float_] = CstrTuple[2]
    K_mat: NDArray[np.float_] = np.diag(K_vec)
    _, L_bar = matrix_functions.buildL(Strctr.DM, K_mat, Cstr, Strctr.NN)
    return np.linalg.solve(L_bar, f)


def _solve_pressure_nonlinear(Strctr: "Network_Structure",
                              CstrTuple: Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]],
                              K_vec: NDArray[np.float_], nonlinear_law: str = 'power_law',
                              nonlinear_exponent: float = 1.0, max_iter: int = 200, tol: float = 1e-10,
                              relaxation: float = 0.5,
                              regularization: float = 1e-12) -> NDArray[np.float_]:
    """
    Solve a nonlinear monotone network by iteratively updating an effective diagonal conductance.
    """
    p = _solve_pressure_linear(Strctr, CstrTuple, K_vec)

    for _ in range(max_iter):
        delta_p = edge_pressure_drop(Strctr, p)
        K_eff = matrix_functions.effective_conductance_from_pressure_drop(
            delta_p, K_vec, law=nonlinear_law, exponent=nonlinear_exponent,
            regularization=regularization
        )
        p_candidate = _solve_pressure_linear(Strctr, CstrTuple, K_eff)
        p_next = relaxation * p_candidate + (1 - relaxation) * p
        if np.max(np.abs(p_next - p)) < tol * max(1.0, np.max(np.abs(p_next))):
            return p_next
        p = p_next
    return p


def edge_pressure_drop(Strctr: "Network_Structure", p: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Edge pressure drops ``delta_p = p_i - p_j`` in the edge orientation defined by ``EI, EJ``.
    """
    return (p[Strctr.EI] - p[Strctr.EJ]).ravel()


def round_small(p: NDArray[np.float_], u: NDArray[np.float_], roundto: float = 10**-10) -> Tuple[NDArray[np.float_],
                                                                                                 NDArray[np.float_]]:
    """
    round_small rounds values of u and p that are close to 0 to get rid of rounding problems
    """
    p[abs(p) < roundto] = 0  # Correct for very low pressures
    u[abs(u) < roundto] = 0  # Correct for very low velocities
    return p, u


def dot_triple(X: Union[NDArray[np.float_], NDArray[np.int_]],
               Y: Union[NDArray[np.float_], NDArray[np.int_]],
               Z: Union[NDArray[np.float_], NDArray[np.int_]]) -> NDArray[np.float_]:
    """
    Matrix triple product X @ Y @ Z.

    Parameters
    ----------
    X, Y, Z : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return np.dot(X, np.dot(Y, Z))
