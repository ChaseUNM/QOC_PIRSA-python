import numpy as np

from util import to_complex, to_real
from objectives import trace_infidelity, energy_term_deriv
from time_stepper import (
    evolve_IMR,
    adjoint_step,
    backward_IMR_step,
)
from Hamiltonian_operators import H_deriv_param_0


def finite_differences(
    t0, T, steps, U0, ω, ω_r, beta, omega, epsilon, U_target
):
    """
    Compute a forward finite-difference approximation of the
    terminal trace-infidelity gradient.

    Note: This computes only the terminal infidelity gradient,
    not the energy contribution.
    """
    beta = np.asarray(beta, dtype=float)
    N_params = len(beta)
    grad = np.zeros(N_params)

    U_end, _ = evolve_IMR(
        t0, T, steps, U0, ω, ω_r, beta, omega
    )
    U_end_complex = to_complex(U_end)
    obj_1 = trace_infidelity(U_end_complex, U_target)

    for i in range(N_params):
        beta_2 = beta.copy()
        beta_2[i] += epsilon

        U_end_2, _ = evolve_IMR(
            t0, T, steps, U0, ω, ω_r, beta_2, omega
        )
        U_end_2_complex = to_complex(U_end_2)

        obj_2 = trace_infidelity(U_end_2_complex, U_target)
        grad[i] = (obj_2 - obj_1) / epsilon

    return grad


def accumulate_gradient(
    t0, T, steps, U0, ω, ω_r, beta, omega,
    U_target, gamma_terminal, gamma_energy
):
    """
    Accumulate the terminal-objective gradient using the discrete
    adjoint method, then add the control-energy gradient.

    beta_idx uses Python's 0-based indexing.
    """
    beta = np.asarray(beta, dtype=float)
    N_params = len(beta)

    grad = np.zeros(N_params)
    energy_grad = np.zeros(N_params)

    # Forward evolution.
    U_end, _ = evolve_IMR(
        t0, T, steps, U0, ω, ω_r, beta, omega
    )

    h = (T - t0) / steps
    U_end_complex = to_complex(U_end)

    # Terminal adjoint condition.
    L_T_complex = (
        -0.5 * np.trace(U_target.conj().T @ U_end_complex)
        * U_target
    )
    L_T = to_real(L_T_complex)

    # Energy-gradient contribution.
    for k in range(N_params):
        energy_grad[k] = (
            gamma_energy
            * energy_term_deriv(
                beta, omega, t0, T, steps, k
            )
        )

    # Backward evolution and gradient accumulation.
    t = T

    for i in range(steps - 1, -1, -1):
        t -= h

        L_m = adjoint_step(
            L_T,
            h,
            ω,
            ω_r,
            beta,
            omega,
            t,
            np.zeros((4, 2)),
        )

        y_m = backward_IMR_step(
            U_end, h, ω, ω_r, beta, omega, t
        )

        for k in range(N_params):
            H_deriv = H_deriv_param_0(
                k, omega, t + h / 2
            )

            b = H_deriv @ (y_m + U_end)

            # NumPy dot product of the flattened real-valued matrices.
            grad[k] += gamma_terminal * np.sum(
                b * (L_m + L_T)
            )

        L_T = L_m
        U_end = y_m

    grad *= h / 4.0

    return grad + energy_grad


def gradient(
    beta,
    *,
    t0, T, steps, U0, ω, ω_r, omega,
    U_target, gamma_terminal, gamma_energy
):
    """
    Return the objective gradient as a new NumPy array.
    """
    return accumulate_gradient(
        t0, T, steps, U0, ω, ω_r, beta, omega,
        U_target, gamma_terminal, gamma_energy
    )


def gradient_inplace(
    grad,
    beta,
    *,
    t0, T, steps, U0, ω, ω_r, omega,
    U_target, gamma_terminal, gamma_energy
):
    """
    Fill a preallocated gradient array in place.

    Useful for optimization libraries that expect the gradient
    callback to write into grad.
    """
    grad[:] = accumulate_gradient(
        t0, T, steps, U0, ω, ω_r, beta, omega,
        U_target, gamma_terminal, gamma_energy
    )