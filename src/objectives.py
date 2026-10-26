import numpy as np

from Hamiltonian_operators import spectral_control
from util import to_complex
from time_stepper import evolve_IMR

def trace_infidelity(U, U_target):
    """
    Compute trace infidelity between two 2x2 matrices.
    """
    overlap = np.trace(U.conj().T @ U_target)
    return 1.0 - 0.25 * np.abs(overlap) ** 2


def energy_term(beta, omega, t0, T, steps):
    """
    Compute control energy using the trapezoidal rule.

    Assumes spectral_control(beta, omega, t) returns (u, v).
    """
    t_range = np.linspace(t0, T, steps + 1)
    h = (T - t0) / steps

    energy_sum = 0.0

    for i in range(steps):
        u1, v1 = spectral_control(beta, omega, t_range[i])
        u2, v2 = spectral_control(beta, omega, t_range[i + 1])

        energy_sum += (
            abs(u1 + 1j * v1) ** 2
            + abs(u2 + 1j * v2) ** 2
        )

    return energy_sum * h / 2.0


def energy_term_deriv(beta, omega, t0, T, steps, beta_idx):
    """
    Compute the derivative of control energy with respect to beta[beta_idx].

    Parameters
    ----------
    beta_idx : int
        Zero-based index into beta.

        Indices 0, ..., len(omega)-1 correspond to cosine coefficients.
        Indices len(omega), ..., 2*len(omega)-1 correspond to sine coefficients.
    """
    N_c = len(omega)

    if not 0 <= beta_idx < 2 * N_c:
        raise ValueError(
            f"beta_idx must be between 0 and {2 * N_c - 1}, "
            f"got {beta_idx}"
        )

    t_range = np.linspace(t0, T, steps + 1)
    h = (T - t0) / steps

    deriv_energy_sum = 0.0

    for i in range(steps):
        t1 = t_range[i]
        t2 = t_range[i + 1]

        u1, v1 = spectral_control(beta, omega, t1)
        u2, v2 = spectral_control(beta, omega, t2)

        if beta_idx < N_c:
            # Derivative with respect to a cosine coefficient.
            w = omega[beta_idx]

            du1 = np.cos(w * t1)
            du2 = np.cos(w * t2)

            dv1 = np.sin(w * t1)
            dv2 = np.sin(w * t2)

        else:
            # Derivative with respect to a sine coefficient.
            w = omega[beta_idx - N_c]

            du1 = -np.sin(w * t1)
            du2 = -np.sin(w * t2)

            dv1 = np.cos(w * t1)
            dv2 = np.cos(w * t2)

        deriv_energy_sum += 2.0 * (
            du1 * u1
            + du2 * u2
            + dv1 * v1
            + dv2 * v2
        )

    return deriv_energy_sum * h / 2.0


def leakage():
    """Placeholder for a future leakage objective."""
    raise NotImplementedError("Leakage objective has not been implemented.")


def deriv_leakage(U, W):
    """Placeholder for a future leakage derivative."""
    raise NotImplementedError("Leakage derivative has not been implemented.")


def objective(
    beta,
    *,
    t0,
    T,
    steps,
    U0,
    ω,
    ω_r,
    omega,
    U_target,
    gamma_terminal,
    gamma_energy,
):
    """
    Evaluate the quantum-control objective.

    Assumes evolve_IMR(...) returns (U_end, history), and that
    to_complex(...) converts the real-valued representation into
    a complex-valued matrix.
    """
    U_end, _ = evolve_IMR(
        t0, T, steps, U0, ω, ω_r, beta, omega
    )

    U_end_complex = to_complex(U_end)

    infidelity = trace_infidelity(U_end_complex, U_target)
    energy = energy_term(beta, omega, t0, T, steps)

    value = (
        gamma_terminal * infidelity
        + gamma_energy * energy
    )

    if value < -1e-10 or value > 1.0 + 1e-10:
        print("beta:", beta)
        print("infidelity:", infidelity)
        print("energy:", energy)
        print("gamma_terminal:", gamma_terminal)
        print("gamma_energy:", gamma_energy)
        print("objective value:", value)

        raise ValueError("Objective outside expected range")

    return value