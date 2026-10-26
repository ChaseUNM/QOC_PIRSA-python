import numpy as np


def drift_H(ω, ω_r):
    """
    Construct the drift Hamiltonian.

    Returns
    -------
    ReH_d, ImH_d : ndarray
        Real and imaginary components of the drift Hamiltonian.
    """
    a = np.array([
        [0.0, 1.0],
        [0.0, 0.0],
    ])

    ReH_d = (ω - ω_r) * (a.conj().T @ a)
    ImH_d = np.zeros_like(ReH_d)

    return ReH_d, ImH_d


def spectral_control(beta, omega, t):
    """
    Compute the real and imaginary components of the spectral control.

    beta[:N_c] contains the cosine coefficients.
    beta[N_c:] contains the sine coefficients.
    """
    beta = np.asarray(beta)
    omega = np.asarray(omega)

    N_c = len(omega)

    beta_real = beta[:N_c]
    beta_imag = beta[N_c:2 * N_c]

    u = np.sum(
        beta_real * np.cos(omega * t)
        - beta_imag * np.sin(omega * t)
    )

    v = np.sum(
        beta_real * np.sin(omega * t)
        + beta_imag * np.cos(omega * t)
    )

    return u, v


def total_H(ω, ω_r, beta, omega, t):
    """
    Construct the total real-valued Hamiltonian representation.

    Returns an ndarray of shape (4, 4), with block structure:
        [ ImH  -ReH ]
        [ ReH   ImH ]
    """
    a = np.array([
        [0.0, 1.0],
        [0.0, 0.0],
    ])

    ReH_d, ImH_d = drift_H(ω, ω_r)

    real_control, imag_control = spectral_control(
        beta, omega, t
    )

    ReH = ReH_d + real_control * (a + a.conj().T)
    ImH = imag_control * (a - a.conj().T)

    H_total = np.block([
        [ImH, -ReH],
        [ReH,  ImH],
    ])

    return H_total


def H_deriv_param_0(beta_idx, omega, t):
    """
    Original alternate formulation of the Hamiltonian derivative.

    beta_idx uses 0-based Python indexing:
        0, ..., N_c - 1     : cosine coefficients
        N_c, ..., 2*N_c - 1 : sine coefficients

    This preserves the block formulas in the original Julia function.
    """
    omega = np.asarray(omega)
    N_c = len(omega)

    if not 0 <= beta_idx < 2 * N_c:
        raise ValueError(
            f"beta_idx must be between 0 and {2 * N_c - 1}"
        )

    a = np.array([
        [0.0, 1.0],
        [0.0, 0.0],
    ])

    K = a - a.conj().T
    S = a + a.conj().T

    if beta_idx < N_c:
        j = beta_idx

        A = np.sin(omega[j] * t) * K
        B = np.cos(omega[j] * t) * S

        return np.block([
            [A, -B],
            [B,  A],
        ])

    else:
        j = beta_idx - N_c

        A = np.sin(omega[j] * t) * S
        B = np.cos(omega[j] * t) * K

        return np.block([
            [B,  A],
            [-A, B],
        ])


def H_deriv_param(beta_idx, omega, t):
    """
    Compute the derivative of the real-valued Hamiltonian
    with respect to beta[beta_idx].

    Uses 0-based indexing.
    """
    omega = np.asarray(omega)
    N_c = len(omega)

    if not 0 <= beta_idx < 2 * N_c:
        raise ValueError(
            f"beta_idx must be between 0 and {2 * N_c - 1}"
        )

    a = np.array([
        [0.0, 1.0],
        [0.0, 0.0],
    ])

    K = a - a.conj().T
    S = a + a.conj().T

    if beta_idx < N_c:
        j = beta_idx
        du = np.cos(omega[j] * t)
        dv = np.sin(omega[j] * t)
    else:
        j = beta_idx - N_c
        du = -np.sin(omega[j] * t)
        dv = np.cos(omega[j] * t)

    dH_R = du * S
    dH_I = dv * K

    return np.block([
        [dH_I, -dH_R],
        [dH_R,  dH_I],
    ])