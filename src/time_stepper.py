import numpy as np

from Hamiltonian_operators import total_H

def IMR_step(U_init, h, ω, ω_r, beta, omega, t_n):
    """
    One implicit midpoint rule (IMR) step using the k formulation.
    """
    H = total_H(ω, ω_r, beta, omega, t_n + h / 2)

    I = np.eye(H.shape[0], dtype=H.dtype)
    k = np.linalg.solve(I - (h / 2) * H, H @ U_init)

    return U_init + h * k


def IMR_step_v2(U_init, h, ω, ω_r, beta, omega, t_n):
    """
    One implicit midpoint rule (IMR) step using the direct formulation.
    """
    H = total_H(ω, ω_r, beta, omega, t_n + h / 2)

    I = np.eye(H.shape[0], dtype=H.dtype)
    LHS = I - 0.5 * h * H
    RHS = (I + 0.5 * h * H) @ U_init

    U = np.linalg.solve(LHS, RHS)

    return U


def backward_IMR_step(U_T, h, ω, ω_r, beta, omega, t_n):
    """
    One backward IMR step.
    """
    H = total_H(ω, ω_r, beta, omega, t_n + h / 2)

    I = np.eye(H.shape[0], dtype=H.dtype)
    LHS = I + 0.5 * h * H
    RHS = (I - 0.5 * h * H) @ U_T

    U = np.linalg.solve(LHS, RHS)

    return U


def adjoint_step(L_T, h, ω, ω_r, beta, omega, t_n, a):
    """
    One adjoint integration step.
    """
    H = total_H(ω, ω_r, beta, omega, t_n + h / 2).conj().T

    I = np.eye(H.shape[0], dtype=H.dtype)
    LHS = I - 0.5 * h * H
    RHS = h * a + (I + 0.5 * h * H) @ L_T

    L = np.linalg.solve(LHS, RHS)

    return L


def evolve_IMR(t0, T, steps, U0, ω, ω_r, beta, omega):
    """
    Evolve U0 forward in time using the implicit midpoint rule.

    Returns
    -------
    U : ndarray
        Final state.
    state_history : ndarray
        State history with shape (rows, columns, steps + 1).
        state_history[:, :, 0] is the initial state.
    """
    U0 = np.asarray(U0)
    state_history = np.zeros(
        (*U0.shape, steps + 1), dtype=U0.dtype
    )
    state_history[:, :, 0] = U0

    h = (T - t0) / steps
    t = t0

    for i in range(steps):
        U0 = IMR_step_v2(U0, h, ω, ω_r, beta, omega, t)
        state_history[:, :, i + 1] = U0
        t += h

    return U0, state_history


def backward_evolve_IMR(t0, T, steps, U_T, ω, ω_r, beta, omega):
    """
    Evolve U_T backward in time using the backward IMR step.

    Returns
    -------
    U : ndarray
        State at t0.
    state_history : ndarray
        State history ordered chronologically, from t0 to T.
    """
    U_T = np.asarray(U_T)
    state_history = np.zeros(
        (*U_T.shape, steps + 1), dtype=U_T.dtype
    )
    state_history[:, :, -1] = U_T

    h = (T - t0) / steps
    t = T

    for i in range(steps - 1, -1, -1):
        t -= h
        U_T = backward_IMR_step(
            U_T, h, ω, ω_r, beta, omega, t
        )
        state_history[:, :, i] = U_T

    return U_T, state_history


def backward_adjoint(t0, T, steps, L_T, ω, ω_r, beta, omega, a):
    """
    Evolve the adjoint variable backward in time.

    Parameters
    ----------
    a : ndarray or callable
        Forcing term in the adjoint equation. If it varies with time,
        this implementation assumes a constant value over the interval,
        matching the original Julia function's use of a.

    Returns
    -------
    L : ndarray
        Adjoint at t0.
    adjoint_history : ndarray
        Adjoint history ordered chronologically, from t0 to T.
    """
    L_T = np.asarray(L_T)
    adjoint_history = np.zeros(
        (*L_T.shape, steps + 1), dtype=L_T.dtype
    )
    adjoint_history[:, :, -1] = L_T

    h = (T - t0) / steps
    t = T

    for i in range(steps - 1, -1, -1):
        t -= h
        L_T = adjoint_step(
            L_T, h, ω, ω_r, beta, omega, t, a
        )
        adjoint_history[:, :, i] = L_T

    return L_T, adjoint_history