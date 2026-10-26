import numpy as np
import matplotlib.pyplot as plt

from Hamiltonian_operators import spectral_control

def to_complex(A):
    """
    Convert a real-valued (4, 2) matrix to a complex-valued (2, 2) matrix.

    The first two rows contain the real parts.
    The last two rows contain the negative imaginary parts.
    """
    A = np.asarray(A)
    assert A.shape == (4, 2)

    return A[:2, :] - 1j * A[2:4, :]


def to_real(A):
    """
    Convert a complex-valued (2, 2) matrix to a real-valued (4, 2) matrix.

    The first two rows contain the real parts.
    The last two rows contain the negative imaginary parts.
    """
    A = np.asarray(A)
    assert A.shape == (2, 2)

    A_real = np.zeros((4, 2), dtype=float)
    A_real[:2, :] = np.real(A)
    A_real[2:4, :] = -np.imag(A)

    return A_real


def to_complex_history(A):
    """
    Convert a real-valued history array of shape (4, 2, steps)
    to a complex-valued history array of shape (2, 2, steps).
    """
    A = np.asarray(A)
    assert A.ndim == 3
    assert A.shape[:2] == (4, 2)

    steps = A.shape[2]
    A_complex = np.zeros((2, 2, steps), dtype=np.complex128)

    for i in range(steps):
        A_complex[:, :, i] = to_complex(A[:, :, i])

    return A_complex


def plot_pulse(beta, omega, t_range):
    """
    Plot the real and imaginary components of the control pulse.

    Assumes spectral_control(beta, omega, t) returns (u, v).
    """
    t_range = np.asarray(t_range)

    real_ts = np.zeros(len(t_range))
    imag_ts = np.zeros(len(t_range))

    for i, t in enumerate(t_range):
        u, v = spectral_control(beta, omega, t)
        real_ts[i] = u
        imag_ts[i] = v

    fig, ax = plt.subplots()
    ax.plot(t_range, real_ts, label="p(t)")
    ax.plot(t_range, imag_ts, label="q(t)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Control amplitude")
    ax.legend()

    return fig, ax


def plot_population(state_history, t0, T):
    """
    Plot the populations of states |0> and |1> for each initial state.

    Parameters
    ----------
    state_history : ndarray, shape (2, N_init, steps)
        Complex-valued state history.
    t0, T : float
        Initial and final times.

    Returns
    -------
    list
        A list of (fig, ax) tuples, one for each initial state.
    """
    state_history = np.asarray(state_history)

    pop_array = np.abs(state_history) ** 2
    steps = state_history.shape[2]
    t_range = np.linspace(t0, T, steps)
    N_init = state_history.shape[1]

    plot_arr = []

    for i in range(N_init):
        pop1 = pop_array[0, i, :]
        pop2 = pop_array[1, i, :]

        fig, ax = plt.subplots()
        ax.plot(t_range, pop1, label="|0>")
        ax.plot(t_range, pop2, label="|1>")
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        ax.legend(loc="upper center", ncol=2)
        ax.set_title(f"Initial state {i + 1}")

        plot_arr.append((fig, ax))

    return plot_arr


def get_phase(z):
    """
    Return the magnitude and phase of a complex number.

    Returns
    -------
    r : float
        Magnitude of z.
    theta : float
        Phase of z in radians, in [-pi, pi].
    """
    z = complex(z)
    r = abs(z)
    theta = np.angle(z)

    return r, theta


def get_global_phase(A, target_mat):
    """
    Estimate the global phase from the matrix overlap and compute
    the squared norm difference.

    Returns
    -------
    theta_global : float
        Phase of tr(target_mat^H @ A).
    norm_diff : float
        Squared Frobenius norm of the difference between A and
        the phase-adjusted target.
    fidelity_expression : complex
        Preserves the third expression from the original Julia code.
    """
    A = np.asarray(A, dtype=complex)
    target_mat = np.asarray(target_mat, dtype=complex)

    # Estimate the global phase using the matrix overlap.
    overlap = np.trace(target_mat.conj().T @ A)
    theta_global = np.angle(overlap)

    # Compare against the phase-adjusted target.
    A_aligned = np.exp(1j * theta_global) * target_mat
    norm_diff = np.linalg.norm(A - A_aligned) ** 2

    fidelity_expression = 1 - 0.25 * np.trace(
        A.conj().T @ target_mat
    )

    return theta_global, norm_diff, fidelity_expression