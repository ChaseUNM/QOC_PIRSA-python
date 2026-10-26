import sys
from pathlib import Path

# Add src/ to Python's module search path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import secrets

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from util import (
    to_complex,
    to_complex_history,
    plot_pulse,
    plot_population,
)
from objectives import trace_infidelity, energy_term
from gradient_eval import accumulate_gradient
from time_stepper import evolve_IMR


# Physical parameters
omega_transition = 4.5
omega_r = 4.5

# Initial condition matrix (real-valued)
U0 = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [0.0, 0.0],
])

# Target matrix: Pauli X gate
U_target = np.array([
    [0.0, 1.0],
    [1.0, 0.0],
], dtype=complex)


# Number of basis functions
N_c = 10

# Maximum frequencies to test
max_frequency_list = [1.0]

# Number of random trials per frequency configuration
n_trials = 1
N_max = len(max_frequency_list)

# Time-stepping conditions
t0 = 0.0
T = 20.0
steps = 1000

gamma_terminal = 1.0
gamma_energy = 0.0

# Optimization settings
use_bounds = False
opt_iter = 100

max_parameter = 0.01
initial_parameter = 0.01


# Store results for each frequency configuration
frequency_configuration_all = [None] * N_max
objective_all = [None] * N_max
seed_all = [None] * N_max
coeffs_all = [None] * N_max
pulse_all = [None] * N_max
plot_all = [None] * N_max


# debug settings 
debug = False

# Run trials for each maximum frequency
for i, max_frequency in enumerate(max_frequency_list):

    objective_values = np.zeros(n_trials)
    frequencies = [None] * n_trials
    seeds = [None] * n_trials
    coeffs_opt = [None] * n_trials
    pulse_plots = []
    population_plots = []

    for j in range(n_trials):

        # Generate a seed and a reproducible random number generator
        seed = secrets.randbits(64)
        seeds[j] = seed
        rng = np.random.default_rng(seed)

        # Generate initial coefficients and frequencies
        beta = rng.uniform(
            -initial_parameter,
            initial_parameter,
            size=2 * N_c,
        )

        omega = rng.uniform(
            -max_frequency,
            max_frequency,
            size=N_c,
        )

        frequencies[j] = omega.copy()
        beta0 = beta.copy()

        # Optional box constraints
        if use_bounds:
            bounds = [
                (-max_parameter, max_parameter)
                for _ in range(len(beta0))
            ]
        else:
            bounds = None

        # Objective for this trial
        def objective_trial(beta):
            U_end, _ = evolve_IMR(
                t0, T, steps, U0,
                omega_transition, omega_r, beta, omega,
            )
            U_complex = to_complex(U_end)

            terminal = trace_infidelity(U_complex, U_target)
            energy = energy_term(beta, omega, t0, T, steps)

            return (
                gamma_terminal * terminal
                + gamma_energy * energy
            )

        # Gradient for this trial
        def gradient_trial(beta):
            return accumulate_gradient(
                t0, T, steps, U0,
                omega_transition, omega_r, beta, omega,
                U_target, gamma_terminal, gamma_energy,
            )

        # Print progress every 20 iterations
        iteration = 0

        def callback(beta):
            nonlocal_placeholder = None
            # The callback is called once per optimizer iteration.
            # The counter is updated below using a mutable container.
            iteration_counter[0] += 1

            if iteration_counter[0] % 5 == 0:
                print(
                    f"Iteration {iteration_counter[0]}: "
                    f"objective = {objective_trial(beta):.8e}"
                )

        iteration_counter = [0]

        if debug:
        
            # Debug: compare analytic gradient with central finite differences
            g_analytic = gradient_trial(beta0)
            g_fd = np.zeros_like(beta0)
            epsilon = 1e-6

            for k in range(len(beta0)):
                beta_plus = beta0.copy()
                beta_minus = beta0.copy()

                beta_plus[k] += epsilon
                beta_minus[k] -= epsilon

                g_fd[k] = (
                    objective_trial(beta_plus) - objective_trial(beta_minus)
                ) / (2 * epsilon)

            print("\nGradient check:")
            print("Analytic gradient:        ", g_analytic)
            print("Finite-difference gradient:", g_fd)
            print("Absolute error norm:      ", np.linalg.norm(g_analytic - g_fd))
            print("Finite-difference norm:   ", np.linalg.norm(g_fd))
            print("Analytic gradient norm:   ", np.linalg.norm(g_analytic))
        
        # Optimize using SciPy's L-BFGS-B implementation.
        # Without bounds, it behaves as an unconstrained L-BFGS optimizer.
        result = minimize(
            objective_trial,
            beta0,
            method="L-BFGS-B",
            jac=gradient_trial,
            bounds=bounds,
            callback=callback,
            options={
                "maxiter": opt_iter,
                "gtol": 1e-8,
            },
        )

        beta_opt = result.x.copy()

        # Evaluate the original objective at the optimized coefficients
        J_opt = objective_trial(beta_opt)

        # Simulate the optimized pulse
        U_end, U_history = evolve_IMR(
            t0, T, steps, U0,
            omega_transition, omega_r, beta_opt, omega,
        )

        U_history_complex = to_complex_history(U_history)
        U_end_complex = to_complex(U_end)

        print(f"\nFrequency configuration {i}, trial {j}")
        print(f"Seed:              {seed}")
        print(f"Initial objective: {objective_trial(beta0):.8e}")
        print(f"Final objective:   {J_opt:.8e}")
        print(
            "Terminal objective: ",
            gamma_terminal
            * trace_infidelity(U_end_complex, U_target),
        )
        print(
            "Energy objective:   ",
            gamma_energy * energy_term(
                beta_opt, omega, t0, T, steps
            ),
        )
        print(f"Converged:          {result.success}")
        print(f"Optimizer message:  {result.message}")
        print(f"Optimized beta:     {beta_opt}")

        # Generate plots
        pulse = plot_pulse(
            beta_opt, omega, np.linspace(t0, T, steps)
        )
        pop_plots = plot_population(
            U_history_complex, t0, T
        )

        pulse_plots.append(pulse)
        population_plots.append(pop_plots)

        # Save trial results
        objective_values[j] = J_opt
        coeffs_opt[j] = beta_opt.copy()

        print("Done!")

    # Save results for this maximum frequency
    objective_all[i] = objective_values
    frequency_configuration_all[i] = frequencies
    seed_all[i] = seeds
    coeffs_all[i] = coeffs_opt
    pulse_all[i] = pulse_plots
    plot_all[i] = population_plots


# Find the best and worst trials for each maximum frequency
best_results = [None] * N_max
worst_results = [None] * N_max

for i, max_frequency in enumerate(max_frequency_list):
    objectives = objective_all[i]

    j_best = int(np.argmin(objectives))
    j_worst = int(np.argmax(objectives))

    best_results[i] = {
        "objective": objectives[j_best],
        "trial": j_best,
        "max_frequency": max_frequency,
        "frequencies": frequency_configuration_all[i][j_best],
        "seed": seed_all[i][j_best],
        "coefficients": coeffs_all[i][j_best],
        "pulse_plot": pulse_all[i][j_best],
        "population_plot": plot_all[i][j_best],
    }

    worst_results[i] = {
        "objective": objectives[j_worst],
        "trial": j_worst,
        "max_frequency": max_frequency,
        "frequencies": frequency_configuration_all[i][j_worst],
        "seed": seed_all[i][j_worst],
        "coefficients": coeffs_all[i][j_worst],
        "pulse_plot": pulse_all[i][j_worst],
        "population_plot": plot_all[i][j_worst],
    }

    print(f"\nMax frequency = {max_frequency}")
    print("  Best objective:", best_results[i]["objective"])
    print("  Best trial (0-based):", best_results[i]["trial"])
    print("  Best seed:    ", best_results[i]["seed"])

    print("  Worst objective:", worst_results[i]["objective"])
    print("  Worst trial (0-based):", worst_results[i]["trial"])
    print("  Worst seed:    ", worst_results[i]["seed"])

fig, ax = best_results[0]["pulse_plot"]
plt.show()

fig, ax = best_results[0]["population_plot"]
plt.show()