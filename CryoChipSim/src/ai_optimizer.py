"""
src/ai_optimizer.py
====================
Owner: Kesav - ML Lead

Implements optimize_pulse(), which searches for input parameters
(f, V) that minimize a combined power+latency objective, using the
physics engine directly (ground truth) rather than the surrogate --
scipy.optimize needs a differentiable/continuous callable, and the
physics engine is fast enough to call directly for a handful of free
variables.
"""

import numpy as np
from scipy.optimize import minimize

from src.physics_engine import power_mw, latency_ns

def _objective(x, N, T, L_cm, power_weight, latency_weight):
    f, V = x
    power = power_mw(N, f, V, T)
    latency = latency_ns(L_cm)
    return power_weight * power + latency_weight * latency


def optimize_pulse(
    N: float,
    T: float,
    L_cm: float,
    f_bounds=(10.0, 2000.0),
    v_bounds=(0.1, 1.2),
    power_weight: float = 1.0,
    latency_weight: float = 1.0,
    initial_guess=None,
):
    """
    Find (f, V) that minimizes power_weight*power + latency_weight*latency,
    for fixed N, T, L_cm.

    Returns a dict with the optimal f, V, resulting power/latency, and
    whether the optimizer converged.
    """
    if initial_guess is None:
        initial_guess = [
            (f_bounds[0] + f_bounds[1]) / 2,
            (v_bounds[0] + v_bounds[1]) / 2,
        ]

        result = minimize(
        _objective,
        x0=initial_guess,
        args=(N, T, L_cm, power_weight, latency_weight),
        bounds=[f_bounds, v_bounds],
        method="L-BFGS-B",
        options={"gtol": 1e-12, "ftol": 1e-15},
    )

    f_opt, V_opt = result.x
    power_opt = power_mw(N, f_opt, V_opt, T)
    latency_opt = latency_ns(L_cm)

    return {
        "f_opt": float(f_opt),
        "V_opt": float(V_opt),
        "power_mw": float(power_opt),
        "latency_ns": float(latency_opt),
        "success": bool(result.success),
        "objective_value": float(result.fun),
    } 
