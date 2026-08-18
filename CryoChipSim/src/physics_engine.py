"""
src/physics_engine.py
=====================

Owner:
    Monisha - Domain Physics Lead

Branch:
    feature/physics-math

Purpose:
    Analytical physics engine for the quantum chiplet project.

This module maps electrical/thermal/interconnect parameters to:

    1. Cryo-CMOS power dissipation
    2. Interconnect latency
    3. Qubit T2* dephasing time
    4. Qubit T1 relaxation time
    5. Estimated gate execution duration

IMPORTANT:
    This module contains ONLY analytical physics calculations.
    It does not generate datasets, train ML models, or perform
    pulse optimization.

    Power, latency, and T2* use the REQUIRED project equations below.
    T1 and gate duration are PROJECT ASSUMPTIONS pending confirmation
    from the Quantum Lead (see notes on params_to_relaxation).

Inputs:
    N       = number of control channels
    f       = clock frequency in MHz
    V       = supply voltage in V
    T       = operating temperature in K
    L_cm    = interconnect length in cm

Outputs:
    Power       = mW
    Latency     = ns
    T1          = us
    T2*         = us
    Gate time   = ns
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# Internal validation helpers
# ============================================================================

def _as_float_array(value, name: str) -> np.ndarray:
    """Convert an input to a NumPy floating-point array."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def _validate_non_negative(value, name: str) -> np.ndarray:
    """Validate that a numerical input is >= 0."""
    array = _as_float_array(value, name)
    if np.any(array < 0):
        raise ValueError(f"{name} must be >= 0.")
    return array


def _validate_positive(value, name: str) -> np.ndarray:
    """Validate that a numerical input is > 0."""
    array = _as_float_array(value, name)
    if np.any(array <= 0):
        raise ValueError(f"{name} must be > 0.")
    return array


def _return_scalar_if_scalar(value):
    """Return a normal Python float for scalar input; arrays pass through."""
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return value


# ============================================================================
# 1. Cryo-CMOS Power Model
# ============================================================================

def power_mw(N, f, V, T):
    """
    Calculate Cryo-CMOS thermal/power load in mW.

    Required project equation:

        P_mW = N * 0.005 * (f / 1000) * sqrt(T / 300)

    V is accepted (it's a specified project input) but intentionally
    unused, since the supplied equation has no voltage term. Do NOT
    add a V-dependent term unless the Quantum Lead explicitly approves
    a revised model.
    """
    N = _validate_non_negative(N, "N")
    f = _validate_non_negative(f, "f")
    V = _validate_non_negative(V, "V")
    T = _validate_positive(T, "T")

    _ = V  # unused by design

    power = N * 0.005 * (f / 1000.0) * np.sqrt(T / 300.0)
    return _return_scalar_if_scalar(power)


# ============================================================================
# 2. Interconnect / Trace Latency Model
# ============================================================================

def latency_ns(L_cm, noise_std=0.0, rng=None):
    """
    Calculate interconnect latency in ns.

    Required project equation:

        Latency_ns = L_cm * 0.05 + noise

    noise_std defaults to 0.0 so the analytical ground truth stays
    deterministic unless noise is explicitly requested.
    """
    L_cm = _validate_non_negative(L_cm, "L_cm")

    noise_std = float(noise_std)
    if not np.isfinite(noise_std):
        raise ValueError("noise_std must be finite.")
    if noise_std < 0:
        raise ValueError("noise_std must be >= 0.")

    base_latency = L_cm * 0.05

    if noise_std == 0.0:
        return _return_scalar_if_scalar(base_latency)

    generator = rng if rng is not None else np.random.default_rng()
    noise = generator.normal(loc=0.0, scale=noise_std, size=base_latency.shape)

    latency = np.maximum(0.0, base_latency + noise)
    return _return_scalar_if_scalar(latency)


# ============================================================================
# 3. Physical-to-Quantum Mapping
# ============================================================================

def params_to_relaxation(temperature_K, latency_ns_value, power_mw_value):
    """
    Convert physical metrics into quantum-control metrics.

    Required T2* project equation:

        T2*_us = max(1.0, 100.0 - (Latency_ns * 0.5 + Power_mW * 0.2))

    T1_us and t_gate_ns are NOT specified by the task, so they use
    clearly-labeled PROJECT ASSUMPTIONS below. Replace these if the
    Quantum Lead provides approved T1 / gate-duration models.
    """
    temperature_K = _validate_positive(temperature_K, "temperature_K")
    latency_ns_value = _validate_non_negative(latency_ns_value, "latency_ns")
    power_mw_value = _validate_non_negative(power_mw_value, "power_mw")

    temperature_K, latency_ns_value, power_mw_value = np.broadcast_arrays(
        temperature_K, latency_ns_value, power_mw_value,
    )

    # Specified T2* model
    t2_star_us = np.maximum(
        1.0,
        100.0 - (latency_ns_value * 0.5 + power_mw_value * 0.2),
    )

    # PROJECT ASSUMPTION: T1 model
    #   thermal_derating = clip(1 - T/1000, 0.5, 1.0)
    #   T1 = 2 * T2* * thermal_derating
    thermal_derating = np.clip(1.0 - (temperature_K / 1000.0), 0.5, 1.0)
    t1_us = 2.0 * t2_star_us * thermal_derating

    # PROJECT ASSUMPTION: Gate duration
    #   t_gate = latency + 10 ns (placeholder overhead)
    t_gate_ns = latency_ns_value + 10.0

    return {
        "T1_us": _return_scalar_if_scalar(t1_us),
        "T2_star_us": _return_scalar_if_scalar(t2_star_us),
        "t_gate_ns": _return_scalar_if_scalar(t_gate_ns),
    }


# ============================================================================
# 4. Complete Physics Evaluation Helper
# ============================================================================

def evaluate_physics(N, f, V, T, L_cm, noise_std=0.0, rng=None):
    """Run the complete analytical physics pipeline (power -> latency -> relaxation)."""
    power = power_mw(N, f, V, T)
    latency = latency_ns(L_cm, noise_std=noise_std, rng=rng)
    relaxation = params_to_relaxation(
        temperature_K=T,
        latency_ns_value=latency,
        power_mw_value=power,
    )

    return {
        "power_mw": power,
        "latency_ns": latency,
        **relaxation,
    }


# ============================================================================
# 5. Built-in Validation Tests
# ============================================================================

def run_self_tests():
    """Run basic analytical and boundary-condition tests. Returns True if all pass."""
    print("\nRunning physics engine self-tests...\n")

    # Test 1: Power equation
    N, f, V, T = 4, 500.0, 0.6, 4.0
    expected_power = N * 0.005 * (f / 1000.0) * np.sqrt(T / 300.0)
    calculated_power = power_mw(N, f, V, T)
    assert np.isclose(calculated_power, expected_power, rtol=1e-12, atol=1e-12)
    print("PASS: Power equation")

    # Test 2: Latency equation
    L_cm = 0.1
    expected_latency = 0.1 * 0.05
    calculated_latency = latency_ns(L_cm)
    assert np.isclose(calculated_latency, expected_latency, rtol=1e-12, atol=1e-12)
    print("PASS: Latency equation")

    # Test 3: T2* equation
    relaxation = params_to_relaxation(
        temperature_K=T,
        latency_ns_value=calculated_latency,
        power_mw_value=calculated_power,
    )
    expected_t2 = max(1.0, 100.0 - (calculated_latency * 0.5 + calculated_power * 0.2))
    assert np.isclose(relaxation["T2_star_us"], expected_t2, rtol=1e-12, atol=1e-12)
    print("PASS: T2* equation")

    # Test 4: T2* lower bound
    extreme_relaxation = params_to_relaxation(
        temperature_K=4.0, latency_ns_value=10000.0, power_mw_value=10000.0,
    )
    assert extreme_relaxation["T2_star_us"] >= 1.0
    assert np.isclose(extreme_relaxation["T2_star_us"], 1.0)
    print("PASS: T2* lower bound >= 1.0 us")

    # Test 5: Invalid temperature
    try:
        power_mw(4, 500, 0.6, 0)
        raise AssertionError("T=0 was accepted.")
    except ValueError:
        pass
    print("PASS: Invalid temperature rejected")

    # Test 6: Invalid negative trace length
    try:
        latency_ns(-1.0)
        raise AssertionError("negative L was accepted.")
    except ValueError:
        pass
    print("PASS: Negative trace length rejected")

    # Test 7: Invalid negative frequency
    try:
        power_mw(4, -500, 0.6, 4)
        raise AssertionError("negative f was accepted.")
    except ValueError:
        pass
    print("PASS: Negative frequency rejected")

    # Test 8: Complete pipeline
    result = evaluate_physics(N=4, f=500.0, V=0.6, T=4.0, L_cm=0.1)
    required_keys = {"power_mw", "latency_ns", "T1_us", "T2_star_us", "t_gate_ns"}
    assert required_keys.issubset(result.keys())
    print("PASS: Complete physics pipeline")

    # Test 9: Array inputs
    array_result = evaluate_physics(
        N=np.array([2, 4, 8]),
        f=np.array([250.0, 500.0, 1000.0]),
        V=np.array([0.5, 0.6, 0.8]),
        T=np.array([4.0, 4.0, 10.0]),
        L_cm=np.array([0.1, 0.5, 1.0]),
    )
    assert np.asarray(array_result["power_mw"]).shape == (3,)
    assert np.asarray(array_result["latency_ns"]).shape == (3,)
    assert np.asarray(array_result["T2_star_us"]).shape == (3,)
    print("PASS: Array/batch inputs")

    print("\nALL PHYSICS ENGINE TESTS PASSED.\n")
    return True


# ============================================================================
# 6. Command-Line Demonstration
# ============================================================================

def main():
    """Run a small demonstration followed by the self-tests."""
    print("=" * 70)
    print("Quantum Chiplet Analytical Physics Engine")
    print("=" * 70)

    N, f, V, T, L_cm = 4, 500.0, 0.6, 4.0, 0.1

    result = evaluate_physics(N=N, f=f, V=V, T=T, L_cm=L_cm)

    print("\nExample operating point:")
    print(f"  Channels       : {N}")
    print(f"  Frequency      : {f:.2f} MHz")
    print(f"  Voltage        : {V:.2f} V")
    print(f"  Temperature    : {T:.2f} K")
    print(f"  Trace length   : {L_cm:.4f} cm")

    print("\nAnalytical outputs:")
    print(f"  Power          : {float(result['power_mw']):.6f} mW")
    print(f"  Latency        : {float(result['latency_ns']):.6f} ns")
    print(f"  T1             : {float(result['T1_us']):.6f} us")
    print(f"  T2*            : {float(result['T2_star_us']):.6f} us")
    print(f"  Gate duration  : {float(result['t_gate_ns']):.6f} ns")

    print("\n" + "-" * 70)
    run_self_tests()


if __name__ == "__main__":
    main()
    
