 import numpy as np
import pytest

from src.ai_optimizer import optimize_pulse
from src.physics_engine import power_mw, latency_ns


def test_optimize_pulse_returns_expected_keys():
    result = optimize_pulse(N=8, T=4.0, L_cm=0.5)
    for key in ("f_opt", "V_opt", "power_mw", "latency_ns", "success", "objective_value"):
        assert key in result

def test_optimize_pulse_converges():
    result = optimize_pulse(N=8, T=4.0, L_cm=0.5)
    assert result["success"] is True

def test_optimize_pulse_respects_bounds():
    f_bounds = (50.0, 500.0)
    v_bounds = (0.2, 0.8)
    result = optimize_pulse(N=8, T=4.0, L_cm=0.5, f_bounds=f_bounds, v_bounds=v_bounds)
    assert f_bounds[0] - 1e-6 <= result["f_opt"] <= f_bounds[1] + 1e-6
    assert v_bounds[0] - 1e-6 <= result["V_opt"] <= v_bounds[1] + 1e-6

def test_optimize_pulse_minimizes_power_weighted_objective():
    result = optimize_pulse(N=8, T=4.0, L_cm=0.5, power_weight=1.0, latency_weight=0.0)
    # minimizing power alone should push f toward its lower bound
    assert result["f_opt"] == pytest.approx(10.0, abs=1.0)

def test_optimize_pulse_outputs_are_non_negative():
    result = optimize_pulse(N=16, T=10.0, L_cm=1.0)
    assert result["power_mw"] >= 0.0
    assert result["latency_ns"] >= 0.0
