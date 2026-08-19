import numpy as np
import pytest

from src.physics_engine import (
    power_mw,
    latency_ns,
    params_to_relaxation,
    compute_cryo_power_and_latency,
    evaluate_physics,
)


# ---------------------------------------------------------------------------
# power_mw
# ---------------------------------------------------------------------------

def test_power_matches_required_equation():
    N, f, V, T = 4, 500.0, 0.6, 4.0
    expected = N * 0.005 * (f / 1000.0) * np.sqrt(T / 300.0)
    assert power_mw(N, f, V, T) == pytest.approx(expected, rel=1e-12)

def test_power_is_non_negative():
    result = power_mw(N=8, f=1000.0, V=0.7, T=4.0)
    assert result >= 0.0

def test_power_zero_channels_gives_zero_power():
    assert power_mw(N=0, f=500.0, V=0.6, T=4.0) == 0.0

def test_power_rejects_zero_or_negative_temperature():
    with pytest.raises(ValueError):
        power_mw(4, 500.0, 0.6, 0.0)
    with pytest.raises(ValueError):
        power_mw(4, 500.0, 0.6, -1.0)

def test_power_rejects_negative_N_or_f():
    with pytest.raises(ValueError):
        power_mw(-1, 500.0, 0.6, 4.0)
    with pytest.raises(ValueError):
        power_mw(4, -500.0, 0.6, 4.0)

def test_power_voltage_has_no_effect():
    p1 = power_mw(4, 500.0, 0.1, 4.0)
    p2 = power_mw(4, 500.0, 5.0, 4.0)
    assert p1 == p2


# ---------------------------------------------------------------------------
# latency_ns
# ---------------------------------------------------------------------------

def test_latency_matches_required_equation():
    L_cm = 2.5
    assert latency_ns(L_cm) == pytest.approx(L_cm * 0.05, rel=1e-12)

def test_latency_is_non_negative_even_with_noise():
    rng = np.random.default_rng(42)
    result = latency_ns(L_cm=0.01, noise_std=5.0, rng=rng)
    assert np.all(np.asarray(result) >= 0.0)

def test_latency_rejects_negative_length():
    with pytest.raises(ValueError):
        latency_ns(-0.5)

def test_latency_rejects_negative_noise_std():
    with pytest.raises(ValueError):
        latency_ns(1.0, noise_std=-1.0)


# ---------------------------------------------------------------------------
# params_to_relaxation
# ---------------------------------------------------------------------------

def test_t2_star_matches_required_equation():
    T, lat, pw = 4.0, 10.0, 5.0
    result = params_to_relaxation(T, lat, pw)
    expected = max(1.0, 100.0 - (lat * 0.5 + pw * 0.2))
    assert result["T2_star_us"] == pytest.approx(expected, rel=1e-12)

def test_t2_star_lower_bound_enforced():
    result = params_to_relaxation(temperature_K=4.0, latency_ns_value=1e6, power_mw_value=1e6)
    assert result["T2_star_us"] == pytest.approx(1.0)

def test_t1_is_derived_from_t2_star_and_stays_positive():
    result = params_to_relaxation(temperature_K=4.0, latency_ns_value=10.0, power_mw_value=5.0)
    assert result["T1_us"] > 0
    assert result["T1_us"] <= 2 * result["T2_star_us"]

def test_gate_time_equals_latency_plus_overhead():
    result = params_to_relaxation(temperature_K=4.0, latency_ns_value=25.0, power_mw_value=5.0)
    assert result["t_gate_ns"] == pytest.approx(35.0)

def test_relaxation_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        params_to_relaxation(0.0, 10.0, 5.0)


# ---------------------------------------------------------------------------
# compute_cryo_power_and_latency (combined wrapper)
# ---------------------------------------------------------------------------

def test_combined_wrapper_matches_individual_calls():
    N, f, V, T, L_cm = 4, 500.0, 0.6, 4.0, 0.1
    combined = compute_cryo_power_and_latency(N, f, V, T, L_cm)
    assert combined["power_mw"] == pytest.approx(power_mw(N, f, V, T))
    assert combined["latency_ns"] == pytest.approx(latency_ns(L_cm))

def test_combined_wrapper_outputs_are_physically_bounded():
    combined = compute_cryo_power_and_latency(N=64, f=2000.0, V=0.6, T=4.0, L_cm=1.0)
    assert combined["power_mw"] >= 0.0
    assert combined["latency_ns"] >= 0.0


# ---------------------------------------------------------------------------
# Full pipeline / array support
# ---------------------------------------------------------------------------

def test_full_pipeline_keys_present():
    result = evaluate_physics(N=4, f=500.0, V=0.6, T=4.0, L_cm=0.1)
    for key in ("power_mw", "latency_ns", "T1_us", "T2_star_us", "t_gate_ns"):
        assert key in result

def test_full_pipeline_supports_array_inputs():
    result = evaluate_physics(
        N=np.array([2, 4, 8]),
        f=np.array([250.0, 500.0, 1000.0]),
        V=np.array([0.5, 0.6, 0.8]),
        T=np.array([4.0, 4.0, 10.0]),
        L_cm=np.array([0.1, 0.5, 1.0]),
    )
    assert np.asarray(result["power_mw"]).shape == (3,)
    assert np.all(np.asarray(result["power_mw"]) >= 0)
    assert np.all(np.asarray(result["T2_star_us"]) >= 1.0)
