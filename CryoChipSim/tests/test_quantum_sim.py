 import numpy as np
import pytest

from src.quantum_sim import build_noise_model, run_fidelity_circuit


# ---------------------------------------------------------------------------
# build_noise_model
# ---------------------------------------------------------------------------

def test_build_noise_model_returns_noise_model():
    from qiskit_aer.noise import NoiseModel
    nm = build_noise_model(t1_us=50.0, t2_us=30.0, gate_time_us=0.05)
    assert isinstance(nm, NoiseModel)

def test_build_noise_model_handles_t2_greater_than_2t1():
    # Physically, T2 can never exceed 2*T1 -- code should clamp it, not crash
    nm = build_noise_model(t1_us=10.0, t2_us=999.0, gate_time_us=0.05)
    assert nm is not None

def test_build_noise_model_handles_zero_or_negative_t1():
    # Code clamps t1 to a 0.1 floor -- should not raise
    nm = build_noise_model(t1_us=0.0, t2_us=0.0, gate_time_us=0.05)
    assert nm is not None

def test_build_noise_model_handles_negative_gate_time():
    # Code clamps gate_time to a 0.1 ns floor -- should not raise
    nm = build_noise_model(t1_us=50.0, t2_us=30.0, gate_time_us=-5.0)
    assert nm is not None


# ---------------------------------------------------------------------------
# run_fidelity_circuit
# ---------------------------------------------------------------------------

def test_run_fidelity_circuit_ideal_case_high_fidelity():
    # No noise model -> should be very close to fidelity 1.0
    result = run_fidelity_circuit(noise_model=None, shots=256)
    assert result["fidelity"] == pytest.approx(1.0, abs=1e-6)

def test_run_fidelity_circuit_returns_expected_keys():
    result = run_fidelity_circuit(noise_model=None, shots=256)
    for key in ("fidelity", "density_matrix", "density_matrix_abs", "counts"):
        assert key in result

def test_fidelity_is_bounded_between_0_and_1():
    result = run_fidelity_circuit(noise_model=None, shots=256)
    assert 0.0 <= result["fidelity"] <= 1.0

def test_noisy_fidelity_stays_bounded_and_below_or_equal_ideal():
    noise_model = build_noise_model(t1_us=5.0, t2_us=3.0, gate_time_us=0.05)
    noisy_result = run_fidelity_circuit(noise_model=noise_model, shots=256)
    ideal_result = run_fidelity_circuit(noise_model=None, shots=256)

    assert 0.0 <= noisy_result["fidelity"] <= 1.0
    # Noise should not make fidelity better than the ideal case
    assert noisy_result["fidelity"] <= ideal_result["fidelity"] + 1e-9

def test_counts_total_matches_shots():
    shots = 512
    result = run_fidelity_circuit(noise_model=None, shots=shots)
    total_counts = sum(result["counts"].values())
    assert total_counts == shots

def test_density_matrix_shape_is_4x4_for_two_qubits():
    result = run_fidelity_circuit(noise_model=None, shots=256)
    dm = result["density_matrix"]
    assert dm.shape == (4, 4)

def test_density_matrix_abs_is_non_negative():
    result = run_fidelity_circuit(noise_model=None, shots=256)
    assert np.all(result["density_matrix_abs"] >= 0.0)

def test_heavy_noise_degrades_fidelity_noticeably():
    # Very short T1/T2 relative to gate time should push fidelity down significantly
    noise_model = build_noise_model(t1_us=0.01, t2_us=0.01, gate_time_us=1.0)
    result = run_fidelity_circuit(noise_model=noise_model, shots=256)
    assert result["fidelity"] < 0.9
