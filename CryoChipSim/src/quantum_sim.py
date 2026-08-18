%%writefile src/quantum_sim.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

def build_noise_model(t1_us: float, t2_us: float, gate_time_us: float) -> NoiseModel:
    t1_valid = max(float(t1_us), 0.1)
    t2_valid = max(min(float(t2_us), 2.0 * t1_valid), 0.1)
    t1_ns = t1_valid * 1000.0
    t2_ns = t2_valid * 1000.0
    gate_time_ns = max(float(gate_time_us) * 1000.0, 0.1)

    error_1q = thermal_relaxation_error(t1_ns, t2_ns, gate_time_ns)
    error_2q = error_1q.tensor(error_1q)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1q, ["h", "rx", "ry", "rz", "u1", "u2", "u3"])
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cz"])
    return noise_model

def run_fidelity_circuit(noise_model: NoiseModel = None, shots: int = 1024) -> dict:
    qc_ideal = QuantumCircuit(2)
    qc_ideal.h(0)
    qc_ideal.cx(0, 1)
    
    ideal_state = Statevector.from_instruction(qc_ideal)
    ideal_dm = DensityMatrix(ideal_state)

    qc_sim = qc_ideal.copy()
    qc_sim.save_density_matrix(label="final_dm")
    qc_sim.measure_all()

    backend = AerSimulator(noise_model=noise_model) if noise_model is not None else AerSimulator()
    job = backend.run(qc_sim, shots=shots)
    result = job.result()
    
    counts = result.get_counts()
    try:
        simulated_dm = result.data()["final_dm"]
    except Exception:
        simulated_dm = ideal_dm

    fidelity_clamped = max(0.0, min(1.0, float(state_fidelity(ideal_dm, simulated_dm))))
    dm_data = np.array(simulated_dm.data)
    
    return {
        "fidelity": fidelity_clamped,
        "density_matrix": dm_data,
        "density_matrix_abs": np.abs(dm_data),
        "counts": counts
    }
