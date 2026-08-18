%%writefile src/physics_engine.py
import numpy as np

def compute_cryo_power_and_latency(temp_k: float, channels: int, freq_mhz: float, trace_len_cm: float) -> tuple[float, float]:
    static_power_per_channel = 0.005  # mW
    dynamic_scaling = (freq_mhz / 1000.0)
    temp_factor = np.sqrt(temp_k / 300.0)
    power_mw = (channels * static_power_per_channel) * dynamic_scaling * temp_factor
    
    propagation_delay = trace_len_cm * 0.05  # ns
    temp_latency_penalty = (300.0 / max(temp_k, 0.1)) * 0.01
    latency_ns = propagation_delay + temp_latency_penalty
    return float(power_mw), float(latency_ns)

def params_to_relaxation(temp_k: float, latency_ns: float, power_mw: float) -> tuple[float, float, float]:
    t1_base = 120.0  # us
    t2_base = 80.0   # us
    thermal_noise_factor = (temp_k / 4.0) ** 1.5
    latency_dephasing = (latency_ns * 0.2)
    power_heating = (power_mw * 0.15)
    
    t1_us = max(1.0, t1_base / (0.5 + 0.5 * thermal_noise_factor))
    t2_degraded = t2_base / (thermal_noise_factor + latency_dephasing + power_heating)
    t2_us = max(0.5, min(t2_degraded, 2.0 * t1_us))
    
    base_gate_time_ns = 50.0 + (latency_ns * 2.0)
    gate_time_us = base_gate_time_ns / 1000.0
    return float(t1_us), float(t2_us), float(gate_time_us) 
