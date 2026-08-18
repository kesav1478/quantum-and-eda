
import numpy as np
from scipy.optimize import minimize

def negative_fidelity(params, temp, fidelity_model):
    """Objective function that extracts the raw score from the AI prediction array."""
    amp, duration = params
    predicted = fidelity_model.predict([[amp, duration, temp]])
    return -predicted[0]

def run_pulse_optimization(current_temp, fidelity_model):
    """
    Finds the optimal pulse amplitude and duration for a given temperature.
    Takes the loaded fidelity_model as an input argument.
    """
    default_amp, default_duration = 0.5, 100.0
    
    # Run the live optimization loop
    result = minimize(
        negative_fidelity, 
        x0=[default_amp, default_duration], 
        args=(current_temp, fidelity_model), 
        bounds=[(0.1, 1.0), (10.0, 200.0)]
    )
    
    best_amp, best_duration = result.x
    optimized_fidelity = -result.fun
    
    return {
        "best_amp": float(best_amp),
        "best_duration": float(best_duration),
        "optimized_fidelity": float(optimized_fidelity)
    }
