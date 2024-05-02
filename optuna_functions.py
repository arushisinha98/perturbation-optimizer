#import cupy as cp
import numpy as cp
import optuna

from cost_functions import ComputeKLD, ComputeProximity


optuna.logging.set_verbosity(optuna.logging.WARNING)

def Objective(trial, original_array, static_array, target, bounds):
    # Get bounds of perturbation
    min_value, min_mult, max_mult = bounds['min_value'], bounds['min_mult'], bounds['max_mult']
    
    # Create a perturbed array where each element can be scaled by a factor
    perturbation_factors = cp.array([trial.suggest_float(f"scale_{i}", max(min_value, min_mult), max_mult, step=0.1) for i in range(original_array.size)])
    perturbed_array = original_array * perturbation_factors

    # Compute KL divergence
    kl_div = ComputeKLD(perturbed_array, static_array)
    
    # Compute proximity to the target sum
    proximity = ComputeProximity(cp.sum(perturbed_array), target)
    
    # Compute cost of perturbation (linear or exponential)
    linear_cost = cp.sum(perturbation_factors - 1)
    exponential_cost = cp.sum((cp.exp(perturbation_factors-1)-1)/(cp.exp(1)-1))

    # Composite objective: KL divergence + proximity to target + cost of perturbation
    return kl_div + proximity + linear_cost


def Optimize(original_array, static_array, target, bounds, n_trials = 100):
    # Ensure arrays are on GPU
    original_array = cp.asarray(original_array, dtype=cp.float32)
    static_array = cp.asarray(static_array, dtype=cp.float32)
    
    # Optuna study
    study = optuna.create_study()
    study.optimize(lambda trial: Objective(trial, original_array, static_array, target, bounds), n_trials=n_trials)

    # Retrieve the best perturbation factors
    best_factors = cp.array([study.best_trial.params[f"scale_{i}"] for i in range(original_array.size)])

    # Apply the best perturbation
    optimized_array = original_array * best_factors
    
    return optimized_array, study.best_trial.value
