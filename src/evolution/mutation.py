"""
Module for implementing mutation operations in evolutionary algorithm.
"""

import numpy as np
from typing import Dict, Any


def mutate_genome(genome: Dict[str, Any], mutation_rate: float, config: Dict) -> Dict[str, Any]:
    """
    Mutate genome by randomly changing parameters based on mutation rate.
    
    Args:
        genome: Genome dictionary to mutate
        mutation_rate: Probability of mutating each parameter
        config: Configuration dictionary with parameter ranges
        
    Returns:
        Mutated genome dictionary
    """
    mutated_genome = genome.copy()
    
    # Get parameter ranges from config
    hp_ranges = config.get('hyperparameter_ranges', {})
    arch_ranges = config.get('architecture_params', {})
    param_ranges = {}
    
    # Combine ranges into one dictionary
    for key, value in hp_ranges.items():
        param_ranges[key] = value
    for key, value in arch_ranges.items():
        param_ranges[key] = value
        
    # Process each parameter in the genome
    for key, value in genome.items():
        # Determine if this parameter should be mutated
        if np.random.random() < mutation_rate:
            # Get range for this parameter if it exists
            if key in param_ranges:
                ranges = param_ranges[key]
            elif key in ['depth_multiple', 'width_multiple', 'mosaic', 'mixup'] and 'yolov8' in param_ranges:
                ranges = param_ranges['yolov8'].get(key, {})
            else:
                ranges = {}
                
            # Mutate based on parameter type
            if isinstance(value, (int, np.integer)):
                if ranges:
                    min_val = ranges.get('min', max(1, int(value * 0.5)))
                    max_val = ranges.get('max', int(value * 1.5) + 1)
                    
                    # Either small change or completely new value
                    if np.random.random() < 0.7:  # 70% small change
                        # Change by up to 20% of range
                        range_width = max_val - min_val
                        max_change = max(1, int(range_width * 0.2))
                        change = np.random.randint(-max_change, max_change + 1)
                        new_value = value + change
                    else:  # 30% completely new value
                        new_value = np.random.randint(min_val, max_val + 1)
                    
                    # Ensure value stays within valid range
                    mutated_genome[key] = np.clip(new_value, min_val, max_val)
                else:
                    # Add/subtract a small random integer
                    max_change = max(1, int(value * 0.2)) if value > 0 else 1
                    change = np.random.randint(-max_change, max_change + 1)
                    mutated_genome[key] = max(1, value + change)  # Ensure positive
                
            elif isinstance(value, (float, np.floating)):
                if ranges:
                    min_val = ranges.get('min', max(0.00001, value * 0.5))
                    max_val = ranges.get('max', value * 1.5)
                    
                    # For some parameters like learning rate, use log-uniform distribution
                    if 'learning_rate' in key or 'lr' in key or 'weight_decay' in key:
                        log_min = np.log10(min_val)
                        log_max = np.log10(max_val)
                        mutated_genome[key] = 10 ** np.random.uniform(log_min, log_max)
                    else:
                        # Gaussian mutation with standard deviation of 10% of the range
                        range_width = max_val - min_val
                        sigma = range_width * 0.1
                        new_value = value + np.random.normal(0, sigma)
                        mutated_genome[key] = np.clip(new_value, min_val, max_val)
                else:
                    # Add Gaussian noise (10% of current value)
                    sigma = abs(value) * 0.1 if value != 0 else 0.01
                    mutated_genome[key] = value + np.random.normal(0, sigma)
                    
            elif isinstance(value, bool):
                # Flip boolean with mutation_rate chance
                mutated_genome[key] = not value
    
    return mutated_genome
