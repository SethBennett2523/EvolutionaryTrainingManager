"""
Module for implementing crossover operations in evolutionary algorithm.
"""

import numpy as np
from typing import List, Dict, Any


def n_parent_crossover(parent_genomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Implement N-parent crossover by combining parameters from multiple parents.
    
    For continuous parameters (like learning rate), we use weighted averaging.
    For discrete parameters (like batch size), we select from one parent or use majority voting.
    
    Args:
        parent_genomes: List of parent genome dictionaries
        
    Returns:
        Child genome dictionary
    """
    if not parent_genomes:
        raise ValueError("Parent genomes list cannot be empty")
    
    # If only one parent, return a copy of its genome
    if len(parent_genomes) == 1:
        return parent_genomes[0].copy()
    
    # Create empty child genome
    child_genome = {}
    
    # Get all keys from all parents
    all_keys = set()
    for genome in parent_genomes:
        all_keys.update(genome.keys())
    
    # For each parameter
    for key in all_keys:
        # Collect values from parents who have this parameter
        values = [genome[key] for genome in parent_genomes if key in genome]
        
        if not values:
            continue
        
        # Handle based on parameter type
        if isinstance(values[0], (int, np.integer)):
            # Integer parameter (like batch size) - pick from one parent with weighted preference for better parents
            weights = np.linspace(1.0, 0.5, len(values))  # Higher weight to earlier parents (assumed to be better)
            weights = weights / np.sum(weights)  # Normalize
            child_genome[key] = int(np.random.choice(values, p=weights))
            
        elif isinstance(values[0], (float, np.floating)):
            # Continuous parameter (like learning rate) - weighted average
            weights = np.linspace(1.0, 0.5, len(values))  # Higher weight to earlier parents
            weights = weights / np.sum(weights)
            child_genome[key] = float(np.sum([w * v for w, v in zip(weights, values)]))
            
        elif isinstance(values[0], bool):
            # For boolean parameters, use majority voting
            votes = sum(1 if v else 0 for v in values)
            child_genome[key] = votes > len(values) / 2
            
        else:
            # For any other type, select from one parent
            child_genome[key] = values[0]  # Take from best parent
    
    return child_genome
