"""
Module for population management in the evolutionary algorithm.
"""

import os
import logging
import numpy as np
import time
import json
import copy
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from src.evolution.individual import Individual
from src.evolution.crossover import n_parent_crossover
from src.evolution.mutation import mutate_genome


class Population:
    """
    Manages a population of individuals for evolutionary training.
    
    Handles initialisation, selection, crossover, mutation, and evolution.
    """
    
    def __init__(self, config: Dict):
        """
        Initialise the population.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        
        # Initialise evolution parameters
        self.population_size = config['evolution']['population_size']
        self.mutation_rate = config['evolution']['mutation_rate']
        self.survivors_count = config['evolution']['survivors_count']
        self.offspring_count = config['evolution']['offspring_count']
        self.crossover_parents = config['evolution']['crossover_parents']
        self.convergence_threshold = config['evolution']['convergence_threshold']
        self.max_generations = config['evolution']['max_generations']
        
        # Initialise population
        self.individuals = [Individual(config) for _ in range(self.population_size)]
        self.current_generation = 0
        self.best_individual = None
        
        # Evolution history
        self.history = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'best_map': [],
            'best_inference': [],
        }
    
    def train_generation(self) -> None:
        """
        Train all individuals in the current generation, one at a time.
        """
        self.logger.info(f"Training generation {self.current_generation}")
        
        output_dir = os.path.join(self.config['paths']['output_dir'], f"gen_{self.current_generation}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Train each individual sequentially
        for i, individual in enumerate(self.individuals):
            self.logger.info(f"Training individual {i+1}/{len(self.individuals)}")
            individual.train(output_dir)
        
        # Update best individual and metrics
        self._update_metrics()
    
    def _update_metrics(self) -> None:
        """Update metrics and best individual after training."""
        # Filter out failed individuals
        trained_individuals = [ind for ind in self.individuals if ind.is_trained and not ind.training_failed]
        
        if not trained_individuals:
            self.logger.warning("No individuals trained successfully in this generation")
            return
            
        # Sort by fitness in descending order
        trained_individuals.sort(key=lambda ind: ind.fitness if ind.fitness is not None else 0, reverse=True)
        
        # Update best individual
        self.best_individual = trained_individuals[0]
        
        # Calculate average fitness
        avg_fitness = np.mean([ind.fitness for ind in trained_individuals if ind.fitness is not None])
        
        # Update history
        self.history['generations'].append(self.current_generation)
        self.history['best_fitness'].append(self.best_individual.fitness)
        self.history['avg_fitness'].append(avg_fitness)
        self.history['best_map'].append(self.best_individual.map_score)
        self.history['best_inference'].append(self.best_individual.inference_time)
        
        self.logger.info(f"Generation {self.current_generation} metrics updated:")
        self.logger.info(f"  Best fitness: {self.best_individual.fitness:.6f}")
        self.logger.info(f"  Average fitness: {avg_fitness:.6f}")
    
    def evolve(self) -> None:
        """
        Evolve the population by performing selection, crossover, and mutation.
        """
        self.logger.info(f"Evolving population for generation {self.current_generation + 1}")
        
        # Selection - identify survivors and candidates for reproduction
        survivors, parents = self._selection()
        
        # Create offspring through crossover and mutation
        offspring = self._reproduction(parents)
        
        # Assemble new population: survivors + offspring
        new_population = survivors + offspring
        
        # Increment generation counter
        self.current_generation += 1
        
        # Update population
        self.individuals = new_population
        
        self.logger.info(f"Population evolved to generation {self.current_generation}")
        self.logger.info(f"  Survivors: {len(survivors)}")
        self.logger.info(f"  Offspring: {len(offspring)}")
    
    def _selection(self) -> Tuple[List[Individual], List[Individual]]:
        """
        Select individuals to survive and reproduce.
        
        Returns:
            Tuple of (survivors, parents)
        """
        # Filter out failed individuals
        valid_individuals = [ind for ind in self.individuals if ind.is_trained and not ind.training_failed]
        
        if len(valid_individuals) == 0:
            self.logger.warning("No valid individuals in population, generating new random population")
            return [], []
        
        # Sort individuals by fitness in descending order
        valid_individuals.sort(key=lambda ind: ind.fitness if ind.fitness is not None else 0, reverse=True)
        
        # Select survivors (top N individuals)
        num_survivors = min(self.survivors_count, len(valid_individuals))
        survivors = valid_individuals[:num_survivors]
        
        # Select parents for reproduction (top crossover_parents individuals)
        num_parents = min(self.crossover_parents, len(valid_individuals))
        parents = valid_individuals[:num_parents]
        
        self.logger.info(f"Selection: {num_survivors} survivors, {num_parents} parents")
        
        # Copy individuals to prevent modifying originals
        survivors = [copy.deepcopy(ind) for ind in survivors]
        parents = [copy.deepcopy(ind) for ind in parents]
        
        return survivors, parents
    
    def _reproduction(self, parents: List[Individual]) -> List[Individual]:
        """
        Create offspring through crossover and mutation.
        
        Args:
            parents: List of parent individuals
            
        Returns:
            List of offspring individuals
        """
        offspring = []
        
        # If no parents, return empty list
        if not parents:
            self.logger.warning("No parents available for reproduction")
            return []
            
        # Get parent genomes
        parent_genomes = [parent.genome for parent in parents]
        
        # Create offspring up to desired count
        for _ in range(self.offspring_count):
            # N-parent crossover
            child_genome = n_parent_crossover(parent_genomes)
            
            # Mutation
            child_genome = mutate_genome(child_genome, self.mutation_rate, self.config)
            
            # Create new individual with the child genome
            child = Individual(self.config, genome=child_genome, generation=self.current_generation + 1)
            offspring.append(child)
        
        self.logger.info(f"Reproduction: created {len(offspring)} offspring")
        return offspring
    
    def has_converged(self) -> bool:
        """
        Check if the population has converged.
        
        Convergence is defined as the top 3 individuals having fitness values
        within the convergence threshold (e.g., 1%) of each other.
        
        Returns:
            True if converged, False otherwise
        """
        # Need at least 3 valid individuals to check convergence
        valid_individuals = [ind for ind in self.individuals 
                            if ind.is_trained and not ind.training_failed and ind.fitness is not None]
        
        if len(valid_individuals) < 3:
            return False
            
        # Sort by fitness
        valid_individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # Get top 3 fitness values
        top_fitness = [ind.fitness for ind in valid_individuals[:3]]
        
        # Check if all are within threshold of the best
        best_fitness = top_fitness[0]
        
        for fitness in top_fitness[1:]:
            relative_diff = abs(best_fitness - fitness) / best_fitness
            if relative_diff > self.convergence_threshold:
                return False
                
        self.logger.info("Population has converged - top 3 individuals within convergence threshold")
        return True
    
    def get_best_individual(self) -> Optional[Individual]:
        """
        Get the best individual in the population.
        
        Returns:
            Best individual, or None if no valid individuals
        """
        valid_individuals = [ind for ind in self.individuals 
                           if ind.is_trained and not ind.training_failed and ind.fitness is not None]
        
        if not valid_individuals:
            return None
            
        return max(valid_individuals, key=lambda ind: ind.fitness)
    
    def get_average_fitness(self) -> float:
        """
        Calculate the average fitness of the population.
        
        Returns:
            Average fitness, or 0 if no valid individuals
        """
        valid_individuals = [ind for ind in self.individuals 
                           if ind.is_trained and not ind.training_failed and ind.fitness is not None]
        
        if not valid_individuals:
            return 0.0
            
        return np.mean([ind.fitness for ind in valid_individuals])
    
    def save_statistics(self, output_dir: str) -> None:
        """
        Save population statistics to files.
        
        Args:
            output_dir: Directory to save statistics
        """
        try:
            stats_dir = os.path.join(output_dir, "statistics")
            os.makedirs(stats_dir, exist_ok=True)
            
            # Save history as JSON
            history_path = os.path.join(stats_dir, "evolution_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            # Save current population state
            population_path = os.path.join(stats_dir, "final_population.json")
            population_data = {
                "generation": self.current_generation,
                "individuals": [ind.to_dict() for ind in self.individuals],
                "best_individual": self.best_individual.to_dict() if self.best_individual else None,
            }
            
            with open(population_path, 'w') as f:
                json.dump(population_data, f, indent=2)
                
            self.logger.info(f"Statistics saved to {stats_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving statistics: {str(e)}")
            
    def to_dict(self) -> Dict:
        """
        Convert the population to a dictionary for serialisation.
        
        Returns:
            Dictionary representation of the population
        """
        return {
            "current_generation": self.current_generation,
            "individuals": [ind.to_dict() for ind in self.individuals],
            "history": self.history,
            "best_individual": self.best_individual.to_dict() if self.best_individual else None,
        }
        
    @classmethod
    def from_dict(cls, data: Dict, config: Dict) -> 'Population':
        """
        Create a population from a dictionary representation.
        
        Args:
            data: Dictionary representation
            config: Configuration dictionary
            
        Returns:
            New Population instance
        """
        population = cls(config)
        population.current_generation = data["current_generation"]
        population.history = data["history"]
        
        # Recreate individuals
        population.individuals = [
            Individual.from_dict(ind_data, config)
            for ind_data in data["individuals"]
        ]
        
        # Set best individual
        if data["best_individual"]:
            best_ind_data = data["best_individual"]
            population.best_individual = Individual.from_dict(best_ind_data, config)
        
        return population
