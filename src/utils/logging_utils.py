"""
Logging utilities for the Evolutionary Training Manager.
"""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: str = 'logs', log_level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"evo_trainer_{timestamp}.log"
    
    # Configure logging
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log setup completion
    logging.info(f"Logging configured. Log file: {log_file}")


class EvoTrainerLogger:
    """
    Custom logger for the Evolutionary Training Manager.
    
    Provides methods for logging training progress, model metrics,
    and generation statistics with appropriate formatting.
    """
    
    def __init__(self, name: str, log_level: int = logging.INFO):
        """
        Initialise the custom logger.
        
        Args:
            name: Logger name
            log_level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
    
    def log_generation_start(self, generation: int, max_generations: int) -> None:
        """
        Log the start of a new generation.
        
        Args:
            generation: Current generation number
            max_generations: Maximum number of generations
        """
        separator = "=" * 20
        self.logger.info(f"{separator} Generation {generation}/{max_generations} {separator}")
    
    def log_training_start(self, individual_id: int, genome: dict) -> None:
        """
        Log the start of individual training.
        
        Args:
            individual_id: Individual identifier
            genome: Individual's genome
        """
        self.logger.info(f"Training individual {individual_id}")
        self.logger.debug(f"Genome: {genome}")
    
    def log_training_metrics(self, map_score: float, inference_time: float, fitness: float) -> None:
        """
        Log training metrics for an individual.
        
        Args:
            map_score: mAP@0.5:0.95 score
            inference_time: Inference time in seconds
            fitness: Calculated fitness score
        """
        self.logger.info(f"Training metrics - mAP: {map_score:.4f}, Inference time: {inference_time:.4f}s, Fitness: {fitness:.4f}")
    
    def log_generation_stats(
        self, 
        generation: int, 
        best_fitness: float, 
        avg_fitness: float, 
        survival_count: int
    ) -> None:
        """
        Log statistics for a generation.
        
        Args:
            generation: Generation number
            best_fitness: Best fitness in the generation
            avg_fitness: Average fitness in the generation
            survival_count: Number of surviving individuals
        """
        self.logger.info(f"Generation {generation} complete")
        self.logger.info(f"Best fitness: {best_fitness:.4f}, Average fitness: {avg_fitness:.4f}")
        self.logger.info(f"Survivors: {survival_count}")
    
    def log_evolution_complete(
        self, 
        generations: int, 
        best_fitness: float, 
        convergence_reached: bool
    ) -> None:
        """
        Log completion of the evolutionary process.
        
        Args:
            generations: Total generations run
            best_fitness: Final best fitness
            convergence_reached: Whether convergence was reached
        """
        reason = "convergence reached" if convergence_reached else "maximum generations reached"
        separator = "=" * 20
        self.logger.info(f"{separator} Evolution complete {separator}")
        self.logger.info(f"Completed after {generations} generations ({reason})")
        self.logger.info(f"Best fitness achieved: {best_fitness:.4f}")
    
    def log_error(self, message: str, exc_info: bool = False) -> None:
        """
        Log an error message.
        
        Args:
            message: Error message
            exc_info: Whether to include exception info
        """
        self.logger.error(message, exc_info=exc_info)
