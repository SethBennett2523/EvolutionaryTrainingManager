#!/usr/bin/env python
"""
Evolutionary Training Manager

This module provides the main entry point for the evolutionary training process
for YOLOv8 models using hardware-aware optimisation for both NVIDIA and AMD GPUs.
"""

import os
import sys
import logging
import argparse
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

# Add TrainingAutomation to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainingAutomation"))

# Import from TrainingAutomation
from TrainingAutomation.src.utils.file_io import FileIO

# Import local modules
from src.utils.config_manager import ConfigManager
from src.utils.logging_utils import setup_logging
from src.evolution.population import Population


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Evolutionary Training Manager for YOLOv8 models"
    )
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help="Path to configuration file")
    parser.add_argument('--resume', action='store_true',
                        help="Resume training from the last checkpoint")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug logging")
    parser.add_argument('--output-dir', type=str,
                        help="Override the output directory specified in the config")
    parser.add_argument('--generations', type=int, default=None,
                        help="Maximum number of generations to run")
    parser.add_argument('--population', type=int, default=None,
                        help="Population size (overrides config)")
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'rocm', 'cpu', 'auto'],
                        help="Device to use (overrides auto-detection)")
    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose output")
    parser.add_argument('--log-dir', type=str, default='logs',
                        help="Directory for log files")
    parser.add_argument('--quick-test', action='store_true',
                        help="Run a quick test: 5 generations, single-epoch models, temporary output.")
    
    return parser.parse_args()


def prepare_environment(args: argparse.Namespace) -> Dict:
    """
    Prepare the environment for evolutionary training.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary with environment setup information
    """
    # Setup file I/O
    file_io = FileIO()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # Override configuration with command-line arguments
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    
    if args.generations:
        config['evolution']['max_generations'] = args.generations
        
    if args.population:
        config['evolution']['population_size'] = args.population
        
    if args.device:
        config['hardware']['device'] = args.device
    
    # Quick test mode: override config for 5 generations, 1 epoch, temp dirs, and smaller population/selection
    if getattr(args, 'quick_test', False):
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="evo_quicktest_")
        config['paths']['output_dir'] = os.path.join(temp_dir, "output")
        config['paths']['checkpoints_dir'] = os.path.join(temp_dir, "checkpoints")
        config['paths']['logs_dir'] = os.path.join(temp_dir, "logs")
        os.makedirs(config['paths']['output_dir'], exist_ok=True)
        os.makedirs(config['paths']['checkpoints_dir'], exist_ok=True)
        os.makedirs(config['paths']['logs_dir'], exist_ok=True)
        config['evolution']['max_generations'] = 5
        config['hyperparameter_ranges']['epochs'] = {'min': 1, 'max': 2}
        # Halve population and selection group sizes for quick test
        pop_size = max(2, config['evolution']['population_size'] // 2)
        config['evolution']['population_size'] = pop_size
        config['evolution']['crossover_parents'] = max(2, config['evolution']['crossover_parents'] // 2)
        config['evolution']['survivors_count'] = max(1, config['evolution']['survivors_count'] // 2)
        config['evolution']['offspring_count'] = max(1, config['evolution']['offspring_count'] // 2)
        if 'batch_size' in config['hyperparameter_ranges']:
            config['hyperparameter_ranges']['batch_size']['min'] = max(1, config['hyperparameter_ranges']['batch_size'].get('min', 1))
            config['hyperparameter_ranges']['batch_size']['max'] = max(1, config['hyperparameter_ranges']['batch_size'].get('max', 1))
        if 'model' in config and 'pretrained' in config['model']:
            config['model']['pretrained'] = False
        print(f"[Quick Test] Output: {config['paths']['output_dir']}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['paths']['output_dir'], f"run_{timestamp}")
    checkpoints_dir = os.path.join(config['paths']['checkpoints_dir'], f"run_{timestamp}")
    logs_dir = os.path.join(config['paths']['logs_dir'], f"run_{timestamp}")
    
    # Override with timestamped paths unless resuming
    if not args.resume:
        config['paths']['output_dir'] = output_dir
        config['paths']['checkpoints_dir'] = checkpoints_dir
        config['paths']['logs_dir'] = logs_dir
    
    # Create directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['checkpoints_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)    # Save the configuration being used
    config_path = os.path.join(config['paths']['output_dir'], "config_used.yaml")
    file_io.save_yaml(config, config_path)
    
    # Return environment information
    return {
        'config': config,
        'file_io': file_io,
        'resume': args.resume,
        'timestamp': timestamp,
    }


def validate_setup(env: Dict) -> None:
    """
    Validate the project setup and print a summary of the configuration.
    
    Args:
        env: Environment dictionary with configuration
    """
    logger = logging.getLogger(__name__)
    config = env['config']
    
    logger.info("Validating project setup")
    
    # Verify paths
    data_yaml_path = config['paths']['data_yaml']
    if not os.path.exists(data_yaml_path):
        logger.warning(f"Data YAML file not found: {data_yaml_path}")
    else:
        logger.info(f"Data YAML file found: {data_yaml_path}")
    
    # Verify hardware configuration
    device = config['hardware']['device']
    memory_threshold = config['hardware']['memory_threshold']
    logger.info(f"Hardware configuration: device={device}, memory_threshold={memory_threshold}")
    
    # Check GPU availability
    from TrainingAutomation.src.training.hardware_manager import HardwareManager
    hw_manager = HardwareManager()
    if hw_manager.has_cuda:
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {hw_manager.vram_total / (1024**3):.2f} GB")
    elif hw_manager.has_rocm:
        logger.info(f"ROCm available")
        logger.info(f"VRAM: {hw_manager.vram_total / (1024**3):.2f} GB")
    else:
        logger.warning("No GPU detected, training will be slow")
    
    # Summarize evolution parameters
    logger.info("Evolution parameters:")
    logger.info(f"  Population size: {config['evolution']['population_size']}")
    logger.info(f"  Mutation rate: {config['evolution']['mutation_rate']}")
    logger.info(f"  Survivors count: {config['evolution']['survivors_count']}")
    logger.info(f"  Offspring count: {config['evolution']['offspring_count']}")
    logger.info(f"  Maximum generations: {config['evolution']['max_generations']}")
    
    # Summarize model parameters
    logger.info(f"Base model type: {config['model']['base_type']}")
    
    # Check output directories
    logger.info(f"Output directory: {config['paths']['output_dir']}")
    logger.info(f"Checkpoints directory: {config['paths']['checkpoints_dir']}")
    
    # Log information about hyperparameter ranges
    logger.info("Hyperparameter ranges configured")
    
    return


def run_evolutionary_training(env: Dict) -> None:
    """
    Run the evolutionary training process.
    
    Args:
        env: Environment dictionary with configuration
    """
    logger = logging.getLogger(__name__)
    config = env['config']
    
    from src.training.checkpoint import CheckpointManager
    checkpoint_manager = CheckpointManager(config['paths']['checkpoints_dir'])
    
    logger.info("Starting evolutionary training")
    
    # Initialize or load population
    if env.get("resume", False):
        logger.info("Resuming from checkpoint")
        population = checkpoint_manager.load_latest_checkpoint()
        if population is None:
            logger.warning("No checkpoint found, starting fresh")
            population = Population(config)
    else:
        logger.info("Starting fresh population")
        population = Population(config)
    
    # Track best fitness over generations
    best_fitness_history = []
    
    # Run generations
    max_generations = config['evolution']['max_generations']
    logger.info(f"Running for up to {max_generations} generations")
    
    try:
        for generation in range(population.current_generation, max_generations):
            generation_start_time = time.time()
            logger.info(f"Starting generation {generation + 1}/{max_generations}")
            
            # Train and evaluate this generation
            population.train_generation()
            
            # Calculate statistics
            best_individual = population.get_best_individual()
            if best_individual is not None:
                fitness_str = f"{best_individual.fitness:.6f}" if best_individual.fitness is not None else "N/A"
                map_str = f"{best_individual.map_score:.6f}" if best_individual.map_score is not None else "N/A"
                logger.info(f"  Best fitness: {fitness_str}")
                logger.info(f"  Best mAP: {map_str}")
                logger.info(f"  Best genome: {best_individual.genome}")
            else:
                logger.error("No individuals were successfully trained in this generation. Skipping fitness summary.")
            avg_fitness = population.get_average_fitness()
            avg_fitness_str = f"{avg_fitness:.6f}" if avg_fitness is not None else "N/A"
            # Log results
            logger.info(f"Generation {generation + 1} results:")
            logger.info(f"  Average fitness: {avg_fitness_str}")
            
            # Record best fitness
            best_fitness_history.append(best_individual.fitness if best_individual else float('nan'))
            
            # Save checkpoint
            checkpoint_manager.save_checkpoint(population, generation + 1)
            
            # Check for convergence
            if population.has_converged():
                logger.info("Population has converged - stopping evolution")
                break
            
            # Evolve the population for next generation
            if generation < max_generations - 1:
                logger.info("Evolving population for next generation")
                population.evolve()
            
            generation_time = time.time() - generation_start_time
            logger.info(f"Generation {generation + 1} completed in {generation_time:.2f} seconds")
        
        logger.info("Evolutionary training completed")
        
        # Export the best model from the final population
        best_model = population.get_best_individual()
        export_path = os.path.join(config['paths']['output_dir'], "best_model")
        os.makedirs(export_path, exist_ok=True)
        
        logger.info(f"Exporting best model (fitness: {best_model.fitness:.6f})")
        best_model.export(export_path)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save checkpoint on interrupt
        checkpoint_manager.save_checkpoint(population, population.current_generation)
    
    except Exception as e:
        logger.error(f"Error during evolutionary training: {str(e)}", exc_info=True)
        # Try to save checkpoint on error
        try:
            checkpoint_manager.save_checkpoint(population, population.current_generation)
        except Exception:
            logger.error("Failed to save checkpoint after error", exc_info=True)
        
        raise
    
    logger.info("Evolutionary training process finished")
    
    # Export the best model
    best_model = population.get_best_individual()
    best_model.export()
    
    logger.info(f"Best model exported to {config['paths']['output_dir']}")


def main():
    """
    Main entry point for the application.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(args.log_dir, log_level)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Prepare the environment
        logger.info("Preparing environment")
        env = prepare_environment(args)

        # Quick test: print summary and clean up temp dir at end
        if getattr(args, 'quick_test', False):
            run_evolutionary_training(env)
            best = env['config']
            print("\n[Quick Test Complete]")
            # Print best model info if available
            from src.evolution.population import Population
            pop = Population(env['config'])
            best_ind = pop.get_best_individual()
            if best_ind is not None:
                print(f"Best fitness: {best_ind.fitness}, mAP: {best_ind.map_score}, inference: {best_ind.inference_time}")
            # Clean up temp dir
            import shutil
            temp_dir = os.path.dirname(env['config']['paths']['output_dir'])
            shutil.rmtree(temp_dir, ignore_errors=True)
            return

        # Validate setup
        validate_setup(env)
        
        # Run evolutionary training
        run_evolutionary_training(env)
        
        logger.info("Operation completed successfully")
        
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

