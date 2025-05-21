"""
Quick test for evolutionary algorithm: trains single-epoch models for 5 generations.
Complies with all requirements in Notes.md (British English, no pretrained, no absolute paths, etc).
"""
import os
import shutil
import tempfile
import logging
import pytest
from src.utils.config_manager import ConfigManager
from src.evolution.population import Population


def test_evolutionary_quick_run():
    # Use a temporary output directory to avoid polluting workspace
    temp_dir = tempfile.mkdtemp(prefix="evo_quicktest_")
    try:
        # Load config and override for quick test
        config_path = "config.yaml"
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()

        # Override output/checkpoints/logs to temp, set single epoch, 5 generations
        config['paths']['output_dir'] = os.path.join(temp_dir, "output")
        config['paths']['checkpoints_dir'] = os.path.join(temp_dir, "checkpoints")
        config['paths']['logs_dir'] = os.path.join(temp_dir, "logs")
        os.makedirs(config['paths']['output_dir'], exist_ok=True)
        os.makedirs(config['paths']['checkpoints_dir'], exist_ok=True)
        os.makedirs(config['paths']['logs_dir'], exist_ok=True)
        config['evolution']['max_generations'] = 5
        config['hyperparameter_ranges']['epochs'] = {'min': 1, 'max': 2}  # Always 1 epoch
        # Ensure batch size is valid
        if 'batch_size' in config['hyperparameter_ranges']:
            config['hyperparameter_ranges']['batch_size']['min'] = max(1, config['hyperparameter_ranges']['batch_size'].get('min', 1))
            config['hyperparameter_ranges']['batch_size']['max'] = max(1, config['hyperparameter_ranges']['batch_size'].get('max', 1))

        # Ensure no pretrained model is used
        if 'model' in config and 'pretrained' in config['model']:
            config['model']['pretrained'] = False

        # Set up population
        population = Population(config)
        for gen in range(5):
            population.train_generation()
            if population.has_converged():
                break
            if gen < 4:
                population.evolve()
        # Check at least one individual trained
        best = population.get_best_individual()
        assert best is not None and best.is_trained and not best.training_failed
        print(f"Quick test complete. Best fitness: {best.fitness}, mAP: {best.map_score}, inference: {best.inference_time}")
    finally:
        shutil.rmtree(temp_dir)
