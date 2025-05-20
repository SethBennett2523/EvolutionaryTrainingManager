"""
Unit tests for the population module.
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evolution.population import Population
from src.evolution.individual import Individual


class TestPopulation(unittest.TestCase):
    """Test cases for the Population class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test configuration
        self.config = {
            'paths': {
                'output_dir': os.path.join(self.test_dir, 'output'),
                'checkpoints_dir': os.path.join(self.test_dir, 'checkpoints'),
                'training_automation': './TrainingAutomation',
                'data_yaml': './TrainingAutomation/data/data.yaml'
            },
            'evolution': {
                'population_size': 5,  # Smaller for tests
                'mutation_rate': 0.1,
                'crossover_parents': 4,
                'survivors_count': 3,
                'offspring_count': 2,
                'convergence_threshold': 0.01,
                'max_generations': 3  # Smaller for tests
            },
            'model': {
                'base_type': 'yolov8m'
            },
            'hardware': {
                'device': 'auto',
                'memory_threshold': 0.85
            },
            'hyperparameter_ranges': {
                'learning_rate': {
                    'min': 0.0001,
                    'max': 0.01
                },
                'batch_size': {
                    'min': 2,
                    'max': 16
                },
                'epochs': {
                    'min': 5,
                    'max': 10
                },
                'img_size': {
                    'min': 320,
                    'max': 640,
                    'step': 32
                },
                'yolov8': {
                    'depth_multiple': {
                        'min': 0.33,
                        'max': 1.0
                    },
                    'width_multiple': {
                        'min': 0.50,
                        'max': 1.0
                    },
                    'mosaic': {
                        'min': 0.0,
                        'max': 1.0
                    },
                    'mixup': {
                        'min': 0.0,
                        'max': 0.1
                    }
                }
            }
        }
        
        # Create output directories
        os.makedirs(self.config['paths']['output_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['checkpoints_dir'], exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_population_initialization(self):
        """Test population initialization."""
        pop = Population(self.config)
        
        # Check population size
        self.assertEqual(len(pop.individuals), self.config['evolution']['population_size'])
        
        # Check that each individual has a genome
        for ind in pop.individuals:
            self.assertIsNotNone(ind.genome)
            self.assertIsInstance(ind.genome, dict)
    
    @patch('src.evolution.individual.Individual.train')
    def test_selection(self, mock_train):
        """Test selection of top individuals."""
        # Mock training to return True
        mock_train.return_value = True
        
        # Create a population
        pop = Population(self.config)
        
        # Manually set fitness values for testing
        for i, ind in enumerate(pop.individuals):
            ind.is_trained = True
            ind.training_failed = False
            ind.fitness = i + 1  # Fitness values 1, 2, 3, 4, 5
            ind.map_score = (i + 1) * 0.1  # mAP values 0.1, 0.2, 0.3, 0.4, 0.5
            ind.inference_time = 1.0
        
        # Run selection
        pop._selection()
        
        # Check that we kept the top 3 individuals (as per survivors_count=3)
        self.assertEqual(len(pop.individuals), self.config['evolution']['survivors_count'])
        
        # Check that we kept the highest fitness individuals
        self.assertEqual(pop.individuals[0].fitness, 5)
        self.assertEqual(pop.individuals[1].fitness, 4)
        self.assertEqual(pop.individuals[2].fitness, 3)
    
    @patch('src.evolution.individual.Individual.train')
    def test_create_offspring(self, mock_train):
        """Test creation of offspring."""
        # Mock training to return True
        mock_train.return_value = True
        
        # Create a population
        pop = Population(self.config)
        
        # Manually set up individuals for testing
        pop.individuals = []
        for i in range(3):  # Create 3 individuals (less than population_size)
            ind = Individual(self.config)
            ind.is_trained = True
            ind.training_failed = False
            ind.fitness = i + 1
            ind.map_score = (i + 1) * 0.1
            ind.inference_time = 1.0
            pop.individuals.append(ind)
        
        # Create offspring
        pop._create_offspring()
        
        # Check that population size is back to the configured size
        self.assertEqual(len(pop.individuals), self.config['evolution']['population_size'])
    
    @patch('src.evolution.individual.Individual.train')
    def test_check_convergence(self, mock_train):
        """Test convergence detection."""
        # Mock training to return True
        mock_train.return_value = True
        
        # Create a population
        pop = Population(self.config)
        
        # Case 1: Not converged (top 3 individuals have different fitness)
        pop.individuals = []
        for i in range(3):
            ind = Individual(self.config)
            ind.is_trained = True
            ind.training_failed = False
            ind.fitness = 1.0 + (i * 0.1)  # Fitness values: 1.0, 1.1, 1.2
            pop.individuals.append(ind)
        
        self.assertFalse(pop._check_convergence())
        
        # Case 2: Converged (top 3 individuals within 1% of each other)
        pop.individuals = []
        for i in range(3):
            ind = Individual(self.config)
            ind.is_trained = True
            ind.training_failed = False
            ind.fitness = 1.0 + (i * 0.005)  # Fitness values: 1.0, 1.005, 1.01
            pop.individuals.append(ind)
        
        self.assertTrue(pop._check_convergence())
        
        # Case 3: Not enough individuals
        pop.individuals = []
        for i in range(2):  # Only 2 individuals
            ind = Individual(self.config)
            ind.is_trained = True
            ind.training_failed = False
            ind.fitness = 1.0
            pop.individuals.append(ind)
        
        self.assertFalse(pop._check_convergence())


if __name__ == '__main__':
    unittest.main()
