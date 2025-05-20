"""
Module for the Individual class that represents a single model in the population.
"""

import os
import logging
import numpy as np
import time
import copy
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Import from TrainingAutomation
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from TrainingAutomation.src.training.yolo_trainer import YoloTrainer


class Individual:
    """
    Represents a single model in the evolutionary population.
    
    Each individual has a genome (hyperparameters and architecture settings),
    fitness score, and methods for training and evaluating the model.
    """
    
    def __init__(self, config: Dict, genome: Optional[Dict] = None, generation: int = 0):
        """
        Initialise an individual with random or specified genome.
        
        Args:
            config: Configuration dictionary
            genome: Optional predefined genome (hyperparameters)
            generation: Generation number this individual belongs to
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.generation = generation
        
        # Initialise genome either from provided parameters or randomly
        self.genome = genome if genome else self._initialise_random_genome()
        
        # Fitness scores (initialised as None)
        self.map_score = None
        self.inference_time = None
        self.fitness = None
        
        # Training state
        self.is_trained = False
        self.training_failed = False
        self.trainer = None
        self.model_path = None
    
    def _initialise_random_genome(self) -> Dict:
        """
        Initialise a random genome with hyperparameters within valid ranges.
        
        Returns:
            Dictionary of hyperparameters
        """
        ranges = self.config.get('hyperparameter_ranges', {})
        arch_ranges = self.config.get('architecture_params', {})
        
        # Create dictionary for hyperparameters
        genome = {
            # Training hyperparameters
            "learning_rate": np.random.uniform(
                ranges.get('learning_rate', {}).get('min', 0.001),
                ranges.get('learning_rate', {}).get('max', 0.01)
            ),
            "momentum": np.random.uniform(
                ranges.get('momentum', {}).get('min', 0.6),
                ranges.get('momentum', {}).get('max', 0.98)
            ),
            "weight_decay": np.random.uniform(
                ranges.get('weight_decay', {}).get('min', 0.0001),
                ranges.get('weight_decay', {}).get('max', 0.001)
            ),
            "batch_size": int(np.random.randint(
                ranges.get('batch_size', {}).get('min', 8),
                ranges.get('batch_size', {}).get('max', 32)
            )),
            "epochs": int(np.random.randint(
                ranges.get('epochs', {}).get('min', 50),
                ranges.get('epochs', {}).get('max', 200)
            )),
            
            # Architecture parameters
            "width_multiple": np.random.uniform(
                arch_ranges.get('width_multiple', {}).get('min', 0.5),
                arch_ranges.get('width_multiple', {}).get('max', 1.0)
            ),
            "depth_multiple": np.random.uniform(
                arch_ranges.get('depth_multiple', {}).get('min', 0.33),
                arch_ranges.get('depth_multiple', {}).get('max', 1.0)
            ),
            
            # Warmup parameters
            "warmup_epochs": np.random.uniform(1.0, 5.0),
            "warmup_momentum": np.random.uniform(0.6, 0.9),
            "warmup_bias_lr": np.random.uniform(0.05, 0.2),
            
            # Loss weights
            "box_loss_gain": np.random.uniform(0.02, 0.1),
            "cls_loss_gain": np.random.uniform(0.2, 0.8),
        }
        
        self.logger.debug(f"Initialised random genome: {genome}")
        return genome
    
    def train(self, output_dir: str) -> bool:
        """
        Train this individual's model using the TrainerWrapper.
        
        Args:
            output_dir: Directory to save training outputs
            
        Returns:
            True if training completed successfully, False otherwise
        """
        if self.is_trained:
            self.logger.info(f"Individual from generation {self.generation} already trained, skipping")
            return True
        
        try:
            self.logger.info(f"Training individual from generation {self.generation}")
            
            # Create trainer
            from src.training.trainer_wrapper import TrainerWrapper
            trainer = TrainerWrapper(
                config=self.config,
                output_dir=output_dir,
                generation=self.generation,
                individual_id=id(self) % 10000,  # Use object id modulo 10000 for a simple unique ID
                genome=self.genome
            )
            self.trainer = trainer
            
            # Train the model
            result = trainer.train_model()
            
            # Extract results
            if result['status'] == 'success':
                metrics = result.get('metrics', {})
                self.map_score = metrics.get('map50-95', 0)
                self.inference_time = metrics.get('inference_time', float('inf'))
                self.model_path = result.get('checkpoint_path')
                self.is_trained = True
                
                # Calculate fitness
                self._calculate_fitness()
                
                self.logger.info(f"Training successful: mAP={self.map_score:.4f}, inference={self.inference_time:.2f}ms, fitness={self.fitness:.6f}")
                return True
            else:
                self.training_failed = True
                self.logger.warning(f"Training failed with error: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.training_failed = True
            return False
    
    def _calculate_fitness(self) -> None:
        """
        Calculate fitness based on mAP and inference time.
        
        Fitness formula: (3 * mAP@0.5:0.95) / inference_time
        """
        if self.inference_time is None or self.inference_time <= 0:
            self.fitness = 0
        elif self.map_score is None:
            self.fitness = 0
        else:
            # Use the specified fitness formula: (3 * mAP) / inference_time
            self.fitness = (3 * self.map_score) / self.inference_time
    
    def export(self, output_dir: str) -> Optional[str]:
        """
        Export the trained model to the specified directory.
        
        Args:
            output_dir: Directory to export the model to
            
        Returns:
            Path to the exported model, or None if export failed
        """
        if not self.is_trained or self.training_failed:
            self.logger.warning("Cannot export untrained or failed model")
            return None
            
        if not self.model_path or not os.path.exists(self.model_path):
            self.logger.warning("Model path not found")
            return None
        
        try:
            # Create export directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Export model
            export_path = os.path.join(output_dir, f"best_model.pt")
            
            # Copy model file
            import shutil
            shutil.copy2(self.model_path, export_path)
            
            # Save genome and metrics
            metadata = {
                "genome": self.genome,
                "fitness": self.fitness,
                "map_score": self.map_score,
                "inference_time": self.inference_time,
                "generation": self.generation
            }
            
            with open(os.path.join(output_dir, "model_metadata.json"), 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
                
            self.logger.info(f"Model exported to {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Error exporting model: {str(e)}")
            return None
    
    def to_dict(self) -> Dict:
        """
        Convert this individual to a dictionary for serialisation.
        
        Returns:
            Dictionary representation of this individual
        """
        return {
            "genome": self.genome,
            "generation": self.generation,
            "fitness": self.fitness,
            "map_score": self.map_score,
            "inference_time": self.inference_time,
            "is_trained": self.is_trained,
            "training_failed": self.training_failed,
            "model_path": str(self.model_path) if self.model_path else None
        }
        
    @classmethod
    def from_dict(cls, data: Dict, config: Dict) -> 'Individual':
        """
        Create an individual from a dictionary representation.
        
        Args:
            data: Dictionary representation of an individual
            config: Configuration dictionary
            
        Returns:
            New Individual instance
        """
        individual = cls(config, genome=data["genome"], generation=data["generation"])
        individual.fitness = data["fitness"]
        individual.map_score = data["map_score"]
        individual.inference_time = data["inference_time"]
        individual.is_trained = data["is_trained"]
        individual.training_failed = data["training_failed"]
        individual.model_path = data["model_path"]
        
        return individual
