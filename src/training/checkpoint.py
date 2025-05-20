"""
Checkpoint management for evolutionary training.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class CheckpointManager:
    """
    Manages checkpoints for the evolutionary algorithm.
    
    Handles saving and loading the state of the population and individuals,
    allowing training to be resumed from a checkpoint.
    """
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialise the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint files
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, population, generation: int = None, is_error: bool = False) -> str:
        """
        Save a checkpoint of the current population.
        
        Args:
            population: Population object to save
            generation: Generation number, if None uses population.current_generation
            is_error: Whether this checkpoint is being saved after an error
            
        Returns:
            Path to the saved checkpoint file
        """
        try:
            gen_num = generation if generation is not None else population.current_generation
            
            # Create checkpoint filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"checkpoint_gen_{gen_num}_{timestamp}"
            if is_error:
                filename += "_error"
            filename += ".json"
            
            checkpoint_path = os.path.join(self.checkpoint_dir, filename)
            
            # Convert population to dictionary
            checkpoint_data = {
                "timestamp": timestamp,
                "generation": gen_num,
                "population": population.to_dict()
            }
            
            # Save to file
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            # Create a 'latest.json' file pointing to the most recent checkpoint
            latest_path = os.path.join(self.checkpoint_dir, "latest.json")
            with open(latest_path, 'w') as f:
                json.dump({"latest_checkpoint": checkpoint_path}, f, indent=2)
                
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
            return None
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a population from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Loaded Population object, or None if loading failed
        """
        from src.evolution.population import Population
        
        try:
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            if not os.path.exists(checkpoint_path):
                self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return None
                
            # Load checkpoint data
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Get population data
            population_data = checkpoint_data["population"]
              # Load configuration
            from src.utils.config_manager import ConfigManager
            import os
            default_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yaml")
            config_manager = ConfigManager(default_config_path)
            config = config_manager.get_config()
            
            # Create population from data
            population = Population.from_dict(population_data, config)
            
            self.logger.info(f"Loaded checkpoint from generation {population.current_generation}")
            return population
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            return None
    
    def load_latest_checkpoint(self):
        """
        Load the most recent checkpoint.
        
        Returns:
            Loaded Population object, or None if loading failed
        """
        latest_path = os.path.join(self.checkpoint_dir, "latest.json")
        
        if not os.path.exists(latest_path):
            self.logger.warning("No latest checkpoint found")
            return None
            
        try:
            # Load latest checkpoint path
            with open(latest_path, 'r') as f:
                latest_data = json.load(f)
                
            checkpoint_path = latest_data["latest_checkpoint"]
            
            # Load the checkpoint
            return self.load_checkpoint(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Error loading latest checkpoint: {str(e)}")
            return None
    
    def list_checkpoints(self) -> List[Dict]:
        """
        List all available checkpoints.
        
        Returns:
            List of dictionaries with checkpoint information
        """
        checkpoints = []
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith("checkpoint_") and filename.endswith(".json"):
                checkpoint_path = os.path.join(self.checkpoint_dir, filename)
                
                try:
                    # Extract basic info without loading the full checkpoint
                    with open(checkpoint_path, 'r') as f:
                        data = json.load(f)
                    
                    checkpoints.append({
                        "path": checkpoint_path,
                        "generation": data["generation"],
                        "timestamp": data["timestamp"],
                        "filename": filename
                    })
                except Exception:
                    # Skip problematic files
                    pass
        
        # Sort by generation and timestamp
        checkpoints.sort(key=lambda x: (x["generation"], x["timestamp"]))
        
        return checkpoints
