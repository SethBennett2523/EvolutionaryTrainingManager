"""
Wrapper for the YoloTrainer from TrainingAutomation.
"""

import os
import logging
import time
import copy
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Import from TrainingAutomation
from TrainingAutomation.src.training.yolo_trainer import YoloTrainer as OriginalYoloTrainer
from TrainingAutomation.src.training.early_stopping import EarlyStopping
from TrainingAutomation.src.training.hardware_manager import HardwareManager


class TrainerWrapper:
    """
    Wrapper around the YoloTrainer from TrainingAutomation.
    
    This class provides additional functionality for handling training failures,
    hardware resource management, and integration with the evolutionary algorithm.
    """
    
    def __init__(
        self,
        config: Dict,
        output_dir: str,
        generation: int,
        individual_id: int,
        genome: Dict
    ):
        """
        Initialise the trainer wrapper.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory for outputs
            generation: Current generation number
            individual_id: ID of the individual being trained
            genome: Genome dictionary with hyperparameters
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.output_dir = output_dir
        self.generation = generation
        self.individual_id = individual_id
        self.genome = genome
        
        # Set paths
        self.model_dir = os.path.join(self.output_dir, f"gen_{generation}", f"ind_{individual_id}")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Hardware manager for resource management
        self.hardware_manager = HardwareManager(config_path=None)
        
        # Extract data YAML path from config
        self.data_yaml_path = os.path.abspath(config['paths']['data_yaml'])
        
        # Create temporary config file for this training run
        self.temp_config_path = os.path.join(self.model_dir, "training_config.yaml")
        self._create_training_config()
        
        # Training metrics
        self.metrics = {}
        self.trained = False
        self.failed = False
        self.checkpoint_path = None
        
    def _create_training_config(self) -> None:
        """
        Create a temporary config file for this training run based on the genome.
        """
        # Start with base configuration
        training_config = {
            "training": {
                "epochs": self.genome.get("epochs", 100),
                "batch_size": self.genome.get("batch_size", 16),
                "patience": 20,  # Early stopping patience
                "imgsz": 640,
                "device": self.config['hardware']['device'],
                "workers": self.config['hardware'].get('workers', 2),
                "verbose": True,
                "pretrained": False,  # Never use pretrained model
            },
            "hyperparameters": {
                "lr0": self.genome.get("learning_rate", 0.01),
                "lrf": self.genome.get("final_lr_factor", 0.01),
                "momentum": self.genome.get("momentum", 0.937),
                "weight_decay": self.genome.get("weight_decay", 0.0005),
                "warmup_epochs": self.genome.get("warmup_epochs", 3.0),
                "warmup_momentum": self.genome.get("warmup_momentum", 0.8),
                "warmup_bias_lr": self.genome.get("warmup_bias_lr", 0.1),
                "box": self.genome.get("box_loss_gain", 0.05),
                "cls": self.genome.get("cls_loss_gain", 0.5),
            },
            "architecture": {
                "model_type": self.config['model'].get('base_type', "yolov8m"),
                "width_multiple": self.genome.get("width_multiple", 1.0),
                "depth_multiple": self.genome.get("depth_multiple", 1.0),
            },
        }
        
        # Write config to file
        with open(self.temp_config_path, 'w') as f:
            import yaml
            yaml.dump(training_config, f, default_flow_style=False)
        
        self.logger.info(f"Created training config for individual {self.individual_id} at {self.temp_config_path}")

    def train_model(self) -> Dict:
        """
        Train the model using the specified genome.
        
        Returns:
            Dictionary with training results and metrics
        """
        try:
            self.logger.info(f"Training model for individual {self.individual_id} (generation {self.generation})")
            
            # Initialize YOLOv8 trainer from TrainingAutomation
            trainer = OriginalYoloTrainer(
                config_path=self.temp_config_path,
                data_yaml_path=self.data_yaml_path,
                output_dir=self.model_dir,
                verbose=True
            )
            
            # Train the model
            start_time = time.time()
            result = trainer.train()
            training_time = time.time() - start_time
            
            # Extract metrics
            metrics = result.get('metrics', {})
            metrics['training_time'] = training_time
            
            # Measure inference time
            inference_time = self._measure_inference_time(trainer)
            metrics['inference_time'] = inference_time
            
            # Save results
            self.metrics = metrics
            self.checkpoint_path = result.get('model_path')
            self.trained = True
            
            self.logger.info(f"Training completed for individual {self.individual_id}")
            self.logger.info(f"mAP@0.5:0.95: {metrics.get('map50-95', 0):.4f}")
            self.logger.info(f"Inference time: {inference_time:.2f}ms")
            
            return {
                'status': 'success',
                'metrics': metrics,
                'checkpoint_path': self.checkpoint_path,
            }
            
        except Exception as e:
            self.logger.error(f"Training failed for individual {self.individual_id}: {str(e)}")
            self.failed = True
            
            return {
                'status': 'failed',
                'error': str(e),
                'metrics': {},
                'checkpoint_path': None,
            }
    
    def _measure_inference_time(self, trainer) -> float:
        """
        Measure the inference time of the trained model.
        
        Args:
            trainer: Trained YoloTrainer instance
            
        Returns:
            Average inference time in milliseconds
        """
        try:
            # Create a model for inference
            model_path = trainer.get_best_model_path()
            if not model_path or not os.path.exists(model_path):
                self.logger.warning("Could not find trained model for inference measurement")
                return float('inf')
            
            # Load the model using Ultralytics YOLO
            from ultralytics import YOLO
            model = YOLO(model_path)
            
            # Run inference multiple times to get average speed
            num_runs = 10
            warmup_runs = 3
            total_time = 0.0
            
            # Use a sample image (create a dummy one if needed)
            import numpy as np
            from PIL import Image
            
            # Create a test image (640x640 blank image)
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            test_img_path = os.path.join(self.model_dir, "test_image.jpg")
            Image.fromarray(test_img).save(test_img_path)
            
            # Warmup runs
            for _ in range(warmup_runs):
                _ = model(test_img_path)
            
            # Timed runs
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(test_img_path)
                total_time += (time.time() - start_time) * 1000  # Convert to ms
            
            # Calculate average inference time
            avg_time = total_time / num_runs
            
            # Clean up
            if os.path.exists(test_img_path):
                os.remove(test_img_path)
            
            return avg_time
            
        except Exception as e:
            self.logger.error(f"Error measuring inference time: {str(e)}")
            return float('inf')  # Return infinity as a fallback

    def export_model(self, format: str = 'onnx') -> str:
        """
        Export the trained model.
        
        Args:
            format: Export format (onnx, torchscript, etc.)
        
        Returns:
            Path to exported model
        """
        if not self.is_trained or self.training_failed:
            self.logger.error("Cannot export untrained or failed model")
            return ""
        
        try:
            export_path = self.trainer.export_model(format=format)
            self.logger.info(f"Model exported to {export_path}")
            return export_path
        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            return ""

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the model based on mAP and inference time.
        
        Returns:
            Tuple of (mAP score, inference time)
        """
        if not self.is_trained or self.training_failed:
            return 0.0, float('inf')
        
        try:
            self.map_score = self._extract_map_from_results(self.results)
            self.inference_time = self._measure_inference_time()
            return self.map_score, self.inference_time
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return 0.0, float('inf')

    def _measure_inference_time(self) -> float:
        """
        Measure model inference time.
        
        Returns:
            Average inference time in seconds
        """
        if not self.trainer or not hasattr(self.trainer, 'model') or self.trainer.model is None:
            self.logger.warning("Model not available for inference time measurement")
            return 1.0  # Default value
        
        try:
            import torch
            import numpy as np
            
            device = self.trainer.model.device
            img_size = self.genome.get('img_size', 640)
            iterations = 10
            
            # Create a dummy input tensor
            dummy_input = torch.zeros((1, 3, img_size, img_size), device=device)
            
            # Warm-up
            for _ in range(5):
                _ = self.trainer.model(dummy_input)
            
            # Timed runs
            start_time = time.time()
            for _ in range(iterations):
                _ = self.trainer.model(dummy_input)
            if device != 'cpu':
                torch.cuda.synchronize()  # Force synchronisation if using GPU
            inference_time = (time.time() - start_time) / iterations
            
            return inference_time
        except Exception as e:
            self.logger.error(f"Error measuring inference time: {e}")
            return 1.0  # Default value on error

    def _extract_map_from_results(self, results: Dict) -> float:
        """
        Extract mAP score from results dictionary.
        
        Args:
            results: Training results dictionary
        
        Returns:
            mAP score as a float
        """
        if isinstance(results, dict):
            if 'metrics' in results and isinstance(results['metrics'], dict):
                if 'mAP50-95' in results['metrics']:
                    return results['metrics']['mAP50-95']
                elif 'map' in results['metrics']:
                    return results['metrics']['map']
                elif 'mAP' in results['metrics']:
                    return results['metrics']['mAP']
            for key in ['mAP', 'map', 'mAP50-95', 'val/map50-95']:
                if key in results:
                    return results[key]
        self.logger.warning("Could not find mAP score in results, using default value 0.0")
        return 0.0

    def _extract_metrics(self) -> None:
        """
        Extract metrics from training results.
        """
        if isinstance(self.results, dict) and 'early_stopping' in self.results:
            es_summary = self.results['early_stopping']
            self.map_score = es_summary.get('best_score', 0)
        else:
            self.map_score = self._extract_map_from_results(self.results)
        self.inference_time = self._measure_inference_time()  # Measure inference time

    def _genome_to_train_params(self) -> Dict:
        """
        Convert genome to training parameters.
        
        Returns:
            Dictionary of training parameters
        """
        train_params = {
            'batch': self.genome.get('batch_size', 16),
            'epochs': self.genome.get('epochs', 100),
            'imgsz': self.genome.get('img_size', 640),
            'lr0': self.genome.get('learning_rate', 0.01),
            'mosaic': self.genome.get('mosaic', 1.0),
            'mixup': self.genome.get('mixup', 0.0),
            'cfg': '',  # Empty string uses default architecture
            'patience': 20,  # Early stopping parameters
            'pretrained': False,  # Ensure we're not using a pretrained model
        }
        
        # Add YOLO-specific architecture parameters if available
        if 'depth_multiple' in self.genome and 'width_multiple' in self.genome:
            # We need to create a custom model configuration
            # This requires more complex handling - would need to create a custom config file
            pass
        
        return train_params

    def train(self) -> bool:
        """
        Train the model using the individual's genome.
        
        Returns:
            True if training completed successfully, False otherwise
        """
        if self.training_failed:
            return False
        if not self.trainer:
            if not self.initialise():
                return False
        try:
            self.trainer.initialize_model()  # Initialise model
            train_params = self._genome_to_train_params()  # Convert genome to training parameters
            
            # Check if we have hardware-specific optimisations
            if self.hardware_manager:
                # Get hardware-aware parameters
                hardware_params = self.hardware_manager.get_training_params(
                    image_size=train_params.get('imgsz', 640)
                )
                # Only override batch size and workers if not specified in genome
                if 'batch' not in train_params:
                    train_params['batch'] = hardware_params.get('batch_size', 16)
                if 'workers' not in train_params:
                    train_params['workers'] = hardware_params.get('workers', 4)
            
            self.logger.info(f"Training individual with genome: {self.genome}")
            start_time = time.time()
            self.results = self.trainer.train(train_params)
            training_time = time.time() - start_time
            self._extract_metrics()  # Extract metrics
            self.is_trained = True
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Get best model path
            if self.early_stopping:
                self.best_model_path = self.early_stopping.get_best_checkpoint()
            
            return True
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.training_failed = True
            return False

    def initialise(self) -> bool:
        """
        Initialise the trainer and hardware manager.
        
        Returns:
            True if initialisation was successful, False otherwise
        """
        try:
            self.trainer = OriginalYoloTrainer(
                config_path=self.config_path,
                data_yaml_path=self.data_yaml_path,
                output_dir=self.output_dir,
                verbose=True
            )
            self.early_stopping = EarlyStopping(
                patience=20,
                min_delta=0.0001,
                metric_name='mAP50-95',
                maximize=True,
                save_dir=self.output_dir,
                verbose=True
            )
            self.hardware_manager = HardwareManager(self.config_path)  # Initialise hardware manager
            return True
        except Exception as e:
            self.logger.error(f"Initialisation failed: {e}")
            self.training_failed = True
            return False

    def export_model(self, format: str = 'onnx') -> str:
        """
        Export the trained model.
        
        Args:
            format: Export format (onnx, torchscript, etc.)
        
        Returns:
            Path to exported model
        """
        if not self.is_trained or self.training_failed:
            self.logger.error("Cannot export untrained or failed model")
            return ""
        
        try:
            export_path = self.trainer.export_model(format=format)
            self.logger.info(f"Model exported to {export_path}")
            return export_path
        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            return ""

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the model based on mAP and inference time.
        
        Returns:
            Tuple of (mAP score, inference time)
        """
        if not self.is_trained or self.training_failed:
            return 0.0, float('inf')
        
        try:
            self.map_score = self._extract_map_from_results(self.results)
            self.inference_time = self._measure_inference_time()
            return self.map_score, self.inference_time
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return 0.0, float('inf')

    def _measure_inference_time(self) -> float:
        """
        Measure model inference time.
        
        Returns:
            Average inference time in seconds
        """
        if not self.trainer or not hasattr(self.trainer, 'model') or self.trainer.model is None:
            self.logger.warning("Model not available for inference time measurement")
            return 1.0  # Default value
        
        try:
            import torch
            import numpy as np
            
            device = self.trainer.model.device
            img_size = self.genome.get('img_size', 640)
            iterations = 10
            
            # Create a dummy input tensor
            dummy_input = torch.zeros((1, 3, img_size, img_size), device=device)
            
            # Warm-up
            for _ in range(5):
                _ = self.trainer.model(dummy_input)
            
            # Timed runs
            start_time = time.time()
            for _ in range(iterations):
                _ = self.trainer.model(dummy_input)
            if device != 'cpu':
                torch.cuda.synchronize()  # Force synchronisation if using GPU
            inference_time = (time.time() - start_time) / iterations
            
            return inference_time
        except Exception as e:
            self.logger.error(f"Error measuring inference time: {e}")
            return 1.0  # Default value on error

    def _extract_map_from_results(self, results: Dict) -> float:
        """
        Extract mAP score from results dictionary.
        
        Args:
            results: Training results dictionary
        
        Returns:
            mAP score as a float
        """
        if isinstance(results, dict):
            if 'metrics' in results and isinstance(results['metrics'], dict):
                if 'mAP50-95' in results['metrics']:
                    return results['metrics']['mAP50-95']
                elif 'map' in results['metrics']:
                    return results['metrics']['map']
                elif 'mAP' in results['metrics']:
                    return results['metrics']['mAP']
            for key in ['mAP', 'map', 'mAP50-95', 'val/map50-95']:
                if key in results:
                    return results[key]
        self.logger.warning("Could not find mAP score in results, using default value 0.0")
        return 0.0

    def _extract_metrics(self) -> None:
        """
        Extract metrics from training results.
        """
        if isinstance(self.results, dict) and 'early_stopping' in self.results:
            es_summary = self.results['early_stopping']
            self.map_score = es_summary.get('best_score', 0)
        else:
            self.map_score = self._extract_map_from_results(self.results)
        self.inference_time = self._measure_inference_time()  # Measure inference time

    def _genome_to_train_params(self) -> Dict:
        """
        Convert genome to training parameters.
        
        Returns:
            Dictionary of training parameters
        """
        train_params = {
            'batch': self.genome.get('batch_size', 16),
            'epochs': self.genome.get('epochs', 100),
            'imgsz': self.genome.get('img_size', 640),
            'lr0': self.genome.get('learning_rate', 0.01),
            'mosaic': self.genome.get('mosaic', 1.0),
            'mixup': self.genome.get('mixup', 0.0),
            'cfg': '',  # Empty string uses default architecture
            'patience': 20,  # Early stopping parameters
            'pretrained': False,  # Ensure we're not using a pretrained model
        }
        
        # Add YOLO-specific architecture parameters if available
        if 'depth_multiple' in self.genome and 'width_multiple' in self.genome:
            # We need to create a custom model configuration
            # This requires more complex handling - would need to create a custom config file
            pass
        
        return train_params

    def train(self) -> bool:
        """
        Train the model using the individual's genome.
        
        Returns:
            True if training completed successfully, False otherwise
        """
        if self.training_failed:
            return False
        if not self.trainer:
            if not self.initialise():
                return False
        try:
            self.trainer.initialize_model()  # Initialise model
            train_params = self._genome_to_train_params()  # Convert genome to training parameters
            
            # Check if we have hardware-specific optimisations
            if self.hardware_manager:
                # Get hardware-aware parameters
                hardware_params = self.hardware_manager.get_training_params(
                    image_size=train_params.get('imgsz', 640)
                )
                # Only override batch size and workers if not specified in genome
                if 'batch' not in train_params:
                    train_params['batch'] = hardware_params.get('batch_size', 16)
                if 'workers' not in train_params:
                    train_params['workers'] = hardware_params.get('workers', 4)
            
            self.logger.info(f"Training individual with genome: {self.genome}")
            start_time = time.time()
            self.results = self.trainer.train(train_params)
            training_time = time.time() - start_time
            self._extract_metrics()  # Extract metrics
            self.is_trained = True
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Get best model path
            if self.early_stopping:
                self.best_model_path = self.early_stopping.get_best_checkpoint()
            
            return True
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.training_failed = True
            return False

    def initialise(self) -> bool:
        """
        Initialise the trainer and hardware manager.
        
        Returns:
            True if initialisation was successful, False otherwise
        """
        try:
            self.trainer = OriginalYoloTrainer(
                config_path=self.config_path,
                data_yaml_path=self.data_yaml_path,
                output_dir=self.output_dir,
                verbose=True
            )
            self.early_stopping = EarlyStopping(
                patience=20,
                min_delta=0.0001,
                metric_name='mAP50-95',
                maximize=True,
                save_dir=self.output_dir,
                verbose=True
            )
            self.hardware_manager = HardwareManager(self.config_path)  # Initialise hardware manager
            return True
        except Exception as e:
            self.logger.error(f"Initialisation failed: {e}")
            self.training_failed = True
            return False
