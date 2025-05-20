"""
Module for evaluating the fitness of YOLO models.
"""

import logging
from typing import Dict, Any, Optional, Union, Tuple

class FitnessEvaluator:
    """
    Evaluates the fitness of YOLO models based on accuracy and speed.
    
    This class calculates fitness scores using the formula:
    fitness = (3 * mAP@0.5:0.95) / inference_time
    
    Also provides methods for comparing models and tracking relative improvements.
    """
    
    def __init__(self, config: Dict):
        """
        Initialise the fitness evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.baseline_map = None
        self.baseline_inference_time = None
    
    def evaluate(self, map_score: float, inference_time: float) -> float:
        """
        Calculate fitness score from mAP and inference time.
        
        Args:
            map_score: mAP@0.5:0.95 score
            inference_time: Inference time in seconds
            
        Returns:
            Fitness score
        """
        # Ensure we don't divide by zero
        inference_time = max(inference_time, 0.001)
        
        # Calculate fitness using the formula (3*mAP@0.5:0.95)/inference time
        fitness = (3 * map_score) / inference_time
        
        return fitness
    
    def set_baseline(self, map_score: float, inference_time: float) -> None:
        """
        Set baseline metrics for calculating relative improvements.
        
        Args:
            map_score: Baseline mAP@0.5:0.95 score
            inference_time: Baseline inference time in seconds
        """
        self.baseline_map = map_score
        self.baseline_inference_time = inference_time
    
    def calculate_relative_improvement(
        self, 
        map_score: float, 
        inference_time: float
    ) -> Tuple[float, float]:
        """
        Calculate relative improvement from baseline.
        
        Args:
            map_score: Current mAP@0.5:0.95 score
            inference_time: Current inference time in seconds
            
        Returns:
            Tuple of (relative_map_improvement, relative_inference_improvement)
        """
        if self.baseline_map is None or self.baseline_inference_time is None:
            self.logger.warning("Baseline not set, unable to calculate relative improvement")
            return 0.0, 0.0
        
        # Calculate relative improvements
        relative_map_improvement = 0.0
        if self.baseline_map > 0:
            relative_map_improvement = (map_score - self.baseline_map) / self.baseline_map
        
        relative_inference_improvement = 0.0
        if self.baseline_inference_time > 0:
            # Improvement means lower inference time
            relative_inference_improvement = (self.baseline_inference_time - inference_time) / self.baseline_inference_time
        
        return relative_map_improvement, relative_inference_improvement
    
    def calculate_weighted_improvement(
        self, 
        map_score: float, 
        inference_time: float,
        map_weight: float = 1.5,  # 150% weighting for accuracy
        inference_weight: float = 1.0
    ) -> float:
        """
        Calculate weighted relative improvement score.
        
        Implements the requirement to prioritise relative improvement in accuracy
        150% as much as relative improvement in speed.
        
        Args:
            map_score: Current mAP@0.5:0.95 score
            inference_time: Current inference time in seconds
            map_weight: Weight for mAP improvement (default: 1.5)
            inference_weight: Weight for inference time improvement (default: 1.0)
            
        Returns:
            Weighted improvement score
        """
        rel_map_imp, rel_inf_imp = self.calculate_relative_improvement(map_score, inference_time)
        
        # Calculate weighted score
        weighted_score = (map_weight * rel_map_imp) + (inference_weight * rel_inf_imp)
        
        return weighted_score
    
    def compare_models(
        self, 
        model1_map: float, 
        model1_inference: float,
        model2_map: float, 
        model2_inference: float
    ) -> int:
        """
        Compare two models based on their fitness.
        
        Args:
            model1_map: mAP@0.5:0.95 score of model 1
            model1_inference: Inference time of model 1
            model2_map: mAP@0.5:0.95 score of model 2
            model2_inference: Inference time of model 2
            
        Returns:
            1 if model 1 is better, -1 if model 2 is better, 0 if equal
        """
        fitness1 = self.evaluate(model1_map, model1_inference)
        fitness2 = self.evaluate(model2_map, model2_inference)
        
        if fitness1 > fitness2:
            return 1
        elif fitness1 < fitness2:
            return -1
        else:
            return 0
