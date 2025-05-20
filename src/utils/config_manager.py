# test
"""
Configuration management utility for the Evolutionary Training Manager.
"""

import os
import yaml
import logging
import string
from typing import Dict, Any, Optional


class ConfigManager:
    """
    Manages configuration loading, validation, and environment variable expansion.
    """
    
    def __init__(self, config_path: str):
        """
        Initialise the configuration manager.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """
        Load and validate configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        try:
            if not os.path.exists(self.config_path):
                self.logger.error(f"Configuration file not found: {self.config_path}")
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Expand environment variables
            config = self._expand_env_vars(config)
            
            # Validate configuration
            self._validate_config(config)
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def _expand_env_vars(self, config: Dict) -> Dict:
        """
        Recursively expand environment variables in configuration values.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with expanded environment variables
        """
        if isinstance(config, dict):
            return {key: self._expand_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._expand_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Expand ${VAR} or $VAR style environment variables
            template = string.Template(config)
            return template.safe_substitute(os.environ)
        else:
            return config
    
    def _validate_config(self, config: Dict) -> None:
        """
        Validate configuration structure and required fields.
        
        Args:
            config: Configuration dictionary to validate
        """
        # Check required sections
        required_sections = ['paths', 'evolution', 'model', 'hardware']
        for section in required_sections:
            if section not in config:
                self.logger.error(f"Missing required configuration section: {section}")
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Check evolution parameters
        evolution = config.get('evolution', {})
        required_evolution_params = [
            'population_size', 'mutation_rate', 'crossover_parents',
            'survivors_count', 'offspring_count'
        ]
        for param in required_evolution_params:
            if param not in evolution:
                self.logger.error(f"Missing required evolution parameter: {param}")
                raise ValueError(f"Missing required evolution parameter: {param}")
    
    def get_config(self) -> Dict:
        """
        Get the loaded and validated configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config
    
    def get_value(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation path.
        
        Args:
            path: Dot-separated path to configuration value (e.g., 'evolution.population_size')
            default: Default value to return if path not found
            
        Returns:
            Configuration value or default
        """
        parts = path.split('.')
        current = self.config
        
        try:
            for part in parts:
                current = current[part]
            return current
        except (KeyError, TypeError):
            return default
