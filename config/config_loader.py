import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration class that loads from YAML and provides attribute access."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            self._config = yaml.safe_load(f)
        
        if self._config is None:
            raise ValueError(f"Configuration file is empty: {config_path}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        
        Examples:
            >>> config.get('training.batch_size')
            64
            >>> config.get('model.architecture.dropout_rate')
            0.3
            >>> config.get('nonexistent.key', 42)
            42
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def to_dict(self) -> Dict:
        """Return the full configuration as a dictionary."""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access: config['general']"""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator: 'general' in config"""
        return key in self._config
    
    # Convenience properties for top-level access
    @property
    def general(self) -> Dict:
        """Access general settings."""
        return self._config.get('general', {})
    
    @property
    def paths(self) -> Dict:
        """Access path settings."""
        return self._config.get('paths', {})
    
    @property
    def data(self) -> Dict:
        """Access data settings."""
        return self._config.get('data', {})
    
    @property
    def training(self) -> Dict:
        """Access training settings."""
        return self._config.get('training', {})
    
    @property
    def model(self) -> Dict:
        """Access model settings."""
        return self._config.get('model', {})
    
    @property
    def pretrained(self) -> Dict:
        """Access pretrained model settings."""
        return self._config.get('pretrained', {})
    
    def __repr__(self) -> str:
        """String representation of config."""
        return f"Config({list(self._config.keys())})"


# Singleton instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None, reload: bool = False) -> Config:
    """
    Get or create the configuration singleton.
    
    Args:
        config_path: Path to config file. If None, uses default path or ENV variable
        reload: Force reload of configuration
        
    Returns:
        Config instance
        
    Examples:
        >>> config = get_config()
        >>> config = get_config('config/config_prod.yaml')
        >>> config = get_config(reload=True)  # Force reload
    """
    global _config
    
    if _config is None or reload:
        if config_path is None:
            # Check for environment variable
            env = os.getenv('CONFIG_ENV', 'config')
            config_path = os.getenv('CONFIG_PATH', f'config/{env}.yaml')
            
            # Fallback to default if env-specific doesn't exist
            if not Path(config_path).exists():
                config_path = 'config/config.yaml'
        
        _config = Config(config_path)
    
    return _config


def reset_config():
    """Reset the configuration singleton. Useful for testing."""
    global _config
    _config = None