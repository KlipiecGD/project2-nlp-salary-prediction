import os
import yaml
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class GeneralConfig(BaseModel):
    """General settings."""
    random_seed: int = Field(ge=0, description="Random seed for reproducibility")


class PathsConfig(BaseModel):
    """Directory and file paths."""
    data_filepath: str
    models_dir: str
    preprocessors_dir: str
    tfidf_features_dir: str
    embeddings_dir: str
    reports_dir: str
    plots_dir: str
    
    @field_validator('*')
    @classmethod
    def validate_paths(cls, v: str) -> str:
        """Ensure paths use forward slashes."""
        return v.replace('\\', '/')


class DataColumnsConfig(BaseModel):
    """Column name configuration."""
    title: str
    description: str
    target: str
    categorical: list[str]
    high_cardinality: list[str]


class DataSplitConfig(BaseModel):
    """Train/validation/test split configuration."""
    test_size: float = Field(gt=0, lt=1, description="Test set proportion")
    valid_size: float = Field(gt=0, lt=1, description="Validation proportion of test set")


class DataConfig(BaseModel):
    """Data-related configuration."""
    columns: DataColumnsConfig
    split: DataSplitConfig


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""
    patience: int = Field(ge=0)
    factor: float = Field(gt=0, lt=1)


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""
    patience: int = Field(ge=0)


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    num_workers: int = Field(ge=0)
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    optimizer: Literal["adam", "sgd"] = "adam"
    loss_function: Literal["mse", "mae"] = "mse"
    scheduler: SchedulerConfig
    early_stopping: EarlyStoppingConfig


class EmbeddingsConfig(BaseModel):
    """Embedding layer configuration."""
    max_seq_length: int = Field(gt=0)
    embedding_dim: int = Field(gt=0)
    w2v_embedding_dim: int = Field(gt=0)


class ArchitectureConfig(BaseModel):
    """Neural network architecture configuration."""
    dropout_rate: float = Field(ge=0, le=1)
    cat_hidden_size: int = Field(gt=0)
    reg_hidden_size: int = Field(gt=0)
    emb_hidden_size: int = Field(gt=0)
    recurrent_hidden_size: int = Field(gt=0)
    num_filters: int = Field(gt=0)
    num_residual_blocks: int = Field(ge=1, le=3)


class LossConfig(BaseModel):
    """Loss function configuration."""
    delta: float = Field(gt=0)


class ModelConfig(BaseModel):
    """Model architecture and configuration."""
    embeddings: EmbeddingsConfig
    architecture: ArchitectureConfig
    loss: LossConfig


class PretrainedConfig(BaseModel):
    """Pretrained model configuration."""
    sentence_transformer: str
    word_embeddings: str


class Config(BaseModel):
    """Root configuration model."""
    general: GeneralConfig
    paths: PathsConfig
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    pretrained: PretrainedConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """
        Load configuration from YAML file with validation.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Validated Config instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config doesn't match schema
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def get(self, key_path: str, default=None):
        """
        Get nested configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self
        
        try:
            for key in keys:
                if hasattr(value, key):
                    value = getattr(value, key)
                else:
                    return default
            return value
        except (AttributeError, KeyError):
            return default
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"  # Prevent extra fields


# Singleton instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None, reload: bool = False) -> Config:
    """
    Get or create the configuration singleton.
    
    Args:
        config_path: Path to config file. If None, uses default path or ENV variable
        reload: Force reload of configuration
        
    Returns:
        Validated Config instance
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
        
        _config = Config.from_yaml(config_path)
    
    return _config


def reset_config():
    """Reset the configuration singleton. Useful for testing."""
    global _config
    _config = None