"""
Configuration management for the neural machine translation project.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model architectures."""
    
    # RNN Configuration
    rnn_embedding_dim: int = 256
    rnn_hidden_dim: int = 512
    rnn_num_layers: int = 2
    rnn_dropout: float = 0.1
    rnn_bidirectional: bool = True
    
    # Transformer Configuration
    transformer_d_model: int = 512
    transformer_nhead: int = 8
    transformer_num_layers: int = 6
    transformer_dim_feedforward: int = 2048
    transformer_dropout: float = 0.1
    transformer_max_position_embeddings: int = 5000


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 10
    warmup_steps: int = 4000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    scheduler: str = "cosine"  # "cosine", "linear", "constant"
    early_stopping_patience: int = 5
    save_best_only: bool = True
    gradient_accumulation_steps: int = 1


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    max_length: int = 50
    min_freq: int = 2
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    src_lang: str = "es"
    tgt_lang: str = "en"
    lowercase: bool = True
    remove_punctuation: bool = False
    data_dir: str = "data"
    cache_dir: str = "cache"


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    
    log_level: str = "INFO"
    log_dir: str = "logs"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "neural-machine-translation"
    wandb_entity: Optional[str] = None
    save_attention_plots: bool = True
    log_interval: int = 100


@dataclass
class Config:
    """Main configuration class."""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # General settings
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create necessary directories
        for dir_path in [self.checkpoint_dir, self.results_dir, 
                        self.logging.log_dir, self.data.data_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested dictionary structure
        flattened = {}
        for section, values in config_dict.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flattened[f"{section}_{key}"] = value
            else:
                flattened[section] = values
        
        return cls(**flattened)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "model": {
                "rnn_embedding_dim": self.model.rnn_embedding_dim,
                "rnn_hidden_dim": self.model.rnn_hidden_dim,
                "rnn_num_layers": self.model.rnn_num_layers,
                "rnn_dropout": self.model.rnn_dropout,
                "rnn_bidirectional": self.model.rnn_bidirectional,
                "transformer_d_model": self.model.transformer_d_model,
                "transformer_nhead": self.model.transformer_nhead,
                "transformer_num_layers": self.model.transformer_num_layers,
                "transformer_dim_feedforward": self.model.transformer_dim_feedforward,
                "transformer_dropout": self.model.transformer_dropout,
                "transformer_max_position_embeddings": self.model.transformer_max_position_embeddings,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "epochs": self.training.epochs,
                "warmup_steps": self.training.warmup_steps,
                "max_grad_norm": self.training.max_grad_norm,
                "weight_decay": self.training.weight_decay,
                "scheduler": self.training.scheduler,
                "early_stopping_patience": self.training.early_stopping_patience,
                "save_best_only": self.training.save_best_only,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            },
            "data": {
                "max_length": self.data.max_length,
                "min_freq": self.data.min_freq,
                "train_split": self.data.train_split,
                "val_split": self.data.val_split,
                "test_split": self.data.test_split,
                "src_lang": self.data.src_lang,
                "tgt_lang": self.data.tgt_lang,
                "lowercase": self.data.lowercase,
                "remove_punctuation": self.data.remove_punctuation,
                "data_dir": self.data.data_dir,
                "cache_dir": self.data.cache_dir,
            },
            "logging": {
                "log_level": self.logging.log_level,
                "log_dir": self.logging.log_dir,
                "tensorboard": self.logging.tensorboard,
                "wandb": self.logging.wandb,
                "wandb_project": self.logging.wandb_project,
                "wandb_entity": self.logging.wandb_entity,
                "save_attention_plots": self.logging.save_attention_plots,
                "log_interval": self.logging.log_interval,
            },
            "seed": self.seed,
            "device": self.device,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "checkpoint_dir": self.checkpoint_dir,
            "results_dir": self.results_dir,
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type."""
        if model_type.lower() == "rnn":
            return {
                "embedding_dim": self.model.rnn_embedding_dim,
                "hidden_dim": self.model.rnn_hidden_dim,
                "num_layers": self.model.rnn_num_layers,
                "dropout": self.model.rnn_dropout,
                "bidirectional": self.model.rnn_bidirectional,
            }
        elif model_type.lower() == "transformer":
            return {
                "d_model": self.model.transformer_d_model,
                "nhead": self.model.transformer_nhead,
                "num_layers": self.model.transformer_num_layers,
                "dim_feedforward": self.model.transformer_dim_feedforward,
                "dropout": self.model.transformer_dropout,
                "max_position_embeddings": self.model.transformer_max_position_embeddings,
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or return default."""
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)
    return Config() 