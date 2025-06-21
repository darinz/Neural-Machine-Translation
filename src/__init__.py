"""
Neural Machine Translation Package

A comprehensive implementation of neural machine translation models
comparing RNN with Attention and Transformer architectures.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models import RNNModel, TransformerModel
from .data import TranslationDataset, Vocabulary
from .trainer import Trainer
from .utils import Config, setup_logging

__all__ = [
    "RNNModel",
    "TransformerModel", 
    "TranslationDataset",
    "Vocabulary",
    "Trainer",
    "Config",
    "setup_logging"
] 