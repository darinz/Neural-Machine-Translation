"""
Training utilities for neural machine translation.

This package contains training loops, evaluation metrics, and utilities
for training neural machine translation models.
"""

from .trainer import Trainer
from .metrics import compute_bleu_score, compute_metrics

__all__ = ["Trainer", "compute_bleu_score", "compute_metrics"] 