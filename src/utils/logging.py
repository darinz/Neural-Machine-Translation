"""
Logging utilities for the neural machine translation project.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb


class Logger:
    """Unified logging interface for TensorBoard and Weights & Biases."""
    
    def __init__(self, config, experiment_name: Optional[str] = None):
        """Initialize logger with configuration."""
        self.config = config
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        
        # Setup basic logging
        self._setup_basic_logging()
        
        # Setup TensorBoard
        self.tensorboard_writer = None
        if config.logging.tensorboard:
            self._setup_tensorboard()
        
        # Setup Weights & Biases
        self.wandb_run = None
        if config.logging.wandb:
            self._setup_wandb()
    
    def _setup_basic_logging(self):
        """Setup basic Python logging."""
        log_dir = Path(self.config.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{self.experiment_name}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config.logging.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logger initialized for experiment: {self.experiment_name}")
    
    def _setup_tensorboard(self):
        """Setup TensorBoard writer."""
        log_dir = Path(self.config.logging.log_dir) / "tensorboard" / self.experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
        self.logger.info(f"TensorBoard logging enabled: {log_dir}")
    
    def _setup_wandb(self):
        """Setup Weights & Biases run."""
        try:
            wandb.init(
                project=self.config.logging.wandb_project,
                entity=self.config.logging.wandb_entity,
                name=self.experiment_name,
                config=self._get_wandb_config(),
                reinit=True
            )
            self.wandb_run = wandb.run
            self.logger.info(f"Weights & Biases logging enabled: {wandb.run.url}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Weights & Biases: {e}")
            self.wandb_run = None
    
    def _get_wandb_config(self) -> Dict[str, Any]:
        """Convert config to W&B compatible format."""
        return {
            "model": {
                "rnn_embedding_dim": self.config.model.rnn_embedding_dim,
                "rnn_hidden_dim": self.config.model.rnn_hidden_dim,
                "transformer_d_model": self.config.model.transformer_d_model,
                "transformer_nhead": self.config.model.transformer_nhead,
            },
            "training": {
                "batch_size": self.config.training.batch_size,
                "learning_rate": self.config.training.learning_rate,
                "epochs": self.config.training.epochs,
            },
            "data": {
                "max_length": self.config.data.max_length,
                "min_freq": self.config.data.min_freq,
            }
        }
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to all enabled logging backends."""
        # Add prefix to metric names
        prefixed_metrics = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
        
        # Log to console
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in prefixed_metrics.items()])
        self.logger.info(f"Step {step}: {metric_str}")
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            for name, value in prefixed_metrics.items():
                self.tensorboard_writer.add_scalar(name, value, step)
        
        # Log to Weights & Biases
        if self.wandb_run:
            wandb.log(prefixed_metrics, step=step)
    
    def log_model_parameters(self, model: torch.nn.Module, step: int = 0):
        """Log model parameters and gradients."""
        if self.tensorboard_writer:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.tensorboard_writer.add_histogram(
                        f"gradients/{name}", param.grad, step
                    )
                self.tensorboard_writer.add_histogram(
                    f"parameters/{name}", param.data, step
                )
        
        if self.wandb_run:
            wandb.watch(model, log="all", log_freq=100)
    
    def log_attention_weights(self, attention_weights: torch.Tensor, 
                            src_tokens: list, tgt_tokens: list, 
                            step: int, sample_idx: int = 0):
        """Log attention weights visualization."""
        if not self.config.logging.save_attention_plots:
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Get attention weights for the sample
            attn = attention_weights[sample_idx].detach().cpu().numpy()
            
            # Create attention heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(attn, 
                       xticklabels=src_tokens[:attn.shape[1]], 
                       yticklabels=tgt_tokens[:attn.shape[0]],
                       cmap='Blues', 
                       annot=True, 
                       fmt='.2f')
            plt.title(f'Attention Weights - Step {step}')
            plt.xlabel('Source Tokens')
            plt.ylabel('Target Tokens')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save to TensorBoard
            if self.tensorboard_writer:
                self.tensorboard_writer.add_figure(
                    'attention_weights', plt.gcf(), step
                )
            
            # Save to Weights & Biases
            if self.wandb_run:
                wandb.log({"attention_weights": wandb.Image(plt)}, step=step)
            
            plt.close()
            
        except ImportError:
            self.logger.warning("matplotlib or seaborn not available for attention visualization")
    
    def log_translation_examples(self, src_sentences: list, 
                               tgt_sentences: list, 
                               pred_sentences: list, 
                               step: int):
        """Log translation examples."""
        examples = []
        for i, (src, tgt, pred) in enumerate(zip(src_sentences, tgt_sentences, pred_sentences)):
            examples.append({
                "Source": src,
                "Target": tgt,
                "Prediction": pred
            })
        
        # Log to console
        self.logger.info(f"Translation examples at step {step}:")
        for i, example in enumerate(examples[:3]):  # Log first 3 examples
            self.logger.info(f"  Example {i+1}:")
            self.logger.info(f"    Source: {example['Source']}")
            self.logger.info(f"    Target: {example['Target']}")
            self.logger.info(f"    Prediction: {example['Prediction']}")
        
        # Log to Weights & Biases
        if self.wandb_run:
            wandb.log({"translation_examples": wandb.Table(
                columns=["Source", "Target", "Prediction"],
                data=[[ex["Source"], ex["Target"], ex["Prediction"]] for ex in examples]
            )}, step=step)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters."""
        self.logger.info("Hyperparameters:")
        for key, value in hyperparams.items():
            self.logger.info(f"  {key}: {value}")
        
        if self.wandb_run:
            wandb.config.update(hyperparams)
    
    def close(self):
        """Close all logging backends."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_run:
            wandb.finish()
        
        self.logger.info("Logger closed")


def setup_logging(config, experiment_name: Optional[str] = None) -> Logger:
    """Setup and return a logger instance."""
    return Logger(config, experiment_name)


class TrainingLogger:
    """Specialized logger for training loops."""
    
    def __init__(self, logger: Logger, log_interval: int = 100):
        self.logger = logger
        self.log_interval = log_interval
        self.step = 0
        self.epoch = 0
        self.metrics_buffer = {}
    
    def log_training_step(self, loss: float, learning_rate: float, 
                         grad_norm: Optional[float] = None):
        """Log training step metrics."""
        self.step += 1
        
        # Buffer metrics
        if "loss" not in self.metrics_buffer:
            self.metrics_buffer["loss"] = []
        self.metrics_buffer["loss"].append(loss)
        
        # Log at intervals
        if self.step % self.log_interval == 0:
            avg_loss = sum(self.metrics_buffer["loss"]) / len(self.metrics_buffer["loss"])
            
            metrics = {
                "loss": avg_loss,
                "learning_rate": learning_rate,
            }
            
            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm
            
            self.logger.log_metrics(metrics, self.step, prefix="train")
            self.metrics_buffer.clear()
    
    def log_validation_step(self, metrics: Dict[str, float]):
        """Log validation step metrics."""
        self.logger.log_metrics(metrics, self.step, prefix="val")
    
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], 
                  val_metrics: Dict[str, float]):
        """Log epoch-level metrics."""
        self.epoch = epoch
        
        # Log training metrics
        self.logger.log_metrics(train_metrics, epoch, prefix="epoch/train")
        
        # Log validation metrics
        self.logger.log_metrics(val_metrics, epoch, prefix="epoch/val")
        
        # Log epoch summary
        self.logger.logger.info(
            f"Epoch {epoch} - "
            f"Train Loss: {train_metrics.get('loss', 0):.4f}, "
            f"Val Loss: {val_metrics.get('loss', 0):.4f}, "
            f"Val BLEU: {val_metrics.get('bleu', 0):.4f}"
        ) 