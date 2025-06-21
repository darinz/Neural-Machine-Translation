"""
Training utilities for neural machine translation models.

This module provides a comprehensive training loop with logging,
checkpointing, and evaluation capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
from typing import Dict, Any, Optional, Callable
import json
from pathlib import Path

from ..utils.logging import get_logger
from .metrics import compute_metrics, evaluate_translations


class Trainer:
    """
    Trainer class for neural machine translation models.
    
    This class handles the training loop, validation, checkpointing,
    and logging for both RNN and Transformer models.
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 optimizer: optim.Optimizer, criterion: nn.Module,
                 device: torch.device, config: Dict[str, Any],
                 src_vocab=None, tgt_vocab=None):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            config: Training configuration
            src_vocab: Source vocabulary (for evaluation)
            tgt_vocab: Target vocabulary (for evaluation)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_bleu_score = 0.0
        self.patience_counter = 0
        
        # Setup logging
        self.logger = get_logger(__name__)
        
        # Setup scheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training history
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'perplexity': []
        }
        self.val_history = {
            'loss': [],
            'accuracy': [],
            'perplexity': [],
            'bleu_scores': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # Move data to device
            src = batch['src_seq'].to(self.device)
            tgt = batch['tgt_seq'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'forward'):
                outputs = self.model(src, tgt)
            else:
                # Handle encoder-decoder models
                enc_output = self.model.encode(src)
                outputs = self.model.decode(tgt, enc_output)
            
            # Compute loss
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                tgt.view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                metrics = compute_metrics(outputs, tgt, ignore_index=self.tgt_vocab.pad_idx if self.tgt_vocab else 0)
            
            total_loss += loss.item()
            total_accuracy += metrics['accuracy']
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            if batch_idx % self.config.get('log_interval', 100) == 0:
                batch_time = time.time() - batch_start_time
                self.logger.info(
                    f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {loss.item():.4f}, Accuracy: {metrics["accuracy"]:.4f}, '
                    f'Time: {batch_time:.2f}s'
                )
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        epoch_time = time.time() - epoch_start_time
        
        self.logger.info(
            f'Epoch {self.current_epoch} completed in {epoch_time:.2f}s. '
            f'Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}, '
            f'Avg Perplexity: {avg_perplexity:.4f}'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'perplexity': avg_perplexity
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                src = batch['src_seq'].to(self.device)
                tgt = batch['tgt_seq'].to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward'):
                    outputs = self.model(src, tgt)
                else:
                    # Handle encoder-decoder models
                    enc_output = self.model.encode(src)
                    outputs = self.model.decode(tgt, enc_output)
                
                # Compute loss
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    tgt.view(-1)
                )
                
                # Compute metrics
                metrics = compute_metrics(outputs, tgt, ignore_index=self.tgt_vocab.pad_idx if self.tgt_vocab else 0)
                
                total_loss += loss.item()
                total_accuracy += metrics['accuracy']
                num_batches += 1
        
        # Compute average metrics
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Compute BLEU scores if vocabularies are available
        bleu_scores = {}
        if self.src_vocab and self.tgt_vocab:
            try:
                eval_results = evaluate_translations(
                    self.model, self.val_loader, self.src_vocab, self.tgt_vocab,
                    self.device, max_length=self.config.get('max_length', 50),
                    beam_size=1
                )
                bleu_scores = eval_results['bleu_scores']
            except Exception as e:
                self.logger.warning(f"Failed to compute BLEU scores: {e}")
        
        self.logger.info(
            f'Validation - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, '
            f'Perplexity: {avg_perplexity:.4f}'
        )
        
        if bleu_scores:
            self.logger.info(f'BLEU Scores: {bleu_scores}')
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'perplexity': avg_perplexity,
            'bleu_scores': bleu_scores
        }
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_bleu_score': self.best_bleu_score,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f'Saved best model to {best_path}')
        
        # Keep only recent checkpoints
        self._cleanup_old_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_bleu_score = checkpoint['best_bleu_score']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        
        self.logger.info(f'Loaded checkpoint from {checkpoint_path}')
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None) -> None:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        if resume_from:
            self.load_checkpoint(resume_from)
        
        self.logger.info(f'Starting training for {num_epochs} epochs')
        self.logger.info(f'Model parameters: {sum(p.numel() for p in self.model.parameters()):,}')
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Store history
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['accuracy'].append(train_metrics['accuracy'])
            self.train_history['perplexity'].append(train_metrics['perplexity'])
            
            self.val_history['loss'].append(val_metrics['loss'])
            self.val_history['accuracy'].append(val_metrics['accuracy'])
            self.val_history['perplexity'].append(val_metrics['perplexity'])
            self.val_history['bleu_scores'].append(val_metrics.get('bleu_scores', {}))
            
            # Check if best model
            is_best = False
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                is_best = True
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Check BLEU score improvement
            if 'bleu_scores' in val_metrics and val_metrics['bleu_scores']:
                current_bleu = val_metrics['bleu_scores'].get('bleu_4', 0.0)
                if current_bleu > self.best_bleu_score:
                    self.best_bleu_score = current_bleu
                    is_best = True
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 10):
                self.logger.info(f'Early stopping after {self.patience_counter} epochs without improvement')
                break
        
        self.logger.info('Training completed')
    
    def _cleanup_old_checkpoints(self, keep_last: int = 5) -> None:
        """
        Remove old checkpoint files.
        
        Args:
            keep_last: Number of recent checkpoints to keep
        """
        checkpoint_files = sorted(
            [f for f in self.checkpoint_dir.glob('checkpoint_epoch_*.pt')],
            key=lambda x: int(x.stem.split('_')[-1])
        )
        
        if len(checkpoint_files) > keep_last:
            for checkpoint_file in checkpoint_files[:-keep_last]:
                checkpoint_file.unlink()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get training summary.
        
        Returns:
            Dictionary with training summary
        """
        return {
            'total_epochs': self.current_epoch,
            'global_steps': self.global_step,
            'best_val_loss': self.best_val_loss,
            'best_bleu_score': self.best_bleu_score,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def save_training_summary(self, output_path: str) -> None:
        """
        Save training summary to file.
        
        Args:
            output_path: Path to save summary
        """
        summary = self.get_training_summary()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f'Saved training summary to {output_path}') 