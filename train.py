#!/usr/bin/env python3
"""
Training script for neural machine translation models.

This script provides a command-line interface for training both RNN and Transformer
models with comprehensive logging, checkpointing, and evaluation.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import load_config, Config
from src.utils.logging import setup_logging, TrainingLogger
from src.data.preprocessing import load_translation_data, split_data, build_vocabularies
from src.data.dataset import TranslationDataset, create_data_loaders
from src.models import RNNModel, TransformerModel
from src.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train neural machine translation models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["rnn", "transformer"], 
        default="transformer",
        help="Model architecture to train"
    )
    
    # Data arguments
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="data/spa.txt",
        help="Path to training data file"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=50000,
        help="Maximum number of samples to use"
    )
    parser.add_argument(
        "--src-lang", 
        type=str, 
        default="es",
        help="Source language code"
    )
    parser.add_argument(
        "--tgt-lang", 
        type=str, 
        default="en",
        help="Target language code"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup-steps", 
        type=int, 
        default=4000,
        help="Number of warmup steps for learning rate scheduler"
    )
    
    # Model configuration
    parser.add_argument(
        "--embedding-dim", 
        type=int, 
        default=256,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--hidden-dim", 
        type=int, 
        default=512,
        help="Hidden dimension (for RNN)"
    )
    parser.add_argument(
        "--d-model", 
        type=int, 
        default=512,
        help="Model dimension (for Transformer)"
    )
    parser.add_argument(
        "--n-heads", 
        type=int, 
        default=8,
        help="Number of attention heads (for Transformer)"
    )
    parser.add_argument(
        "--n-layers", 
        type=int, 
        default=6,
        help="Number of layers"
    )
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=0.1,
        help="Dropout probability"
    )
    
    # Logging and saving
    parser.add_argument(
        "--experiment-name", 
        type=str, 
        default=None,
        help="Experiment name for logging"
    )
    parser.add_argument(
        "--checkpoint-dir", 
        type=str, 
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="logs",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--save-interval", 
        type=int, 
        default=1,
        help="Save checkpoint every N epochs"
    )
    
    # Other arguments
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=4,
        help="Number of data loader workers"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Setup training environment."""
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def load_data(args, config):
    """Load and preprocess training data."""
    print("Loading and preprocessing data...")
    
    # Load raw data
    df = load_translation_data(
        data_path=args.data_path,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        max_samples=args.max_samples,
        lowercase=config.data.lowercase,
        remove_punctuation=config.data.remove_punctuation
    )
    
    print(f"Loaded {len(df)} sentence pairs")
    
    # Split data
    train_df, val_df, test_df = split_data(
        df, 
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Build vocabularies
    src_vocab, tgt_vocab = build_vocabularies(
        train_df, 
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        min_freq=config.data.min_freq
    )
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Create datasets
    from src.data.preprocessing import preprocess_data_to_tensor
    
    train_src, train_tgt, max_src_len, max_tgt_len = preprocess_data_to_tensor(
        train_df, src_vocab, tgt_vocab, config.data.max_length
    )
    val_src, val_tgt, _, _ = preprocess_data_to_tensor(
        val_df, src_vocab, tgt_vocab, config.data.max_length
    )
    test_src, test_tgt, _, _ = preprocess_data_to_tensor(
        test_df, src_vocab, tgt_vocab, config.data.max_length
    )
    
    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab)
    test_dataset = TranslationDataset(test_src, test_tgt, src_vocab, tgt_vocab)
    
    # Create data loaders
    train_loader = create_data_loaders(
        train_src, train_tgt,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = create_data_loaders(
        val_src, val_tgt,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = create_data_loaders(
        test_src, test_tgt,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'max_src_len': max_src_len,
        'max_tgt_len': max_tgt_len
    }


def create_model(args, data_info, device):
    """Create model based on architecture."""
    print(f"Creating {args.model} model...")
    
    src_vocab_size = len(data_info['src_vocab'])
    tgt_vocab_size = len(data_info['tgt_vocab'])
    
    if args.model == "rnn":
        model = RNNModel(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.n_layers,
            dropout=args.dropout,
            bidirectional=True
        )
    elif args.model == "transformer":
        model = TransformerModel(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            num_layers=args.n_layers,
            dim_feedforward=args.d_model * 4,
            dropout=args.dropout,
            max_position_embeddings=5000
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    model = model.to(device)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.warmup_steps = args.warmup_steps
    config.device = args.device
    config.num_workers = args.num_workers
    config.checkpoint_dir = args.checkpoint_dir
    config.logging.log_dir = args.log_dir
    
    # Setup environment
    device = setup_environment(args)
    config.device = str(device)
    
    # Setup logging
    experiment_name = args.experiment_name or f"{args.model}_{int(time.time())}"
    logger = setup_logging(config, experiment_name)
    
    # Log hyperparameters
    hyperparams = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'seed': args.seed
    }
    logger.log_hyperparameters(hyperparams)
    
    try:
        # Load data
        data_info = load_data(args, config)
        
        # Create model
        model = create_model(args, data_info, device)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=config,
            device=device,
            logger=logger
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Train model
        print("Starting training...")
        trainer.train(
            train_loader=data_info['train_loader'],
            val_loader=data_info['val_loader'],
            epochs=args.epochs,
            save_interval=args.save_interval
        )
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_metrics = trainer.evaluate(data_info['test_loader'])
        print(f"Test BLEU: {test_metrics['bleu']:.4f}")
        
        # Save final model
        final_checkpoint_path = Path(args.checkpoint_dir) / f"{experiment_name}_final.pt"
        trainer.save_checkpoint(str(final_checkpoint_path))
        print(f"Final model saved to: {final_checkpoint_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main() 