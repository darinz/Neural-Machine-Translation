#!/usr/bin/env python3
"""
Quick start example for Neural Machine Translation.

This script demonstrates how to use the project for training and inference.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.utils.logging import setup_logging
from src.data.preprocessing import (
    load_translation_data, 
    split_data, 
    build_vocabularies,
    preprocess_data_to_tensor
)
from src.data.dataset import TranslationDataset, create_data_loaders
from src.models import RNNModel, TransformerModel
from src.trainer import Trainer


def main():
    """Quick start example."""
    print("🚀 Neural Machine Translation - Quick Start")
    print("=" * 50)
    
    # Setup configuration
    config = Config()
    config.training.epochs = 2  # Quick training for demo
    config.training.batch_size = 32
    config.data.max_length = 30
    config.data.min_freq = 2
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup logging
    logger = setup_logging(config, "quick_start_demo")
    
    try:
        # Load data (assuming spa.txt is in data/ directory)
        data_path = Path("data/spa.txt")
        if not data_path.exists():
            print("❌ Data file not found. Please download the Spanish-English dataset:")
            print("   wget http://www.manythings.org/anki/spa-eng.zip")
            print("   unzip spa-eng.zip -d data/")
            return
        
        print("📊 Loading data...")
        df = load_translation_data(
            data_path=str(data_path),
            max_samples=10000,  # Use smaller dataset for demo
            lowercase=True,
            remove_punctuation=False
        )
        
        print(f"   Loaded {len(df)} sentence pairs")
        
        # Split data
        train_df, val_df, test_df = split_data(df, 0.8, 0.1, 0.1)
        print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Build vocabularies
        print("🔤 Building vocabularies...")
        src_vocab, tgt_vocab = build_vocabularies(
            train_df, 
            src_lang="es",
            tgt_lang="en",
            min_freq=config.data.min_freq
        )
        print(f"   Source vocabulary size: {len(src_vocab)}")
        print(f"   Target vocabulary size: {len(tgt_vocab)}")
        
        # Create datasets
        print("📦 Creating datasets...")
        train_src, train_tgt, max_src_len, max_tgt_len = preprocess_data_to_tensor(
            train_df, src_vocab, tgt_vocab, config.data.max_length
        )
        val_src, val_tgt, _, _ = preprocess_data_to_tensor(
            val_df, src_vocab, tgt_vocab, config.data.max_length
        )
        
        train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab)
        val_dataset = TranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab)
        
        # Create data loaders
        train_loader = create_data_loaders(
            train_src, train_tgt,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = create_data_loaders(
            val_src, val_tgt,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Create model
        print("🤖 Creating model...")
        model = TransformerModel(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=256,  # Smaller model for demo
            n_heads=8,
            num_layers=3,
            dim_feedforward=1024,
            dropout=0.1,
            max_position_embeddings=1000
        )
        model = model.to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Model parameters: {total_params:,}")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=config,
            device=device,
            logger=logger
        )
        
        # Train model
        print("🎯 Starting training...")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.training.epochs,
            save_interval=1
        )
        
        # Test translation
        print("🌐 Testing translation...")
        test_sentences = [
            "Hola, ¿cómo estás?",
            "Me gusta el café",
            "¿Dónde está la biblioteca?",
            "Buenos días",
            "Gracias por tu ayuda"
        ]
        
        for sentence in test_sentences:
            translation = trainer.translate(sentence, src_vocab, tgt_vocab)
            print(f"   '{sentence}' → '{translation}'")
        
        print("\n✅ Quick start completed successfully!")
        print("📁 Check the 'checkpoints' and 'logs' directories for outputs.")
        
    except Exception as e:
        print(f"❌ Error during quick start: {e}")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main() 