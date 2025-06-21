"""
Tests for trainer modules.

This module contains unit tests for the trainer, metrics, and training utilities.
"""

import torch
import pytest
import sys
import os
import tempfile
import numpy as np
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trainer.trainer import Trainer
from trainer.metrics import (
    greedy_decode, beam_decode, calculate_bleu_score, 
    calculate_perplexity, calculate_accuracy
)
from models import RNNModel, TransformerModel
from data.vocabulary import Vocabulary
from data.dataset import TranslationDataset, create_data_loader


class TestMetrics:
    """Test cases for evaluation metrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.src_len = 5
        self.tgt_len = 4
        self.src_vocab_size = 100
        self.tgt_vocab_size = 100
        self.device = torch.device('cpu')
        
        # Create vocabularies
        self.src_vocab = Vocabulary()
        self.tgt_vocab = Vocabulary()
        
        # Add some tokens
        for i in range(50):
            self.src_vocab.add_token(f"src_token_{i}")
            self.tgt_vocab.add_token(f"tgt_token_{i}")
    
    def test_greedy_decode(self):
        """Test greedy decoding."""
        # Create a simple model
        model = RNNModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=64,
            hidden_dim=128
        )
        model.eval()
        
        # Create dummy input
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.src_len))
        
        # Test greedy decoding
        with torch.no_grad():
            translation = greedy_decode(
                model=model,
                src=src,
                src_vocab=self.src_vocab,
                tgt_vocab=self.tgt_vocab,
                device=self.device,
                max_length=10
            )
        
        assert isinstance(translation, str)
        assert len(translation) > 0
    
    def test_beam_decode(self):
        """Test beam search decoding."""
        # Create a simple model
        model = RNNModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=64,
            hidden_dim=128
        )
        model.eval()
        
        # Create dummy input
        src = torch.randint(0, self.src_vocab_size, (1, self.src_len))  # Single sample for beam search
        
        # Test beam decoding
        with torch.no_grad():
            translation = beam_decode(
                model=model,
                src=src,
                src_vocab=self.src_vocab,
                tgt_vocab=self.tgt_vocab,
                device=self.device,
                max_length=10,
                beam_size=3
            )
        
        assert isinstance(translation, str)
        assert len(translation) > 0
    
    def test_calculate_bleu_score(self):
        """Test BLEU score calculation."""
        # Test with exact match
        references = [["hello", "world"]]
        candidates = [["hello", "world"]]
        bleu = calculate_bleu_score(candidates, references)
        assert bleu == 1.0  # Perfect match should give BLEU = 1.0
        
        # Test with partial match
        references = [["hello", "world", "test"]]
        candidates = [["hello", "world"]]
        bleu = calculate_bleu_score(candidates, references)
        assert 0.0 < bleu < 1.0  # Partial match should give BLEU < 1.0
        
        # Test with no match
        references = [["hello", "world"]]
        candidates = [["different", "words"]]
        bleu = calculate_bleu_score(candidates, references)
        assert bleu == 0.0  # No match should give BLEU = 0.0
    
    def test_calculate_bleu_score_multiple_references(self):
        """Test BLEU score with multiple references."""
        references = [
            ["hello", "world"],
            ["hi", "world"],
            ["hello", "earth"]
        ]
        candidates = [["hello", "world"]]
        bleu = calculate_bleu_score(candidates, references)
        assert bleu > 0.0  # Should have some score with multiple references
    
    def test_calculate_perplexity(self):
        """Test perplexity calculation."""
        # Create dummy logits and targets
        batch_size = 2
        seq_len = 3
        vocab_size = 10
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Calculate perplexity
        perplexity = calculate_perplexity(logits, targets)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0.0
        assert not np.isnan(perplexity)
        assert not np.isinf(perplexity)
    
    def test_calculate_accuracy(self):
        """Test accuracy calculation."""
        # Create dummy predictions and targets
        batch_size = 4
        seq_len = 5
        vocab_size = 10
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Calculate accuracy
        accuracy = calculate_accuracy(logits, targets)
        
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert not np.isnan(accuracy)
    
    def test_calculate_accuracy_with_padding(self):
        """Test accuracy calculation with padding."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Create padding mask (ignore last token in each sequence)
        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        padding_mask[:, -1] = False
        
        # Calculate accuracy with padding mask
        accuracy = calculate_accuracy(logits, targets, padding_mask)
        
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0


class TestTrainer:
    """Test cases for Trainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.src_len = 5
        self.tgt_len = 4
        self.src_vocab_size = 100
        self.tgt_vocab_size = 100
        self.device = torch.device('cpu')
        
        # Create vocabularies
        self.src_vocab = Vocabulary()
        self.tgt_vocab = Vocabulary()
        
        # Add some tokens
        for i in range(50):
            self.src_vocab.add_token(f"src_token_{i}")
            self.tgt_vocab.add_token(f"tgt_token_{i}")
        
        # Create sample data
        self.src_texts = ["hola mundo", "como estas", "buenos dias"]
        self.tgt_texts = ["hello world", "how are you", "good morning"]
        
        # Build vocabularies
        self.src_vocab.build_from_texts(self.src_texts, min_freq=1)
        self.tgt_vocab.build_from_texts(self.tgt_texts, min_freq=1)
        
        # Create dataset and data loader
        self.dataset = TranslationDataset(
            src_texts=self.src_texts,
            tgt_texts=self.tgt_texts,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab
        )
        
        self.train_loader = create_data_loader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        self.val_loader = create_data_loader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        model = RNNModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=64,
            hidden_dim=128
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tgt_vocab.pad_idx)
        
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device
        )
        
        assert trainer.model == model
        assert trainer.train_loader == self.train_loader
        assert trainer.val_loader == self.val_loader
        assert trainer.optimizer == optimizer
        assert trainer.criterion == criterion
        assert trainer.device == self.device
    
    def test_trainer_train_epoch(self):
        """Test training for one epoch."""
        model = RNNModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=64,
            hidden_dim=128
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tgt_vocab.pad_idx)
        
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device
        )
        
        # Train for one epoch
        train_loss = trainer.train_epoch()
        
        assert isinstance(train_loss, float)
        assert train_loss > 0.0
        assert not np.isnan(train_loss)
        assert not np.isinf(train_loss)
    
    def test_trainer_validate(self):
        """Test validation."""
        model = RNNModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=64,
            hidden_dim=128
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tgt_vocab.pad_idx)
        
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device
        )
        
        # Validate
        val_loss, val_accuracy, val_perplexity = trainer.validate()
        
        assert isinstance(val_loss, float)
        assert isinstance(val_accuracy, float)
        assert isinstance(val_perplexity, float)
        assert val_loss > 0.0
        assert 0.0 <= val_accuracy <= 1.0
        assert val_perplexity > 0.0
        assert not np.isnan(val_loss)
        assert not np.isnan(val_accuracy)
        assert not np.isnan(val_perplexity)
    
    def test_trainer_save_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        model = RNNModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=64,
            hidden_dim=128
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tgt_vocab.pad_idx)
        
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device
        )
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            trainer.save_checkpoint(checkpoint_path, epoch=1, val_loss=0.5)
            
            # Load checkpoint
            loaded_trainer = Trainer.load_checkpoint(
                checkpoint_path, 
                model, 
                optimizer, 
                criterion, 
                self.device
            )
            
            assert loaded_trainer.epoch == 1
            assert loaded_trainer.best_val_loss == 0.5
            assert loaded_trainer.model is not None
            assert loaded_trainer.optimizer is not None
        
        finally:
            # Clean up
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)
    
    def test_trainer_early_stopping(self):
        """Test early stopping functionality."""
        model = RNNModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=64,
            hidden_dim=128
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tgt_vocab.pad_idx)
        
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            patience=2
        )
        
        # Simulate worsening validation loss
        trainer.best_val_loss = 1.0
        
        # Should not stop early
        assert not trainer.should_stop_early(0.9)
        
        # Should stop early after patience exceeded
        assert not trainer.should_stop_early(1.1)  # First bad epoch
        assert not trainer.should_stop_early(1.2)  # Second bad epoch
        assert trainer.should_stop_early(1.3)      # Third bad epoch - should stop
    
    def test_trainer_learning_rate_scheduling(self):
        """Test learning rate scheduling."""
        model = RNNModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=64,
            hidden_dim=128
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tgt_vocab.pad_idx)
        
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=self.device
        )
        
        # Check initial learning rate
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Step scheduler
        trainer.step_scheduler()
        
        # Check that learning rate decreased
        new_lr = optimizer.param_groups[0]['lr']
        assert new_lr == initial_lr * 0.5


class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.src_vocab_size = 100
        self.tgt_vocab_size = 100
        self.device = torch.device('cpu')
        
        # Create vocabularies
        self.src_vocab = Vocabulary()
        self.tgt_vocab = Vocabulary()
        
        # Create sample data
        self.src_texts = [
            "hola mundo",
            "como estas", 
            "buenos dias",
            "gracias por todo"
        ]
        self.tgt_texts = [
            "hello world",
            "how are you",
            "good morning", 
            "thank you for everything"
        ]
        
        # Build vocabularies
        self.src_vocab.build_from_texts(self.src_texts, min_freq=1)
        self.tgt_vocab.build_from_texts(self.tgt_texts, min_freq=1)
        
        # Create dataset and data loader
        self.dataset = TranslationDataset(
            src_texts=self.src_texts,
            tgt_texts=self.tgt_texts,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab
        )
        
        self.train_loader = create_data_loader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        self.val_loader = create_data_loader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def test_rnn_training_pipeline(self):
        """Test complete RNN training pipeline."""
        # Create RNN model
        model = RNNModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=64,
            hidden_dim=128
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tgt_vocab.pad_idx)
        
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device
        )
        
        # Train for a few epochs
        for epoch in range(3):
            train_loss = trainer.train_epoch()
            val_loss, val_accuracy, val_perplexity = trainer.validate()
            
            assert isinstance(train_loss, float)
            assert isinstance(val_loss, float)
            assert isinstance(val_accuracy, float)
            assert isinstance(val_perplexity, float)
            assert train_loss > 0.0
            assert val_loss > 0.0
            assert 0.0 <= val_accuracy <= 1.0
            assert val_perplexity > 0.0
    
    def test_transformer_training_pipeline(self):
        """Test complete Transformer training pipeline."""
        # Create Transformer model
        model = TransformerModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=64,
            n_heads=4,
            num_layers=2
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tgt_vocab.pad_idx)
        
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device
        )
        
        # Train for a few epochs
        for epoch in range(3):
            train_loss = trainer.train_epoch()
            val_loss, val_accuracy, val_perplexity = trainer.validate()
            
            assert isinstance(train_loss, float)
            assert isinstance(val_loss, float)
            assert isinstance(val_accuracy, float)
            assert isinstance(val_perplexity, float)
            assert train_loss > 0.0
            assert val_loss > 0.0
            assert 0.0 <= val_accuracy <= 1.0
            assert val_perplexity > 0.0
    
    def test_training_with_gradient_clipping(self):
        """Test training with gradient clipping."""
        model = RNNModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=64,
            hidden_dim=128
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tgt_vocab.pad_idx)
        
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            max_grad_norm=1.0
        )
        
        # Train for one epoch with gradient clipping
        train_loss = trainer.train_epoch()
        
        assert isinstance(train_loss, float)
        assert train_loss > 0.0
        assert not np.isnan(train_loss)
        assert not np.isinf(train_loss)


class TestTrainingUtilities:
    """Test cases for training utility functions."""
    
    def test_model_parameter_counting(self):
        """Test model parameter counting."""
        model = RNNModel(
            src_vocab_size=100,
            tgt_vocab_size=100,
            embedding_dim=64,
            hidden_dim=128
        )
        
        params = model.count_parameters()
        
        assert 'total' in params
        assert 'trainable' in params
        assert 'encoder' in params
        assert 'decoder' in params
        assert params['total'] > 0
        assert params['trainable'] > 0
        assert params['encoder'] > 0
        assert params['decoder'] > 0
    
    def test_model_device_management(self):
        """Test model device management."""
        model = RNNModel(
            src_vocab_size=100,
            tgt_vocab_size=100,
            embedding_dim=64,
            hidden_dim=128
        )
        
        # Test moving to CPU
        model.to('cpu')
        assert next(model.parameters()).device.type == 'cpu'
        
        # Test moving to CUDA if available
        if torch.cuda.is_available():
            model.to('cuda')
            assert next(model.parameters()).device.type == 'cuda'
    
    def test_optimizer_state_management(self):
        """Test optimizer state management."""
        model = RNNModel(
            src_vocab_size=100,
            tgt_vocab_size=100,
            embedding_dim=64,
            hidden_dim=128
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Test optimizer state
        assert len(optimizer.state_dict()['state']) > 0
        assert 'param_groups' in optimizer.state_dict() 