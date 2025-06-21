"""
Tests for neural machine translation models.

This module contains unit tests for the RNN and Transformer models.
"""

import torch
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import RNNModel, TransformerModel
from data.vocabulary import Vocabulary


class TestRNNModel:
    """Test cases for RNN model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.src_len = 10
        self.tgt_len = 8
        self.src_vocab_size = 1000
        self.tgt_vocab_size = 1000
        self.embedding_dim = 256
        self.hidden_dim = 512
        
        # Create vocabularies
        self.src_vocab = Vocabulary()
        self.tgt_vocab = Vocabulary()
        
        # Add some tokens
        for i in range(self.src_vocab_size):
            self.src_vocab.add_token(f"src_token_{i}")
        for i in range(self.tgt_vocab_size):
            self.tgt_vocab.add_token(f"tgt_token_{i}")
    
    def test_rnn_model_creation(self):
        """Test RNN model creation."""
        model = RNNModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        assert model is not None
        assert model.src_vocab_size == self.src_vocab_size
        assert model.tgt_vocab_size == self.tgt_vocab_size
        assert model.embedding_dim == self.embedding_dim
        assert model.hidden_dim == self.hidden_dim
    
    def test_rnn_model_forward(self):
        """Test RNN model forward pass."""
        model = RNNModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Create dummy input
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.src_len))
        tgt = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.tgt_len))
        
        # Forward pass
        output = model(src, tgt)
        
        assert output.shape == (self.batch_size, self.tgt_len, self.tgt_vocab_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_rnn_model_parameter_count(self):
        """Test RNN model parameter counting."""
        model = RNNModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        params = model.count_parameters()
        
        assert 'total' in params
        assert 'trainable' in params
        assert 'encoder' in params
        assert 'decoder' in params
        assert params['total'] > 0
        assert params['trainable'] > 0


class TestTransformerModel:
    """Test cases for Transformer model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.src_len = 10
        self.tgt_len = 8
        self.src_vocab_size = 1000
        self.tgt_vocab_size = 1000
        self.d_model = 512
        self.n_heads = 8
        self.num_layers = 6
        
        # Create vocabularies
        self.src_vocab = Vocabulary()
        self.tgt_vocab = Vocabulary()
        
        # Add some tokens
        for i in range(self.src_vocab_size):
            self.src_vocab.add_token(f"src_token_{i}")
        for i in range(self.tgt_vocab_size):
            self.tgt_vocab.add_token(f"tgt_token_{i}")
    
    def test_transformer_model_creation(self):
        """Test Transformer model creation."""
        model = TransformerModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers
        )
        
        assert model is not None
        assert model.src_vocab_size == self.src_vocab_size
        assert model.tgt_vocab_size == self.tgt_vocab_size
        assert model.d_model == self.d_model
        assert model.n_heads == self.n_heads
        assert model.num_layers == self.num_layers
    
    def test_transformer_model_forward(self):
        """Test Transformer model forward pass."""
        model = TransformerModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers
        )
        
        # Create dummy input
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.src_len))
        tgt = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.tgt_len))
        
        # Forward pass
        output = model(src, tgt)
        
        assert output.shape == (self.batch_size, self.tgt_len, self.tgt_vocab_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_transformer_model_masks(self):
        """Test Transformer model with attention masks."""
        model = TransformerModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers
        )
        
        # Create dummy input
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.src_len))
        tgt = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.tgt_len))
        
        # Create masks
        src_mask = torch.ones(self.batch_size, self.src_len, self.src_len)
        tgt_mask = model.generate_square_subsequent_mask(self.tgt_len).unsqueeze(0).expand(self.batch_size, -1, -1)
        
        # Forward pass with masks
        output = model(src, tgt, src_mask, tgt_mask)
        
        assert output.shape == (self.batch_size, self.tgt_len, self.tgt_vocab_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_transformer_model_parameter_count(self):
        """Test Transformer model parameter counting."""
        model = TransformerModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers
        )
        
        params = model.count_parameters()
        
        assert 'total' in params
        assert 'trainable' in params
        assert 'encoder' in params
        assert 'decoder' in params
        assert params['total'] > 0
        assert params['trainable'] > 0


class TestModelComparison:
    """Test cases comparing RNN and Transformer models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.src_len = 8
        self.tgt_len = 6
        self.src_vocab_size = 500
        self.tgt_vocab_size = 500
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.d_model = 256
        self.n_heads = 4
        self.num_layers = 2
    
    def test_model_output_shapes(self):
        """Test that both models produce correct output shapes."""
        # Create RNN model
        rnn_model = RNNModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # Create Transformer model
        transformer_model = TransformerModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers
        )
        
        # Create dummy input
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.src_len))
        tgt = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.tgt_len))
        
        # Forward pass
        rnn_output = rnn_model(src, tgt)
        transformer_output = transformer_model(src, tgt)
        
        # Check shapes
        expected_shape = (self.batch_size, self.tgt_len, self.tgt_vocab_size)
        assert rnn_output.shape == expected_shape
        assert transformer_output.shape == expected_shape
    
    def test_model_parameter_comparison(self):
        """Test parameter count comparison between models."""
        # Create RNN model
        rnn_model = RNNModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # Create Transformer model
        transformer_model = TransformerModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers
        )
        
        # Get parameter counts
        rnn_params = rnn_model.count_parameters()
        transformer_params = transformer_model.count_parameters()
        
        # Both should have parameters
        assert rnn_params['total'] > 0
        assert transformer_params['total'] > 0
        
        # Transformer typically has more parameters for same vocab sizes
        # (this may not always be true depending on exact configuration)
        print(f"RNN parameters: {rnn_params['total']:,}")
        print(f"Transformer parameters: {transformer_params['total']:,}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 