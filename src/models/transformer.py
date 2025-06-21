"""
Transformer-based neural machine translation model.

This module implements a Transformer encoder-decoder architecture
for neural machine translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math

from .attention import MultiHeadAttention, PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer.
    
    This layer consists of multi-head self-attention followed by
    a feed-forward network with residual connections and layer normalization.
    """
    
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, 
                 dropout: float = 0.1):
        """
        Initialize transformer encoder layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dim_feedforward: Dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of encoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.
    
    This layer consists of masked multi-head self-attention, multi-head
    cross-attention, and a feed-forward network with residual connections.
    """
    
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, 
                 dropout: float = 0.1):
        """
        Initialize transformer decoder layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dim_feedforward: Dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Masked multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Multi-head cross-attention
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of decoder layer.
        
        Args:
            x: Input tensor [batch_size, tgt_len, d_model]
            enc_output: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Target attention mask [batch_size, tgt_len, tgt_len]
            src_mask: Source attention mask [batch_size, tgt_len, src_len]
            
        Returns:
            Output tensor [batch_size, tgt_len, d_model]
        """
        # Masked self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with residual connection
        cross_output, _ = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder.
    
    This encoder consists of multiple transformer encoder layers
    and processes the source sequence.
    """
    
    def __init__(self, src_vocab_size: int, d_model: int, n_heads: int, 
                 num_layers: int, dim_feedforward: int, max_position_embeddings: int = 5000,
                 dropout: float = 0.1):
        """
        Initialize transformer encoder.
        
        Args:
            src_vocab_size: Size of source vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: Dimension of feed-forward networks
            max_position_embeddings: Maximum sequence length for positional encoding
            dropout: Dropout probability
        """
        super().__init__()
        
        self.src_vocab_size = src_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        
        # Word embeddings
        self.embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_position_embeddings, dropout)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of encoder.
        
        Args:
            src: Source sequences [batch_size, src_len]
            src_mask: Attention mask [batch_size, src_len, src_len]
            
        Returns:
            Encoder output [batch_size, src_len, d_model]
        """
        # Create embeddings
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder.
    
    This decoder consists of multiple transformer decoder layers
    and generates the target sequence.
    """
    
    def __init__(self, tgt_vocab_size: int, d_model: int, n_heads: int,
                 num_layers: int, dim_feedforward: int, max_position_embeddings: int = 5000,
                 dropout: float = 0.1):
        """
        Initialize transformer decoder.
        
        Args:
            tgt_vocab_size: Size of target vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            num_layers: Number of decoder layers
            dim_feedforward: Dimension of feed-forward networks
            max_position_embeddings: Maximum sequence length for positional encoding
            dropout: Dropout probability
        """
        super().__init__()
        
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        
        # Word embeddings
        self.embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_position_embeddings, dropout)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, tgt: torch.Tensor, enc_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of decoder.
        
        Args:
            tgt: Target sequences [batch_size, tgt_len]
            enc_output: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Target attention mask [batch_size, tgt_len, tgt_len]
            src_mask: Source attention mask [batch_size, tgt_len, src_len]
            
        Returns:
            Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # Create embeddings
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        
        # Project to vocabulary
        output = self.output_projection(x)
        
        return output


class TransformerModel(nn.Module):
    """
    Complete Transformer-based neural machine translation model.
    
    This model combines a Transformer encoder and decoder for
    end-to-end neural machine translation.
    """
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 n_heads: int = 8, num_layers: int = 6, dim_feedforward: int = 2048,
                 max_position_embeddings: int = 5000, dropout: float = 0.1):
        """
        Initialize Transformer model.
        
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            num_layers: Number of encoder/decoder layers
            dim_feedforward: Dimension of feed-forward networks
            max_position_embeddings: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = TransformerEncoder(
            src_vocab_size=src_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the complete model.
        
        Args:
            src: Source sequences [batch_size, src_len]
            tgt: Target sequences [batch_size, tgt_len]
            src_mask: Source attention mask [batch_size, src_len, src_len]
            tgt_mask: Target attention mask [batch_size, tgt_len, tgt_len]
            
        Returns:
            Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # Encode source sequence
        enc_output = self.encoder(src, src_mask)
        
        # Decode target sequence
        output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        
        return output
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src: Source sequences [batch_size, src_len]
            src_mask: Source attention mask [batch_size, src_len, src_len]
            
        Returns:
            Encoder output [batch_size, src_len, d_model]
        """
        return self.encoder(src, src_mask)
    
    def decode(self, tgt: torch.Tensor, enc_output: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode target sequence.
        
        Args:
            tgt: Target sequences [batch_size, tgt_len]
            enc_output: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Target attention mask [batch_size, tgt_len, tgt_len]
            src_mask: Source attention mask [batch_size, tgt_len, src_len]
            
        Returns:
            Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        return self.decoder(tgt, enc_output, tgt_mask, src_mask)
    
    def generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """
        Generate square subsequent mask for decoder.
        
        Args:
            size: Size of the mask
            
        Returns:
            Subsequent mask [size, size]
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'encoder': sum(p.numel() for p in self.encoder.parameters()),
            'decoder': sum(p.numel() for p in self.decoder.parameters())
        } 