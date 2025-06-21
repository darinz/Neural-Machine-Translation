"""
Attention mechanisms for neural machine translation.

This module implements various attention mechanisms including:
- Bahdanau attention (additive attention)
- Luong attention (multiplicative attention)
- Multi-head attention (as used in Transformers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class Attention(nn.Module):
    """
    Base attention mechanism class.
    
    This class provides a common interface for different attention mechanisms
    and implements the basic attention computation.
    """
    
    def __init__(self, hidden_dim: int, attention_dim: Optional[int] = None):
        """
        Initialize attention mechanism.
        
        Args:
            hidden_dim: Dimension of hidden states
            attention_dim: Dimension of attention space (default: hidden_dim)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim or hidden_dim
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention.
        
        Args:
            query: Query tensor [batch_size, query_len, hidden_dim]
            keys: Key tensor [batch_size, key_len, hidden_dim]
            values: Value tensor [batch_size, key_len, value_dim]
            mask: Attention mask [batch_size, query_len, key_len]
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        raise NotImplementedError("Subclasses must implement forward method")


class BahdanauAttention(Attention):
    """
    Bahdanau attention (additive attention).
    
    This attention mechanism computes attention scores using a feed-forward network
    that takes the concatenation of query and key as input.
    """
    
    def __init__(self, hidden_dim: int, attention_dim: Optional[int] = None):
        """
        Initialize Bahdanau attention.
        
        Args:
            hidden_dim: Dimension of hidden states
            attention_dim: Dimension of attention space
        """
        super().__init__(hidden_dim, attention_dim)
        
        # Attention scoring network
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Bahdanau attention.
        
        Args:
            query: Query tensor [batch_size, query_len, hidden_dim]
            keys: Key tensor [batch_size, key_len, hidden_dim]
            values: Value tensor [batch_size, key_len, value_dim]
            mask: Attention mask [batch_size, query_len, key_len]
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        batch_size, query_len, hidden_dim = query.size()
        key_len = keys.size(1)
        
        # Expand query and keys for attention computation
        # query: [batch_size, query_len, 1, hidden_dim]
        # keys: [batch_size, 1, key_len, hidden_dim]
        query_expanded = query.unsqueeze(2)
        keys_expanded = keys.unsqueeze(1)
        
        # Concatenate query and keys
        # [batch_size, query_len, key_len, hidden_dim * 2]
        query_key_concat = torch.cat([query_expanded, keys_expanded], dim=-1)
        
        # Compute attention scores
        # [batch_size, query_len, key_len, 1]
        attention_scores = self.attention_net(query_key_concat)
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, query_len, key_len]
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute context vector
        # [batch_size, query_len, value_dim]
        context_vector = torch.bmm(attention_weights, values)
        
        return context_vector, attention_weights


class LuongAttention(Attention):
    """
    Luong attention (multiplicative attention).
    
    This attention mechanism computes attention scores using a simple dot product
    between query and key, optionally with a scaling factor.
    """
    
    def __init__(self, hidden_dim: int, attention_type: str = 'dot', 
                 attention_dim: Optional[int] = None):
        """
        Initialize Luong attention.
        
        Args:
            hidden_dim: Dimension of hidden states
            attention_type: Type of attention ('dot', 'general', 'concat')
            attention_dim: Dimension of attention space (for 'general' type)
        """
        super().__init__(hidden_dim, attention_dim)
        self.attention_type = attention_type
        
        if attention_type == 'general':
            self.attention_net = nn.Linear(hidden_dim, attention_dim)
        elif attention_type == 'concat':
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, attention_dim),
                nn.Tanh(),
                nn.Linear(attention_dim, 1)
            )
        else:  # dot
            self.attention_net = None
            
    def forward(self, query: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Luong attention.
        
        Args:
            query: Query tensor [batch_size, query_len, hidden_dim]
            keys: Key tensor [batch_size, key_len, hidden_dim]
            values: Value tensor [batch_size, key_len, value_dim]
            mask: Attention mask [batch_size, query_len, key_len]
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        if self.attention_type == 'dot':
            # Simple dot product attention
            attention_scores = torch.bmm(query, keys.transpose(1, 2))
        elif self.attention_type == 'general':
            # General attention with learned transformation
            transformed_keys = self.attention_net(keys)
            attention_scores = torch.bmm(query, transformed_keys.transpose(1, 2))
        elif self.attention_type == 'concat':
            # Concat attention
            batch_size, query_len, hidden_dim = query.size()
            key_len = keys.size(1)
            
            # Expand dimensions for concatenation
            query_expanded = query.unsqueeze(2).expand(-1, -1, key_len, -1)
            keys_expanded = keys.unsqueeze(1).expand(-1, query_len, -1, -1)
            
            # Concatenate and compute scores
            query_key_concat = torch.cat([query_expanded, keys_expanded], dim=-1)
            attention_scores = self.attention_net(query_key_concat).squeeze(-1)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute context vector
        context_vector = torch.bmm(attention_weights, values)
        
        return context_vector, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism as used in Transformers.
    
    This attention mechanism allows the model to jointly attend to information
    from different representation subspaces at different positions.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.w_o(context)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    
    This module adds positional information to input embeddings using
    sine and cosine functions of different frequencies.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            
        Returns:
            Embeddings with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create padding mask for attention.
    
    Args:
        seq: Input sequence [batch_size, seq_len]
        pad_idx: Padding token index
        
    Returns:
        Padding mask [batch_size, 1, 1, seq_len] where True indicates padding
    """
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(size: int) -> torch.Tensor:
    """
    Create look-ahead mask for decoder self-attention.
    
    Args:
        size: Sequence length
        
    Returns:
        Look-ahead mask [size, size] where True indicates positions to mask
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask


def create_combined_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create combined mask for decoder (padding + look-ahead).
    
    Args:
        seq: Input sequence [batch_size, seq_len]
        pad_idx: Padding token index
        
    Returns:
        Combined mask [batch_size, 1, seq_len, seq_len]
    """
    seq_len = seq.size(1)
    
    # Create padding mask
    padding_mask = create_padding_mask(seq, pad_idx)
    
    # Create look-ahead mask
    look_ahead_mask = create_look_ahead_mask(seq_len).unsqueeze(0).unsqueeze(0)
    
    # Combine masks
    combined_mask = padding_mask | look_ahead_mask
    
    return combined_mask 