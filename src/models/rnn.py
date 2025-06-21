"""
RNN-based neural machine translation model.

This module implements an RNN encoder-decoder with attention mechanism
for neural machine translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math

from .attention import BahdanauAttention


class RNNEncoder(nn.Module):
    """
    RNN Encoder with bidirectional GRU.
    
    This encoder processes the source sequence and produces hidden states
    that will be used by the decoder with attention.
    """
    
    def __init__(self, src_vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.1, bidirectional: bool = True):
        """
        Initialize RNN encoder.
        
        Args:
            src_vocab_size: Size of source vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden states
            num_layers: Number of RNN layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional RNN
        """
        super().__init__()
        
        self.src_vocab_size = src_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Word embeddings
        self.embedding = nn.Embedding(src_vocab_size, embedding_dim, padding_idx=0)
        
        # RNN layers
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Projection layer for attention (if bidirectional)
        if bidirectional:
            self.attention_projection = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, src: torch.Tensor, src_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.
        
        Args:
            src: Source sequences [batch_size, src_len]
            src_lengths: Length of each sequence [batch_size]
            
        Returns:
            Tuple of (outputs, hidden_state)
            - outputs: [batch_size, src_len, hidden_dim * num_directions]
            - hidden_state: [num_layers * num_directions, batch_size, hidden_dim]
        """
        batch_size, src_len = src.size()
        
        # Create embeddings
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, embedding_dim]
        
        # Pack sequences if lengths are provided
        if src_lengths is not None:
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_outputs, hidden = self.rnn(packed_embedded)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            outputs, hidden = self.rnn(embedded)
        
        # Project outputs for attention if bidirectional
        if self.bidirectional:
            outputs = self.attention_projection(outputs)
        
        return outputs, hidden
    
    def get_hidden_for_decoder(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Prepare hidden state for decoder initialization.
        
        Args:
            hidden: Hidden state from encoder [num_layers * num_directions, batch_size, hidden_dim]
            
        Returns:
            Hidden state for decoder [num_layers, batch_size, hidden_dim]
        """
        if self.bidirectional:
            # Sum bidirectional states
            hidden = hidden.view(self.num_layers, self.num_directions, -1, self.hidden_dim)
            hidden = hidden.sum(dim=1)  # [num_layers, batch_size, hidden_dim]
        
        return hidden


class RNNDecoder(nn.Module):
    """
    RNN Decoder with attention mechanism.
    
    This decoder generates the target sequence one token at a time,
    using attention over the encoder outputs.
    """
    
    def __init__(self, tgt_vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.1, attention_dim: Optional[int] = None):
        """
        Initialize RNN decoder.
        
        Args:
            tgt_vocab_size: Size of target vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden states
            num_layers: Number of RNN layers
            dropout: Dropout probability
            attention_dim: Dimension of attention space
        """
        super().__init__()
        
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Word embeddings
        self.embedding = nn.Embedding(tgt_vocab_size, embedding_dim, padding_idx=0)
        
        # Attention mechanism
        self.attention = BahdanauAttention(hidden_dim, attention_dim or hidden_dim)
        
        # RNN layers
        self.rnn = nn.GRU(
            input_size=embedding_dim + hidden_dim,  # embedding + context vector
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt: torch.Tensor, hidden: torch.Tensor, 
                encoder_outputs: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            tgt: Target sequences [batch_size, tgt_len]
            hidden: Initial hidden state [num_layers, batch_size, hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, hidden_dim]
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        batch_size, tgt_len = tgt.size()
        device = tgt.device
        
        # Initialize outputs
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=device)
        
        # Get start token
        start_token = tgt[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # Decode step by step
        for t in range(tgt_len):
            # Use teacher forcing or previous prediction
            if t == 0 or torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = tgt[:, t].unsqueeze(1)  # [batch_size, 1]
            else:
                decoder_input = torch.argmax(outputs[:, t-1], dim=1).unsqueeze(1)
            
            # Forward step
            output, hidden = self.forward_step(decoder_input, hidden, encoder_outputs)
            outputs[:, t] = output.squeeze(1)
        
        return outputs
    
    def forward_step(self, decoder_input: torch.Tensor, hidden: torch.Tensor,
                    encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single decoding step.
        
        Args:
            decoder_input: Input token [batch_size, 1]
            hidden: Current hidden state [num_layers, batch_size, hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, hidden_dim]
            
        Returns:
            Tuple of (output, new_hidden)
            - output: [batch_size, 1, tgt_vocab_size]
            - new_hidden: [num_layers, batch_size, hidden_dim]
        """
        batch_size = decoder_input.size(0)
        
        # Get embedding
        embedded = self.dropout(self.embedding(decoder_input))  # [batch_size, 1, embedding_dim]
        
        # Get attention context
        # Use the last layer's hidden state for attention
        last_hidden = hidden[-1].unsqueeze(1)  # [batch_size, 1, hidden_dim]
        context, attention_weights = self.attention(last_hidden, encoder_outputs, encoder_outputs)
        
        # Combine embedding and context
        rnn_input = torch.cat([embedded, context], dim=2)  # [batch_size, 1, embedding_dim + hidden_dim]
        
        # RNN step
        output, new_hidden = self.rnn(rnn_input, hidden)
        
        # Project to vocabulary
        output = self.output_projection(output)  # [batch_size, 1, tgt_vocab_size]
        
        return output, new_hidden


class RNNModel(nn.Module):
    """
    Complete RNN-based neural machine translation model.
    
    This model combines an RNN encoder and decoder with attention mechanism
    for end-to-end neural machine translation.
    """
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embedding_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.1,
                 bidirectional: bool = True, attention_dim: Optional[int] = None):
        """
        Initialize RNN model.
        
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden states
            num_layers: Number of RNN layers
            dropout: Dropout probability
            bidirectional: Whether encoder is bidirectional
            attention_dim: Dimension of attention space
        """
        super().__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Encoder
        self.encoder = RNNEncoder(
            src_vocab_size=src_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Decoder
        self.decoder = RNNDecoder(
            tgt_vocab_size=tgt_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_dim=attention_dim
        )
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_lengths: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass of the complete model.
        
        Args:
            src: Source sequences [batch_size, src_len]
            tgt: Target sequences [batch_size, tgt_len]
            src_lengths: Length of source sequences [batch_size]
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # Encode source sequence
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
        
        # Prepare hidden state for decoder
        decoder_hidden = self.encoder.get_hidden_for_decoder(encoder_hidden)
        
        # Decode target sequence
        outputs = self.decoder(tgt, decoder_hidden, encoder_outputs, teacher_forcing_ratio)
        
        return outputs
    
    def encode(self, src: torch.Tensor, src_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode source sequence.
        
        Args:
            src: Source sequences [batch_size, src_len]
            src_lengths: Length of source sequences [batch_size]
            
        Returns:
            Tuple of (encoder_outputs, encoder_hidden)
        """
        return self.encoder(src, src_lengths)
    
    def decode(self, tgt: torch.Tensor, hidden: torch.Tensor, 
               encoder_outputs: torch.Tensor, teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        """
        Decode target sequence.
        
        Args:
            tgt: Target sequences [batch_size, tgt_len]
            hidden: Initial hidden state [num_layers, batch_size, hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, hidden_dim]
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        return self.decoder(tgt, hidden, encoder_outputs, teacher_forcing_ratio)
    
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