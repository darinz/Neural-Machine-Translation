#!/usr/bin/env python3
"""
02_model_comparison.py
Model Comparison Notebook for Neural Machine Translation

This notebook compares RNN and Transformer models for Spanish-English translation,
analyzing their architectures, training dynamics, and performance characteristics.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import defaultdict
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import RNNModel, TransformerModel
from data import TranslationDataset, Vocabulary, preprocess_sentence, build_vocabulary
from trainer import Trainer

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("=== Neural Machine Translation - Model Comparison ===\n")

# 1. Model Architecture Analysis
print("1. Model Architecture Analysis")
print("-" * 50)

def analyze_model_architecture():
    """Analyze and compare model architectures"""
    
    # Model parameters
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    embedding_dim = 256
    hidden_dim = 512
    d_model = 512
    nhead = 8
    num_layers = 6
    
    # Create models
    rnn_model = RNNModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.1
    )
    
    transformer_model = TransformerModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=2048,
        dropout=0.1
    )
    
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    rnn_params = count_parameters(rnn_model)
    transformer_params = count_parameters(transformer_model)
    
    print(f"RNN Model Parameters: {rnn_params:,}")
    print(f"Transformer Model Parameters: {transformer_params:,}")
    print(f"Parameter Ratio (Transformer/RNN): {transformer_params/rnn_params:.2f}")
    
    # Model size analysis
    def get_model_size(model):
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    rnn_size = get_model_size(rnn_model)
    transformer_size = get_model_size(transformer_model)
    
    print(f"\nRNN Model Size: {rnn_size:.2f} MB")
    print(f"Transformer Model Size: {transformer_size:.2f} MB")
    
    return rnn_model, transformer_model, {
        'rnn_params': rnn_params,
        'transformer_params': transformer_params,
        'rnn_size': rnn_size,
        'transformer_size': transformer_size
    }

rnn_model, transformer_model, arch_stats = analyze_model_architecture()

# 2. Training Dynamics Comparison
print("\n2. Training Dynamics Comparison")
print("-" * 50)

def simulate_training_dynamics():
    """Simulate training dynamics for both models"""
    
    # Training parameters
    batch_size = 32
    seq_length = 20
    num_batches = 100
    
    # Create dummy data
    src_data = torch.randint(1, 5000, (batch_size, seq_length))
    tgt_data = torch.randint(1, 5000, (batch_size, seq_length))
    
    # Move to device
    src_data = src_data.to(device)
    tgt_data = tgt_data.to(device)
    
    # Initialize models
    rnn_model.to(device)
    transformer_model.to(device)
    
    # Training metrics storage
    rnn_losses = []
    transformer_losses = []
    rnn_times = []
    transformer_times = []
    
    # Optimizers
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print("Training RNN model...")
    rnn_model.train()
    for batch in range(num_batches):
        start_time = time.time()
        
        rnn_optimizer.zero_grad()
        
        # Forward pass
        outputs = rnn_model(src_data, tgt_data[:, :-1])
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_data[:, 1:].reshape(-1))
        
        # Backward pass
        loss.backward()
        rnn_optimizer.step()
        
        batch_time = time.time() - start_time
        
        rnn_losses.append(loss.item())
        rnn_times.append(batch_time)
        
        if batch % 20 == 0:
            print(f"  Batch {batch}: Loss = {loss.item():.4f}, Time = {batch_time:.4f}s")
    
    print("Training Transformer model...")
    transformer_model.train()
    for batch in range(num_batches):
        start_time = time.time()
        
        transformer_optimizer.zero_grad()
        
        # Forward pass
        outputs = transformer_model(src_data, tgt_data[:, :-1])
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_data[:, 1:].reshape(-1))
        
        # Backward pass
        loss.backward()
        transformer_optimizer.step()
        
        batch_time = time.time() - start_time
        
        transformer_losses.append(loss.item())
        transformer_times.append(batch_time)
        
        if batch % 20 == 0:
            print(f"  Batch {batch}: Loss = {loss.item():.4f}, Time = {batch_time:.4f}s")
    
    return {
        'rnn_losses': rnn_losses,
        'transformer_losses': transformer_losses,
        'rnn_times': rnn_times,
        'transformer_times': transformer_times
    }

training_stats = simulate_training_dynamics()

# 3. Memory Usage Analysis
print("\n3. Memory Usage Analysis")
print("-" * 50)

def analyze_memory_usage():
    """Analyze memory usage during training"""
    
    batch_sizes = [16, 32, 64, 128]
    seq_lengths = [10, 20, 30, 50]
    
    memory_results = {
        'rnn': {'batch_sizes': [], 'seq_lengths': [], 'memory': []},
        'transformer': {'batch_sizes': [], 'seq_lengths': [], 'memory': []}
    }
    
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            # Create dummy data
            src_data = torch.randint(1, 5000, (batch_size, seq_length))
            tgt_data = torch.randint(1, 5000, (batch_size, seq_length))
            
            # RNN memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            rnn_model.to(device)
            src_data = src_data.to(device)
            tgt_data = tgt_data.to(device)
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = rnn_model(src_data, tgt_data[:, :-1])
            
            if torch.cuda.is_available():
                rnn_memory = torch.cuda.max_memory_allocated() / 1024**2
            else:
                rnn_memory = 0
            
            memory_results['rnn']['batch_sizes'].append(batch_size)
            memory_results['rnn']['seq_lengths'].append(seq_length)
            memory_results['rnn']['memory'].append(rnn_memory)
            
            # Transformer memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            transformer_model.to(device)
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = transformer_model(src_data, tgt_data[:, :-1])
            
            if torch.cuda.is_available():
                transformer_memory = torch.cuda.max_memory_allocated() / 1024**2
            else:
                transformer_memory = 0
            
            memory_results['transformer']['batch_sizes'].append(batch_size)
            memory_results['transformer']['seq_lengths'].append(seq_length)
            memory_results['transformer']['memory'].append(transformer_memory)
    
    return memory_results

memory_stats = analyze_memory_usage()

# 4. Visualization
print("\n4. Creating Visualizations")
print("-" * 50)

# Create comprehensive comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Training loss comparison
axes[0, 0].plot(training_stats['rnn_losses'], label='RNN', alpha=0.8)
axes[0, 0].plot(training_stats['transformer_losses'], label='Transformer', alpha=0.8)
axes[0, 0].set_xlabel('Training Steps')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Training time comparison
axes[0, 1].plot(training_stats['rnn_times'], label='RNN', alpha=0.8)
axes[0, 1].plot(training_stats['transformer_times'], label='Transformer', alpha=0.8)
axes[0, 1].set_xlabel('Training Steps')
axes[0, 1].set_ylabel('Time per Batch (s)')
axes[0, 1].set_title('Training Time Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Model size comparison
models = ['RNN', 'Transformer']
sizes = [arch_stats['rnn_size'], arch_stats['transformer_size']]
colors = ['blue', 'red']

axes[0, 2].bar(models, sizes, color=colors, alpha=0.7)
axes[0, 2].set_ylabel('Model Size (MB)')
axes[0, 2].set_title('Model Size Comparison')
for i, v in enumerate(sizes):
    axes[0, 2].text(i, v + 0.1, f'{v:.1f}MB', ha='center', va='bottom')

# Parameter count comparison
params = [arch_stats['rnn_params'], arch_stats['transformer_params']]
axes[1, 0].bar(models, params, color=colors, alpha=0.7)
axes[1, 0].set_ylabel('Number of Parameters')
axes[1, 0].set_title('Parameter Count Comparison')
for i, v in enumerate(params):
    axes[1, 0].text(i, v + max(params)*0.01, f'{v:,}', ha='center', va='bottom')

# Memory usage heatmap for RNN
rnn_memory_matrix = np.array(memory_stats['rnn']['memory']).reshape(len(batch_sizes), len(seq_lengths))
im1 = axes[1, 1].imshow(rnn_memory_matrix, cmap='Blues', aspect='auto')
axes[1, 1].set_title('RNN Memory Usage (MB)')
axes[1, 1].set_xlabel('Sequence Length')
axes[1, 1].set_ylabel('Batch Size')
axes[1, 1].set_xticks(range(len(seq_lengths)))
axes[1, 1].set_yticks(range(len(batch_sizes)))
axes[1, 1].set_xticklabels(seq_lengths)
axes[1, 1].set_yticklabels(batch_sizes)
plt.colorbar(im1, ax=axes[1, 1])

# Memory usage heatmap for Transformer
transformer_memory_matrix = np.array(memory_stats['transformer']['memory']).reshape(len(batch_sizes), len(seq_lengths))
im2 = axes[1, 2].imshow(transformer_memory_matrix, cmap='Reds', aspect='auto')
axes[1, 2].set_title('Transformer Memory Usage (MB)')
axes[1, 2].set_xlabel('Sequence Length')
axes[1, 2].set_ylabel('Batch Size')
axes[1, 2].set_xticks(range(len(seq_lengths)))
axes[1, 2].set_yticks(range(len(batch_sizes)))
axes[1, 2].set_xticklabels(seq_lengths)
axes[1, 2].set_yticklabels(batch_sizes)
plt.colorbar(im2, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('../results/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Performance Analysis
print("\n5. Performance Analysis")
print("-" * 50)

# Calculate average metrics
avg_rnn_loss = np.mean(training_stats['rnn_losses'][-20:])  # Last 20 batches
avg_transformer_loss = np.mean(training_stats['transformer_losses'][-20:])

avg_rnn_time = np.mean(training_stats['rnn_times'])
avg_transformer_time = np.mean(training_stats['transformer_times'])

print(f"Average RNN Loss (last 20 batches): {avg_rnn_loss:.4f}")
print(f"Average Transformer Loss (last 20 batches): {avg_transformer_loss:.4f}")
print(f"Loss Improvement: {(avg_rnn_loss - avg_transformer_loss) / avg_rnn_loss * 100:.2f}%")

print(f"\nAverage RNN Training Time per Batch: {avg_rnn_time:.4f}s")
print(f"Average Transformer Training Time per Batch: {avg_transformer_time:.4f}s")
print(f"Time Ratio (Transformer/RNN): {avg_transformer_time/avg_rnn_time:.2f}")

# 6. Architecture Comparison Table
print("\n6. Architecture Comparison")
print("-" * 50)

comparison_data = {
    'Feature': [
        'Architecture Type',
        'Sequential Processing',
        'Parallel Processing',
        'Attention Mechanism',
        'Position Encoding',
        'Memory Efficiency',
        'Training Speed',
        'Inference Speed',
        'Scalability',
        'Best Use Case'
    ],
    'RNN': [
        'Recurrent',
        'Yes (sequential)',
        'No',
        'Bahdanau/Luong',
        'Implicit',
        'Good',
        'Fast',
        'Fast',
        'Limited',
        'Short sequences'
    ],
    'Transformer': [
        'Attention-based',
        'No (parallel)',
        'Yes',
        'Multi-head Self-attention',
        'Explicit',
        'Moderate',
        'Slower',
        'Fast',
        'Excellent',
        'Long sequences'
    ]
}

# Print comparison table
print(f"{'Feature':<20} {'RNN':<15} {'Transformer':<15}")
print("-" * 50)
for i in range(len(comparison_data['Feature'])):
    print(f"{comparison_data['Feature'][i]:<20} {comparison_data['RNN'][i]:<15} {comparison_data['Transformer'][i]:<15}")

# 7. Recommendations
print("\n7. Model Selection Recommendations")
print("-" * 50)

print("Based on the analysis:")
print("\nChoose RNN when:")
print("- Working with short sequences (< 50 tokens)")
print("- Limited computational resources")
print("- Need fast training times")
print("- Memory constraints are tight")

print("\nChoose Transformer when:")
print("- Working with long sequences (> 50 tokens)")
print("- Need high translation quality")
print("- Have sufficient computational resources")
print("- Parallel processing is important")

print("\n=== Model Comparison Complete ===") 