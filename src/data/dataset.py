"""
Dataset classes for neural machine translation.
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, Any
import numpy as np


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for translation data.
    
    This dataset handles source and target sequences, providing efficient
    access to training pairs with optional data augmentation and caching.
    """
    
    def __init__(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor,
                 src_vocab=None, tgt_vocab=None, max_length: Optional[int] = None):
        """
        Initialize translation dataset.
        
        Args:
            src_tensor: Source sequences tensor [num_samples, src_len]
            tgt_tensor: Target sequences tensor [num_samples, tgt_len]
            src_vocab: Source vocabulary (optional, for decoding)
            tgt_vocab: Target vocabulary (optional, for decoding)
            max_length: Maximum sequence length (None for unlimited)
        """
        self.src_tensor = src_tensor
        self.tgt_tensor = tgt_tensor
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
        # Validate input
        assert len(src_tensor) == len(tgt_tensor), "Source and target tensors must have same length"
        
        # Calculate sequence lengths for efficient batching
        self.src_lengths = self._calculate_lengths(src_tensor)
        self.tgt_lengths = self._calculate_lengths(tgt_tensor)
        
        # Create indices for sorting by length (for efficient batching)
        self.sorted_indices = self._create_sorted_indices()
    
    def _calculate_lengths(self, tensor: torch.Tensor) -> torch.Tensor:
        """Calculate actual sequence lengths (excluding padding)."""
        # Assuming padding token is 0
        return (tensor != 0).sum(dim=1)
    
    def _create_sorted_indices(self) -> torch.Tensor:
        """Create indices sorted by source sequence length for efficient batching."""
        # Sort by source length, then by target length
        src_lengths = self.src_lengths.numpy()
        tgt_lengths = self.tgt_lengths.numpy()
        
        # Create sorting key: (src_length, tgt_length, index)
        sorting_key = list(zip(src_lengths, tgt_lengths, range(len(self))))
        sorting_key.sort()
        
        return torch.tensor([idx for _, _, idx in sorting_key])
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.src_tensor)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary containing source and target sequences
        """
        src_seq = self.src_tensor[idx]
        tgt_seq = self.tgt_tensor[idx]
        src_len = self.src_lengths[idx]
        tgt_len = self.tgt_lengths[idx]
        
        return {
            'src_seq': src_seq,
            'tgt_seq': tgt_seq,
            'src_len': src_len,
            'tgt_len': tgt_len,
            'index': idx
        }
    
    def get_sorted_batch(self, batch_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get a batch of examples sorted by length.
        
        Args:
            batch_indices: Indices of examples in the batch
            
        Returns:
            Dictionary containing batched sequences
        """
        # Sort indices by source length for efficient processing
        batch_src_lengths = self.src_lengths[batch_indices]
        sorted_indices = torch.argsort(batch_src_lengths, descending=True)
        sorted_batch_indices = batch_indices[sorted_indices]
        
        # Get sequences
        src_seqs = self.src_tensor[sorted_batch_indices]
        tgt_seqs = self.tgt_tensor[sorted_batch_indices]
        src_lens = self.src_lengths[sorted_batch_indices]
        tgt_lens = self.tgt_lengths[sorted_batch_indices]
        
        return {
            'src_seq': src_seqs,
            'tgt_seq': tgt_seqs,
            'src_len': src_lens,
            'tgt_len': tgt_lens,
            'indices': sorted_batch_indices
        }
    
    def decode_example(self, idx: int, remove_special: bool = True) -> Dict[str, str]:
        """
        Decode an example back to text.
        
        Args:
            idx: Index of the example
            remove_special: Whether to remove special tokens
            
        Returns:
            Dictionary with decoded source and target text
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            raise ValueError("Vocabularies must be provided for decoding")
        
        src_indices = self.src_tensor[idx].tolist()
        tgt_indices = self.tgt_tensor[idx].tolist()
        
        src_text = self.src_vocab.decode(src_indices, remove_special)
        tgt_text = self.tgt_vocab.decode(tgt_indices, remove_special)
        
        return {
            'src_text': src_text,
            'tgt_text': tgt_text,
            'src_len': self.src_lengths[idx].item(),
            'tgt_len': self.tgt_lengths[idx].item()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        src_lengths = self.src_lengths.numpy()
        tgt_lengths = self.tgt_lengths.numpy()
        
        return {
            'num_samples': len(self),
            'src_length_stats': {
                'mean': float(np.mean(src_lengths)),
                'std': float(np.std(src_lengths)),
                'min': int(np.min(src_lengths)),
                'max': int(np.max(src_lengths)),
                'median': float(np.median(src_lengths))
            },
            'tgt_length_stats': {
                'mean': float(np.mean(tgt_lengths)),
                'std': float(np.std(tgt_lengths)),
                'min': int(np.min(tgt_lengths)),
                'max': int(np.max(tgt_lengths)),
                'median': float(np.median(tgt_lengths))
            },
            'src_vocab_size': len(self.src_vocab) if self.src_vocab else None,
            'tgt_vocab_size': len(self.tgt_vocab) if self.tgt_vocab else None
        }
    
    def filter_by_length(self, min_src_len: int = 0, max_src_len: Optional[int] = None,
                        min_tgt_len: int = 0, max_tgt_len: Optional[int] = None) -> 'TranslationDataset':
        """
        Filter dataset by sequence lengths.
        
        Args:
            min_src_len: Minimum source sequence length
            max_src_len: Maximum source sequence length
            min_tgt_len: Minimum target sequence length
            max_tgt_len: Maximum target sequence length
            
        Returns:
            Filtered dataset
        """
        # Create mask for filtering
        mask = torch.ones(len(self), dtype=torch.bool)
        
        if min_src_len > 0:
            mask &= (self.src_lengths >= min_src_len)
        if max_src_len is not None:
            mask &= (self.src_lengths <= max_src_len)
        if min_tgt_len > 0:
            mask &= (self.tgt_lengths >= min_tgt_len)
        if max_tgt_len is not None:
            mask &= (self.tgt_lengths <= max_tgt_len)
        
        # Apply filter
        filtered_src = self.src_tensor[mask]
        filtered_tgt = self.tgt_tensor[mask]
        
        return TranslationDataset(
            filtered_src, filtered_tgt, 
            self.src_vocab, self.tgt_vocab, 
            self.max_length
        )
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
              test_ratio: float = 0.1, shuffle: bool = True) -> Tuple['TranslationDataset', 'TranslationDataset', 'TranslationDataset']:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            shuffle: Whether to shuffle before splitting
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        n_samples = len(self)
        indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        def create_subset(indices):
            return TranslationDataset(
                self.src_tensor[indices],
                self.tgt_tensor[indices],
                self.src_vocab,
                self.tgt_vocab,
                self.max_length
            )
        
        return (
            create_subset(train_indices),
            create_subset(val_indices),
            create_subset(test_indices)
        )


class TranslationDataLoader:
    """
    Custom DataLoader for translation data with efficient batching.
    
    This loader provides optimized batching strategies and collation
    functions for translation datasets.
    """
    
    def __init__(self, dataset: TranslationDataset, batch_size: int,
                 shuffle: bool = True, num_workers: int = 4,
                 pin_memory: bool = True, drop_last: bool = True):
        """
        Initialize translation data loader.
        
        Args:
            dataset: Translation dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
        # Create standard PyTorch DataLoader
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: list) -> Dict[str, torch.Tensor]:
        """
        Custom collation function for batching.
        
        Args:
            batch: List of dataset items
            
        Returns:
            Dictionary with batched tensors
        """
        # Extract sequences and lengths
        src_seqs = [item['src_seq'] for item in batch]
        tgt_seqs = [item['tgt_seq'] for item in batch]
        src_lens = [item['src_len'] for item in batch]
        tgt_lens = [item['tgt_len'] for item in batch]
        
        # Pad sequences to maximum length in batch
        max_src_len = max(src_lens)
        max_tgt_len = max(tgt_lens)
        
        # Pad source sequences
        padded_src_seqs = []
        for seq in src_seqs:
            if len(seq) < max_src_len:
                padded_seq = torch.cat([seq, torch.zeros(max_src_len - len(seq), dtype=seq.dtype)])
            else:
                padded_seq = seq[:max_src_len]
            padded_src_seqs.append(padded_seq)
        
        # Pad target sequences
        padded_tgt_seqs = []
        for seq in tgt_seqs:
            if len(seq) < max_tgt_len:
                padded_seq = torch.cat([seq, torch.zeros(max_tgt_len - len(seq), dtype=seq.dtype)])
            else:
                padded_seq = seq[:max_tgt_len]
            padded_tgt_seqs.append(padded_seq)
        
        return {
            'src_seq': torch.stack(padded_src_seqs),
            'tgt_seq': torch.stack(padded_tgt_seqs),
            'src_len': torch.tensor(src_lens),
            'tgt_len': torch.tensor(tgt_lens)
        }
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.loader)
    
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.loader)
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current batch.
        
        Returns:
            Dictionary with batch statistics
        """
        return {
            'batch_size': self.batch_size,
            'num_batches': len(self),
            'total_samples': len(self.dataset)
        }


def create_data_loaders(src_tensor: torch.Tensor, tgt_tensor: torch.Tensor,
                       batch_size: int, shuffle: bool = True,
                       num_workers: int = 4, pin_memory: bool = True) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for training.
    
    Args:
        src_tensor: Source sequences tensor
        tgt_tensor: Target sequences tensor
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    dataset = TranslationDataset(src_tensor, tgt_tensor)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    ) 