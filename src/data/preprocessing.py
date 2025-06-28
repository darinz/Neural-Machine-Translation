"""
Text preprocessing utilities for neural machine translation.
"""

import re
import unicodedata
from typing import List, Tuple, Dict
from collections import Counter
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


def unicode_to_ascii(s: str) -> str:
    """
    Normalize latin chars with accent to their canonical decomposition.
    
    Args:
        s: Input string with unicode characters
        
    Returns:
        String with normalized ASCII characters
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) 
        if unicodedata.category(c) != 'Mn'
    )


def preprocess_sentence(sentence: str, lowercase: bool = True, 
                       remove_punctuation: bool = False) -> str:
    """
    Preprocess a sentence by adding start/end tokens and applying text normalization.
    
    Args:
        sentence: Input sentence
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        
    Returns:
        Preprocessed sentence with start/end tokens
    """
    # Normalize unicode characters
    sentence = unicode_to_ascii(sentence.strip())
    
    # Convert to lowercase if specified
    if lowercase:
        sentence = sentence.lower()
    
    # Handle punctuation
    if remove_punctuation:
        # Remove all punctuation except sentence boundaries
        sentence = re.sub(r'[^\w\s]', '', sentence)
    else:
        # Add spaces around punctuation for tokenization
        sentence = re.sub(r'([?.!,¿])', r' \1 ', sentence)
    
    # Clean up multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # Remove non-alphabetic characters (except punctuation if not removed)
    if remove_punctuation:
        sentence = re.sub(r'[^a-zA-Z\s]', ' ', sentence)
    else:
        sentence = re.sub(r'[^a-zA-Z?.!,¿\s]', ' ', sentence)
    
    # Clean up and add start/end tokens
    sentence = sentence.strip()
    sentence = f'<start> {sentence} <end>'
    
    return sentence


def preprocess_text(text: str, lowercase: bool = True, 
                   remove_punctuation: bool = False) -> str:
    """
    Preprocess text by applying normalization and cleaning.
    
    Args:
        text: Input text
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        
    Returns:
        Preprocessed text
    """
    # Normalize unicode characters
    text = unicode_to_ascii(text.strip())
    
    # Convert to lowercase if specified
    if lowercase:
        text = text.lower()
    
    # Handle punctuation
    if remove_punctuation:
        # Remove all punctuation
        text = re.sub(r'[^\w\s]', '', text)
    else:
        # Add spaces around punctuation for tokenization
        text = re.sub(r'([?.!,¿])', r' \1 ', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-alphabetic characters (except punctuation if not removed)
    if remove_punctuation:
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    else:
        text = re.sub(r'[^a-zA-Z?.!,¿\s]', ' ', text)
    
    # Clean up
    text = text.strip()
    
    return text


def build_vocabulary(sentences: List[str], min_freq: int = 2) -> Dict[str, int]:
    """
    Build vocabulary from a list of sentences.
    
    Args:
        sentences: List of preprocessed sentences
        min_freq: Minimum frequency for a word to be included in vocabulary
        
    Returns:
        Dictionary mapping words to their frequencies
    """
    # Tokenize all sentences
    all_tokens = []
    for sentence in sentences:
        tokens = sentence.split()
        all_tokens.extend(tokens)
    
    # Count word frequencies
    word_counts = Counter(all_tokens)
    
    # Filter by minimum frequency
    vocabulary = {
        word: count for word, count in word_counts.items() 
        if count >= min_freq
    }
    
    return vocabulary


def build_vocabularies(src_sentences: List[str], tgt_sentences: List[str], 
                      min_freq: int = 2) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build vocabularies for both source and target languages.
    
    Args:
        src_sentences: List of source language sentences or DataFrame
        tgt_sentences: List of target language sentences or language codes
        min_freq: Minimum frequency for a word to be included in vocabulary
        
    Returns:
        Tuple of (src_vocabulary, tgt_vocabulary)
    """
    import pandas as pd
    
    # Handle DataFrame input (from quick_start.py)
    if isinstance(src_sentences, pd.DataFrame):
        df = src_sentences
        src_lang = tgt_sentences  # In this case, tgt_sentences is actually src_lang
        tgt_lang = min_freq       # In this case, min_freq is actually tgt_lang
        min_freq = 2              # Default min_freq
        
        # Extract sentences from DataFrame
        src_sentences = df[src_lang].tolist()
        tgt_sentences = df[tgt_lang].tolist()
    
    src_vocab = build_vocabulary(src_sentences, min_freq)
    tgt_vocab = build_vocabulary(tgt_sentences, min_freq)
    
    return src_vocab, tgt_vocab


def pad_sequences(sequences: List[List[int]], max_length: int, 
                 pad_value: int = 0) -> torch.Tensor:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of token sequences
        max_length: Maximum sequence length
        pad_value: Value to use for padding
        
    Returns:
        Padded sequences as a tensor
    """
    padded_sequences = []
    
    for seq in sequences:
        if len(seq) > max_length:
            # Truncate if too long
            padded_seq = seq[:max_length]
        else:
            # Pad if too short
            padded_seq = seq + [pad_value] * (max_length - len(seq))
        
        padded_sequences.append(padded_seq)
    
    return torch.tensor(padded_sequences, dtype=torch.long)


def create_attention_mask(sequences: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
    """
    Create attention mask for padded sequences.
    
    Args:
        sequences: Padded sequences tensor [batch_size, seq_len]
        pad_value: Value used for padding
        
    Returns:
        Attention mask tensor [batch_size, seq_len] where True indicates padding
    """
    return (sequences == pad_value)


def load_translation_data(data_path: str, src_lang: str = 'es', 
                         tgt_lang: str = 'en', max_samples: int = None,
                         lowercase: bool = True, remove_punctuation: bool = False) -> pd.DataFrame:
    """
    Load and preprocess translation data from file.
    
    Args:
        data_path: Path to the data file
        src_lang: Source language code
        tgt_lang: Target language code
        max_samples: Maximum number of samples to load
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        
    Returns:
        DataFrame with preprocessed source and target sentences
    """
    # Load data
    df = pd.read_csv(
        data_path,
        sep='\t',
        header=None,
        usecols=[0, 1],
        names=[tgt_lang, src_lang],
        nrows=max_samples,
        encoding='utf-8'
    )
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Preprocess sentences
    df[tgt_lang] = df[tgt_lang].apply(
        lambda x: preprocess_sentence(x, lowercase, remove_punctuation)
    )
    df[src_lang] = df[src_lang].apply(
        lambda x: preprocess_sentence(x, lowercase, remove_punctuation)
    )
    
    return df


def split_data(df: pd.DataFrame, train_split: float = 0.8, 
               val_split: float = 0.1, test_split: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    n_samples = len(df)
    train_end = int(n_samples * train_split)
    val_end = int(n_samples * (train_split + val_split))
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    return train_df, val_df, test_df


def preprocess_data_to_tensor(df: pd.DataFrame, src_vocab: 'Vocabulary', 
                             tgt_vocab: 'Vocabulary', max_length: int = 50) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Convert DataFrame to tensor format for training.
    
    Args:
        df: DataFrame with source and target sentences
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (src_tensor, tgt_tensor, max_src_len, max_tgt_len)
    """
    # Convert sentences to token indices
    src_sequences = []
    tgt_sequences = []
    
    for _, row in df.iterrows():
        # Source sequence
        src_tokens = row.iloc[1].split()  # Assuming source is second column
        src_indices = [src_vocab.word2idx.get(token, src_vocab.unk_idx) for token in src_tokens]
        src_sequences.append(src_indices)
        
        # Target sequence
        tgt_tokens = row.iloc[0].split()  # Assuming target is first column
        tgt_indices = [tgt_vocab.word2idx.get(token, tgt_vocab.unk_idx) for token in tgt_tokens]
        tgt_sequences.append(tgt_indices)
    
    # Find actual maximum lengths
    max_src_len = min(max(len(seq) for seq in src_sequences), max_length)
    max_tgt_len = min(max(len(seq) for seq in tgt_sequences), max_length)
    
    # Pad sequences
    src_tensor = pad_sequences(src_sequences, max_src_len, src_vocab.pad_idx)
    tgt_tensor = pad_sequences(tgt_sequences, max_tgt_len, tgt_vocab.pad_idx)
    
    return src_tensor, tgt_tensor, max_src_len, max_tgt_len


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
    from .dataset import TranslationDataset
    
    dataset = TranslationDataset(src_tensor, tgt_tensor)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    ) 