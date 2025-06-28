"""
Vocabulary management for neural machine translation.
"""

from typing import Dict, List, Optional, Tuple
from collections import Counter
import pickle
import json
from pathlib import Path


class Vocabulary:
    """
    Vocabulary class for managing word-to-index mappings.
    
    This class handles the conversion between words and their corresponding
    indices, including special tokens like PAD, UNK, START, and END.
    """
    
    def __init__(self, word_counts: Optional[Dict[str, int]] = None, 
                 min_freq: int = 2, max_size: Optional[int] = None):
        """
        Initialize vocabulary.
        
        Args:
            word_counts: Dictionary mapping words to their frequencies
            min_freq: Minimum frequency for a word to be included
            max_size: Maximum vocabulary size (None for unlimited)
        """
        # Special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.start_token = '<start>'
        self.end_token = '<end>'
        
        # Special token indices
        self.pad_idx = 0
        self.unk_idx = 1
        self.start_idx = 2
        self.end_idx = 3
        
        # Initialize mappings
        self.word2idx = {
            self.pad_token: self.pad_idx,
            self.unk_token: self.unk_idx,
            self.start_token: self.start_idx,
            self.end_token: self.end_idx
        }
        self.idx2word = {
            self.pad_idx: self.pad_token,
            self.unk_idx: self.unk_token,
            self.start_idx: self.start_token,
            self.end_idx: self.end_token
        }
        
        # Build vocabulary from word counts if provided
        if word_counts is not None:
            self._build_from_counts(word_counts, min_freq, max_size)
    
    def _build_from_counts(self, word_counts: Dict[str, int], 
                          min_freq: int, max_size: Optional[int]):
        """Build vocabulary from word frequency counts."""
        # Filter by minimum frequency
        filtered_words = {
            word: count for word, count in word_counts.items()
            if count >= min_freq and word not in self.word2idx
        }
        
        # Sort by frequency (descending)
        sorted_words = sorted(
            filtered_words.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Limit vocabulary size if specified
        if max_size is not None:
            max_regular_words = max_size - len(self.word2idx)
            sorted_words = sorted_words[:max_regular_words]
        
        # Add words to vocabulary
        for idx, (word, _) in enumerate(sorted_words):
            actual_idx = idx + len(self.word2idx)
            self.word2idx[word] = actual_idx
            self.idx2word[actual_idx] = word
    
    def add_word(self, word: str) -> int:
        """
        Add a word to the vocabulary.
        
        Args:
            word: Word to add
            
        Returns:
            Index of the added word
        """
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            return idx
        return self.word2idx[word]
    
    def encode(self, sentence: str, max_length: Optional[int] = None) -> List[int]:
        """
        Encode a sentence to token indices.
        
        Args:
            sentence: Input sentence
            max_length: Maximum sequence length (None for unlimited)
            
        Returns:
            List of token indices
        """
        tokens = sentence.split()
        indices = []
        
        for token in tokens:
            idx = self.word2idx.get(token, self.unk_idx)
            indices.append(idx)
            
            if max_length is not None and len(indices) >= max_length:
                break
        
        return indices
    
    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """
        Decode token indices back to sentence.
        
        Args:
            indices: List of token indices
            remove_special: Whether to remove special tokens
            
        Returns:
            Decoded sentence
        """
        tokens = []
        
        for idx in indices:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                
                # Skip special tokens if requested
                if remove_special and token in [self.pad_token, self.start_token, self.end_token]:
                    continue
                
                # Stop at end token
                if token == self.end_token:
                    break
                
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def encode_batch(self, sentences: List[str], 
                    max_length: Optional[int] = None) -> List[List[int]]:
        """
        Encode a batch of sentences.
        
        Args:
            sentences: List of input sentences
            max_length: Maximum sequence length
            
        Returns:
            List of encoded sequences
        """
        return [self.encode(sentence, max_length) for sentence in sentences]
    
    def decode_batch(self, sequences: List[List[int]], 
                    remove_special: bool = True) -> List[str]:
        """
        Decode a batch of sequences.
        
        Args:
            sequences: List of token index sequences
            remove_special: Whether to remove special tokens
            
        Returns:
            List of decoded sentences
        """
        return [self.decode(seq, remove_special) for seq in sequences]
    
    def get_word_frequency(self, word: str) -> int:
        """
        Get the frequency of a word in the vocabulary.
        
        Args:
            word: Word to look up
            
        Returns:
            Frequency of the word (0 if not found)
        """
        return getattr(self, '_word_counts', {}).get(word, 0)
    
    def get_most_common(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get the n most common words in the vocabulary.
        
        Args:
            n: Number of words to return
            
        Returns:
            List of (word, frequency) tuples
        """
        word_counts = getattr(self, '_word_counts', {})
        return Counter(word_counts).most_common(n)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)
    
    def __contains__(self, word: str) -> bool:
        """Check if word is in vocabulary."""
        return word in self.word2idx
    
    def __getitem__(self, word: str) -> int:
        """Get index for word (returns UNK index if not found)."""
        return self.word2idx.get(word, self.unk_idx)
    
    def __repr__(self) -> str:
        """String representation of vocabulary."""
        return f"Vocabulary(size={len(self)}, special_tokens={list(self.word2idx.keys())[:4]})"
    
    def save(self, filepath: str, format: str = 'pickle'):
        """
        Save vocabulary to file.
        
        Args:
            filepath: Path to save file
            format: File format ('pickle' or 'json')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        elif format == 'json':
            data = {
                'word2idx': self.word2idx,
                'idx2word': {str(k): v for k, v in self.idx2word.items()},
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'start_token': self.start_token,
                'end_token': self.end_token
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, filepath: str, format: str = 'pickle') -> 'Vocabulary':
        """
        Load vocabulary from file.
        
        Args:
            filepath: Path to vocabulary file
            format: File format ('pickle' or 'json')
            
        Returns:
            Loaded vocabulary instance
        """
        filepath = Path(filepath)
        
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            vocab = cls()
            vocab.word2idx = data['word2idx']
            vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
            vocab.pad_token = data['pad_token']
            vocab.unk_token = data['unk_token']
            vocab.start_token = data['start_token']
            vocab.end_token = data['end_token']
            
            return vocab
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_coverage_stats(self, sentences: List[str]) -> Dict[str, float]:
        """
        Get vocabulary coverage statistics.
        
        Args:
            sentences: List of sentences to analyze
            
        Returns:
            Dictionary with coverage statistics
        """
        total_tokens = 0
        covered_tokens = 0
        unk_tokens = 0
        
        for sentence in sentences:
            tokens = sentence.split()
            total_tokens += len(tokens)
            
            for token in tokens:
                if token in self.word2idx:
                    covered_tokens += 1
                else:
                    unk_tokens += 1
        
        return {
            'total_tokens': total_tokens,
            'covered_tokens': covered_tokens,
            'unk_tokens': unk_tokens,
            'coverage_rate': covered_tokens / total_tokens if total_tokens > 0 else 0.0,
            'unk_rate': unk_tokens / total_tokens if total_tokens > 0 else 0.0
        }


def build_vocabularies(train_df, src_lang: str = 'es', tgt_lang: str = 'en',
                      min_freq: int = 2, max_size: Optional[int] = None) -> Tuple[Vocabulary, Vocabulary]:
    """
    Build source and target vocabularies from training data.
    
    Args:
        train_df: Training DataFrame
        src_lang: Source language column name
        tgt_lang: Target language column name
        min_freq: Minimum word frequency
        max_size: Maximum vocabulary size
        
    Returns:
        Tuple of (src_vocab, tgt_vocab)
    """
    from collections import Counter
    
    # Count words in source and target sentences
    src_word_counts = Counter()
    tgt_word_counts = Counter()
    
    for _, row in train_df.iterrows():
        src_tokens = row[src_lang].split()
        tgt_tokens = row[tgt_lang].split()
        
        src_word_counts.update(src_tokens)
        tgt_word_counts.update(tgt_tokens)
    
    # Create vocabularies
    src_vocab = Vocabulary(src_word_counts, min_freq, max_size)
    tgt_vocab = Vocabulary(tgt_word_counts, min_freq, max_size)
    
    return src_vocab, tgt_vocab 