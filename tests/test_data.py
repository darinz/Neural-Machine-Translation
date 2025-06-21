"""
Tests for data processing modules.

This module contains unit tests for vocabulary, preprocessing, and dataset classes.
"""

import torch
import pytest
import sys
import os
import tempfile
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.vocabulary import Vocabulary
from data.preprocessing import preprocess_text, tokenize_text, normalize_text
from data.dataset import TranslationDataset, create_data_loader


class TestVocabulary:
    """Test cases for Vocabulary class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.vocab = Vocabulary()
        self.test_tokens = ['hello', 'world', 'test', 'vocabulary', 'machine', 'translation']
    
    def test_vocabulary_creation(self):
        """Test vocabulary creation."""
        assert self.vocab is not None
        assert len(self.vocab) == 4  # Special tokens: PAD, UNK, START, END
        assert self.vocab.pad_idx == 0
        assert self.vocab.unk_idx == 1
        assert self.vocab.start_idx == 2
        assert self.vocab.end_idx == 3
    
    def test_add_token(self):
        """Test adding tokens to vocabulary."""
        for token in self.test_tokens:
            self.vocab.add_token(token)
        
        assert len(self.vocab) == 4 + len(self.test_tokens)
        
        # Check that tokens were added
        for token in self.test_tokens:
            assert token in self.vocab.token_to_idx
            assert self.vocab.token_to_idx[token] == self.vocab.idx_to_token.index(token)
    
    def test_get_token_id(self):
        """Test getting token ID."""
        # Add tokens
        for token in self.test_tokens:
            self.vocab.add_token(token)
        
        # Test existing tokens
        for token in self.test_tokens:
            token_id = self.vocab.get_token_id(token)
            assert token_id == self.vocab.token_to_idx[token]
        
        # Test unknown token
        unknown_id = self.vocab.get_token_id('unknown_token')
        assert unknown_id == self.vocab.unk_idx
    
    def test_get_token(self):
        """Test getting token from ID."""
        # Add tokens
        for token in self.test_tokens:
            self.vocab.add_token(token)
        
        # Test existing tokens
        for token in self.test_tokens:
            token_id = self.vocab.get_token_id(token)
            retrieved_token = self.vocab.get_token(token_id)
            assert retrieved_token == token
        
        # Test special tokens
        assert self.vocab.get_token(self.vocab.pad_idx) == '<PAD>'
        assert self.vocab.get_token(self.vocab.unk_idx) == '<UNK>'
        assert self.vocab.get_token(self.vocab.start_idx) == '<START>'
        assert self.vocab.get_token(self.vocab.end_idx) == '<END>'
    
    def test_encode_decode(self):
        """Test encoding and decoding sequences."""
        # Add tokens
        for token in self.test_tokens:
            self.vocab.add_token(token)
        
        # Test encoding
        text = "hello world test"
        encoded = self.vocab.encode(text, add_special_tokens=True)
        expected = [self.vocab.start_idx] + [self.vocab.get_token_id(t) for t in text.split()] + [self.vocab.end_idx]
        assert encoded == expected
        
        # Test decoding
        decoded = self.vocab.decode(encoded, remove_special=True)
        assert decoded == text
    
    def test_build_from_texts(self):
        """Test building vocabulary from texts."""
        texts = [
            "hello world test",
            "machine translation is fun",
            "vocabulary building test"
        ]
        
        vocab = Vocabulary()
        vocab.build_from_texts(texts, min_freq=1)
        
        # Check that all tokens are included
        all_tokens = set()
        for text in texts:
            all_tokens.update(text.split())
        
        for token in all_tokens:
            assert token in vocab.token_to_idx
    
    def test_build_from_texts_min_freq(self):
        """Test building vocabulary with minimum frequency."""
        texts = [
            "hello world test",
            "hello world machine",
            "test translation fun"
        ]
        
        vocab = Vocabulary()
        vocab.build_from_texts(texts, min_freq=2)
        
        # 'hello', 'world', 'test' should be included (freq >= 2)
        assert 'hello' in vocab.token_to_idx
        assert 'world' in vocab.token_to_idx
        assert 'test' in vocab.token_to_idx
        
        # 'machine', 'translation', 'fun' should not be included (freq < 2)
        assert 'machine' not in vocab.token_to_idx
        assert 'translation' not in vocab.token_to_idx
        assert 'fun' not in vocab.token_to_idx
    
    def test_save_load(self):
        """Test saving and loading vocabulary."""
        # Add tokens
        for token in self.test_tokens:
            self.vocab.add_token(token)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            vocab_path = f.name
        
        try:
            self.vocab.save(vocab_path)
            
            # Load vocabulary
            loaded_vocab = Vocabulary.load(vocab_path)
            
            # Check that vocabularies are identical
            assert len(self.vocab) == len(loaded_vocab)
            assert self.vocab.token_to_idx == loaded_vocab.token_to_idx
            assert self.vocab.idx_to_token == loaded_vocab.idx_to_token
            assert self.vocab.token_freq == loaded_vocab.token_freq
        
        finally:
            # Clean up
            if os.path.exists(vocab_path):
                os.unlink(vocab_path)
    
    def test_vocabulary_size_limits(self):
        """Test vocabulary size limits."""
        vocab = Vocabulary(max_size=10)
        
        # Add more tokens than max_size
        for i in range(15):
            vocab.add_token(f"token_{i}")
        
        # Should only keep the most frequent tokens
        assert len(vocab) <= 10


class TestPreprocessing:
    """Test cases for preprocessing functions."""
    
    def test_normalize_text(self):
        """Test text normalization."""
        # Test basic normalization
        text = "Hello, World!  This is a TEST."
        normalized = normalize_text(text)
        assert normalized == "hello world this is a test"
        
        # Test with numbers
        text = "Test123 with 456 numbers"
        normalized = normalize_text(text)
        assert normalized == "test123 with 456 numbers"
        
        # Test with special characters
        text = "Test@#$%^&*()_+-=[]{}|;':\",./<>?"
        normalized = normalize_text(text)
        assert normalized == "test"
    
    def test_tokenize_text(self):
        """Test text tokenization."""
        # Test basic tokenization
        text = "hello world test"
        tokens = tokenize_text(text)
        assert tokens == ['hello', 'world', 'test']
        
        # Test with punctuation
        text = "hello, world! test."
        tokens = tokenize_text(text)
        assert tokens == ['hello', 'world', 'test']
        
        # Test with extra whitespace
        text = "  hello   world  test  "
        tokens = tokenize_text(text)
        assert tokens == ['hello', 'world', 'test']
    
    def test_preprocess_text(self):
        """Test complete text preprocessing."""
        # Test Spanish text
        spanish_text = "¡Hola, mundo! ¿Cómo estás?"
        processed = preprocess_text(spanish_text)
        assert processed == "hola mundo como estas"
        
        # Test English text
        english_text = "Hello, world! How are you?"
        processed = preprocess_text(english_text)
        assert processed == "hello world how are you"
        
        # Test with numbers and special characters
        text = "Test123 @#$%^&*()_+-=[]{}|;':\",./<>?"
        processed = preprocess_text(text)
        assert processed == "test123"
    
    def test_preprocess_text_lowercase(self):
        """Test preprocessing with lowercase option."""
        text = "Hello WORLD Test"
        
        # With lowercase=True (default)
        processed = preprocess_text(text, lowercase=True)
        assert processed == "hello world test"
        
        # With lowercase=False
        processed = preprocess_text(text, lowercase=False)
        assert processed == "Hello WORLD Test"
    
    def test_preprocess_text_remove_punctuation(self):
        """Test preprocessing with punctuation removal."""
        text = "Hello, world! Test."
        
        # With remove_punctuation=True (default)
        processed = preprocess_text(text, remove_punctuation=True)
        assert processed == "hello world test"
        
        # With remove_punctuation=False
        processed = preprocess_text(text, remove_punctuation=False)
        assert processed == "hello world test"


class TestTranslationDataset:
    """Test cases for TranslationDataset class."""
    
    def setup_method(self):
        """Set up test fixtures."""
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
        
        # Create vocabularies
        self.src_vocab = Vocabulary()
        self.tgt_vocab = Vocabulary()
        
        # Build vocabularies from texts
        self.src_vocab.build_from_texts(self.src_texts, min_freq=1)
        self.tgt_vocab.build_from_texts(self.tgt_texts, min_freq=1)
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = TranslationDataset(
            src_texts=self.src_texts,
            tgt_texts=self.tgt_texts,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab
        )
        
        assert len(dataset) == len(self.src_texts)
        assert dataset.src_vocab == self.src_vocab
        assert dataset.tgt_vocab == self.tgt_vocab
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = TranslationDataset(
            src_texts=self.src_texts,
            tgt_texts=self.tgt_texts,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab
        )
        
        # Get first item
        src_tensor, tgt_tensor = dataset[0]
        
        assert isinstance(src_tensor, torch.Tensor)
        assert isinstance(tgt_tensor, torch.Tensor)
        assert src_tensor.dim() == 1
        assert tgt_tensor.dim() == 1
        
        # Check that tensors contain valid indices
        assert src_tensor.min() >= 0
        assert src_tensor.max() < len(self.src_vocab)
        assert tgt_tensor.min() >= 0
        assert tgt_tensor.max() < len(self.tgt_vocab)
    
    def test_dataset_with_max_length(self):
        """Test dataset with maximum length constraint."""
        dataset = TranslationDataset(
            src_texts=self.src_texts,
            tgt_texts=self.tgt_texts,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
            max_src_length=5,
            max_tgt_length=5
        )
        
        # Check that all sequences respect max length
        for i in range(len(dataset)):
            src_tensor, tgt_tensor = dataset[i]
            assert src_tensor.size(0) <= 5
            assert tgt_tensor.size(0) <= 5
    
    def test_dataset_filtering(self):
        """Test dataset filtering by length."""
        # Add some longer texts
        long_src_texts = self.src_texts + ["este es un texto muy largo que deberia ser filtrado"]
        long_tgt_texts = self.tgt_texts + ["this is a very long text that should be filtered"]
        
        dataset = TranslationDataset(
            src_texts=long_src_texts,
            tgt_texts=long_tgt_texts,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
            max_src_length=10,
            max_tgt_length=10
        )
        
        # Should filter out the long text
        assert len(dataset) == len(self.src_texts)
    
    def test_dataset_padding(self):
        """Test dataset padding functionality."""
        dataset = TranslationDataset(
            src_texts=self.src_texts,
            tgt_texts=self.tgt_texts,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
            pad_to_max_length=True
        )
        
        # Get all items and check padding
        src_lengths = []
        tgt_lengths = []
        
        for i in range(len(dataset)):
            src_tensor, tgt_tensor = dataset[i]
            src_lengths.append(src_tensor.size(0))
            tgt_lengths.append(tgt_tensor.size(0))
        
        # All sequences should have the same length when padded
        if dataset.pad_to_max_length:
            assert len(set(src_lengths)) == 1
            assert len(set(tgt_lengths)) == 1


class TestDataLoader:
    """Test cases for data loader creation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data
        self.src_texts = [
            "hola mundo",
            "como estas",
            "buenos dias",
            "gracias por todo",
            "adios amigo",
            "hasta luego"
        ]
        self.tgt_texts = [
            "hello world",
            "how are you",
            "good morning",
            "thank you for everything",
            "goodbye friend",
            "see you later"
        ]
        
        # Create vocabularies
        self.src_vocab = Vocabulary()
        self.tgt_vocab = Vocabulary()
        
        # Build vocabularies from texts
        self.src_vocab.build_from_texts(self.src_texts, min_freq=1)
        self.tgt_vocab.build_from_texts(self.tgt_texts, min_freq=1)
    
    def test_create_data_loader(self):
        """Test data loader creation."""
        dataset = TranslationDataset(
            src_texts=self.src_texts,
            tgt_texts=self.tgt_texts,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab
        )
        
        data_loader = create_data_loader(
            dataset=dataset,
            batch_size=2,
            shuffle=True
        )
        
        assert data_loader is not None
        assert data_loader.batch_size == 2
        
        # Test iteration
        batch_count = 0
        for batch in data_loader:
            src_batch, tgt_batch = batch
            assert isinstance(src_batch, torch.Tensor)
            assert isinstance(tgt_batch, torch.Tensor)
            assert src_batch.dim() == 2  # [batch_size, seq_len]
            assert tgt_batch.dim() == 2  # [batch_size, seq_len]
            batch_count += 1
        
        # Should have 3 batches with batch_size=2 and 6 samples
        assert batch_count == 3
    
    def test_data_loader_padding(self):
        """Test data loader with padding."""
        dataset = TranslationDataset(
            src_texts=self.src_texts,
            tgt_texts=self.tgt_texts,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab
        )
        
        data_loader = create_data_loader(
            dataset=dataset,
            batch_size=3,
            shuffle=False,
            pad_to_max_length=True
        )
        
        # Get first batch
        src_batch, tgt_batch = next(iter(data_loader))
        
        # All sequences in batch should have same length
        assert src_batch.size(1) == src_batch.size(1)  # All same length
        assert tgt_batch.size(1) == tgt_batch.size(1)  # All same length
    
    def test_data_loader_collate_fn(self):
        """Test data loader with custom collate function."""
        dataset = TranslationDataset(
            src_texts=self.src_texts,
            tgt_texts=self.tgt_texts,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab
        )
        
        def custom_collate(batch):
            src_tensors, tgt_tensors = zip(*batch)
            return torch.stack(src_tensors), torch.stack(tgt_tensors)
        
        data_loader = create_data_loader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=custom_collate
        )
        
        # Test iteration
        for batch in data_loader:
            src_batch, tgt_batch = batch
            assert src_batch.size(0) == 2  # batch_size
            break


class TestDataIntegration:
    """Integration tests for data processing pipeline."""
    
    def test_full_pipeline(self):
        """Test complete data processing pipeline."""
        # Sample Spanish-English pairs
        spanish_texts = [
            "hola como estas",
            "buenos dias senor",
            "gracias por la ayuda",
            "adios hasta luego"
        ]
        english_texts = [
            "hello how are you",
            "good morning sir",
            "thank you for the help",
            "goodbye see you later"
        ]
        
        # Create vocabularies
        src_vocab = Vocabulary()
        tgt_vocab = Vocabulary()
        
        # Build vocabularies
        src_vocab.build_from_texts(spanish_texts, min_freq=1)
        tgt_vocab.build_from_texts(english_texts, min_freq=1)
        
        # Create dataset
        dataset = TranslationDataset(
            src_texts=spanish_texts,
            tgt_texts=english_texts,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab
        )
        
        # Create data loader
        data_loader = create_data_loader(
            dataset=dataset,
            batch_size=2,
            shuffle=False
        )
        
        # Test pipeline
        batch_count = 0
        for src_batch, tgt_batch in data_loader:
            assert src_batch.shape[0] == 2  # batch_size
            assert tgt_batch.shape[0] == 2  # batch_size
            batch_count += 1
        
        assert batch_count == 2  # 4 samples / batch_size 2 = 2 batches 