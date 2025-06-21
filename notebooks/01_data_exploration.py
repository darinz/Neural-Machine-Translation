#!/usr/bin/env python3
"""
01_data_exploration.py
Data Exploration Notebook for Neural Machine Translation

This notebook explores the Spanish-English translation dataset,
analyzing data distribution, vocabulary statistics, and text characteristics.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import re

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import TranslationDataset, Vocabulary, preprocess_sentence, build_vocabulary

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== Neural Machine Translation - Data Exploration ===\n")

# 1. Load and Explore Dataset
print("1. Loading Dataset...")
print("-" * 50)

# Load sample data (assuming data is in data/ directory)
data_path = "../data/spa-eng.txt"

try:
    # Read the dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total lines in dataset: {len(lines)}")
    
    # Parse Spanish-English pairs
    spanish_sentences = []
    english_sentences = []
    
    for line in lines[:10000]:  # Sample first 10k lines for exploration
        if '\t' in line:
            english, spanish = line.strip().split('\t', 1)
            spanish_sentences.append(spanish)
            english_sentences.append(english)
    
    print(f"Parsed {len(spanish_sentences)} sentence pairs")
    
except FileNotFoundError:
    print(f"Dataset not found at {data_path}")
    print("Creating sample data for demonstration...")
    
    # Create sample data for demonstration
    sample_data = [
        ("Hello, how are you?", "Hola, ¿cómo estás?"),
        ("I like coffee", "Me gusta el café"),
        ("Where is the library?", "¿Dónde está la biblioteca?"),
        ("The weather is nice today", "El clima está agradable hoy"),
        ("I want to learn Spanish", "Quiero aprender español"),
        ("What time is it?", "¿Qué hora es?"),
        ("I don't understand", "No entiendo"),
        ("Please help me", "Por favor ayúdame"),
        ("Good morning", "Buenos días"),
        ("Thank you very much", "Muchas gracias")
    ]
    
    english_sentences = [pair[0] for pair in sample_data]
    spanish_sentences = [pair[1] for pair in sample_data]
    
    print(f"Using {len(sample_data)} sample sentence pairs")

# 2. Basic Statistics
print("\n2. Basic Statistics")
print("-" * 50)

# Sentence length analysis
spanish_lengths = [len(sentence.split()) for sentence in spanish_sentences]
english_lengths = [len(sentence.split()) for sentence in english_sentences]

print(f"Spanish sentences - Mean length: {np.mean(spanish_lengths):.2f}, "
      f"Std: {np.std(spanish_lengths):.2f}, "
      f"Min: {min(spanish_lengths)}, Max: {max(spanish_lengths)}")

print(f"English sentences - Mean length: {np.mean(english_lengths):.2f}, "
      f"Std: {np.std(english_lengths):.2f}, "
      f"Min: {min(english_lengths)}, Max: {max(english_lengths)}")

# 3. Vocabulary Analysis
print("\n3. Vocabulary Analysis")
print("-" * 50)

# Tokenize and build vocabulary
def tokenize_sentences(sentences):
    """Tokenize sentences and return all tokens"""
    all_tokens = []
    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        all_tokens.extend(tokens)
    return all_tokens

spanish_tokens = tokenize_sentences(spanish_sentences)
english_tokens = tokenize_sentences(english_sentences)

# Count frequencies
spanish_freq = Counter(spanish_tokens)
english_freq = Counter(english_tokens)

print(f"Spanish vocabulary size: {len(spanish_freq)}")
print(f"English vocabulary size: {len(english_freq)}")

# Most common words
print("\nTop 10 Spanish words:")
for word, count in spanish_freq.most_common(10):
    print(f"  {word}: {count}")

print("\nTop 10 English words:")
for word, count in english_freq.most_common(10):
    print(f"  {word}: {count}")

# 4. Data Visualization
print("\n4. Data Visualization")
print("-" * 50)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Sentence length distribution
axes[0, 0].hist(spanish_lengths, bins=30, alpha=0.7, label='Spanish', color='blue')
axes[0, 0].hist(english_lengths, bins=30, alpha=0.7, label='English', color='red')
axes[0, 0].set_xlabel('Sentence Length (words)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Sentence Length Distribution')
axes[0, 0].legend()

# Word frequency distribution (log scale)
spanish_word_counts = list(spanish_freq.values())
english_word_counts = list(english_freq.values())

axes[0, 1].hist(spanish_word_counts, bins=50, alpha=0.7, label='Spanish', color='blue')
axes[0, 1].hist(english_word_counts, bins=50, alpha=0.7, label='English', color='red')
axes[0, 1].set_xlabel('Word Frequency')
axes[0, 1].set_ylabel('Number of Words')
axes[0, 1].set_title('Word Frequency Distribution')
axes[0, 1].set_xscale('log')
axes[0, 1].legend()

# Vocabulary size vs frequency threshold
def get_vocab_size_by_threshold(freq_dict, thresholds):
    sizes = []
    for threshold in thresholds:
        size = sum(1 for count in freq_dict.values() if count >= threshold)
        sizes.append(size)
    return sizes

thresholds = [1, 2, 5, 10, 20, 50, 100]
spanish_sizes = get_vocab_size_by_threshold(spanish_freq, thresholds)
english_sizes = get_vocab_size_by_threshold(english_freq, thresholds)

axes[1, 0].plot(thresholds, spanish_sizes, 'o-', label='Spanish', color='blue')
axes[1, 0].plot(thresholds, english_sizes, 's-', label='English', color='red')
axes[1, 0].set_xlabel('Minimum Frequency Threshold')
axes[1, 0].set_ylabel('Vocabulary Size')
axes[1, 0].set_title('Vocabulary Size vs Frequency Threshold')
axes[1, 0].legend()
axes[1, 0].set_xscale('log')

# Sample sentence pairs
sample_indices = np.random.choice(len(spanish_sentences), min(5, len(spanish_sentences)), replace=False)
sample_text = ""
for i, idx in enumerate(sample_indices):
    sample_text += f"{i+1}. Spanish: {spanish_sentences[idx]}\n"
    sample_text += f"   English: {english_sentences[idx]}\n\n"

axes[1, 1].text(0.1, 0.9, 'Sample Sentence Pairs:', transform=axes[1, 1].transAxes, 
                fontsize=12, fontweight='bold')
axes[1, 1].text(0.1, 0.8, sample_text, transform=axes[1, 1].transAxes, 
                fontsize=10, verticalalignment='top')
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('../results/data_exploration.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Text Characteristics Analysis
print("\n5. Text Characteristics Analysis")
print("-" * 50)

# Character-level analysis
spanish_chars = ''.join(spanish_sentences)
english_chars = ''.join(english_sentences)

print(f"Spanish character count: {len(spanish_chars)}")
print(f"English character count: {len(english_chars)}")

# Unique characters
spanish_unique_chars = set(spanish_chars)
english_unique_chars = set(english_chars)

print(f"Spanish unique characters: {len(spanish_unique_chars)}")
print(f"English unique characters: {len(english_unique_chars)}")

# Special characters in Spanish
spanish_special = [char for char in spanish_unique_chars if not char.isalnum() and char != ' ']
print(f"Spanish special characters: {spanish_special}")

# 6. Data Quality Checks
print("\n6. Data Quality Checks")
print("-" * 50)

# Check for empty sentences
empty_spanish = sum(1 for s in spanish_sentences if not s.strip())
empty_english = sum(1 for s in english_sentences if not s.strip())

print(f"Empty Spanish sentences: {empty_spanish}")
print(f"Empty English sentences: {empty_english}")

# Check for very long sentences
long_spanish = sum(1 for s in spanish_sentences if len(s.split()) > 50)
long_english = sum(1 for s in english_sentences if len(s.split()) > 50)

print(f"Very long Spanish sentences (>50 words): {long_spanish}")
print(f"Very long English sentences (>50 words): {long_english}")

# Check for sentence length mismatch
length_diff = [abs(len(s.split()) - len(e.split())) for s, e in zip(spanish_sentences, english_sentences)]
large_diff = sum(1 for diff in length_diff if diff > 10)

print(f"Sentences with large length difference (>10 words): {large_diff}")

# 7. Recommendations
print("\n7. Data Preprocessing Recommendations")
print("-" * 50)

print("Based on the analysis, consider the following preprocessing steps:")
print("1. Remove sentences with extreme length differences")
print("2. Set maximum sentence length (e.g., 50 words)")
print("3. Apply minimum frequency threshold for vocabulary (e.g., 2)")
print("4. Normalize text (lowercase, remove extra spaces)")
print("5. Handle special characters appropriately")
print("6. Consider data augmentation for rare words")

print("\n=== Data Exploration Complete ===") 