"""
Evaluation metrics for neural machine translation.

This module provides various metrics for evaluating translation quality,
including BLEU scores and other common NLP metrics.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import Counter
import re

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. BLEU scores will not be computed.")


def compute_bleu_score(references: List[List[str]], candidates: List[List[str]], 
                      weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)) -> float:
    """
    Compute BLEU score for translation evaluation.
    
    Args:
        references: List of reference translations (each is a list of tokens)
        candidates: List of candidate translations (each is a list of tokens)
        weights: Weights for n-gram precision (default: uniform weights for 1-4 grams)
        
    Returns:
        BLEU score
    """
    if not NLTK_AVAILABLE:
        return 0.0
    
    if len(references) != len(candidates):
        raise ValueError("Number of references and candidates must match")
    
    smoother = SmoothingFunction()
    total_bleu = 0.0
    
    for ref, cand in zip(references, candidates):
        if len(cand) == 0:
            continue
        score = sentence_bleu([ref], cand, weights=weights, smoothing_function=smoother.method1)
        total_bleu += score
    
    return total_bleu / len(references) if references else 0.0


def compute_bleu_scores(references: List[List[str]], candidates: List[List[str]]) -> Dict[str, float]:
    """
    Compute BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores.
    
    Args:
        references: List of reference translations
        candidates: List of candidate translations
        
    Returns:
        Dictionary with BLEU scores
    """
    scores = {}
    
    # BLEU-1
    scores['bleu_1'] = compute_bleu_score(references, candidates, weights=(1.0,))
    
    # BLEU-2
    scores['bleu_2'] = compute_bleu_score(references, candidates, weights=(0.5, 0.5))
    
    # BLEU-3
    scores['bleu_3'] = compute_bleu_score(references, candidates, weights=(0.33, 0.33, 0.34))
    
    # BLEU-4
    scores['bleu_4'] = compute_bleu_score(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
    
    return scores


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                   ignore_index: int = 0) -> Dict[str, float]:
    """
    Compute various metrics for model evaluation.
    
    Args:
        predictions: Model predictions [batch_size, seq_len, vocab_size]
        targets: Target sequences [batch_size, seq_len]
        ignore_index: Index to ignore in loss computation (usually padding)
        
    Returns:
        Dictionary with computed metrics
    """
    # Compute loss
    loss = F.cross_entropy(
        predictions.view(-1, predictions.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    # Compute accuracy
    with torch.no_grad():
        pred_indices = torch.argmax(predictions, dim=-1)
        correct = (pred_indices == targets) & (targets != ignore_index)
        total = (targets != ignore_index).sum()
        accuracy = correct.sum().float() / total if total > 0 else 0.0
    
    # Compute perplexity
    perplexity = torch.exp(loss)
    
    return {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'perplexity': perplexity.item()
    }


def tokenize_sentence(sentence: str, lowercase: bool = True) -> List[str]:
    """
    Tokenize a sentence into words.
    
    Args:
        sentence: Input sentence
        lowercase: Whether to convert to lowercase
        
    Returns:
        List of tokens
    """
    if lowercase:
        sentence = sentence.lower()
    
    # Simple word tokenization
    tokens = re.findall(r'\b\w+\b', sentence)
    return tokens


def detokenize_sentence(tokens: List[str]) -> str:
    """
    Convert tokens back to a sentence.
    
    Args:
        tokens: List of tokens
        
    Returns:
        Detokenized sentence
    """
    return ' '.join(tokens)


def remove_special_tokens(tokens: List[str], special_tokens: List[str] = None) -> List[str]:
    """
    Remove special tokens from a token list.
    
    Args:
        tokens: List of tokens
        special_tokens: List of special tokens to remove
        
    Returns:
        List of tokens with special tokens removed
    """
    if special_tokens is None:
        special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
    
    return [token for token in tokens if token not in special_tokens]


def compute_length_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                          ignore_index: int = 0) -> Dict[str, float]:
    """
    Compute length-related metrics.
    
    Args:
        predictions: Model predictions [batch_size, seq_len, vocab_size]
        targets: Target sequences [batch_size, seq_len]
        ignore_index: Index to ignore in computation
        
    Returns:
        Dictionary with length metrics
    """
    with torch.no_grad():
        # Get predicted lengths (excluding padding)
        pred_indices = torch.argmax(predictions, dim=-1)
        pred_lengths = (pred_indices != ignore_index).sum(dim=1).float()
        target_lengths = (targets != ignore_index).sum(dim=1).float()
        
        # Length statistics
        avg_pred_length = pred_lengths.mean().item()
        avg_target_length = target_lengths.mean().item()
        length_ratio = avg_pred_length / avg_target_length if avg_target_length > 0 else 1.0
        
        # Length correlation
        if len(pred_lengths) > 1:
            length_correlation = torch.corrcoef(torch.stack([pred_lengths, target_lengths]))[0, 1].item()
        else:
            length_correlation = 0.0
    
    return {
        'avg_pred_length': avg_pred_length,
        'avg_target_length': avg_target_length,
        'length_ratio': length_ratio,
        'length_correlation': length_correlation
    }


def compute_vocabulary_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                              vocab_size: int, ignore_index: int = 0) -> Dict[str, float]:
    """
    Compute vocabulary-related metrics.
    
    Args:
        predictions: Model predictions [batch_size, seq_len, vocab_size]
        targets: Target sequences [batch_size, seq_len]
        vocab_size: Size of vocabulary
        ignore_index: Index to ignore in computation
        
    Returns:
        Dictionary with vocabulary metrics
    """
    with torch.no_grad():
        # Get predicted tokens
        pred_indices = torch.argmax(predictions, dim=-1)
        
        # Count unique tokens
        pred_tokens = pred_indices[pred_indices != ignore_index]
        target_tokens = targets[targets != ignore_index]
        
        pred_vocab_size = len(torch.unique(pred_tokens))
        target_vocab_size = len(torch.unique(target_tokens))
        
        # Vocabulary coverage
        vocab_coverage = pred_vocab_size / vocab_size if vocab_size > 0 else 0.0
        
        # Token frequency analysis
        pred_token_counts = torch.bincount(pred_tokens, minlength=vocab_size)
        target_token_counts = torch.bincount(target_tokens, minlength=vocab_size)
        
        # Most common token overlap
        top_k = 100
        pred_top_k = torch.topk(pred_token_counts, min(top_k, vocab_size)).indices
        target_top_k = torch.topk(target_token_counts, min(top_k, vocab_size)).indices
        
        overlap = len(set(pred_top_k.tolist()) & set(target_top_k.tolist()))
        overlap_ratio = overlap / top_k if top_k > 0 else 0.0
    
    return {
        'pred_vocab_size': pred_vocab_size,
        'target_vocab_size': target_vocab_size,
        'vocab_coverage': vocab_coverage,
        'top_k_overlap': overlap,
        'top_k_overlap_ratio': overlap_ratio
    }


def evaluate_translations(model, data_loader, src_vocab, tgt_vocab, 
                         device: torch.device, max_length: int = 50,
                         beam_size: int = 1) -> Dict[str, Any]:
    """
    Evaluate model on a dataset and compute various metrics.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: Device to run evaluation on
        max_length: Maximum generation length
        beam_size: Beam size for decoding (1 for greedy)
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_src_sentences = []
    all_tgt_sentences = []
    all_pred_sentences = []
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            src = batch['src_seq'].to(device)
            tgt = batch['tgt_seq'].to(device)
            
            # Forward pass
            if hasattr(model, 'forward'):
                outputs = model(src, tgt)
            else:
                # Handle encoder-decoder models
                enc_output = model.encode(src)
                outputs = model.decode(tgt, enc_output)
            
            # Compute loss
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                tgt.view(-1),
                ignore_index=tgt_vocab.pad_idx,
                reduction='mean'
            )
            total_loss += loss.item()
            num_batches += 1
            
            # Generate translations
            for i in range(src.size(0)):
                src_sentence = src_vocab.decode(src[i].tolist(), remove_special=True)
                tgt_sentence = tgt_vocab.decode(tgt[i].tolist(), remove_special=True)
                
                # Generate prediction
                if beam_size == 1:
                    pred_sentence = greedy_decode(model, src[i:i+1], src_vocab, tgt_vocab, 
                                                device, max_length)
                else:
                    pred_sentence = beam_decode(model, src[i:i+1], src_vocab, tgt_vocab,
                                              device, max_length, beam_size)
                
                all_src_sentences.append(src_sentence)
                all_tgt_sentences.append(tgt_sentence)
                all_pred_sentences.append(pred_sentence)
                
                # Tokenize for BLEU
                tgt_tokens = tokenize_sentence(tgt_sentence)
                pred_tokens = tokenize_sentence(pred_sentence)
                
                all_targets.append(tgt_tokens)
                all_predictions.append(pred_tokens)
    
    # Compute metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    bleu_scores = compute_bleu_scores(all_targets, all_predictions)
    
    results = {
        'loss': avg_loss,
        'bleu_scores': bleu_scores,
        'num_samples': len(all_predictions),
        'examples': list(zip(all_src_sentences[:5], all_tgt_sentences[:5], all_pred_sentences[:5]))
    }
    
    return results


def greedy_decode(model, src: torch.Tensor, src_vocab, tgt_vocab, 
                 device: torch.device, max_length: int = 50) -> str:
    """
    Perform greedy decoding.
    
    Args:
        model: Trained model
        src: Source sequence [1, src_len]
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: Device to run on
        max_length: Maximum generation length
        
    Returns:
        Decoded sentence
    """
    model.eval()
    
    with torch.no_grad():
        # Encode source
        if hasattr(model, 'encode'):
            enc_output = model.encode(src)
        else:
            # For RNN models
            enc_output, _ = model.encoder(src)
        
        # Initialize decoder input
        decoder_input = torch.tensor([[tgt_vocab.start_idx]], device=device)
        
        decoded_tokens = []
        
        for _ in range(max_length):
            # Decode step
            if hasattr(model, 'decode'):
                output = model.decode(decoder_input, enc_output)
            else:
                # For RNN models
                output, _ = model.decoder.forward_step(decoder_input, None, enc_output)
            
            # Get next token
            next_token = torch.argmax(output[:, -1:], dim=-1)
            decoded_tokens.append(next_token.item())
            
            # Stop if end token
            if next_token.item() == tgt_vocab.end_idx:
                break
            
            # Update decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
    
    # Decode to string
    return tgt_vocab.decode(decoded_tokens, remove_special=True)


def beam_decode(model, src: torch.Tensor, src_vocab, tgt_vocab,
               device: torch.device, max_length: int = 50, beam_size: int = 5) -> str:
    """
    Perform beam search decoding.
    
    Args:
        model: Trained model
        src: Source sequence [1, src_len]
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: Device to run on
        max_length: Maximum generation length
        beam_size: Beam size
        
    Returns:
        Decoded sentence
    """
    model.eval()
    
    with torch.no_grad():
        # Encode source
        if hasattr(model, 'encode'):
            enc_output = model.encode(src)
        else:
            # For RNN models
            enc_output, _ = model.encoder(src)
        
        # Initialize beam
        beams = [([tgt_vocab.start_idx], 0.0)]  # (tokens, score)
        
        for _ in range(max_length):
            candidates = []
            
            for tokens, score in beams:
                if tokens[-1] == tgt_vocab.end_idx:
                    candidates.append((tokens, score))
                    continue
                
                # Decode step
                decoder_input = torch.tensor([tokens], device=device)
                
                if hasattr(model, 'decode'):
                    output = model.decode(decoder_input, enc_output)
                else:
                    # For RNN models
                    output, _ = model.decoder.forward_step(decoder_input, None, enc_output)
                
                # Get top-k candidates
                log_probs = F.log_softmax(output[:, -1], dim=-1)
                top_k_probs, top_k_indices = torch.topk(log_probs, beam_size, dim=-1)
                
                for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
                    new_tokens = tokens + [idx.item()]
                    new_score = score + prob.item()
                    candidates.append((new_tokens, new_score))
            
            # Select top beam_size candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]
            
            # Check if all beams end with end token
            if all(beam[0][-1] == tgt_vocab.end_idx for beam in beams):
                break
        
        # Return best beam
        best_tokens = beams[0][0]
        return tgt_vocab.decode(best_tokens, remove_special=True) 