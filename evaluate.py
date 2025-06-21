#!/usr/bin/env python3
"""
Evaluation script for neural machine translation models.

This script evaluates trained models on test data and generates comprehensive
performance reports including BLEU scores, accuracy, and translation examples.
"""

import argparse
import torch
import torch.nn as nn
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# Add src to path
sys.path.append('src')

from models import RNNModel, TransformerModel
from data.vocabulary import Vocabulary
from data.dataset import TranslationDataset
from data.preprocessing import preprocess_text
from trainer.metrics import compute_bleu_scores, evaluate_translations
from utils.config import Config
from utils.logging import get_logger


def load_model(model_path: str, model_type: str, config: Dict[str, Any], device: torch.device):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model ('rnn' or 'transformer')
        config: Model configuration
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    logger = get_logger(__name__)
    
    # Create model
    if model_type.lower() == 'rnn':
        model = RNNModel(
            src_vocab_size=config['src_vocab_size'],
            tgt_vocab_size=config['tgt_vocab_size'],
            embedding_dim=config.get('embedding_dim', 256),
            hidden_dim=config.get('hidden_dim', 512),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1)
        )
    elif model_type.lower() == 'transformer':
        model = TransformerModel(
            src_vocab_size=config['src_vocab_size'],
            tgt_vocab_size=config['tgt_vocab_size'],
            d_model=config.get('d_model', 512),
            n_heads=config.get('n_heads', 8),
            num_layers=config.get('num_layers', 6),
            dim_feedforward=config.get('dim_feedforward', 2048),
            dropout=config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded {model_type.upper()} model from {model_path}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def load_vocabularies(vocab_dir: str, src_vocab_name: str = 'src_vocab.json', 
                     tgt_vocab_name: str = 'tgt_vocab.json'):
    """
    Load source and target vocabularies.
    
    Args:
        vocab_dir: Directory containing vocabulary files
        src_vocab_name: Source vocabulary filename
        tgt_vocab_name: Target vocabulary filename
        
    Returns:
        Tuple of (src_vocab, tgt_vocab)
    """
    vocab_path = Path(vocab_dir)
    
    src_vocab_path = vocab_path / src_vocab_name
    tgt_vocab_path = vocab_path / tgt_vocab_name
    
    if not src_vocab_path.exists():
        raise FileNotFoundError(f"Source vocabulary not found: {src_vocab_path}")
    if not tgt_vocab_path.exists():
        raise FileNotFoundError(f"Target vocabulary not found: {tgt_vocab_path}")
    
    src_vocab = Vocabulary.load(str(src_vocab_path))
    tgt_vocab = Vocabulary.load(str(tgt_vocab_path))
    
    return src_vocab, tgt_vocab


def create_test_dataset(test_file: str, src_vocab: Vocabulary, tgt_vocab: Vocabulary, 
                       max_length: int = 50):
    """
    Create test dataset from file.
    
    Args:
        test_file: Path to test data file
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        max_length: Maximum sequence length
        
    Returns:
        Test dataset
    """
    logger = get_logger(__name__)
    
    if not os.path.exists(test_file):
        logger.warning(f"Test file not found: {test_file}. Creating sample dataset.")
        # Create sample test data
        sample_data = [
            ("Hola, ¿cómo estás?", "Hello, how are you?"),
            ("Me gusta este lugar.", "I like this place."),
            ("¿Dónde está el restaurante?", "Where is the restaurant?"),
            ("Gracias por tu ayuda.", "Thank you for your help."),
            ("¿Qué hora es?", "What time is it?")
        ]
        
        # Save sample data
        with open(test_file, 'w', encoding='utf-8') as f:
            for src, tgt in sample_data:
                f.write(f"{src}\t{tgt}\n")
        
        logger.info(f"Created sample test file: {test_file}")
    
    dataset = TranslationDataset(
        data_file=test_file,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_length=max_length
    )
    
    logger.info(f"Created test dataset with {len(dataset)} samples")
    return dataset


def evaluate_model(model, dataset, src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                  device: torch.device, batch_size: int = 32, beam_size: int = 5,
                  max_length: int = 50) -> Dict[str, Any]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained model
        dataset: Test dataset
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        beam_size: Beam size for decoding
        max_length: Maximum generation length
        
    Returns:
        Dictionary with evaluation results
    """
    logger = get_logger(__name__)
    
    from torch.utils.data import DataLoader
    
    # Create data loader
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Starting evaluation with {len(dataset)} test samples")
    logger.info(f"Batch size: {batch_size}, Beam size: {beam_size}")
    
    # Evaluate using the metrics module
    results = evaluate_translations(
        model=model,
        data_loader=test_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        max_length=max_length,
        beam_size=beam_size
    )
    
    logger.info(f"Evaluation completed. Loss: {results['loss']:.4f}")
    logger.info(f"BLEU scores: {results['bleu_scores']}")
    
    return results


def generate_translation_examples(model, dataset, src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                                 device: torch.device, num_examples: int = 10) -> List[Dict[str, Any]]:
    """
    Generate translation examples for analysis.
    
    Args:
        model: Trained model
        dataset: Test dataset
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: Device to run on
        num_examples: Number of examples to generate
        
    Returns:
        List of translation examples
    """
    logger = get_logger(__name__)
    
    examples = []
    model.eval()
    
    with torch.no_grad():
        for i in range(min(num_examples, len(dataset))):
            sample = dataset[i]
            src_seq = sample['src_seq'].unsqueeze(0).to(device)
            tgt_seq = sample['tgt_seq'].unsqueeze(0).to(device)
            
            # Get reference translation
            reference = tgt_vocab.decode(tgt_seq[0].tolist(), remove_special=True)
            source = src_vocab.decode(src_seq[0].tolist(), remove_special=True)
            
            # Generate prediction
            if hasattr(model, 'encode') and hasattr(model, 'decode'):
                # Transformer model
                enc_output = model.encode(src_seq)
                prediction = model.decode(tgt_seq, enc_output)
            else:
                # RNN model
                prediction = model(src_seq, tgt_seq)
            
            # Get predicted tokens
            pred_tokens = torch.argmax(prediction, dim=-1)[0].tolist()
            predicted = tgt_vocab.decode(pred_tokens, remove_special=True)
            
            # Calculate BLEU score
            from trainer.metrics import tokenize_sentence
            ref_tokens = tokenize_sentence(reference)
            pred_tokens = tokenize_sentence(predicted)
            
            bleu_score = compute_bleu_scores([ref_tokens], [pred_tokens])
            
            examples.append({
                'source': source,
                'reference': reference,
                'prediction': predicted,
                'bleu_scores': bleu_score
            })
    
    logger.info(f"Generated {len(examples)} translation examples")
    return examples


def save_evaluation_results(results: Dict[str, Any], examples: List[Dict[str, Any]], 
                           output_dir: str, model_name: str):
    """
    Save evaluation results to files.
    
    Args:
        results: Evaluation results
        examples: Translation examples
        output_dir: Output directory
        model_name: Name of the model
    """
    logger = get_logger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save main results
    results_file = output_path / f"{model_name}_evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save examples
    examples_file = output_path / f"{model_name}_translation_examples.json"
    with open(examples_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    # Save detailed report
    report_file = output_path / f"{model_name}_evaluation_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"EVALUATION REPORT FOR {model_name.upper()} MODEL\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"  Loss: {results['loss']:.4f}\n")
        f.write(f"  Number of samples: {results['num_samples']}\n\n")
        
        f.write("BLEU SCORES:\n")
        for metric, score in results['bleu_scores'].items():
            f.write(f"  {metric}: {score:.4f}\n")
        f.write("\n")
        
        f.write("TRANSLATION EXAMPLES:\n")
        for i, example in enumerate(examples, 1):
            f.write(f"\nExample {i}:\n")
            f.write(f"  Source: {example['source']}\n")
            f.write(f"  Reference: {example['reference']}\n")
            f.write(f"  Prediction: {example['prediction']}\n")
            f.write(f"  BLEU-4: {example['bleu_scores']['bleu_4']:.4f}\n")
    
    logger.info(f"Saved evaluation results to {output_path}")
    logger.info(f"  - Results: {results_file}")
    logger.info(f"  - Examples: {examples_file}")
    logger.info(f"  - Report: {report_file}")


def print_evaluation_summary(results: Dict[str, Any], examples: List[Dict[str, Any]], 
                           model_name: str):
    """
    Print evaluation summary to console.
    
    Args:
        results: Evaluation results
        examples: Translation examples
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY: {model_name.upper()} MODEL")
    print(f"{'='*60}")
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Number of samples: {results['num_samples']}")
    
    print(f"\nBLEU SCORES:")
    for metric, score in results['bleu_scores'].items():
        print(f"  {metric}: {score:.4f}")
    
    print(f"\nTRANSLATION EXAMPLES:")
    for i, example in enumerate(examples[:5], 1):  # Show first 5 examples
        print(f"\n  Example {i}:")
        print(f"    Source: {example['source']}")
        print(f"    Reference: {example['reference']}")
        print(f"    Prediction: {example['prediction']}")
        print(f"    BLEU-4: {example['bleu_scores']['bleu_4']:.4f}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate neural machine translation models')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['rnn', 'transformer'], required=True,
                       help='Type of model (rnn or transformer)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test_file', type=str, default='data/test.txt',
                       help='Path to test data file')
    parser.add_argument('--vocab_dir', type=str, default='data/vocab',
                       help='Directory containing vocabulary files')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--beam_size', type=int, default=5,
                       help='Beam size for decoding')
    parser.add_argument('--max_length', type=int, default=50,
                       help='Maximum generation length')
    parser.add_argument('--num_examples', type=int, default=10,
                       help='Number of translation examples to generate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = get_logger(__name__)
    logger.info("Starting model evaluation")
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    try:
        # Load configuration
        config = Config.from_yaml(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Load vocabularies
        src_vocab, tgt_vocab = load_vocabularies(args.vocab_dir)
        logger.info(f"Loaded vocabularies - Source: {len(src_vocab)}, Target: {len(tgt_vocab)}")
        
        # Load model
        model = load_model(args.model_path, args.model_type, config.model, device)
        
        # Create test dataset
        dataset = create_test_dataset(args.test_file, src_vocab, tgt_vocab, args.max_length)
        
        # Evaluate model
        start_time = time.time()
        results = evaluate_model(
            model=model,
            dataset=dataset,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
            max_length=args.max_length
        )
        evaluation_time = time.time() - start_time
        
        # Generate examples
        examples = generate_translation_examples(
            model=model,
            dataset=dataset,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            num_examples=args.num_examples
        )
        
        # Add timing information
        results['evaluation_time'] = evaluation_time
        results['samples_per_second'] = len(dataset) / evaluation_time
        
        # Save results
        model_name = f"{args.model_type}_{Path(args.model_path).stem}"
        save_evaluation_results(results, examples, args.output_dir, model_name)
        
        # Print summary
        print_evaluation_summary(results, examples, model_name)
        
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        logger.info(f"Processing speed: {results['samples_per_second']:.2f} samples/second")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main() 