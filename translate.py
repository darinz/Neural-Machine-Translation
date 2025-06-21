#!/usr/bin/env python3
"""
Interactive translation script for neural machine translation models.

This script provides an interactive interface for translating Spanish text
to English using trained RNN or Transformer models.
"""

import argparse
import torch
import sys
import os
from pathlib import Path
from typing import Optional, List
import time

# Add src to path
sys.path.append('src')

from models import RNNModel, TransformerModel
from data.vocabulary import Vocabulary
from data.preprocessing import preprocess_text
from trainer.metrics import greedy_decode, beam_decode
from utils.config import Config
from utils.logging import get_logger


class InteractiveTranslator:
    """
    Interactive translator for real-time translation.
    """
    
    def __init__(self, model_path: str, model_type: str, config: dict, 
                 vocab_dir: str, device: torch.device):
        """
        Initialize the translator.
        
        Args:
            model_path: Path to model checkpoint
            model_type: Type of model ('rnn' or 'transformer')
            config: Model configuration
            vocab_dir: Directory containing vocabulary files
            device: Device to run on
        """
        self.logger = get_logger(__name__)
        self.device = device
        self.model_type = model_type
        
        # Load vocabularies
        self.src_vocab, self.tgt_vocab = self._load_vocabularies(vocab_dir)
        
        # Load model
        self.model = self._load_model(model_path, model_type, config)
        
        self.logger.info(f"Interactive translator initialized with {model_type.upper()} model")
    
    def _load_vocabularies(self, vocab_dir: str):
        """Load source and target vocabularies."""
        vocab_path = Path(vocab_dir)
        
        src_vocab_path = vocab_path / 'src_vocab.json'
        tgt_vocab_path = vocab_path / 'tgt_vocab.json'
        
        if not src_vocab_path.exists():
            raise FileNotFoundError(f"Source vocabulary not found: {src_vocab_path}")
        if not tgt_vocab_path.exists():
            raise FileNotFoundError(f"Target vocabulary not found: {tgt_vocab_path}")
        
        src_vocab = Vocabulary.load(str(src_vocab_path))
        tgt_vocab = Vocabulary.load(str(tgt_vocab_path))
        
        self.logger.info(f"Loaded vocabularies - Source: {len(src_vocab)}, Target: {len(tgt_vocab)}")
        return src_vocab, tgt_vocab
    
    def _load_model(self, model_path: str, model_type: str, config: dict):
        """Load trained model from checkpoint."""
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
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"Loaded {model_type.upper()} model from {model_path}")
        return model
    
    def preprocess_input(self, text: str) -> str:
        """
        Preprocess input text for translation.
        
        Args:
            text: Input Spanish text
            
        Returns:
            Preprocessed text
        """
        # Apply preprocessing
        processed_text = preprocess_text(text)
        return processed_text
    
    def translate(self, text: str, beam_size: int = 1, max_length: int = 50) -> str:
        """
        Translate Spanish text to English.
        
        Args:
            text: Spanish text to translate
            beam_size: Beam size for decoding (1 for greedy)
            max_length: Maximum generation length
            
        Returns:
            English translation
        """
        # Preprocess input
        processed_text = self.preprocess_input(text)
        
        # Tokenize
        tokens = processed_text.split()
        token_ids = [self.src_vocab.get_token_id(token) for token in tokens]
        
        # Add start token
        token_ids = [self.src_vocab.start_idx] + token_ids + [self.src_vocab.end_idx]
        
        # Convert to tensor
        src_tensor = torch.tensor([token_ids], device=self.device)
        
        # Translate
        start_time = time.time()
        
        if beam_size == 1:
            translation = greedy_decode(
                model=self.model,
                src=src_tensor,
                src_vocab=self.src_vocab,
                tgt_vocab=self.tgt_vocab,
                device=self.device,
                max_length=max_length
            )
        else:
            translation = beam_decode(
                model=self.model,
                src=src_tensor,
                src_vocab=self.src_vocab,
                tgt_vocab=self.tgt_vocab,
                device=self.device,
                max_length=max_length,
                beam_size=beam_size
            )
        
        translation_time = time.time() - start_time
        
        return translation, translation_time
    
    def batch_translate(self, texts: List[str], beam_size: int = 1, 
                       max_length: int = 50) -> List[tuple]:
        """
        Translate multiple texts in batch.
        
        Args:
            texts: List of Spanish texts to translate
            beam_size: Beam size for decoding
            max_length: Maximum generation length
            
        Returns:
            List of (translation, time) tuples
        """
        results = []
        
        for text in texts:
            translation, trans_time = self.translate(text, beam_size, max_length)
            results.append((translation, trans_time))
        
        return results


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*60)
    print("🌍 INTERACTIVE SPANISH-ENGLISH TRANSLATOR")
    print("="*60)
    print("Type Spanish text to translate to English.")
    print("Commands:")
    print("  - 'quit' or 'exit': Exit the translator")
    print("  - 'help': Show this help message")
    print("  - 'batch': Enter batch translation mode")
    print("  - 'settings': Change translation settings")
    print("="*60 + "\n")


def print_help():
    """Print help information."""
    print("\n📖 HELP:")
    print("  • Simply type Spanish text and press Enter to translate")
    print("  • Use 'quit' or 'exit' to close the translator")
    print("  • Use 'batch' to translate multiple sentences at once")
    print("  • Use 'settings' to adjust beam size and max length")
    print("  • The translator supports both RNN and Transformer models")
    print()


def get_translation_settings() -> tuple:
    """Get translation settings from user."""
    print("\n⚙️  TRANSLATION SETTINGS:")
    
    try:
        beam_size = int(input("Beam size (1-10, default 1): ") or "1")
        beam_size = max(1, min(10, beam_size))
    except ValueError:
        beam_size = 1
        print("Invalid beam size, using default: 1")
    
    try:
        max_length = int(input("Max length (10-100, default 50): ") or "50")
        max_length = max(10, min(100, max_length))
    except ValueError:
        max_length = 50
        print("Invalid max length, using default: 50")
    
    return beam_size, max_length


def interactive_mode(translator: InteractiveTranslator):
    """Run interactive translation mode."""
    beam_size, max_length = 1, 50
    
    while True:
        try:
            # Get input
            text = input("\n🇪🇸 Spanish: ").strip()
            
            # Handle commands
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif text.lower() in ['help', 'h']:
                print_help()
                continue
            elif text.lower() == 'batch':
                batch_mode(translator, beam_size, max_length)
                continue
            elif text.lower() == 'settings':
                beam_size, max_length = get_translation_settings()
                continue
            elif not text:
                continue
            
            # Translate
            print("🔄 Translating...")
            translation, trans_time = translator.translate(text, beam_size, max_length)
            
            # Display result
            print(f"🇬🇧 English: {translation}")
            print(f"⏱️  Time: {trans_time:.3f}s")
            
            if beam_size > 1:
                print(f"🔍 Beam size: {beam_size}")
            print(f"📏 Max length: {max_length}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again or type 'help' for assistance.")


def batch_mode(translator: InteractiveTranslator, beam_size: int, max_length: int):
    """Run batch translation mode."""
    print("\n📝 BATCH TRANSLATION MODE")
    print("Enter multiple Spanish sentences, one per line.")
    print("Type 'done' on a new line when finished.")
    print("Type 'cancel' to return to interactive mode.\n")
    
    texts = []
    
    while True:
        text = input("🇪🇸 Spanish: ").strip()
        
        if text.lower() == 'done':
            break
        elif text.lower() == 'cancel':
            print("❌ Batch translation cancelled.")
            return
        elif text:
            texts.append(text)
    
    if not texts:
        print("❌ No text to translate.")
        return
    
    print(f"\n🔄 Translating {len(texts)} sentences...")
    start_time = time.time()
    
    results = translator.batch_translate(texts, beam_size, max_length)
    
    total_time = time.time() - start_time
    
    # Display results
    print(f"\n📊 TRANSLATION RESULTS:")
    print("="*60)
    
    for i, (text, (translation, trans_time)) in enumerate(zip(texts, results), 1):
        print(f"\n{i}. 🇪🇸 Spanish: {text}")
        print(f"   🇬🇧 English: {translation}")
        print(f"   ⏱️  Time: {trans_time:.3f}s")
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total sentences: {len(texts)}")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Average time per sentence: {total_time/len(texts):.3f}s")
    print(f"   Beam size: {beam_size}")
    print(f"   Max length: {max_length}")


def demo_mode(translator: InteractiveTranslator):
    """Run demo mode with sample translations."""
    print("\n🎯 DEMO MODE - Sample Translations")
    print("="*40)
    
    demo_sentences = [
        "Hola, ¿cómo estás?",
        "Me gusta este lugar.",
        "¿Dónde está el restaurante?",
        "Gracias por tu ayuda.",
        "¿Qué hora es?",
        "El clima está agradable hoy.",
        "Quiero aprender español.",
        "¿Puedes ayudarme?",
        "Muchas gracias.",
        "Estoy muy feliz."
    ]
    
    print("Sample Spanish sentences:")
    for i, sentence in enumerate(demo_sentences, 1):
        print(f"  {i}. {sentence}")
    
    print(f"\n🔄 Translating {len(demo_sentences)} sentences...")
    start_time = time.time()
    
    results = translator.batch_translate(demo_sentences, beam_size=1, max_length=50)
    
    total_time = time.time() - start_time
    
    print(f"\n📊 DEMO RESULTS:")
    print("="*40)
    
    for i, (sentence, (translation, trans_time)) in enumerate(zip(demo_sentences, results), 1):
        print(f"\n{i}. 🇪🇸 {sentence}")
        print(f"   🇬🇧 {translation}")
        print(f"   ⏱️  {trans_time:.3f}s")
    
    print(f"\n📈 Demo completed in {total_time:.3f}s")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Interactive Spanish-English translator')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['rnn', 'transformer'], required=True,
                       help='Type of model (rnn or transformer)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--vocab_dir', type=str, default='data/vocab',
                       help='Directory containing vocabulary files')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode with sample sentences')
    parser.add_argument('--batch', action='store_true',
                       help='Run in batch mode')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = get_logger(__name__)
    logger.info("Starting interactive translator")
    
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
        
        # Initialize translator
        translator = InteractiveTranslator(
            model_path=args.model_path,
            model_type=args.model_type,
            config=config.model,
            vocab_dir=args.vocab_dir,
            device=device
        )
        
        # Run appropriate mode
        if args.demo:
            demo_mode(translator)
        elif args.batch:
            batch_mode(translator, beam_size=1, max_length=50)
        else:
            print_banner()
            interactive_mode(translator)
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        print(f"❌ Error: {e}")
        print("Please check your model path and configuration.")
        raise


if __name__ == "__main__":
    main() 