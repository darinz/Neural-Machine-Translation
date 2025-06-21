"""
Inference module for neural machine translation models.

This module provides a high-level Translator class for easy model inference
and translation functionality.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union
import time
import json
from pathlib import Path

from ..models import RNNModel, TransformerModel
from ..data.vocabulary import Vocabulary
from ..data.preprocessing import preprocess_text
from ..trainer.metrics import greedy_decode, beam_decode
from ..utils.config import Config
from ..utils.logging import get_logger


class Translator:
    """
    High-level translator class for neural machine translation.
    
    This class provides a simple interface for loading models and performing
    translations with various decoding strategies.
    """
    
    def __init__(self, model_path: str, model_type: str, config_path: str = 'config.yaml',
                 vocab_dir: str = 'data/vocab', device: str = 'auto'):
        """
        Initialize the translator.
        
        Args:
            model_path: Path to model checkpoint
            model_type: Type of model ('rnn' or 'transformer')
            config_path: Path to configuration file
            vocab_dir: Directory containing vocabulary files
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.logger = get_logger(__name__)
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Initializing translator on device: {self.device}")
        
        # Load configuration
        self.config = Config.from_yaml(config_path)
        self.model_type = model_type.lower()
        
        # Load vocabularies
        self.src_vocab, self.tgt_vocab = self._load_vocabularies(vocab_dir)
        
        # Load model
        self.model = self._load_model(model_path, model_type, self.config.model)
        
        self.logger.info(f"Translator initialized with {model_type.upper()} model")
    
    def _load_vocabularies(self, vocab_dir: str) -> tuple:
        """
        Load source and target vocabularies.
        
        Args:
            vocab_dir: Directory containing vocabulary files
            
        Returns:
            Tuple of (src_vocab, tgt_vocab)
        """
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
    
    def _load_model(self, model_path: str, model_type: str, config: Dict[str, Any]) -> nn.Module:
        """
        Load trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            model_type: Type of model ('rnn' or 'transformer')
            config: Model configuration
            
        Returns:
            Loaded model
        """
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
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text for translation.
        
        Args:
            text: Input Spanish text
            
        Returns:
            Preprocessed text
        """
        return preprocess_text(text)
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text into token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize
        tokens = processed_text.split()
        token_ids = [self.src_vocab.get_token_id(token) for token in tokens]
        
        # Add special tokens
        token_ids = [self.src_vocab.start_idx] + token_ids + [self.src_vocab.end_idx]
        
        return token_ids
    
    def translate(self, text: str, beam_size: int = 1, max_length: int = 50,
                 return_timing: bool = False) -> Union[str, tuple]:
        """
        Translate Spanish text to English.
        
        Args:
            text: Spanish text to translate
            beam_size: Beam size for decoding (1 for greedy)
            max_length: Maximum generation length
            return_timing: Whether to return timing information
            
        Returns:
            English translation (and timing if requested)
        """
        # Tokenize input
        token_ids = self.tokenize(text)
        
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
        
        if return_timing:
            return translation, translation_time
        else:
            return translation
    
    def batch_translate(self, texts: List[str], beam_size: int = 1, 
                       max_length: int = 50, return_timing: bool = False) -> Union[List[str], List[tuple]]:
        """
        Translate multiple texts in batch.
        
        Args:
            texts: List of Spanish texts to translate
            beam_size: Beam size for decoding
            max_length: Maximum generation length
            return_timing: Whether to return timing information
            
        Returns:
            List of translations (and timing if requested)
        """
        results = []
        total_start_time = time.time()
        
        for i, text in enumerate(texts):
            if return_timing:
                translation, trans_time = self.translate(text, beam_size, max_length, return_timing=True)
                results.append((translation, trans_time))
            else:
                translation = self.translate(text, beam_size, max_length, return_timing=False)
                results.append(translation)
            
            # Log progress for large batches
            if len(texts) > 10 and (i + 1) % 10 == 0:
                self.logger.info(f"Translated {i + 1}/{len(texts)} texts")
        
        total_time = time.time() - total_start_time
        
        if return_timing:
            return results, total_time
        else:
            return results
    
    def translate_with_confidence(self, text: str, beam_size: int = 5, 
                                max_length: int = 50) -> Dict[str, Any]:
        """
        Translate text with confidence scores using beam search.
        
        Args:
            text: Spanish text to translate
            beam_size: Beam size for decoding
            max_length: Maximum generation length
            
        Returns:
            Dictionary with translation and confidence information
        """
        # Tokenize input
        token_ids = self.tokenize(text)
        src_tensor = torch.tensor([token_ids], device=self.device)
        
        # Get multiple candidates using beam search
        candidates = []
        self.model.eval()
        
        with torch.no_grad():
            # Encode source
            if hasattr(self.model, 'encode'):
                enc_output = self.model.encode(src_tensor)
            else:
                # For RNN models
                enc_output, _ = self.model.encoder(src_tensor)
            
            # Initialize beam
            beams = [([self.tgt_vocab.start_idx], 0.0)]
            
            for _ in range(max_length):
                new_beams = []
                
                for tokens, score in beams:
                    if tokens[-1] == self.tgt_vocab.end_idx:
                        new_beams.append((tokens, score))
                        continue
                    
                    # Decode step
                    decoder_input = torch.tensor([tokens], device=self.device)
                    
                    if hasattr(self.model, 'decode'):
                        output = self.model.decode(decoder_input, enc_output)
                    else:
                        # For RNN models
                        output, _ = self.model.decoder.forward_step(decoder_input, None, enc_output)
                    
                    # Get top-k candidates
                    import torch.nn.functional as F
                    log_probs = F.log_softmax(output[:, -1], dim=-1)
                    top_k_probs, top_k_indices = torch.topk(log_probs, beam_size, dim=-1)
                    
                    for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
                        new_tokens = tokens + [idx.item()]
                        new_score = score + prob.item()
                        new_beams.append((new_tokens, new_score))
                
                # Select top beam_size candidates
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_size]
                
                # Check if all beams end with end token
                if all(beam[0][-1] == self.tgt_vocab.end_idx for beam in beams):
                    break
            
            # Get top candidates
            for tokens, score in beams[:3]:  # Top 3 candidates
                translation = self.tgt_vocab.decode(tokens, remove_special=True)
                confidence = torch.exp(torch.tensor(score)).item()
                candidates.append({
                    'translation': translation,
                    'score': score,
                    'confidence': confidence
                })
        
        return {
            'source': text,
            'best_translation': candidates[0]['translation'],
            'candidates': candidates,
            'beam_size': beam_size
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        params = self.model.count_parameters()
        
        return {
            'model_type': self.model_type,
            'device': str(self.device),
            'src_vocab_size': len(self.src_vocab),
            'tgt_vocab_size': len(self.tgt_vocab),
            'total_parameters': params['total'],
            'trainable_parameters': params['trainable'],
            'encoder_parameters': params['encoder'],
            'decoder_parameters': params['decoder']
        }
    
    def save_translations(self, translations: List[Dict[str, Any]], 
                         output_file: str):
        """
        Save translations to a file.
        
        Args:
            translations: List of translation dictionaries
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translations, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(translations)} translations to {output_file}")
    
    def benchmark(self, test_texts: List[str], beam_sizes: List[int] = [1, 5, 10],
                 max_length: int = 50) -> Dict[str, Any]:
        """
        Benchmark translation performance with different settings.
        
        Args:
            test_texts: List of test texts
            beam_sizes: List of beam sizes to test
            max_length: Maximum generation length
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        for beam_size in beam_sizes:
            self.logger.info(f"Benchmarking with beam size {beam_size}")
            
            start_time = time.time()
            translations, total_time = self.batch_translate(
                test_texts, beam_size, max_length, return_timing=True
            )
            
            avg_time = total_time / len(test_texts)
            
            results[f'beam_{beam_size}'] = {
                'total_time': total_time,
                'avg_time_per_text': avg_time,
                'texts_per_second': len(test_texts) / total_time,
                'translations': translations
            }
        
        return results


# Convenience functions for easy usage
def create_translator(model_path: str, model_type: str, **kwargs) -> Translator:
    """
    Create a translator instance with default settings.
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model ('rnn' or 'transformer')
        **kwargs: Additional arguments for Translator
        
    Returns:
        Translator instance
    """
    return Translator(model_path, model_type, **kwargs)


def quick_translate(text: str, model_path: str, model_type: str, **kwargs) -> str:
    """
    Quick translation function for single text.
    
    Args:
        text: Spanish text to translate
        model_path: Path to model checkpoint
        model_type: Type of model ('rnn' or 'transformer')
        **kwargs: Additional arguments for Translator
        
    Returns:
        English translation
    """
    translator = create_translator(model_path, model_type, **kwargs)
    return translator.translate(text) 