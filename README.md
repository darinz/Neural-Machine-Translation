# Neural Machine Translation: RNN vs Transformer

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A comprehensive implementation of neural machine translation models comparing Recurrent Neural Networks (RNN) with Attention and Transformer architectures for Spanish-to-English translation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements and compares two state-of-the-art neural machine translation approaches:

1. **RNN with Attention**: Based on the Bahdanau attention mechanism
2. **Transformer**: Based on the "Attention is All You Need" architecture

Both models are trained on the Spanish-English translation dataset from [ManyThings.org](http://www.manythings.org/anki/spa-eng.zip) and evaluated using BLEU scores.

## Features

- **Dual Architecture Implementation**: Both RNN and Transformer models
- **Attention Mechanisms**: Bahdanau attention for RNN, multi-head attention for Transformer
- **Modern PyTorch**: Built with PyTorch 2.0+ features and best practices
- **Comprehensive Evaluation**: BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scoring
- **GPU Support**: Automatic device detection and GPU acceleration
- **Modular Design**: Clean, maintainable code structure
- **Logging & Monitoring**: TensorBoard and Weights & Biases integration
- **Data Preprocessing**: Robust text preprocessing and vocabulary management

## Architecture

### RNN with Attention
- **Encoder**: Bidirectional GRU with attention mechanism
- **Decoder**: GRU with Bahdanau attention
- **Attention**: Computes context vectors using encoder hidden states

### Transformer
- **Encoder**: Multi-layer transformer encoder with positional embeddings
- **Decoder**: Multi-layer transformer decoder with causal masking
- **Attention**: Multi-head self-attention and cross-attention mechanisms

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/darinz/Neural-Machine-Translation.git
   cd Neural-Machine-Translation
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt')"
   ```

## Usage

### Quick Start

```python
from src.models import RNNModel, TransformerModel
from src.data import TranslationDataset
from src.trainer import Trainer

# Initialize models
rnn_model = RNNModel(src_vocab_size, tgt_vocab_size, embedding_dim=256, hidden_dim=512)
transformer_model = TransformerModel(src_vocab_size, tgt_vocab_size, d_model=512, nhead=8)

# Train models
trainer = Trainer(device='cuda' if torch.cuda.is_available() else 'cpu')
trainer.train(rnn_model, train_loader, epochs=10)
trainer.train(transformer_model, train_loader, epochs=10)
```

### Training Scripts

```bash
# Train RNN model
python train.py --model rnn --epochs 10 --batch-size 64

# Train Transformer model
python train.py --model transformer --epochs 10 --batch-size 64

# Evaluate models
python evaluate.py --model rnn
python evaluate.py --model transformer
```

### Interactive Translation

```python
from src.inference import Translator

translator = Translator('checkpoints/transformer_best.pt')
translation = translator.translate("Hola, ¿cómo estás?")
print(translation)  # "Hello, how are you?"
```

## Results

### Performance Metrics

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Training Time |
|-------|--------|--------|--------|--------|---------------|
| RNN + Attention | 0.312 | 0.089 | 0.067 | 0.061 | ~2 hours |
| Transformer | 0.345 | 0.102 | 0.078 | 0.072 | ~1.5 hours |

### Sample Translations

| Spanish | RNN Translation | Transformer Translation |
|---------|----------------|------------------------|
| "Hola, ¿cómo estás?" | "Hello, how are you?" | "Hello, how are you?" |
| "Me gusta el café" | "I like coffee" | "I like coffee" |
| "¿Dónde está la biblioteca?" | "Where is the library?" | "Where is the library?" |

## Project Structure

```
neural-machine-translation/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── rnn.py          # RNN encoder-decoder with attention
│   │   ├── transformer.py  # Transformer architecture
│   │   └── attention.py    # Attention mechanisms
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py      # Custom dataset classes
│   │   ├── preprocessing.py # Text preprocessing utilities
│   │   └── vocabulary.py   # Vocabulary management
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── trainer.py      # Training loop and utilities
│   │   └── metrics.py      # Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       ├── config.py       # Configuration management
│       └── logging.py      # Logging utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_results_analysis.ipynb
├── checkpoints/            # Model checkpoints
├── data/                   # Dataset files
├── logs/                   # Training logs
├── results/                # Evaluation results
├── requirements.txt
├── setup.py
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── translate.py           # Interactive translation
└── README.md
```

## Configuration

The project uses a configuration system for easy hyperparameter management:

```python
# config.yaml
model:
  rnn:
    embedding_dim: 256
    hidden_dim: 512
    num_layers: 2
    dropout: 0.1
  
  transformer:
    d_model: 512
    nhead: 8
    num_layers: 6
    dim_feedforward: 2048
    dropout: 0.1

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 10
  warmup_steps: 4000
  max_grad_norm: 1.0

data:
  max_length: 50
  min_freq: 2
  train_split: 0.8
  val_split: 0.1
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_models.py
pytest tests/test_data.py
pytest tests/test_trainer.py
```

## Monitoring

The project integrates with popular monitoring tools:

### TensorBoard
```bash
tensorboard --logdir logs/
```

### Weights & Biases
```python
import wandb
wandb.init(project="neural-machine-translation")
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## References

- [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset provided by [ManyThings.org](http://www.manythings.org/)
- PyTorch team for the excellent deep learning framework
- The open-source community for inspiration and tools
---

⭐ If you find this project helpful, please give it a star! 