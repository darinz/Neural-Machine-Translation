# Contributing to Neural Machine Translation

Thank you for your interest in contributing to the Neural Machine Translation project! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

- Use the GitHub issue tracker to report bugs
- Include a clear and descriptive title
- Provide detailed steps to reproduce the bug
- Include system information (OS, Python version, etc.)
- Include error messages and stack traces

### Suggesting Enhancements

- Use the GitHub issue tracker for feature requests
- Describe the enhancement clearly
- Explain why this enhancement would be useful
- Provide examples of how it would work

### Pull Requests

- Fork the repository
- Create a feature branch
- Make your changes
- Add tests for new functionality
- Ensure all tests pass
- Update documentation if needed
- Submit a pull request

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/darinz/Neural-Machine-Translation.git
   cd Neural-Machine-Translation
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Download data**
   ```bash
   wget http://www.manythings.org/anki/spa-eng.zip
   unzip spa-eng.zip -d data/
   ```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use type hints for all function parameters and return values
- Use docstrings for all public functions and classes
- Use f-strings for string formatting

### Code Formatting

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run these tools before submitting a pull request:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check types
mypy src/

# Lint code
flake8 src/ tests/
```

### Documentation

- Use Google-style docstrings
- Include type hints
- Provide examples for complex functions
- Update README.md for new features

Example docstring:

```python
def translate_sentence(
    model: nn.Module, 
    sentence: str, 
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    max_length: int = 50
) -> str:
    """Translate a single sentence using the provided model.
    
    Args:
        model: Trained translation model
        sentence: Source sentence to translate
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        max_length: Maximum output length
        
    Returns:
        Translated sentence
        
    Example:
        >>> model = load_model('checkpoints/best.pt')
        >>> translation = translate_sentence(model, "Hola", src_vocab, tgt_vocab)
        >>> print(translation)
        "Hello"
    """
    # Implementation here
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Test both success and failure cases
- Use fixtures for common setup
- Mock external dependencies

Example test:

```python
import pytest
import torch
from src.models import RNNModel

def test_rnn_model_forward():
    """Test RNN model forward pass."""
    model = RNNModel(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        embedding_dim=256,
        hidden_dim=512
    )
    
    batch_size, seq_len = 4, 10
    src = torch.randint(0, 1000, (batch_size, seq_len))
    tgt = torch.randint(0, 1000, (batch_size, seq_len))
    
    output = model(src, tgt)
    
    assert output.shape == (batch_size, seq_len, 1000)
    assert not torch.isnan(output).any()
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes**
   - Write code following our standards
   - Add tests for new functionality
   - Update documentation

3. **Run quality checks**
   ```bash
   pre-commit run --all-files
   pytest
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Create a pull request**
   - Use a clear title
   - Describe the changes in detail
   - Link related issues
   - Request reviews from maintainers

### Pull Request Guidelines

- **Title**: Clear and descriptive
- **Description**: Explain what and why, not how
- **Tests**: Ensure all tests pass
- **Documentation**: Update docs for new features
- **Breaking changes**: Clearly mark and explain

## Reporting Bugs

When reporting bugs, please include:

1. **Environment information**
   - Operating system
   - Python version
   - Package versions (requirements.txt)

2. **Steps to reproduce**
   - Clear, step-by-step instructions
   - Minimal example code

3. **Expected vs actual behavior**
   - What you expected to happen
   - What actually happened

4. **Additional information**
   - Error messages and stack traces
   - Screenshots if relevant
   - Related issues

## Feature Requests

When requesting features, please include:

1. **Problem description**
   - What problem does this solve?
   - Why is this feature needed?

2. **Proposed solution**
   - How should this feature work?
   - Any design considerations?

3. **Alternatives considered**
   - What other approaches were considered?
   - Why was this approach chosen?

4. **Additional context**
   - Related issues or discussions
   - Examples from other projects

## Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check the README and docstrings first

Thank you for contributing to Neural Machine Translation! 