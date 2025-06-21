# Notebooks

This folder contains Jupyter notebooks and Python scripts for data exploration, model comparison, and results analysis in the Neural Machine Translation (NMT) project.

## Contents

- **01_data_exploration.ipynb / 01_data_exploration.py**
  - Explore the Spanish-English translation dataset.
  - Analyze data distribution, sentence lengths, vocabulary statistics, and text characteristics.
  - Visualize key properties of the dataset and provide recommendations for preprocessing.

- **02_model_comparison.py**
  - Compare RNN and Transformer models for NMT.
  - Analyze model architectures, parameter counts, training dynamics, and memory usage.
  - Visualize training loss, speed, and resource requirements.
  - Provide recommendations for model selection based on use case.

- **03_results_analysis.py**
  - Analyze the results of trained models.
  - Evaluate translation quality using BLEU scores and error analysis.
  - Visualize performance metrics and provide statistical comparisons.
  - Summarize findings and offer recommendations for future improvements.

## Usage

- You can open the `.ipynb` notebook(s) in JupyterLab, Jupyter Notebook, or VSCode for interactive exploration.
- The `.py` scripts can be run directly with Python for reproducible, non-interactive analysis:
  ```bash
  python 01_data_exploration.py
  python 02_model_comparison.py
  python 03_results_analysis.py
  ```
- Make sure to install the required dependencies listed in the project's `requirements.txt` and download the dataset as described in the main project README.

## Notes
- If the dataset file (`data/spa-eng.txt`) is missing, the notebooks/scripts will use a small sample for demonstration.
- Visualizations and analysis outputs are saved to the `results/` directory.

---
For more details on the project, see the main [README.md](../README.md). 