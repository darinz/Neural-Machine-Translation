"""
Setup script for Neural Machine Translation package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open(this_directory / "requirements.txt", "r", encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)

setup(
    name="Neural-Machine-Translation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive implementation of neural machine translation models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/darinz/Neural-Machine-Translation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "nmt-train=src.cli.train:main",
            "nmt-evaluate=src.cli.evaluate:main",
            "nmt-translate=src.cli.translate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
    keywords=[
        "machine-translation",
        "neural-networks",
        "transformer",
        "rnn",
        "attention",
        "pytorch",
        "nlp",
        "deep-learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/darinz/Neural-Machine-Translation/issues",
        "Source": "https://github.com/darinz/Neural-Machine-Translation",
        "Documentation": "https://neural-machine-translation.readthedocs.io/",
    },
) 