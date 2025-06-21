#!/usr/bin/env python3
"""
Data download script for Neural Machine Translation.

This script downloads the Spanish-English translation dataset from ManyThings.org.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path


def download_file(url: str, filename: str) -> bool:
    """
    Download a file from URL.
    
    Args:
        url: URL to download from
        filename: Local filename to save as
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        print(f"Downloading {filename} from {url}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """
    Extract a ZIP file.
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract to
        
    Returns:
        True if extraction successful, False otherwise
    """
    try:
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted {zip_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to extract {zip_path}: {e}")
        return False


def main():
    """Download and prepare the dataset."""
    print("📥 Neural Machine Translation - Data Download")
    print("=" * 50)
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Dataset information
    dataset_url = "http://www.manythings.org/anki/spa-eng.zip"
    zip_filename = "spa-eng.zip"
    zip_path = data_dir / zip_filename
    txt_filename = "spa.txt"
    txt_path = data_dir / txt_filename
    
    # Check if data already exists
    if txt_path.exists():
        print(f"✅ Dataset already exists at {txt_path}")
        print("   Skipping download...")
        return
    
    # Download dataset
    if not download_file(dataset_url, str(zip_path)):
        print("❌ Failed to download dataset")
        return
    
    # Extract dataset
    if not extract_zip(str(zip_path), str(data_dir)):
        print("❌ Failed to extract dataset")
        return
    
    # Clean up ZIP file
    try:
        zip_path.unlink()
        print(f"🗑️  Removed {zip_filename}")
    except Exception as e:
        print(f"⚠️  Could not remove {zip_filename}: {e}")
    
    # Verify extraction
    if txt_path.exists():
        # Count lines
        with open(txt_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        
        print(f"✅ Dataset ready!")
        print(f"   File: {txt_path}")
        print(f"   Lines: {line_count:,}")
        print(f"   Size: {txt_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Show sample
        print("\n📋 Sample data:")
        with open(txt_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Show first 5 lines
                    break
                eng, spa, _ = line.strip().split('\t')
                print(f"   {eng} → {spa}")
        
        print("\n🎉 Dataset download completed successfully!")
        print("   You can now run the training script.")
        
    else:
        print("❌ Dataset extraction failed - spa.txt not found")


if __name__ == "__main__":
    main() 