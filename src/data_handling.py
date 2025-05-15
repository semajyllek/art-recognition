"""
Data handling module for the art-recognition system.

This module contains functions for loading the WikiArt dataset,
extracting metadata from captions, and preparing the data for
further processing.
"""

import pandas as pd
from pathlib import Path

def load_artwork_dataset():
    """Load the WikiArt dataset and extract metadata."""
    print("Loading WikiArt dataset...")
    
    # Always load the full dataset
    df = pd.read_csv("hf://datasets/matrixglitch/wikiart-215k/metadata.csv")
    
    # Extract metadata directly - no configurable options
    df['artist'] = df['file_link'].apply(lambda x: Path(x).parts[-2].replace('-', ' '))
    df['title'] = df['wikiart_caption'].apply(lambda caption: 
        caption.split(', by ')[0].strip() if isinstance(caption, str) and ', by ' in caption else "Unknown")
    
    print(f"Loaded {len(df)} artworks")
    return df

