"""
create_test_split.py - Create a test dataset split for the art-recognition system

This script creates a test dataset split from the WikiArt dataset
for evaluating the performance of the art-recognition system.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional
from pathlib import Path


def create_balanced_test_set(
    wikiart_csv_path: str,
    output_path: str,
    samples_per_artist: int = 5,
    min_artworks_per_artist: int = 10,
    random_seed: int = 42,
    max_artists: Optional[int] = None
) -> pd.DataFrame:
    """
    Create a balanced test set with the same number of samples per artist.
    
    Args:
        wikiart_csv_path: Path to the full WikiArt CSV file
        output_path: Path to save the test CSV file
        samples_per_artist: Number of samples to include per artist
        min_artworks_per_artist: Minimum number of artworks an artist must have
        random_seed: Random seed for reproducibility
        max_artists: Maximum number of artists to include
        
    Returns:
        Test DataFrame
    """
    print(f"Loading WikiArt dataset from {wikiart_csv_path}...")
    df = pd.read_csv(wikiart_csv_path)
    
    print(f"Total artworks in dataset: {len(df)}")
    
    # Extract artist information
    df['artist'] = df['file_link'].apply(lambda x: Path(x).parts[-2].replace('-', ' '))
    
    # Extract title information
    df['title'] = df['wikiart_caption'].apply(lambda caption: 
        caption.split(', by ')[0].strip() if isinstance(caption, str) and ', by ' in caption else "Unknown")
    
    # Count artworks per artist
    artist_counts = df['artist'].value_counts()
    print(f"Total artists: {len(artist_counts)}")
    
    # Filter artists with enough artworks
    valid_artists = artist_counts[artist_counts >= min_artworks_per_artist].index
    print(f"Artists with at least {min_artworks_per_artist} artworks: {len(valid_artists)}")
    
    # Limit number of artists if specified
    if max_artists is not None and max_artists < len(valid_artists):
        valid_artists = valid_artists[:max_artists]
        print(f"Using the top {max_artists} artists with most artworks")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create test set
    test_df = pd.DataFrame()
    
    for artist in valid_artists:
        # Get all artworks by this artist
        artist_df = df[df['artist'] == artist]
        
        # Randomly select test samples
        if len(artist_df) >= samples_per_artist:
            artist_test_df = artist_df.sample(samples_per_artist, random_state=random_seed)
            test_df = pd.concat([test_df, artist_test_df])
    
    # Reset index
    test_df = test_df.reset_index(drop=True)
    
    print("\nTest set statistics:")
    print(f"Total artworks: {len(test_df)}")
    print(f"Total artists: {test_df['artist'].nunique()}")
    print(f"Artworks per artist: {samples_per_artist}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    test_df.to_csv(output_path, index=False)
    
    print(f"\nSaved test set to: {output_path}")
    
    return test_df


def create_variety_test_set(
    wikiart_csv_path: str,
    output_path: str,
    total_samples: int = 500,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Create a test set with a variety of different artists and styles.
    
    Args:
        wikiart_csv_path: Path to the full WikiArt CSV file
        output_path: Path to save the test CSV file
        total_samples: Total number of samples to include
        random_seed: Random seed for reproducibility
        
    Returns:
        Test DataFrame
    """
    print(f"Loading WikiArt dataset from {wikiart_csv_path}...")
    df = pd.read_csv(wikiart_csv_path)
    
    print(f"Total artworks in dataset: {len(df)}")
    
    # Extract artist information
    df['artist'] = df['file_link'].apply(lambda x: Path(x).parts[-2].replace('-', ' '))
    
    # Extract title information
    df['title'] = df['wikiart_caption'].apply(lambda caption: 
        caption.split(', by ')[0].strip() if isinstance(caption, str) and ', by ' in caption else "Unknown")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Try to extract style information if available
    if 'style' in df.columns:
        # Stratified sampling by style
        test_df = df.groupby('style', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(total_samples * len(x) / len(df)))), random_state=random_seed)
        )
    else:
        # Stratified sampling by artist
        test_df = df.groupby('artist', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(total_samples * len(x) / len(df)))), random_state=random_seed)
        )
    
    # If we have too many samples, take a random subset
    if len(test_df) > total_samples:
        test_df = test_df.sample(total_samples, random_state=random_seed)
    
    # Reset index
    test_df = test_df.reset_index(drop=True)
    
    print("\nTest set statistics:")
    print(f"Total artworks: {len(test_df)}")
    print(f"Total artists: {test_df['artist'].nunique()}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    test_df.to_csv(output_path, index=False)
    
    print(f"\nSaved test set to: {output_path}")
    
    return test_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create test datasets for artwork recognition evaluation')
    parser.add_argument('--input_csv', type=str, required=True, 
                        help='Path to the WikiArt CSV file (usually "hf://datasets/matrixglitch/wikiart-215k/train-00000-of-00001.csv")')
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Path to save the test CSV file')
    parser.add_argument('--mode', type=str, choices=['balanced', 'variety'], default='balanced',
                        help='Test set creation mode: balanced (equal samples per artist) or variety (diverse)')
    parser.add_argument('--samples_per_artist', type=int, default=5,
                        help='Number of samples per artist for balanced mode')
    parser.add_argument('--min_artworks', type=int, default=10,
                        help='Minimum number of artworks an artist must have')
    parser.add_argument('--max_artists', type=int, default=None,
                        help='Maximum number of artists to include in balanced mode')
    parser.add_argument('--total_samples', type=int, default=500,
                        help='Total samples for variety mode')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.mode == 'balanced':
        create_balanced_test_set(
            args.input_csv,
            args.output_path,
            samples_per_artist=args.samples_per_artist,
            min_artworks_per_artist=args.min_artworks,
            random_seed=args.random_seed,
            max_artists=args.max_artists
        )
    elif args.mode == 'variety':
        create_variety_test_set(
            args.input_csv,
            args.output_path,
            total_samples=args.total_samples,
            random_seed=args.random_seed
        )