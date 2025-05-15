"""
Data handling module for the ArtRecognition system.

This module contains functions for loading the WikiArt dataset,
extracting metadata from captions, and preparing the data for
further processing.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Union, List, Tuple, Any

def load_artwork_dataset(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load the WikiArt dataset directly using pandas.
    
    Args:
        limit: Optional limit on number of examples to load
        
    Returns:
        DataFrame containing artwork data with extracted metadata
    """
    print("Loading WikiArt dataset...")
    
    # Load dataset from Hugging Face
    try:
        df = pd.read_csv("hf://datasets/matrixglitch/wikiart-215k/train-00000-of-00001.csv")
        print(f"Successfully loaded dataset with {len(df)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise RuntimeError(f"Failed to load WikiArt dataset: {e}")
    
    if limit:
        print(f"Limiting to {limit} examples")
        df = df.head(limit)
    
    print(f"DataFrame shape: {df.shape}")
    
    # Extract metadata
    print("Extracting metadata...")
    df = extract_metadata(df)
    
    # Display sample
    if not df.empty:
        print("\nSample entry:")
        display_columns = ['artist', 'title', 'year', 'file_link']
        display_columns = [col for col in display_columns if col in df.columns]
        print(df[display_columns].iloc[0])
    
    return df

def extract_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract artwork metadata from the dataset.
    
    Args:
        df: DataFrame containing raw WikiArt data
        
    Returns:
        DataFrame with added metadata columns
    """
    # Extract artist from file_link
    df['artist'] = df['file_link'].apply(lambda x: extract_artist_from_path(x))
    
    # Extract title from wikiart_caption
    df['title'] = df['wikiart_caption'].apply(extract_title_from_caption)
    
    # Extract year from wikiart_caption
    df['year'] = df['wikiart_caption'].apply(extract_year_from_caption)
    
    # Extract style/genre from wikiart_caption (optional)
    df['style_genre'] = df['wikiart_caption'].apply(extract_style_genre_from_caption)
    
    return df

def extract_artist_from_path(file_path: str) -> str:
    """
    Extract artist name from the file path.
    
    Args:
        file_path: Path to the artwork image
        
    Returns:
        Artist name
    """
    try:
        # The artist name is typically the second-to-last part of the path
        # Example: https://uploads8.wikiart.org/images/vincent-van-gogh/the-starry-night-1889.jpg
        parts = Path(file_path).parts
        if len(parts) > 1:
            # Convert hyphens to spaces for readability
            return parts[-2].replace('-', ' ')
    except Exception:
        pass
    
    return "Unknown"

def extract_title_from_caption(caption: str) -> str:
    """
    Extract artwork title from the caption.
    
    Args:
        caption: Caption text from WikiArt
        
    Returns:
        Artwork title
    """
    if pd.isna(caption) or not isinstance(caption, str):
        return "Unknown"
    
    # Title is typically before "by" in the caption
    # Example: "Starry Night, by Vincent van Gogh, 1889, Post-Impressionism"
    if ', by ' in caption:
        return caption.split(', by ')[0].strip()
    
    return "Unknown"

def extract_year_from_caption(caption: str) -> Optional[str]:
    """
    Extract creation year from the caption.
    
    Args:
        caption: Caption text from WikiArt
        
    Returns:
        Year or None if not found
    """
    if pd.isna(caption) or not isinstance(caption, str):
        return None
    
    # Year typically comes after artist name, separated by commas
    # Example: "Starry Night, by Vincent van Gogh, 1889, Post-Impressionism"
    if ', by ' in caption and ',' in caption.split(', by ')[1]:
        parts = caption.split(', by ')[1].split(',')
        if len(parts) > 1:
            year_part = parts[1].strip()
            # Check if the first word is a 4-digit number
            if year_part and year_part.split()[0].isdigit():
                return year_part.split()[0]
    
    return None

def extract_style_genre_from_caption(caption: str) -> List[str]:
    """
    Extract style and genre information from the caption.
    
    Args:
        caption: Caption text from WikiArt
        
    Returns:
        List of style/genre tags
    """
    if pd.isna(caption) or not isinstance(caption, str):
        return []
    
    # Style/genre typically comes after the year in the caption
    # Example: "Starry Night, by Vincent van Gogh, 1889, Post-Impressionism, landscape"
    if ', by ' in caption and ',' in caption.split(', by ')[1]:
        parts = caption.split(', by ')[1].split(',')
        if len(parts) > 2:  # Artist, year, then style/genre
            # Skip artist and year, take the rest
            style_genre = [item.strip() for item in parts[2:]]
            return style_genre
    
    return []

def filter_by_artists(df: pd.DataFrame, artist_list: Optional[List[str]] = None, 
                      artist_file: Optional[str] = None) -> pd.DataFrame:
    """
    Filter the dataset to include only specific artists.
    
    Args:
        df: DataFrame containing artwork data
        artist_list: List of artist names to include
        artist_file: Path to file containing artist names (one per line)
        
    Returns:
        Filtered DataFrame
    """
    # If neither list nor file is provided, return original DataFrame
    if artist_list is None and artist_file is None:
        return df
    
    # Load artist names from file if provided
    if artist_file and os.path.exists(artist_file):
        with open(artist_file, 'r', encoding='utf-8') as f:
            artist_list = [line.strip() for line in f]
    
    if not artist_list:
        return df
    
    # Normalize artist names
    artist_list = [name.lower() for name in artist_list]
    
    # Filter DataFrame
    filtered_df = df[df['artist'].str.lower().isin(artist_list)].reset_index(drop=True)
    
    print(f"Filtered to {len(filtered_df)} examples from {len(artist_list)} artists")
    found_artists = set(filtered_df['artist'].str.lower())
    missing_artists = set(artist_list) - found_artists
    
    if missing_artists:
        print(f"Note: Could not find examples for {len(missing_artists)} artists: " +
              f"{', '.join(list(missing_artists)[:5])}" +
              f"{' and more' if len(missing_artists) > 5 else ''}")
    
    return filtered_df

def get_random_sample(df: pd.DataFrame, n: int = 1000, 
                      stratify_by: Optional[str] = 'artist') -> pd.DataFrame:
    """
    Get a stratified random sample from the dataset.
    
    Args:
        df: DataFrame containing artwork data
        n: Number of examples to sample
        stratify_by: Column to stratify by (None for random sampling)
        
    Returns:
        Sampled DataFrame
    """
    if n >= len(df):
        return df
    
    if stratify_by and stratify_by in df.columns:
        # Stratified sampling
        sample_df = pd.DataFrame()
        categories = df[stratify_by].unique()
        
        # Calculate samples per category
        samples_per_category = max(1, n // len(categories))
        
        # Sample from each category
        for category in categories:
            category_df = df[df[stratify_by] == category]
            if len(category_df) <= samples_per_category:
                # Take all samples if there are fewer than needed
                sample_df = pd.concat([sample_df, category_df])
            else:
                # Take a random sample
                category_sample = category_df.sample(samples_per_category)
                sample_df = pd.concat([sample_df, category_sample])
        
        # If we need more samples to reach n, take random samples
        if len(sample_df) < n:
            remaining = n - len(sample_df)
            # Get indexes not already in sample_df
            remaining_df = df[~df.index.isin(sample_df.index)]
            if len(remaining_df) > 0:
                additional = remaining_df.sample(min(remaining, len(remaining_df)))
                sample_df = pd.concat([sample_df, additional])
    else:
        # Simple random sampling
        sample_df = df.sample(n)
    
    return sample_df.reset_index(drop=True)

def save_dataset_info(df: pd.DataFrame, file_path: str) -> None:
    """
    Save information about the dataset to a text file.
    
    Args:
        df: DataFrame containing artwork data
        file_path: Path to save the information
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"Total artworks: {len(df)}\n\n")
        
        # Artist statistics
        f.write("Artist statistics:\n")
        artist_counts = df['artist'].value_counts()
        f.write(f"Total artists: {len(artist_counts)}\n")
        f.write(f"Top 10 artists by number of works:\n")
        for artist, count in artist_counts.head(10).items():
            f.write(f"  {artist}: {count} works\n")
        
        # Year statistics
        if 'year' in df.columns:
            non_null_years = df['year'].dropna()
            if len(non_null_years) > 0:
                f.write("\nYear statistics:\n")
                f.write(f"Artworks with year information: {len(non_null_years)}\n")
                f.write(f"Earliest year: {non_null_years.min()}\n")
                f.write(f"Latest year: {non_null_years.max()}\n")
        
        # Style/genre statistics
        if 'style_genre' in df.columns:
            f.write("\nStyle/genre statistics:\n")
            all_styles = []
            for styles in df['style_genre'].dropna():
                if isinstance(styles, list):
                    all_styles.extend(styles)
            
            if all_styles:
                style_counts = pd.Series(all_styles).value_counts()
                f.write(f"Total unique styles/genres: {len(style_counts)}\n")
                f.write(f"Top 10 styles/genres:\n")
                for style, count in style_counts.head(10).items():
                    f.write(f"  {style}: {count} works\n")