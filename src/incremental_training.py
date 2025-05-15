"""
Incremental training module for the ArtRecognition system.

This module provides functionality for incrementally updating the
system with new artworks, allowing the database to grow over time.
"""

import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Dict, Any, Optional, Set
from tqdm import tqdm

# Type aliases
FeatureVectorType = np.ndarray
MetadataType = Dict[str, Any]


def get_processed_urls(metadata: List[MetadataType]) -> Set[str]:
    """
    Get the set of URLs that have already been processed.
    
    Args:
        metadata: List of metadata for all processed artworks
        
    Returns:
        Set of processed URLs
    """
    urls = set()
    for item in metadata:
        url = item.get('url', '')
        if url:
            urls.add(url)
    return urls


def filter_new_artworks(dataframe: pd.DataFrame, processed_urls: Set[str]) -> pd.DataFrame:
    """
    Filter a dataframe to include only new artworks not yet processed.
    
    Args:
        dataframe: DataFrame with artwork data
        processed_urls: Set of URLs already processed
        
    Returns:
        Filtered DataFrame with only new artworks
    """
    # Check if 'file_link' column exists
    if 'file_link' not in dataframe.columns:
        raise ValueError("DataFrame must have a 'file_link' column")
    
    # Filter out already processed URLs
    new_df = dataframe[~dataframe['file_link'].isin(processed_urls)].reset_index(drop=True)
    
    print(f"Found {len(new_df)} new artworks out of {len(dataframe)} total")
    return new_df


def train_incrementally(dataframe: pd.DataFrame, 
                       save_dir: str,
                       chunk_size: int = 1000) -> bool:
    """
    Incrementally update the artwork recognition system with new artworks.
    
    This function either:
    1. Loads an existing system and adds new artworks, or
    2. Creates a new system if none exists
    
    Args:
        dataframe: DataFrame with artwork data
        save_dir: Directory to save/load system files
        chunk_size: Number of new artworks to add at once
        
    Returns:
        True if successful, False otherwise
    """
    # Import required functions
    # These would normally be imported at the top, but are included
    # here to show the dependencies clearly
    from feature_extraction import create_feature_extractor, extract_features
    from index_management import build_index
    from system_management import save_system, load_system, system_exists
    
    print(f"Starting incremental training with up to {chunk_size} new artworks...")
    
    # Check if an existing system exists
    if system_exists(save_dir):
        try:
            # Load existing system
            print("Loading existing system...")
            index, embeddings, metadata = load_system(save_dir)
            
            # Get already processed URLs
            processed_urls = get_processed_urls(metadata)
            print(f"Existing system has {len(processed_urls)} artworks")
            
            # Filter dataframe to include only new artworks
            new_df = filter_new_artworks(dataframe, processed_urls)
            
            # If no new artworks, just return success
            if len(new_df) == 0:
                print("No new artworks to add. System is up to date.")
                return True
            
            # Take a chunk of new artworks
            if len(new_df) > chunk_size:
                print(f"Taking first {chunk_size} new artworks...")
                chunk_df = new_df.head(chunk_size).reset_index(drop=True)
            else:
                chunk_df = new_df
            
            # Create feature extractor
            feature_extractor = create_feature_extractor()
            
            # Extract features for new artworks
            print(f"Extracting features for {len(chunk_df)} new artworks...")
            new_embeddings, new_metadata = extract_features(chunk_df, feature_extractor)
            
            # Combine with existing embeddings and metadata
            print("Combining with existing data...")
            combined_embeddings = np.vstack([embeddings, new_embeddings])
            combined_metadata = metadata + new_metadata
            
            # Create a new index with combined data
            print("Building new index with combined data...")
            combined_index = build_index(combined_embeddings)
            
            # Save the updated system
            print("Saving updated system...")
            success = save_system(save_dir, combined_index, combined_embeddings, combined_metadata)
            
            if success:
                print(f"System successfully updated with {len(new_metadata)} new artworks")
                print(f"Total artworks in system: {len(combined_metadata)}")
                
                # Report progress through the dataset
                total = len(dataframe)
                processed = len(processed_urls) + len(new_metadata)
                percent = (processed / total) * 100 if total > 0 else 0
                print(f"Progress: {processed}/{total} artworks processed ({percent:.1f}%)")
                
                return True
            else:
                print("Failed to save updated system")
                return False
            
        except Exception as e:
            print(f"Error updating existing system: {e}")
            print("Starting fresh...")
    
    # If no existing system or failed to update, create a new one
    try:
        print("Creating new system...")
        
        # Take a chunk of artworks
        if len(dataframe) > chunk_size:
            print(f"Taking first {chunk_size} artworks...")
            chunk_df = dataframe.head(chunk_size).reset_index(drop=True)
        else:
            chunk_df = dataframe
        
        # Create feature extractor
        feature_extractor = create_feature_extractor()
        
        # Extract features
        print(f"Extracting features for {len(chunk_df)} artworks...")
        embeddings, metadata = extract_features(chunk_df, feature_extractor)
        
        # Build index
        print("Building search index...")
        index = build_index(embeddings)
        
        # Save the system
        print("Saving system...")
        success = save_system(save_dir, index, embeddings, metadata)
        
        if success:
            print(f"New system created with {len(metadata)} artworks")
            
            # Report progress through the dataset
            total = len(dataframe)
            processed = len(metadata)
            percent = (processed / total) * 100 if total > 0 else 0
            print(f"Progress: {processed}/{total} artworks processed ({percent:.1f}%)")
            
            return True
        else:
            print("Failed to save new system")
            return False
        
    except Exception as e:
        print(f"Error creating new system: {e}")
        return False


def run_incremental_demo(dataframe: pd.DataFrame, save_dir: str) -> None:
    """
    Run the incremental training demo with user interaction.
    
    This function guides the user through the process of incrementally
    adding artworks to the system.
    
    Args:
        dataframe: DataFrame with artwork data
        save_dir: Directory to save/load system files
    """
    from system_management import system_exists
    
    print("Starting Artwork Recognition System with Incremental Training...")
    
    # Check if a system exists
    if system_exists(save_dir):
        print("Found existing system.")
        add_more = input("Would you like to add more artworks? (y/n): ")
        
        if add_more.lower() == 'y':
            # Add more artworks with default chunk size (1000)
            success = train_incrementally(dataframe, save_dir)
            
            if success:
                print("System successfully updated")
                
                # Ask if user wants to add more
                another_round = input("Would you like to add another batch of artworks? (y/n): ")
                if another_round.lower() == 'y':
                    run_incremental_demo(dataframe, save_dir)
            else:
                print("Failed to update system")
        else:
            print("Using existing system without changes")
    else:
        print("No existing system found.")
        create_new = input("Would you like to create a new system? (y/n): ")
        
        if create_new.lower() == 'y':
            # Create new system
            success = train_incrementally(dataframe, save_dir)
            
            if success:
                print("New system successfully created")
                
                # Ask if user wants to add more
                another_round = input("Would you like to add another batch of artworks? (y/n): ")
                if another_round.lower() == 'y':
                    run_incremental_demo(dataframe, save_dir)
            else:
                print("Failed to create system")
        else:
            print("Exiting without creating a system")


def add_more_artworks(dataframe: pd.DataFrame, save_dir: str) -> bool:
    """
    Add more artworks to the system without user interaction.
    
    This is a simpler interface for adding artworks without
    the interactive prompts.
    
    Args:
        dataframe: DataFrame with artwork data
        save_dir: Directory to save/load system files
        
    Returns:
        True if successful, False otherwise
    """
    from system_management import system_exists
    
    if not system_exists(save_dir):
        print("No existing system found. Creating new system...")
    else:
        print("Updating existing system...")
    
    return train_incrementally(dataframe, save_dir)