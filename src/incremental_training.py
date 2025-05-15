"""
Incremental training module for the ArtRecognition system.

This module provides functionality for incrementally updating
the recognition system with new artworks, avoiding duplicates
and maintaining system performance over time.
"""

import numpy as np
import pandas as pd
import faiss
import os
import time
from typing import List, Tuple, Dict, Any, Optional, Union, Set
from tqdm import tqdm

# Type aliases
FeatureVectorType = np.ndarray
MetadataType = Dict[str, Any]
FeatureExtractorType = Any  # This would be the actual feature extractor type

# These would normally be imported from other modules
# For demonstration, we'll reference them by name
# from feature_extraction import extract_features
# from index_management import build_index, save_index
# from system_management import save_system, load_system, system_exists


def get_processed_urls(metadata: List[MetadataType]) -> Set[str]:
    """
    Get the set of URLs that have already been processed.
    
    Args:
        metadata: List of metadata for all artworks
        
    Returns:
        Set of URLs
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
        Filtered DataFrame
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
                       feature_extractor: FeatureExtractorType,
                       chunk_size: int = 1000,
                       batch_size: int = 32) -> Tuple[Optional[faiss.Index], Optional[List[MetadataType]]]:
    """
    Incrementally update the artwork recognition system with new artworks.
    
    Args:
        dataframe: DataFrame with artwork data
        save_dir: Directory to save/load system files
        feature_extractor: Feature extractor for processing new images
        chunk_size: Number of new artworks to add
        batch_size: Batch size for feature extraction
        
    Returns:
        Tuple of (index, metadata) or (None, None) if failed
    """
    from feature_extraction import extract_features
    from index_management import build_index
    from system_management import save_system, load_system, system_exists
    
    print(f"Starting incremental training with up to {chunk_size} new artworks...")
    
    # Check if existing system exists
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
            
            # If no new artworks, just return the existing system
            if len(new_df) == 0:
                print("No new artworks to add. System is up to date.")
                return index, metadata
            
            # Take a chunk of new artworks
            if len(new_df) > chunk_size:
                print(f"Taking first {chunk_size} new artworks...")
                chunk_df = new_df.head(chunk_size).reset_index(drop=True)
            else:
                chunk_df = new_df
            
            # Extract features for new artworks
            print(f"Extracting features for {len(chunk_df)} new artworks...")
            new_embeddings, new_metadata = extract_features(chunk_df, feature_extractor, batch_size)
            
            # Combine with existing embeddings and metadata
            print("Combining with existing data...")
            combined_embeddings = np.vstack([embeddings, new_embeddings])
            combined_metadata = metadata + new_metadata
            
            # Create a new index with combined data
            # (Some index types like HNSW can't be updated, so we rebuild)
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
                
                return combined_index, combined_metadata
            else:
                print("Failed to save updated system")
                return index, metadata
            
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
        
        # Extract features
        print(f"Extracting features for {len(chunk_df)} artworks...")
        embeddings, metadata = extract_features(chunk_df, feature_extractor, batch_size)
        
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
            
            return index, metadata
        else:
            print("Failed to save new system")
            return None, None
        
    except Exception as e:
        print(f"Error creating new system: {e}")
        return None, None


def analyze_incremental_performance(embeddings: FeatureVectorType, 
                                   metadata: List[MetadataType],
                                   chunk_sizes: List[int] = [1000, 5000, 10000]) -> None:
    """
    Analyze how index performance scales with increasing dataset size.
    
    Args:
        embeddings: Full set of embeddings
        metadata: Full set of metadata
        chunk_sizes: Different dataset sizes to test
    """
    from index_management import build_index
    import matplotlib.pyplot as plt
    
    print("Analyzing incremental performance...")
    
    # Performance metrics
    build_times = []
    query_times = []
    memory_usage = []
    
    # Test query
    query_vector = embeddings[0].reshape(1, -1)  # Use first vector as test query
    
    # Test for different chunk sizes
    for size in chunk_sizes:
        if size > len(embeddings):
            print(f"Skipping size {size} (larger than dataset)")
            continue
        
        print(f"Testing with {size} artworks...")
        
        # Get chunk of embeddings
        chunk_embeddings = embeddings[:size]
        
        # Measure build time
        start_time = time.time()
        index = build_index(chunk_embeddings)
        build_time = time.time() - start_time
        build_times.append(build_time)
        
        # Measure query time (average of 100 queries)
        query_time_sum = 0
        for _ in range(100):
            start_time = time.time()
            index.search(query_vector, 5)
            query_time_sum += time.time() - start_time
        query_times.append(query_time_sum / 100)
        
        # Memory usage (approximate based on embeddings size)
        memory_mb = chunk_embeddings.nbytes / (1024 * 1024)
        memory_usage.append(memory_mb)
        
        print(f"  Build time: {build_time:.3f}s")
        print(f"  Query time: {(query_time_sum / 100) * 1000:.3f}ms")
        print(f"  Memory: {memory_mb:.2f}MB")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Build time
    plt.subplot(1, 3, 1)
    plt.plot(chunk_sizes[:len(build_times)], build_times, 'o-', label='Build Time')
    plt.xlabel('Number of Artworks')
    plt.ylabel('Time (seconds)')
    plt.title('Index Build Time')
    plt.grid(alpha=0.3)
    
    # Query time
    plt.subplot(1, 3, 2)
    plt.plot(chunk_sizes[:len(query_times)], [t * 1000 for t in query_times], 'o-', label='Query Time')
    plt.xlabel('Number of Artworks')
    plt.ylabel('Time (milliseconds)')
    plt.title('Query Time')
    plt.grid(alpha=0.3)
    
    # Memory usage
    plt.subplot(1, 3, 3)
    plt.plot(chunk_sizes[:len(memory_usage)], memory_usage, 'o-', label='Memory Usage')
    plt.xlabel('Number of Artworks')
    plt.ylabel('Memory (MB)')
    plt.title('Memory Usage')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def estimate_full_dataset_resources(sample_embeddings: FeatureVectorType, 
                                   total_artworks: int) -> Dict[str, float]:
    """
    Estimate resources needed for the full dataset based on a sample.
    
    Args:
        sample_embeddings: Sample of embeddings
        total_artworks: Total number of artworks in the full dataset
        
    Returns:
        Dictionary with resource estimates
    """
    from index_management import build_index
    
    print(f"Estimating resources for {total_artworks} artworks based on sample of {len(sample_embeddings)}...")
    
    # Build index for sample
    start_time = time.time()
    index = build_index(sample_embeddings)
    build_time = time.time() - start_time
    
    # Memory usage of sample
    sample_memory_mb = sample_embeddings.nbytes / (1024 * 1024)
    
    # Scaling factor
    scaling_factor = total_artworks / len(sample_embeddings)
    
    # Estimates (memory scales linearly, time might scale worse)
    estimated_memory_mb = sample_memory_mb * scaling_factor
    
    # Time often scales superlinearly, use a conservative power law
    time_scaling_power = 1.2  # Empirical factor (could be adjusted)
    estimated_build_time = build_time * (scaling_factor ** time_scaling_power)
    
    # Prepare results
    estimates = {
        'sample_size': len(sample_embeddings),
        'full_size': total_artworks,
        'sample_build_time': build_time,
        'estimated_build_time': estimated_build_time,
        'sample_memory_mb': sample_memory_mb,
        'estimated_memory_mb': estimated_memory_mb,
        'estimated_memory_gb': estimated_memory_mb / 1024
    }
    
    # Print summary
    print(f"Sample build time: {build_time:.2f}s")
    print(f"Estimated full dataset build time: {estimated_build_time:.2f}s ({estimated_build_time/60:.2f}min)")
    print(f"Sample memory: {sample_memory_mb:.2f}MB")
    print(f"Estimated full dataset memory: {estimated_memory_mb:.2f}MB ({estimated_memory_mb/1024:.2f}GB)")
    
    return estimates


def run_incremental_demo(dataframe: pd.DataFrame, 
                        save_dir: str,
                        feature_extractor: FeatureExtractorType,
                        chunk_size: int = 1000) -> None:
    """
    Run the incremental training demo with user interaction.
    
    Args:
        dataframe: DataFrame with artwork data
        save_dir: Directory to save/load system files
        feature_extractor: Feature extractor for processing new images
        chunk_size: Number of new artworks to add per iteration
    """
    from system_management import system_exists, get_system_info
    
    print("Starting Artwork Recognition System with Incremental Training...")
    
    # Check if a system exists
    if system_exists(save_dir):
        system_info = get_system_info(save_dir)
        current_size = system_info.get('num_artworks', 0)
        
        print(f"Found existing system with {current_size} artworks")
        add_more = input("Would you like to add more artworks? (y/n): ")
        
        if add_more.lower() == 'y':
            # Add more artworks
            index, metadata = train_incrementally(
                dataframe=dataframe,
                save_dir=save_dir,
                feature_extractor=feature_extractor,
                chunk_size=chunk_size
            )
            
            if index is not None:
                print("System successfully updated")
                
                # Ask if user wants to add more
                if len(metadata) < len(dataframe):
                    another_round = input("Would you like to add another batch of artworks? (y/n): ")
                    if another_round.lower() == 'y':
                        run_incremental_demo(dataframe, save_dir, feature_extractor, chunk_size)
            else:
                print("Failed to update system")
        else:
            # Just load the system
            print("Using existing system without changes")
    else:
        print("No existing system found.")
        create_new = input("Would you like to create a new system? (y/n): ")
        
        if create_new.lower() == 'y':
            # Create new system
            index, metadata = train_incrementally(
                dataframe=dataframe,
                save_dir=save_dir,
                feature_extractor=feature_extractor,
                chunk_size=chunk_size
            )
            
            if index is not None:
                print("New system successfully created")
                
                # Ask if user wants to add more
                if len(metadata) < len(dataframe):
                    another_round = input("Would you like to add another batch of artworks? (y/n): ")
                    if another_round.lower() == 'y':
                        run_incremental_demo(dataframe, save_dir, feature_extractor, chunk_size)
            else:
                print("Failed to create system")
        else:
            print("Exiting without creating a system")