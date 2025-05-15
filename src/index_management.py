"""
Index management module for the ArtRecognition system.

This module contains functions for creating and searching FAISS indexes
specifically optimized for the WikiArt dataset with ~215K images.
"""

import numpy as np
import faiss
import os
import time
from typing import List, Tuple, Dict, Any, Optional

# Type aliases
FeatureVectorType = np.ndarray
MetadataType = Dict[str, Any]
SearchResultType = Dict[str, Any]

def build_index(features: FeatureVectorType) -> faiss.Index:
    """
    Build a FAISS index optimized for our specific use case (~215K artwork images).
    
    For a dataset of this size, we use an HNSW index which offers an excellent
    balance of search speed and accuracy.
    
    Args:
        features: Feature vectors of shape (n_samples, dimension)
        
    Returns:
        FAISS index with vectors added
    """
    # Get dimensions
    num_vectors, dimension = features.shape
    print(f"Building index for {num_vectors} vectors of dimension {dimension}")
    
    # For ~215K images, HNSW is an excellent choice
    m = 32  # Number of connections per node (higher = more accurate but slower to build)
    ef_construction = 200  # Size of the dynamic candidate list during construction
    
    # Create HNSW index
    index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = 64  # Dynamic candidate list size during search
    
    # Add vectors to the index
    start_time = time.time()
    index.add(features)
    elapsed = time.time() - start_time
    
    print(f"Index built in {elapsed:.2f} seconds with {index.ntotal} vectors")
    return index

def search(index: faiss.Index, query: FeatureVectorType, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for similar vectors in the index.
    
    Args:
        index: FAISS index
        query: Query vector(s) of shape (n_queries, dimension)
        k: Number of results to return per query
        
    Returns:
        Tuple of (distances, indices) arrays
    """
    # Ensure query has the right shape
    if len(query.shape) == 1:
        query = query.reshape(1, -1)
    
    # Perform search
    distances, indices = index.search(query, k)
    return distances, indices

def format_search_results(distances: np.ndarray, indices: np.ndarray, 
                          metadata_list: List[MetadataType]) -> List[SearchResultType]:
    """
    Format search results with metadata.
    
    Args:
        distances: Array of distances from search
        indices: Array of indices from search
        metadata_list: List of metadata for all artworks
        
    Returns:
        List of search results with metadata and similarity scores
    """
    results = []
    
    # We only handle a single query at a time
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and idx < len(metadata_list):  # Check valid index
            # Calculate similarity score (0-100%)
            similarity = 100.0 * (1.0 - distances[0][i] / (distances[0][i] + 10.0))
            
            results.append({
                'distance': float(distances[0][i]),
                'similarity': float(similarity),
                'metadata': metadata_list[idx]
            })
    
    return results

def save_index(index: faiss.Index, filepath: str) -> None:
    """
    Save the index to a file.
    
    Args:
        index: FAISS index
        filepath: Path to save the index
    """
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the index
        faiss.write_index(index, filepath)
        print(f"Index saved to {filepath}")
    except Exception as e:
        print(f"Error saving index: {e}")

def load_index(filepath: str) -> faiss.Index:
    """
    Load an index from a file.
    
    Args:
        filepath: Path to the index file
        
    Returns:
        Loaded FAISS index
        
    Raises:
        FileNotFoundError: If index file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Index file not found: {filepath}")
    
    try:
        # Load the index
        index = faiss.read_index(filepath)
        print(f"Loaded index with {index.ntotal} vectors of dimension {index.d}")
        return index
    except Exception as e:
        raise RuntimeError(f"Error loading index: {e}")

def merge_with_existing_index(existing_index: faiss.Index, new_features: FeatureVectorType) -> faiss.Index:
    """
    Merge new features with an existing index.
    
    This function is useful for incremental training.
    
    Args:
        existing_index: Existing FAISS index
        new_features: New feature vectors to add
        
    Returns:
        New index with all vectors
    """
    # Get index properties
    dimension = existing_index.d
    
    # For HNSW index, we need to create a new one and add all vectors
    # Create a new index with the same parameters
    m = 32  # Same as in build_index
    index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_L2)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64
    
    # Add existing vectors (need to reconstruct them)
    existing_count = existing_index.ntotal
    print(f"Extracting {existing_count} existing vectors...")
    
    # Extract in batches to avoid memory issues
    batch_size = 10000
    for start_idx in range(0, existing_count, batch_size):
        end_idx = min(start_idx + batch_size, existing_count)
        batch_vectors = np.vstack([
            existing_index.reconstruct(i) for i in range(start_idx, end_idx)
        ])
        index.add(batch_vectors)
    
    # Add new vectors
    print(f"Adding {len(new_features)} new vectors...")
    index.add(new_features)
    
    print(f"Merged index created with {index.ntotal} vectors")
    return index