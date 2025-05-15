"""
Index management module for the ArtRecognition system.

This module handles the creation and searching of FAISS indexes for artwork embeddings.
It uses the HNSW index which is optimized for the WikiArt dataset (~215K images).
"""

import numpy as np
import faiss
import os
import time
from typing import List, Tuple, Dict, Any

# Type aliases
FeatureVectorType = np.ndarray
MetadataType = Dict[str, Any]
SearchResultType = Dict[str, Any]


def build_index(features: FeatureVectorType) -> faiss.Index:
    """
    Build a HNSW index for the artwork embeddings.
    
    This function creates a Hierarchical Navigable Small World (HNSW) index,
    which offers an excellent balance of search speed and accuracy for the
    WikiArt dataset size (~215K images).
    
    Args:
        features: Feature vectors of shape (n_samples, dimension)
        
    Returns:
        FAISS index with vectors added
    """
    # Get dimensions
    num_vectors, dimension = features.shape
    print(f"Building index for {num_vectors} vectors of dimension {dimension}")
    
    # For WikiArt dataset, HNSW is an excellent choice
    # These settings are optimized for ~215K images
    m = 32  # Number of connections per node
    ef_construction = 200  # Dynamic candidate list size during construction
    
    # Create HNSW index
    index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = 64  # Dynamic candidate list size during search
    
    # Add vectors to the index
    start_time = time.time()
    index.add(features)
    build_time = time.time() - start_time
    
    print(f"Index built in {build_time:.2f} seconds with {index.ntotal} vectors")
    return index


def search(index: faiss.Index, query: FeatureVectorType, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for similar vectors in the index.
    
    Args:
        index: FAISS index
        query: Query vector of shape (dimension) or (1, dimension)
        k: Number of results to return
        
    Returns:
        Tuple of (distances, indices) arrays
    """
    # Ensure query has the right shape (1, dimension)
    if len(query.shape) == 1:
        query = query.reshape(1, -1)
    
    # Perform search
    distances, indices = index.search(query, k)
    return distances, indices


def search_artwork(query_vector: FeatureVectorType, 
                  index: faiss.Index, 
                  metadata: List[MetadataType], 
                  k: int = 5) -> List[SearchResultType]:
    """
    Search for similar artworks and format the results.
    
    Args:
        query_vector: Query feature vector
        index: FAISS index
        metadata: List of metadata for all artworks
        k: Number of results to return
        
    Returns:
        List of search results with metadata and similarity scores
    """
    # Search the index
    distances, indices = search(index, query_vector, k)
    
    # Format results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(metadata):
            # Invalid index, skip
            continue
            
        # Convert distance to similarity score (0-100%)
        similarity = 100.0 * (1.0 - distances[0][i] / (distances[0][i] + 10.0))
        
        results.append({
            'distance': float(distances[0][i]),
            'similarity': float(similarity),
            'metadata': metadata[idx]
        })
    
    return results


def save_index(index: faiss.Index, filepath: str) -> None:
    """
    Save the index to a file.
    
    Args:
        index: FAISS index
        filepath: Path to save the index
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the index
    faiss.write_index(index, filepath)
    print(f"Index saved to {filepath}")


def load_index(filepath: str) -> faiss.Index:
    """
    Load an index from a file.
    
    Args:
        filepath: Path to the index file
        
    Returns:
        Loaded FAISS index
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Index file not found: {filepath}")
    
    # Load the index
    index = faiss.read_index(filepath)
    print(f"Loaded index with {index.ntotal} vectors of dimension {index.d}")
    return index


def get_index_info(index: faiss.Index) -> Dict[str, Any]:
    """
    Get information about a FAISS index.
    
    Args:
        index: FAISS index
        
    Returns:
        Dictionary with index information
    """
    info = {
        'num_vectors': index.ntotal,
        'dimension': index.d,
        'index_type': 'HNSW',
        'memory_usage_mb': estimate_memory_usage(index) / (1024 * 1024)
    }
    
    # Get HNSW-specific parameters if available
    if hasattr(index, 'hnsw'):
        info['connections_per_node'] = index.hnsw.M
        info['ef_search'] = index.hnsw.efSearch
        info['ef_construction'] = index.hnsw.efConstruction
    
    return info


def estimate_memory_usage(index: faiss.Index) -> int:
    """
    Estimate memory usage of the index in bytes.
    
    Args:
        index: FAISS index
        
    Returns:
        Estimated memory usage in bytes
    """
    # Basic estimate: vectors + index overhead
    vector_size = index.ntotal * index.d * 4  # 4 bytes per float32
    
    # HNSW overhead (approximate)
    if hasattr(index, 'hnsw'):
        # Each node has connections to other nodes
        connections_overhead = index.ntotal * index.hnsw.M * 8  # 8 bytes per connection (4 for index, 4 for distance)
        return vector_size + connections_overhead
    else:
        # Simple flat index
        return vector_size