"""
System management module for the ArtRecognition system.

This module handles the saving, loading, and management of the artwork
recognition system state, including the FAISS index, embeddings, and metadata.
"""

import os
import pickle
import numpy as np
import faiss
import json
import time
from typing import List, Tuple, Dict, Any, Optional

# Type aliases
FeatureVectorType = np.ndarray
MetadataType = Dict[str, Any]


def save_system(save_dir: str, index: faiss.Index, embeddings: FeatureVectorType, 
               metadata: List[MetadataType]) -> bool:
    """
    Save the complete system state to disk.
    
    Args:
        save_dir: Directory to save the system files
        index: FAISS index
        embeddings: Feature embeddings
        metadata: Metadata for all artworks
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Define paths
        index_path = os.path.join(save_dir, "artwork_index.faiss")
        embeddings_path = os.path.join(save_dir, "artwork_embeddings.npy")
        metadata_path = os.path.join(save_dir, "artwork_metadata.pkl")
        info_path = os.path.join(save_dir, "system_info.json")
        
        # Save index
        faiss.write_index(index, index_path)
        print(f"✅ Index saved: {index_path}")
        
        # Save embeddings
        np.save(embeddings_path, embeddings)
        print(f"✅ Embeddings saved: {embeddings_path}")
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f, protocol=4)
        print(f"✅ Metadata saved: {metadata_path}")
        
        # Save system info
        system_info = {
            "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_artworks": len(metadata),
            "embedding_dim": embeddings.shape[1],
            "version": "1.0"
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(system_info, f, indent=2)
        print(f"✅ System info saved: {info_path}")
        
        print(f"\nSystem successfully saved to: {save_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving system: {e}")
        return False


def load_system(save_dir: str) -> Tuple[faiss.Index, FeatureVectorType, List[MetadataType]]:
    """
    Load the complete system state from disk.
    
    Args:
        save_dir: Directory with the system files
        
    Returns:
        Tuple of (index, embeddings, metadata)
        
    Raises:
        FileNotFoundError: If any required file is missing
        RuntimeError: If there's an error loading the system
    """
    # Define paths
    index_path = os.path.join(save_dir, "artwork_index.faiss")
    embeddings_path = os.path.join(save_dir, "artwork_embeddings.npy")
    metadata_path = os.path.join(save_dir, "artwork_metadata.pkl")
    
    # Check if files exist
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    try:
        # Load index
        index = faiss.read_index(index_path)
        print(f"✅ Loaded index with {index.ntotal} vectors")
        
        # Load embeddings
        embeddings = np.load(embeddings_path)
        print(f"✅ Loaded embeddings with shape {embeddings.shape}")
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"✅ Loaded metadata for {len(metadata)} artworks")
        
        # Consistency check
        if index.ntotal != len(metadata) or index.ntotal != embeddings.shape[0]:
            print(f"⚠️ Warning: Inconsistent system state:")
            print(f"  - Index has {index.ntotal} vectors")
            print(f"  - Metadata has {len(metadata)} items")
            print(f"  - Embeddings has {embeddings.shape[0]} vectors")
        
        return index, embeddings, metadata
        
    except Exception as e:
        raise RuntimeError(f"Error loading system: {e}")


def delete_system(save_dir: str) -> bool:
    """
    Delete all system files.
    
    Args:
        save_dir: Directory with the system files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Define paths
        index_path = os.path.join(save_dir, "artwork_index.faiss")
        embeddings_path = os.path.join(save_dir, "artwork_embeddings.npy")
        metadata_path = os.path.join(save_dir, "artwork_metadata.pkl")
        info_path = os.path.join(save_dir, "system_info.json")
        
        # Delete files if they exist
        for path in [index_path, embeddings_path, metadata_path, info_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Deleted: {path}")
        
        print("System deleted.")
        return True
        
    except Exception as e:
        print(f"Error deleting system: {e}")
        return False


def system_exists(save_dir: str) -> bool:
    """
    Check if a system exists at the specified location.
    
    Args:
        save_dir: Directory to check
        
    Returns:
        True if a complete system exists, False otherwise
    """
    # Define paths
    index_path = os.path.join(save_dir, "artwork_index.faiss")
    embeddings_path = os.path.join(save_dir, "artwork_embeddings.npy")
    metadata_path = os.path.join(save_dir, "artwork_metadata.pkl")
    
    # Check if all required files exist
    return (os.path.exists(index_path) and 
            os.path.exists(embeddings_path) and 
            os.path.exists(metadata_path))


def get_system_info(save_dir: str) -> Optional[Dict[str, Any]]:
    """
    Get information about the system.
    
    Args:
        save_dir: Directory with the system files
        
    Returns:
        Dictionary with system info or None if no system found
    """
    # Check if system exists
    if not system_exists(save_dir):
        return None
    
    # Define paths
    info_path = os.path.join(save_dir, "system_info.json")
    
    # Try to load info file
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    
    # If info file doesn't exist or couldn't be loaded,
    # create basic info from existing files
    try:
        index_path = os.path.join(save_dir, "artwork_index.faiss")
        metadata_path = os.path.join(save_dir, "artwork_metadata.pkl")
        
        # Load index
        index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return {
            "num_artworks": len(metadata),
            "index_size": index.ntotal,
            "embedding_dim": index.d
        }
        
    except Exception:
        # If everything fails, return minimal info
        return {
            "status": "exists",
            "info_available": False
        }