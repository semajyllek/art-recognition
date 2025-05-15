"""
System management module for the ArtRecognition system.

This module handles saving, loading, and managing the complete artwork 
recognition system, including the index, embeddings, and metadata.
"""

import os
import pickle
import numpy as np
import faiss
import json
import time
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path

# Import from other modules
# These would normally be import statements pointing to other files
# For demonstration, we're defining the expected interfaces
FeatureVectorType = np.ndarray
MetadataType = Dict[str, Any]
FeatureExtractorType = Any  # This would typically be the FeatureExtractor class


class SystemManager:
    """
    Manages the ArtRecognition system state.
    
    This class handles saving, loading, and maintaining the system state,
    including the index, embeddings, and metadata.
    """
    
    def __init__(self, save_dir: str):
        """
        Initialize the system manager.
        
        Args:
            save_dir: Directory to save system files
        """
        self.save_dir = save_dir
        
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Define paths for system files
        self.index_path = os.path.join(save_dir, "artwork_index.faiss")
        self.embeddings_path = os.path.join(save_dir, "artwork_embeddings.npy")
        self.metadata_path = os.path.join(save_dir, "artwork_metadata.pkl")
        self.metadata_json_path = os.path.join(save_dir, "artwork_metadata.json")
        self.info_path = os.path.join(save_dir, "system_info.json")
        
        # Initialize system state
        self.index = None
        self.embeddings = None
        self.metadata = None
        self.system_info = {
            "creation_time": None,
            "last_update_time": None,
            "num_artworks": 0,
            "embedding_dim": 0,
            "version": "1.0"
        }
    
    def save_system(self, index: faiss.Index, embeddings: FeatureVectorType, 
                   metadata: List[MetadataType]) -> bool:
        """
        Save the complete system state.
        
        Args:
            index: FAISS index
            embeddings: Feature vectors
            metadata: Metadata for all artworks
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update system info
            self.system_info["last_update_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if self.system_info["creation_time"] is None:
                self.system_info["creation_time"] = self.system_info["last_update_time"]
            self.system_info["num_artworks"] = len(metadata)
            self.system_info["embedding_dim"] = embeddings.shape[1]
            
            # Save system info
            with open(self.info_path, 'w', encoding='utf-8') as f:
                json.dump(self.system_info, f, indent=2)
            
            # Save index
            faiss.write_index(index, self.index_path)
            print(f"✅ Index saved: {self.index_path}")
            
            # Save embeddings
            np.save(self.embeddings_path, embeddings)
            print(f"✅ Embeddings saved: {self.embeddings_path}")
            
            # Save metadata using pickle (for complete object preservation)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f, protocol=4)
            print(f"✅ Metadata saved: {self.metadata_path}")
            
            # Also save a JSON version of metadata for better compatibility
            try:
                # Convert metadata to JSON-compatible format
                json_compatible = []
                for item in metadata:
                    json_item = {}
                    for key, value in item.items():
                        # Handle non-serializable types
                        if isinstance(value, (str, int, float, bool, type(None))):
                            json_item[key] = value
                        elif isinstance(value, list):
                            # Handle lists of simple types
                            if all(isinstance(x, (str, int, float, bool)) for x in value):
                                json_item[key] = value
                            else:
                                json_item[key] = str(value)
                        else:
                            json_item[key] = str(value)
                    json_compatible.append(json_item)
                
                with open(self.metadata_json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_compatible, f, ensure_ascii=False, indent=2)
                print(f"✅ Backup metadata saved as JSON: {self.metadata_json_path}")
            except Exception as e:
                print(f"⚠️ Could not save JSON backup: {e}")
            
            # Update internal state
            self.index = index
            self.embeddings = embeddings
            self.metadata = metadata
            
            print(f"\nSystem successfully saved to: {self.save_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving system: {e}")
            return False
    
    def load_system(self) -> Tuple[Optional[faiss.Index], Optional[FeatureVectorType], 
                                 Optional[List[MetadataType]]]:
        """
        Load the complete system state.
        
        Returns:
            Tuple of (index, embeddings, metadata) or (None, None, None) if load fails
        """
        # Check if required files exist
        if not os.path.exists(self.index_path):
            print(f"❌ Index file not found: {self.index_path}")
            return None, None, None
        
        if not os.path.exists(self.embeddings_path):
            print(f"❌ Embeddings file not found: {self.embeddings_path}")
            return None, None, None
        
        if not os.path.exists(self.metadata_path) and not os.path.exists(self.metadata_json_path):
            print(f"❌ No metadata file found")
            return None, None, None
        
        try:
            # Load index
            index = faiss.read_index(self.index_path)
            print(f"✅ Loaded index with {index.ntotal} vectors")
            
            # Load embeddings
            embeddings = np.load(self.embeddings_path)
            print(f"✅ Loaded embeddings with shape {embeddings.shape}")
            
            # Load metadata - try pickle first, then JSON
            metadata = None
            try:
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"✅ Loaded metadata for {len(metadata)} artworks")
            except Exception as e:
                print(f"⚠️ Error loading metadata from pickle: {e}")
                
                # Try loading from JSON backup
                if os.path.exists(self.metadata_json_path):
                    try:
                        with open(self.metadata_json_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        print(f"✅ Loaded metadata from JSON backup ({len(metadata)} artworks)")
                    except Exception as json_e:
                        print(f"❌ Failed to load metadata from both formats: {e}, {json_e}")
                        return None, None, None
                else:
                    return None, None, None
            
            # Load system info if available
            if os.path.exists(self.info_path):
                try:
                    with open(self.info_path, 'r', encoding='utf-8') as f:
                        self.system_info = json.load(f)
                    print(f"✅ Loaded system info: {self.system_info}")
                except Exception as e:
                    print(f"⚠️ Could not load system info: {e}")
            
            # Update internal state
            self.index = index
            self.embeddings = embeddings
            self.metadata = metadata
            
            # Consistency check
            if index.ntotal != len(metadata) or index.ntotal != embeddings.shape[0]:
                print(f"⚠️ Warning: Consistency check failed")
                print(f"  - Index has {index.ntotal} vectors")
                print(f"  - Metadata has {len(metadata)} items")
                print(f"  - Embeddings has {embeddings.shape[0]} vectors")
            
            return index, embeddings, metadata
            
        except Exception as e:
            print(f"❌ Error loading system: {e}")
            return None, None, None
    
    def delete_system(self) -> bool:
        """
        Delete all system files.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete each file if it exists
            for path in [self.index_path, self.embeddings_path, self.metadata_path, 
                         self.metadata_json_path, self.info_path]:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Deleted: {path}")
            
            # Reset internal state
            self.index = None
            self.embeddings = None
            self.metadata = None
            self.system_info = {
                "creation_time": None,
                "last_update_time": None,
                "num_artworks": 0,
                "embedding_dim": 0,
                "version": "1.0"
            }
            
            print("System deleted. You can now train a new one.")
            return True
            
        except Exception as e:
            print(f"Error deleting system: {e}")
            return False
    
    def system_exists(self) -> bool:
        """
        Check if a saved system exists.
        
        Returns:
            True if a system exists, False otherwise
        """
        # We need at least the index and embeddings to consider the system usable
        return os.path.exists(self.index_path) and os.path.exists(self.embeddings_path)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the current system.
        
        Returns:
            Dictionary of system information
        """
        if self.system_exists():
            # If we haven't loaded the system yet, try to load the info file
            if self.system_info["num_artworks"] == 0 and os.path.exists(self.info_path):
                try:
                    with open(self.info_path, 'r', encoding='utf-8') as f:
                        self.system_info = json.load(f)
                except Exception:
                    pass
            
            return self.system_info
        else:
            return {
                "status": "not_found",
                "message": "No system found at the specified location"
            }
    
    def rebuild_system(self, feature_extractor: FeatureExtractorType, 
                      dataframe, batch_size: int = 32) -> Tuple[Optional[faiss.Index], 
                                                             Optional[List[MetadataType]]]:
        """
        Rebuild the system from scratch.
        
        Args:
            feature_extractor: Feature extractor model
            dataframe: DataFrame with artwork data
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (index, metadata) or (None, None) if rebuild fails
        """
        # First delete existing system
        if self.system_exists():
            print("Deleting existing system...")
            self.delete_system()
        
        # Extract features
        try:
            print(f"Extracting features from {len(dataframe)} artworks...")
            
            # This would normally be imported from feature_extraction.py
            # For demonstration, assume extract_features is available
            from feature_extraction import extract_features
            embeddings, metadata = extract_features(dataframe, feature_extractor, batch_size)
            
            # Build index
            print("Building search index...")
            
            # This would normally be imported from index_management.py
            # For demonstration, assume build_index is available
            from index_management import build_index
            index = build_index(embeddings)
            
            # Save the system
            print("Saving system...")
            self.save_system(index, embeddings, metadata)
            
            return index, metadata
            
        except Exception as e:
            print(f"Error rebuilding system: {e}")
            return None, None


# Standalone functions for simpler usage

def save_system(save_dir: str, index: faiss.Index, embeddings: FeatureVectorType, 
               metadata: List[MetadataType]) -> bool:
    """
    Save the complete system state.
    
    Args:
        save_dir: Directory to save system files
        index: FAISS index
        embeddings: Feature vectors
        metadata: Metadata for all artworks
        
    Returns:
        True if successful, False otherwise
    """
    manager = SystemManager(save_dir)
    return manager.save_system(index, embeddings, metadata)

def load_system(save_dir: str) -> Tuple[Optional[faiss.Index], Optional[FeatureVectorType], 
                                       Optional[List[MetadataType]]]:
    """
    Load the complete system state.
    
    Args:
        save_dir: Directory with system files
        
    Returns:
        Tuple of (index, embeddings, metadata) or (None, None, None) if load fails
    """
    manager = SystemManager(save_dir)
    return manager.load_system()

def delete_system(save_dir: str) -> bool:
    """
    Delete all system files.
    
    Args:
        save_dir: Directory with system files
        
    Returns:
        True if successful, False otherwise
    """
    manager = SystemManager(save_dir)
    return manager.delete_system()

def system_exists(save_dir: str) -> bool:
    """
    Check if a saved system exists.
    
    Args:
        save_dir: Directory to check
        
    Returns:
        True if a system exists, False otherwise
    """
    manager = SystemManager(save_dir)
    return manager.system_exists()

def get_system_info(save_dir: str) -> Dict[str, Any]:
    """
    Get information about a saved system.
    
    Args:
        save_dir: Directory with system files
        
    Returns:
        Dictionary of system information
    """
    manager = SystemManager(save_dir)
    return manager.get_system_info()