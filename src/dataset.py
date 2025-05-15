"""
Dataset module for the ArtRecognition system.

This module provides dataset classes for loading and processing artwork images
from the WikiArt dataset, handling image loading, transformations, and metadata.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import time
from typing import List, Dict, Any, Optional, Tuple

class ArtworkDataset(Dataset):
    """Dataset for loading artwork images from URLs."""
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the dataset with fixed transformation settings.
        
        Args:
            dataframe: DataFrame containing artwork data with at least a 'file_link' column
        """
        self.df = dataframe
        
        # Fixed transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and process a single artwork image.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing the image tensor and metadata
        """
        try:
            # Get image URL and metadata
            row = self.df.iloc[idx]
            image_url = row['file_link']
            
            # Load image
            img = self._load_image_from_url(image_url)
            
            # Apply transformations
            img_tensor = self.transform(img)
            
            # Extract metadata
            metadata = {
                'url': image_url,
                'artist': row.get('artist', 'Unknown'),
                'title': row.get('title', 'Unknown')
            }
            
            # Add year if available
            if 'year' in row and not pd.isna(row['year']):
                metadata['year'] = str(row['year'])
            
            return {
                'image': img_tensor,
                'metadata': metadata
            }
            
        except Exception as e:
            # Return a placeholder for error cases
            return self._create_placeholder(idx, str(e))
    
    def _load_image_from_url(self, url: str) -> Image.Image:
        """
        Load an image from a URL with retries.
        
        Args:
            url: URL to the image
            
        Returns:
            PIL Image
        """
        # Fixed settings
        timeout = 10
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert('RGB')
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        # If all attempts fail, return a blank image
        return Image.new('RGB', (224, 224), color=(0, 0, 0))
    
    def _create_placeholder(self, idx: int, error_msg: str) -> Dict[str, Any]:
        """
        Create a placeholder for error cases.
        
        Args:
            idx: Index of the item
            error_msg: Error message
            
        Returns:
            Dictionary with placeholder image and metadata
        """
        # Create a blank image
        img = Image.new('RGB', (224, 224), color=(0, 0, 0))
        img_tensor = self.transform(img)
        
        # Basic metadata
        metadata = {
            'url': self.df.iloc[idx].get('file_link', '') if idx < len(self.df) else '',
            'artist': 'Error',
            'title': 'Error Loading Image',
            'error': True,
            'error_message': error_msg
        }
        
        return {
            'image': img_tensor,
            'metadata': metadata
        }


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to handle errors in batches.
    
    Args:
        batch: List of items from the dataset
        
    Returns:
        Dictionary with batched tensors and metadata
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    # If the batch is empty, return a minimal valid batch
    if not batch:
        dummy_img = torch.zeros((1, 3, 224, 224))
        return {
            'image': dummy_img,
            'metadata': [{'error': True, 'artist': 'Error', 'title': 'Error', 'url': ''}]
        }
    
    # Collate images into a batch
    images = torch.stack([item['image'] for item in batch])
    
    # Collect metadata
    metadata = [item['metadata'] for item in batch]
    
    return {
        'image': images,
        'metadata': metadata
    }


def create_dataloader(dataframe: pd.DataFrame, 
                     batch_size: Optional[int] = None,
                     num_workers: Optional[int] = None) -> Tuple[DataLoader, bool]:
    """
    Create a DataLoader for the artwork dataset.
    
    Args:
        dataframe: DataFrame containing artwork data
        batch_size: Optional batch size (auto-determined if None)
        num_workers: Optional number of workers (auto-determined if None)
        
    Returns:
        Tuple of (DataLoader, is_gpu_available)
    """
    # Detect GPU
    is_gpu_available = torch.cuda.is_available()
    
    # Set batch size and workers based on device
    if batch_size is None:
        batch_size = 64 if is_gpu_available else 32
    
    if num_workers is None:
        num_workers = 4 if is_gpu_available else 2
    
    # Create dataset
    dataset = ArtworkDataset(dataframe)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=is_gpu_available,
        collate_fn=custom_collate_fn
    )
    
    return dataloader, is_gpu_available