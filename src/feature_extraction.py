"""
Feature extraction module for the ArtRecognition system.

This module contains functionality for extracting visual features
from artwork images using deep learning models. It includes custom
dataset classes, model handlers, and optimization for GPU processing.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import time
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from tqdm.auto import tqdm

# Type aliases
Tensor = torch.Tensor
ImageType = Union[str, bytes, Image.Image]
MetadataType = Dict[str, Any]

class ArtworkDataset(Dataset):
    """
    Dataset for loading and processing artwork images.
    
    This class handles loading images from URLs/files, applying transformations,
    and returning them along with their metadata.
    """
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 transform: Optional[Callable] = None,
                 timeout: int = 10,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize the dataset.
        
        Args:
            dataframe: DataFrame containing artwork data with at least a 'file_link' column
            transform: PyTorch transforms to apply to images (default: standard normalization)
            timeout: Timeout for image downloads in seconds
            max_retries: Maximum number of download retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.df = dataframe
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Default transformation pipeline if none provided
        self.transform = transform or transforms.Compose([
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
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing the image tensor and metadata
        """
        # Get image URL or path
        try:
            row = self.df.iloc[idx]
            image_url = row['file_link']
            
            # Load the image
            img = self._load_image(image_url)
            
            # Apply transformations
            img_tensor = self.transform(img)
            
            # Get metadata
            metadata = self._extract_metadata(row)
            
            return {
                'image': img_tensor,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return a placeholder item instead of failing
            return self._create_placeholder(idx)
    
    def _load_image(self, image_source: str) -> Image.Image:
        """
        Load an image from a URL or file path.
        
        Args:
            image_source: URL or path to the image
            
        Returns:
            PIL Image object
        
        Raises:
            RuntimeError: If image cannot be loaded after max retries
        """
        # Handle URL
        if image_source.startswith(('http://', 'https://')):
            return self._load_image_from_url(image_source)
        # Handle local file
        else:
            return Image.open(image_source).convert('RGB')
    
    def _load_image_from_url(self, url: str) -> Image.Image:
        """
        Load an image from a URL with retries.
        
        Args:
            url: URL to the image
            
        Returns:
            PIL Image object
            
        Raises:
            RuntimeError: If image cannot be loaded after max retries
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()  # Raise exception for HTTP errors
                return Image.open(BytesIO(response.content)).convert('RGB')
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:  # Don't sleep on the last attempt
                    time.sleep(self.retry_delay)
        
        # If we get here, all attempts failed
        print(f"Failed to load image from {url}: {last_exception}")
        return self._create_blank_image()
    
    def _create_blank_image(self) -> Image.Image:
        """Create a blank image for error cases."""
        return Image.new('RGB', (224, 224), color=(0, 0, 0))
    
    def _extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """
        Extract metadata from a DataFrame row.
        
        Args:
            row: Row from the DataFrame
            
        Returns:
            Dictionary with metadata fields
        """
        metadata = {
            'url': row.get('file_link', ''),
            'artist': row.get('artist', 'Unknown'),
            'title': row.get('title', 'Unknown')
        }
        
        # Add optional fields if they exist
        if 'year' in row and not pd.isna(row['year']):
            metadata['year'] = str(row['year'])
            
        if 'style_genre' in row and not pd.isna(row['style_genre']):
            metadata['style_genre'] = row['style_genre']
            
        return metadata
    
    def _create_placeholder(self, idx: int) -> Dict[str, Any]:
        """
        Create a placeholder item for error cases.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary with placeholder image and metadata
        """
        # Create a blank black image
        img = self._create_blank_image()
        img_tensor = self.transform(img)
        
        # Create minimal metadata
        metadata = {
            'url': self.df.iloc[idx].get('file_link', '') if idx < len(self.df) else '',
            'artist': 'Error',
            'title': 'Error Loading Image',
            'year': '',
            'error': True
        }
        
        return {
            'image': img_tensor,
            'metadata': metadata
        }


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Union[Tensor, List[MetadataType]]]:
    """
    Custom collate function to handle None values and errors in batches.
    
    Args:
        batch: List of items from the dataset
        
    Returns:
        Dictionary with batched tensors and metadata
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    # If the batch is empty, return a minimal valid batch
    if not batch:
        # Create a dummy tensor with the right shape
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


class FeatureExtractor:
    """
    Feature extractor using pre-trained deep learning models.
    
    This class manages the feature extraction process, handling model loading,
    batch processing, and GPU acceleration.
    """
    def __init__(self, 
                model_name: str = 'resnet50', 
                device: Optional[torch.device] = None,
                use_mixed_precision: bool = True):
        """
        Initialize the feature extractor.
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to use for computation (default: GPU if available, else CPU)
            use_mixed_precision: Whether to use mixed precision for faster computation on supported GPUs
        """
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mixed_precision = use_mixed_precision and self.device.type == 'cuda'
        self.model = self._build_model()
        
        print(f"Feature extractor initialized with {model_name} on {self.device}")
        if self.use_mixed_precision:
            print("Using mixed precision for faster computation")
    
    def _build_model(self) -> nn.Module:
        """
        Build the feature extraction model.
        
        Returns:
            PyTorch model for feature extraction
        
        Raises:
            ValueError: If the model name is not supported
        """
        if self.model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Remove classification layer
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
            feature_dim = 2048
        elif self.model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
            feature_dim = 512
        elif self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
            feature_dim = 1280
        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Move to device and set to evaluation mode
        feature_extractor.to(self.device)
        feature_extractor.eval()
        
        # Store feature dimension for later use
        self.feature_dim = feature_dim
        
        return feature_extractor
    
    def extract_features(self, 
                         dataframe: pd.DataFrame, 
                         batch_size: int = 32, 
                         num_workers: int = 4) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Extract features from all images in the DataFrame.
        
        Args:
            dataframe: DataFrame containing artwork data
            batch_size: Batch size for processing
            num_workers: Number of worker threads for data loading
            
        Returns:
            Tuple containing (features array, metadata list)
        """
        # Optimize parameters based on hardware
        if self.device.type == 'cuda':
            batch_size = batch_size or 64
            num_workers = num_workers or 4
            pin_memory = True
            prefetch_factor = 4
        else:
            batch_size = batch_size or 32
            num_workers = num_workers or 2
            pin_memory = False
            prefetch_factor = 2
        
        # Create dataset and dataloader
        dataset = ArtworkDataset(dataframe)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn,
            prefetch_factor=prefetch_factor
        )
        
        # Arrays to store results
        all_features = []
        all_metadata = []
        
        # Process batches
        total_batches = len(dataloader)
        print(f"Processing {len(dataframe)} images in {total_batches} batches")
        
        with torch.no_grad():  # Disable gradient calculation
            for batch in tqdm(dataloader, desc="Extracting features", total=total_batches):
                images = batch['image'].to(self.device, non_blocking=True)
                
                # Use mixed precision if available
                if self.use_mixed_precision:
                    try:
                        from torch.cuda.amp import autocast
                        with autocast():
                            features = self.model(images)
                    except ImportError:
                        features = self.model(images)
                else:
                    features = self.model(images)
                
                # Process features
                features = features.squeeze().cpu().numpy()
                
                # Handle single image case
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)
                
                all_features.append(features)
                all_metadata.extend(batch['metadata'])
        
        # Combine all features into a single array
        features_array = np.vstack(all_features)
        
        print(f"Extracted features for {len(all_metadata)} images, shape: {features_array.shape}")
        return features_array, all_metadata
    
    def extract_single_feature(self, image: ImageType) -> np.ndarray:
        """
        Extract features from a single image.
        
        Args:
            image: Image to process (URL, path, or PIL Image)
            
        Returns:
            Feature vector as numpy array
        """
        # Prepare image
        if isinstance(image, str):
            if image.startswith(('http://', 'https://')):
                response = requests.get(image, timeout=10)
                img = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                img = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply transformations
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            if self.use_mixed_precision:
                try:
                    from torch.cuda.amp import autocast
                    with autocast():
                        features = self.model(img_tensor)
                except ImportError:
                    features = self.model(img_tensor)
            else:
                features = self.model(img_tensor)
        
        # Convert to numpy
        return features.squeeze().cpu().numpy()


# Factory function for easy creation
def create_feature_extractor(model_name: str = 'resnet50') -> FeatureExtractor:
    """
    Create and initialize a feature extractor.
    
    Args:
        model_name: Name of the pre-trained model to use
        
    Returns:
        Initialized FeatureExtractor
    """
    return FeatureExtractor(model_name=model_name)