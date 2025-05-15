"""
Feature extraction module for the ArtRecognition system.

This module handles the extraction of visual features from artwork images
using a pre-trained ResNet50 neural network. It provides a streamlined
approach with fixed settings optimized for artwork recognition.
"""


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple, Dict, Any
from tqdm.auto import tqdm

# Import from dataset module
from dataset import create_dataloader


class FeatureExtractor:
    """Feature extractor using ResNet50."""
    
    def __init__(self):
        """Initialize with fixed settings - no configurable options."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()
        print(f"Feature extractor initialized on {self.device}")
    
    def _build_model(self) -> nn.Module:
        """
        Build ResNet50 feature extractor with fixed settings.
        
        Returns:
            PyTorch model for feature extraction
        """
        from torchvision import models
        
        model = models.resnet50(pretrained=True)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor.to(self.device)
        feature_extractor.eval()
        self.feature_dim = 2048  # Fixed for ResNet50
        return feature_extractor
    
    def extract_features(self, dataframe: pd.DataFrame) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Extract features from all images in the DataFrame.
        
        Args:
            dataframe: DataFrame containing artwork data
            
        Returns:
            Tuple containing (features array, metadata list)
        """
        # Create dataloader
        dataloader, _ = create_dataloader(dataframe)
        
        # Arrays to store results
        all_features = []
        all_metadata = []
        
        # Process batches
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                # Move images to device
                images = batch['image'].to(self.device, non_blocking=True)
                
                # Try using mixed precision on compatible GPUs
                if self.device.type == 'cuda':
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
        
        # Combine all features
        features_array = np.vstack(all_features)
        
        print(f"Extracted features for {len(all_metadata)} images, shape: {features_array.shape}")
        return features_array, all_metadata
    
    def extract_single_feature(self, image) -> np.ndarray:
        """
        Extract features from a single image.
        
        Args:
            image: Image to process (PIL Image or tensor)
            
        Returns:
            Feature vector as numpy array
        """
        # Handle different input types
        if isinstance(image, Image.Image):
            # Apply transformations if it's a PIL Image
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(image).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            # Already a tensor
            if image.dim() == 3:
                img_tensor = image.unsqueeze(0)  # Add batch dimension
            else:
                img_tensor = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Move to device
        img_tensor = img_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
        
        # Return as numpy array
        return features.squeeze().cpu().numpy()


def extract_features(dataframe: pd.DataFrame, extractor=None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Extract features from all images in the DataFrame.
    
    Args:
        dataframe: DataFrame containing artwork data
        extractor: Optional feature extractor (created if None)
        
    Returns:
        Tuple containing (features array, metadata list)
    """
    if extractor is None:
        extractor = create_feature_extractor()
    return extractor.extract_features(dataframe)


def create_feature_extractor() -> FeatureExtractor:
    """
    Create and initialize a feature extractor.
    
    Returns:
        Initialized FeatureExtractor
    """
    return FeatureExtractor()