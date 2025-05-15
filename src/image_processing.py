"""
Image processing module for the ArtRecognition system.

This module provides functions for image loading, transformation,
and augmentation to prepare images for feature extraction and
improve recognition robustness.
"""

import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import requests
from io import BytesIO
import time
import random
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

# Type aliases
ImageType = Union[str, bytes, Image.Image]


# Standard transformation pipeline for artwork images
def get_standard_transform() -> transforms.Compose:
    """
    Get the standard image transformation pipeline.
    
    This transformation prepares images for the neural network:
    1. Resize to 256x256 pixels
    2. Center crop to 224x224 pixels (standard input size for many CNNs)
    3. Convert to PyTorch tensor
    4. Normalize with ImageNet mean and std values
    
    Returns:
        PyTorch transforms composition
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet standard deviations
        )
    ])


def load_image_from_url(url: str, timeout: int = 10, 
                      max_retries: int = 3, retry_delay: float = 1.0) -> Optional[Image.Image]:
    """
    Load an image from a URL with retry logic.
    
    Args:
        url: URL to the image
        timeout: Timeout for the request in seconds
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        PIL Image or None if loading fails
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Raise exception for HTTP errors
            return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt+1}/{max_retries} for URL: {url}")
                time.sleep(retry_delay)
            else:
                print(f"Failed to load image from {url}: {e}")
                return None


def load_image(image_source: ImageType) -> Optional[Image.Image]:
    """
    Load an image from various sources.
    
    Args:
        image_source: URL, file path, bytes, or PIL Image
        
    Returns:
        PIL Image or None if loading fails
    """
    try:
        # URL
        if isinstance(image_source, str) and image_source.startswith(('http://', 'https://')):
            return load_image_from_url(image_source)
        
        # File path
        elif isinstance(image_source, str):
            return Image.open(image_source).convert('RGB')
        
        # Bytes
        elif isinstance(image_source, bytes):
            return Image.open(BytesIO(image_source)).convert('RGB')
        
        # PIL Image
        elif isinstance(image_source, Image.Image):
            return image_source.convert('RGB')  # Ensure RGB mode
        
        else:
            print(f"Unsupported image source type: {type(image_source)}")
            return None
    
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def process_image(image: ImageType) -> Optional[torch.Tensor]:
    """
    Process an image for feature extraction.
    
    Args:
        image: Image to process (URL, path, bytes, or PIL Image)
        
    Returns:
        Processed image as a PyTorch tensor or None if processing fails
    """
    try:
        # Load the image
        img = load_image(image)
        if img is None:
            return None
        
        # Apply standard transformations
        transform = get_standard_transform()
        img_tensor = transform(img)
        
        return img_tensor
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def create_augmented_versions(image: Image.Image, num_versions: int = 5) -> List[Image.Image]:
    """
    Create augmented versions of an image to improve recognition robustness.
    
    The augmentations simulate different viewing conditions:
    - Slight rotations (as if photo is taken at an angle)
    - Brightness/contrast variations (different lighting conditions)
    - Slight crops (different framing of the artwork)
    - Perspective changes (different viewing angles)
    
    Args:
        image: Original image
        num_versions: Number of augmented versions to create
        
    Returns:
        List of augmented images including the original
    """
    results = [image]  # Include original image
    
    # Create multiple variations
    for _ in range(num_versions - 1):
        # Start with a copy of the original
        img_copy = image.copy()
        
        # Apply a random sequence of transformations
        
        # 1. Random rotation (-15 to 15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            img_copy = img_copy.rotate(angle, resample=Image.BICUBIC, expand=False)
        
        # 2. Random brightness adjustment (0.8 to 1.2)
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            img_copy = ImageEnhance.Brightness(img_copy).enhance(brightness_factor)
        
        # 3. Random contrast adjustment (0.8 to 1.2)
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            img_copy = ImageEnhance.Contrast(img_copy).enhance(contrast_factor)
        
        # 4. Random crop and resize
        if random.random() > 0.5:
            width, height = img_copy.size
            # Crop 80-95% of the image
            crop_percent = random.uniform(0.8, 0.95)
            left = random.uniform(0, 1 - crop_percent) * width
            top = random.uniform(0, 1 - crop_percent) * height
            right = left + crop_percent * width
            bottom = top + crop_percent * height
            
            img_copy = img_copy.crop((left, top, right, bottom))
            img_copy = img_copy.resize((width, height), Image.BICUBIC)
        
        # 5. Small perspective transform
        if random.random() > 0.5:
            width, height = img_copy.size
            
            # Define perspective transform
            # Slightly move the corners to simulate change in viewing angle
            max_shift = 0.05  # Max shift as proportion of width/height
            
            # Calculate random shifts for corners
            shifts = [
                random.uniform(-max_shift, max_shift) * width,  # Top left x
                random.uniform(-max_shift, max_shift) * height, # Top left y
                random.uniform(-max_shift, max_shift) * width,  # Top right x
                random.uniform(-max_shift, max_shift) * height, # Top right y
                random.uniform(-max_shift, max_shift) * width,  # Bottom right x
                random.uniform(-max_shift, max_shift) * height, # Bottom right y
                random.uniform(-max_shift, max_shift) * width,  # Bottom left x
                random.uniform(-max_shift, max_shift) * height  # Bottom left y
            ]
            
            # Original coordinates of the corners
            coords = [
                0, 0,                  # Top left
                width, 0,              # Top right
                width, height,         # Bottom right
                0, height              # Bottom left
            ]
            
            # Apply shifts
            coeffs = []
            for i in range(4):
                coeffs.extend([
                    coords[i*2] + shifts[i*2], 
                    coords[i*2+1] + shifts[i*2+1]
                ])
            
            # Apply perspective transform
            try:
                img_copy = img_copy.transform(
                    (width, height), 
                    Image.PERSPECTIVE, 
                    coeffs, 
                    Image.BICUBIC
                )
            except Exception:
                # Fall back to affine transform if perspective fails
                img_copy = img_copy
        
        # Add the augmented image to results
        results.append(img_copy)
    
    return results


def preprocess_batch(images: List[ImageType]) -> torch.Tensor:
    """
    Process a batch of images for feature extraction.
    
    Args:
        images: List of images to process
        
    Returns:
        Batch of processed images as a PyTorch tensor
    """
    # Load and process each image
    processed_images = []
    transform = get_standard_transform()
    
    for image in images:
        img = load_image(image)
        if img is not None:
            # Apply transformations
            try:
                img_tensor = transform(img)
                processed_images.append(img_tensor)
            except Exception as e:
                print(f"Error processing image: {e}")
    
    # Stack into a batch
    if processed_images:
        return torch.stack(processed_images)
    else:
        return torch.zeros((0, 3, 224, 224))  # Empty batch


def prepare_query_image(image: ImageType) -> Tuple[torch.Tensor, Optional[Image.Image]]:
    """
    Prepare a query image for searching.
    
    Args:
        image: Image to prepare (URL, path, bytes, or PIL Image)
        
    Returns:
        Tuple of (processed tensor, original PIL Image)
    """
    # Load the image
    img = load_image(image)
    if img is None:
        # Return empty tensor and None
        return torch.zeros((1, 3, 224, 224)), None
    
    # Process for the network
    transform = get_standard_transform()
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    return img_tensor, img


def process_image_for_display(image: ImageType, max_size: int = 800) -> Optional[Image.Image]:
    """
    Process an image for display purposes.
    
    Args:
        image: Image to process
        max_size: Maximum size for either dimension
        
    Returns:
        Processed PIL Image or None if processing fails
    """
    # Load the image
    img = load_image(image)
    if img is None:
        return None
    
    # Resize if necessary, maintaining aspect ratio
    width, height = img.size
    if width > max_size or height > max_size:
        if width >= height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        img = img.resize((new_width, new_height), Image.BICUBIC)
    
    return img