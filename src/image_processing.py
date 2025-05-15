"""
Image processing module for the ArtRecognition system.

This module handles image loading, transformation, and processing
operations needed for the artwork recognition system.
"""

import torch
from torchvision import transforms
from PIL import Image, ImageOps
import requests
from io import BytesIO
import time
from typing import Optional, Union, Tuple

# Type alias for various image inputs
ImageType = Union[str, bytes, Image.Image]


def get_transform() -> transforms.Compose:
    """
    Get the standard image transformation pipeline.
    
    The transformation pipeline includes:
    1. Resize to 256x256 (preserving aspect ratio)
    2. Center crop to 224x224 (standard input for ResNet)
    3. Convert to tensor
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


def load_image(image_source: ImageType) -> Optional[Image.Image]:
    """
    Load an image from various sources.
    
    Args:
        image_source: URL, file path, bytes, or PIL Image
        
    Returns:
        PIL Image or None if loading fails
    """
    try:
        # Handle URL
        if isinstance(image_source, str) and image_source.startswith(('http://', 'https://')):
            return load_image_from_url(image_source)
        
        # Handle file path
        elif isinstance(image_source, str):
            return Image.open(image_source).convert('RGB')
        
        # Handle bytes
        elif isinstance(image_source, bytes):
            return Image.open(BytesIO(image_source)).convert('RGB')
        
        # Handle PIL Image
        elif isinstance(image_source, Image.Image):
            return image_source.convert('RGB')  # Ensure RGB mode
        
        else:
            print(f"Unsupported image source type: {type(image_source)}")
            return None
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def load_image_from_url(url: str) -> Optional[Image.Image]:
    """
    Load an image from a URL with retry logic.
    
    Args:
        url: URL to the image
        
    Returns:
        PIL Image or None if loading fails
    """
    # Fixed settings
    timeout = 10
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Raise exception for HTTP errors
            return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                print(f"Retry {attempt+1}/{max_retries} for URL: {url}")
                time.sleep(retry_delay)
            else:
                print(f"Failed to load image from {url}: {e}")
                return None
    
    return None  # Should not reach here, but just in case


def process_image(image: ImageType) -> Optional[torch.Tensor]:
    """
    Process an image for feature extraction.
    
    This function:
    1. Loads the image from the given source
    2. Applies the standard transformations
    3. Returns a tensor ready for the neural network
    
    Args:
        image: Image to process (URL, path, bytes, or PIL Image)
        
    Returns:
        Processed image tensor or None if processing fails
    """
    # Load the image
    img = load_image(image)
    if img is None:
        return None
    
    # Apply standard transformations
    transform = get_transform()
    try:
        img_tensor = transform(img)
        return img_tensor
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def create_blank_image(width: int = 224, height: int = 224) -> Image.Image:
    """
    Create a blank black image.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Blank PIL Image
    """
    return Image.new('RGB', (width, height), color=(0, 0, 0))


def resize_for_display(image: ImageType, max_size: int = 800) -> Optional[Image.Image]:
    """
    Resize an image for display purposes, preserving aspect ratio.
    
    Args:
        image: Image to resize
        max_size: Maximum size (width or height)
        
    Returns:
        Resized PIL Image or None if processing fails
    """
    # Load the image
    img = load_image(image)
    if img is None:
        return None
    
    # Get original size
    width, height = img.size
    
    # Check if resizing is needed
    if width <= max_size and height <= max_size:
        return img
    
    # Calculate new size
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # Resize
    try:
        return img.resize((new_width, new_height), Image.BILINEAR)
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None


def crop_center(image: ImageType, crop_width: int = 224, crop_height: int = 224) -> Optional[Image.Image]:
    """
    Crop the center region of an image.
    
    Args:
        image: Image to crop
        crop_width: Width of the crop
        crop_height: Height of the crop
        
    Returns:
        Cropped PIL Image or None if processing fails
    """
    # Load the image
    img = load_image(image)
    if img is None:
        return None
    
    # Get dimensions
    width, height = img.size
    
    # Calculate crop coordinates
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    
    # Crop
    try:
        return img.crop((left, top, right, bottom))
    except Exception as e:
        print(f"Error cropping image: {e}")
        return None


def process_query_image(image: ImageType) -> Tuple[Optional[torch.Tensor], Optional[Image.Image]]:
    """
    Process a query image for artwork recognition.
    
    This function:
    1. Loads the image
    2. Creates a processed tensor for the neural network
    3. Returns both the tensor and the original image for display
    
    Args:
        image: Image to process
        
    Returns:
        Tuple of (processed tensor, original image) or (None, None) if processing fails
    """
    # Load the image
    img = load_image(image)
    if img is None:
        return None, None
    
    # Process for the neural network
    try:
        transform = get_transform()
        img_tensor = transform(img)
        return img_tensor, img
    except Exception as e:
        print(f"Error processing query image: {e}")
        return None, img  # Return original image even if processing fails