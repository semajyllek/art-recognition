"""
Visualization module for the ArtRecognition system.

This module provides functions for visualizing search results
and displaying artwork matches with their metadata.
"""

import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from typing import List, Dict, Any, Optional, Union

# Type alias
ImageType = Union[str, bytes, Image.Image]
SearchResultType = Dict[str, Any]


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
            response = requests.get(image_source, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        
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


def format_metadata(metadata: Dict[str, Any]) -> str:
    """
    Format artwork metadata for display.
    
    Args:
        metadata: Artwork metadata dictionary
        
    Returns:
        Formatted string for display
    """
    # Extract key information
    title = metadata.get('title', 'Unknown Title')
    artist = metadata.get('artist', 'Unknown Artist')
    year = metadata.get('year', '')
    
    # Format the display string
    display_text = f"{title}"
    
    if artist != 'Unknown Artist':
        display_text += f"\nby {artist}"
    
    if year:
        display_text += f", {year}"
    
    return display_text


def display_search_results(query_image: ImageType, results: List[SearchResultType]) -> None:
    """
    Display search results with the query image and matches.
    
    This function creates a figure showing the query image and the
    top matching artworks with their metadata and similarity scores.
    
    Args:
        query_image: The query image
        results: List of search results from index_management.search_artwork
    """
    # Fixed settings
    num_results = min(5, len(results))  # Show at most 5 results
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Display query image
    plt.subplot(1, num_results + 1, 1)
    query_img = load_image(query_image)
    
    if query_img is not None:
        plt.imshow(query_img)
        plt.title("Query Image", fontsize=12)
        plt.axis('off')
    else:
        plt.text(0.5, 0.5, "Query image could not be loaded", 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
    
    # Display results
    for i, result in enumerate(results[:num_results]):
        plt.subplot(1, num_results + 1, i + 2)
        
        # Get metadata and URL
        metadata = result['metadata']
        url = metadata.get('url', '')
        
        # Get similarity score
        similarity = result.get('similarity', None)
        if similarity is None and 'distance' in result:
            # Convert distance to similarity
            distance = result['distance']
            similarity = 100.0 * (1.0 - distance / (distance + 10.0))
        
        # Load and display the image
        img = load_image(url)
        if img is not None:
            plt.imshow(img)
            
            # Format title with metadata and similarity
            title = format_metadata(metadata)
            if similarity is not None:
                title += f"\nSimilarity: {similarity:.1f}%"
            
            plt.title(title, fontsize=10)
            plt.axis('off')
        else:
            # Display a placeholder for failed image loads
            plt.text(0.5, 0.5, f"Image could not be loaded\n{format_metadata(metadata)}", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def print_search_results(results: List[SearchResultType]) -> None:
    """
    Print search results in a readable format.
    
    Args:
        results: List of search results from index_management.search_artwork
    """
    print("\n===== Search Results =====")
    
    for i, result in enumerate(results):
        metadata = result['metadata']
        similarity = result.get('similarity', None)
        
        if similarity is None and 'distance' in result:
            # Convert distance to similarity
            distance = result['distance']
            similarity = 100.0 * (1.0 - distance / (distance + 10.0))
        
        # Print result details
        print(f"\nMatch #{i+1}:")
        print(f"  Title: {metadata.get('title', 'Unknown Title')}")
        print(f"  Artist: {metadata.get('artist', 'Unknown Artist')}")
        
        if 'year' in metadata and metadata['year']:
            print(f"  Year: {metadata['year']}")
            
        if similarity is not None:
            print(f"  Similarity: {similarity:.1f}%")
    
    print("\n=========================")


def display_progress(current: int, total: int, message: str = "Processing") -> None:
    """
    Display a simple progress message.
    
    Args:
        current: Current progress value
        total: Total value
        message: Progress message prefix
    """
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"{message}: {current}/{total} ({percentage:.1f}%)")