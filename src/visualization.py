"""
Visualization module for the ArtRecognition system.

This module provides functions for visualizing search results,
comparing query images with matches, and displaying system status.
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import io
import requests
from io import BytesIO
from typing import List, Dict, Any, Optional, Union, Tuple
import time

# Type aliases
ImageType = Union[str, bytes, Image.Image]
MetadataType = Dict[str, Any]
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
        # URL
        if isinstance(image_source, str) and image_source.startswith(('http://', 'https://')):
            response = requests.get(image_source, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        
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


def format_metadata(metadata: MetadataType) -> str:
    """
    Format artwork metadata for display.
    
    Args:
        metadata: Artwork metadata
        
    Returns:
        Formatted string
    """
    # Extract key information
    title = metadata.get('title', 'Unknown Title')
    artist = metadata.get('artist', 'Unknown Artist')
    year = metadata.get('year', '')
    
    # Format the string
    result = f"{title}"
    if artist != 'Unknown Artist':
        result += f"\nby {artist}"
    if year:
        result += f", {year}"
    
    return result


def display_search_results(query_image: ImageType, results: List[SearchResultType], 
                          max_results: int = 5, figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Display search results with the query image and matches.
    
    Args:
        query_image: Query image
        results: List of search results
        max_results: Maximum number of results to display
        figsize: Figure size (width, height) in inches
    """
    # Limit the number of results
    results = results[:max_results]
    num_results = len(results)
    
    # Create figure
    plt.figure(figsize=figsize)
    
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
    for i, result in enumerate(results):
        plt.subplot(1, num_results + 1, i + 2)
        
        # Get metadata and image URL
        metadata = result['metadata']
        url = metadata.get('url', '')
        
        # Calculate similarity score
        similarity = result.get('similarity')
        if similarity is None and 'distance' in result:
            # Convert distance to similarity if not provided
            distance = result['distance']
            similarity = 100.0 * (1.0 - distance / (distance + 10.0))
        
        # Load and display the image
        img = load_image(url)
        if img is not None:
            plt.imshow(img)
            
            # Create title with metadata and similarity
            title = format_metadata(metadata)
            if similarity is not None:
                title += f"\nSimilarity: {similarity:.1f}%"
            
            plt.title(title, fontsize=10)
            plt.axis('off')
        else:
            # Display a placeholder if image can't be loaded
            plt.text(0.5, 0.5, f"Image could not be loaded\n{format_metadata(metadata)}", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def create_result_grid(query_image: ImageType, results: List[SearchResultType], 
                      max_results: int = 5, image_size: int = 256) -> Image.Image:
    """
    Create a grid image with query and results for saving or sharing.
    
    Args:
        query_image: Query image
        results: List of search results
        max_results: Maximum number of results to display
        image_size: Size of each image in the grid
        
    Returns:
        PIL Image with the grid
    """
    # Limit the number of results
    results = results[:max_results]
    num_results = len(results)
    
    # Calculate grid dimensions and create canvas
    grid_width = (num_results + 1) * image_size
    grid_height = image_size + 80  # Extra space for captions
    grid_img = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid_img)
    
    try:
        # Try to load a font, fall back to default if not available
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    # Add query image
    query_img = load_image(query_image)
    if query_img is not None:
        # Resize image
        query_img = query_img.resize((image_size, image_size), Image.BICUBIC)
        grid_img.paste(query_img, (0, 0))
        
        # Add label
        draw.text((10, image_size + 10), "Query Image", fill=(0, 0, 0), font=font)
    
    # Add result images
    for i, result in enumerate(results):
        # Get metadata and image URL
        metadata = result['metadata']
        url = metadata.get('url', '')
        
        # Calculate similarity score
        similarity = result.get('similarity')
        if similarity is None and 'distance' in result:
            # Convert distance to similarity if not provided
            distance = result['distance']
            similarity = 100.0 * (1.0 - distance / (distance + 10.0))
        
        # Calculate position
        x_pos = (i + 1) * image_size
        
        # Load and add the image
        img = load_image(url)
        if img is not None:
            # Resize image
            img = img.resize((image_size, image_size), Image.BICUBIC)
            grid_img.paste(img, (x_pos, 0))
            
            # Create caption with metadata and similarity
            caption = format_metadata(metadata)
            if similarity is not None:
                caption += f"\nSimilarity: {similarity:.1f}%"
            
            # Add caption
            draw.text((x_pos + 10, image_size + 10), caption, fill=(0, 0, 0), font=font)
    
    return grid_img


def show_feature_visualization(embeddings: np.ndarray, metadata: List[MetadataType], 
                             num_samples: int = 1000, method: str = 'tsne') -> None:
    """
    Visualize feature embeddings using dimensionality reduction.
    
    Args:
        embeddings: Feature embeddings
        metadata: Metadata for each embedding
        num_samples: Number of samples to visualize
        method: Dimensionality reduction method ('tsne' or 'pca')
    """
    # Import visualization libraries
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError:
        print("scikit-learn is required for embedding visualization.")
        print("Install it with: pip install scikit-learn")
        return
    
    # Sample embeddings if there are too many
    total_samples = len(embeddings)
    if total_samples > num_samples:
        indices = np.random.choice(total_samples, num_samples, replace=False)
        sampled_embeddings = embeddings[indices]
        sampled_metadata = [metadata[i] for i in indices]
    else:
        sampled_embeddings = embeddings
        sampled_metadata = metadata
    
    # Apply dimensionality reduction
    if method == 'tsne':
        print("Applying t-SNE dimensionality reduction...")
        reducer = TSNE(n_components=2, random_state=42)
    else:  # PCA
        print("Applying PCA dimensionality reduction...")
        reducer = PCA(n_components=2, random_state=42)
    
    # Reduce dimensions
    reduced_embeddings = reducer.fit_transform(sampled_embeddings)
    
    # Extract artist information for coloring
    artists = [m.get('artist', 'Unknown') for m in sampled_metadata]
    unique_artists = list(set(artists))
    
    # Limit to top 20 artists for clarity
    if len(unique_artists) > 20:
        # Count artworks per artist
        artist_counts = {}
        for artist in artists:
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
        
        # Take top 20 by count
        top_artists = sorted(unique_artists, key=lambda a: artist_counts.get(a, 0), reverse=True)[:20]
        other_mask = np.array([a not in top_artists for a in artists])
        
        # Replace other artists with 'Other'
        for i in range(len(artists)):
            if artists[i] not in top_artists:
                artists[i] = 'Other'
        
        unique_artists = top_artists + ['Other']
    
    # Create color map
    artist_to_color = {artist: i for i, artist in enumerate(unique_artists)}
    colors = [artist_to_color.get(artist, 0) for artist in artists]
    
    # Create plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                         c=colors, alpha=0.7, cmap='tab20')
    
    # Add legend for artists
    if len(unique_artists) <= 20:  # Only show legend if not too cluttered
        legend1 = plt.legend(handles=scatter.legend_elements()[0], 
                           labels=unique_artists, 
                           title="Artists", 
                           loc="upper right")
        plt.gca().add_artist(legend1)
    
    plt.title(f"Feature Embedding Visualization using {method.upper()}")
    plt.tight_layout()
    plt.show()


def plot_similarity_distribution(distances: np.ndarray, bins: int = 50) -> None:
    """
    Plot the distribution of similarity scores.
    
    Args:
        distances: Array of distance values
        bins: Number of histogram bins
    """
    # Convert distances to similarities
    similarities = 100.0 * (1.0 - distances / (distances + 10.0))
    
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Similarity Score (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarity Scores')
    plt.axvline(x=np.mean(similarities), color='red', linestyle='--', 
               label=f'Mean: {np.mean(similarities):.2f}%')
    plt.axvline(x=np.median(similarities), color='green', linestyle='--', 
               label=f'Median: {np.median(similarities):.2f}%')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_system_stats(system_info: Dict[str, Any], metadata: List[MetadataType]) -> None:
    """
    Display system statistics and information.
    
    Args:
        system_info: System information dictionary
        metadata: Metadata for all artworks
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create a text summary
    summary = []
    summary.append(f"System Version: {system_info.get('version', 'Unknown')}")
    summary.append(f"Created: {system_info.get('creation_time', 'Unknown')}")
    summary.append(f"Last Updated: {system_info.get('last_update_time', 'Unknown')}")
    summary.append(f"Total Artworks: {system_info.get('num_artworks', len(metadata))}")
    summary.append(f"Embedding Dimension: {system_info.get('embedding_dim', 'Unknown')}")
    
    # Count unique artists, years, etc.
    artists = set()
    years = set()
    
    for item in metadata:
        artist = item.get('artist')
        if artist:
            artists.add(artist)
        
        year = item.get('year')
        if year:
            years.add(year)
    
    summary.append(f"Unique Artists: {len(artists)}")
    summary.append(f"Year Range: {min(years) if years else 'Unknown'} - {max(years) if years else 'Unknown'}")
    
    # Join the summary
    summary_text = "\n".join(summary)
    
    # Add to plot
    plt.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=1", facecolor='white', alpha=0.8))
    
    plt.title("ArtRecognition System Statistics")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_search_process(query_image: ImageType, results: List[SearchResultType], 
                           k_values: List[int] = [1, 3, 5, 10]) -> None:
    """
    Visualize how search results change with different k values.
    
    Args:
        query_image: Query image
        results: Full list of search results
        k_values: List of k values to visualize
    """
    # Load query image
    query_img = load_image(query_image)
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(k_values), 1, figsize=(15, 5 * len(k_values)))
    
    # Single value case
    if len(k_values) == 1:
        axes = [axes]
    
    # For each k value
    for i, k in enumerate(k_values):
        ax = axes[i]
        
        # Select top-k results
        top_k_results = results[:k]
        
        # Create a horizontal display of images
        ax.set_title(f"Top {k} Results", fontsize=14)
        ax.axis('off')
        
        # Calculate grid positions
        num_images = len(top_k_results) + 1  # +1 for query image
        width = 1.0 / num_images
        
        # Add query image
        if query_img is not None:
            ax.imshow(query_img, extent=(0, width, 0, 1))
            ax.text(width/2, 1.05, "Query", ha='center', va='bottom', fontsize=12)
        
        # Add result images
        for j, result in enumerate(top_k_results):
            # Calculate position
            x_start = (j + 1) * width
            
            # Get metadata and URL
            metadata = result['metadata']
            url = metadata.get('url', '')
            
            # Calculate similarity
            similarity = result.get('similarity')
            if similarity is None and 'distance' in result:
                distance = result['distance']
                similarity = 100.0 * (1.0 - distance / (distance + 10.0))
            
            # Load and display image
            img = load_image(url)
            if img is not None:
                ax.imshow(img, extent=(x_start, x_start + width, 0, 1))
                
                # Add metadata as text
                title = metadata.get('title', 'Unknown')
                artist = metadata.get('artist', 'Unknown')
                
                ax.text(x_start + width/2, 1.05, f"{title}", 
                       ha='center', va='bottom', fontsize=10)
                ax.text(x_start + width/2, -0.05, f"{artist}", 
                       ha='center', va='top', fontsize=10)
                
                if similarity is not None:
                    ax.text(x_start + width/2, -0.15, f"{similarity:.1f}%", 
                           ha='center', va='top', fontsize=10, 
                           color='green' if similarity > 75 else 'black')
    
    plt.tight_layout()
    plt.show()