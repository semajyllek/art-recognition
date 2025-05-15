# art-recognition: Deep Learning-Based Artwork Recognition System

art-recognition is a Python-based system that can identify artwork from photographs, similar to how TinEye works for general images. Using deep learning and similarity search techniques, it can recognize artwork even when photographed from different angles, with varying lighting conditions, or at different sizes.

## Overview

The system creates "embeddings" (numerical representations) of artwork images using a pre-trained neural network. These embeddings capture the visual essence of each artwork, allowing for similarity matching with new query images. The system is designed to work with large datasets of artwork images and to be run in GPU environments like Google Colab.


## Key Features

- Identify artworks from photographs with different lighting, angles, and sizes
- Incremental training - build your database over time, 1000 examples at a time
- GPU-accelerated processing for faster feature extraction
- HNSW indexing for efficient similarity search even with large datasets
- Works with the art dataset (215,000+ artworks) from Hugging Face
- Persistent storage in Google Drive for reuse across sessions

## Installation Instructions

### Option 1: Install from GitHub

```bash
# Clone the repository
git clone https://github.com/semajyllek/art-recognition.git
cd art-recognition

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Install using pip (if published)

```bash
pip install art-recognition
```

### Option 3: For Google Colab

```python
# Install from GitHub
!pip install git+https://github.com/semajyllek/art-recognition.git

# Or install required packages manually
!pip install -q datasets faiss-cpu torch torchvision tqdm matplotlib pillow requests pandas
```

## System Architecture

The system is organized into several focused modules, each handling a specific aspect:

1. **data_handling.py**: Loads and processes the artwork dataset, extracting metadata
2. **dataset.py**: Handles loading and preprocessing of artwork images
3. **feature_extraction.py**: Extracts visual features from images using neural networks
4. **index_management.py**: Creates and manages FAISS indexes for similarity search
5. **system_management.py**: Handles saving, loading, and maintaining system state
6. **image_processing.py**: Processes images for feature extraction and display
7. **visualization.py**: Visualizes search results and system status
8. **incremental_training.py**: Enables adding artworks to the system incrementally

## Logic and Data Flow

Here's how the system works:

1. **Data Loading**: The art dataset is loaded and parsed to extract artwork metadata
2. **Feature Extraction**:
   - Each artwork image is processed through a series of transformations (resize, crop, normalize)
   - A pre-trained ResNet50 neural network extracts a 2048-dimensional feature vector
   - These vectors capture the visual essence of each artwork
3. **Index Building**:
   - Feature vectors are added to a FAISS index (specifically HNSW for large datasets)
   - The index enables fast similarity searching
4. **Saving the System**:
   - The index, raw feature vectors, and metadata are saved to Google Drive
   - This enables incremental training and persistence across sessions
5. **Query Processing**:
   - When a query image is uploaded, it undergoes the same transformations
   - Its feature vector is extracted using the same neural network
   - The FAISS index finds the most similar vectors (nearest neighbors)
   - Matching artworks are displayed with similarity scores

## Using art-recognition in Google Colab

### Quick Start

The simplest way to use art-recognition is with the main function:

```python
def run_artwork_recognition(save_dir='/content/drive/MyDrive/artwork_recognition'):
    """
    Main function to run the artwork recognition system.
    Handles the entire workflow in a prescribed manner.
    """
    # Import required modules
    import os
    from google.colab import drive, files
    
    # Mount Google Drive for persistent storage
    print("Mounting Google Drive...")
    drive.mount('/content/drive')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Files will be saved to: {save_dir}")
    
    # Import system modules
    import data_handling
    import feature_extraction
    import incremental_training
    import image_processing
    import index_management
    import visualization
    import system_management
    
    # 1. Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Add more artworks to the database")
    print("2. Identify an artwork from an image")
    print("3. Delete the existing system and start fresh")
    
    choice = input("Enter your choice (1/2/3): ")
    
    # 2. Handle user choice
    if choice == '1':
        # Add more artworks
        print("\nAdding artworks to the database...")
        
        # Load dataset
        df = data_handling.load_artwork_dataset()
        
        # Add artworks incrementally
        incremental_training.add_more_artworks(df, save_dir)
        
    elif choice == '2':
        # Identify artwork
        print("\nIdentifying artwork from image...")
        
        # Check if system exists
        if not system_management.system_exists(save_dir):
            print("No artwork database found. Please add artworks first (option 1).")
            return
        
        # Load system
        index, embeddings, metadata = system_management.load_system(save_dir)
        feature_extractor = feature_extraction.create_feature_extractor()
        
        # Upload image
        print("Please upload an image of the artwork:")
        uploaded = files.upload()
        
        if not uploaded:
            print("No image uploaded.")
            return
            
        # Process the single uploaded image
        filename = next(iter(uploaded.keys()))
        print(f"Processing image: {filename}")
        
        # Extract features
        img_tensor = image_processing.process_image(filename)
        query_vector = feature_extractor.extract_single_feature(img_tensor)
        
        # Search
        results = index_management.search_artwork(query_vector, index, metadata, k=5)
        
        # Display results
        visualization.display_search_results(filename, results)
        visualization.print_search_results(results)
        
    elif choice == '3':
        # Delete and start fresh
        print("\nDeleting existing system...")
        
        confirm = input("Are you sure you want to delete the existing system? (y/n): ")
        if confirm.lower() == 'y':
            system_management.delete_system(save_dir)
            print("System deleted. Run again to create a new one.")
        else:
            print("Deletion cancelled.")
        
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    print("\nOperation completed. Run this function again to perform another action.")

# Example usage
run_artwork_recognition()
```

### Setting Up a Colab Notebook

1. Create a new Google Colab notebook
2. Install the required packages:

```python
# Install required packages
!pip install -q datasets faiss-cpu torch torchvision tqdm matplotlib pillow requests pandas
```

3. Create a new code cell for each module, and paste the corresponding code
4. Run the main function in a final cell

## Typical Workflow

art-recognition provides a streamlined, prescribed workflow:

1. **Initial Setup**:
   - Run the system for the first time
   - Choose option 1 to add artworks to the database
   - The system will create a database with the first 1000 artworks

2. **Adding More Artworks**:
   - Run the system again
   - Choose option 1 to add more artworks
   - The system will incrementally add another 1000 artworks

3. **Identifying Artwork**:
   - Run the system
   - Choose option 2 to identify an artwork
   - Upload an image
   - View the matching results

4. **Starting Fresh**:
   - If needed, choose option 3 to delete the system and start over

## GPU Optimization

The system automatically optimizes for GPU usage:

- Detects if a GPU is available
- Uses larger batch sizes on GPU (64 vs 32 on CPU)
- Implements mixed precision where supported
- Uses non-blocking data transfers to improve performance

## Incremental Training

For the full art dataset (215K images), incremental training is essential:

- Processes artworks in manageable chunks of 1000
- Tracks which artworks have been processed to avoid duplicates
- Combines new embeddings with existing ones
- Updates the search index to include all artworks

This approach allows you to build a comprehensive database over time, even in resource-constrained environments.

## System Requirements

- Python 3.7+
- PyTorch 1.9+
- FAISS 1.7+
- 8+ GB RAM (16+ GB recommended for larger batches)
- GPU with 8+ GB VRAM for optimal performance (can run on CPU but slower)
- Google Drive space: ~500 MB for 1000 artworks, ~10 GB for full dataset

## Troubleshooting

### Common Issues

1. **"Out of memory" errors**:
   - Reduce batch size
   - Process fewer artworks at once
   - Restart the Colab runtime to clear GPU memory

2. **System loading errors**:
   - Check that all files exist in the save directory
   - If corrupted, use option 3 to delete the system and start fresh

3. **Slow processing**:
   - Verify GPU is enabled
   - Check for other processes using GPU resources
   - Consider using a more powerful Colab instance (Colab Pro)

4. **Image not recognized**:
   - Try a clearer photo with better lighting
   - Ensure the artwork is clearly visible and not severely distorted
   - Add more artworks to the database for better coverage

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for hosting the artwork dataset
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [torchvision](https://pytorch.org/vision/stable/index.html) for pre-trained models and image transformations

---

This system lets you build a powerful artwork recognition engine that can identify artworks from photographs. By using incremental training, you can scale it to handle the full dataset of 215,000+ images even in resource-constrained environments.