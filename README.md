# art-recognition: Deep Learning-Based Artwork Recognition System

art-recognition is a Python-based system that can identify artwork from photographs, similar to how TinEye works for general images. Using deep learning and similarity search techniques, it can recognize artwork even when photographed from different angles, with varying lighting conditions, or at different sizes.

## Overview

The system creates "embeddings" (numerical representations) of artwork images using a pre-trained neural network. These embeddings capture the visual essence of each artwork, allowing for similarity matching with new query images. The system is designed to work with the WikiArt dataset (215,000+ images) and to be run in GPU environments like Google Colab.

![System Overview](https://i.imgur.com/0NJA8A9.png)

## Key Features

- Identify artworks from photographs with different lighting, angles, and sizes
- Incremental training - build your database over time, 1000 examples at a time
- GPU-accelerated processing for faster feature extraction
- HNSW indexing for efficient similarity search even with large datasets
- Works with the WikiArt dataset (215,000+ artworks)
- Persistent storage in Google Drive for reuse across sessions

## System Architecture

### Architecture Overview

The system is organized into several modules, each handling a specific aspect:

1. **data_handling.py**: Loads and processes the WikiArt dataset, extracting metadata
2. **feature_extraction.py**: Extracts visual features from images using neural networks
3. **index_management.py**: Creates and manages FAISS indexes for similarity search
4. **system_management.py**: Handles saving, loading, and maintaining system state
5. **image_processing.py**: Processes images for feature extraction and display
6. **visualization.py**: Visualizes search results and system status
7. **incremental_training.py**: Enables adding artworks to the system incrementally

### Logic and Data Flow

Here's how the system works:

1. **Data Loading**: The WikiArt dataset is loaded and parsed to extract artwork metadata
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

### Setting Up a Colab Notebook

1. Create a new Google Colab notebook
2. Install the required packages:

```python
# Install required packages
!pip install -q datasets faiss-cpu torch torchvision tqdm matplotlib pillow requests
```

3. Mount Google Drive for persistent storage:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create directory for saving system
import os
SAVE_DIR = '/content/drive/MyDrive/artwork_recognition'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Files will be saved to: {SAVE_DIR}")
```

4. Import art-recognition (either as modules or by copying code):

```python
# Option 1: Clone repository (if available on GitHub)
!git clone https://github.com/semajyllek/art-recognition.git
```

### Basic Usage

Once you have set up the environment, you can use the system as follows:

```python
# Import from art-recognition
from art-recognition import data_handling, feature_extraction, index_management
from art-recognition import system_management, image_processing, visualization
from art-recognition import incremental_training

# Define paths
SAVE_DIR = '/content/drive/MyDrive/artwork_recognition'

# Load dataset
df = data_handling.load_artwork_dataset(limit=None)  # Remove limit for full dataset
```

### Training the System

For initial training with a small batch:

```python
# Create feature extractor
feature_extractor = feature_extraction.create_feature_extractor()

# Run incremental training (adds 1000 artworks)
index, metadata = incremental_training.train_incrementally(
    dataframe=df,
    save_dir=SAVE_DIR,
    feature_extractor=feature_extractor,
    chunk_size=1000  # Start with 1000 for quick testing
)
```

### Searching for Artwork

To identify an artwork from an uploaded image:

```python
# Import Google Colab tools
from google.colab import files

# Function to identify artwork
def identify_artwork():
    # Load the system
    index, embeddings, metadata = system_management.load_system(SAVE_DIR)
    feature_extractor = feature_extraction.create_feature_extractor()
    
    # Upload an image
    print("Upload an image of artwork to identify:")
    uploaded = files.upload()
    
    # Get the filename (only expecting one image)
    if not uploaded:
        print("No image uploaded")
        return
        
    filename = next(iter(uploaded.keys()))
    print(f"Processing image: {filename}")
    
    # Process the query image
    img_tensor = image_processing.process_image(filename)
    query_vector = feature_extractor.extract_single_feature(img_tensor)
    
    # Search for matches
    results = index_management.search_artwork(query_vector, index, metadata, k=5)
    
    # Display results
    visualization.display_search_results(filename, results)

# Run the identification function
identify_artwork()
```

### Incremental Training

To add more artworks to the system over time:

```python
# Run the incremental training demo
incremental_training.run_incremental_demo(
    dataframe=df,
    save_dir=SAVE_DIR,
    feature_extractor=feature_extractor,
    chunk_size=1000  # Add another 1000 artworks
)
```

## Tips for GPU Environments

To maximize performance in GPU environments:

1. **Enable GPU acceleration** in Google Colab:
   - Go to Runtime → Change runtime type → GPU

2. **Optimize batch size** based on available GPU memory:
   - For high-end GPUs: `batch_size=64` or higher
   - For limited memory: `batch_size=32` or lower

3. **Use mixed precision** for faster computation:
   - The feature extraction module automatically uses mixed precision on compatible GPUs

4. **Process in manageable chunks**:
   - For the full WikiArt dataset (215K images), use incremental training
   - Process 1000-5000 images at a time to avoid memory issues

5. **Save frequently**:
   - The system saves progress after each chunk
   - This allows you to continue from where you left off even if the session terminates

## Typical Workflow

1. **Initial Setup**:
   - Install dependencies
   - Mount Google Drive
   - Import art-recognition modules

2. **First Training Run**:
   - Load WikiArt dataset
   - Train on first 1000 artworks
   - Test the system with a few queries

3. **Incremental Growth**:
   - Run the system again later
   - Add another batch of artworks
   - Repeat until you've processed as many artworks as desired

4. **Production Use**:
   - Load the fully trained system
   - Use it to identify artworks from photos

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
   - If corrupted, use `system_management.delete_system(SAVE_DIR)` and retrain

3. **Slow processing**:
   - Verify GPU is enabled
   - Check for other processes using GPU resources
   - Consider using a more powerful Colab instance (Colab Pro)

4. **Image not recognized**:
   - Try a clearer photo with better lighting
   - Ensure the artwork is clearly visible and not severely distorted
   - Add more artworks to the database for better coverage

## Project Structure

```
art-recognition/
├── data_handling.py        # Dataset loading and metadata extraction
├── feature_extraction.py   # Neural network feature extraction
├── index_management.py     # FAISS index operations
├── system_management.py    # Save, load, delete system functions
├── image_processing.py     # Image transformation and augmentation
├── visualization.py        # Result visualization functions
└── incremental_training.py # Incremental training functionality
```

## Acknowledgments

- [WikiArt](https://www.wikiart.org/) for the artwork dataset
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [torchvision](https://pytorch.org/vision/stable/index.html) for pre-trained models and image transformations

---

This system lets you build a powerful artwork recognition engine that can identify artworks from photographs. By using incremental training, you can scale it to handle the full WikiArt dataset of 215,000+ images even in resource-constrained environments.