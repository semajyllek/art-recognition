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


if __name__ == "__main__":
    run_artwork_recognition()