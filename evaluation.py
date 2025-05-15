"""
evaluation.py - Evaluation script for the art-recognition system

This script evaluates the performance of the art-recognition system
by testing it against a set of test images and calculating metrics
such as accuracy, precision, and recall.
"""

import os
import numpy as np
import pandas as pd
import time
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json
from pathlib import Path


def load_test_dataset(test_csv_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load a test dataset from a CSV file.
    
    Args:
        test_csv_path: Path to the test CSV file
        limit: Optional limit on number of test samples
        
    Returns:
        DataFrame with test data
    """
    print(f"Loading test dataset from {test_csv_path}")
    df = pd.read_csv(test_csv_path)
    
    # Apply basic validation
    required_columns = ['file_link']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Test dataset missing required column: {col}")
    
    # Extract essential metadata if not already present
    if 'artist' not in df.columns:
        df['artist'] = df['file_link'].apply(lambda x: Path(x).parts[-2].replace('-', ' '))
    
    if 'title' not in df.columns and 'wikiart_caption' in df.columns:
        df['title'] = df['wikiart_caption'].apply(lambda caption: 
            caption.split(', by ')[0].strip() if isinstance(caption, str) and ', by ' in caption else "Unknown")
    
    # Apply optional limit
    if limit is not None and limit < len(df):
        print(f"Limiting test set to {limit} samples (from {len(df)} total)")
        df = df.sample(limit, random_state=42).reset_index(drop=True)
    
    print(f"Loaded {len(df)} test samples")
    return df


def evaluate_system(test_df: pd.DataFrame, save_dir: str, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
    """
    Evaluate the artwork recognition system on a test dataset.
    
    Args:
        test_df: DataFrame with test data
        save_dir: Directory where the system is saved
        k_values: List of k values to consider for top-k accuracy
        
    Returns:
        Dictionary with evaluation results
    """
    # Import required modules
    import image_processing
    import feature_extraction
    import index_management
    import system_management
    
    # Check if system exists
    if not system_management.system_exists(save_dir):
        raise FileNotFoundError(f"No artwork recognition system found at {save_dir}")
    
    # Load the system
    print("Loading the artwork recognition system...")
    index, embeddings, metadata = system_management.load_system(save_dir)
    feature_extractor = feature_extraction.create_feature_extractor()
    
    # Initialize metrics
    total_samples = len(test_df)
    results = {
        'total_samples': total_samples,
        'successful_queries': 0,
        'failed_queries': 0,
        'top_k_accuracy': {k: 0 for k in k_values},
        'avg_query_time': 0,
        'samples': []
    }
    
    total_query_time = 0
    
    # Build a lookup for faster matching
    metadata_lookup = {}
    for item in metadata:
        artist = item.get('artist', '').lower()
        if artist and artist not in metadata_lookup:
            metadata_lookup[artist] = []
        if artist:
            metadata_lookup[artist].append(item)
    
    # Process each test sample
    print(f"Evaluating {total_samples} test samples...")
    for idx, row in tqdm(test_df.iterrows(), total=total_samples):
        sample_result = {
            'test_url': row['file_link'],
            'test_artist': row.get('artist', 'Unknown'),
            'test_title': row.get('title', 'Unknown'),
            'matches': [],
            'success': False,
            'rank': None,
            'query_time': 0
        }
        
        try:
            # Load and process the test image
            test_image = image_processing.load_image(row['file_link'])
            if test_image is None:
                results['failed_queries'] += 1
                sample_result['error'] = 'Failed to load image'
                results['samples'].append(sample_result)
                continue
            
            # Process the image
            img_tensor = image_processing.process_image(test_image)
            if img_tensor is None:
                results['failed_queries'] += 1
                sample_result['error'] = 'Failed to process image'
                results['samples'].append(sample_result)
                continue
            
            # Time the query process
            start_time = time.time()
            
            # Extract features
            query_vector = feature_extractor.extract_single_feature(img_tensor)
            
            # Search for matches
            max_k = max(k_values)
            search_results = index_management.search_artwork(query_vector, index, metadata, k=max_k)
            
            # Record query time
            query_time = time.time() - start_time
            total_query_time += query_time
            sample_result['query_time'] = query_time
            
            # Process results
            results['successful_queries'] += 1
            
            # Check if any of the top results match the test artist
            test_artist = row.get('artist', '').lower()
            found_match = False
            match_rank = None
            
            for rank, result in enumerate(search_results):
                result_meta = result['metadata']
                result_artist = result_meta.get('artist', '').lower()
                
                # Add to sample results
                sample_result['matches'].append({
                    'rank': rank + 1,
                    'artist': result_meta.get('artist', 'Unknown'),
                    'title': result_meta.get('title', 'Unknown'),
                    'similarity': result.get('similarity', 0),
                    'matches_test': result_artist == test_artist
                })
                
                # Check for artist match
                if result_artist == test_artist and not found_match:
                    found_match = True
                    match_rank = rank + 1
                    
                    # Update top-k accuracy
                    for k in k_values:
                        if rank < k:
                            results['top_k_accuracy'][k] += 1
            
            sample_result['success'] = found_match
            sample_result['rank'] = match_rank
            
        except Exception as e:
            results['failed_queries'] += 1
            sample_result['error'] = str(e)
        
        results['samples'].append(sample_result)
    
    # Calculate final metrics
    successful_queries = results['successful_queries']
    if successful_queries > 0:
        # Calculate average query time
        results['avg_query_time'] = total_query_time / successful_queries
        
        # Calculate top-k accuracy percentages
        for k in k_values:
            if successful_queries > 0:
                results['top_k_accuracy'][k] = {
                    'count': results['top_k_accuracy'][k],
                    'percentage': (results['top_k_accuracy'][k] / successful_queries) * 100
                }
    
    return results


def plot_evaluation_results(results: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plot evaluation results.
    
    Args:
        results: Evaluation results dictionary
        save_path: Optional path to save the plots
    """
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Top-K Accuracy
    plt.subplot(2, 2, 1)
    k_values = []
    accuracy = []
    
    for k, data in results['top_k_accuracy'].items():
        if isinstance(data, dict):
            k_values.append(k)
            accuracy.append(data['percentage'])
    
    plt.bar(k_values, accuracy)
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')
    plt.title('Top-K Accuracy')
    plt.xticks(k_values)
    plt.ylim(0, 100)
    
    for i, v in enumerate(accuracy):
        plt.text(k_values[i], v + 2, f"{v:.1f}%", ha='center')
    
    # Plot 2: Success vs Failure
    plt.subplot(2, 2, 2)
    success_counts = [
        results['successful_queries'],
        results['failed_queries']
    ]
    labels = ['Successful', 'Failed']
    
    plt.pie(success_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
    plt.axis('equal')
    plt.title('Query Success Rate')
    
    # Plot 3: Rank Distribution
    plt.subplot(2, 2, 3)
    
    ranks = [sample['rank'] for sample in results['samples'] if sample.get('rank') is not None]
    max_rank = max(ranks) if ranks else 10
    plt.hist(ranks, bins=range(1, max_rank + 2), alpha=0.7, color='#2196F3')
    plt.xlabel('Rank of Correct Match')
    plt.ylabel('Frequency')
    plt.title('Distribution of Correct Match Ranks')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 4: Query Time
    plt.subplot(2, 2, 4)
    
    query_times = [sample['query_time'] for sample in results['samples'] if 'query_time' in sample]
    plt.hist(query_times, bins=20, alpha=0.7, color='#FF9800')
    plt.xlabel('Query Time (seconds)')
    plt.ylabel('Frequency')
    plt.title(f'Query Time Distribution (Avg: {results["avg_query_time"]:.3f}s)')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        print(f"Evaluation plots saved to {save_path}")
    else:
        plt.show()


def save_evaluation_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save the results
    """
    # Convert NumPy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Process the results dictionary
    serializable_results = json.loads(
        json.dumps(results, default=convert_to_serializable)
    )
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Evaluation results saved to {output_path}")


def create_error_analysis(results: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Create a human-readable error analysis report.
    
    Args:
        results: Evaluation results dictionary
        output_path: Optional path to save the report
        
    Returns:
        Report as a string
    """
    # Create report text
    report = "# Artwork Recognition System Evaluation\n\n"
    
    # Overall statistics
    report += "## Overall Statistics\n\n"
    report += f"- Total test samples: {results['total_samples']}\n"
    report += f"- Successful queries: {results['successful_queries']} ({results['successful_queries']/results['total_samples']*100:.1f}%)\n"
    report += f"- Failed queries: {results['failed_queries']} ({results['failed_queries']/results['total_samples']*100:.1f}%)\n"
    report += f"- Average query time: {results['avg_query_time']:.3f} seconds\n\n"
    
    # Top-K accuracy
    report += "## Top-K Accuracy\n\n"
    for k, data in results['top_k_accuracy'].items():
        if isinstance(data, dict):
            report += f"- Top-{k}: {data['count']} correct matches ({data['percentage']:.1f}%)\n"
    report += "\n"
    
    # Error analysis
    report += "## Error Analysis\n\n"
    
    # Count common error types
    error_types = {}
    for sample in results['samples']:
        if 'error' in sample:
            error_type = sample['error']
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    if error_types:
        report += "### Common Error Types\n\n"
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            report += f"- {error_type}: {count} occurrences\n"
        report += "\n"
    
    # Misidentified artists analysis
    report += "### Misidentified Artists\n\n"
    artist_confusion = {}
    
    for sample in results['samples']:
        if not sample.get('success', False) and 'error' not in sample and sample.get('matches'):
            test_artist = sample.get('test_artist', 'Unknown')
            top_match_artist = sample['matches'][0]['artist']
            
            key = f"{test_artist} → {top_match_artist}"
            artist_confusion[key] = artist_confusion.get(key, 0) + 1
    
    if artist_confusion:
        report += "Most common artist misidentifications:\n\n"
        for confusion, count in sorted(artist_confusion.items(), key=lambda x: x[1], reverse=True)[:10]:
            report += f"- {confusion}: {count} occurrences\n"
        report += "\n"
    
    # Save report if requested
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Error analysis report saved to {output_path}")
    
    return report


def run_evaluation(
    test_csv_path: str,
    save_dir: str,
    output_dir: str = './evaluation_results',
    test_limit: Optional[int] = None,
    k_values: List[int] = [1, 3, 5, 10]
) -> None:
    """
    Run a complete evaluation and save the results.
    
    Args:
        test_csv_path: Path to the test CSV file
        save_dir: Directory where the system is saved
        output_dir: Directory to save evaluation results
        test_limit: Optional limit on number of test samples
        k_values: List of k values to consider for top-k accuracy
    """
    print("=" * 70)
    print(f"Starting evaluation of artwork recognition system")
    print(f"System location: {save_dir}")
    print(f"Test data: {test_csv_path}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test dataset
    test_df = load_test_dataset(test_csv_path, limit=test_limit)
    
    # Run evaluation
    results = evaluate_system(test_df, save_dir, k_values=k_values)
    
    # Generate timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save results
    json_path = os.path.join(output_dir, f"eval_results_{timestamp}.json")
    save_evaluation_results(results, json_path)
    
    # Create plots
    plot_path = os.path.join(output_dir, f"eval_plots_{timestamp}.png")
    plot_evaluation_results(results, save_path=plot_path)
    
    # Create error analysis
    report_path = os.path.join(output_dir, f"eval_report_{timestamp}.md")
    create_error_analysis(results, output_path=report_path)
    
    print("\nEvaluation complete!")
    print(f"Results saved to {output_dir}")
    
    # Print summary
    print("\nSummary:")
    print(f"- Total test samples: {results['total_samples']}")
    print(f"- Successful queries: {results['successful_queries']} ({results['successful_queries']/results['total_samples']*100:.1f}%)")
    print(f"- Failed queries: {results['failed_queries']} ({results['failed_queries']/results['total_samples']*100:.1f}%)")
    print(f"- Average query time: {results['avg_query_time']:.3f} seconds")
    print("\nTop-K Accuracy:")
    for k, data in results['top_k_accuracy'].items():
        if isinstance(data, dict):
            print(f"- Top-{k}: {data['percentage']:.1f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate the artwork recognition system')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to the test CSV file')
    parser.add_argument('--save_dir', type=str, default='/content/drive/MyDrive/artwork_recognition', 
                        help='Directory where the system is saved')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', 
                        help='Directory to save evaluation results')
    parser.add_argument('--limit', type=int, default=None, 
                        help='Limit the number of test samples')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 3, 5, 10],
                        help='List of k values for top-k accuracy')
    
    args = parser.parse_args()
    
    run_evaluation(
        args.test_csv,
        args.save_dir,
        args.output_dir,
        args.limit,
        args.k_values
    )