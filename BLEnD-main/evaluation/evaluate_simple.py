#!/usr/bin/env python3
"""
Simple evaluation script for BLEnD results
Evaluates model responses using Soft Exact Match (SEM) methodology
"""

import os
import sys
import json
import argparse
import pandas as pd

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from evaluation.exact_match import soft_exact_match

def evaluate_simple(predictions_file, annotations_file, results_dir='results'):
    """
    Simple evaluation function
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Load predictions
    print(f"Loading predictions from {predictions_file}")
    predictions_df = pd.read_csv(predictions_file, encoding='utf-8')
    
    # Load annotations
    print(f"Loading annotations from {annotations_file}")
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Determine country from filename
    if 'US' in annotations_file or 'US' in predictions_file:
        country = 'US'
        language = 'English'
    elif 'Algeria' in annotations_file:
        country = 'Algeria'
        language = 'Arabic'
    else:
        country = 'US'  # Default
        language = 'English'
    
    print(f"\nEvaluating for {country} ({language})")
    print(f"Total predictions: {len(predictions_df)}")
    
    # Run evaluation
    sem_b, sem_w, scored_df = soft_exact_match(
        country=country,
        language=language,
        annotation_dict=annotations,
        response_df=predictions_df,
        id_col='ID',
        r_col='response',
        annotations_key='annotations'
    )
    
    # Save results
    results_file = os.path.join(results_dir, f'{country}_detailed_results.csv')
    scored_df.to_csv(results_file, index=False, encoding='utf-8')
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Evaluation Results for {country}")
    print(f"{'='*50}")
    print(f"SEM-B (Binary): {sem_b:.2f}%")
    print(f"SEM-W (Weighted): {sem_w:.2f}%")
    print(f"Total Questions: {len(scored_df)}")
    print(f"Correct Answers: {scored_df['binary_score'].sum()}")
    print(f"{'='*50}")
    
    # Show some examples
    print("\nSample Results:")
    print("-" * 50)
    for idx, row in scored_df.head(10).iterrows():
        status = "✓" if row['binary_score'] == 1 else "✗"
        print(f"{status} {row['ID']}: {row.get('response', 'N/A')[:50]}...")
    
    return sem_b, sem_w, scored_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple evaluation for BLEnD results')
    parser.add_argument('--predictions_file', type=str, required=True,
                       help='Path to CSV file with predictions')
    parser.add_argument('--annotations_file', type=str, required=True,
                       help='Path to JSON file with annotations')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    evaluate_simple(args.predictions_file, args.annotations_file, args.results_dir)
