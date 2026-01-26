"""
SemEval 2026 Task 7, Track 1: Short Answer Questions (SAQ) Evaluation Script
Adapted for CodaBench submission format

This script evaluates model responses using Soft Exact Match (SEM) methodology:
- SEM-B: Binary metric (correct/incorrect)
- SEM-W: Weighted metric (accounts for annotator vote counts)
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from evaluation_utils import *
from exact_match import *

def evaluate_semeval(
    predictions_file,
    annotations_dir,
    results_dir,
    id_col='ID',
    response_col='response',
    annotations_key='annotations',
    annotation_template='{country}_data.json'
):
    """
    Evaluate predictions for SemEval 2026 Task 7 format.
    
    Args:
        predictions_file: Path to CSV file with model predictions
        annotations_dir: Directory containing annotation JSON files
        results_dir: Directory to save evaluation results
        id_col: Column name for question IDs
        response_col: Column name for model responses
        annotations_key: Key for annotations in JSON files
        annotation_template: Template for annotation filenames
    """
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Load predictions
    print(f"Loading predictions from {predictions_file}")
    predictions_df = pd.read_csv(predictions_file, encoding='utf-8')
    
    # Determine country from predictions file or use default
    # Try to infer from filename first
    filename = os.path.basename(predictions_file)
    country = None
    language = None
    
    # Try to extract country and language from filename (e.g., "mt5-small-Mexico_Spanish_inst-4_result.csv")
    if 'Mexico' in filename:
        country = 'Mexico'
        language = 'Spanish'
    elif 'Spain' in filename:
        country = 'Spain'
        language = 'Spanish'
    elif 'US' in filename or 'United_States' in filename:
        country = 'US'
        language = 'English'
    elif 'UK' in filename:
        country = 'UK'
        language = 'English'
    else:
        # Try to infer from COUNTRY_LANG mapping
        from utils import COUNTRY_LANG
        for c, l in COUNTRY_LANG.items():
            if c in filename:
                country = c
                language = l
                break
    
    # Fallback: use country column or default
    if country is None:
        if 'country' in predictions_df.columns:
            countries = predictions_df['country'].unique()
            country = countries[0] if len(countries) > 0 else 'US'
        else:
            country = 'US'  # Default
    
    # Determine language from COUNTRY_LANG if not set
    if language is None:
        from utils import COUNTRY_LANG
        language = COUNTRY_LANG.get(country, 'English')
    
    print(f"\nProcessing country: {country}, language: {language}")
    
    # Filter predictions for this country
    if 'country' in predictions_df.columns:
        country_predictions = predictions_df[predictions_df['country'] == country]
    else:
        country_predictions = predictions_df
    
    # Load annotations
    annotation_file = annotation_template.replace('{country}', country.replace(' ', '_'))
    annotation_path = os.path.join(annotations_dir, annotation_file)
    
    if not os.path.exists(annotation_path):
        print(f"Warning: Annotation file not found: {annotation_path}")
        return None
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Load existing results if they exist
    results_json_file = os.path.join(results_dir, 'results.json')
    existing_results = []
    if os.path.exists(results_json_file):
        with open(results_json_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_results = existing_data.get('detailed_results', [])
            print(f"Found {len(existing_results)} existing result(s)")
    
    all_results = existing_results.copy()
        
    # Run evaluation
    sem_b, sem_w, scored_df = soft_exact_match(
        country=country,
        language=language,
        annotation_dict=annotations,
        response_df=country_predictions,
        id_col=id_col,
        r_col=response_col,
        annotations_key=annotations_key
    )
    
    # Save detailed results
    results_file = os.path.join(results_dir, f'{country}_{language}_detailed_results.csv')
    scored_df.to_csv(results_file, index=False, encoding='utf-8')
    
    # Check if this country/language combination already exists
    new_result = {
        'country': country,
        'language': language,
        'SEM-B': sem_b,
        'SEM-W': sem_w,
        'num_questions': len(country_predictions)
    }
    
    # Check if this exact combination already exists
    existing_idx = None
    for idx, r in enumerate(all_results):
        if r['country'] == country and r['language'] == language:
            existing_idx = idx
            break
    
    if existing_idx is not None:
        # Replace existing entry
        print(f"Replacing existing results for {country} ({language})")
        all_results[existing_idx] = new_result
    else:
        # Add new result (different country/language)
        print(f"Adding new results for {country} ({language})")
        all_results.append(new_result)
    
    print(f"SEM-B: {sem_b:.2f}%")
    print(f"SEM-W: {sem_w:.2f}%")
    
    # Save summary results
    summary_df = pd.DataFrame(all_results)
    summary_file = os.path.join(results_dir, 'evaluation_summary.csv')
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    
    # Save results in CodaBench format (JSON)
    # Calculate weighted average across all results
    total_questions = sum(r['num_questions'] for r in all_results)
    if total_questions > 0:
        overall_sem_b = sum(r['SEM-B'] * r['num_questions'] for r in all_results) / total_questions
        overall_sem_w = sum(r['SEM-W'] * r['num_questions'] for r in all_results) / total_questions
    else:
        overall_sem_b = 0
        overall_sem_w = 0
    
    results_json = {
        'SEM-B': overall_sem_b,
        'SEM-W': overall_sem_w,
        'detailed_results': all_results
    }
    
    results_json_file = os.path.join(results_dir, 'results.json')
    with open(results_json_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation complete. Results saved to {results_dir}")
    print(f"Overall SEM-B: {results_json['SEM-B']:.2f}%")
    print(f"Overall SEM-W: {results_json['SEM-W']:.2f}%")
    
    return results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate predictions for SemEval 2026 Task 7, Track 1 (SAQ)'
    )
    
    parser.add_argument(
        '--predictions_file',
        type=str,
        required=True,
        help='Path to CSV file with model predictions (must contain ID and response columns)'
    )
    
    parser.add_argument(
        '--annotations_dir',
        type=str,
        default='../data/annotations',
        help='Directory containing annotation JSON files'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--id_col',
        type=str,
        default='ID',
        help='Column name for question IDs in predictions file'
    )
    
    parser.add_argument(
        '--response_col',
        type=str,
        default='response',
        help='Column name for model responses in predictions file'
    )
    
    parser.add_argument(
        '--annotations_key',
        type=str,
        default='annotations',
        help='Key for annotations in JSON files'
    )
    
    parser.add_argument(
        '--annotation_template',
        type=str,
        default='{country}_data.json',
        help='Template for annotation filenames (use {country} placeholder)'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_semeval(
        predictions_file=args.predictions_file,
        annotations_dir=args.annotations_dir,
        results_dir=args.results_dir,
        id_col=args.id_col,
        response_col=args.response_col,
        annotations_key=args.annotations_key,
        annotation_template=args.annotation_template
    )
