#!/usr/bin/env python3
"""
Simplified script for getting model responses on BLEnD dataset questions.
Works with local files without requiring Google Sheets.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add current directory to path for importing utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *

def load_prompts_from_csv(country, prompt_dir=None):
    """Load prompts from local CSV file"""
    if prompt_dir is None:
        prompt_dir = os.path.join(os.path.dirname(__file__), 'data', 'prompts')
    
    prompt_file = os.path.join(prompt_dir, f'{country}_prompts.csv')
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    return pd.read_csv(prompt_file, encoding='utf-8')

def make_prompt_from_csv(question, prompt_no, language, prompt_sheet):
    """Create prompt from CSV file"""
    prompt_row = prompt_sheet[prompt_sheet['id'] == prompt_no]
    if len(prompt_row) == 0:
        raise ValueError(f"Prompt {prompt_no} not found in prompt sheet")
    
    if language == 'English':
        prompt_template = prompt_row['English'].values[0]
    else:
        prompt_template = prompt_row['Translation'].values[0]
    
    return prompt_template.replace('{q}', question)

def get_questions_from_csv(country, question_dir=None):
    """Load questions from CSV file"""
    if question_dir is None:
        question_dir = os.path.join(os.path.dirname(__file__), 'data', 'questions')
    
    # Try different possible filenames
    possible_files = [
        os.path.join(question_dir, f'{country}_questions.csv'),
        os.path.join(question_dir, f'{country}_full_final_questions.csv'),
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            return pd.read_csv(file_path, encoding='utf-8')
    
    raise FileNotFoundError(f"Question file not found for {country}. Tried: {possible_files}")

def generate_responses_simple(
    model_name,
    country,
    language,
    prompt_no='inst-4',
    question_dir=None,
    prompt_dir=None,
    output_dir='./model_inference_results',
    temperature=0,
    top_p=1,
    gpt_azure=False,
    max_questions=None
):
    """
    Simplified function for getting model responses
    
    Args:
        model_name: Model name from MODEL_PATHS
        country: Country (e.g., 'US', 'Algeria')
        language: Language (e.g., 'English', 'Arabic')
        prompt_no: Prompt ID (e.g., 'inst-4', 'pers-3')
        question_dir: Directory with questions
        prompt_dir: Directory with prompts
        output_dir: Directory to save results
        temperature: Generation temperature
        top_p: Top-p parameter
        gpt_azure: Whether to use Azure OpenAI
        max_questions: Maximum number of questions (for testing)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load questions
    print(f"Loading questions for {country}...")
    questions_df = get_questions_from_csv(country, question_dir)
    
    # Limit number of questions for testing
    if max_questions:
        questions_df = questions_df.head(max_questions)
        print(f"Limited to {max_questions} questions for testing")
    
    # Determine question column
    if language == COUNTRY_LANG.get(country, 'English'):
        question_col = 'Translation'
    else:
        question_col = 'Question'
    
    id_col = 'ID'
    
    # Load prompts
    print(f"Loading prompts for {country}...")
    prompt_sheet = load_prompts_from_csv(country, prompt_dir)
    
    # Load model
    print(f"Loading model {model_name}...")
    model_path = MODEL_PATHS.get(model_name)
    if model_path is None:
        raise ValueError(f"Model {model_name} not found in MODEL_PATHS. Available models: {list(MODEL_PATHS.keys())}")
    
    tokenizer, model = get_tokenizer_model(model_name, model_path, '.cache')
    
    # Create output filename
    output_filename = os.path.join(
        output_dir,
        f"{model_name}-{country}_{language}_{prompt_no}_result.csv"
    )
    
    # Check if results already exist
    guid_list = set()
    if os.path.exists(output_filename):
        print(f"Found existing results file: {output_filename}")
        already = pd.read_csv(output_filename, encoding='utf-8')
        guid_list = set(already[id_col].values)
        print(f"Skipping {len(guid_list)} already processed questions")
    else:
        # Create CSV header
        write_csv_row([id_col, question_col, 'prompt', 'response', 'prompt_no'], output_filename)
    
    # Process questions
    print(f"Processing {len(questions_df)} questions...")
    from tqdm import tqdm
    
    for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc=f"Processing {model_name}"):
        question_id = row[id_col]
        question = row[question_col]
        
        # Skip already processed
        if question_id in guid_list:
            continue
        
        # Create prompt
        if prompt_no:
            prompt = make_prompt_from_csv(question, prompt_no, language, prompt_sheet)
        else:
            prompt = question
        
        # Get model response
        try:
            response = get_model_response(
                model_name=model_name,  # Use key from MODEL_PATHS, not path
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                temperature=temperature,
                top_p=top_p,
                gpt_azure=gpt_azure
            )
            
            # Save result
            write_csv_row([question_id, question, prompt, response, prompt_no], output_filename)
            
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            # Save error as response
            write_csv_row([question_id, question, prompt, f"ERROR: {str(e)}", prompt_no], output_filename)
            continue
    
    print(f"\nResults saved to: {output_filename}")
    return output_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Get model responses on BLEnD dataset questions'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., gpt-4-0125-preview, claude-3-sonnet-20240229)'
    )
    
    parser.add_argument(
        '--country',
        type=str,
        default='US',
        help='Country (e.g., US, Algeria, South_Korea)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default=None,
        help='Language (e.g., English, Arabic). If not specified, uses country language'
    )
    
    parser.add_argument(
        '--prompt_no',
        type=str,
        default='inst-4',
        help='Prompt ID (e.g., inst-4, pers-3)'
    )
    
    parser.add_argument(
        '--question_dir',
        type=str,
        default=None,
        help='Directory with questions (default: data/questions)'
    )
    
    parser.add_argument(
        '--prompt_dir',
        type=str,
        default=None,
        help='Directory with prompts (default: data/prompts)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./model_inference_results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0,
        help='Generation temperature'
    )
    
    parser.add_argument(
        '--top_p',
        type=float,
        default=1,
        help='Top-p parameter'
    )
    
    parser.add_argument(
        '--gpt_azure',
        action='store_true',
        help='Use Azure OpenAI'
    )
    
    parser.add_argument(
        '--max_questions',
        type=int,
        default=None,
        help='Maximum number of questions (for testing)'
    )
    
    args = parser.parse_args()
    
    # Determine language if not specified
    if args.language is None:
        args.language = COUNTRY_LANG.get(args.country, 'English')
    
    # Set default paths
    if args.question_dir is None:
        args.question_dir = os.path.join(os.path.dirname(__file__), 'data', 'questions')
    
    if args.prompt_dir is None:
        args.prompt_dir = os.path.join(os.path.dirname(__file__), 'data', 'prompts')
    
    # Run response generation
    generate_responses_simple(
        model_name=args.model,
        country=args.country,
        language=args.language,
        prompt_no=args.prompt_no,
        question_dir=args.question_dir,
        prompt_dir=args.prompt_dir,
        output_dir=args.output_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        gpt_azure=args.gpt_azure,
        max_questions=args.max_questions
    )
