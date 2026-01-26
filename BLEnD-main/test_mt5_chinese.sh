#!/bin/bash
# Test mt5-small model on Chinese (China) with questions

cd "$(dirname "$0")"

echo "Testing google/mt5-small on Chinese (China)..."
echo "This will test on 100 questions (you can increase with --max_questions)"
echo ""

# Step 1: Generate model responses
python3 get_model_responses_simple.py \
    --model mt5-small \
    --country China \
    --language Chinese \
    --prompt_no inst-4 \
    --max_questions 100

echo ""
echo "Step 1 complete! Results saved to: model_inference_results/mt5-small-China_Chinese_inst-4_result.csv"
echo ""

# Step 2: Evaluate the results
echo "Step 2: Running evaluation..."
cd evaluation

python3 evaluate_semeval.py \
    --predictions_file ../model_inference_results/mt5-small-China_Chinese_inst-4_result.csv \
    --annotations_dir ../data/annotations \
    --results_dir results \
    --id_col ID \
    --response_col response

echo ""
echo "Done! Check results/results.json to see all results (English, Spanish, and Chinese)."
