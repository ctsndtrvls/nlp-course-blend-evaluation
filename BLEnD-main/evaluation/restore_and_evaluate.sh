#!/bin/bash
# Restore English results and add Spanish results

cd "$(dirname "$0")"

echo "Step 1: Re-evaluating English results..."
python3 evaluate_semeval.py \
    --predictions_file ../model_inference_results/Qwen2.5-3B-Instruct-US_English_inst-4_result.csv \
    --annotations_dir ../data/annotations \
    --results_dir results \
    --id_col ID \
    --response_col response

echo ""
echo "Step 2: Adding Spanish results..."
python3 evaluate_semeval.py \
    --predictions_file ../model_inference_results/mt5-small-Mexico_Spanish_inst-4_result.csv \
    --annotations_dir ../data/annotations \
    --results_dir results \
    --id_col ID \
    --response_col response

echo ""
echo "Done! Check results/results.json to see both English and Spanish results."
