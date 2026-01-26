#!/bin/bash
# Test mt5-small model on Spanish (Mexico) with more questions

cd "$(dirname "$0")"

echo "Testing google/mt5-small on Spanish (Mexico)..."
echo "This will test on 100 questions (you can increase with --max_questions)"
echo ""

python3 get_model_responses_simple.py \
    --model mt5-small \
    --country Mexico \
    --language Spanish \
    --prompt_no inst-4 \
    --max_questions 100

echo ""
echo "Done! Results saved to: model_inference_results/mt5-small-Mexico_Spanish_inst-4_result.csv"
echo ""
echo "To evaluate the results, run:"
echo "cd evaluation && python3 evaluate_semeval.py \\"
echo "    --predictions_file ../model_inference_results/mt5-small-Mexico_Spanish_inst-4_result.csv \\"
echo "    --annotations_dir ../data/annotations \\"
echo "    --results_dir results \\"
echo "    --id_col ID \\"
echo "    --response_col response"
