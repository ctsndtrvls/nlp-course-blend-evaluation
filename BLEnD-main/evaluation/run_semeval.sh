#!/bin/bash
# SemEval 2026 Task 7, Track 1: Short Answer Questions (SAQ) Evaluation Script
# For CodaBench submission

# Set default paths (CodaBench typically provides these via environment variables)
PREDICTIONS_FILE="${1:-predictions.csv}"
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-../data/annotations}"
RESULTS_DIR="${RESULTS_DIR:-results}"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run evaluation
python evaluate_semeval.py \
    --predictions_file "$PREDICTIONS_FILE" \
    --annotations_dir "$ANNOTATIONS_DIR" \
    --results_dir "$RESULTS_DIR" \
    --id_col ID \
    --response_col response \
    --annotations_key annotations \
    --annotation_template "{country}_data.json"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results saved to: $RESULTS_DIR"
    echo "Summary:"
    cat "$RESULTS_DIR/evaluation_summary.csv" 2>/dev/null || echo "Summary file not found"
else
    echo "Evaluation failed!"
    exit 1
fi
