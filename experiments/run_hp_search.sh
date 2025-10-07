#!/bin/bash
# Helper script for running hyperparameter searches
# Usage: ./run_hp_search.sh <search_type> <input_path>

set -e

# Default values
SEARCH_TYPE=${1:-"focused"}
INPUT_PATH=${2:-""}
CONFIG_PATH="maveric_config.yaml"
OUTPUT_DIR="./hp_search_results"

# Check if input path is provided
if [ -z "$INPUT_PATH" ]; then
    echo "❌ Error: Input path not provided"
    echo "Usage: ./run_hp_search.sh <search_type> <input_path>"
    echo ""
    echo "Search types:"
    echo "  focused        - Search around optimal regularization_weight=0.5"
    echo "  regularization - Fine-grained regularization weight search"
    echo "  learning_rate  - Learning rate optimization"
    echo "  broad          - Broad multi-dimensional search"
    echo ""
    echo "Example:"
    echo "  ./run_hp_search.sh focused /path/to/cifar10_training_data/"
    exit 1
fi

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR}/${SEARCH_TYPE}_${TIMESTAMP}"

echo "🔍 Starting ${SEARCH_TYPE} hyperparameter search"
echo "📁 Input: ${INPUT_PATH}"
echo "📋 Config: ${CONFIG_PATH}"
echo "📊 Output: ${OUTPUT_DIR}"
echo ""

# Run hyperparameter search
python 05_hyperparameter_search.py \
    --input "${INPUT_PATH}" \
    --config "${CONFIG_PATH}" \
    --output "${OUTPUT_DIR}" \
    --search-type "${SEARCH_TYPE}" \
    --method grid

echo ""
echo "✅ Search complete! Results saved to: ${OUTPUT_DIR}"
echo "📊 Check ${OUTPUT_DIR}/search_summary.json for detailed results"
