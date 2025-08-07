# Model and test data
MODEL_PATH="./checkpoints/stage_2/skill_relevance_model_seed_42_v2"
TEST_FILE="./Data/stage_2/test_data/test_tech_20"

# Note: Optimal threshold should be determined during training
# The threshold below should be the one found during training
# Check ./outputs/training_results.json for the optimal_threshold value
# Classification threshold
THRESHOLD=0.2 

# Output settings
OUTPUT_DIR="./outputs/stage_2"
SAVE_PREDICTIONS="--save_predictions"  # Remove to disable
SAVE_EVALUATION="--save_evaluation"  # Remove to disable

# Logging
VERBOSE="--verbose"  # Remove to disable
LOG_LEVEL="INFO"

# Create directories
mkdir -p "$OUTPUT_DIR"

python -m ranker.skill_ranker_main \
  --mode test \
  --model_path "$MODEL_PATH" \
  --test_file "$TEST_FILE" \
  --threshold $THRESHOLD \
  --output_dir "$OUTPUT_DIR" \
  $SAVE_PREDICTIONS \
  $SAVE_EVALUATION \
  $VERBOSE \
  --log_level $LOG_LEVEL