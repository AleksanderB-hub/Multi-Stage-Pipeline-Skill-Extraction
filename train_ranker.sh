export CUDA_VISIBLE_DEVICES=0
# Data paths 
TRAIN_FILE="./Datasets/stage_2/full_train_100"
VAL_FILE=""  # Leave empty to auto-split from training data
TEST_FILE="./Datasets/stage_2/test_data/test_full_100"

# Model configuration
MODEL_NAME="cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH=128
DEVICE="cuda"  # "auto", "cuda", or "cpu"

# Training parameters
BATCH_SIZE=64
EPOCHS=5
LEARNING_RATE=2e-5
SEED=42
VAL_SIZE=0.2  # Only used if VAL_FILE is empty

# Focal Loss parameters (for handling class imbalance)
FOCAL_ALPHA=0.8
FOCAL_GAMMA=3.0

# Batch sampling
USE_BALANCED_BATCHES="--use_balanced_batches"  # Remove to disable
POSITIVE_RATIO=0.3

# Data augmentation
AUGMENTATION_PROB=0.2
APPLY_AUGMENTATION="--apply_augmentation"  # Remove to disable

# Training control
VALIDATION_STEPS=8000 #This needs to be adjusted per batch and training size to avoid too rare/often validation. 
GRADIENT_CLIP_NORM=1.0
WARMUP_RATIO=0.1

# Output paths
OUTPUT_DIR="./outputs/ranker_stage_2"
MODEL_SAVE_DIR="./checkpoints/ranker_stage_2"

# Inference settings
FIND_OPTIMAL_THRESHOLD="--find_optimal_threshold"  # Remove to disable
THRESHOLD_METRIC="f1"

# Logging
VERBOSE="--verbose"  # Remove to disable
LOG_LEVEL="INFO"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$MODEL_SAVE_DIR"

python -m ranker.skill_ranker_main \
  --mode train \
  --train_file "$TRAIN_FILE" \
  ${VAL_FILE:+--val_file "$VAL_FILE"} \
  ${TEST_FILE:+--test_file "$TEST_FILE"} \
  --val_size $VAL_SIZE \
  --model_name "$MODEL_NAME" \
  --max_length $MAX_LENGTH \
  --device $DEVICE \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --seed $SEED \
  --focal_alpha $FOCAL_ALPHA \
  --focal_gamma $FOCAL_GAMMA \
  $USE_BALANCED_BATCHES \
  --positive_ratio $POSITIVE_RATIO \
  --augmentation_prob $AUGMENTATION_PROB \
  $APPLY_AUGMENTATION \
  --validation_steps $VALIDATION_STEPS \
  --gradient_clip_norm $GRADIENT_CLIP_NORM \
  --warmup_ratio $WARMUP_RATIO \
  --output_dir "$OUTPUT_DIR" \
  --model_save_dir "$MODEL_SAVE_DIR" \
  $FIND_OPTIMAL_THRESHOLD \
  --threshold_metric $THRESHOLD_METRIC \
  $VERBOSE \
  --log_level $LOG_LEVEL