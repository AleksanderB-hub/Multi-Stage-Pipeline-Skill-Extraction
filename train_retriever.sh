# set the parameters
MODEL_NAME="sentence-transformers/all-mpnet-base-v2"

DATA_PATH="./Datasets/stage_1/full_train.json" 
VAL_DATA_PATH="./Datasets/stage_1/skape_val.json"
TEST_DATA_PATH="./Datasets/stage_1/test_data/skape_test.json" # options: [house_test, tech_test, techwolf_test, skape_test]
MAPPING_PATH="./Datasets/stage_1/mapping_esco.json" #List of all ESCO skills
LABELS_TO_EXCLUDE_PATH="./Datasets/stage_1/labels_to_exclude_zero_shot.json" # in zero-shot mode specify the model to the excluded skills

OUTPUT_DIR="./outputs/retriever_stage_1"
CHECKPOINT_DIR="./checkpoints/retriever_stage_1"

DATA_PATH_PT="./Datasets/stage_1/full_pretrain.json" # the path to data used for pre_training   esco_pre_train.json
OUTPUT_DIR_PT="./outputs/retriever_pre_train/retriever_pre_train_stage_1"
CHECKPOINT_DIR_PT="./checkpoints/retriever_pre_train/retriever_pre_train_stage_1"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$OUTPUT_DIR_PT"
mkdir -p "$CHECKPOINT_DIR_PT"

python -m retriever.main \
  --model_name $MODEL_NAME \
  --run_name test_v1 \
  --data_path $DATA_PATH \
  --val_data_path $VAL_DATA_PATH \
  --test_data_path $TEST_DATA_PATH \
  --mapping_path $MAPPING_PATH \
  --output_dir $OUTPUT_DIR \
  --checkpoint_dir $CHECKPOINT_DIR \
  --fp16 \
  --hard_negative_strategy "in_batch" \
  --dataset skape \
  --seed 42 \
  --data_path_pt $DATA_PATH_PT \
  --output_dir_pt $OUTPUT_DIR_PT \
  --checkpoint_dir_pt $CHECKPOINT_DIR_PT \
  --pre_train \
  --use_pre_trained \
  --enable_augmentation \

  # for zero-shot you will have to provide the following. Just make sure that the adequate training data is provided for both phases:
  # --excluded_path $LABELS_TO_EXCLUDE_PATH \
  # --zero_shot

  # if you already pre-trained model once you can simply delete "--pre_train" from the list above, assuming the path to the pre-trained model is provided, the main training phase will use the model if "--use_pre_trained" is present. 



