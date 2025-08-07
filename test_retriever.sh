LABELS_TO_EXCLUDE_PATH="./Datasets/Stage_1/labels_to_exclude_zero_shot.json"
TEST_DATA_PATH="./Datasets/Stage_1/test_data/skape_test.json" # options: [house_test, tech_test, techwolf_test, skape_test]
MAPPING_PATH="./Datasets/Stage_1/mapping_esco.json"

python -m retriever.main \
  --test_only \
  --test_data_path $TEST_DATA_PATH \
  --mapping_path $MAPPING_PATH \
  --model_name "sentence-transformers/all-mpnet-base-v2" \
  --top_k 5 \
  --dataset skape \
  --fp16 \
  --resume_from_checkpoint "./checkpoints/retriever_stage_1/best_model" 

  # If you want to test zero-shot performance you must provide:
  # --zero_shot \
  # --excluded_path $LABELS_TO_EXCLUDE_PATH

  # If you want to test the pure baseline performance (no training)
  # --pure_baseline \
