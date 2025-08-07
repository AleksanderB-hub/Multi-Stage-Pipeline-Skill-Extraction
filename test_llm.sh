API_KEY="" # Insert your Open AI api key here
MODEL="gpt-4o-mini"
TEST_DATA="./Datasets/stage_2/test_data/test_house_20"
REFERENCE_DATA="./Datasets/stage_2/full_train_20"
OUTPUT_DIR="./outputs/stage_2/llm_baseline"

python -m llm_ranker.llm_main \
  --api_key "$API_KEY" \
  --model "$MODEL" \
  --test_file "$TEST_DATA" \
  --reference_file "$REFERENCE_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_type house \
  --use_demo