import argparse
import json
import logging
import os
from .llm_run import run_llm_extraction, evaluate_skill_extraction_v2

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM Skill Extraction with optional demonstration examples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name (e.g. gpt-4o-mini, gpt-4.0, gpt-3.5-turbo)")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test data JSON file")
    parser.add_argument("--reference_file", type=str, default=None, help="Path to reference data for demonstrations")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save predictions")
    parser.add_argument("--dataset_type", type=str, required=True, help="Sets the name of the dataset to save as")
    parser.add_argument("--use_demo", action="store_true", help="Whether to use demonstration examples")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    with open(args.test_file, 'r') as f:
        test_data = json.load(f)

    reference_data = None
    if args.use_demo and args.reference_file:
        with open(args.reference_file, 'r') as f:
            reference_data = json.load(f)

    results = run_llm_extraction(
        api_key=args.api_key,
        test_data=test_data,
        reference_data=reference_data,
        model=args.model,
        use_demo=args.use_demo
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"llm_predictions_{args.dataset_type}.json")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved predictions to {output_path}")

    # Evaluate and log metrics
    metrics = evaluate_skill_extraction_v2(results)
    logging.info(f"Precision: {metrics['precision']:.4f}")
    logging.info(f"Recall: {metrics['recall']:.4f}")
    logging.info(f"F1 Score: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()