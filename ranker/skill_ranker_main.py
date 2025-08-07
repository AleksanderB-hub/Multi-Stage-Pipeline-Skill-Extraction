#!/usr/bin/env python3
import os
import sys
import json
import logging
from datetime import datetime

from .config_2 import create_parser
from .skill_ranker import (
    SkillExtractorTrainer, 
    SkillExtractorInference,
    set_seed,
    create_train_val_split
)


def setup_logging(args):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout) if args.verbose else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_data(args, logger):
    """Load training, validation, and test data"""
    data = {}
    
    # Load training data
    if args.train_file:
        logger.info(f"Loading training data from: {args.train_file}")
        with open(args.train_file, 'r') as f:
            train_data_full = json.load(f)
        
        # Create train/val split if no validation file provided
        if args.val_file:
            logger.info(f"Loading validation data from: {args.val_file}")
            with open(args.val_file, 'r') as f:
                val_data = json.load(f)
            data['train'] = train_data_full
            data['val'] = val_data
        else:
            logger.info(f"Creating train/val split with ratio: {args.val_size}")
            train_data, val_data = create_train_val_split(
                train_data_full, 
                val_size=args.val_size, 
                seed=args.seed
            )
            data['train'] = train_data
            data['val'] = val_data
        
        logger.info(f"Training examples: {len(data['train'])}")
        logger.info(f"Validation examples: {len(data['val'])}")
    
    # Load test data
    if args.test_file:
        logger.info(f"Loading test data from: {args.test_file}")
        with open(args.test_file, 'r') as f:
            test_data = json.load(f)
        data['test'] = test_data
        logger.info(f"Test examples: {len(test_data)}")
    
    return data


def train_model(args, data, logger):
    """Train the cross-encoder model"""
    logger.info("Starting model training...")

    
    # Create model save path
    model_name = f"skill_relevance_model_seed_{args.seed}_v2"
    model_save_path = os.path.join(args.model_save_dir, model_name)
    os.makedirs(args.model_save_dir, exist_ok=True)
    
    logger.info(f"Model will be saved to: {model_save_path}")
    
    # Initialize trainer
    trainer = SkillExtractorTrainer(args)
    
    # Train model
    model, optimal_threshold = trainer.train(
        train_data=data['train'],
        val_data=data.get('val'),
        test_data=data.get('test'),
        model_save_path=model_save_path
    )
    
    # Save training results
    os.makedirs(args.output_dir, exist_ok=True)
    training_results = {
        'model_path': model_save_path,
        'optimal_threshold': optimal_threshold,
        'training_args': vars(args),
        'training_date': datetime.now().isoformat()
    }
    
    results_file = os.path.join(args.output_dir, 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump(training_results, f, indent=2)
    
    logger.info(f"Training complete. Results saved to: {results_file}")
    logger.info(f"Optimal threshold: {optimal_threshold}")
    
    return model_save_path, optimal_threshold


def test_model(args, data, logger):
    """Test the cross-encoder model"""
    logger.info("Starting model testing...")
    
    if 'test' not in data or not data['test']:
        raise ValueError("Test data is required for testing mode")
    
    # Initialize inference
    inference = SkillExtractorInference(args)
    
    # Use the provided threshold (should be optimal threshold from training)
    threshold = args.threshold
    logger.info(f"Using threshold: {threshold}")
    
    # Test model
    evaluation_results = inference.evaluate(
        test_data=data['test'],
        threshold=threshold
    )
    
    # Add metadata
    evaluation_results['metadata'] = {
        'model_path': args.model_path,
        'threshold': threshold,  
        'test_file': args.test_file,
        'evaluation_date': datetime.now().isoformat(),
        'test_args': vars(args)
    }
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.save_evaluation:
        eval_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        logger.info(f"Evaluation results saved to: {eval_file}")
    
    if args.save_predictions:
        pred_file = os.path.join(args.output_dir, 'predictions.json')
        predictions = {
            'predictions': evaluation_results['detailed_results'],
            'metadata': evaluation_results['metadata']
        }
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Predictions saved to: {pred_file}")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Test examples: {len(data['test'])}")
    logger.info(f"Threshold: {threshold:.3f}")
    logger.info(f"Precision: {evaluation_results['precision']:.4f}")
    logger.info(f"Recall: {evaluation_results['recall']:.4f}")
    logger.info(f"F1 Score: {evaluation_results['f1']:.4f}")
    logger.info(f"True Positives: {evaluation_results['total_true_positives']}")
    logger.info(f"False Positives: {evaluation_results['total_false_positives']}")
    logger.info(f"False Negatives: {evaluation_results['total_false_negatives']}")
    logger.info("="*50)
    
    # Show example predictions
    if args.verbose:
        logger.info("\nExample Predictions:")
        for i, result in enumerate(evaluation_results['detailed_results'][:3]):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"Sentence: {result['sentence'][:100]}...")
            logger.info(f"True labels: {result['true_labels']}")
            logger.info(f"Predicted labels: {result['predicted_labels']}")
            if result['false_positives']:
                logger.info(f"False positives: {result['false_positives']}")
            if result['false_negatives']:
                logger.info(f"False negatives: {result['false_negatives']}")
    
    return evaluation_results


def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logging(args)
    
    # Validate arguments
    if args.mode in ['train'] and not args.train_file:
        parser.error("--train_file is required for training modes")
    
    if args.mode == 'test' and not args.model_path:
        parser.error("--model_path is required for testing mode")
    
    if args.mode == 'test' and not args.test_file:
        parser.error("--test_file is required for testing mode")
    
    try:
        # Load data
        data = load_data(args, logger)
        
        # Execute based on mode
        if args.mode == 'train':
            model_path, optimal_threshold = train_model(args, data, logger)
            logger.info(f"Training completed. Model saved to: {model_path}")
            
            # Update args for testing
            args.model_path = model_path
            args.threshold = optimal_threshold
            
            # Test the trained model
            logger.info("Starting testing phase...")
            evaluation_results = test_model(args, data, logger)
            logger.info("Training and testing completed successfully!")
            
        elif args.mode == 'test':
            evaluation_results = test_model(args, data, logger)
            logger.info("Testing completed successfully!")
    
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()