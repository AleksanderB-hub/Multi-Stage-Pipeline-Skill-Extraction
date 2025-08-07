import argparse

def create_parser():
    """Create argument parser with organized groups"""
    parser = argparse.ArgumentParser(
        description="Cross-Encoder for Skill Extraction (Binary Ranking)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows defaults in help
    )
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'test', 'train_test'], 
                       default='train', help='Operation mode')
    
    # Data arguments
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--train_file', type=str, required=False,
                           help='Path to training data JSON file')
    data_group.add_argument('--val_file', type=str, default=None,
                           help='Path to validation data JSON file (optional)')
    data_group.add_argument('--test_file', type=str, default=None,
                           help='Path to test data JSON file')
    data_group.add_argument('--val_size', type=float, default=0.2,
                           help='Validation split size if no val_file provided')
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model_name', type=str, 
                            default='cross-encoder/ms-marco-MiniLM-L-6-v2',
                            help='Pre-trained cross-encoder model to use')
    model_group.add_argument('--max_length', type=int, default=128,
                            help='Maximum sequence length')
    model_group.add_argument('--device', choices=['auto', 'cuda', 'cpu'], 
                            default='auto', help='Device to use')
    
    # Training arguments
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument('--batch_size', type=int, default=64, 
                               help='Training batch size')
    training_group.add_argument('--epochs', type=int, default=5, 
                               help='Number of training epochs')
    training_group.add_argument('--learning_rate', type=float, default=2e-5, 
                               help='Learning rate')
    training_group.add_argument('--warmup_ratio', type=float, default=0.1,
                               help='Warmup ratio for learning rate scheduler')
    training_group.add_argument('--seed', type=int, default=583912, 
                               help='Random seed for reproducibility')
    training_group.add_argument('--gradient_clip_norm', type=float, default=1.0,
                               help='Gradient clipping norm')
    training_group.add_argument('--validation_steps', type=int, default=8000,
                               help='Steps between validation runs')
    
    # Focal Loss arguments
    focal_group = parser.add_argument_group('Focal Loss Configuration')
    focal_group.add_argument('--focal_alpha', type=float, default=0.8,
                            help='Focal loss alpha parameter (class balancing)')
    focal_group.add_argument('--focal_gamma', type=float, default=3.0,
                            help='Focal loss gamma parameter (focusing)')
    
    # Batch sampling arguments
    sampling_group = parser.add_argument_group('Batch Sampling Configuration')
    sampling_group.add_argument('--use_balanced_batches', action='store_true', default=True,
                               help='Use balanced batch sampling')
    sampling_group.add_argument('--positive_ratio', type=float, default=0.3,
                               help='Ratio of positive examples in each batch')
    
    # Data augmentation arguments
    augmentation_group = parser.add_argument_group('Data Augmentation Configuration')
    augmentation_group.add_argument('--augmentation_prob', type=float, default=0.2,
                                   help='Probability of applying data augmentation')
    augmentation_group.add_argument('--apply_augmentation', action='store_true', default=True,
                                   help='Whether to apply data augmentation')
    
    # Inference arguments
    inference_group = parser.add_argument_group('Inference Configuration')
    inference_group.add_argument('--model_path', type=str, default=None,
                                help='Path to trained model for testing')
    inference_group.add_argument('--threshold', type=float, default=0.5,
                                help='Classification threshold')
    inference_group.add_argument('--find_optimal_threshold', action='store_true',
                                help='Find optimal threshold on validation set')
    inference_group.add_argument('--threshold_metric', choices=['f1', 'precision', 'recall'],
                                default='f1', help='Metric to optimize for threshold selection')
    
    # Output arguments
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output_dir', type=str, default='./outputs',
                             help='Directory for output files')
    output_group.add_argument('--model_save_dir', type=str, default='./models',
                             help='Directory to save trained models')
    output_group.add_argument('--save_predictions', action='store_true',
                             help='Save predictions to file')
    output_group.add_argument('--save_evaluation', action='store_true', default=True,
                             help='Save evaluation results to file')
    
    # Logging arguments
    logging_group = parser.add_argument_group('Logging Configuration')
    logging_group.add_argument('--verbose', action='store_true', default=True,
                              help='Verbose output')
    logging_group.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                              default='INFO', help='Logging level')
    
    return parser