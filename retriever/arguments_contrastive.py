import argparse

def create_parser():
    """Create argument parser with organized groups"""
    parser = argparse.ArgumentParser(
        description="Contrastive Learning for Skill Extraction (Stage 1 (Bi-encoder))",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows defaults in help
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model_name', type=str, 
                            default='sentence-transformers/all-mpnet-base-v2',
                            help='Pre-trained model to use')
    model_group.add_argument('--tokenizer', type=str, 
                            default=None,
                            help='The model tokenizer to use, based on the model name')
    model_group.add_argument('--embedding_dim', type=int, default=768,
                            help='Dimension of embeddings') 
    model_group.add_argument('--dropout_rate', type=int, default=0.1,
                            help='Dropout rate for training')  
    model_group.add_argument('--run_name', type=str,
                            help='The version of the run')
    model_group.add_argument("--test_only", action='store_true', required=False, 
                            help="Determines whether you want only to load the fine-tuned model for testing")
    model_group.add_argument("--resume_from_checkpoint", type=str, default=None, required=False, 
                            help="A path to a partially or fully fine-tuned model, only used when the training was disrupted or you want only test")                                     
    model_group.add_argument("--seed", type=int, default=42, required=False, 
                            help="A random seed for reproducibility")
    model_group.add_argument("--pure_baseline", action='store_true', required=False, 
                            help="IF you want to run pure model baseline")                                   
    model_group.add_argument("--dataset", required=True, type=str, 
                            help="A name of the dataset to use. This will reflect the name of the predicted sample.")
    model_group.add_argument('--pre_train', action='store_true',
                             help='Enables the pre-training phase') 
                      
    # Data arguments
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data_path', type=str, required=False,
                           help='Path to skills JSON file (training data)')
    data_group.add_argument('--val_data_path', type=str, required=False,
                           help='Path to skills JSON file (validation)')
    data_group.add_argument('--test_data_path', type=str, required=False,
                           help='Path to skills JSON file (test)')
    data_group.add_argument('--mapping_path', type=str, required=False,
                           help='Path to definitions to labels dictionary, used for eval and testing') 
    data_group.add_argument('--excluded_path', type=str, required=False,
                           help='Path to set of labels to exclude')     
    data_group.add_argument('--top_k', type=int, default=5, required=False,
                           help='The number of top k most similar items to retrieve')           
    data_group.add_argument('--max_length_sent', type=int, default=128,
                           help='Maximum length for sentence tokenization, consider the augmentation strategy')
    data_group.add_argument('--max_length_label', type=int, default=32,
                           help='Maximum length for skill label tokenization')
    data_group.add_argument('--augmentation_ratio', type=float, default=0.8,
                           help='Probability of augmenting a sentence')
    data_group.add_argument('--enable_augmentation', action='store_true',
                           help='Enable sentence augmentation')
    data_group.add_argument('--zero_shot', action='store_true',
                           help='Enable zero-shot setting')    
    
    # Negative sampling arguments
    neg_group = parser.add_argument_group('Negative Sampling Configuration')
    neg_group.add_argument('--hard_negative_strategy', type=str, default='in_batch',
                          choices=['in_batch', 'esco', 'mixed_batch_esco'],
                          help='Negative sampling strategy')
    neg_group.add_argument('--num_hard_negatives', type=int, default=20,
                          help='Number of hard negatives to sample')

    # Loss function arguments
    loss_group = parser.add_argument_group('Loss Function Configuration')
    loss_group.add_argument('--loss_type', type=str, default='ntxent',
                           choices=['ntxent', 'margin'],
                           help='Loss function type')
    loss_group.add_argument('--temperature_main', type=float, default=0.05,
                           help='Temperature for NT-Xent loss (main phase)')   
    loss_group.add_argument('--temperature_pre_train', type=float, default=0.03,
                           help='Temperature for NT-Xent loss (pre-train phase)')
    loss_group.add_argument('--margin', type=float, default=0.2,
                           help='Margin for margin loss')
    loss_group.add_argument('--reduction', type=str, default='mean',
                           choices=['mean', 'sum'],
                           help='Loss reduction method')
    
    # Training arguments
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--batch_size', type=int, default=64,
                            help='Training batch size')
    train_group.add_argument('--log_with', type=str, default='tensorboard',
                            help='The logging strategy. Model configured to use tensorboard only.')    
    train_group.add_argument('--learning_rate_main', type=float, default=2e-5,
                            help='Learning rate for main training phase')
    train_group.add_argument('--learning_rate_pre_train', type=float, default=3e-5,
                            help='Learning rate for pre_train phase')
    train_group.add_argument('--num_epochs', type=int, default=1,
                            help='Number of training epochs')
    train_group.add_argument('--weight_decay', type=float, default=0.01,
                            help='Weight Decay')    
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help='Gradient accumulation steps')
    train_group.add_argument('--early_stopping_patience', type=int, default=2,
                            help='Determines the number of epoch after which if the performance is not improved, training must stop')
    train_group.add_argument('--fp16', action="store_true", required=False,
                            help='A boolean to determine whether to use half-precision')
    train_group.add_argument('--run_initial_eval', type=bool, default=False,
                            help='A boolean to determine whether to run evaluation using a vanilla model (before training starts)')
    train_group.add_argument("--save_baseline", action="store_true",
                            help="Save the initial (pre-training) model as baseline_initial")   
    train_group.add_argument("--log_freq", type=int, default=10,
                            help="The frequency of parameters updates as logged in logger")        
    train_group.add_argument("--max_grad_norm", type=int, default=1, required=False, 
                            help="Gradient Clipping")
    train_group.add_argument("--warmup_percent", type=float, default=0.05, required=False, 
                            help="The warmup ratio")  
    train_group.add_argument("--use_cosine_schedule", type=bool, default=True, required=False, 
                            help="Whether to use cosine schedule or not")

    # pre_train stage
    pt_group = parser.add_argument_group('Pre-train Stage Training Configuration')
    pt_group.add_argument('--data_path_pt', type=str, required=False,
                             help='Path to the Pre-train data')
    pt_group.add_argument('--output_dir_pt', type=str, required=False,
                             help='Path to the output directory for pre-train models (tensorboard + results)')
    pt_group.add_argument('--checkpoint_dir_pt', type=str, required=False,
                             help='Path to the checkpoint directory for pre-train models (models)')
    pt_group.add_argument('--use_pre_trained', action='store_true',
                             help='If the base model is available, use it for main bi encoder training')
    
    # Collator arguments
    collator_group = parser.add_argument_group('Collator Configuration')
    collator_group.add_argument('--truncation', type=bool, default=True,
                               help='Whether to truncate sequences')
    collator_group.add_argument('--enable_relative_ranking', action='store_true',
                               help='Enable relative ranking loss')
    collator_group.add_argument('--ranking_margin', type=float, default=0.2,
                               help='Margin for relative ranking')
    
    # Monitoring arguments
    monitor_group = parser.add_argument_group('Monitoring Configuration')
    monitor_group.add_argument('--enable_monitoring', action='store_true',
                              help='Enable negative count monitoring')
    monitor_group.add_argument('--log_warnings', action='store_true',
                              help='Log warnings for low negative counts')
    monitor_group.add_argument('--min_valid_negatives', type=int, default=10,
                              help='Minimum valid negatives threshold')
    
    # Output arguments
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output_dir', type=str, default='./outputs',
                             help='Directory for outputs')
    output_group.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                             help='Directory for checkpoints')
    output_group.add_argument('--save_freq', type=int, default=5,
                             help='Save checkpoint every N epochs')
    
    return parser