# build in libraries
import logging
from transformers import AutoModel, AutoTokenizer
import json
import torch
import os

# custom functions

from ContrastiveTrainer import ContrastiveTrainer
from ContrastiveSkillDataset import ContrastiveSkillDataset
from ContrastiveSkillEncoder import ContrastiveSkillEncoder
from SkillEvaluator import SkillEvaluator
from Loss import ContrastiveLossWrapper
from SimplifiedContrastiveCollator import SimplifiedContrastiveCollator
from arguments_contrastive import create_parser

# env params
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    """training script"""
    
    # parser
    parser = create_parser()
    args = parser.parse_args()
    finish_training = False
    # logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    zero_shot = getattr(args, "zero_shot", False)
    excluded_path = getattr(args, "excluded_path", None)
    if args.test_only:
        assert args.test_data_path, "You must provide --test_data_path"
        assert args.mapping_path, "You must provide --mapping_path"
        logger.info("Running in test-only mode")
        # Load tokenizer + model

        if args.pure_baseline:
            logger.info("Loading raw baseline model from HuggingFace")
            model = ContrastiveSkillEncoder(args)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            args.tokenizer = tokenizer
            model.encoder = AutoModel.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if getattr(args, "fp16", False) else torch.float32
            )
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            assert args.resume_from_checkpoint, "You must set --resume_from_checkpoint when not using --pure_baseline"
            logger.info("Loading fine-tuned model from checkpoint")
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.resume_from_checkpoint, "tokenizer"))
            args.tokenizer = tokenizer
            model = ContrastiveSkillEncoder.from_pretrained(
                args.resume_from_checkpoint, args=args
            )
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Load test data
        with open(args.test_data_path) as f:
            test_data = json.load(f)
        with open(args.mapping_path) as f:
            definition_to_label = json.load(f)
        if zero_shot:
            with open(excluded_path) as f:
                labels_to_exclude = json.load(f)
                labels_to_exclude = set(labels_to_exclude)
        else:
            labels_to_exclude = None

        evaluator = SkillEvaluator(model, definition_to_label)
        dataset_test = args.dataset
        output_path_path = os.path.join(args.output_dir, dataset_test, "test_predictions_2.json")
        os.makedirs(os.path.dirname(output_path_path), exist_ok=True)
        
        results = evaluator.evaluate(
            test_data,
            k=args.top_k,
            return_preds=True,
            output_path=output_path_path,
            zero_shot_mode=zero_shot,
            excluded_labels=labels_to_exclude
            )
        print(f"Test R-Precision@{args.top_k}: {results[f'rprecision@{args.top_k}']:.4f}")
        print(f"Test MRR: {results[f'mrr']:.4f}")
        return
    
    if args.pre_train:
        
        # load stage 1 data
        logger.info(f"Initiating the Pre Training")
        logger.info(f"Loading data from {args.data_path_pt}")
        with open(args.data_path_pt) as f:
            skill_data = json.load(f)
            
        train_dataset = ContrastiveSkillDataset(args, skill_data, allow_pre_train=True)
            
        # initialise tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        args.tokenizer = tokenizer
        model = ContrastiveSkillEncoder(args) 
        # loss
        loss = ContrastiveLossWrapper(args)
        # collator
        collator = SimplifiedContrastiveCollator(args, allow_pre_train=True)
        # trainer
        trainer = ContrastiveTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            collate_fn=collator,
            loss_fn=loss,
            allow_pre_train=True
        )
        # Start training
        trainer.train()
        logger.info("Pre-training completed.")
        
        if args.use_pre_trained:
            logger.info(f"Initiating the main bi-encoder training with pre-training phase")          
            logger.info(f"Loading data from {args.data_path}")
            # switch off the pre_train configuration 
            args.pre_train = False
            if not args.pre_train:
                print('YOu are golden......') 
            with open(args.data_path) as f:
                skill_data = json.load(f)   
                
            train_dataset = ContrastiveSkillDataset(args, skill_data, allow_pre_train=False)   
        
            # load the pre-trained model
            assert args.checkpoint_dir_pt
            logger.info("Loading pre-trained model from checkpoint")
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.checkpoint_dir_pt, "best_model/tokenizer"))
            args.tokenizer = tokenizer
            model = ContrastiveSkillEncoder.from_pretrained(os.path.join(args.checkpoint_dir_pt, "best_model"), args=args)
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")

            # loss
            loss = ContrastiveLossWrapper(args)
            # collator
            collator = SimplifiedContrastiveCollator(args, allow_pre_train=False)
            # trainer
            trainer = ContrastiveTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                collate_fn=collator,
                loss_fn=loss,
                allow_pre_train=False
            )
            # Start training
            trainer.train()
            finish_training = True
            logger.info("Bi-encoder Training completed.")
            
    if not args.pre_train and not finish_training:
        logger.info(f"Initiating the main bi-encoder training (no pre-train).")         
        logger.info(f"Loading data from {args.data_path}")
        with open(args.data_path) as f:
            skill_data = json.load(f)      
            
        train_dataset = ContrastiveSkillDataset(args, skill_data, allow_pre_train=False)
        
        if args.use_pre_trained:
            assert args.checkpoint_dir_pt
            logger.info(f"Pre-trained model found, initiating stage 2 from pre-trained checkpoint")
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.checkpoint_dir_pt, "best_model/tokenizer"))
            args.tokenizer = tokenizer
            model = ContrastiveSkillEncoder.from_pretrained(os.path.join(args.checkpoint_dir_pt, "best_model"), args=args)
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            logger.info(f"No pre-trained model found, Stage 2 trained from base model")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            args.tokenizer = tokenizer
            model = ContrastiveSkillEncoder(args) 

        # loss
        loss = ContrastiveLossWrapper(args)
        # collator
        collator = SimplifiedContrastiveCollator(args, allow_pre_train=False)
        # trainer
        trainer = ContrastiveTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            collate_fn=collator,
            loss_fn=loss,
            allow_pre_train=False
        )
        # Start training
        trainer.train()
        logger.info("Training Stage 2 completed.")

if __name__ == "__main__":
    main()
