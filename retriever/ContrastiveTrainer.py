import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
import os
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR
from torch.optim import AdamW
import json 
from tqdm import tqdm
import shutil
from typing import List, Dict, Set, Tuple, Optional, Any

# our modules
from .SkillEvaluator import SkillEvaluator

class ContrastiveTrainer:
    """
    Trainer for our contrastive learning pipeline
    """
    
    def __init__(
        self,
        model: nn.Module,
        args,
        train_dataset,
        allow_pre_train,
        collate_fn=None,
        loss_fn=None
        ):
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.collate_fn = collate_fn
        self.loss_fn = loss_fn
        self.zero_shot = getattr(args, 'zero_shot', False)
        self.excluded_path = getattr(args, 'excluded_path', None)
        self.pre_train = allow_pre_train

        # Early stopping parameters
        self.patience = getattr(args, 'early_stopping_patience', 1)
        self.best_score = 0.0  
        self.epochs_without_improvement = 0     
        
        # validation parameters
        self.val_data = None
        self.evaluator = None
        self.top_k = getattr(args, 'top_k', 5)
        
        # learning rate (depending on training phase)
        if self.pre_train:
            self.learning_rate = args.learning_rate_pre_train
        else:
            self.learning_rate = args.learning_rate_main

        # Initialise the huggingface's accelerate 
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision='fp16' if getattr(args, 'fp16', False) else 'no',
            log_with=getattr(args, 'log_with', None), # use tensorboard but could use a different one if needed (e.g., comet)
            project_dir=args.output_dir if self.pre_train else args.output_dir_pt
        )
        
        # configure logging
        self.setup_logging()
        
        # configure zero shot
        if self.zero_shot:
            assert self.excluded_path, "You must provide a path to excluded labels list if using zero-shot"
            with open(self.excluded_path) as f:
                lab_to_exclude = json.load(f)
                self.lab_to_exclude = set(lab_to_exclude)
        else:
            self.lab_to_exclude = None
        
        # set random seed if not specified 
        if getattr(args, 'seed', None):
            set_seed(args.seed)
            
        # output directories
        if self.pre_train:
            os.makedirs(args.output_dir_pt, exist_ok=True)
            os.makedirs(args.checkpoint_dir_pt, exist_ok=True)
        else:   
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            
        # data loaders
        self.train_dataloader = self.create_dataloader(train_dataset, shuffle=True)
        
        # optimizer and scheduler
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_lr_scheduler()
            
        # prepare
        objs = [self.model,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler]

        prepared = self.accelerator.prepare(*objs)

        # unpack; order is the same as in `objs`
        (self.model,
         self.optimizer,
         self.train_dataloader,
         self.lr_scheduler
         ) = prepared
            
        # validation data
        self.val_data   = None
        self.evaluator  = None
        if getattr(args, "val_data_path", None):
            self.setup_validation(args.val_data_path, args.mapping_path)

        # tracking
        self.global_step = 0
        self.start_epoch = 0

    # logging 
    def setup_logging(self):
        """
        The function to track model training
        """          
            
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        safe_args = {
            k: v for k, v in vars(self.args).items()
            if isinstance(v, (int, float, str, bool))
        }
        
        # in case of distributed training, log form main process only 
        if self.accelerator.is_main_process:
            if self.args.log_with == 'tensorboard':
                run_name = getattr(self.args, 'run_name',
                                   f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") 
                if self.pre_train:
                    dir_to_log = os.path.join(self.args.output_dir_pt, 'tensorboard', run_name)
                else: 
                    dir_to_log = os.path.join(self.args.output_dir, 'tensorboard', run_name)
                tb_log_dir = dir_to_log
                  
            self.logging_dir = tb_log_dir        
            # Initialise Accelerate with TensorBoard
            self.accelerator.init_trackers(
                project_name="contrastive-skill-classification",
                config=safe_args
            )
            if self.pre_train:
                self.logger.info(f"TensorBoard logging initialized. Run 'tensorboard --logdir {self.args.output_dir_pt}/tensorboard' to view")
            else:
                self.logger.info(f"TensorBoard logging initialized. Run 'tensorboard --logdir {self.args.output_dir}/tensorboard' to view")
            
            # Store the run name for later use
            self.run_name = run_name
        
    # dataloader 
    def create_dataloader(self, dataset, shuffle=True):
        if dataset is None:
            return None    
        # with drop_last set to shuffle we set it to true for training and false for evaluation
        # this is to prevent very small batches to be used for training. 
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=getattr(self.args, 'num_workers', 4),
            pin_memory=True,
            drop_last=shuffle 
        )
        
    def create_optimizer(self):
        # skip bias and LayerNorm parameters 
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
    def create_lr_scheduler(self):
        """Create learning rate scheduler with optional warmup."""
        steps_per_epoch = len(self.train_dataloader)
        num_training_steps = steps_per_epoch * self.args.num_epochs
        warmup_pct = getattr(self.args, 'warmup_percent', 0.0)
        num_warmup_steps = int(num_training_steps * warmup_pct)
        
        # print(f"Training setup:")
        # print(f"  Total epochs: {args.num_epochs}")
        # print(f"  Steps per epoch: {steps_per_epoch}")
        # print(f"  Total steps: {num_training_steps}")
        # print(f"  Warmup: {warmup_pct*100:.1f}% = {num_warmup_steps} steps")
        
        # no warmup (still cosine annealing if enabled)
        if num_warmup_steps == 0:
            if getattr(self.args, 'use_cosine_schedule', True):
                return CosineAnnealingLR(
                    self.optimizer,
                    T_max=num_training_steps,
                    eta_min=self.learning_rate * getattr(self.args, 'final_lr_percent', 0.1)
                )
            else:
                return ConstantLR(self.optimizer, factor=1.0, total_iters=num_training_steps)
            
        # with warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=getattr(self.args, 'warmup_start_percent', 0.1),  # Start at 10% of LR
            end_factor=1.0,
            total_iters=num_warmup_steps
        )  
        
        if getattr(self.args, 'use_cosine_schedule', True):
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                # adjust by warmup
                T_max=num_training_steps - num_warmup_steps,
                eta_min=self.learning_rate * getattr(self.args, 'final_lr_percent', 0.1)
                
            )
        else:
            main_scheduler = ConstantLR(
            self.optimizer, 
            factor=1.0, 
            total_iters=num_training_steps - num_warmup_steps
        )
            
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[num_warmup_steps]
        )
    # test
    def run_test(self, top_k: int):
        test_path = self.args.test_data_path
        if not test_path:
            return

        with open(test_path) as f:
            test_data = json.load(f)
        if self.pre_train:
            best_dir = os.path.join(self.args.checkpoint_dir_pt, "best_model")
            outfile = os.path.join(self.args.output_dir_pt, "test_predictions.json")
        else: 
            best_dir = os.path.join(self.args.checkpoint_dir, "best_model")
            outfile = os.path.join(self.args.output_dir, "test_predictions.json")
        # loaded model must use the same precision as the original, make sure to adequately adjust params
        self.model = self.model.__class__.from_pretrained(best_dir).to(self.accelerator.device)
        if self.args.fp16:
            self.model.half()
        self.evaluator.refresh_model(self.model)

        # this time we save the predictions as well.
        self.logger.info(f"Running Test...")
        test_stats = self.evaluator.evaluate(
            test_data,
            k=top_k,
            return_preds=True,
            zero_shot_mode=self.zero_shot,
            excluded_labels=self.lab_to_exclude
        )
        r5_test   = test_stats[f"rprecision@{top_k}"]
        mrr_test  = test_stats[f"mrr"]
        preds_all = test_stats["predictions"]  

        output = []
        for ex, labels in zip(test_data, preds_all):
            output.append({
                "sentence": ex["sentence"],
                "predicted_labels": labels,          # list[(label, score)]
                "true_labels": ex["labels"],
            })
        
        
        if self.accelerator.is_main_process:
            with open(outfile, "w") as f:
                json.dump(output, f, indent=2)
            self.logger.info(f"Saved predictions to {outfile}")   
            self.accelerator.log({f"test/rprecision@{top_k}": r5_test},
                             step=self.global_step)
            self.logger.info(f"TEST  R-Precision@{top_k} = {r5_test:.4f}")
            self.logger.info(f"TEST  MRR = {mrr_test:.4f}")
    # validation
    def _early_stopping(self, current_score: float) -> bool:
        """
        Return True  → halt training
               False → continue
        """
        if current_score > self.best_score:
            self.best_score = current_score
            self.epochs_without_improvement = 0

            if self.accelerator.is_main_process:
                if self.pre_train:
                    best_dir = os.path.join(self.args.checkpoint_dir_pt, "best_model")
                else:
                    best_dir = os.path.join(self.args.checkpoint_dir, "best_model")
                if os.path.exists(best_dir):
                    shutil.rmtree(best_dir)
                self.save_checkpoint("best_model")
                self.logger.info("New best model saved.")

            return False

        # no improvement
        self.epochs_without_improvement += 1
        self.logger.info(
            f"No improvement for {self.epochs_without_improvement} epoch(s). "
            f"Best so far: {self.best_score:.4f}"
        )

        return self.epochs_without_improvement >= self.patience
    
    def setup_validation(self, val_data_path: str, mapping_path: str):
        # 1) validation sentences/labels
        with open(val_data_path) as f:
            self.val_data = json.load(f)

        # 2) full definition → label dict  (JSON: {"definition": "skill_label", ...})
        with open(mapping_path) as f:
            def2lab = json.load(f)

        self.evaluator = SkillEvaluator(
            model=self.accelerator.unwrap_model(self.model),
            definition_to_label=def2lab,
            device=self.accelerator.device,
        )
        self.logger.info(f"Loaded {len(def2lab)} definitions for retrieval.")
            
    def train(self):
        """Main training loop."""
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataset)}")
        self.logger.info(f"  Num Epochs = {self.args.num_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {len(self.train_dataloader) * self.args.num_epochs}")
        self.logger.info(f"  Current Seed = {self.args.seed}")
        # when validation data is present
        if self.val_data:
            self.logger.info(f"  Validation samples = {len(self.val_data)}")
            self.logger.info(f"  Early stopping patience = {self.patience}")
        # Training Loop
        # save the baseline if enabled
        if self.args.save_baseline and self.accelerator.is_main_process:
            self.save_checkpoint("baseline_initial")
        for epoch in range(self.start_epoch, self.args.num_epochs):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            self.logger.info(f"{'='*50}")

            # train 
            train_metrics = self.train_epoch(epoch)

            # log
            if self.accelerator.is_main_process:
                self.accelerator.log(train_metrics, step=self.global_step)
                
            # evaluate and check early stopping
            if self.val_data:
                # "refresh" the model
                self.evaluator.refresh_model(
                    self.accelerator.unwrap_model(self.model)
                )
                self.logger.info(f"Running Evaluation...")
                val_stats = self.evaluator.evaluate(self.val_data, k=self.top_k, zero_shot_mode=self.zero_shot, excluded_labels=self.lab_to_exclude)
                r5 = val_stats[f"rprecision@{self.top_k}"]
                
                if self.accelerator.is_main_process:
                    self.logger.info(f"EVAL  R-Precision@{self.top_k} = {r5:.4f}")
                    self.accelerator.log({f"eval/rprecision@{self.top_k}": r5},
                                         step=self.global_step)

                if self._early_stopping(r5):
                    self.logger.info("Early stopping triggered.")
                    break
    
            # Save periodic checkpoint
            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}")
                
        self.logger.info("Training completed!")

        # for test data (after training)
        if getattr(self.args, "test_data_path", None):
            self.logger.info("Testing the model")
            self.run_test(top_k=self.top_k)
            
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Training for a single epoch, returns metrics"""
        
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Progress bar
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training epoch {epoch + 1}",
            disable=not self.accelerator.is_local_main_process
        )
        for batch in progress_bar:
            with self.accelerator.accumulate(self.model):
                # Forward pass
                loss_dict = self.training_step(batch)
                loss = loss_dict['loss']
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping (if configured)
                if self.accelerator.sync_gradients and getattr(self.args, 'max_grad_norm', None):
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
                # Optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Gather metrics from all processes
                gathered_loss = self.accelerator.gather(loss).mean().item()
                
                epoch_loss += gathered_loss
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': gathered_loss,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    if self.global_step % self.args.log_freq == 0:
                        self.accelerator.log({
                            'train/loss_step': gathered_loss,
                            'train/lr': self.optimizer.param_groups[0]['lr'],
                        }, step=self.global_step)
        # Return epoch metrics
        return {
            'train/loss_epoch': epoch_loss / num_batches,
            'train/epoch': epoch + 1
        }
        
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Single training step."""
        # Get embeddings for anchors and positives
        anchor_embeddings = self.model(
            batch['anchor_input_ids'],
            batch['anchor_attention_mask']
        )
        
        positive_embeddings = self.model(
            batch['positive_input_ids'],
            batch['positive_attention_mask']
        )
        
        # Handle negative embeddings if present
        negative_embeddings = None
        if batch['negative_input_ids'] is not None:
            # Reshape negatives for batch processing
            neg_ids = batch['negative_input_ids']
            neg_mask = batch['negative_attention_mask']
            batch_size, num_negs, seq_len = neg_ids.shape
            
            # Flatten batch and negatives dimensions
            neg_ids = neg_ids.view(-1, seq_len)
            neg_mask = neg_mask.view(-1, seq_len)
            
            # Get embeddings
            neg_embeddings_flat = self.model(neg_ids, neg_mask)
            
            # Reshape back
            negative_embeddings = neg_embeddings_flat.view(batch_size, num_negs, -1)
        
        # Compute loss
        loss_output = self.loss_fn(
            anchor_embeddings=anchor_embeddings,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
            in_batch_negative_masks=batch.get('in_batch_negative_masks')
        )
        
        return {"loss": loss_output}
        
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        if self.pre_train:
            save_path = os.path.join(self.args.checkpoint_dir_pt, name)
        else:
            save_path = os.path.join(self.args.checkpoint_dir, name)
            
        os.makedirs(save_path, exist_ok=True)
        
        # Unwrap model for saving
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Save model
        unwrapped_model.save_pretrained(save_path)
        # save tokenizer (especially important for mix strategy)
        unwrapped_model.tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
        
        # Save trainer state
        trainer_state = {
            'epoch': self.global_step // len(self.train_dataloader),
            'global_step': self.global_step,
            'best_metric': self.best_score,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'args': vars(self.args),
        }
        
        torch.save(trainer_state, os.path.join(save_path, 'trainer_state.pt'))
        self.logger.info(f"Checkpoint saved to {save_path}")

