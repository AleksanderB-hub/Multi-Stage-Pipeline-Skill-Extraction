import json
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any

# Setup logging
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to: {seed}")


def create_train_val_split(data: List[Dict], val_size: float = 0.2, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Create train/validation split"""
    train_data, val_data = train_test_split(
        data, test_size=val_size, shuffle=True, random_state=seed
    )
    logger.info(f"Created train/val split: Train={len(train_data)}, Val={len(val_data)}")
    return train_data, val_data


def load_data_from_dict_format(data_list: List[Dict]) -> Tuple[List[str], List[List[str]], List[Set[str]]]:
    """Convert dictionary format data to model format"""
    sentences = []
    candidates_list = []
    true_skills_list = []
    
    for item in data_list:
        sentences.append(item['sentence'])
        candidates_list.append(item['candidate_labels'])
        true_skills_list.append(set(item['true_labels']))
    
    return sentences, candidates_list, true_skills_list


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.8, gamma=3.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class BalancedBatchSampler:
    """Custom sampler for balanced positive/negative examples"""
    def __init__(self, examples: List[InputExample], batch_size: int, positive_ratio: float = 0.3):
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        
        # Separate positive and negative examples
        self.positive_indices = []
        self.negative_indices = []
        
        for idx, example in enumerate(examples):
            if example.label == 1.0:
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)
        
        self.n_positives_per_batch = max(1, int(batch_size * positive_ratio))
        self.n_negatives_per_batch = batch_size - self.n_positives_per_batch
        
    def __iter__(self):
        # Shuffle indices
        positive_indices = self.positive_indices.copy()
        negative_indices = self.negative_indices.copy()
        random.shuffle(positive_indices)
        random.shuffle(negative_indices)
        
        # Create balanced batches
        batches = []
        pos_idx = 0
        neg_idx = 0
        
        while pos_idx < len(positive_indices) or neg_idx < len(negative_indices):
            batch = []
            
            # Add positives
            for _ in range(self.n_positives_per_batch):
                if pos_idx < len(positive_indices):
                    batch.append(positive_indices[pos_idx])
                    pos_idx += 1
                elif neg_idx < len(negative_indices):
                    batch.append(negative_indices[neg_idx])
                    neg_idx += 1
            
            # Add negatives
            for _ in range(self.n_negatives_per_batch):
                if neg_idx < len(negative_indices):
                    batch.append(negative_indices[neg_idx])
                    neg_idx += 1
                elif pos_idx < len(positive_indices):
                    batch.append(positive_indices[pos_idx])
                    pos_idx += 1
            
            if batch:
                random.shuffle(batch)
                batches.append(batch)
        
        # Shuffle batches
        random.shuffle(batches)
        
        # Yield indices
        for batch in batches:
            yield batch
    
    def __len__(self):
        total_examples = len(self.positive_indices) + len(self.negative_indices)
        return (total_examples + self.batch_size - 1) // self.batch_size


class RelevanceDataPreparer:
    """Prepare training data with augmentation"""
    def __init__(self, args):
        self.args = args
        self.augmentation_prob = args.augmentation_prob
        self.mask_token = "[MASK]"
        
    def prepare_training_examples(self, sentences: List[str], candidates_list: List[List[str]], 
                                true_skills_list: List[Set[str]], apply_augmentation: bool = None) -> List[InputExample]:
        """Convert data into training examples with optional augmentation"""
        if apply_augmentation is None:
            apply_augmentation = self.args.apply_augmentation
            
        examples = []
        
        for sentence, candidates, true_skills in zip(sentences, candidates_list, true_skills_list):
            for candidate in candidates:
                label = 1.0 if candidate in true_skills else 0.0
                
                # Standard example
                examples.append(InputExample(texts=[sentence, candidate], label=label))
                
                # Apply augmentation if enabled
                if apply_augmentation and random.random() < self.augmentation_prob:
                    # Label masking
                    if len(candidate.split()) > 1:
                        words = candidate.split()
                        mask_idx = random.randint(0, len(words) - 1)
                        words[mask_idx] = self.mask_token
                        masked_candidate = " ".join(words)
                        examples.append(InputExample(texts=[sentence, masked_candidate], label=label))
                    
                    # Sentence noise
                    if random.random() < 0.5:
                        words = sentence.split()
                        if len(words) > 10:
                            remove_idx = random.randint(0, len(words) - 1)
                            noisy_sentence = " ".join(words[:remove_idx] + words[remove_idx+1:])
                            examples.append(InputExample(texts=[noisy_sentence, candidate], label=label))
        
        logger.info(f"Created {len(examples)} training examples (augmentation: {apply_augmentation})")
        return examples


class SkillExtractorTrainer:
    """Main trainer class for the cross-encoder model"""
    
    def __init__(self, args):
        self.args = args
        
        # Set device
        if args.device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = args.device
        
        # Normalize device string (cuda and cuda:0 are the same)
        if self.device == "cuda":
            self.device = "cuda:0"
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize cross-encoder
        self.model = CrossEncoder(
            args.model_name,
            num_labels=1,
            max_length=args.max_length,
            device=self.device
        )
        
        # Verify the model is on the correct device
        model_device = str(next(self.model.model.parameters()).device)
        logger.info(f"Model initialized on device: {model_device}")
        
        # Only move if actually different
        if model_device != self.device:
            logger.info(f"Moving model from {model_device} to {self.device}")
            self.model.model = self.model.model.to(self.device)
            logger.info(f"Model moved to: {next(self.model.model.parameters()).device}")
        else:
            logger.info("Model is already on the correct device")
        
        # Initialize data preparer
        self.data_preparer = RelevanceDataPreparer(args)
        
        # Track training history
        self.training_history = defaultdict(list)
        
    def train_with_focal_loss(self, train_examples: List[InputExample], val_examples: List[InputExample] = None,
                            model_save_path: str = None) -> CrossEncoder:
        """Train using Focal Loss"""
        args = self.args
        
        # Print device info once
        model_device = str(next(self.model.model.parameters()).device)
        logger.info(f"Training starting - Model on: {model_device}, Target: {self.device}")
        
        # Prepare data loader
        if args.use_balanced_batches:
            train_dataset = list(range(len(train_examples)))
            sampler = BalancedBatchSampler(train_examples, args.batch_size, args.positive_ratio)
            
            def collate_fn(indices):
                batch_examples = [train_examples[i] for i in indices]
                texts1 = [ex.texts[0] for ex in batch_examples]
                texts2 = [ex.texts[1] for ex in batch_examples]
                labels = torch.tensor([ex.label for ex in batch_examples], dtype=torch.float, device=self.device)
                return texts1, texts2, labels
            
            train_dataloader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=collate_fn)
            logger.info("Using balanced batch sampling")
        else:
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
            logger.info("Using standard DataLoader")
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.model.parameters(), lr=args.learning_rate)
        num_training_steps = len(train_dataloader) * args.epochs
        num_warmup_steps = int(num_training_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        
        # Initialize focal loss
        focal_loss = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        
        # Training loop
        self.model.model.train()
        global_step = 0
        best_val_f1 = 0
        
        for epoch in range(args.epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for batch in progress_bar:
                if args.use_balanced_batches:
                    texts1, texts2, labels = batch
                    inputs = self.model.tokenizer(texts1, texts2, padding=True, truncation=True,
                                                return_tensors="pt", max_length=self.model.max_length)
                    # Labels are already on correct device from collate_fn
                else:
                    texts = [[ex.texts[0], ex.texts[1]] for ex in batch]
                    labels = torch.tensor([ex.label for ex in batch], dtype=torch.float)
                    inputs = self.model.tokenizer([t[0] for t in texts], [t[1] for t in texts],
                                                padding=True, truncation=True, return_tensors="pt",
                                                max_length=self.model.max_length)
                    labels = labels.to(self.device)
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model.model(**inputs, return_dict=True)
                logits = outputs.logits.squeeze(-1)
                
                # Compute focal loss
                loss = focal_loss(logits, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), args.gradient_clip_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                global_step += 1
                
                progress_bar.set_postfix({'loss': loss.item()})
                
                # Validation
                if val_examples and global_step % args.validation_steps == 0:
                    val_metrics = self.evaluate(val_examples, threshold=0.5)
                    self.training_history['val_f1'].append(val_metrics['f1'])
                    
                    if val_metrics['f1'] > best_val_f1:
                        best_val_f1 = val_metrics['f1']
                        if model_save_path:
                            self.model.save(model_save_path, save_transformer_model=True)
                            logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            self.training_history['train_loss'].append(avg_epoch_loss)
            logger.info(f"Epoch {epoch+1} - Average loss: {avg_epoch_loss:.4f}")
        
        # Save final model
        if model_save_path and not val_examples:
            self.model.save(model_save_path, save_transformer_model=True)
            logger.info(f"Final model saved to: {model_save_path}")
        
        return self.model
    
    def evaluate(self, examples: List[InputExample], threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate model performance"""
        all_predictions = []
        all_labels = []
        
        batch_size = 32
        for i in tqdm(range(0, len(examples), batch_size), desc="Evaluating", leave=False):
            batch = examples[i:i+batch_size]
            texts = [[ex.texts[0], ex.texts[1]] for ex in batch]
            labels = [ex.label for ex in batch]
            
            scores = self.model.predict(texts, show_progress_bar=False)
            all_predictions.extend(scores)
            all_labels.extend(labels)
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        pred_binary = (all_predictions > threshold).astype(int)
        true_binary = all_labels.astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_binary, pred_binary, average='binary', zero_division=0
        )
        
        return {'precision': precision, 'recall': recall, 'f1': f1, 'threshold': threshold}
    
    def find_optimal_threshold(self, val_examples: List[InputExample], metric: str = 'f1') -> float:
        """Find optimal classification threshold"""
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            metrics = self.evaluate(val_examples, threshold=threshold)
            score = metrics[metric]
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.2f} with {metric}: {best_score:.4f}")
        return best_threshold
    
    def train(self, train_data: List[Dict], val_data: List[Dict] = None, test_data: List[Dict] = None,
             model_save_path: str = None) -> Tuple[CrossEncoder, float]:
        """Complete training pipeline"""
        # Convert data format
        logger.info("Converting training data format...")
        sentences_train, candidates_train, true_skills_train = load_data_from_dict_format(train_data)
        
        # Prepare training examples
        train_examples = self.data_preparer.prepare_training_examples(
            sentences_train, candidates_train, true_skills_train, apply_augmentation=True
        )
        
        # Prepare validation examples
        val_examples = None
        if val_data:
            logger.info("Converting validation data format...")
            sentences_val, candidates_val, true_skills_val = load_data_from_dict_format(val_data)
            val_examples = self.data_preparer.prepare_training_examples(
                sentences_val, candidates_val, true_skills_val, apply_augmentation=False
            )
        
        # Train model
        logger.info("Training with Focal Loss...")
        model = self.train_with_focal_loss(
            train_examples=train_examples,
            val_examples=val_examples,
            model_save_path=model_save_path
        )
        
        # Find optimal threshold
        optimal_threshold = self.args.threshold
        if val_examples and self.args.find_optimal_threshold:
            optimal_threshold = self.find_optimal_threshold(val_examples, metric=self.args.threshold_metric)
        
        # Test evaluation if provided
        if test_data:
            logger.info("Evaluating on test set...")
            sentences_test, candidates_test, true_skills_test = load_data_from_dict_format(test_data)
            test_examples = self.data_preparer.prepare_training_examples(
                sentences_test, candidates_test, true_skills_test, apply_augmentation=False
            )
            
            test_metrics = self.evaluate(test_examples, threshold=optimal_threshold)
            logger.info(f"Test Set Metrics (threshold={optimal_threshold:.2f}):")
            logger.info(f"Precision: {test_metrics['precision']:.4f}")
            logger.info(f"Recall: {test_metrics['recall']:.4f}")
            logger.info(f"F1: {test_metrics['f1']:.4f}")
        
        return model, optimal_threshold


class SkillExtractorInference:
    """Inference class for trained models"""
    
    def __init__(self, args):
        self.args = args
        
        if not args.model_path:
            raise ValueError("Model path must be specified for inference")
        
        logger.info(f"Loading model from: {args.model_path}")
        self.model = CrossEncoder(args.model_path)
        self.threshold = args.threshold
    
    def predict_relevant_skills(self, sentence: str, candidates: List[str], 
                              threshold: float = None, return_scores: bool = False) -> List[str]:
        """Predict relevant skills for a sentence"""
        if threshold is None:
            threshold = self.threshold
        
        # Prepare pairs and get predictions
        pairs = [[sentence, candidate] for candidate in candidates]
        scores = self.model.predict(pairs)
        
        # Filter by threshold
        relevant = []
        for candidate, score in zip(candidates, scores):
            if score > threshold:
                if return_scores:
                    relevant.append((candidate, score))
                else:
                    relevant.append(candidate)
        
        return relevant
    
    def find_optimal_threshold(self, val_data: List[Dict], metric: str = None) -> float:
        """Find optimal threshold on validation data"""
        if metric is None:
            metric = self.args.threshold_metric
        
        # Convert validation data
        sentences_val, candidates_val, true_skills_val = load_data_from_dict_format(val_data)
        
        # Prepare validation examples
        data_preparer = RelevanceDataPreparer(self.args)
        val_examples = data_preparer.prepare_training_examples(
            sentences_val, candidates_val, true_skills_val, apply_augmentation=False
        )
        
        # Search for optimal threshold
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = self.threshold
        best_score = 0
        
        for threshold in thresholds:
            # Evaluate at this threshold
            all_predictions = []
            all_labels = []
            
            batch_size = 32
            for i in range(0, len(val_examples), batch_size):
                batch = val_examples[i:i+batch_size]
                texts = [[ex.texts[0], ex.texts[1]] for ex in batch]
                labels = [ex.label for ex in batch]
                
                scores = self.model.predict(texts, show_progress_bar=False)
                all_predictions.extend(scores)
                all_labels.extend(labels)
            
            # Calculate metrics
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            pred_binary = (all_predictions > threshold).astype(int)
            true_binary = all_labels.astype(int)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_binary, pred_binary, average='binary', zero_division=0
            )
            
            score = {'precision': precision, 'recall': recall, 'f1': f1}[metric]
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.2f} with {metric}: {best_score:.4f}")
        return best_threshold
    
    def evaluate(self, test_data: List[Dict], threshold: float = None) -> Dict[str, Any]:
        """Evaluate model on test data"""
        if threshold is None:
            threshold = self.threshold
        
        # Evaluate each example
        results = []
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        
        for item in tqdm(test_data, desc="Evaluating"):
            sentence = item['sentence']
            candidates = item['candidate_labels']
            true_labels = set(item['true_labels'])
            
            # Get predictions
            predicted_relevant = set(self.predict_relevant_skills(
                sentence, candidates, threshold=threshold, return_scores=False
            ))
            
            # Calculate metrics for this example
            true_positives = predicted_relevant & true_labels
            false_positives = predicted_relevant - true_labels
            false_negatives = true_labels - predicted_relevant
            
            total_true_positives += len(true_positives)
            total_false_positives += len(false_positives)
            total_false_negatives += len(false_negatives)
            
            results.append({
                'sentence': sentence,
                'true_labels': list(true_labels),
                'predicted_labels': list(predicted_relevant),
                'true_positives': list(true_positives),
                'false_positives': list(false_positives),
                'false_negatives': list(false_negatives)
            })
        
        # Calculate overall metrics
        precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
        recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_true_positives': total_true_positives,
            'total_false_positives': total_false_positives,
            'total_false_negatives': total_false_negatives,
            'detailed_results': results,
            'threshold_used': threshold
        }