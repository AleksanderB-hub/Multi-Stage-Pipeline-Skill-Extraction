from typing import List, Dict, Set, Tuple, Optional, Any
import torch

class SimplifiedContrastiveCollator:
    """
    Simplified collator with dynamic negative counting for mixed strategies.
    Accepts varying negative counts, no resampling or deduplication. 
    """

    def __init__(self, args, allow_pre_train):
        self.tokenizer = args.tokenizer
        self.max_length_sent = args.max_length_sent
        self.max_length_label = args.max_length_label
        self.truncation = args.truncation
        self.return_tensors = 'pt'
        self.padding = 'max_length'
        self.enable_relative_ranking = args.enable_relative_ranking
        self.ranking_margin = args.ranking_margin
        self.enable_monitoring = args.enable_monitoring
        self.log_warnings = args.log_warnings
        self.pre_train=allow_pre_train

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)

        # Extract components
        anchors = [item['anchor'] for item in batch]
        positives = [item['positive'] for item in batch]
        pre_selected_negatives = [item['negatives'] for item in batch]
        skill_indices = [item['skill_idx'] for item in batch]
        excluded_skill_indices = [item.get('excluded_skill_idx') for item in batch]
        strategies = [item['strategy'] for item in batch]
        is_augmented = [item.get('is_augmented', False) for item in batch]

        # Tokenize anchors and positives
        anchor_enc = self.tokenizer(
            anchors,
            max_length=self.max_length_sent if not self.pre_train else self.max_length_label,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors
        )
        
        positive_enc = self.tokenizer(
            positives,
            max_length=self.max_length_label if not self.pre_train else self.max_length_sent,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors
        )

        # Create in-batch negative masks
        if self.enable_monitoring:
            in_batch_negative_masks, monitoring_info = self._create_masks_with_monitoring(
                batch_size, skill_indices, excluded_skill_indices, strategies, pre_selected_negatives
            )
        else:
            in_batch_negative_masks = self._create_in_batch_negative_masks(
                batch_size, skill_indices, excluded_skill_indices, strategies
            )
            monitoring_info = None

        # Tokenize pre-selected negatives if any
        negatives_enc = None
        if any(pre_selected_negatives):
            negatives_enc = self._tokenize_pre_selected_negatives(
                pre_selected_negatives, batch_size
            )
        
        # Prepare relative ranking targets if enabled
        ranking_targets = None
        if self.enable_relative_ranking:
            ranking_targets = {
                "enable_ranking": torch.ones(batch_size, dtype=torch.bool),
                "ranking_margin": torch.full((batch_size,), self.ranking_margin)
            }
        
        output = {
            "anchor_input_ids": anchor_enc["input_ids"],
            "anchor_attention_mask": anchor_enc["attention_mask"],
            "positive_input_ids": positive_enc["input_ids"],
            "positive_attention_mask": positive_enc["attention_mask"],
            "negative_input_ids": negatives_enc["input_ids"] if negatives_enc else None,
            "negative_attention_mask": negatives_enc["attention_mask"] if negatives_enc else None,
            "in_batch_negative_masks": in_batch_negative_masks,
            "skill_indices": torch.tensor(skill_indices),
            "excluded_skill_indices": torch.tensor([-1 if x is None else x for x in excluded_skill_indices]),
            "is_augmented": torch.tensor(is_augmented),
            "ranking_targets": ranking_targets,
            "strategies": strategies
        }

        # Add monitoring info if enabled
        if monitoring_info is not None:
            output["monitoring_info"] = monitoring_info
            
        return output
    
    def _create_in_batch_negative_masks(
        self, 
        batch_size: int,
        skill_indices: List[int],
        excluded_skill_indices: List[Optional[int]],
        strategies: List[str]
    ) -> torch.Tensor:
        """
        Create masks to prevent conflicting in-batch negatives.
        Simple version without monitoring.
        """
        
        mask = torch.ones(batch_size, batch_size, dtype=torch.bool)
        mask.fill_diagonal_(False)
        
        in_batch_strategies = {"in_batch", "mixed_batch_esco"}
        
        for i in range(batch_size):
            if strategies[i] not in in_batch_strategies:
                mask[i, :] = False
                continue
                
            for j in range(batch_size):
                if i == j:
                    continue
                    
                # Exclude same skill
                if skill_indices[j] == skill_indices[i]:
                    mask[i, j] = False
                    
                # Exclude augmented skill
                if (excluded_skill_indices[i] is not None and 
                    excluded_skill_indices[i] != -1 and
                    skill_indices[j] == excluded_skill_indices[i]):
                    mask[i, j] = False
                    
        return mask
    
    def _create_masks_with_monitoring(
        self, 
        batch_size: int,
        skill_indices: List[int],
        excluded_skill_indices: List[Optional[int]],
        strategies: List[str],
        pre_selected_negatives: List[List[str]]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Create masks with additional monitoring information.
        Only used when enable_monitoring=True.
        """
        
        # Create the mask
        mask = self._create_in_batch_negative_masks(
            batch_size, skill_indices, excluded_skill_indices, strategies
        )
        
        # Calculate monitoring info
        valid_in_batch_counts = mask.sum(dim=1)
        
        # Count pre-selected negatives
        pre_selected_counts = torch.tensor([
            len(negs) if negs else 0 for negs in pre_selected_negatives
        ])
        
        # Calculate total negatives by strategy
        total_negatives = torch.zeros(batch_size, dtype=torch.long)
        for i, strategy in enumerate(strategies):
            if strategy == "in_batch":
                total_negatives[i] = valid_in_batch_counts[i]
            elif strategy in ["esco"]:
                total_negatives[i] = pre_selected_counts[i]
            elif strategy in ["mixed_batch_esco"]:
                total_negatives[i] = valid_in_batch_counts[i] + pre_selected_counts[i]
        
        monitoring_info = {
            "valid_in_batch_counts": valid_in_batch_counts,
            "pre_selected_counts": pre_selected_counts,
            "total_negative_counts": total_negatives,
            "stats": {
                "min_in_batch": valid_in_batch_counts.min().item() if valid_in_batch_counts.numel() > 0 else 0,
                "max_in_batch": valid_in_batch_counts.max().item() if valid_in_batch_counts.numel() > 0 else 0,
                "avg_in_batch": valid_in_batch_counts.float().mean().item() if valid_in_batch_counts.numel() > 0 else 0,
                "min_total": total_negatives.min().item(),
                "max_total": total_negatives.max().item(),
                "avg_total": total_negatives.float().mean().item()
            }
        }
        
        # Log warnings if enabled
        if self.log_warnings:
            min_in_batch = monitoring_info["stats"]["min_in_batch"]
            if min_in_batch < 10 and min_in_batch > 0:
                print(f"Warning: Some samples have only {min_in_batch} valid in-batch negatives")
                
        return mask, monitoring_info

    def _tokenize_pre_selected_negatives(
        self,
        pre_selected_negatives: List[List[str]],
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """Tokenize pre-selected negatives with padding."""
        
        all_negatives, max_negatives = [], 0
        
        for negatives in pre_selected_negatives:
            if negatives:
                all_negatives.append(negatives)
                max_negatives = max(max_negatives, len(negatives))
            else:
                all_negatives.append([])
                
        if max_negatives == 0:
            return {}
        
        # Pad negative lists to same length
        padded_negatives = []
        for negatives in all_negatives:
            current_negatives = negatives.copy()
            if len(current_negatives) < max_negatives:
                padding_needed = max_negatives - len(current_negatives)
                current_negatives.extend([""] * padding_needed)
            padded_negatives.extend(current_negatives)
            
        # Tokenize all negatives
        negative_encodings = self.tokenizer(
            padded_negatives,
            max_length=self.max_length_label,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors
        )
        
        # Reshape to [batch_size, max_negatives, seq_length]
        seq_length = negative_encodings["input_ids"].size(1)
        
        return {
            "input_ids": negative_encodings["input_ids"].view(batch_size, max_negatives, seq_length),
            "attention_mask": negative_encodings["attention_mask"].view(batch_size, max_negatives, seq_length)
        }