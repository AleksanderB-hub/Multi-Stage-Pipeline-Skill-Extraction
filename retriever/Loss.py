import torch
import torch.nn as nn
from typing import List, Dict, Set, Tuple, Optional, Any
import torch.nn.functional as F
class NTXent(nn.Module):
    """A variant of InfoNCE loss accepting both in-batch negatives
    as well as pre-selected hard negatives. 
    """
    
    def __init__(self, args):
        super(NTXent, self).__init__()
        self.reduction = args.reduction
        self.hard_negative_strategy = args.hard_negative_strategy
        
        if args.pre_train:
            self.temperature = args.temperature_pre_train
        else:
            self.temperature = args.temperature_main
        
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
        in_batch_negatives: bool = None,
        in_batch_negative_masks: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        """
        Compute NTXent loss

        Args:
            anchor_embeddings: [batch_size, embedding_dim]
            positive_embeddings: [batch_size, embedding_dim]
            negative_embeddings: [batch_size, num_negatives, embedding_dim] or None
            in_batch_negatives: Whether to use in-batch negatives
            in_batch_negative_masks: [batch_size, batch_size] mask for valid in-batch negatives

        Returns:
            Dict containing loss and metrics
        """
        batch_size = anchor_embeddings.size(0)
        device = anchor_embeddings.device
        
        # auto-detect strategies utilising in-batch negatives
        if in_batch_negatives is None:
            in_batch_strategies = [
                "in_batch", 
                "mixed_batch_hard", 
                "mixed_batch_esco", 
                "mixed_all"
            ]
            # a boolean for strategies that use in-batch negatives
            in_batch_negatives = self.hard_negative_strategy in in_batch_strategies
        
        # Normalize embeddings and compute cosine similarity
        anchor_embeddings = F.normalize(anchor_embeddings, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, dim=1)
        if negative_embeddings is not None:
            negative_embeddings = F.normalize(negative_embeddings, dim=2)
            
        # compute similarities
        positive_similarities = F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=1)
        # adjust by temperature 
        positive_similarities = positive_similarities / self.temperature 
        
        # now for the negative similarities
        all_negative_similarities = []
        if in_batch_negatives:
            # use all other positives in batch as negatives
            batch_similarities = torch.matmul(anchor_embeddings, positive_embeddings.T)
            batch_similarities = batch_similarities / self.temperature
            
            # masking for augmented sentences
            if in_batch_negative_masks is not None:
                batch_negative_similarities = batch_similarities.masked_fill(
                    ~in_batch_negative_masks, float('-inf')
                )
            else:
                # mask out positive pairs (over diagonal)
                mask = torch.eye(batch_size, device=device, dtype=torch.bool)
                batch_negative_similarities = batch_similarities.masked_fill(mask, float('-inf'))
                
            all_negative_similarities.append(batch_negative_similarities)
            
        # in case of hard negative sampling
        if negative_embeddings is not None:
            # negative_embeddings: [batch_size, num_negatives, embedding_dim]
            hard_negative_similarities = torch.bmm(
                anchor_embeddings.unsqueeze(1), # [batch_size, 1, embedding_dim]
                negative_embeddings.transpose(1, 2) # [batch_size, embedding_dim, num_negatives]
            ).squeeze(1) # [batch_size, num_negatives]
            
            hard_negative_similarities = hard_negative_similarities / self.temperature
            all_negative_similarities.append(hard_negative_similarities)
            
        # combine all negative similarities
        if all_negative_similarities:
            negative_similarities = torch.cat(all_negative_similarities, dim=1)
        else:
            raise ValueError("No negatives provided (either in_batch_negatives=True or negative_embeddings must be provided)")
        
        # compute the loss
        numerator = torch.exp(positive_similarities) # exp(positive_similarity)
        all_similarities = torch.cat([
            positive_similarities.unsqueeze(1),
            negative_similarities
        ], dim=1)
        
        denominator = torch.sum(torch.exp(all_similarities), dim=1) # sum of exp(all_similarities)

        loss = -torch.log(numerator / denominator)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
    
        return loss
        
class MarginLoss(nn.Module):
    """
    Margin-based contrastive loss for comparison with NT-Xent
    """
    
    def __init__(self, args):
        super(MarginLoss, self).__init__()
        self.margin = args.margin
        self.reduction = args.reduction
        self.hard_negative_strategy = args.hard_negative_strategy

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
        in_batch_negatives: bool = None,
        in_batch_negative_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # Changed return type
        """
        Compute Margin loss

        Args:
            anchor_embeddings: [batch_size, embedding_dim]
            positive_embeddings: [batch_size, embedding_dim]
            negative_embeddings: [batch_size, num_negatives, embedding_dim] or None
            in_batch_negatives: Whether to use in-batch negatives
            in_batch_negative_masks: [batch_size, batch_size] mask for in-batch negatives

        Returns:
            loss tensor
        """
        batch_size = anchor_embeddings.size(0)
        device = anchor_embeddings.device
        
        # Auto-detect strategies utilizing in-batch negatives
        if in_batch_negatives is None:
            in_batch_strategies = [
                "in_batch", 
                "mixed_batch_hard", 
                "mixed_batch_esco", 
                "mixed_all"
            ]
            in_batch_negatives = self.hard_negative_strategy in in_batch_strategies
            
        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, dim=1)
        if negative_embeddings is not None:
            negative_embeddings = F.normalize(negative_embeddings, dim=2)
        
        # Compute positive similarities
        positive_similarities = F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=1)
        
        # Collect all negative similarities
        all_negative_similarities = []
        
        # In-batch negatives
        if in_batch_negatives:
            batch_similarities = torch.matmul(anchor_embeddings, positive_embeddings.T)
            
            # Apply masking for augmented sentences
            if in_batch_negative_masks is not None:
                batch_negative_similarities = batch_similarities.masked_fill(
                    ~in_batch_negative_masks, float('-inf')
                )
            else:
                # Default: mask out positive pairs (diagonal)
                mask = torch.eye(batch_size, device=device, dtype=torch.bool)
                batch_negative_similarities = batch_similarities.masked_fill(mask, float('-inf'))
                
            all_negative_similarities.append(batch_negative_similarities)    
            
        # Hard/ESCO negatives
        if negative_embeddings is not None:
            hard_negative_similarities = torch.bmm(
                anchor_embeddings.unsqueeze(1),  # [batch_size, 1, embedding_dim]
                negative_embeddings.transpose(1, 2)  # [batch_size, embedding_dim, num_negatives]
            ).squeeze(1)  # [batch_size, num_negatives]
    
            all_negative_similarities.append(hard_negative_similarities)
        
        # Combine all negative similarities
        if all_negative_similarities:
            negative_similarities = torch.cat(all_negative_similarities, dim=1)
        else:
            raise ValueError("No negatives provided (either in_batch_negatives=True or negative_embeddings must be provided)")
        
        # Filter out invalid negatives (inf values)
        valid_negatives_mask = negative_similarities != float('-inf')
        
        # Margin loss: max(0, margin - pos_sim + neg_sim)
        positive_similarities_expanded = positive_similarities.unsqueeze(1)  # [batch_size, 1]
        margin_losses = torch.clamp(
            self.margin - positive_similarities_expanded + negative_similarities, 
            min=0.0
        )
        
        # Only average over valid negatives
        if valid_negatives_mask.any(dim=1).all():
            # Average over negatives per sample
            loss_per_sample = (margin_losses * valid_negatives_mask.float()).sum(dim=1) / valid_negatives_mask.float().sum(dim=1)
        else:
            # Fallback if some samples have no valid negatives
            loss_per_sample = margin_losses.mean(dim=1)
        
        # Reduce over batch
        if self.reduction == "mean":
            loss = loss_per_sample.mean()
        elif self.reduction == "sum":
            loss = loss_per_sample.sum()
        else:
            loss = loss_per_sample

        return loss
    
class SymmetricNTXent(nn.Module):
    """
    A very thin wrapper that calls your current NTXent loss twice
    (anchor➜positive  AND  positive➜anchor) and averages the results.
    """

    def __init__(self, base_loss: NTXent):
        """
        Args
        ----
        base_loss : an *instance* of your existing NTXent class,
                    already initialised with temperature, reduction,
                    hard-negative strategy, etc.
        """
        super().__init__()
        self.base_loss = base_loss              # reuse all its internals

    @torch.no_grad()
    def _transpose_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        # keep None as None; otherwise transpose in-batch mask so
        # the same anchor-positive pair is protected in the mirrored call
        return None if mask is None else mask.T

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        *,
        negative_embeddings: Optional[torch.Tensor] = None,
        in_batch_negative_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # original direction: anchor ➜ positive
        loss_ab = self.base_loss(
            anchor_embeddings        = anchor_embeddings,
            positive_embeddings      = positive_embeddings,
            negative_embeddings      = negative_embeddings,
            in_batch_negative_masks  = in_batch_negative_masks,
        )

        # mirrored direction: positive ➜ anchor
        loss_ba = self.base_loss(
            anchor_embeddings        = positive_embeddings,          # <─ swapped
            positive_embeddings      = anchor_embeddings,
            negative_embeddings      = negative_embeddings,          #  same tensor
            in_batch_negative_masks  = self._transpose_mask(in_batch_negative_masks),
        )

        # average keeps scale identical to the original loss
        return 0.5 * (loss_ab + loss_ba)
        
class ContrastiveLossWrapper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        loss_type = args.loss_type.lower()
        negative_sampling = args.hard_negative_strategy
        if loss_type == "ntxent" and negative_sampling != 'in_batch':
            self.loss_fn = NTXent(args)
        elif loss_type == "ntxent" and negative_sampling == 'in_batch':
            base_loss = NTXent(args)
            self.loss_fn = SymmetricNTXent(base_loss)
        elif loss_type == "margin":
            self.loss_fn = MarginLoss(args)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)