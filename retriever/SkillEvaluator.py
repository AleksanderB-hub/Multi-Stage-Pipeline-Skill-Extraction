from typing import List, Dict, Set, Tuple, Optional, Any
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import json
from collections import defaultdict
import random 

class SkillEvaluator:
    """
    Computes retrieval metrics.
    Works for both validation during training and final evaluation.
    """
    
    def __init__(
        self,
        model, 
        definition_to_label: Dict[str, str],
        device="cuda", 
        batch_size_defs=256
        ):
        self.model = model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size_defs = batch_size_defs
        
        self.definitions  = list(definition_to_label.keys())
        self.skill_labels = list(definition_to_label.values())

        # will be re-built whenever the model refreshes
        self.definition_embeddings: Optional[torch.Tensor] = None

    def refresh_model(self, model) -> None:
        """Point evaluator at new model weights and clear caches."""
        self.model = model
        self.definition_embeddings = None

    def evaluate(
        self,
        val_data: List[Dict],
        k: int = 5,
        return_preds: bool = False,
        output_path: Optional[str] = None,
        zero_shot_mode: bool = False,
        excluded_labels: Optional[Set[str]] = None
    ):
        """
        Computes R-Precision@k and MRR. If `return_preds` is True, the returned
        dict will have an extra key "predictions" with the top-k
        lists per sentence (no extra matmul cost).
        If `output_path` is provided, predictions are saved to JSON.
    
        Zero-shot evaluation parameters:
        - excluded_labels: Labels to test zero-shot (kept in evaluation)
        - zero_shot_mode: If True, applies Strategy 2 filtering
        """

        val_data = self._group_labels_by_sentence(val_data)
   
        # Apply Strategy 2 filtering if zero-shot mode is enabled
        if zero_shot_mode:
            if excluded_labels is None:
                raise ValueError("excluded_labels must be provided for zero_shot_mode")
    
            # Automatically infer trained labels
            print(f" The labels are {len(self.skill_labels)} (full train)")
            print(f" The labels are {len(excluded_labels)} (excluded)")
            trained_labels = set(self.skill_labels) - excluded_labels
            print(f" The labels are {len(trained_labels)} (control)")
            filtered_data, filter_stats = self._apply_strategy2_filtering(
                val_data, excluded_labels, trained_labels
            )
    
            print(f"Zero-shot filtering applied:")
            print(f"  Original examples: {filter_stats['original_examples']}")
            print(f"  Examples kept: {filter_stats['examples_kept']}")
            print(f"  Examples removed (no excluded labels): {filter_stats['examples_removed_empty']}")
            print(f"  Total excluded labels in test set: {filter_stats['total_excluded_labels']}")
    
            # Use filtered data for evaluation
            eval_data = filtered_data
            queries = [sample["sentence"] for sample in eval_data]
            gold_labels = [set(sample["filtered_labels"]) for sample in eval_data]
        else:
            # Standard evaluation
            eval_data = val_data
            queries = [sample["sentence"] for sample in eval_data]
            gold_labels = [set(label for label in sample["labels"] if label not in ["UNDERSPECIFIED", 'UNK']) for sample in eval_data]

        retrieve_texts = self.skill_labels
    
        # Encode once
        self.definition_embeddings = self._encode_texts(retrieve_texts, "labels", self.device)
        query_embeddings = self._encode_texts(queries, "queries", self.device)
    
        # Normalize
        self.definition_embeddings = F.normalize(self.definition_embeddings, dim=-1)
        query_embeddings = F.normalize(query_embeddings, dim=-1)
    
        # Compute cosine similarity
        sim_matrix = torch.matmul(query_embeddings, self.definition_embeddings.T)
    
        scores = []
        mrr_scores = []
        preds_all = [] if return_preds else None
    
        for i in range(len(queries)):
            sims = sim_matrix[i]
            top_idx = torch.topk(sims, k).indices
            top_idx_other = torch.topk(sims, 100).indices
            predicted_labels = [self.skill_labels[j.item()] for j in top_idx]
            predicted_labels_other = [self.skill_labels[j.item()] for j in top_idx_other]

            if zero_shot_mode:
                # Only consider relevant labels that were excluded from training
                relevant_labels = [label for label in gold_labels[i] if label in excluded_labels]

                if not relevant_labels:  # Skip queries with no excluded gold labels
                    continue
            else:
                relevant_labels = gold_labels[i]
                
            # Calculate R-Precision@k
            rprecision = 0.0
            if not relevant_labels:
                continue
            if len(predicted_labels) > 0:
                relevant_in_topk = sum(1 for label in predicted_labels if label in relevant_labels)
                if len(relevant_labels) >= k:
                    rprecision = relevant_in_topk / k
                else:
                    rprecision = relevant_in_topk / len(relevant_labels)
            else:
                rprecision = 0.0
    
            scores.append(rprecision)
            
            # Calculate MRR
            mrr_score = 0.0
            if not relevant_labels: 
                continue
            if len(predicted_labels_other) > 0:
                for rank, pred_label in enumerate(predicted_labels_other, 1):
                    if pred_label in relevant_labels:
                        mrr_score = 1.0 / rank 
                        break  
            else:
                mrr_score = 0.0  

            mrr_scores.append(mrr_score)
    
            if return_preds:
                # Get scores for the filtered predictions
                if zero_shot_mode:
                    # Find indices of excluded labels only
                    excluded_indices = [j for j in top_idx if self.skill_labels[j.item()] in excluded_labels]
                    if excluded_indices:
                        top_scores = [sims[j].item() for j in excluded_indices]
                        pred_with_scores = list(zip(predicted_labels, top_scores))
                    else:
                        pred_with_scores = []
                else:
                    top_scores = sims[top_idx].cpu().tolist()
                    pred_with_scores = list(zip(predicted_labels, top_scores))
    
                pred_entry = {
                    "sentence": queries[i],
                    "predicted_labels": pred_with_scores,
                    "true_labels": list(relevant_labels),
                    "rprecision": rprecision,
                    "mrr": mrr_score
                }
    
                # Add extra info for zero-shot mode
                if zero_shot_mode:
                    pred_entry["original_labels"] = eval_data[i]["original_labels"]
                    pred_entry["removed_labels"] = eval_data[i]["removed_labels"]
    
                preds_all.append(pred_entry)
    
        # Prepare output with both metrics
        out = {
            f"rprecision@{k}": float(np.mean(scores)),
            f"mrr": float(np.mean(mrr_scores))
        }
    
        if zero_shot_mode:
            out["zero_shot_stats"] = filter_stats
    
        if return_preds:
            out["predictions"] = preds_all
    
        if return_preds and output_path:
            with open(output_path, "w") as f:
                json.dump(preds_all, f, indent=2)
    
        return out
    
    
    def _apply_strategy2_filtering(
        self,
        data: List[Dict], 
        excluded_labels: Set[str], 
        trained_labels: Set[str]
    ) -> Tuple[List[Dict], Dict]:
        """
        Apply Strategy 2 filtering: keep examples with excluded labels, 
        remove trained labels from evaluation
        """

        filtered_data = []
        stats = {
            'original_examples': len(data),
            'examples_with_excluded': 0,
            'examples_kept': 0,
            'examples_removed_empty': 0,
            'total_excluded_labels': 0,
            'total_trained_labels_removed': 0
        }

        for item in data:
            # Filter out UNDERSPECIFIED and UNK first
            clean_labels = [label for label in item['labels'] if label not in ["UNDERSPECIFIED", 'UNK']]

            # Separate excluded and trained labels
            item_excluded = [label for label in clean_labels if label in excluded_labels]
            item_trained = [label for label in clean_labels if label in trained_labels]

            # Count examples that originally had excluded labels
            if item_excluded:
                stats['examples_with_excluded'] += 1

            # Only keep examples that have at least one excluded label
            if len(item_excluded) > 0:
                filtered_data.append({
                    'sentence': item['sentence'],
                    'original_labels': clean_labels,
                    'filtered_labels': item_excluded,  # Only excluded labels (gold standard)
                    'removed_labels': item_trained,    # Trained labels (for reference)
                    'labels': item_excluded  # For compatibility with existing code
                })
                stats['examples_kept'] += 1
                stats['total_excluded_labels'] += len(item_excluded)
                stats['total_trained_labels_removed'] += len(item_trained)
            else:
                stats['examples_removed_empty'] += 1

        return filtered_data, stats

    def _encode_texts(self, texts, desc, device):
        self.model.eval()
        results = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size_defs),
                          desc=f"Encoding {desc}",
                          disable=self.device != "cuda"):
                batch = texts[i : i + self.batch_size_defs]
                emb = self.model.encode(batch, convert_to_tensor=True,
                                        device=device, show_progress_bar=False)
                emb = F.normalize(emb, p=2, dim=-1)
                results.append(emb)
        return torch.cat(results, dim=0).to(device)

    def _group_labels_by_sentence(self, dataset):

        if any('labels' in d for d in dataset):
            return dataset
        else:
            sentence_to_labels = defaultdict(set)

            # First, iterate through all rows in the dataset
            for row in dataset:
                sentence = row['sentence']
                label = row['label']
                if label != 'LABEL NOT PRESENT':
                    sentence_to_labels[sentence].add(label)
                else:
                    sentence_to_labels[sentence]  # ensure sentence is recorded even without label

            # Now construct the final list of dictionaries
            grouped_data = [
                {'sentence': sentence, 'labels': sorted(list(labels))}
                for sentence, labels in sentence_to_labels.items()
            ]

        return grouped_data

        
        
    def process_data(data, max_candidates=20):
        dict_new = []
        for item in data:
            sentence = item['sentence']
            true_labels = set(item['true_labels'])
            predicted_labels = [label for label, _ in item['predicted_labels']]
            # Ensure all true_labels are present
            candidates = list(true_labels)
            # Add top predicted labels not in true_labels
            for label in predicted_labels:
                if label not in true_labels:
                    candidates.append(label)
                if len(candidates) >= max_candidates:
                    break  # stop adding if we reach the limit
                
            # Shuffle the candidate labels
            random.shuffle(candidates)
            dict_new.append({
                "sentence": sentence,
                "true_labels": list(true_labels),
                "candidate_labels": candidates
            })
        return dict_new
    
    def process_test_data(data):
        dict_new = []
        for item in data:
            sentence = item['sentence']
            true_labels = set(item['true_labels'])
            predicted_labels = [label for label, _ in item['predicted_labels']]
            dict_new.append({
                "sentence": sentence,
                "true_labels": list(true_labels),
                "candidate_labels": predicted_labels
            })
        return dict_new