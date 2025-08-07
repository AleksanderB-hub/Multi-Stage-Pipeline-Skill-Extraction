import logging
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from typing import List, Dict, Set, Tuple, Optional, Union, Any
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class ContrastiveSkillEncoder(nn.Module):
    """
    This class contains the actual contrastive model.
    The model utilises job description sentences and the skill labels.
    The architecture is as follows:
    1. Pre-trained Transformer (e.g, MPNet)
    2. Pooling layer to get sentence embeddings 
    3. Optional Projection Head
    4. L2 normalization
    """

    def __init__(self, args):
        super().__init__()

        self.model_name = args.model_name
        self.dropout_rate = getattr(args, 'dropout_rate', 0.1)
        self.pooling_strategy = getattr(args, 'pooling_strategy', 'mean')
        self.logger = logging.getLogger(__name__)
        # Load the pre-trained encoder
        logging.info(f"Loading pre-trained model: {self.model_name}")
        self.encoder = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.output_dim = self.hidden_size

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_dict: bool=False,
                ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the encoder.

        Args:
            input_ids: Token IDs from tokenizer [batch_size, seq_length]
            attention_mask: Attention mask (1 for real tokens, 0 for padding) [batch_size, seq_length]
            return_dict: If True, return dictionary with additional info
            
        Returns:
            If return_dict=False: Normalized embeddings [batch_size, output_dim]
            If return_dict=True: Dictionary with embeddings and intermediate outputs
        """

        # get the encodings
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # get token embeddings [batch_size, seq_length, hidden_size]
        token_emb = encoder_outputs.last_hidden_state
        # pool to get the sentence embedding
        sentence_emb = self._pool_embeddings(token_emb, attention_mask)

        # since we use AutoModel we need to manually normalize 
        normalized_emb = F.normalize(sentence_emb, dim=1)

        # for monitoring
        if return_dict:
            return {
                'embeddings': normalized_emb,
                'pooled_embeddings': sentence_emb, 
                'token_embeddings': token_emb,  
            }
        
        return normalized_emb

    def _pool_embeddings(self,
                         token_embeddings: torch.Tensor,
                         attention_mask: torch.Tensor
                         ) -> torch.Tensor:
        """
        Pool token embeddings into a single sentence embedding.
        
        Args:
            token_embeddings: All token embeddings [batch_size, seq_length, hidden_size]
            attention_mask: Mask indicating real tokens [batch_size, seq_length]
            
        Returns:
            Pooled sentence embeddings [batch_size, hidden_size]
        """
        if self.pooling_strategy == 'mean':
            # Expand mask to match embedding dimensions
            # [batch_size, seq_length] -> [batch_size, seq_length, hidden_size]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            # Sum embeddings for non-padding tokens
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            
            # Count non-padding tokens for each sentence
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            
            # Compute mean by dividing sum by count
            return sum_embeddings / sum_mask
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def encode(self,
               sentences: List[str],
               batch_size: int = 256,
               convert_to_tensor: bool = True,
               show_progress_bar: bool = True,
               device: Optional[torch.device] = None
               ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode a list of sentences into embeddings.
        
        This method is used by the FAISS sampler and during inference.
    
        Args:
            sentences: List of strings to encode
            batch_size: Batch size to use (falls back to self.batch_size or 32)
            convert_to_tensor: Return tensor (True) or numpy array (False)
            show_progress_bar: Whether to show tqdm bar
            device: Device to use for encoding
    
        Returns:
            Embeddings as tensor or numpy array [num_sentences, output_dim]
        """
    
        if device is None:
            device = next(self.parameters()).device
    
        self.eval()
        all_embeddings = []
    
        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding Sentences", total=(len(sentences) // batch_size))
    
        with torch.no_grad():
            for start_idx in iterator:
                batch_sentences = sentences[start_idx:start_idx + batch_size]
    
                enc = self.tokenizer(
                    batch_sentences,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt'
                )
    
                enc = {k: v.to(device) for k, v in enc.items()}
    
                emb = self.forward(
                    input_ids=enc['input_ids'],
                    attention_mask=enc['attention_mask'],
                    return_dict=False
                )
                all_embeddings.append(emb.cpu())
    
        all_embeddings = torch.cat(all_embeddings, dim=0)
    
        return all_embeddings if convert_to_tensor else all_embeddings.numpy()
        
    def get_embedding_dimension(self) -> int:
        """Get the output embedding dimension."""
        return self.output_dim
    
    def save_pretrained(self, save_path: str):
        """
        Save model weights and configuration.
        
        Saves:
        - Model weights
        - Configuration (pooling strategy, etc.)
        - Base encoder and tokenizer for easy loading
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights and config
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'dropout_rate': self.dropout_rate,
            'pooling_strategy': self.pooling_strategy,
            'output_dim': self.output_dim,
            'hidden_size': self.hidden_size,
        }, os.path.join(save_path, 'pytorch_model.bin'))
        
        # Save config as JSON for easy inspection
        import json
        config = {
            'model_name': self.model_name,
            'dropout_rate': self.dropout_rate,
            'pooling_strategy': self.pooling_strategy,
            'output_dim': self.output_dim,
            'hidden_size': self.hidden_size,
        }
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Model saved to {save_path}")    

    @classmethod
    def from_pretrained(cls, load_path: str, args=None):
        """
        Load model from saved checkpoint.
        
        Args:
            load_path: Path to saved model
            args: Optional args object (will use saved config if None)
            
        Returns:
            Loaded model instance
        """
        import os
        import json
        
        # Load configuration
        with open(os.path.join(load_path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Create args-like object if not provided
        if args is None:
            class Args:
                pass
            args = Args()
            for key, value in config.items():
                setattr(args, key, value)
        
        # Initialize model
        model = cls(args)
        
        # Load weights
        checkpoint = torch.load(os.path.join(load_path, 'pytorch_model.bin'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model

def create_model(args):
    """
    This function creates a model instance used for training. 
    """
    model = ContrastiveSkillEncoder(args)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model initialized: {args.model_name}")
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Output dimension: {model.output_dim}")
    
    return model