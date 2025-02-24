import torch.nn as nn


class PoolingLayer(nn.Module):
    def __init__(self, pooling_type="mean"):
        super().__init__()
        assert pooling_type in ["mean", "max", "cls"], "Invalid pooling type!"
        self.pooling_type = pooling_type

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: tensor (batch_size, seq_len, d_model)
        attention_mask: tensor  (batch_size, seq_len), 1 for valid tokens, 0 for padding (optional)
        final answ: pooled sentence embeddings of (batch_size, d_model)
        """
        if attention_mask is not None:
          # shape: (batch_size, seq_len, 1)
          attention_mask = attention_mask.unsqueeze(-1)  
          # mask padding
          hidden_states = hidden_states * attention_mask  
           # mean pooling
          return hidden_states.sum(dim=1) / attention_mask.sum(dim=1) 
        else:
          return hidden_states.mean(dim=1)