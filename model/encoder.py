from model.positional_embeddings import TokenPositionEmbeddings
from model.attention import GlobalSelfAttention, FeedForward
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self,d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def forward(self, x):
        x, attn_scores = self.self_attention(x)  
        x = self.ffn(x) 
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, model_name="bert-base-uncased", dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers

        # pretrained BERT embeddings
        self.token_embedding = TokenPositionEmbeddings(model_name=model_name)

        # stack ncoder layers
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the Transformer Encoder"""
        x = self.token_embedding(input_ids, attention_mask) 
        #dropout
        x = self.dropout(x)  
        
        #pass through encoder layers
        for layer in self.enc_layers:
            x = layer(x)

        return x