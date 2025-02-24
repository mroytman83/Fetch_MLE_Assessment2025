import torch.nn as nn
from model.encoder import Encoder
from model.pooling import PoolingLayer

class SentenceTransformerMultiTask(nn.Module):
    def __init__(self, num_layers=6, d_model=768, num_heads=8, dff=2048, hidden_size=256, num_classes=2, model_name="bert-base-uncased", dropout_rate=0.1):
        """
        Hard parameter sharing for multi-task learning
        One layer is sentence classification, the layer is sentiment analysis
        """
        super().__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, model_name, dropout_rate)
        self.pooling = PoolingLayer(pooling_type="mean")

        #classification layer
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

        #sentiment layer, similar to classification, just uses sigmoid to map log probs
        self.sentiment_head = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids, attention_mask)
        pooled_output = self.pooling(x, attention_mask)

        classification_output = self.classification_head(pooled_output)
        sentiment_output = self.sentiment_head(pooled_output)

        return classification_output, sentiment_output