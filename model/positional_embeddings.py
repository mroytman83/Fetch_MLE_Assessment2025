from transformers import BertTokenizer, BertModel
import torch.nn as nn

class TokenPositionEmbeddings(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name) 

    def forward(self, input_ids, attention_mask):
        """Contextualized token embeddings from BERT"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state 