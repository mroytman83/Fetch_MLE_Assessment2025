from model.encoder import Encoder
from model.pooling import PoolingLayer
from model.utils import pytorch_cos_sim
from model.multitask import SentenceTransformerMultiTask
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import pandas as pd
import time

#tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#multi-task model
model = SentenceTransformerMultiTask()

#testing dataset for Task1
splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
task1_df = pd.read_parquet("hf://datasets/sentence-transformers/stsb/" + splits["train"])

def task1_eval():

    # Tokenize Inputs
    tokens = tokenizer(

        [
            task1_df["sentence1"][0],
            task1_df["sentence2"][0]
        ],

        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    encoder = Encoder(num_layers=6, d_model=768, num_heads=8, dff=2048, model_name="bert-base-uncased")
    pooling = PoolingLayer(pooling_type="mean")


    hidden_states = encoder(tokens["input_ids"], tokens["attention_mask"])
    pooled_output = pooling(hidden_states, tokens["attention_mask"])
    #testing embeddings 
    cos_sim = pytorch_cos_sim(pooled_output[0].unsqueeze(0), pooled_output[1].unsqueeze(0))
    print("Evaluating Task1")
    return ("Cosine Similarity:", cos_sim.item())

def task2_eval():
    answ_dict={}

    inputs = tokenizer(["This is a great day!", "I hate the weather today!"], padding=True, truncation=True, return_tensors="pt")

    classification_output, sentiment_output = model(inputs["input_ids"], inputs["attention_mask"])

    answ_dict["Sentence Classification Output:"]=classification_output
    answ_dict["Sentiment Analysis Output:"]=sentiment_output

    print("Evaluating Task2")
    return answ_dict

def task4_eval():
    print("Evaluating Task4")
    #optimizer for gradient descent, handles weight decay better than Adam
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn_classification = nn.CrossEntropyLoss()  
    #two categories
    loss_fn_sentiment = nn.BCELoss()  

    # Hypothetical dataset (batch of 4 samples)
    input_ids = torch.randint(0, 30522, (4, 128))  
    attention_mask = torch.ones((4, 128))  
    labels_classification = torch.tensor([0, 1, 1, 0])  
    labels_sentiment = torch.tensor([1.0, 0.0, 1.0, 0.0]).unsqueeze(1)  

    dataset = TensorDataset(input_ids, attention_mask, labels_classification, labels_sentiment)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    #just 1 epoch for simulation
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()

        batch_input_ids, batch_attention_mask, batch_labels_class, batch_labels_sent = batch

        #forward
        classification_output, sentiment_output = model(batch_input_ids, batch_attention_mask)

        #loss computation
        loss_classification = loss_fn_classification(classification_output, batch_labels_class)
        loss_sentiment = loss_fn_sentiment(sentiment_output, batch_labels_sent)

        #loss aggregation 
        total_loss = loss_classification + loss_sentiment

        # backpropagation
        total_loss.backward()
        optimizer.step()

        print(f"Loss - Classification: {loss_classification.item():.4f}, Sentiment: {loss_sentiment.item():.4f}, Total: {total_loss.item():.4f}")

if __name__ == '__main__':
    task1_eval()
    time.sleep(5)
    task2_eval()
    time.sleep(5)
    task4_eval()
