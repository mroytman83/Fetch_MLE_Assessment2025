# Sentence Transformer Implementation - Writeup

### Task 1

Instead of training token embeddings from scratch, I used a pre-trained BERT model through the transformers library maintained by hugging face since it is considered a deep learning library. BERT has already been trained on vast amounts of text data, allowing it to produce contextually rich representations right away instead of using nn. Embedding would require extensive training(I don't have the hardware, haha).

To generate a fixed-length sentence representation, I experimented with different pooling strategies, and a combined strategy yielded the best results. Using just the CLS token sometimes misses the critical context, and on the other hand, simple mean pooling diluted key information. So, I combined CLS and mean pooling, which kept important features while preserving overall meaning. Layer normalization/residual connections would stabilize training(future tasks) and allow deeper layers to retain earlier information, solving potential struggles with gradient issues and making training less effective.

### Task 2

For improved efficiency and generalization across multiple NLP tasks, I implemented hard parameter sharing with two task-specific layers. I wanted to reduce redundancy while the layers of sentence classification and sentiment analysis allow the model to specialize in each task. Using dropout regularization to prevent overfitting by randomly deactivating neurons during training helped the model generalize better instead of memorizing patterns. I also chose mean pooling to average all token embeddings, making it more robust for varying sentence lengths. The ReLU activation keeps computations efficient by introducing non-linearity, while sigmoid ensures sentiment outputs are between 0 and 1, making them interpretable as probabilities.


### Task 3

##### Training Considerations for My Sentence Transformer Architecture

1. If the entire network should be frozen
Freezing the entire network means that none of the model’s parameters will be updated during training. This would drastically reduce training time/ computational cost but would come with major issues. Since my architecture includes task-specific layers for sentence classification and sentiment analysis, freezing everything would prevent these layers from adapting to new data, yielding poor in-context performance. Freezing is only useful when using the model as a feature extractor in scenarios where labels are scarce, but it shouldn't be done for fine-tuning on new tasks. 

2. If only the transformer backbone should be frozen
Freezing only the transformer backbone while training the task-specific heads is a more balanced approach. The advantage is that it retains BERT’s pre-trained language knowledge while allowing the task-specific layers to adapt to my dataset. This speeds up training and avoids catastrophic forgetting, making it a great choice when I have a small labeled dataset or when the pre-trained transformer is already well-suited for my tasks. However, if my dataset is domain-specific (e.g., medical or legal text), I might need to unfreeze some of the later transformer layers to help the model adapt.

3. If only one of the task-specific heads (either for Task A or Task B) should be frozen
If I freeze one, I assume that one head is already well-trained and does not require further updates, while the other needs fine-tuning. This can be beneficial if the two tasks are related but not identical, like if they are within the same task level, such as token or sentence level. For example, if I trained the sentiment analysis head on a massive dataset but sentence classification is new, I could freeze the sentiment head and only fine-tune the classification head. However, if the tasks are closely linked, it might be better to fine-tune both heads together.

##### Transfer Learning Approach

1. Choice of a Pretrained Model
Since I am working with general sentence embeddings and classification tasks, my best option is BERT-base-uncased or RoBERTa-base, as they are already trained on massive corpora like Wikipedia and BookCorpus. If my tasks require niche knowledge, I will use a pre-trained model on that domain to get better initial representations.

2. Layers to Freeze/Unfreeze
I would start by freezing the transformer backbone and training only the task-specific layers. This ensures that I leverage BERT’s knowledge without overfitting too quickly. Gradually unfreeze the later transformer layers if my dataset is domain-specific or significantly different from BERT’s training data. Keep the initial transformer layers frozen because they capture general language representations that are transferable across tasks.

3. Rationale Behind These Choices
If I were to freeze early layers, I would prevent unnecessary updates to useful general features. Training task-specific heads first ensures they learn from BERT’s fixed representations before modifying them. Unfreezing later layers selectively allows the model to adapt without losing its pre-trained knowledge. A domain-specific model would avoid excessive fine-tuning and improve results when working with specialized text.

### Task 4
For task 4, the hard parameter sharing architecture with two task-specific heads ensures that general language features learned by the transformer are leveraged effectively for different tasks while keeping task-specific learning distinct. Since I didn’t have a real dataset, I simulated tokenized inputs, set attention masks to assume no padding, and created fake labels for both tasks. The model processes each batch through the transformer backbone, a pooling layer, and the task-specific heads. Loss is computed using CrossEntropyLoss for classification and BCELoss for sentiment analysis, summing both for balanced training. I evaluated accuracy with softmax for classification and thresholding sigmoid outputs at 0.5 for sentiment labels, ensuring an efficient and adaptable model.