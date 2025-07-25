## LLMS for toxicity classification


## Dataset used

I used the Toxicity Comments Dataset from the Toxic Comment Classification Challenge - https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fc%2Fjigsaw-toxic-comment-classification-challenge%2Fdata  

# How to decide which LLM model to use based on the structure of the data?

Dataset constraints - Here I saw that there were multiple classes in this dataset beyond just "toxic". 
Hence, it was a multi-label classification problem. That is, labels are not mutually exclusive. Hence, I realized that while building our classifier, we will need to use a model capable of producing multiple labels for each input.

# LLM models explored

## RoBERTa-base model

I used the 'roberta-base' model but realized that this model is only used for sequence classification. Hence, this was only used to give me the predicted probablities for each toxicity label for any sample sentence.

Few important characteristics of the RoBERTa-base model
1. Type - Encoder only (BERT-style)
2. Pretraining Task - Masked Language Modeling (MLM)
3. Primary use case - Classification, feature extraction
4. Input format - Just the raw text
5. Ouput - Predicts logits over class labels
6. Tokernizer - 	Byte-Pair Encoding (BPE)
7. Uses final hidden states for label logits
8. Natural fit (logits per label + sigmoid + BCE)
9. Target = binary vector like [1, 0, 1, 0, 0, 1]
10. Training time- Low (just classification head)
11. Inference time - Single forward pass
12. Model size - 125M parameters
   
## 






