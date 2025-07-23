## LLMS for toxicity classification


## Dataset used

I used the Toxicity Comments Dataset from the Toxic Comment Classification Challenge - https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fc%2Fjigsaw-toxic-comment-classification-challenge%2Fdata  

# Baseline Model Testing

* Why a sequence classifier could not be used? *

Dataset constraints - Here I saw that there were multiple classes in this dataset beyond just "toxic". 
Hence, it was a multi-label classification problem. That is, multiple nonexclusive labels (or none at all) may be assigned to each instance. Hence, I realized that while building our classifier, we will need to use a model capable of producing multiple labels for each input.

Initially, I used the 'roberta-base' model but realized that this model is only used for sequence classification. Hence, this was only used to give me the predicted probablities for each toxicity label for any sample sentence.






