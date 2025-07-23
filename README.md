# Toxicity-Detector-using-LLMs

Developing a toxicity detector using large language models (LLMs) 

Baseline Model Testing:
Select a pre-trained LLM that seems appropriate for the task.
Evaluate the performance of this pre-trained model on a test set from a toxicity-related dataset. This involves measuring baseline metrics such as precision, recall, F1-score, etc., to understand how well the model performs without any fine-tuning.

Fine-Tuning on Training Data:
Use the training portion of a toxicity-related dataset to fine-tune the selected LLM. This process involves adjusting the model's weights specifically to improve its performance on the toxicity detection task.
After fine-tuning, evaluate the model on the test set again and compare the metrics to the baseline to see the impact of fine-tuning.

Cross-Dataset Evaluation:
Test the fine-tuned model on a different dataset to assess the model's generalization capability.
This step checks if the improvements from fine-tuning hold when the model is exposed to data from a different source.

Training a Model from Scratch:
Train a text-to-text model solely on the training data from scratch without leveraging a pre-trained model.
Evaluate the performance of this model and compare the results with those of the fine-tuned pre-trained model to determine the benefits of using pre-trained models versus training from scratch.

## Dataset used

I used the Toxicity Comments Dataset - https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fc%2Fjigsaw-toxic-comment-classification-challenge%2Fdata  

# Baseline Model Testing

* Why a sequence classifier could not be used? *

Dataset constraints - Here I saw that there were multiple classes in this dataset beyond just "toxic". 
Hence, it was a multi-label classification problem. That is, multiple nonexclusive labels (or none at all) may be assigned to each instance. Hence, I realized that while building our classifier, we will need to use a model capable of producing multiple labels for each input.

Initially, I used the 'roberta-base' model but realized that this model is only used for sequence classification. Hence, this was only used to give me the predicted probablities for each toxicity label for any sample sentence.






