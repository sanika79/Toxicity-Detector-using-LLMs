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
