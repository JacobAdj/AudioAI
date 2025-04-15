import torch

from sparktts.SparkTTS import SparkTTS
from datasets import load_dataset

# Load dataset
dataset = load_dataset("Spark-TTS-finetune/dataset")


model = SparkTTS('D:/LanguageModels/sparktts', 0)

# Load pre-trained Spark-TTS model
# model = SparkTTS.from_pretrained("SparkAudio/Spark-TTS-0.5B")

# Fine-tuning parameters
training_args = {
    "batch_size": 16,
    "learning_rate": 0.0001,
    "epochs": 10
}

# Fine-tune model
model.train(dataset, **training_args)

# Save fine-tuned model
model.save_pretrained("D:/LanguageModels/fine-tuned-sparktts")
