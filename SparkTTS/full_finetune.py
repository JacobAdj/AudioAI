import torch

from cli.SparkTTS import SparkTTS
from sparktts.models.audio_tokenizer import BiCodecTokenizer

import torchaudio

from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer


# Load Spark-TTS pretrained model
MODEL_PATH = "D:/LanguageModels/sparktts"

AUDIO_PATH = "D:/LanguageModels/sparktts/dataset/audio/"

# Load Spark-TTS model
model = SparkTTS("D:/LanguageModels/sparktts", "cpu")

# Load dataset
dataset = load_dataset("json", data_files="D:/LanguageModels/sparktts/dataset/train.json")


# Load audio tokenizer
# audio_tokenizer = BiCodecTokenizer(MODEL_PATH, device=torch.device("cpu"))


def preprocess(batch):

    # global_token_ids, semantic_token_ids = audio_tokenizer.tokenize(prompt_audio_path)
    
    # global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()])
    # semantic_tokens = "".join([f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()])
    
    # input_text = f"<|start_content|>{text}{global_tokens}{semantic_tokens}<|end_content|>"
 

    batch["input_values"] = batch["text"]

    waveform, sampling_rate = torchaudio.load(AUDIO_PATH + batch["audio"])
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate)(waveform)
    batch["labels"] = mel_spectrogram

    return batch


dataset = dataset.map(preprocess)


training_args = TrainingArguments(
    output_dir="D:/LanguageModels/fine_tuned_sparktts",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=100,
    save_total_limit=2,
    # logging_dir="./logs",
    learning_rate=0.0002,
    optim="adamw_torch_fused",
    eval_strategy="no",
    eval_steps=500,
    weight_decay=0.0,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()


# Save fine-tuned model
model.save_pretrained("D:/LanguageModels/fine-tuned-sparktts")
