from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
import torch
import torchaudio

from datasets import load_dataset, Audio

import os
os.environ["WANDB_MODE"] = "disabled"


MODEL_NAME = "microsoft/speecht5_tts"

processor = SpeechT5Processor.from_pretrained(MODEL_NAME)
# model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_NAME)


MODEL_NAME = "microsoft/speecht5_tts"

CACHE_DIR = "D:/LanguageModels/cache"
DATASET_DIR = "D:/LanguageModels/dataset/"
AUDIO_DIR = "D:/LanguageModels/dataset/audio/"

processor = SpeechT5Processor.from_pretrained(MODEL_NAME , cache_dir=CACHE_DIR)



# dataset = load_dataset("facebook/voxpopuli", "nl", split="train[:10]", trust_remote_code=True, cache_dir=CACHE_DIR)
# len(dataset)

dataset = load_dataset("json", data_files = DATASET_DIR + "train.json")

dataset = dataset["train"]  # Ensure correct split selection

print(dataset[0])

# exit(0)

def preprocess(batch):
    # Tokenize text for model input
    batch["input_values"] = processor.tokenizer(batch["text"], 
                                                return_tensors="pt", 
                                                padding="max_length", max_length=21, truncation=True).input_ids

    print('tokenized text shape' , batch["input_values"].shape)

    # Convert audio into spectrogram (or mel-spectrogram)
    speech_array, sampling_rate = torchaudio.load(AUDIO_DIR + batch["audio"])

    print(f"Audio sample rate: {sampling_rate}")
    print(f"Waveform min: {speech_array.min()}, max: {speech_array.max()}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, n_mels=10, n_fft=512)(speech_array)

    print(f"Mel Spectrogram Shape: {mel_spectrogram.shape}")
    # print(mel_spectrogram)
    if not torch.any(mel_spectrogram):
        print("Warning: Spectrogram contains only zeros!")
    else:
        print('Spectrogram ok')

    batch["labels"] = mel_spectrogram

    batch["labels"] = batch["labels"][0]

    # batch["labels"] = torch.nn.functional.pad(mel_spectrogram, (0, max(0, 150 - mel_spectrogram.shape[-1])), 
    #                                           mode="constant", value=0)
    
    # print('tokenized sound ' , batch["labels"])

    return batch

dataset = dataset.map(preprocess)

print(dataset[0])


# max_len = 0
# max_label_len = 0

# for sample in dataset:
#     tokenized_text = processor.tokenizer(sample["text"], return_tensors="pt").input_ids
#     max_len = max(max_len, tokenized_text.shape[-1])

#     speech_array, sampling_rate = torchaudio.load(AUDIO_DIR + sample["audio"])
#     mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate)(speech_array)
    
#     max_label_len = max(max_label_len, mel_spectrogram.shape[-1])  # Length dimension

# print(f"Max spectrogram length: {max_label_len}")

# print(f"Max sequence length: {max_len}")





training = True

if training:

    from transformers import TrainingArguments, Trainer

    model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_NAME , cache_dir=CACHE_DIR)

    # print(model.config)


    training_args = TrainingArguments(
        output_dir="D:/LanguageModels/out_T5tts",
        run_name="tts_exp", 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        save_steps=100,
        save_total_limit=2,
        logging_dir="D:/LanguageModels/logs",
        learning_rate=0.0002,
        optim="adamw_torch_fused",
        eval_strategy="no",
        # eval_strategy="steps",
        eval_steps=500,
        weight_decay=0.0,
    )

    from transformers import DataCollatorWithPadding

    collator = DataCollatorWithPadding(tokenizer=processor.tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )


    trainer.train()


    model.save_pretrained("D:/LanguageModels/fine_tuned_tts")
    processor.save_pretrained("D:/LanguageModels/fine_tuned_tts")



