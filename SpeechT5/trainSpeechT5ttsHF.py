
import os
os.environ["WANDB_MODE"] = "disabled"

import torch
import torchaudio

MODEL_NAME = "microsoft/speecht5_tts"

CACHE_DIR = "D:/LanguageModels/cache"
DATASET_DIR = "D:/LanguageModels/dataset/"
AUDIO_DIR = "D:/LanguageModels/dataset/audio/"



from datasets import load_dataset , Audio


# dataset = load_dataset("facebook/voxpopuli", "nl", split="train[:10]", trust_remote_code=True, cache_dir=CACHE_DIR)
# len(dataset)

dataset = load_dataset("json", data_files = DATASET_DIR + "train.json")

dataset = dataset["train"]  # Ensure correct split selection

# gender_mapping = {"number1.wav": "male", "number2.wav": "female"}  # Example mapping
# dataset = dataset.map(lambda example: {**example, "gender": gender_mapping.get(example["audio"], "unknown")})
print(type(dataset))
print(len(dataset))
print(dataset[0])

audio_id = 1

def update_example(example, audio_id):
    example['audio_id'] = audio_id
    example['language'] = 9
    example['gender'] = 'female'
    example['speaker_id'] = '1122'
    example['is_gold_transcript'] = True
    example['accent'] = 'None'

    waveform, sampling_rate = torchaudio.load(AUDIO_DIR + example['audiofile'])
    # Convert to NumPy
    audio_array = waveform.numpy()
    # print(audio_array.flatten())
    example['audio'] = {}
    example['audio']['array'] = audio_array.flatten()
    example['audio']['sampling_rate'] = sampling_rate

    return example

# Use `enumerate` with `map()` to update dataset
dataset = dataset.map(lambda example, idx: update_example(example, idx + 1), with_indices=True)

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
# exit(0)

from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor

processor = SpeechT5Processor.from_pretrained(MODEL_NAME , cache_dir=CACHE_DIR)




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


def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        text=example["text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    # example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example


processed_example = prepare_dataset(dataset[0])
print(list(processed_example.keys()))

# import matplotlib.pyplot as plt

# plt.figure()
# plt.imshow(processed_example["labels"].T)
# plt.show()

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# dataset = dataset.map(prepare_dataset)
# dataset = dataset.map(preprocess)

# print(dataset[0])

# dataset = dataset.train_test_split(test_size=0.1)

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        
        # speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [
                    length - length % model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        # batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch
    

data_collator = TTSDataCollatorWithPadding(processor=processor)


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

    # collator = DataCollatorWithPadding(tokenizer=processor.tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )


    trainer.train()


    model.save_pretrained("D:/LanguageModels/fine_tuned_tts")
    processor.save_pretrained("D:/LanguageModels/fine_tuned_tts")



