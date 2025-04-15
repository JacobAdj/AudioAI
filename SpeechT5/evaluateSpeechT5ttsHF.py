
import os
os.environ["WANDB_MODE"] = "disabled"

import torch
import torchaudio

MODEL_NAME = "microsoft/speecht5_tts"

CACHE_DIR = "D:/LanguageModels/cache"
DATASET_DIR = "D:/LanguageModels/dataset/"
AUDIO_DIR = "D:/LanguageModels/dataset/audio/"



from datasets import load_dataset , Audio

from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor

processor = SpeechT5Processor.from_pretrained(MODEL_NAME , cache_dir=CACHE_DIR)
model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_NAME , cache_dir=CACHE_DIR)

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


dataset = dataset.map(prepare_dataset)




model.eval()  # Set model to evaluation mode

total_loss = 0
for batch in dataset:
    input_ids = processor.tokenizer(batch["text"], return_tensors="pt").input_ids
    labels = batch["audio"]["array"]  # Ensure correct format

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss.item()
        total_loss += loss

print(f"Pre-trained SpeechT5 Loss: {total_loss / len(dataset):.4f}")