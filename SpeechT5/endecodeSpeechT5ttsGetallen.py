
import os
os.environ["WANDB_MODE"] = "disabled"

import torch
import torchaudio

MODEL_NAME = "microsoft/speecht5_tts"

CACHE_DIR = "D:/LanguageModels/cache"
DATASET_DIR = "D:/LanguageModels/audiodata/"
AUDIO_DIR = "D:/LanguageModels/audiodata/"

from transformers import SpeechT5HifiGan

from datasets import load_dataset , Audio


# dataset = load_dataset("facebook/voxpopuli", "nl", split="train[:10]", trust_remote_code=True, cache_dir=CACHE_DIR)
# len(dataset)

dataset = load_dataset("json", data_files = DATASET_DIR + "trainNL.json")

dataset = dataset["train"]  # Ensure correct split selection

# gender_mapping = {"number1.wav": "male", "number2.wav": "female"}  # Example mapping
# dataset = dataset.map(lambda example: {**example, "gender": gender_mapping.get(example["audio"], "unknown")})
print(type(dataset))
print(len(dataset))
print('example to encode' , dataset[8])

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

print('example to encode with audio' , dataset[8])


from transformers import SpeechT5Processor

processor = SpeechT5Processor.from_pretrained(MODEL_NAME , cache_dir=CACHE_DIR)



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

# print('example after encoding' , dataset[8])

spectrogramnegen = dataset[8]['labels']

print('spectrogram' , spectrogramnegen)
print('spectrogram type' , type(spectrogramnegen))


vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan" , cache_dir=CACHE_DIR)

# Convert list to torch.Tensor
spectrogram_tensor = torch.tensor(spectrogramnegen, dtype=torch.float32)
spectrogram_tensor = spectrogram_tensor.unsqueeze(0)

waveform = vocoder(spectrogram_tensor)

print('waveform' , waveform)
print('waveform' , waveform)

# Convert tensor to NumPy before saving
waveform_numpy = waveform.squeeze(0).cpu().detach().numpy() 

# Reshape to 2D format: [1, samples] (mono-channel audio)
waveform_tensor = torch.from_numpy(waveform_numpy).unsqueeze(0)

# Save generated speech
import torchaudio
torchaudio.save("output/negenTfHiFi.wav", waveform_tensor, 16000)


